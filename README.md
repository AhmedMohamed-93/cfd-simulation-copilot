# CFD Simulation Copilot
### LLM Agent for OpenFOAM Simulation Setup | Free via Hugging Face | No Docker, No Local GPU/RAM | Production-Ready on Mistral API

[![CI](https://github.com/AhmedMohamed-93/cfd-simulation-copilot/actions/workflows/ci.yml/badge.svg)](https://github.com/AhmedMohamed-93/cfd-simulation-copilot/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Hugging Face](https://img.shields.io/badge/LLM-Hugging%20Face%20Inference%20API-yellow)
![Mistral](https://img.shields.io/badge/production-Mistral%20API-orange)
![No Docker](https://img.shields.io/badge/runtime-no%20Docker%20required-success)
![License](https://img.shields.io/badge/license-MIT-green)

---

CFD Simulation Copilot is a production-grade AI agent that integrates LLMs with engineering simulation workflows, translating natural language problem descriptions into complete, physics-validated OpenFOAM case configurations. It is built on **LangGraph** agentic orchestration and **RAG over OpenFOAM documentation**, and demonstrates the full AI4Engineering stack: domain knowledge retrieval, physics-constrained reasoning, structured output generation, and rigorous validation against engineering standards.

**Everything runs free and container-free by default: no Docker, no local GPU/RAM requirement, no cost.** LLM reasoning runs on the [Hugging Face Inference API](https://huggingface.co/settings/tokens) (`mistralai/Mistral-7B-Instruct-v0.2`), free with an HF account, so there's no local model to load and no RAM ceiling. Embeddings and reranking run on local `sentence-transformers` models (small enough to run on any machine), vector storage runs on embedded (serverless) Qdrant, and agent traces are written to a local structured JSON log. The entire setup is two commands: `python setup.py && python run.py`.

> **Designed to be LLM-agnostic.** Every LLM call goes through a single provider-pluggable client ([`src/common/llm_client.py`](src/common/llm_client.py)) supporting three backends: Hugging Face (default, free, no local RAM), a fully local Ollama server (free but RAM-hungry, see [Production deployment](#production-deployment)), and the **Mistral API** (`mistral-large-latest`) for production. Switching is one setting; the agent graph, tools, and prompts are unchanged. This project doubles as a direct demonstration of building a production agentic + RAG system on Mistral's own models, without requiring a paid key just to try it out.

🔗 Live demo coming soon: deploying to Hugging Face Spaces

**Source:** [github.com/AhmedMohamed-93/cfd-simulation-copilot](https://github.com/AhmedMohamed-93/cfd-simulation-copilot)

---

## Running for free: no Docker, no local GPU/RAM requirement

| Layer               | Free, container-free implementation                  |
|----------------------|--------------------------------------------------------------|
| LLM reasoning         | [Hugging Face Inference API](https://huggingface.co/settings/tokens) (`mistralai/Mistral-7B-Instruct-v0.2`), free HF account required |
| Embeddings              | `sentence-transformers` / `all-MiniLM-L6-v2` (384-dim, CPU, local)   |
| Reranking                 | `sentence-transformers` cross-encoder (`ms-marco-MiniLM-L-6-v2`, local) |
| Vector database              | Embedded Qdrant (`qdrant_storage/` folder on disk, serverless) |
| Agent traces                  | Structured JSON log (`logs/agent_traces.json`)              |

No credit card, no Docker daemon, and no local model to load are needed anywhere in this list: only a free Hugging Face account and token for LLM calls (get one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)). Embedding/reranking model weights download once (from Hugging Face) and are cached locally. Prefer a fully local, zero-account setup and have the RAM to spare? Set `LLM_PROVIDER=ollama`. See [Production deployment](#production-deployment). A Docker path also still exists in [`docker-compose.yml`](docker-compose.yml) for anyone who prefers containers, but it is entirely optional. See that file's header comment.

---

## Architecture

```
                                   ┌─────────────────────────────┐
                                   │      Streamlit Frontend     │
                                   │  (CFD Copilot / KB / Traces)│
                                   └──────────────┬───────────────┘
                                                  │ HTTP
                                   ┌──────────────▼───────────────┐
                                   │        FastAPI Layer         │
                                   │  /simulate  /health  /kb ... │
                                   └──────────────┬───────────────┘
                                                  │
                                   ┌──────────────▼───────────────┐
                                   │     LangGraph ReAct Agent    │
                                   └───────────────────────────────┘

  START
    │
    ▼
┌─────────────────────────┐
│ parse_flow_description  │  LLMClient (Hugging Face by default, structured
│                          │  output) extracts Re, Ma, geometry, fluid, BCs
└────────────┬─────────────┘
             ▼
┌─────────────────────────┐  ◄──────────────────┐
│ retrieve_cfd_knowledge  │                      │ retry (quality < 0.5,
│  stage 1: Qdrant dense  │──────────────────────┘  max 2 attempts)
│  stage 2: local cross-  │
│  encoder rerank         │
└────────────┬─────────────┘
             ▼
┌─────────────────────────┐
│ select_solver_and_models│  LLMClient reasons over decision tree +
│                          │  retrieved docs → SolverConfiguration
└────────────┬─────────────┘◄─────────────────────┐
             ▼                                     │ retry (critical
┌─────────────────────────┐                        │ validation error,
│ generate_openfoam_files │                        │ max 1 retry)
│  Pydantic → OpenFOAM     │                        │
└────────────┬─────────────┘                        │
             ▼                                     │
┌─────────────────────────┐──────────────────────┘
│    validate_physics     │  rule-based physics + syntax checks
└────────────┬─────────────┘
             ▼
┌─────────────────────────┐
│  format_final_response  │  markdown report + citations + next steps
└────────────┬─────────────┘
             ▼
            END

  Cross-cutting: iteration_count capped at MAX_AGENT_ITERATIONS (10);
  every run is appended as one structured entry (timestamp, query, steps,
  latency_ms, result) to the local logs/agent_traces.json file.

┌────────────────────────── RAG Layer ──────────────────────────┐
│  document_loader → chunker → all-MiniLM-L6-v2 →                │
│  embedded Qdrant (qdrant_storage/, cosine similarity)           │
│  Sources: OpenFOAM User Guide, OpenFOAM tutorials, CFD-Online   │
│  Wiki (turbulence models), curated arXiv CFD+ML papers          │
│  (synthetic fallback content if any source is unreachable)      │
└──────────────────────────────────────────────────────────────┘
```

---

## Agent Reasoning Example

**Input query:**
> "Turbulent flow of air through a pipe at Reynolds number 50000, steady state, interested in pressure drop."

**Agent trace:**

```
[parse_flow_description] completed in 30.71s
  → Re=50000, fluid=air, geometry=pipe, is_steady=True, is_compressible=False

[retrieve_cfd_knowledge] completed in 18.80s
  → query: "solver selection for a flow with geometry 'pipe', Re=50000, fluid air"
  → 3 chunks retrieved, quality=0.30 from mean raw cross-encoder score (above 0.2 threshold, no retry)

[select_solver_and_models] completed in 29.74s
  → Re >= 5e4 → kOmegaSST recommended; incompressible + steady → simpleFoam
  → justification grounded in retrieved OpenFOAM User Guide + CFD-Online Wiki excerpts

[generate_openfoam_files] completed in 0.00s
  → 9 files generated: controlDict, fvSchemes, fvSolution,
    transportProperties, turbulenceProperties, 0/U, 0/p, 0/k, 0/omega

[validate_physics] completed in 0.00s
  → 8/9 checks passed, 1 warning (viscosity_consistency), 0 errors

[format_final_response] completed in 0.00s

Total: 80.4s (local Ollama, llama3.2:latest, CPU). See Engineering Notes below.
```

**Output (excerpt):** solver `simpleFoam`, turbulence model `kOmegaSST`, 9 syntactically valid OpenFOAM files, full physics validation report, and citations back to the specific documentation chunks that justified each choice.

---

## Evaluation Results

Run via `python -m src.evaluation.ragas_eval --cases 5` (a subset of the 20 scenarios in [`evaluation/test_cases.json`](evaluation/test_cases.json); omit `--cases` to run all 20). This script always judges via a local Ollama model (`llama3.2:latest`), regardless of the app's own `LLM_PROVIDER`, so it's free and rate-limit-free to re-run.

*Evaluated on 5 CFD scenarios covering laminar, turbulent, internal and external flows.*

| Metric                          | Score   |
|----------------------------------|---------|
| Solver accuracy                  | 0.80²   |
| Turbulence model accuracy        | 1.00    |
| File completeness                | 1.00    |
| Physics validation pass rate     | 1.00    |
| RAGAS faithfulness               | N/A¹    |
| RAGAS answer relevancy           | N/A¹    |
| RAGAS context precision          | N/A¹    |

Turbulence model selection is governed by a deterministic physics decision layer that overrides LLM proposals failing physical consistency checks, so it is reproducible run-to-run regardless of LLM nondeterminism. Solver selection is deterministic except where the input query itself is physically ambiguous.

¹ Not computed in this run: `ragas==0.2.6`'s judge call is incompatible with the installed `langchain-ollama`/`ollama` client version (`AsyncClient.chat() got an unexpected keyword argument 'temperature'`), a dependency-version issue independent of the 4 metrics above (which score deterministically, without an LLM judge call). Full per-case results are in [`evaluation/results/ragas_results.json`](evaluation/results/ragas_results.json).

² tc04 (lid-driven cavity) is the single miss: the query does not state whether the case is steady or transient, so the LLM's is_steady parse is genuinely ambiguous and varies run-to-run. This was left unforced deliberately: hardcoding is_steady=False for cavity geometries would produce wrong answers for the classic steady Ghia et al. benchmark. Turbulence model selection is fully deterministic and reproducible.

---

## Engineering Notes

### Latency optimization: 310s → 80s

| Step | Baseline | Final |
|---|---|---|
| parse_flow_description | 78.4s | 30.7s |
| retrieve_cfd_knowledge | 14.6s (retried) | 18.8s |
| select_solver_and_models | 227.9s | 29.7s |
| **Total** | **309.8s** | **80.4s** |

Three levers, in order of impact: capping `max_tokens` at 300 for structured-output calls (the outputs are fixed-shape JSON objects, so unbounded generation was pure latency cost), compressing the solver-selection prompt from 4,524 to 801 characters, and replacing the LLM-based retrieval-quality grader with a deterministic cross-encoder score threshold, removing one LLM round-trip per run entirely.

### Prompt compression as a correctness test

Cutting the solver-selection prompt by 82% caused turbulence-model accuracy to drop from 1.00 to 0.60. The verbose prompt had been carrying soft guidance (steering away from kEpsilon for wall-bounded flows) that the deterministic rule layer never actually enforced: the system appeared correct only because the LLM was reading a paragraph.

The fix was to close the rule-layer gap rather than restore the prompt: kEpsilon is now accepted only for genuinely free-shear or multiphase flows, cross-checked against the full 20-case suite (tc09 T-junction mixing and tc17 dam-break both legitimately expect kEpsilon) to avoid over-correcting.

A second, related bug surfaced the same way: `_check_solver_algorithm_consistency` was validating the LLM's proposed solver against the LLM's own `is_steady` field rather than against the parsed flow's actual `is_steady`, so an internally-consistent-but-wrong proposal passed validation. Extended to compare against the flow description directly.

Both gaps existed before the prompts were shortened; compression only made them observable. The resulting system's engineering-critical outputs are enforced by deterministic physics rules rather than prompt wording.

---

## Generated File Example

`constant/turbulenceProperties` for a `kOmegaSST` case, generated by [`src/generation/schemas.py`](src/generation/schemas.py):

```cpp
/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox          |
|  \\    /   O peration     | Version:  10                                   |
|   \\  /    A nd           | Web:      www.openfoam.com                     |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      turbulenceProperties;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

simulationType  RAS;

RAS
{
    RASModel        kOmegaSST;

    turbulence      on;
    printCoeffs     on;
}

// ************************************************************************* //
```

---

## Tech Stack

| Component           | Technology                          | Purpose                                             |
|----------------------|--------------------------------------|------------------------------------------------------|
| LLM reasoning         | Hugging Face Inference API (`Mistral-7B-Instruct-v0.2`), free, swappable to local Ollama or Mistral API (`mistral-large-latest`) | Flow parsing, solver selection, reporting |
| Embeddings             | `sentence-transformers` (`all-MiniLM-L6-v2`), local, free | 384-dim vectors for the RAG knowledge base |
| Reranking               | `sentence-transformers` cross-encoder (`ms-marco-MiniLM-L-6-v2`), local, free | Stage-2 relevance reranking |
| Agent orchestration    | LangGraph                           | Stateful ReAct loop with conditional retries          |
| RAG pipeline           | LangChain                           | Chunking, document abstractions, retrieval glue       |
| Vector database        | Qdrant, embedded/serverless mode, local, free | Dense retrieval over the CFD knowledge base (no server, no Docker) |
| Structured outputs      | Pydantic v2                        | OpenFOAM file schemas, agent I/O contracts             |
| REST API                | FastAPI                            | `/simulate`, `/health`, `/sessions`, `/feedback`        |
| Frontend                 | Streamlit                          | Copilot UI, knowledge base explorer, agent traces        |
| Observability             | Structured local JSON log (`logs/agent_traces.json`) | Per-run trace: query, steps, latency, result |
| RAG evaluation              | RAGAS                          | Faithfulness, answer relevancy, context precision            |
| Process orchestration          | `run.py` / `setup.py`        | Single-command local launch (no containers, no orchestrator)    |
| CI/CD                            | GitHub Actions                | Lint (ruff), test (pytest + coverage) on every push              |

---

## Physics Validation Rules

Implemented in [`src/generation/physics_validator.py`](src/generation/physics_validator.py):

1. **Reynolds/turbulence consistency**: flags a turbulence model selected for Re < 2300, or a laminar model selected for Re ≥ 2300.
2. **CFL feasibility**: estimates the Courant number from the generated `deltaT`, inlet velocity, and characteristic length for transient runs.
3. **Boundary condition completeness**: every field file (`U`, `p`, `k`, `omega`/`epsilon`) must define the exact same set of boundary patches.
4. **Turbulence intensity range**: generated `k`/`epsilon`/`omega` values must correspond to a physically realistic turbulence intensity (0.001–0.2).
5. **Viscosity consistency**: the generated kinematic viscosity must be within a factor of 2 of the reference value for the stated fluid.
6. **Pressure reference well-posedness**: incompressible solvers must have a `fixedValue` pressure boundary to avoid an under-determined pressure field.
7. **OpenFOAM syntax validity**: every generated file must contain a valid `FoamFile` header and have balanced braces/parentheses.

---

## Quick Start

> **Free tier:** Ollama (local) | **Higher quality:** Hugging Face Inference API (free account required) | **Production:** Mistral API

**Prerequisites:** Python 3.11+, [Ollama](https://ollama.com) installed locally (no account, no API key).

```bash
# Step 1
ollama pull llama3.2:latest
cp .env.example .env   # defaults to LLM_PROVIDER=ollama, nothing else to edit

# Step 2
python setup.py

# Step 3
python run.py

# Step 4
open http://localhost:8501
```

`setup.py` installs dependencies into your current environment, downloads the local embedding model, and builds the CFD knowledge base (embedded Qdrant, a `qdrant_storage/` folder is created automatically, nothing to run separately). `run.py` starts the FastAPI backend and the Streamlit frontend together and opens your browser; press Ctrl+C to stop both. No Docker, no account signup, no GPU required.

- Streamlit UI: [http://localhost:8501](http://localhost:8501)
- FastAPI docs: [http://localhost:8000/docs](http://localhost:8000/docs)

Want better-quality responses and don't mind a free account? Set `LLM_PROVIDER=huggingface` and `HF_API_TOKEN` in `.env` instead: no local RAM/GPU needed, but subject to the HF free tier's monthly usage limits. See [Production deployment](#production-deployment) for all three provider options, including the paid Mistral API.

Prefer containers instead? An optional [`docker-compose.yml`](docker-compose.yml) is still included. See its header comment.

---

## Knowledge Base

Ingested by [`src/ingestion/document_loader.py`](src/ingestion/document_loader.py), with a rich synthetic fallback for any source that is unreachable, so the system always has a working knowledge base:

- **OpenFOAM User Guide**: solver descriptions, boundary conditions, numerical schemes, meshing guidelines.
- **OpenFOAM tutorial catalogue**: representative case setups per solver family (GitHub tutorials directory).
- **CFD-Online Wiki**: turbulence model reference pages (k-ε, k-ω SST, Spalart-Allmaras, LES, DNS).
- **Curated arXiv papers**: 20 papers on CFD surrogate modeling and machine learning for fluid mechanics.

Every chunk carries metadata: `source`, `title`, `topic_tags` (solver / turbulence_model / flow_type / boundary_condition / mesh_type), and `difficulty_level`, enabling metadata-filtered retrieval in [`src/retrieval/vector_store.py`](src/retrieval/vector_store.py).

---

## Evaluation

[`evaluation/test_cases.json`](evaluation/test_cases.json) defines 20 realistic CFD scenarios spanning laminar and turbulent internal flows, external aerodynamics, compressible/supersonic flow, buoyancy-driven convection, and multiphase free-surface problems, each with an expected solver, turbulence model, and evaluation criteria.

[`src/evaluation/ragas_eval.py`](src/evaluation/ragas_eval.py) runs every scenario through the full agent graph and reports:

- **RAGAS metrics**: faithfulness, answer relevancy, context precision. The evaluation script always forces `LLM_PROVIDER=ollama` (`llama3.2:latest`) for both the agent and the judge (regardless of the deployed app's own `LLM_PROVIDER`), since evaluation runs repeatedly and locally, and Ollama has no rate/credit limit, unlike the HF Inference API free tier the app defaults to. Judge embeddings are always the local `sentence-transformers` model.
- **Custom engineering metrics**: solver accuracy, turbulence model accuracy, file completeness, physics validation pass rate.

Results are saved to `evaluation/results/ragas_results.json` and surfaced in the Streamlit **Agent Traces** page.

---

## Production Deployment

Three interchangeable LLM providers are supported via [`src/common/llm_client.py`](src/common/llm_client.py): the single seam every agent tool calls through, so `select_solver_and_models` and `parse_flow_description` transparently switch backend with identical prompts, schemas, and validation logic. No other code changes between them. (Retrieval-quality grading doesn't use an LLM at all. It's a deterministic read of the cross-encoder's own scores, so it's unaffected by provider choice and costs no extra round-trip.)

| `LLM_PROVIDER` | Backend | When to use |
|---|---|---|
| `huggingface` (default) | Hugging Face Inference API, `mistralai/Mistral-7B-Instruct-v0.2` | Free demos/dev, no local RAM/GPU, just an HF account |
| `ollama` | Local Ollama server, e.g. `llama3.1:8b` | Free, fully local/offline, needs ~5GB+ free RAM to load the model |
| `mistral` | Mistral API, `mistral-large-latest` | Production, highest quality, paid |

```bash
# .env: production
LLM_PROVIDER=mistral
MISTRAL_API_KEY=your_key_here

# .env: fully local/offline alternative
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1:8b   # ollama pull llama3.1:8b first
```

Embeddings and reranking remain local (`sentence-transformers`) regardless of `LLM_PROVIDER`, since they are already free, fast, and require no GPU. There is no cost benefit to moving them to a paid API. The system is designed to be LLM-agnostic: adding OpenAI or Anthropic as a fourth option would mean one more branch in `LLMClient`, not restructuring the agent.

---

## About the Author

**Ahmed Mohamed**, Aerospace Engineer & PhD in Fluid Mechanics (École Centrale de Lyon). 4 years of DNS research on CNRS national HPC clusters. Expertise in OpenFOAM, ANSYS, turbulence modeling, and scientific computing. Now working at the intersection of CFD and AI.

[GitHub](https://github.com/AhmedMohamed-93) | [LinkedIn](https://linkedin.com/in/ahmed-mohamed11)
