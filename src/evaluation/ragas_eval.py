"""RAGAS + custom evaluation pipeline for the CFD Simulation Copilot agent.

Runs every scenario in evaluation/test_cases.json through the full agent
graph, then scores the results two ways:

1. RAGAS metrics (faithfulness, answer_relevancy, context_precision), judged
   by a local Ollama model (llama3.2:latest) via ``langchain_ollama`` — this
   script always forces LLM_PROVIDER=ollama for both the agent and the judge
   (see the top of this module), regardless of the deployed app's own
   LLM_PROVIDER setting, since evaluation runs repeatedly and locally and
   Ollama has no rate/credit limit, unlike the HF Inference API free tier the
   app defaults to. Judge embeddings always use the same local
   sentence-transformers model as retrieval, via a thin LangChain Embeddings
   adapter around LocalEmbedder.
2. Custom engineering metrics: solver_accuracy, turbulence_model_accuracy,
   file_completeness, and physics_validation_pass_rate.

RAGAS evaluation is best-effort: if the installed ragas version's API
differs or the judge backend is unavailable, RAGAS scores are recorded as
``None`` rather than crashing the whole evaluation run, so custom metrics
(which do not require an extra LLM judge call) are always produced.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

# This script runs the agent + judge locally and repeatedly, so it forces
# LLM_PROVIDER=ollama regardless of the environment's own setting: Ollama is
# free and has no rate/credit limit, unlike the HF Inference API's free tier
# (which the deployed app defaults to and which this script would otherwise
# burn through on every evaluation run). Must be set before `config` (or
# anything importing it) is imported, since Settings() reads the environment
# at import time.
os.environ["LLM_PROVIDER"] = "ollama"
os.environ["OLLAMA_MODEL"] = "llama3.2:latest"

from config import settings  # noqa: E402
from src.agent.graph import run_agent  # noqa: E402
from src.generation.schemas import SolverConfiguration, ValidationResult  # noqa: E402
from src.retrieval.embedder import LocalEmbedder  # noqa: E402

logger = logging.getLogger(__name__)


class _LocalEmbeddingsAdapter:
    """Adapts LocalEmbedder to the LangChain/RAGAS Embeddings interface."""

    def __init__(self, embedder: LocalEmbedder | None = None) -> None:
        """Initialize the adapter.

        Args:
            embedder: A LocalEmbedder instance; a new one is created if omitted.
        """
        self._embedder = embedder or LocalEmbedder()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents.

        Args:
            texts: The texts to embed.

        Returns:
            One embedding vector per input text.
        """
        return self._embedder.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query.

        Args:
            text: The text to embed.

        Returns:
            The embedding vector for the query.
        """
        return self._embedder.embed_query(text)


def _build_ragas_judge() -> tuple[Any, Any]:
    """Build the judge LLM and embeddings RAGAS uses to score each case.

    Mirrors whichever LLM_PROVIDER the agent itself is configured to use,
    so the judge is never silently pointed at a different backend than the
    one being evaluated. Judge embeddings are always the local
    sentence-transformers model, regardless of LLM_PROVIDER.

    Returns:
        A (judge_llm, judge_embeddings) tuple compatible with ragas.evaluate.
    """
    judge_embeddings = _LocalEmbeddingsAdapter()

    if settings.LLM_PROVIDER == "mistral":
        from langchain_mistralai import ChatMistralAI  # noqa: PLC0415

        judge_llm = ChatMistralAI(model=settings.MISTRAL_MODEL, mistral_api_key=settings.MISTRAL_API_KEY)
    elif settings.LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama  # noqa: PLC0415

        judge_llm = ChatOllama(model=settings.OLLAMA_MODEL, base_url=settings.OLLAMA_BASE_URL)
    else:
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint  # noqa: PLC0415

        endpoint = HuggingFaceEndpoint(repo_id=settings.HF_MODEL, huggingfacehub_api_token=settings.HF_API_TOKEN)
        judge_llm = ChatHuggingFace(llm=endpoint)

    return judge_llm, judge_embeddings


TEST_CASES_PATH = Path("evaluation/test_cases.json")
RESULTS_PATH = Path("evaluation/results/ragas_results.json")


def _load_test_cases() -> list[dict[str, Any]]:
    """Load the 20 CFD evaluation scenarios from disk.

    Returns:
        The list of test case dicts.
    """
    return json.loads(TEST_CASES_PATH.read_text(encoding="utf-8"))


def _run_case(test_case: dict[str, Any]) -> dict[str, Any]:
    """Run one test case through the full agent graph and score it.

    Args:
        test_case: A single test case dict from test_cases.json.

    Returns:
        A dict with the raw agent outputs plus custom per-case metrics.
    """
    final_state = run_agent(test_case["description"])

    solver_config_raw = final_state.get("solver_config") or {}
    validation_raw = final_state.get("validation_results") or {}
    generated_files = final_state.get("generated_files") or {}

    solver_accuracy = 0.0
    turbulence_accuracy = 0.0
    if solver_config_raw:
        try:
            config = SolverConfiguration.model_validate(solver_config_raw)
            solver_accuracy = float(
                config.solver_name.value == test_case["expected_solver"]
            )
            turbulence_accuracy = float(
                config.turbulence_model.value == test_case["expected_turbulence_model"]
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not parse solver_config for %s: %s", test_case["id"], exc)

    required_files = {"system/controlDict", "system/fvSchemes", "system/fvSolution",
                       "constant/transportProperties", "constant/turbulenceProperties",
                       "0/U", "0/p"}
    present = required_files.intersection(generated_files.keys())
    file_completeness = len(present) / len(required_files)

    validation_passed = False
    if validation_raw:
        try:
            validation_passed = ValidationResult.model_validate(validation_raw).passed
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not parse validation_results for %s: %s", test_case["id"], exc)

    return {
        "id": test_case["id"],
        "description": test_case["description"],
        "expected_solver": test_case["expected_solver"],
        "expected_turbulence_model": test_case["expected_turbulence_model"],
        "actual_solver": solver_config_raw.get("solver_name"),
        "actual_turbulence_model": solver_config_raw.get("turbulence_model"),
        "solver_accuracy": solver_accuracy,
        "turbulence_model_accuracy": turbulence_accuracy,
        "file_completeness": file_completeness,
        "physics_validation_pass": validation_passed,
        "answer": final_state.get("final_response", ""),
        "contexts": [c["content"] for c in final_state.get("retrieved_chunks", [])],
        "error": final_state.get("error"),
    }


def _run_ragas_metrics(case_results: list[dict[str, Any]]) -> dict[str, float | None]:
    """Score faithfulness, answer_relevancy, and context_precision via RAGAS.

    Args:
        case_results: Per-case results including answer and contexts.

    Returns:
        A dict mapping metric name to its mean score, or None for any
        metric that could not be computed (missing deps, API error, or
        incompatible ragas version).
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness

        rows = [
            {
                "question": c["description"],
                "answer": c["answer"],
                "contexts": c["contexts"] or [""],
                "ground_truth": f"Use solver {c['expected_solver']} with turbulence model {c['expected_turbulence_model']}.",
            }
            for c in case_results
            if c["answer"]
        ]
        if not rows:
            return {"faithfulness": None, "answer_relevancy": None, "context_precision": None}

        dataset = Dataset.from_list(rows)
        judge_llm, judge_embeddings = _build_ragas_judge()
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
            llm=judge_llm,
            embeddings=judge_embeddings,
        )
        scores = result.to_pandas().mean(numeric_only=True).to_dict()

        def _clean(value: Any) -> float | None:
            # A per-row judge-call failure (e.g. a judge-backend API
            # incompatibility) makes ragas record NaN for that row rather
            # than raising, so an all-failed metric column means() to NaN,
            # not a missing key. Normalize that to None so the documented
            # "unavailable metrics are None" contract holds either way.
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return None
            return float(value)

        return {
            "faithfulness": _clean(scores.get("faithfulness")),
            "answer_relevancy": _clean(scores.get("answer_relevancy")),
            "context_precision": _clean(scores.get("context_precision")),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("RAGAS evaluation unavailable/failed, skipping: %s", exc)
        return {"faithfulness": None, "answer_relevancy": None, "context_precision": None}


def run_evaluation(n_cases: int | None = None) -> dict[str, Any]:
    """Run the evaluation suite across the test cases.

    Args:
        n_cases: If given, only the first `n_cases` scenarios from
            test_cases.json are run (in file order). Defaults to all 20.

    Returns:
        A summary dict with per-case results and aggregate metrics,
        matching the structure written to evaluation/results/ragas_results.json.
    """
    test_cases = _load_test_cases()
    if n_cases is not None:
        test_cases = test_cases[:n_cases]
    case_results = [_run_case(tc) for tc in test_cases]

    n = len(case_results)
    aggregate = {
        "solver_accuracy": sum(c["solver_accuracy"] for c in case_results) / n,
        "turbulence_model_accuracy": sum(c["turbulence_model_accuracy"] for c in case_results) / n,
        "file_completeness": sum(c["file_completeness"] for c in case_results) / n,
        "physics_validation_pass_rate": sum(c["physics_validation_pass"] for c in case_results) / n,
    }
    aggregate.update(_run_ragas_metrics(case_results))

    summary = {"n_cases": n, "aggregate_metrics": aggregate, "case_results": case_results}

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _print_summary_table(aggregate, case_results)
    return summary


def _print_summary_table(aggregate: dict[str, Any], case_results: list[dict[str, Any]]) -> None:
    """Print a human-readable summary table to the console.

    Args:
        aggregate: Aggregate metric scores.
        case_results: Per-case result dicts.
    """
    print("\n=== CFD Simulation Copilot — Evaluation Summary ===\n")
    print(f"{'Metric':<32}{'Score':>10}")
    print("-" * 42)
    for name, value in aggregate.items():
        display = f"{value:.3f}" if isinstance(value, int | float) else "N/A"
        print(f"{name:<32}{display:>10}")
    print("\n" + f"{'Case':<8}{'Expected Solver':<18}{'Actual Solver':<18}{'Valid?':<8}")
    print("-" * 52)
    for c in case_results:
        print(
            f"{c['id']:<8}{c['expected_solver']:<18}{str(c['actual_solver']):<18}"
            f"{str(c['physics_validation_pass']):<8}"
        )


def _parse_args() -> Any:
    """Parse command-line arguments for the evaluation script.

    Returns:
        The parsed argparse Namespace, with a `cases` attribute (int or None).
    """
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Run the CFD Simulation Copilot evaluation suite.")
    parser.add_argument(
        "--cases",
        type=int,
        default=None,
        help="Run only the first N scenarios from evaluation/test_cases.json (default: all 20).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    run_evaluation(n_cases=args.cases)
