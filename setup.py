"""One-time project setup — no Docker required.

Installs Python dependencies into the current environment, downloads the
local embedding model, and builds the CFD knowledge base (document
ingestion + chunking + embedding + indexing into embedded Qdrant).

Usage:
    python setup.py

Prerequisites: Python 3.11+, and an HF_API_TOKEN set in .env (free at
https://huggingface.co/settings/tokens) — see README.md. If you'd rather run
fully local instead, set LLM_PROVIDER=ollama in .env and pull a model with
`ollama pull llama3.1:8b` before running this script.
"""

from __future__ import annotations

import subprocess
import sys


def _run(cmd: list[str], description: str) -> None:
    """Run a subprocess step, streaming its output, and exit on failure.

    Args:
        cmd: The command and arguments to run.
        description: Human-readable label printed before running.
    """
    print(f"\n=== {description} ===")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\nFAILED: {description} (exit code {result.returncode})")
        sys.exit(result.returncode)


def _install_dependencies() -> None:
    """Install requirements.txt into the currently active Python environment."""
    _run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        "Installing Python dependencies",
    )


def _download_embedding_model() -> None:
    """Download and cache the local sentence-transformers embedding model."""
    print("\n=== Downloading local embedding model ===")
    from sentence_transformers import SentenceTransformer

    from config import settings

    SentenceTransformer(settings.EMBEDDING_MODEL)
    print(f"Embedding model '{settings.EMBEDDING_MODEL}' downloaded and cached.")


def _build_knowledge_base() -> None:
    """Run the full ingestion pipeline to build the embedded Qdrant knowledge base."""
    print("\n=== Building the CFD knowledge base ===")
    from src.ingestion.ingest_pipeline import run_ingestion_pipeline

    summary = run_ingestion_pipeline(rebuild_collection=True)
    print(f"Knowledge base built: {summary}")


def main() -> None:
    """Run the full setup sequence: install, download model, build KB."""
    _install_dependencies()
    _download_embedding_model()
    _build_knowledge_base()

    from config import settings

    if settings.LLM_PROVIDER == "ollama":
        llm_step = "Make sure Ollama is running with the model pulled: ollama pull llama3.1:8b"
    elif settings.LLM_PROVIDER == "mistral":
        llm_step = "Make sure MISTRAL_API_KEY is set in .env"
    else:
        llm_step = "Make sure HF_API_TOKEN is set in .env (free at huggingface.co/settings/tokens)"

    print(
        "\n"
        "Setup complete.\n"
        "Next steps:\n"
        f"  1. {llm_step}\n"
        "  2. Run: python run.py\n"
        "  3. Open: http://localhost:8501\n"
    )


if __name__ == "__main__":
    main()
