"""CLI entry point to build the full CFD knowledge base from scratch.

Usage:
    python knowledge_base/setup_knowledge_base.py [--rebuild]

Loads documents from all configured sources (with synthetic fallbacks),
chunks them, embeds them via the Mistral API, and indexes them into Qdrant.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.ingest_pipeline import run_ingestion_pipeline  # noqa: E402


def main() -> None:
    """Parse CLI arguments and run the knowledge base ingestion pipeline."""
    parser = argparse.ArgumentParser(description="Build the CFD knowledge base.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Drop and recreate the Qdrant collection before indexing.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = run_ingestion_pipeline(rebuild_collection=args.rebuild)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
