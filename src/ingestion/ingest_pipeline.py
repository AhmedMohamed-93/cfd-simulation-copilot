"""Orchestrates the full ingestion pipeline: load -> chunk -> embed -> index."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from langchain_core.documents import Document

from config import settings
from src.ingestion.chunker import chunk_documents
from src.ingestion.document_loader import RawDocument, load_all_documents
from src.retrieval.embedder import LocalEmbedder
from src.retrieval.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


def _reset_local_vector_storage() -> None:
    """Physically wipe the embedded Qdrant storage directory for a clean rebuild.

    QdrantVectorStore.create_collection(recreate=True) alone is not always
    sufficient: qdrant-client's embedded/local storage mode has been
    observed (reproducibly, on Windows) to leave the previous collection's
    on-disk SQLite file locked even after delete_collection() and an
    explicit client.close(), meaning stale points can silently survive a
    "rebuild" and merge with freshly indexed ones under a fresh set of
    point IDs, roughly doubling the collection instead of replacing it.
    Removing the entire storage directory before any client in this
    process opens it sidesteps the issue entirely: there is nothing left
    to hold a lock on.
    """
    from src.retrieval import vector_store as _vector_store_module  # noqa: PLC0415

    cached_client = _vector_store_module._client_cache.pop(settings.QDRANT_LOCAL_PATH, None)
    if cached_client is not None:
        try:
            cached_client.close()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to close cached Qdrant client before reset: %s", exc)

    storage_path = Path(settings.QDRANT_LOCAL_PATH)
    if storage_path.exists():
        shutil.rmtree(storage_path, ignore_errors=True)
        logger.info(
            "Removed existing embedded Qdrant storage at '%s' for a clean rebuild.", storage_path
        )


def _persist_raw_documents(raw_docs: list[RawDocument]) -> None:
    """Persist raw documents to data/raw as JSON for auditability.

    Args:
        raw_docs: The raw documents to persist.
    """
    raw_dir = Path(settings.DATA_RAW_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "content": d.content,
            "source": d.source,
            "title": d.title,
            "topic_tags": d.topic_tags,
            "difficulty_level": d.difficulty_level,
            "metadata": d.metadata,
        }
        for d in raw_docs
    ]
    (raw_dir / "raw_documents.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _persist_chunks(chunks: list[Document]) -> None:
    """Persist processed chunks to data/processed as JSON.

    Args:
        chunks: The chunked documents to persist.
    """
    processed_dir = Path(settings.DATA_PROCESSED_DIR)
    processed_dir.mkdir(parents=True, exist_ok=True)
    payload = [
        {"page_content": c.page_content, "metadata": c.metadata} for c in chunks
    ]
    (processed_dir / "chunks.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def run_ingestion_pipeline(rebuild_collection: bool = False) -> dict[str, int]:
    """Run the full ingestion pipeline end-to-end.

    Loads documents from all configured sources, chunks them, embeds the
    chunks via the local sentence-transformers embedding model, and upserts
    them into Qdrant.

    Args:
        rebuild_collection: If True, drop and recreate the Qdrant collection
            before indexing.

    Returns:
        A summary dict with counts: raw_documents, chunks, indexed.
    """
    logger.info("Starting ingestion pipeline (rebuild_collection=%s).", rebuild_collection)

    if rebuild_collection:
        _reset_local_vector_storage()

    raw_docs = load_all_documents()
    _persist_raw_documents(raw_docs)

    chunks = chunk_documents(raw_docs)
    _persist_chunks(chunks)

    embedder = LocalEmbedder()
    vector_store = QdrantVectorStore()
    vector_store.create_collection(recreate=rebuild_collection)

    texts = [c.page_content for c in chunks]
    vectors = embedder.embed_documents(texts)

    indexed = vector_store.upsert_documents(chunks, vectors)

    summary = {
        "raw_documents": len(raw_docs),
        "chunks": len(chunks),
        "indexed": indexed,
    }
    logger.info("Ingestion pipeline complete: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_ingestion_pipeline(rebuild_collection=True)
    print(json.dumps(result, indent=2))
