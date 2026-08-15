"""Embedded (serverless) Qdrant vector store operations for the CFD knowledge base.

Uses Qdrant's local/embedded mode (`QdrantClient(path=...)`), which persists
vectors directly to a folder on disk: no Docker, no server process, and no
network connection required. Embedded-mode storage can only be opened by one
QdrantClient at a time per path, so every QdrantVectorStore in this process
shares a single process-wide client instance for a given path rather than
each opening its own (which would otherwise raise a storage-lock error).
"""

from __future__ import annotations

import logging
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from config import settings

logger = logging.getLogger(__name__)

_UPSERT_BATCH_SIZE = 50
_FILTERABLE_TAGS = {"solver_type", "turbulence_model", "flow_regime"}

_client_lock = threading.Lock()
_client_cache: dict[str, QdrantClient] = {}


def _get_local_qdrant_client(path: str) -> QdrantClient:
    """Return a process-wide singleton embedded QdrantClient for `path`.

    Args:
        path: Filesystem directory to persist/open the embedded Qdrant
            storage at.

    Returns:
        A QdrantClient instance shared by every caller that requests the
        same path within this process.
    """
    if path not in _client_cache:
        with _client_lock:
            if path not in _client_cache:
                Path(path).mkdir(parents=True, exist_ok=True)
                logger.info("Opening embedded Qdrant storage at '%s'.", path)
                _client_cache[path] = QdrantClient(path=path)
    return _client_cache[path]


class QdrantVectorStore:
    """Thin wrapper around the embedded Qdrant client for the CFD knowledge base.

    Attributes:
        collection_name: Name of the Qdrant collection used for the KB.
        vector_size: Dimensionality of stored embedding vectors.
    """

    def __init__(
        self,
        path: str = settings.QDRANT_LOCAL_PATH,
        collection_name: str = settings.QDRANT_COLLECTION_NAME,
        vector_size: int = settings.EMBEDDING_DIM,
    ) -> None:
        """Initialize the vector store against the embedded Qdrant instance.

        Args:
            path: Filesystem directory the embedded Qdrant storage lives in.
            collection_name: Name of the collection to use.
            vector_size: Dimensionality of the embedding vectors stored.
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._storage_path = path
        self._client = _get_local_qdrant_client(path)

    def create_collection(self, recreate: bool = False) -> None:
        """Create the KB collection if it does not already exist.

        Args:
            recreate: If True, delete and recreate the collection even if
                it already exists (used for a full reindex).
        """
        exists = self._client.collection_exists(self.collection_name)
        if exists and not recreate:
            logger.info("Collection '%s' already exists.", self.collection_name)
            return
        if exists and recreate:
            self._client.delete_collection(self.collection_name)
            # qdrant-client's embedded/local storage mode has been observed
            # to not fully purge a deleted collection's on-disk segment data
            # (Storage/RocksDB deletes there are logical, not physical);
            # stale points can otherwise resurface on the next fresh client
            # open in a new process, silently duplicating the "rebuilt" KB.
            # Removing the on-disk collection directory outright guarantees
            # a genuinely clean slate regardless of that internal behavior.
            collection_dir = Path(self._storage_path) / "collection" / self.collection_name
            if collection_dir.exists():
                shutil.rmtree(collection_dir, ignore_errors=True)
            logger.info("Deleted existing collection '%s'.", self.collection_name)

        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(
                size=self.vector_size, distance=qmodels.Distance.COSINE
            ),
        )
        logger.info("Created collection '%s'.", self.collection_name)

    def upsert_documents(
        self, docs: list[Document], vectors: list[list[float]]
    ) -> int:
        """Upsert chunked documents and their embeddings into Qdrant.

        Args:
            docs: Chunked LangChain Document objects to index.
            vectors: Embedding vectors, one per document, in the same order.

        Returns:
            The total number of points upserted.

        Raises:
            ValueError: If docs and vectors have mismatched lengths.
        """
        if len(docs) != len(vectors):
            raise ValueError(
                f"docs ({len(docs)}) and vectors ({len(vectors)}) length mismatch"
            )

        total = 0
        for start in range(0, len(docs), _UPSERT_BATCH_SIZE):
            batch_docs = docs[start : start + _UPSERT_BATCH_SIZE]
            batch_vectors = vectors[start : start + _UPSERT_BATCH_SIZE]
            points = [
                qmodels.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={"page_content": doc.page_content, **doc.metadata},
                )
                for doc, vector in zip(batch_docs, batch_vectors, strict=False)
            ]
            self._client.upsert(collection_name=self.collection_name, points=points)
            total += len(points)
        logger.info("Upserted %d points into '%s'.", total, self.collection_name)
        return total

    def _build_filter(
        self, filter_by_tags: dict[str, Any] | None
    ) -> qmodels.Filter | None:
        """Build a Qdrant filter from a dict of allowed metadata tags.

        Args:
            filter_by_tags: Mapping restricted to solver_type,
                turbulence_model, and/or flow_regime keys.

        Returns:
            A Qdrant Filter object, or None if no valid tags were provided.
        """
        if not filter_by_tags:
            return None
        conditions = [
            qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=value))
            for key, value in filter_by_tags.items()
            if key in _FILTERABLE_TAGS and value is not None
        ]
        if not conditions:
            return None
        return qmodels.Filter(must=conditions)

    def search(
        self,
        query_vector: list[float],
        top_k: int = settings.TOP_K_RETRIEVAL,
        filter_by_tags: dict[str, Any] | None = None,
    ) -> list[qmodels.ScoredPoint]:
        """Search the collection for the nearest neighbors of a query vector.

        Args:
            query_vector: The embedding vector of the query.
            top_k: Number of nearest neighbors to return.
            filter_by_tags: Optional metadata filter restricted to
                solver_type, turbulence_model, and/or flow_regime.

        Returns:
            A list of ScoredPoint results ordered by decreasing similarity.
        """
        query_filter = self._build_filter(filter_by_tags)
        results = self._client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return results.points

    def get_collection_info(self) -> dict[str, Any]:
        """Return health/status information about the KB collection.

        Returns:
            A dict with keys: exists, points_count, vector_size, status.
        """
        if not self._client.collection_exists(self.collection_name):
            return {
                "exists": False,
                "points_count": 0,
                "vector_size": self.vector_size,
                "status": "not_created",
            }
        info = self._client.get_collection(self.collection_name)
        return {
            "exists": True,
            "points_count": info.points_count,
            "vector_size": self.vector_size,
            "status": info.status,
        }
