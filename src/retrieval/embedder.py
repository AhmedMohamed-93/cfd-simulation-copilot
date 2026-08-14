"""Local embedding client backed by sentence-transformers, with disk caching.

Runs 100% locally: the model weights download once from Hugging Face on
first use and are cached by sentence-transformers/huggingface_hub in the
usual local cache directory, after which no network access is required.
No API key of any kind is needed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path

from config import settings

logger = logging.getLogger(__name__)

_BATCH_SIZE = 32

_model_lock = threading.Lock()
_model_cache: dict[str, object] = {}


def _get_sentence_transformer(model_name: str):
    """Lazily construct and cache a SentenceTransformer for the process.

    Loading a transformer model is expensive, so a single instance is
    shared across every LocalEmbedder created with the same model name.

    Args:
        model_name: The sentence-transformers model identifier.

    Returns:
        A loaded SentenceTransformer instance.
    """
    if model_name not in _model_cache:
        with _model_lock:
            if model_name not in _model_cache:
                from sentence_transformers import SentenceTransformer  # noqa: PLC0415

                logger.info("Loading local embedding model '%s'...", model_name)
                _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


class LocalEmbedder:
    """Wraps a local sentence-transformers model with batching and caching.

    Attributes:
        model_name: The sentence-transformers embedding model identifier.
        cache_path: Path to the local JSON embedding cache.
    """

    def __init__(
        self,
        model_name: str = settings.EMBEDDING_MODEL,
        cache_path: str = settings.EMBEDDING_CACHE_PATH,
    ) -> None:
        """Initialize the embedder.

        Args:
            model_name: The sentence-transformers embedding model identifier.
            cache_path: Path to a local JSON file used to cache embeddings
                keyed by a hash of their input text, avoiding recomputation
                across restarts.
        """
        self.model_name = model_name
        self.cache_path = Path(cache_path)
        self._cache: dict[str, list[float]] = self._load_cache()

    @property
    def _model(self):
        """The underlying (lazily loaded, process-wide shared) SentenceTransformer."""
        return _get_sentence_transformer(self.model_name)

    def _load_cache(self) -> dict[str, list[float]]:
        """Load the embedding cache from disk if it exists.

        Returns:
            A mapping from text hash to embedding vector, empty if no cache
            file is present or it fails to parse.
        """
        if not self.cache_path.exists():
            return {}
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load embedding cache: %s", exc)
            return {}

    def _save_cache(self) -> None:
        """Persist the in-memory embedding cache to disk."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self._cache), encoding="utf-8")

    @staticmethod
    def _hash_text(text: str) -> str:
        """Compute a stable cache key for a piece of text.

        Args:
            text: The text to hash.

        Returns:
            A hex digest uniquely identifying the text.
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _encode_batch(self, batch: list[str]) -> list[list[float]]:
        """Run the local model over one batch of texts.

        Args:
            batch: A batch of input texts.

        Returns:
            The list of embedding vectors, in the same order as the batch.
        """
        vectors = self._model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        return [vector.tolist() for vector in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents, using the cache and batching by 32.

        Args:
            texts: The document texts to embed.

        Returns:
            A list of embedding vectors, one per input text, in order.
        """
        results: list[list[float] | None] = [None] * len(texts)
        to_embed_indices: list[int] = []
        to_embed_texts: list[str] = []

        for i, text in enumerate(texts):
            cache_key = self._hash_text(text)
            if cache_key in self._cache:
                results[i] = self._cache[cache_key]
            else:
                to_embed_indices.append(i)
                to_embed_texts.append(text)

        for start in range(0, len(to_embed_texts), _BATCH_SIZE):
            batch = to_embed_texts[start : start + _BATCH_SIZE]
            batch_indices = to_embed_indices[start : start + _BATCH_SIZE]
            embeddings = self._encode_batch(batch)
            for idx, text, vector in zip(batch_indices, batch, embeddings, strict=False):
                results[idx] = vector
                self._cache[self._hash_text(text)] = vector

        if to_embed_texts:
            self._save_cache()

        return [r for r in results if r is not None]

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.

        Args:
            text: The query text to embed.

        Returns:
            The embedding vector for the query.
        """
        return self.embed_documents([text])[0]
