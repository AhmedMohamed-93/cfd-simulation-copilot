"""Two-stage retrieval: dense Qdrant search followed by local cross-encoder reranking.

Both stages run entirely locally: stage 1 uses the sentence-transformers
embedding model against Qdrant, and stage 2 uses a local cross-encoder
(``cross-encoder/ms-marco-MiniLM-L-6-v2`` by default) to rerank candidates.
No LLM API call and no API key are involved in retrieval at all.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from config import settings
from src.retrieval.embedder import LocalEmbedder
from src.retrieval.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)

_cross_encoder_lock = threading.Lock()
_cross_encoder_cache: dict[str, object] = {}


def _get_cross_encoder(model_name: str):
    """Lazily construct and cache a CrossEncoder for the process.

    Args:
        model_name: The sentence-transformers cross-encoder model identifier.

    Returns:
        A loaded CrossEncoder instance.
    """
    if model_name not in _cross_encoder_cache:
        with _cross_encoder_lock:
            if model_name not in _cross_encoder_cache:
                from sentence_transformers import CrossEncoder  # noqa: PLC0415

                logger.info("Loading local cross-encoder reranker '%s'...", model_name)
                _cross_encoder_cache[model_name] = CrossEncoder(model_name)
    return _cross_encoder_cache[model_name]


def _min_max_normalize(raw_scores: list[float]) -> list[float]:
    """Rescale raw cross-encoder logits to [0, 10] via min-max normalization.

    ``cross-encoder/ms-marco-MiniLM-L-6-v2`` is calibrated on short,
    web-search-style query/passage pairs, where a raw logit of 0 is roughly
    the relevance boundary. Verified empirically against this project's
    actual corpus (long, technical, encyclopedia-style passages) and
    keyword-style queries: a genuinely accurate, on-topic real passage
    scored -4.96, while a hand-written, clearly on-topic one-sentence
    answer scored +0.69. The model's absolute zero-point does not transfer
    to this domain, so a fixed ``sigmoid(raw) * 10`` centered at 0 collapses
    every real result toward 0/10 regardless of how good the reranking
    itself is. Rescaling within each batch instead reports each chunk's
    *relative* standing among the candidates actually retrieved for this
    query, which is what the score is used for (ranking + display) and
    stays meaningful across corpora without a hand-tuned offset.

    This does not change which chunks get selected: the rescaling is
    monotonic, so it preserves the raw cross-encoder's ranking exactly:
    only the displayed 0-10 value changes, not the underlying selection.

    Args:
        raw_scores: Raw cross-encoder logits, one per candidate chunk.

    Returns:
        Scores rescaled to [0, 10], with the batch's best candidate at 10
        and the worst at 0. If every score is equal (or there is only one
        candidate), every chunk gets a neutral 5.0: there is no basis to
        differentiate them.
    """
    if not raw_scores:
        return []
    lo, hi = min(raw_scores), max(raw_scores)
    if hi - lo < 1e-9:
        return [5.0] * len(raw_scores)
    return [(s - lo) / (hi - lo) * 10.0 for s in raw_scores]


_STRONG_MATCH_THRESHOLD = 0.0
_MODERATE_MATCH_THRESHOLD = -5.0


def _match_quality(raw_score: float | None) -> str | None:
    """Classify a raw cross-encoder logit into an absolute relevance band.

    Min-max normalization (see `_min_max_normalize`) is batch-relative by
    design: the best chunk in any batch always displays as 100%, even when
    every candidate in that batch is genuinely irrelevant (e.g. the
    knowledge base has nothing on-topic for this query). This absolute
    floor, based on the raw (pre-normalization) logit, is what lets a
    citation be shown as "100% relevance (weak match)" instead of silently
    implying strong relevance whenever it happens to rank first.

    Args:
        raw_score: The raw cross-encoder logit, or None if unavailable
            (e.g. the cross-encoder failed and dense-score fallback was
            used instead, which is not comparable to a logit).

    Returns:
        "strong" (raw >= 0), "moderate" (-5 <= raw < 0), "weak" (raw < -5),
        or None if no raw score is available.
    """
    if raw_score is None:
        return None
    if raw_score >= _STRONG_MATCH_THRESHOLD:
        return "strong"
    if raw_score >= _MODERATE_MATCH_THRESHOLD:
        return "moderate"
    return "weak"


@dataclass
class RetrievedChunk:
    """A single retrieved and (optionally) reranked knowledge chunk.

    Attributes:
        content: The chunk's text content.
        metadata: The chunk's associated metadata (source, title, tags...).
        dense_score: The raw cosine similarity score from Qdrant.
        rerank_score: The cross-encoder relevance score, min-max normalized
            to 0-10 relative to the other candidates in the same retrieval
            batch (see `_min_max_normalize`): the batch's best match scores
            10, the worst scores 0. Used for ranking/selection and as the
            displayed relevance percentage; batch-relative, not an absolute
            quality signal (see `match_quality`).
        raw_rerank_score: The cross-encoder's raw (pre-normalization) logit,
            or None if the cross-encoder failed and a dense-score fallback
            was used instead. This is what `match_quality` is derived from.
        match_quality: "strong", "moderate", or "weak", an absolute
            relevance band derived from `raw_rerank_score` (see
            `_match_quality`), independent of how this chunk ranks within
            its batch. None if `raw_rerank_score` is unavailable.
    """

    content: str
    metadata: dict[str, Any]
    dense_score: float
    rerank_score: float | None = field(default=None)
    raw_rerank_score: float | None = field(default=None)
    match_quality: str | None = field(default=None)


_HIGH_RETRIEVAL_QUALITY = 0.9
_LOW_RETRIEVAL_QUALITY = 0.3
_NEUTRAL_RETRIEVAL_QUALITY = 0.5


def compute_retrieval_quality(chunks: list[RetrievedChunk]) -> float:
    """Deterministically estimate retrieval quality from cross-encoder scores.

    Replaces an LLM self-grading round-trip (previously one of three LLM
    calls per agent run, on local CPU-bound Ollama ~40s each) with a
    direct read of the reranker's own signal: the mean raw
    (pre-normalization) cross-encoder logit across the retrieved batch. A
    positive mean logit means the reranker itself judged the batch as, on
    average, genuinely relevant (raw logits, not the batch-relative 0-10
    `rerank_score`, are the meaningful absolute signal here (see
    `_min_max_normalize`'s docstring for why); a non-positive mean means
    weak/irrelevant retrieval. This also makes retrieval-quality grading
    deterministic and reproducible, unlike an LLM's free-text self-rating.

    Args:
        chunks: The reranked candidate chunks for this query.

    Returns:
        0.0 if there are no chunks at all; 0.5 (neutral) if none of the
        chunks have a raw score available (e.g. the cross-encoder failed
        and a dense-score fallback was used, which isn't comparable to a
        logit); otherwise 0.9 if the mean raw score is positive, else 0.3.
    """
    if not chunks:
        return 0.0
    raw_scores = [c.raw_rerank_score for c in chunks if c.raw_rerank_score is not None]
    if not raw_scores:
        return _NEUTRAL_RETRIEVAL_QUALITY
    mean_raw_score = sum(raw_scores) / len(raw_scores)
    return _HIGH_RETRIEVAL_QUALITY if mean_raw_score > 0 else _LOW_RETRIEVAL_QUALITY


class CFDRetriever:
    """Two-stage retriever: dense vector search + local cross-encoder reranking.

    Attributes:
        embedder: The local embedding client used to encode queries.
        vector_store: The Qdrant vector store to search against.
    """

    def __init__(
        self,
        embedder: LocalEmbedder | None = None,
        vector_store: QdrantVectorStore | None = None,
        cross_encoder_model: str = settings.CROSS_ENCODER_MODEL,
    ) -> None:
        """Initialize the retriever and its dependencies.

        Args:
            embedder: Embedder instance; a new one is created if omitted.
            vector_store: Vector store instance; a new one is created if
                omitted.
            cross_encoder_model: The local cross-encoder model used for
                stage-2 reranking.
        """
        self.embedder = embedder or LocalEmbedder()
        self.vector_store = vector_store or QdrantVectorStore()
        self._cross_encoder_model = cross_encoder_model

    def _dense_search(
        self,
        query: str,
        top_k: int,
        filter_by_tags: dict[str, Any] | None,
    ) -> tuple[list[RetrievedChunk], float]:
        """Run stage-1 dense retrieval and time it.

        Args:
            query: The search query text.
            top_k: Number of nearest neighbors to fetch.
            filter_by_tags: Optional metadata filter.

        Returns:
            A tuple of (retrieved chunks, elapsed time in seconds).
        """
        start = time.perf_counter()
        query_vector = self.embedder.embed_query(query)
        points = self.vector_store.search(
            query_vector=query_vector, top_k=top_k, filter_by_tags=filter_by_tags
        )
        elapsed = time.perf_counter() - start

        chunks = []
        for point in points:
            payload = dict(point.payload or {})
            content = payload.pop("page_content", "")
            chunks.append(
                RetrievedChunk(
                    content=content, metadata=payload, dense_score=point.score
                )
            )
        logger.info(
            "Stage 1 (dense retrieval): %d chunks in %.3fs", len(chunks), elapsed
        )
        return chunks, elapsed

    def _score_chunks(
        self, query: str, chunks: list[RetrievedChunk]
    ) -> tuple[list[float], list[float | None]]:
        """Score every chunk's relevance to the query using the local cross-encoder.

        Args:
            query: The search query text.
            chunks: The candidate chunks to score.

        Returns:
            A (normalized_scores, raw_scores) tuple, each a list in the same
            order as `chunks`. normalized_scores are in [0, 10], min-max
            normalized across the batch (see `_min_max_normalize`), used
            for ranking/selection and the displayed relevance percentage.
            raw_scores are the pre-normalization cross-encoder logits, used
            for the absolute `match_quality` band (see `_match_quality`).
            Falls back to the dense score (scaled) for normalized_scores,
            and None for every raw_scores entry, if the cross-encoder fails
            to load or run (a dense-score fallback is not comparable to a
            cross-encoder logit).
        """
        try:
            cross_encoder = _get_cross_encoder(self._cross_encoder_model)
            pairs = [(query, chunk.content) for chunk in chunks]
            raw_scores = [float(s) for s in cross_encoder.predict(pairs)]
            logger.debug(
                "Raw cross-encoder scores for query %r: %s",
                query,
                list(zip([c.metadata.get("title") for c in chunks], raw_scores, strict=True)),
            )
            return _min_max_normalize(raw_scores), raw_scores
        except Exception as exc:  # noqa: BLE001
            logger.warning("Cross-encoder reranking failed, falling back to dense scores: %s", exc)
            return [chunk.dense_score * 10.0 for chunk in chunks], [None] * len(chunks)

    def _rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> tuple[list[RetrievedChunk], float]:
        """Run stage-2 local cross-encoder reranking and time it.

        Args:
            query: The search query text.
            chunks: Candidate chunks from stage 1.
            top_k: Number of top chunks to keep after reranking.

        Returns:
            A tuple of (top-k reranked chunks, elapsed time in seconds).
        """
        start = time.perf_counter()
        normalized_scores, raw_scores = self._score_chunks(query, chunks)
        for chunk, norm_score, raw_score in zip(chunks, normalized_scores, raw_scores, strict=True):
            chunk.rerank_score = norm_score
            chunk.raw_rerank_score = raw_score
            chunk.match_quality = _match_quality(raw_score)
        elapsed = time.perf_counter() - start

        ranked = sorted(chunks, key=lambda c: c.rerank_score or 0.0, reverse=True)
        logger.info(
            "Stage 2 (cross-encoder rerank): scored %d chunks in %.3fs", len(chunks), elapsed
        )
        return ranked[:top_k], elapsed

    def retrieve(
        self,
        query: str,
        top_k_dense: int = settings.TOP_K_RETRIEVAL,
        top_k_reranked: int = settings.TOP_K_RERANKED,
        filter_by_tags: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the full two-stage retrieval pipeline for a query.

        Args:
            query: The natural language search query.
            top_k_dense: Number of candidates to fetch from Qdrant.
            top_k_reranked: Number of chunks to keep after reranking.
            filter_by_tags: Optional metadata filter (solver_type,
                turbulence_model, flow_regime).

        Returns:
            A dict with keys: chunks (list[RetrievedChunk]),
            dense_latency_s, rerank_latency_s, total_latency_s.
        """
        dense_chunks, dense_latency = self._dense_search(
            query, top_k_dense, filter_by_tags
        )
        if not dense_chunks:
            return {
                "chunks": [],
                "dense_latency_s": dense_latency,
                "rerank_latency_s": 0.0,
                "total_latency_s": dense_latency,
            }
        reranked_chunks, rerank_latency = self._rerank(
            query, dense_chunks, top_k_reranked
        )
        return {
            "chunks": reranked_chunks,
            "dense_latency_s": dense_latency,
            "rerank_latency_s": rerank_latency,
            "total_latency_s": dense_latency + rerank_latency,
        }
