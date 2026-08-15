"""Tests for the retrieval layer: local embedder caching, vector store
filtering, and local cross-encoder reranking (no network/API calls)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.retrieval.embedder import LocalEmbedder
from src.retrieval.retriever import CFDRetriever, RetrievedChunk, _match_quality
from src.retrieval.vector_store import QdrantVectorStore


def test_embedded_qdrant_create_upsert_search_roundtrip(tmp_path):
    """The embedded (serverless) Qdrant client supports a full create->upsert->search cycle."""
    store = QdrantVectorStore(
        path=str(tmp_path / "qdrant_storage"),
        collection_name="test_collection",
        vector_size=4,
    )
    store.create_collection(recreate=True)

    docs = [
        Document(page_content="pipe flow turbulence content", metadata={"title": "doc-a"}),
        Document(page_content="airfoil aerodynamics content", metadata={"title": "doc-b"}),
    ]
    vectors = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    indexed = store.upsert_documents(docs, vectors)
    assert indexed == 2

    info = store.get_collection_info()
    assert info["exists"] is True
    assert info["points_count"] == 2

    results = store.search(query_vector=[1.0, 0.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].payload["title"] == "doc-a"


def test_vector_store_build_filter_with_valid_tags():
    """_build_filter accepts recognized metadata tag keys."""
    store = QdrantVectorStore()
    result = store._build_filter({"solver_type": "simpleFoam"})
    assert result is not None
    assert len(result.must) == 1


def test_vector_store_build_filter_ignores_unknown_tags():
    """_build_filter silently drops keys outside the allowed filterable set."""
    store = QdrantVectorStore()
    result = store._build_filter({"not_a_real_tag": "value"})
    assert result is None


def test_vector_store_build_filter_none_for_empty_input():
    """_build_filter returns None when no filter dict is provided."""
    store = QdrantVectorStore()
    assert store._build_filter(None) is None
    assert store._build_filter({}) is None


def test_embedder_cache_roundtrip(tmp_path):
    """Embeddings served from cache do not trigger a new model encode call."""
    cache_path = tmp_path / "cache.json"
    embedder = LocalEmbedder(cache_path=str(cache_path))
    text_hash = embedder._hash_text("hello world")
    embedder._cache[text_hash] = [0.1, 0.2, 0.3]
    embedder._save_cache()

    reloaded = LocalEmbedder(cache_path=str(cache_path))
    assert text_hash in reloaded._cache
    assert reloaded._cache[text_hash] == [0.1, 0.2, 0.3]


def test_embedder_embed_documents_skips_model_call_for_cached_text(tmp_path):
    """embed_documents does not re-run the local model for already-cached text."""
    cache_path = tmp_path / "cache.json"
    embedder = LocalEmbedder(cache_path=str(cache_path))
    with patch.object(embedder, "_encode_batch") as mock_encode:
        mock_encode.return_value = [[1.0, 2.0]]
        first = embedder.embed_documents(["some text"])
        assert mock_encode.call_count == 1

        second = embedder.embed_documents(["some text"])
        assert mock_encode.call_count == 1
        assert first == second


def test_retriever_score_chunks_falls_back_on_cross_encoder_failure():
    """_score_chunks falls back to dense-score-derived values if the local model errors."""
    embedder = MagicMock()
    vector_store = MagicMock()
    retriever = CFDRetriever(embedder=embedder, vector_store=vector_store)

    with patch("src.retrieval.retriever._get_cross_encoder") as mock_get_ce:
        mock_get_ce.side_effect = RuntimeError("model failed to load")
        chunk = RetrievedChunk(content="some content", metadata={}, dense_score=0.8)
        normalized, raw = retriever._score_chunks("query", [chunk])

    assert normalized == pytest.approx([8.0])
    assert raw == [None]


def test_retriever_score_chunks_min_max_normalizes_across_batch():
    """_score_chunks min-max normalizes raw logits to [0, 10] within the batch.

    Regression test: a plain sigmoid(raw) * 10 centered at 0 collapsed
    every real-world result toward 0/10, since this project's actual
    corpus (long technical passages + keyword-style queries) reliably
    produces negative cross-encoder logits even for genuinely relevant
    matches — verified empirically (see _min_max_normalize's docstring).
    Min-max normalizing within the batch keeps the score meaningful
    regardless of the corpus's absolute logit range: the best candidate in
    *any* batch reaches 10, the worst reaches 0, matching how the score is
    actually used (ranking + display).
    """
    embedder = MagicMock()
    vector_store = MagicMock()
    retriever = CFDRetriever(embedder=embedder, vector_store=vector_store)

    mock_cross_encoder = MagicMock()
    # All-negative logits, as observed against the real corpus — a plain
    # sigmoid would have squashed every one of these toward 0/10.
    mock_cross_encoder.predict.return_value = [-10.0, -5.0, 0.0]

    chunks = [
        RetrievedChunk(content="worst match", metadata={}, dense_score=0.1),
        RetrievedChunk(content="middle match", metadata={}, dense_score=0.2),
        RetrievedChunk(content="best match", metadata={}, dense_score=0.3),
    ]
    with patch("src.retrieval.retriever._get_cross_encoder", return_value=mock_cross_encoder):
        normalized, raw = retriever._score_chunks("query", chunks)

    assert normalized == pytest.approx([0.0, 5.0, 10.0])
    assert raw == pytest.approx([-10.0, -5.0, 0.0])


def test_retriever_score_chunks_neutral_when_scores_indistinguishable():
    """_score_chunks returns a neutral 5.0 for every chunk when all logits are equal."""
    embedder = MagicMock()
    vector_store = MagicMock()
    retriever = CFDRetriever(embedder=embedder, vector_store=vector_store)

    mock_cross_encoder = MagicMock()
    mock_cross_encoder.predict.return_value = [-3.0, -3.0]

    chunks = [
        RetrievedChunk(content="a", metadata={}, dense_score=0.1),
        RetrievedChunk(content="b", metadata={}, dense_score=0.1),
    ]
    with patch("src.retrieval.retriever._get_cross_encoder", return_value=mock_cross_encoder):
        normalized, raw = retriever._score_chunks("query", chunks)

    assert normalized == pytest.approx([5.0, 5.0])
    assert raw == pytest.approx([-3.0, -3.0])


def test_retriever_score_chunks_preserves_raw_ranking_order():
    """Min-max normalization is monotonic: it never changes which chunk ranks best.

    This is the property that makes the rescaling safe to apply purely as
    a display fix — it cannot change which chunks get selected as
    citations, only the number shown next to them.
    """
    embedder = MagicMock()
    vector_store = MagicMock()
    retriever = CFDRetriever(embedder=embedder, vector_store=vector_store)

    mock_cross_encoder = MagicMock()
    # Raw logits: chunk "best" has the highest (least negative) score.
    mock_cross_encoder.predict.return_value = [-2.0, -8.0, -4.0]

    chunks = [
        RetrievedChunk(content="best", metadata={}, dense_score=0.1),
        RetrievedChunk(content="worst", metadata={}, dense_score=0.1),
        RetrievedChunk(content="middle", metadata={}, dense_score=0.1),
    ]
    with patch("src.retrieval.retriever._get_cross_encoder", return_value=mock_cross_encoder):
        normalized, _raw = retriever._score_chunks("query", chunks)

    scored = list(zip(chunks, normalized, strict=True))
    ranked = [c.content for c, _ in sorted(scored, key=lambda pair: pair[1], reverse=True)]
    assert ranked == ["best", "middle", "worst"]

    score_by_content = {c.content: s for c, s in scored}
    assert score_by_content["best"] == 10.0
    assert score_by_content["worst"] == 0.0


def test_retriever_retrieve_returns_empty_when_no_dense_hits():
    """retrieve() short-circuits and skips reranking when stage 1 finds nothing."""
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.0] * 384
    vector_store = MagicMock()
    vector_store.search.return_value = []

    retriever = CFDRetriever(embedder=embedder, vector_store=vector_store)
    result = retriever.retrieve("empty query")
    assert result["chunks"] == []
    assert result["rerank_latency_s"] == 0.0


def test_match_quality_strong_for_nonnegative_raw_score():
    """A raw cross-encoder logit >= 0 is classified as a strong match."""
    assert _match_quality(0.0) == "strong"
    assert _match_quality(3.2) == "strong"


def test_match_quality_moderate_for_raw_score_in_middle_band():
    """A raw logit in [-5, 0) is classified as a moderate match."""
    assert _match_quality(-0.01) == "moderate"
    assert _match_quality(-5.0) == "moderate"
    assert _match_quality(-2.5) == "moderate"


def test_match_quality_weak_for_raw_score_below_negative_five():
    """A raw logit < -5 is classified as a weak match.

    Regression test: min-max normalization alone would show a chunk like
    this at 99-100% relevance whenever it happens to be the best of a bad
    batch (e.g. "Reynolds stress model" wiki pages surfacing for a laminar
    pipe flow query) — match_quality is the absolute floor that catches
    that case regardless of batch-relative rank.
    """
    assert _match_quality(-5.01) == "weak"
    assert _match_quality(-10.0) == "weak"


def test_match_quality_none_when_raw_score_unavailable():
    """No raw score (cross-encoder fallback) means no quality band, not a guess."""
    assert _match_quality(None) is None


def test_retriever_rerank_sets_raw_score_and_match_quality_on_chunks():
    """_rerank populates raw_rerank_score and match_quality alongside rerank_score."""
    embedder = MagicMock()
    vector_store = MagicMock()
    retriever = CFDRetriever(embedder=embedder, vector_store=vector_store)

    mock_cross_encoder = MagicMock()
    # strong, moderate, weak, in that order
    mock_cross_encoder.predict.return_value = [1.0, -2.0, -8.0]

    chunks = [
        RetrievedChunk(content="strong", metadata={}, dense_score=0.1),
        RetrievedChunk(content="moderate", metadata={}, dense_score=0.1),
        RetrievedChunk(content="weak", metadata={}, dense_score=0.1),
    ]
    with patch("src.retrieval.retriever._get_cross_encoder", return_value=mock_cross_encoder):
        ranked, _elapsed = retriever._rerank("query", chunks, top_k=3)

    by_content = {c.content: c for c in ranked}
    assert by_content["strong"].raw_rerank_score == pytest.approx(1.0)
    assert by_content["strong"].match_quality == "strong"
    assert by_content["moderate"].raw_rerank_score == pytest.approx(-2.0)
    assert by_content["moderate"].match_quality == "moderate"
    assert by_content["weak"].raw_rerank_score == pytest.approx(-8.0)
    assert by_content["weak"].match_quality == "weak"
    # rerank_score (batch-relative) still spans the full 0-10 range
    # regardless of the absolute quality band — that's the bug being fixed.
    assert by_content["strong"].rerank_score == pytest.approx(10.0)


def test_retriever_rerank_match_quality_none_on_cross_encoder_failure():
    """match_quality is None (not a misleading guess) when the cross-encoder failed."""
    embedder = MagicMock()
    vector_store = MagicMock()
    retriever = CFDRetriever(embedder=embedder, vector_store=vector_store)

    with patch("src.retrieval.retriever._get_cross_encoder") as mock_get_ce:
        mock_get_ce.side_effect = RuntimeError("model failed to load")
        chunk = RetrievedChunk(content="some content", metadata={}, dense_score=0.8)
        ranked, _elapsed = retriever._rerank("query", [chunk], top_k=1)

    assert ranked[0].match_quality is None
    assert ranked[0].raw_rerank_score is None
