"""Tests for the document ingestion pipeline: loaders and chunker."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from config import settings
from src.ingestion import document_loader
from src.ingestion.chunker import chunk_document, chunk_documents
from src.ingestion.document_loader import RawDocument, _extract_article_text
from src.ingestion.ingest_pipeline import _reset_local_vector_storage


def test_synthetic_user_guide_fallback_when_network_fails():
    """load_openfoam_user_guide falls back to synthetic docs on network failure."""
    with patch.object(document_loader, "_safe_get", return_value=None):
        docs = document_loader.load_openfoam_user_guide()
    assert len(docs) > 0
    assert all(d.source == "openfoam-user-guide-synthetic" for d in docs)


def test_synthetic_tutorials_fallback_when_network_fails():
    """load_openfoam_tutorials falls back to a synthetic catalogue on failure."""
    with patch.object(document_loader, "_safe_get", return_value=None):
        docs = document_loader.load_openfoam_tutorials()
    assert len(docs) > 0
    assert all("solver" in d.topic_tags for d in docs)


def test_load_openfoam_tutorials_extracts_only_real_named_cases():
    """load_openfoam_tutorials keeps only depth-4 case dirs, not category/solver dirs.

    Regression test for the git-trees-based rewrite: the tree includes
    entries at every depth (tutorials/, tutorials/incompressible/,
    tutorials/incompressible/simpleFoam/, tutorials/incompressible/simpleFoam/pitzDaily/,
    files, etc.) — only the last of those (an actual named tutorial case)
    should become a document.
    """
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "tree": [
            {"path": "tutorials", "type": "tree"},
            {"path": "tutorials/incompressible", "type": "tree"},
            {"path": "tutorials/incompressible/simpleFoam", "type": "tree"},
            {"path": "tutorials/incompressible/simpleFoam/pitzDaily", "type": "tree"},
            {"path": "tutorials/incompressible/simpleFoam/pitzDaily/0/U", "type": "blob"},
            {"path": "tutorials/incompressible/pimpleFoam/TJunction", "type": "tree"},
        ]
    }
    with patch.object(document_loader, "_safe_get", return_value=fake_response):
        docs = document_loader.load_openfoam_tutorials()
    assert len(docs) == 2
    titles = {d.title for d in docs}
    assert "OpenFOAM Tutorial: tutorials/incompressible/simpleFoam/pitzDaily" in titles
    assert "OpenFOAM Tutorial: tutorials/incompressible/pimpleFoam/TJunction" in titles
    assert all(d.source == "openfoam-tutorials-github" for d in docs)


def test_synthetic_cfd_online_wiki_fallback_covers_all_models():
    """load_cfd_online_wiki falls back to synthetic summaries for every model."""
    with patch.object(document_loader, "_safe_get", return_value=None):
        docs = document_loader.load_cfd_online_wiki()
    model_names = {d.metadata.get("model") for d in docs}
    assert model_names == set(document_loader.CFD_ONLINE_WIKI_PAGES.keys())


def test_synthetic_arxiv_fallback_respects_max_results():
    """load_arxiv_papers falls back to synthetic entries capped at max_results."""
    with patch.object(document_loader, "_safe_get", return_value=None):
        docs = document_loader.load_arxiv_papers(max_results=5)
    assert len(docs) == 5


def test_extract_article_text_strips_scripts_and_styles():
    """_extract_article_text drops non-content markup and keeps paragraph prose."""
    html = """
    <html><head><script>evil()</script><style>.x{color:red}</style></head>
    <body><nav><a href="#">menu item</a></nav>
    <p>Real article sentence one.</p>
    <p>Real article sentence two.</p>
    </body></html>
    """
    text = _extract_article_text(html)
    assert "evil()" not in text
    assert "color:red" not in text
    assert "Real article sentence one." in text
    assert "Real article sentence two." in text


def test_extract_article_text_scopes_to_content_selector():
    """_extract_article_text ignores content outside the given selector.

    Regression test: without scoping, site-wide navigation megamenus (far
    larger than the real article) drown out the actual content.
    """
    html = """
    <html><body>
    <div id="menu"><p>huge sitewide menu paragraph that is not the article</p></div>
    <div id="bodyContent"><p>The actual article content.</p></div>
    </body></html>
    """
    text = _extract_article_text(html, content_selector="#bodyContent")
    assert text == "The actual article content."


def test_load_openfoam_user_guide_falls_back_when_page_is_content_free():
    """A 200 OK response with no real paragraph content still triggers the fallback.

    Regression test: the live page returned 200 but was a JS-rendered shell
    with zero <p> tags; raw response.text was non-empty (all nav markup),
    so the old "response.text truthy" check wrongly accepted it as real.
    """
    fake_response = MagicMock()
    fake_response.text = "<html><body><ul><li><a href='#'>Just a nav link</a></li></ul></body></html>"
    with patch.object(document_loader, "_safe_get", return_value=fake_response):
        docs = document_loader.load_openfoam_user_guide()
    assert all(d.source == "openfoam-user-guide-synthetic" for d in docs)


def test_load_cfd_online_wiki_uses_real_content_when_substantial():
    """A page with real substantial article prose in #bodyContent is used as-is."""
    real_paragraph = "This is genuine turbulence model documentation text. " * 10
    fake_response = MagicMock()
    fake_response.text = f"<html><body><div id='bodyContent'><p>{real_paragraph}</p></div></body></html>"
    with patch.object(document_loader, "_safe_get", return_value=fake_response):
        docs = document_loader.load_cfd_online_wiki()
    assert all(d.source == "cfd-online-wiki" for d in docs)
    assert all("genuine turbulence model documentation" in d.content for d in docs)


def test_load_all_documents_aggregates_all_sources():
    """load_all_documents concatenates results from every loader."""
    with patch.object(document_loader, "_safe_get", return_value=None):
        docs = document_loader.load_all_documents()
    sources = {d.source for d in docs}
    assert "openfoam-user-guide-synthetic" in sources
    assert "cfd-online-wiki-synthetic" in sources
    assert "arxiv-synthetic" in sources


def test_chunk_document_preserves_metadata_and_adds_chunk_fields():
    """chunk_document propagates parent metadata and adds chunk-level fields."""
    raw = RawDocument(
        content="This is a sentence. " * 200,
        source="unit-test",
        title="Test Document",
        topic_tags=["solver"],
        difficulty_level="beginner",
    )
    chunks = chunk_document(raw, chunk_size=64, chunk_overlap=8)
    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["source"] == "unit-test"
        assert chunk.metadata["title"] == "Test Document"
        assert chunk.metadata["chunk_index"] == i
        assert chunk.metadata["total_chunks"] == len(chunks)
        assert "document_id" in chunk.metadata


def test_chunk_documents_flattens_across_multiple_docs():
    """chunk_documents returns a flat list spanning all input documents."""
    raw_docs = [
        RawDocument(content="Alpha content. " * 50, source="a", title="A"),
        RawDocument(content="Beta content. " * 50, source="b", title="B"),
    ]
    chunks = chunk_documents(raw_docs)
    doc_ids = {c.metadata["document_id"] for c in chunks}
    assert len(doc_ids) == 2


def test_reset_local_vector_storage_closes_cached_client_and_wipes_directory(tmp_path, monkeypatch):
    """_reset_local_vector_storage evicts+closes any cached client and removes the directory.

    Regression test for a reproducible Windows bug: qdrant-client's
    embedded/local mode can leave a deleted collection's SQLite file locked
    even after delete_collection() + client.close(), silently letting stale
    points survive a "rebuild". Wiping the whole directory upfront (before
    any client in the process has opened it) is the reliable fix.
    """
    storage_dir = tmp_path / "qdrant_storage"
    storage_dir.mkdir()
    (storage_dir / "meta.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(settings, "QDRANT_LOCAL_PATH", str(storage_dir))

    from src.retrieval import vector_store as vector_store_module

    fake_client = MagicMock()
    vector_store_module._client_cache[str(storage_dir)] = fake_client

    _reset_local_vector_storage()

    fake_client.close.assert_called_once()
    assert str(storage_dir) not in vector_store_module._client_cache
    assert not storage_dir.exists()


def test_reset_local_vector_storage_noop_when_nothing_to_clean(tmp_path, monkeypatch):
    """_reset_local_vector_storage does nothing (no error) if storage never existed."""
    storage_dir = tmp_path / "does_not_exist"
    monkeypatch.setattr(settings, "QDRANT_LOCAL_PATH", str(storage_dir))

    _reset_local_vector_storage()  # must not raise

    assert not storage_dir.exists()
