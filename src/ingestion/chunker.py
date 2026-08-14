"""Semantic chunking of raw documents into LangChain Document objects."""

from __future__ import annotations

import hashlib
import logging

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from src.ingestion.document_loader import RawDocument

logger = logging.getLogger(__name__)

_ENCODING = tiktoken.get_encoding("cl100k_base")


def _token_length(text: str) -> int:
    """Count tokens in text using the cl100k_base tokenizer.

    Args:
        text: The text to tokenize.

    Returns:
        The number of tokens in the text.
    """
    return len(_ENCODING.encode(text))


def _document_id(raw_doc: RawDocument) -> str:
    """Derive a stable document id from a raw document's title and source.

    Args:
        raw_doc: The raw document to derive an id for.

    Returns:
        A short, stable hex digest identifying the document.
    """
    key = f"{raw_doc.source}::{raw_doc.title}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def chunk_document(
    raw_doc: RawDocument,
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
) -> list[Document]:
    """Split a single raw document into overlapping, metadata-rich chunks.

    Args:
        raw_doc: The raw document to chunk.
        chunk_size: Target chunk size in tokens.
        chunk_overlap: Token overlap between consecutive chunks.

    Returns:
        A list of LangChain Document objects, each carrying the parent
        document's metadata plus chunk_index, total_chunks, and document_id.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=_token_length,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    raw_chunks = splitter.split_text(raw_doc.content)
    doc_id = _document_id(raw_doc)
    total_chunks = len(raw_chunks)

    documents = []
    for idx, chunk_text in enumerate(raw_chunks):
        metadata = {
            "source": raw_doc.source,
            "title": raw_doc.title,
            "topic_tags": raw_doc.topic_tags,
            "difficulty_level": raw_doc.difficulty_level,
            "document_id": doc_id,
            "chunk_index": idx,
            "total_chunks": total_chunks,
            **raw_doc.metadata,
        }
        documents.append(Document(page_content=chunk_text, metadata=metadata))
    return documents


def chunk_documents(raw_docs: list[RawDocument]) -> list[Document]:
    """Chunk a batch of raw documents.

    Args:
        raw_docs: The raw documents to chunk.

    Returns:
        A flat list of chunked LangChain Document objects across all inputs.
    """
    all_chunks: list[Document] = []
    for raw_doc in raw_docs:
        all_chunks.extend(chunk_document(raw_doc))
    logger.info(
        "Chunked %d documents into %d chunks.", len(raw_docs), len(all_chunks)
    )
    return all_chunks
