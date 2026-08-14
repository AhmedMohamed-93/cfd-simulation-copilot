"""Centralized configuration for the CFD Simulation Copilot.

All hyperparameters, model names, and environment-derived secrets live here.
No other module should hardcode a value that belongs in this file.

The system runs 100% free and container-free by default: LLM reasoning via
the Hugging Face Inference API (free with an HF account, no local RAM/GPU
needed), embeddings + reranking via local sentence-transformers models,
vector storage via embedded (serverless) Qdrant, and agent execution traces
via structured local JSON logs. Docker is never required. The LLM layer is
provider-pluggable — see LLM_PROVIDER — so the same codebase can also run
against a local Ollama server (LLM_PROVIDER=ollama, free but RAM-hungry) or
the Mistral API for a production deployment (LLM_PROVIDER=mistral).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings, loaded from environment variables / .env.

    Attributes:
        LLM_PROVIDER: Which LLM backend to use for agent reasoning:
            "huggingface" (default, free with an HF account, no local
            RAM/GPU needed), "ollama" (free, local, but requires enough RAM
            to load the model), or "mistral" (production, requires
            MISTRAL_API_KEY).
        HF_API_TOKEN: API token for the Hugging Face Inference API, required
            when LLM_PROVIDER=huggingface. Get a free one at
            https://huggingface.co/settings/tokens.
        HF_MODEL: Hugging Face model repo id used for agent reasoning, used
            only when LLM_PROVIDER=huggingface.
        OLLAMA_BASE_URL: Base URL of the Ollama server, used only when
            LLM_PROVIDER=ollama.
        OLLAMA_MODEL: Ollama model tag used for agent reasoning (must be
            pulled locally first, e.g. `ollama pull llama3.1:8b`), used only
            when LLM_PROVIDER=ollama.
        MISTRAL_API_KEY: Optional API key for the Mistral platform, only
            required when LLM_PROVIDER=mistral.
        MISTRAL_MODEL: Mistral chat model identifier, used only when
            LLM_PROVIDER=mistral.
        QDRANT_MODE: Vector store deployment mode. Only "local" (embedded,
            serverless, on-disk) is currently implemented; the field exists
            so a future remote-server mode can be added without an API change.
        QDRANT_LOCAL_PATH: Filesystem directory where the embedded Qdrant
            instance persists its data. Created automatically if missing.
        QDRANT_COLLECTION_NAME: Name of the Qdrant collection holding the KB.
        EMBEDDING_MODEL: Local sentence-transformers embedding model name.
        EMBEDDING_DIM: Dimensionality of the embedding vectors (384 for
            all-MiniLM-L6-v2).
        CROSS_ENCODER_MODEL: Local sentence-transformers cross-encoder model
            used for reranking retrieved chunks.
        LLM_TEMPERATURE: Sampling temperature for the agent LLM.
        CHUNK_SIZE: Target chunk size (in tokens) for document chunking.
        CHUNK_OVERLAP: Overlap (in tokens) between consecutive chunks.
        TOP_K_RETRIEVAL: Number of chunks pulled from the vector store (stage 1).
        TOP_K_RERANKED: Number of chunks kept after reranking (stage 2).
        MAX_AGENT_ITERATIONS: Hard cap on total agent graph steps.
        MAX_RETRIEVAL_ATTEMPTS: Max retrieval retries when quality is low.
        MAX_VALIDATION_RETRIES: Max regeneration retries after validation failure.
        RETRIEVAL_QUALITY_THRESHOLD: Minimum self-graded retrieval quality.
        EMBEDDING_CACHE_PATH: Local JSON file used to cache embeddings.
        AGENT_TRACES_LOG_PATH: Local JSON file that structured agent
            execution traces are appended to (timestamp, query, steps,
            latency_ms, result) — replaces an external tracing service.
        API_HOST: Host the FastAPI server binds to.
        API_PORT: Port the FastAPI server binds to.
        STREAMLIT_PORT: Port the Streamlit frontend binds to (used by run.py).
        DATA_RAW_DIR: Directory holding raw downloaded documents.
        DATA_PROCESSED_DIR: Directory holding chunked/processed documents.
        SESSIONS_DB_PATH: SQLite file used to persist simulation sessions.
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- LLM provider (free via Hugging Face by default; no local RAM needed) ---
    LLM_PROVIDER: str = Field(default="huggingface")
    HF_API_TOKEN: str = Field(default="")
    # v0.2, not v0.3: as of this writing, v0.3's only listed HF Inference
    # Providers mapping (novita) reports status="error" (verified via
    # huggingface_hub.model_info(..., expand=["inferenceProviderMapping"])),
    # so calls to it 404 regardless of token permissions. v0.2 has a live
    # mapping (featherless-ai) and was confirmed working end-to-end.
    HF_MODEL: str = Field(default="mistralai/Mistral-7B-Instruct-v0.2")

    # --- Ollama (optional alternative: free, local, but RAM-hungry) ---
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434")
    OLLAMA_MODEL: str = Field(default="llama3.1:8b")

    # --- Mistral API (optional, production-only alternative provider) ---
    MISTRAL_API_KEY: str = Field(default="")
    MISTRAL_MODEL: str = Field(default="mistral-large-latest")

    # --- Qdrant vector database (embedded, serverless — no Docker/server) ---
    QDRANT_MODE: str = Field(default="local")
    QDRANT_LOCAL_PATH: str = Field(default="./qdrant_storage")
    QDRANT_COLLECTION_NAME: str = Field(default="cfd_knowledge_base")

    # --- Local embedding + reranking models ---
    EMBEDDING_MODEL: str = Field(default="all-MiniLM-L6-v2")
    EMBEDDING_DIM: int = Field(default=384)
    CROSS_ENCODER_MODEL: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")

    LLM_TEMPERATURE: float = Field(default=0.1)

    # --- Chunking ---
    CHUNK_SIZE: int = Field(default=512)
    CHUNK_OVERLAP: int = Field(default=64)

    # --- Retrieval ---
    TOP_K_RETRIEVAL: int = Field(default=10)
    TOP_K_RERANKED: int = Field(default=3)

    # --- Agent control flow ---
    MAX_AGENT_ITERATIONS: int = Field(default=10)
    MAX_RETRIEVAL_ATTEMPTS: int = Field(default=2)
    MAX_VALIDATION_RETRIES: int = Field(default=1)
    RETRIEVAL_QUALITY_THRESHOLD: float = Field(default=0.5)

    # --- Embedding cache ---
    EMBEDDING_CACHE_PATH: str = Field(default="data/processed/embedding_cache.json")

    # --- Local structured trace logging (replaces Phoenix/LangSmith) ---
    AGENT_TRACES_LOG_PATH: str = Field(default="logs/agent_traces.json")

    # --- API ---
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    STREAMLIT_PORT: int = Field(default=8501)

    # --- Data locations ---
    DATA_RAW_DIR: str = Field(default="data/raw")
    DATA_PROCESSED_DIR: str = Field(default="data/processed")
    SESSIONS_DB_PATH: str = Field(default="data/processed/sessions.json")

    @property
    def LLM_MODEL(self) -> str:  # noqa: N802 - kept upper-case for settings convention
        """Return the active chat model name for the configured provider.

        Returns:
            HF_MODEL, OLLAMA_MODEL, or MISTRAL_MODEL, matching LLM_PROVIDER.
        """
        if self.LLM_PROVIDER == "ollama":
            return self.OLLAMA_MODEL
        if self.LLM_PROVIDER == "mistral":
            return self.MISTRAL_MODEL
        return self.HF_MODEL


@lru_cache
def get_settings() -> Settings:
    """Return a cached, process-wide Settings instance.

    Returns:
        The singleton Settings object built from environment variables.
    """
    return Settings()


settings = get_settings()
