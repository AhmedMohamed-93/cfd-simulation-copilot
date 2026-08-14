"""Pydantic v2 request/response models for the FastAPI layer."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SimulateRequest(BaseModel):
    """Request body for POST /simulate.

    Attributes:
        query: Natural-language description of the CFD simulation problem.
        stream: Whether to stream intermediate agent steps (reserved for
            future use; the current implementation returns a single
            synchronous response regardless of this flag).
    """

    query: str = Field(..., min_length=1, description="Natural language CFD problem description.")
    stream: bool = Field(default=False, description="Reserved for future streaming support.")


class SimulateResponse(BaseModel):
    """Response body for POST /simulate.

    Attributes:
        session_id: Unique identifier for this simulation session.
        solver: Selected OpenFOAM solver name.
        turbulence_model: Selected turbulence closure model.
        generated_files: Mapping of OpenFOAM file path to file content.
        validation: Serialized physics validation results.
        explanation: Full markdown explanation of the agent's reasoning.
        citations: Source citations gathered during retrieval.
        latency_ms: End-to-end wall-clock latency of the agent run.
    """

    session_id: str
    solver: str | None
    turbulence_model: str | None
    generated_files: dict[str, str]
    validation: dict[str, Any]
    explanation: str
    citations: list[dict[str, Any]]
    latency_ms: float
    error: str | None = None


class HealthResponse(BaseModel):
    """Response body for GET /health.

    Attributes:
        status: "ok" if the API and all its dependencies are reachable,
            "degraded" if one or more are not.
        documents_indexed: Number of points currently in the Qdrant collection.
        model: The configured LLM model identifier.
        llm_provider: The active LLM provider ("ollama" or "mistral").
        ollama_reachable: Whether the configured Ollama server responded.
            Always True when LLM_PROVIDER is not "ollama" (not applicable).
    """

    status: str
    documents_indexed: int
    model: str
    llm_provider: str
    ollama_reachable: bool


class SessionRecord(BaseModel):
    """A persisted simulation session, as returned by GET /sessions/{id}.

    Attributes:
        session_id: Unique identifier for the session.
        query: The original user query.
        response: The full SimulateResponse payload for this session.
        created_at: ISO-8601 timestamp of when the session was created.
    """

    session_id: str
    query: str
    response: SimulateResponse
    created_at: str


class FeedbackRequest(BaseModel):
    """Request body for POST /feedback.

    Attributes:
        session_id: The session this feedback refers to.
        rating: Integer rating, typically 1-5.
        comment: Optional free-text comment.
    """

    session_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: str = Field(default="")


class FeedbackResponse(BaseModel):
    """Response body for POST /feedback.

    Attributes:
        status: "recorded" on success.
    """

    status: str


class KnowledgeBaseStatsResponse(BaseModel):
    """Response body for GET /knowledge-base/stats.

    Attributes:
        total_documents: Total number of indexed chunks.
        topics: Mapping of topic tag -> count of chunks carrying that tag.
        last_updated: ISO-8601 timestamp of the last known ingestion run,
            or an empty string if unknown.
    """

    total_documents: int
    topics: dict[str, int]
    last_updated: str
