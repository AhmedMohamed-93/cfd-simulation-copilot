"""Pydantic v2 request/response models for the FastAPI layer."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SimulateRequest(BaseModel):
    """Request body for POST /simulate.

    Attributes:
        query: Natural-language description of the CFD simulation problem.
    """

    query: str = Field(..., min_length=1, description="Natural language CFD problem description.")


class SimulateStartedResponse(BaseModel):
    """Response body for POST /simulate.

    The agent run happens in a background task (a local Ollama run can take
    a couple of minutes on CPU); this response returns immediately.
    Poll GET /sessions/{session_id}/status for live progress, then fetch
    GET /sessions/{session_id} for the full SimulateResponse once its
    status is "complete".

    Attributes:
        session_id: Unique identifier for this simulation session.
        status: Always "running" at this point.
    """

    session_id: str
    status: str


class SimulationStatusResponse(BaseModel):
    """Response body for GET /sessions/{session_id}/status.

    Attributes:
        session_id: The session identifier.
        status: "running", "complete", or "error".
        current_node: The agent graph node currently executing (or the last
            one that completed), or None if the run hasn't reported its
            first completed node yet.
        completed_steps: Human-readable trace entries for each node
            completed so far, in order (same format as
            AgentState.reasoning_steps).
        error: Error message if status is "error", else None.
    """

    session_id: str
    status: str
    current_node: str | None
    completed_steps: list[str]
    error: str | None = None


class SimulateResponse(BaseModel):
    """Full simulation result, returned by GET /sessions/{session_id} once complete.

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
