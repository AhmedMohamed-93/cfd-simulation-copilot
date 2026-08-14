"""FastAPI routes for the CFD Simulation Copilot API."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from config import settings
from src.agent.graph import run_agent
from src.api.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    KnowledgeBaseStatsResponse,
    SessionRecord,
    SimulateRequest,
    SimulateResponse,
)
from src.retrieval.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)
router = APIRouter()

_SESSIONS_PATH = Path(settings.SESSIONS_DB_PATH)


def _load_sessions() -> dict[str, dict[str, Any]]:
    """Load all persisted simulation sessions from disk.

    Returns:
        A mapping of session_id -> session record dict, empty if no
        sessions file exists yet.
    """
    if not _SESSIONS_PATH.exists():
        return {}
    try:
        return json.loads(_SESSIONS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load sessions store: %s", exc)
        return {}


def _save_sessions(sessions: dict[str, dict[str, Any]]) -> None:
    """Persist all simulation sessions to disk.

    Args:
        sessions: The full session store to persist.
    """
    _SESSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SESSIONS_PATH.write_text(json.dumps(sessions, indent=2), encoding="utf-8")


@router.post("/simulate", response_model=SimulateResponse)
def simulate(request: SimulateRequest) -> SimulateResponse:
    """Run the CFD copilot agent on a natural-language problem description.

    Args:
        request: The simulation request containing the user's query.

    Returns:
        The full agent output: solver/turbulence selection, generated
        OpenFOAM files, physics validation, explanation, and citations.
    """
    start = time.perf_counter()
    session_id = str(uuid.uuid4())

    final_state = run_agent(request.query)
    latency_ms = (time.perf_counter() - start) * 1000

    solver_config = final_state.get("solver_config") or {}
    response = SimulateResponse(
        session_id=session_id,
        solver=solver_config.get("solver_name"),
        turbulence_model=solver_config.get("turbulence_model"),
        generated_files=final_state.get("generated_files", {}),
        validation=final_state.get("validation_results", {}),
        explanation=final_state.get("final_response", ""),
        citations=final_state.get("citations", []),
        latency_ms=latency_ms,
        error=final_state.get("error"),
    )

    sessions = _load_sessions()
    sessions[session_id] = {
        "session_id": session_id,
        "query": request.query,
        "response": response.model_dump(mode="json"),
        "created_at": datetime.now(UTC).isoformat(),
    }
    _save_sessions(sessions)

    return response


def _check_ollama_reachable() -> bool:
    """Check whether the configured Ollama server responds to a list-models call.

    Returns:
        True if LLM_PROVIDER is not "ollama" (check not applicable), or if
        the Ollama server responded successfully. False if it did not.
    """
    if settings.LLM_PROVIDER != "ollama":
        return True
    try:
        import ollama  # noqa: PLC0415

        ollama.Client(host=settings.OLLAMA_BASE_URL).list()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Health check could not reach Ollama: %s", exc)
        return False


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report API health, knowledge base status, and Ollama connectivity.

    Returns:
        A HealthResponse with the current document count, configured model,
        and whether the configured LLM backend (Ollama, by default) is reachable.
    """
    try:
        info = QdrantVectorStore().get_collection_info()
        documents_indexed = info.get("points_count", 0)
        qdrant_ok = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Health check could not reach Qdrant: %s", exc)
        documents_indexed = 0
        qdrant_ok = False

    ollama_reachable = _check_ollama_reachable()
    status = "ok" if qdrant_ok and ollama_reachable else "degraded"

    return HealthResponse(
        status=status,
        documents_indexed=documents_indexed,
        model=settings.LLM_MODEL,
        llm_provider=settings.LLM_PROVIDER,
        ollama_reachable=ollama_reachable,
    )


@router.get("/sessions/{session_id}", response_model=SessionRecord)
def get_session(session_id: str) -> SessionRecord:
    """Retrieve a previously run simulation session.

    Args:
        session_id: The unique session identifier returned by /simulate.

    Returns:
        The persisted SessionRecord.

    Raises:
        HTTPException: 404 if no session with the given id exists.
    """
    sessions = _load_sessions()
    record = sessions.get(session_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return SessionRecord.model_validate(record)


@router.get("/sessions", response_model=list[SessionRecord])
def list_sessions() -> list[SessionRecord]:
    """List all persisted simulation sessions, most recent first.

    Returns:
        A list of SessionRecord objects.
    """
    sessions = _load_sessions()
    records = [SessionRecord.model_validate(r) for r in sessions.values()]
    return sorted(records, key=lambda r: r.created_at, reverse=True)


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Record user feedback on a past simulation session.

    Args:
        request: The feedback payload (session id, rating, comment).

    Returns:
        A FeedbackResponse confirming the feedback was recorded.

    Raises:
        HTTPException: 404 if the referenced session does not exist.
    """
    sessions = _load_sessions()
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session '{request.session_id}' not found.")

    sessions[request.session_id].setdefault("feedback", []).append(
        {
            "rating": request.rating,
            "comment": request.comment,
            "submitted_at": datetime.now(UTC).isoformat(),
        }
    )
    _save_sessions(sessions)
    return FeedbackResponse(status="recorded")


@router.get("/knowledge-base/stats", response_model=KnowledgeBaseStatsResponse)
def knowledge_base_stats() -> KnowledgeBaseStatsResponse:
    """Report knowledge base indexing statistics.

    Returns:
        A KnowledgeBaseStatsResponse with total document count, a topic
        breakdown, and the last known ingestion timestamp.
    """
    processed_path = Path(settings.DATA_PROCESSED_DIR) / "chunks.json"
    if not processed_path.exists():
        return KnowledgeBaseStatsResponse(total_documents=0, topics={}, last_updated="")

    chunks = json.loads(processed_path.read_text(encoding="utf-8"))
    topic_counter: Counter[str] = Counter()
    for chunk in chunks:
        for tag in chunk.get("metadata", {}).get("topic_tags", []):
            topic_counter[tag] += 1

    last_updated = datetime.fromtimestamp(
        processed_path.stat().st_mtime, tz=UTC
    ).isoformat()

    return KnowledgeBaseStatsResponse(
        total_documents=len(chunks), topics=dict(topic_counter), last_updated=last_updated
    )
