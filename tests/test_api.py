"""Tests for the FastAPI application: routes, error handling, and schemas."""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.agent.state import initial_state
from src.api.main import app

client = TestClient(app)


def test_health_endpoint_returns_ok_shape():
    """GET /health returns the expected keys even if Qdrant/Ollama are unreachable."""
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert "documents_indexed" in body
    assert "model" in body
    assert "llm_provider" in body
    assert "ollama_reachable" in body


def test_health_endpoint_reports_degraded_when_ollama_unreachable():
    """GET /health reports status=degraded when the Ollama server does not respond."""
    with patch("src.api.routes._check_ollama_reachable", return_value=False):
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["ollama_reachable"] is False


def test_health_endpoint_checks_ollama_when_reachable():
    """GET /health reflects a reachable Ollama server in ollama_reachable."""
    with patch("src.api.routes._check_ollama_reachable", return_value=True):
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["ollama_reachable"] is True


def test_simulate_endpoint_returns_agent_output(tmp_path):
    """POST /simulate returns the agent's solver, files, and validation output."""
    import src.api.routes as routes_module

    routes_module._SESSIONS_PATH = tmp_path / "sessions.json"

    fake_state = initial_state("Turbulent pipe flow at Re=50000")
    fake_state["solver_config"] = {
        "solver_name": "simpleFoam",
        "turbulence_model": "kOmegaSST",
        "is_compressible": False,
        "is_steady": True,
        "simulation_type": "RAS",
        "justification": "test",
        "numerical_schemes_notes": "",
    }
    fake_state["generated_files"] = {"system/controlDict": "dummy content"}
    fake_state["validation_results"] = {"findings": []}
    fake_state["final_response"] = "## Test response"
    fake_state["citations"] = []

    with patch("src.api.routes.run_agent", return_value=fake_state):
        response = client.post("/simulate", json={"query": "Turbulent pipe flow at Re=50000"})

    assert response.status_code == 200
    body = response.json()
    assert body["solver"] == "simpleFoam"
    assert body["turbulence_model"] == "kOmegaSST"
    assert "system/controlDict" in body["generated_files"]
    assert body["error"] is None


def test_simulate_endpoint_surfaces_agent_error_without_crashing(tmp_path, monkeypatch):
    """POST /simulate returns a 200 with an error field if the agent fails internally."""
    import src.api.routes as routes_module

    routes_module._SESSIONS_PATH = tmp_path / "sessions.json"

    fake_state = initial_state("bad query")
    fake_state["error"] = "Something went wrong upstream."

    with patch("src.api.routes.run_agent", return_value=fake_state):
        response = client.post("/simulate", json={"query": "bad query"})

    assert response.status_code == 200
    assert response.json()["error"] == "Something went wrong upstream."


def test_simulate_endpoint_rejects_empty_query():
    """POST /simulate returns a 422 validation error for an empty query string."""
    response = client.post("/simulate", json={"query": ""})
    assert response.status_code == 422


def test_get_session_returns_404_for_unknown_id(tmp_path):
    """GET /sessions/{id} returns 404 when the session does not exist."""
    import src.api.routes as routes_module

    routes_module._SESSIONS_PATH = tmp_path / "sessions.json"
    response = client.get("/sessions/does-not-exist")
    assert response.status_code == 404


def test_feedback_returns_404_for_unknown_session(tmp_path):
    """POST /feedback returns 404 when referencing a nonexistent session."""
    import src.api.routes as routes_module

    routes_module._SESSIONS_PATH = tmp_path / "sessions.json"
    response = client.post("/feedback", json={"session_id": "nope", "rating": 5, "comment": "great"})
    assert response.status_code == 404


def test_knowledge_base_stats_handles_missing_processed_data(tmp_path, monkeypatch):
    """GET /knowledge-base/stats returns zeroed stats when no ingestion has run yet."""
    monkeypatch.setattr("src.api.routes.settings.DATA_PROCESSED_DIR", str(tmp_path / "does_not_exist"))
    response = client.get("/knowledge-base/stats")
    assert response.status_code == 200
    body = response.json()
    assert body["total_documents"] == 0
    assert body["topics"] == {}
