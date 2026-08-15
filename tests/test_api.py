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


def _fake_success_state(query: str) -> dict:
    """Build a fake successful AgentState for mocking run_agent in tests.

    Args:
        query: The user query the fake state should be built for.

    Returns:
        A dict matching AgentState with a complete, successful result.
    """
    fake_state = initial_state(query)
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
    return fake_state


def test_simulate_endpoint_returns_session_id_immediately(tmp_path):
    """POST /simulate returns session_id + status='running' without blocking on the agent.

    The agent run happens in a FastAPI background task specifically because
    a local Ollama run can take a couple of minutes on CPU, and the HTTP
    response itself must not wait for it.
    """
    import src.api.routes as routes_module

    routes_module._SESSIONS_PATH = tmp_path / "sessions.json"

    with patch("src.api.routes.run_agent", return_value=_fake_success_state("Turbulent pipe flow at Re=50000")):
        response = client.post("/simulate", json={"query": "Turbulent pipe flow at Re=50000"})

    assert response.status_code == 200
    body = response.json()
    assert "session_id" in body
    assert body["status"] == "running"


def test_simulate_background_run_completes_and_full_result_is_fetchable(tmp_path):
    """After POST /simulate, status becomes 'complete' and the full result is fetchable.

    TestClient runs FastAPI BackgroundTasks to completion before .post()
    returns, so the mocked agent run has already finished by the time this
    test inspects /status and /sessions/{id}.
    """
    import src.api.routes as routes_module

    routes_module._SESSIONS_PATH = tmp_path / "sessions.json"

    with patch("src.api.routes.run_agent", return_value=_fake_success_state("Turbulent pipe flow at Re=50000")):
        started = client.post("/simulate", json={"query": "Turbulent pipe flow at Re=50000"}).json()
        session_id = started["session_id"]

        status_response = client.get(f"/sessions/{session_id}/status")
        assert status_response.status_code == 200
        status_body = status_response.json()
        assert status_body["status"] == "complete"
        assert status_body["error"] is None

        result_response = client.get(f"/sessions/{session_id}")
        assert result_response.status_code == 200
        result_body = result_response.json()["response"]
        assert result_body["solver"] == "simpleFoam"
        assert result_body["turbulence_model"] == "kOmegaSST"
        assert "system/controlDict" in result_body["generated_files"]
        assert result_body["error"] is None


def test_simulate_status_reports_error_without_crashing(tmp_path):
    """When the agent fails internally, the session's /status becomes 'error'."""
    import src.api.routes as routes_module

    routes_module._SESSIONS_PATH = tmp_path / "sessions.json"

    fake_state = initial_state("bad query")
    fake_state["error"] = "Something went wrong upstream."

    with patch("src.api.routes.run_agent", return_value=fake_state):
        started = client.post("/simulate", json={"query": "bad query"}).json()
        session_id = started["session_id"]

        status_response = client.get(f"/sessions/{session_id}/status")

    assert status_response.status_code == 200
    status_body = status_response.json()
    assert status_body["status"] == "error"
    assert status_body["error"] == "Something went wrong upstream."


def test_simulate_status_reports_error_when_background_task_crashes(tmp_path):
    """An unexpected exception in the background task itself still marks the session 'error'.

    Regression coverage for a real hang risk: if anything other than
    run_agent's own caught graph errors raises (building the response,
    disk I/O), the session must not get stuck reporting 'running' forever.
    """
    import src.api.routes as routes_module

    routes_module._SESSIONS_PATH = tmp_path / "sessions.json"

    with patch("src.api.routes.run_agent", side_effect=RuntimeError("boom")):
        started = client.post("/simulate", json={"query": "Turbulent pipe flow at Re=50000"}).json()
        session_id = started["session_id"]

        status_response = client.get(f"/sessions/{session_id}/status")

    assert status_response.status_code == 200
    status_body = status_response.json()
    assert status_body["status"] == "error"
    assert "boom" in status_body["error"]


def test_simulate_endpoint_rejects_empty_query():
    """POST /simulate returns a 422 validation error for an empty query string."""
    response = client.post("/simulate", json={"query": ""})
    assert response.status_code == 422


def test_session_status_returns_404_for_unknown_id():
    """GET /sessions/{id}/status returns 404 for a session that was never started."""
    response = client.get("/sessions/does-not-exist/status")
    assert response.status_code == 404


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
