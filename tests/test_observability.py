"""Tests for local structured agent trace logging (no external tracing service)."""

from __future__ import annotations

import json

from src.observability.logger import log_agent_trace


def test_log_agent_trace_creates_file_with_expected_shape(tmp_path):
    """A first trace entry creates the log file with the documented fields."""
    log_path = tmp_path / "traces.json"
    log_agent_trace(
        query="Turbulent pipe flow at Re=50000",
        steps=["[parse_flow_description] completed in 0.10s"],
        latency_ms=123.4,
        result={"solver": "simpleFoam", "turbulence_model": "kOmegaSST", "error": None},
        log_path=str(log_path),
    )

    entries = json.loads(log_path.read_text(encoding="utf-8"))
    assert len(entries) == 1
    entry = entries[0]
    assert entry["query"] == "Turbulent pipe flow at Re=50000"
    assert entry["latency_ms"] == 123.4
    assert entry["result"]["solver"] == "simpleFoam"
    assert "timestamp" in entry
    assert entry["steps"] == ["[parse_flow_description] completed in 0.10s"]


def test_log_agent_trace_appends_to_existing_entries(tmp_path):
    """Multiple runs accumulate as a growing JSON array rather than overwriting."""
    log_path = tmp_path / "traces.json"
    for i in range(3):
        log_agent_trace(
            query=f"query {i}",
            steps=[],
            latency_ms=float(i),
            result={},
            log_path=str(log_path),
        )

    entries = json.loads(log_path.read_text(encoding="utf-8"))
    assert len(entries) == 3
    assert [e["query"] for e in entries] == ["query 0", "query 1", "query 2"]


def test_log_agent_trace_recovers_from_corrupted_existing_file(tmp_path):
    """A corrupted/unparseable existing log file does not crash logging; it restarts fresh."""
    log_path = tmp_path / "traces.json"
    log_path.write_text("not valid json", encoding="utf-8")

    log_agent_trace(query="q", steps=[], latency_ms=1.0, result={}, log_path=str(log_path))

    entries = json.loads(log_path.read_text(encoding="utf-8"))
    assert len(entries) == 1
