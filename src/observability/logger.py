"""Simple structured JSON logging of agent execution traces.

Replaces an external tracing service (Phoenix/LangSmith) with a local,
dependency-free JSON log: every agent run appends one entry to
`AGENT_TRACES_LOG_PATH` containing a timestamp, the original query, the
step-by-step reasoning trace, end-to-end latency, and a summary of the
result. No account, server, or network access is required.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config import settings

logger = logging.getLogger(__name__)

_write_lock = threading.Lock()


def log_agent_trace(
    query: str,
    steps: list[str],
    latency_ms: float,
    result: dict[str, Any],
    log_path: str = settings.AGENT_TRACES_LOG_PATH,
) -> None:
    """Append one structured trace entry to the local agent traces log.

    Best-effort: a logging failure (e.g. a locked or unwritable file) is
    caught and logged as a warning rather than propagated, so tracing can
    never break an actual agent run.

    Args:
        query: The user's original natural-language query.
        steps: The agent's reasoning/step trace, one string per graph node.
        latency_ms: End-to-end wall-clock latency of the agent run.
        result: A summary of the run's outcome (e.g. solver, turbulence
            model, validation pass/fail, error if any).
        log_path: Path to the JSON trace log file.
    """
    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "query": query,
        "steps": steps,
        "latency_ms": latency_ms,
        "result": result,
    }
    path = Path(log_path)
    try:
        with _write_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            traces: list[dict[str, Any]] = []
            if path.exists():
                try:
                    traces = json.loads(path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning("Failed to read existing trace log, starting fresh: %s", exc)
                    traces = []
            traces.append(entry)
            path.write_text(json.dumps(traces, indent=2), encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to write agent trace log: %s", exc)
