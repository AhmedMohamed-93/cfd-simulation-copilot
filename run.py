"""Single-command launcher for the CFD Simulation Copilot, no Docker needed.

Starts the FastAPI backend (uvicorn, in a background thread of this same
process) and the Streamlit frontend (as a subprocess), waits for the API to
report healthy, then opens the app in your default browser. Ctrl+C stops
both cleanly.

Usage:
    python run.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import webbrowser

import requests
import uvicorn

from config import settings

_API_READY_TIMEOUT_S = 60


class _ApiServerThread(threading.Thread):
    """Runs the FastAPI app via uvicorn in a background thread.

    Attributes:
        server: The underlying uvicorn.Server instance, once started.
    """

    def __init__(self) -> None:
        """Initialize the thread as a daemon so it never blocks process exit."""
        super().__init__(daemon=True)
        self.server: uvicorn.Server | None = None

    def run(self) -> None:
        """Build and run the uvicorn server (blocks until shutdown)."""
        config = uvicorn.Config(
            "src.api.main:app",
            host=settings.API_HOST,
            port=settings.API_PORT,
            log_level="info",
        )
        self.server = uvicorn.Server(config)
        self.server.run()

    def stop(self) -> None:
        """Signal the uvicorn server to shut down gracefully."""
        if self.server is not None:
            self.server.should_exit = True


def _wait_for_api(timeout_s: int = _API_READY_TIMEOUT_S) -> bool:
    """Poll the API's /health endpoint until it responds or times out.

    Args:
        timeout_s: Maximum seconds to wait.

    Returns:
        True if the API responded, False if the timeout was reached.
    """
    url = f"http://localhost:{settings.API_PORT}/health"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(1)
    return False


def _start_streamlit() -> subprocess.Popen:
    """Launch the Streamlit frontend as a subprocess.

    Returns:
        The subprocess.Popen handle for the running Streamlit process.
    """
    env = os.environ.copy()
    env["API_BASE_URL"] = f"http://localhost:{settings.API_PORT}"
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app/streamlit_app.py",
            "--server.address=0.0.0.0",
            f"--server.port={settings.STREAMLIT_PORT}",
            "--server.headless=true",
        ],
        env=env,
    )


def main() -> None:
    """Start the API and frontend, wait for readiness, and block until Ctrl+C."""
    print(f"Starting FastAPI backend on http://localhost:{settings.API_PORT} ...")
    api_thread = _ApiServerThread()
    api_thread.start()

    if _wait_for_api():
        print("Backend is healthy.")
    else:
        print(
            f"WARNING: backend did not report healthy within {_API_READY_TIMEOUT_S}s; "
            "starting the frontend anyway."
        )

    print(f"Starting Streamlit frontend on http://localhost:{settings.STREAMLIT_PORT} ...")
    streamlit_process = _start_streamlit()

    time.sleep(3)
    try:
        webbrowser.open(f"http://localhost:{settings.STREAMLIT_PORT}")
    except Exception:  # noqa: BLE001 - opening a browser is best-effort only
        pass

    print("\nCFD Simulation Copilot is running:")
    print(f"  Streamlit UI:  http://localhost:{settings.STREAMLIT_PORT}")
    print(f"  FastAPI docs:  http://localhost:{settings.API_PORT}/docs")
    print("\nPress Ctrl+C to stop.\n")

    try:
        while True:
            if streamlit_process.poll() is not None:
                print("Streamlit process exited unexpectedly; shutting down.")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if streamlit_process.poll() is None:
            streamlit_process.terminate()
            try:
                streamlit_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                streamlit_process.kill()
        api_thread.stop()
        print("Stopped.")


if __name__ == "__main__":
    main()
