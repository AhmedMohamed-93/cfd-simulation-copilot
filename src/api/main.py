"""FastAPI application entry point for the CFD Simulation Copilot."""

from __future__ import annotations

import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from src.api.routes import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _warm_ollama_model() -> None:
    """Fire a trivial Ollama call so the model is loaded into memory eagerly.

    A cold model load is a large, avoidable chunk of the first real
    request's latency (observed as tens of seconds of run-to-run variance
    on parse_flow_description); this loads the model ahead of time instead
    of on whichever request happens to arrive first. Runs in a background
    thread so it never delays server startup/health readiness; a real
    request that races the warm-up just pays the cold-load cost itself,
    same as before this existed.
    """
    try:
        import ollama  # noqa: PLC0415

        ollama.Client(host=settings.OLLAMA_BASE_URL).chat(
            model=settings.OLLAMA_MODEL,
            messages=[{"role": "user", "content": "Respond with OK."}],
            options={"num_predict": 5},
        )
        logger.info("Warmed Ollama model '%s' into memory.", settings.OLLAMA_MODEL)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Ollama warm-up failed (model will load lazily on first request): %s", exc)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Kick off Ollama model warm-up at startup; no-op for other providers.

    Args:
        _app: The FastAPI application (unused; required by the lifespan protocol).
    """
    if settings.LLM_PROVIDER == "ollama":
        threading.Thread(target=_warm_ollama_model, daemon=True).start()
    yield


app = FastAPI(
    title="CFD Simulation Copilot API",
    description=(
        "LLM agent that turns natural-language CFD problem descriptions into "
        "physics-validated OpenFOAM case configurations. Runs free via the "
        "Hugging Face Inference API by default; swappable to local Ollama or "
        "the Mistral API for production."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


def main() -> None:
    """Run the FastAPI app with uvicorn using the configured host/port."""
    import uvicorn

    uvicorn.run("src.api.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=False)


if __name__ == "__main__":
    main()
