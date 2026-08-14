"""FastAPI application entry point for the CFD Simulation Copilot."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from src.api.routes import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(
    title="CFD Simulation Copilot API",
    description=(
        "LLM agent that turns natural-language CFD problem descriptions into "
        "physics-validated OpenFOAM case configurations. Runs free via the "
        "Hugging Face Inference API by default; swappable to local Ollama or "
        "the Mistral API for production."
    ),
    version="0.1.0",
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
