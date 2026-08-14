"""Shared pytest fixtures for the CFD Simulation Copilot test suite."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("MISTRAL_API_KEY", "test-key-not-real")

# Route the embedded (serverless) Qdrant storage and the agent trace log to
# the OS temp dir for the whole test session, so running the suite never
# creates or touches real qdrant_storage/logs directories in the repo.
_TEST_TMP = Path(tempfile.gettempdir()) / "cfd_copilot_pytest"
os.environ.setdefault("QDRANT_LOCAL_PATH", str(_TEST_TMP / "qdrant_storage"))
os.environ.setdefault("AGENT_TRACES_LOG_PATH", str(_TEST_TMP / "agent_traces.json"))

from src.generation.schemas import (  # noqa: E402
    FlowDescription,
    SimulationType,
    SolverConfiguration,
    SolverName,
    TurbulenceModel,
)


@pytest.fixture
def laminar_pipe_flow() -> FlowDescription:
    """A simple laminar internal pipe flow, well below the transition Re."""
    return FlowDescription(
        reynolds_number=500,
        is_compressible=False,
        is_steady=True,
        geometry="pipe",
        fluid="water",
        characteristic_length=0.02,
        inlet_velocity=0.025,
        desired_outputs=["pressure drop"],
    )


@pytest.fixture
def turbulent_pipe_flow() -> FlowDescription:
    """A fully turbulent internal pipe flow of air."""
    return FlowDescription(
        reynolds_number=50000,
        is_compressible=False,
        is_steady=True,
        geometry="pipe",
        fluid="air",
        characteristic_length=0.1,
        inlet_velocity=7.5,
        desired_outputs=["pressure drop"],
    )


@pytest.fixture
def solver_config_simple_komega() -> SolverConfiguration:
    """A simpleFoam + kOmegaSST solver configuration."""
    return SolverConfiguration(
        solver_name=SolverName.SIMPLE_FOAM,
        turbulence_model=TurbulenceModel.K_OMEGA_SST,
        is_compressible=False,
        is_steady=True,
        simulation_type=SimulationType.RAS,
        justification="Re >= 5e4, kOmegaSST recommended for robustness.",
    )


@pytest.fixture
def solver_config_laminar() -> SolverConfiguration:
    """A simpleFoam + laminar solver configuration."""
    return SolverConfiguration(
        solver_name=SolverName.SIMPLE_FOAM,
        turbulence_model=TurbulenceModel.LAMINAR,
        is_compressible=False,
        is_steady=True,
        simulation_type=SimulationType.LAMINAR,
        justification="Re < 2300, laminar flow.",
    )
