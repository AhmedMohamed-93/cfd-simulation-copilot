"""Tests for the LangGraph agent: state, tools fallback logic, and routing."""

from __future__ import annotations

import json

import pytest

from src.agent.graph import _route_after_retrieval, _route_after_validation
from src.agent.state import initial_state
from src.agent.tools import (
    _extract_json,
    _field_hint,
    _response_instruction,
    _rule_based_solver_selection,
    select_solver_and_models,
)
from src.generation.schemas import FlowDescription, SolverName, TurbulenceModel, ValidationResult


class _FakeLLMClient:
    """Minimal stand-in for LLMClient that returns a fixed JSON response."""

    def __init__(self, response_json: dict) -> None:
        self._response_json = response_json

    def complete(self, messages, temperature=0.0, max_tokens=None, json_mode=False):
        return json.dumps(self._response_json)


def test_initial_state_has_expected_defaults():
    """initial_state produces a well-formed, empty AgentState."""
    state = initial_state("Turbulent pipe flow at Re=50000")
    assert state["user_query"] == "Turbulent pipe flow at Re=50000"
    assert state["iteration_count"] == 0
    assert state["retrieval_attempts"] == 0
    assert state["error"] is None
    assert state["generated_files"] == {}


def test_extract_json_handles_markdown_code_fences():
    """_extract_json strips ```json fences before parsing."""
    text = '```json\n{"a": 1, "b": 2}\n```'
    parsed = _extract_json(text)
    assert parsed == {"a": 1, "b": 2}


def test_extract_json_handles_plain_json():
    """_extract_json parses a bare JSON object with no fences."""
    parsed = _extract_json('{"solver": "simpleFoam"}')
    assert parsed == {"solver": "simpleFoam"}


def test_extract_json_rejects_echoed_schema():
    """_extract_json raises when the model echoes the JSON Schema instead of data.

    Regression test: smaller/weaker LLMs sometimes misread "respond matching
    this schema: <schema>" as an instruction to return the schema itself.
    """
    schema_echo = FlowDescription.model_json_schema()
    with pytest.raises(ValueError, match="echoed a JSON Schema"):
        _extract_json(json.dumps(schema_echo))


def test_field_hint_handles_plain_type():
    """_field_hint reports a plain JSON Schema type directly."""
    assert _field_hint({"type": "string"}) == "string"


def test_field_hint_handles_optional_anyof():
    """_field_hint reports optional (anyOf [type, null]) fields readably."""
    assert _field_hint({"anyOf": [{"type": "number"}, {"type": "null"}]}) == "number or null"


def test_field_hint_handles_enum():
    """_field_hint lists enum members instead of the raw JSON Schema shape."""
    hint = _field_hint({"enum": ["simpleFoam", "pimpleFoam"]})
    assert hint == "one of simpleFoam, pimpleFoam"


def test_response_instruction_lists_fields_without_full_schema_dump():
    """_response_instruction lists field:hint pairs, not the raw nested schema."""
    schema = FlowDescription.model_json_schema()
    instruction = _response_instruction(schema)
    assert '"reynolds_number"' in instruction
    assert "do not return this field list" in instruction.lower()
    # The raw JSON Schema's own descriptive keys must not leak into the prompt.
    assert '"properties"' not in instruction
    assert '"title"' not in instruction


def test_rule_based_solver_selection_laminar_below_transition_re(laminar_pipe_flow):
    """The rule-based fallback selects laminar for Re < 2300."""
    config = _rule_based_solver_selection(laminar_pipe_flow)
    assert config.turbulence_model == TurbulenceModel.LAMINAR
    assert config.solver_name == SolverName.SIMPLE_FOAM


def test_rule_based_solver_selection_turbulent_above_transition_re(turbulent_pipe_flow):
    """The rule-based fallback selects kOmegaSST for Re >= 5e4."""
    config = _rule_based_solver_selection(turbulent_pipe_flow)
    assert config.turbulence_model == TurbulenceModel.K_OMEGA_SST


def test_rule_based_solver_selection_multiphase_selects_interfoam(laminar_pipe_flow):
    """Multiphase flows must route to interFoam regardless of Reynolds number."""
    laminar_pipe_flow.multiphase = True
    config = _rule_based_solver_selection(laminar_pipe_flow)
    assert config.solver_name == SolverName.INTER_FOAM


def test_rule_based_solver_selection_buoyant_selects_buoyant_solver(laminar_pipe_flow):
    """Temperature-dependent (buoyancy-driven) flows must route to a buoyant solver."""
    laminar_pipe_flow.temperature_dependent = True
    laminar_pipe_flow.is_steady = True
    config = _rule_based_solver_selection(laminar_pipe_flow)
    assert config.solver_name == SolverName.BUOYANT_SIMPLE_FOAM


def test_rule_based_solver_selection_external_aero_never_selects_buoyant_solver():
    """Regression test: external aerodynamics must never fall through to a buoyant solver.

    Even with a mis-set temperature_dependent=True flag, an airfoil/wing/NACA
    geometry must route to simpleFoam/rhoSimpleFoam, not buoyantSimpleFoam —
    external aerodynamics is checked before temperature_dependent.
    """
    flow = FlowDescription(
        reynolds_number=500000,
        geometry="External aerodynamics of a NACA0012 airfoil at low speed",
        fluid="air",
        temperature_dependent=True,
        is_compressible=False,
        is_steady=True,
    )
    config = _rule_based_solver_selection(flow)
    assert config.solver_name == SolverName.SIMPLE_FOAM


def test_rule_based_solver_selection_external_aero_high_speed_selects_rho_solver():
    """High-speed (compressible) external aerodynamics routes to rhoSimpleFoam."""
    flow = FlowDescription(
        reynolds_number=3000000,
        mach_number=0.8,
        geometry="External aerodynamics of a transonic wing",
        fluid="air",
        is_compressible=True,
        is_steady=True,
    )
    config = _rule_based_solver_selection(flow)
    assert config.solver_name == SolverName.RHO_SIMPLE_FOAM


def test_rule_based_solver_selection_cavity_selects_icofoam():
    """A canonical lid-driven cavity case (laminar, unsteady) routes to icoFoam.

    Regression test: evaluation on this exact scenario (tc04) showed the
    fallback picking pimpleFoam instead, because it had no icoFoam branch at
    all — icoFoam is OpenFOAM's own bundled tutorial solver for this case.
    """
    flow = FlowDescription(
        reynolds_number=1000,
        geometry="Lid-driven cavity flow",
        fluid="water",
        is_compressible=False,
        is_steady=False,
    )
    config = _rule_based_solver_selection(flow)
    assert config.solver_name == SolverName.ICO_FOAM
    assert config.turbulence_model == TurbulenceModel.LAMINAR


def test_rule_based_solver_selection_cylinder_wake_does_not_select_icofoam():
    """A non-cavity laminar unsteady case (cylinder wake) stays on pimpleFoam.

    icoFoam is deliberately scoped to the canonical cavity-type validation
    cases, not any low-Re unsteady flow — a cylinder vortex-shedding case at
    the same Reynolds number should not be swept into the cavity branch.
    """
    flow = FlowDescription(
        reynolds_number=1000,
        geometry="flow past a circular cylinder",
        fluid="air",
        is_compressible=False,
        is_steady=False,
    )
    config = _rule_based_solver_selection(flow)
    assert config.solver_name == SolverName.PIMPLE_FOAM


def test_rule_based_solver_selection_external_aero_turbulent_prefers_spalart_allmaras():
    """External aerodynamics above the laminar threshold prefers SpalartAllmaras.

    Regression test: evaluation on the NACA0012 case (tc05) showed the
    fallback always choosing kOmegaSST, never SpalartAllmaras — the
    standard, economical choice for attached external aero flow.
    """
    flow = FlowDescription(
        reynolds_number=3000000,
        geometry="External aerodynamics of a NACA0012 airfoil at low speed",
        fluid="air",
        is_compressible=False,
        is_steady=True,
    )
    config = _rule_based_solver_selection(flow)
    assert config.turbulence_model == TurbulenceModel.SPALART_ALLMARAS


def test_select_solver_and_models_overrides_solver_only_preserves_turbulence_choice():
    """An LLM's valid turbulence pick survives a solver-only override.

    Regression test for internal (non-aero) flow: evaluation on turbulent
    pipe flow (tc02) showed the LLM picking rhoSimpleFoam for a stated
    incompressible flow while its kOmegaSST turbulence choice was correct;
    the override must fix the solver without discarding the correct half.
    """
    flow = FlowDescription(
        reynolds_number=50000,
        geometry="circular pipe",
        fluid="air",
        is_compressible=False,
        is_steady=True,
    )
    fake_client = _FakeLLMClient(
        {
            "solver_name": "rhoSimpleFoam",
            "turbulence_model": "kOmegaSST",
            "is_compressible": False,
            "is_steady": True,
            "simulation_type": "RAS",
            "justification": "(wrong solver from a fake LLM, for testing)",
        }
    )
    config = select_solver_and_models(flow, retrieved_chunks=[], client=fake_client)
    assert config.solver_name == SolverName.SIMPLE_FOAM
    assert config.turbulence_model == TurbulenceModel.K_OMEGA_SST


def test_select_solver_and_models_overrides_turbulence_only_preserves_solver_choice():
    """An LLM's valid solver pick survives a turbulence-only override.

    Regression test: an LLM stating "Re is below 2300" and then recommending
    kOmegaSST instead of laminar (a self-contradiction observed with
    llama3.2:latest) must have its turbulence model corrected without
    discarding an otherwise-correct solver choice.
    """
    flow = FlowDescription(
        reynolds_number=500,
        geometry="straight pipe",
        fluid="water",
        is_compressible=False,
        is_steady=True,
    )
    fake_client = _FakeLLMClient(
        {
            "solver_name": "simpleFoam",
            "turbulence_model": "kOmegaSST",
            "is_compressible": False,
            "is_steady": True,
            "simulation_type": "RAS",
            "justification": "(self-contradictory: says Re<2300 but picks kOmegaSST)",
        }
    )
    config = select_solver_and_models(flow, retrieved_chunks=[], client=fake_client)
    assert config.solver_name == SolverName.SIMPLE_FOAM
    assert config.turbulence_model == TurbulenceModel.LAMINAR


def test_select_solver_and_models_overrides_both_solver_and_turbulence():
    """When both the solver and turbulence choices are wrong, both get replaced."""
    flow = FlowDescription(
        reynolds_number=500,
        geometry="External aerodynamics of a NACA0012 airfoil at low speed",
        fluid="air",
        is_compressible=False,
        is_steady=True,
    )
    fake_client = _FakeLLMClient(
        {
            "solver_name": "buoyantPimpleFoam",
            "turbulence_model": "kOmegaSST",
            "is_compressible": False,
            "is_steady": False,
            "simulation_type": "RAS",
            "justification": "(wrong on both counts, for testing)",
        }
    )
    config = select_solver_and_models(flow, retrieved_chunks=[], client=fake_client)
    assert config.solver_name == SolverName.SIMPLE_FOAM
    assert config.turbulence_model == TurbulenceModel.LAMINAR


def test_select_solver_and_models_does_not_override_turbulence_when_reynolds_unknown():
    """A missing Reynolds number must not force a turbulence-model override.

    _check_reynolds_turbulence_consistency reports a non-PASS "cannot
    verify" finding when Re is unknown; that is not evidence the LLM's
    choice is wrong and must not trigger the override.
    """
    flow = FlowDescription(
        geometry="circular pipe",
        fluid="air",
        is_compressible=False,
        is_steady=True,
    )
    fake_client = _FakeLLMClient(
        {
            "solver_name": "simpleFoam",
            "turbulence_model": "kOmegaSST",
            "is_compressible": False,
            "is_steady": True,
            "simulation_type": "RAS",
            "justification": "Assumed turbulent given no other information.",
        }
    )
    config = select_solver_and_models(flow, retrieved_chunks=[], client=fake_client)
    assert config.turbulence_model == TurbulenceModel.K_OMEGA_SST
    assert config.justification == "Assumed turbulent given no other information."


def test_select_solver_and_models_overrides_buoyant_solver_for_external_aero():
    """select_solver_and_models overrides an LLM's buoyant pick for an airfoil case.

    Regression test for the reported bug: the agent selected buoyantPimpleFoam
    for a NACA0012 airfoil. The LLM response is faked here to reproduce that
    exact wrong answer and verify the post-LLM sanity check catches it.
    """
    flow = FlowDescription(
        reynolds_number=500000,
        geometry="External aerodynamics of a NACA0012 airfoil at low speed",
        fluid="air",
        is_compressible=False,
        is_steady=True,
    )
    fake_client = _FakeLLMClient(
        {
            "solver_name": "buoyantPimpleFoam",
            "turbulence_model": "kOmegaSST",
            "is_compressible": False,
            "is_steady": False,
            "simulation_type": "RAS",
            "justification": "(wrong answer from a fake LLM, for testing)",
        }
    )
    config = select_solver_and_models(flow, retrieved_chunks=[], client=fake_client)
    assert config.solver_name == SolverName.SIMPLE_FOAM
    assert "overridden" in config.justification.lower()
    assert "simpleFoam" in config.justification


def test_select_solver_and_models_overrides_compressible_solver_for_low_speed_aero():
    """select_solver_and_models overrides an LLM's compressible pick for a low-speed airfoil case."""
    flow = FlowDescription(
        reynolds_number=500000,
        geometry="External aerodynamics of a NACA0012 airfoil at low speed",
        fluid="air",
        is_compressible=False,
        is_steady=True,
    )
    fake_client = _FakeLLMClient(
        {
            "solver_name": "rhoSimpleFoam",
            "turbulence_model": "kOmegaSST",
            "is_compressible": False,
            "is_steady": True,
            "simulation_type": "RAS",
            "justification": "(wrong answer from a fake LLM, for testing)",
        }
    )
    config = select_solver_and_models(flow, retrieved_chunks=[], client=fake_client)
    assert config.solver_name == SolverName.SIMPLE_FOAM


def test_select_solver_and_models_overrides_algorithm_mismatch_for_natural_convection():
    """select_solver_and_models overrides an internally-inconsistent buoyantPimpleFoam+steady pick."""
    flow = FlowDescription(
        geometry="Natural convection in a heated square cavity",
        fluid="air",
        temperature_dependent=True,
        is_compressible=False,
        is_steady=True,
    )
    fake_client = _FakeLLMClient(
        {
            "solver_name": "buoyantPimpleFoam",
            "turbulence_model": "kOmegaSST",
            "is_compressible": False,
            "is_steady": True,
            "simulation_type": "RAS",
            "justification": "(internally inconsistent answer from a fake LLM, for testing)",
        }
    )
    config = select_solver_and_models(flow, retrieved_chunks=[], client=fake_client)
    assert config.solver_name == SolverName.BUOYANT_SIMPLE_FOAM


def test_select_solver_and_models_keeps_valid_llm_choice():
    """select_solver_and_models does not override a physically sound LLM choice."""
    flow = FlowDescription(
        reynolds_number=500000,
        geometry="External aerodynamics of a NACA0012 airfoil at low speed",
        fluid="air",
        is_compressible=False,
        is_steady=True,
    )
    fake_client = _FakeLLMClient(
        {
            "solver_name": "simpleFoam",
            "turbulence_model": "kOmegaSST",
            "is_compressible": False,
            "is_steady": True,
            "simulation_type": "RAS",
            "justification": "Low-speed external aerodynamics.",
        }
    )
    config = select_solver_and_models(flow, retrieved_chunks=[], client=fake_client)
    assert config.solver_name == SolverName.SIMPLE_FOAM
    assert config.justification == "Low-speed external aerodynamics."


def test_route_after_retrieval_retries_on_low_quality():
    """The retrieval routing loop retries when quality is below threshold and attempts remain."""
    state = initial_state("query")
    state["retrieval_quality"] = 0.1
    state["retrieval_attempts"] = 1
    state["iteration_count"] = 2
    assert _route_after_retrieval(state) == "retrieve_cfd_knowledge"


def test_route_after_retrieval_proceeds_when_attempts_exhausted():
    """The retrieval routing loop proceeds once max retrieval attempts are reached."""
    state = initial_state("query")
    state["retrieval_quality"] = 0.1
    state["retrieval_attempts"] = 2
    state["iteration_count"] = 2
    assert _route_after_retrieval(state) == "select_solver_and_models"


def test_route_after_retrieval_proceeds_when_iteration_cap_hit():
    """The retrieval routing loop hard-stops once MAX_AGENT_ITERATIONS is reached."""
    state = initial_state("query")
    state["retrieval_quality"] = 0.0
    state["retrieval_attempts"] = 0
    state["iteration_count"] = 999
    assert _route_after_retrieval(state) == "select_solver_and_models"


def test_route_after_validation_retries_on_failed_validation():
    """The validation routing loop retries solver selection on a failed validation."""
    state = initial_state("query")
    state["iteration_count"] = 3
    state["validation_retries"] = 0
    failed = ValidationResult(findings=[])
    from src.generation.schemas import ValidationFinding, ValidationSeverity

    failed.findings.append(
        ValidationFinding(rule="test", severity=ValidationSeverity.ERROR, message="bad")
    )
    state["validation_results"] = failed.model_dump(mode="json")
    assert _route_after_validation(state) == "select_solver_and_models"
    assert state["validation_retries"] == 1


def test_route_after_validation_proceeds_when_passed():
    """The validation routing loop proceeds to the final response when validation passes."""
    state = initial_state("query")
    state["iteration_count"] = 3
    state["validation_retries"] = 0
    state["validation_results"] = ValidationResult(findings=[]).model_dump(mode="json")
    assert _route_after_validation(state) == "format_final_response"
