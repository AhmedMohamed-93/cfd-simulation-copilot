"""Tests for the LangGraph agent: state, tools fallback logic, and routing."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from src.agent.graph import _route_after_retrieval, _route_after_validation, run_agent
from src.agent.state import initial_state
from src.agent.tools import (
    _extract_json,
    _field_hint,
    _response_instruction,
    _rule_based_solver_selection,
    format_final_response,
    select_solver_and_models,
)
from src.generation.schemas import (
    FlowDescription,
    SimulationType,
    SolverConfiguration,
    SolverName,
    TurbulenceModel,
    ValidationResult,
)


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


def test_rule_based_solver_selection_justification_is_framed_as_physics_rules(laminar_pipe_flow):
    """The justification reads as a deliberate physics decision, not a failure path.

    Regression test: "Rule-based fallback: ..." undersold this as a
    last-resort path when it's actually the deterministic invariant
    enforcement layer the system is designed around.
    """
    config = _rule_based_solver_selection(laminar_pipe_flow)
    assert config.justification.startswith("Selected via physics decision rules: ")
    assert "Rule-based fallback" not in config.justification
    assert "Re=" in config.justification
    assert "-> simpleFoam." in config.justification


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
    geometry must route to simpleFoam/rhoSimpleFoam, not buoyantSimpleFoam:
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
    all; icoFoam is OpenFOAM's own bundled tutorial solver for this case.
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
    cases, not any low-Re unsteady flow: a cylinder vortex-shedding case at
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
    fallback always choosing kOmegaSST, never SpalartAllmaras, the
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


def test_rule_based_solver_selection_transonic_airfoil_does_not_select_spalart_allmaras():
    """A transonic/compressible airfoil does not get SpalartAllmaras from the fallback.

    Regression test: evaluation suite case tc15 ("Transonic flow over a
    supercritical airfoil at Mach 0.78... shock/boundary-layer interaction")
    expects kOmegaSST: a blanket "any external aero -> SpalartAllmaras"
    rule would get this wrong.
    """
    flow = FlowDescription(
        reynolds_number=5000000,
        mach_number=0.78,
        geometry="Transonic flow over a supercritical airfoil",
        fluid="air",
        is_compressible=True,
        is_steady=True,
    )
    config = _rule_based_solver_selection(flow)
    assert config.turbulence_model == TurbulenceModel.K_OMEGA_SST


def test_rule_based_solver_selection_turbine_blade_does_not_select_spalart_allmaras():
    """A wind turbine blade does not get SpalartAllmaras from the fallback.

    Regression test: evaluation suite case tc20 ("Turbulent external flow
    over a wind turbine blade section") expects kOmegaSST: external
    aerodynamics keyword matching alone is too broad for the SpalartAllmaras
    preference; it must be scoped to airfoil/wing/NACA geometries.
    """
    flow = FlowDescription(
        reynolds_number=1000000,
        geometry="Turbulent external flow over a wind turbine blade section",
        fluid="air",
        is_compressible=False,
        is_steady=True,
    )
    config = _rule_based_solver_selection(flow)
    assert config.turbulence_model == TurbulenceModel.K_OMEGA_SST


def test_select_solver_and_models_overrides_turbulence_preference_for_low_speed_airfoil():
    """select_solver_and_models corrects kOmegaSST to SpalartAllmaras for a low-speed airfoil.

    Regression test for evaluation case tc05: the LLM's own solver pick
    (simpleFoam) was correct, but its turbulence pick (kOmegaSST) merely
    passed the coarse laminar/non-laminar check without matching the
    case-type standard: this is what the turbulence_model_preference check
    is for, distinct from reynolds_turbulence_consistency.
    """
    flow = FlowDescription(
        reynolds_number=3000000,
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
            "justification": "(technically valid but non-standard, for testing)",
        }
    )
    config = select_solver_and_models(flow, retrieved_chunks=[], client=fake_client)
    assert config.solver_name == SolverName.SIMPLE_FOAM
    assert config.turbulence_model == TurbulenceModel.SPALART_ALLMARAS
    assert "corrected LLM proposal of 'kOmegaSST'" in config.justification
    assert "turbulence_model_preference" in config.justification


def test_select_solver_and_models_overrides_solver_disagreeing_with_flow_steadiness():
    """select_solver_and_models corrects a self-consistent-but-flow-wrong solver proposal.

    Regression test for a real gap exposed by evaluation (case tc03): the
    LLM proposed simpleFoam + is_steady=True, internally self-consistent,
    so the old algorithm-consistency check (config-only) passed it, for a
    query explicitly describing "Unsteady flow past a circular cylinder...
    vortex shedding" (flow.is_steady=False). The override must catch this
    via the flow-vs-config comparison, not just config's own consistency.
    """
    flow = FlowDescription(
        reynolds_number=1000,
        geometry="flow past a circular cylinder",
        fluid="air",
        is_compressible=False,
        is_steady=False,
    )
    fake_client = _FakeLLMClient(
        {
            "solver_name": "simpleFoam",
            "turbulence_model": "kOmegaSST",
            "is_compressible": False,
            "is_steady": True,
            "simulation_type": "RAS",
            "justification": "(internally consistent but wrong for this flow, for testing)",
        }
    )
    config = select_solver_and_models(flow, retrieved_chunks=[], client=fake_client)
    assert config.solver_name == SolverName.PIMPLE_FOAM
    assert config.is_steady is False


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
    assert config.justification.startswith("Selected via physics decision rules")
    assert "corrected LLM proposal of 'rhoSimpleFoam'" in config.justification
    assert "retained" in config.justification.lower()


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
    assert config.justification.startswith("Selected via physics decision rules")
    assert "corrected LLM proposal of 'kOmegaSST'" in config.justification
    assert "retained" in config.justification.lower()


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
    assert config.justification.startswith("Selected via physics decision rules")
    assert "corrected LLM proposal of 'buoyantPimpleFoam / kOmegaSST'" in config.justification


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
            # SpalartAllmaras (not kOmegaSST) so this test isolates the
            # solver-only override path; a NACA0012 airfoil expects
            # SpalartAllmaras (see turbulence_model_preference), which
            # would otherwise also flag turbulence as bad here.
            "turbulence_model": "SpalartAllmaras",
            "is_compressible": False,
            "is_steady": False,
            "simulation_type": "RAS",
            "justification": "(wrong answer from a fake LLM, for testing)",
        }
    )
    config = select_solver_and_models(flow, retrieved_chunks=[], client=fake_client)
    assert config.solver_name == SolverName.SIMPLE_FOAM
    assert config.justification.startswith("Selected via physics decision rules")
    assert "corrected LLM proposal of 'buoyantPimpleFoam'" in config.justification
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
            # SpalartAllmaras is the standard choice for a low-speed
            # NACA0012 airfoil (see turbulence_model_preference); this is
            # what makes the whole answer "physically sound" and therefore
            # not subject to override.
            "turbulence_model": "SpalartAllmaras",
            "is_compressible": False,
            "is_steady": True,
            "simulation_type": "RAS",
            "justification": "Low-speed external aerodynamics.",
        }
    )
    config = select_solver_and_models(flow, retrieved_chunks=[], client=fake_client)
    assert config.solver_name == SolverName.SIMPLE_FOAM
    assert config.justification == "Low-speed external aerodynamics."


def test_format_final_response_citation_line_shows_title_relevance_and_quality(
    laminar_pipe_flow, solver_config_laminar
):
    """A citation renders as 'title, relevance X% (quality match)'.

    Every ingested title already embeds its human-readable category (e.g.
    "OpenFOAM User Guide: Meshing Guidelines"; see document_loader.py);
    `source` is a separate internal slug ("openfoam-user-guide-synthetic"),
    not a display name, so it must not be prepended to the title.

    Regression test for the reported UX bug: min-max normalization alone
    let a "Reynolds stress model" wiki page show 99-100% relevance for a
    laminar pipe flow query. match_quality is the absolute floor that
    surfaces that mismatch to the reader instead of hiding it behind a
    batch-relative percentage.
    """
    citations = [
        {
            "title": "OpenFOAM User Guide: Meshing Guidelines",
            "source": "openfoam-user-guide-synthetic",
            "url": "",
            "rerank_score": 10.0,
            "raw_rerank_score": 2.0,
            "match_quality": "strong",
        },
        {
            "title": "CFD-Online Wiki: Reynolds stress model",
            "source": "cfd-online-wiki-synthetic",
            "url": "",
            "rerank_score": 9.9,
            "raw_rerank_score": -7.0,
            "match_quality": "weak",
        },
    ]
    response = format_final_response(
        laminar_pipe_flow,
        solver_config_laminar,
        generated_files={},
        validation_result=ValidationResult(findings=[]),
        citations=citations,
    )
    assert "OpenFOAM User Guide: Meshing Guidelines, relevance 100% (strong match)" in response
    assert "CFD-Online Wiki: Reynolds stress model, relevance 99% (weak match)" in response
    assert "openfoam-user-guide-synthetic" not in response


def test_format_final_response_adds_low_confidence_note_when_all_citations_weak(
    laminar_pipe_flow, solver_config_laminar
):
    """A note is appended when every citation is a weak match."""
    citations = [
        {
            "title": "Reynolds stress model",
            "source": "CFD-Online Wiki",
            "url": "",
            "rerank_score": 10.0,
            "raw_rerank_score": -6.0,
            "match_quality": "weak",
        },
        {
            "title": "Large Eddy Simulation",
            "source": "CFD-Online Wiki",
            "url": "",
            "rerank_score": 8.0,
            "raw_rerank_score": -9.0,
            "match_quality": "weak",
        },
    ]
    response = format_final_response(
        laminar_pipe_flow,
        solver_config_laminar,
        generated_files={},
        validation_result=ValidationResult(findings=[]),
        citations=citations,
    )
    assert "retrieval confidence was low for this query" in response
    assert "physics decision rules rather than retrieved documentation" in response


def test_format_final_response_omits_low_confidence_note_when_not_all_weak(
    laminar_pipe_flow, solver_config_laminar
):
    """No low-confidence note when at least one citation is a good match."""
    citations = [
        {
            "title": "Meshing Guidelines",
            "source": "OpenFOAM User Guide",
            "url": "",
            "rerank_score": 10.0,
            "raw_rerank_score": 2.0,
            "match_quality": "strong",
        },
        {
            "title": "Reynolds stress model",
            "source": "CFD-Online Wiki",
            "url": "",
            "rerank_score": 3.0,
            "raw_rerank_score": -9.0,
            "match_quality": "weak",
        },
    ]
    response = format_final_response(
        laminar_pipe_flow,
        solver_config_laminar,
        generated_files={},
        validation_result=ValidationResult(findings=[]),
        citations=citations,
    )
    assert "retrieval confidence was low" not in response


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


def test_run_agent_streams_progress_via_on_step_callback():
    """run_agent's on_step callback fires once per node, in graph order.

    Regression test for switching run_agent's internals from .invoke() to
    manually consuming .stream(..., stream_mode="updates"); this must
    still traverse every node exactly as before and produce an equivalent
    final state, while additionally reporting live progress (needed so the
    API can report which step a long local-Ollama run is on).

    Every tool call is mocked so this stays fast and deterministic (no real
    LLM, Qdrant, or embedding model calls), and the fakes are chosen to
    pass on the first try (retrieval quality above threshold, validation
    passed) so the graph takes the single-pass route with no retry loops.
    """
    fake_flow = FlowDescription(
        geometry="pipe", reynolds_number=50000, fluid="water", is_steady=True, is_compressible=False
    )
    fake_config = SolverConfiguration(
        solver_name=SolverName.SIMPLE_FOAM,
        turbulence_model=TurbulenceModel.K_OMEGA_SST,
        is_compressible=False,
        is_steady=True,
        simulation_type=SimulationType.RAS,
        justification="test",
    )
    fake_files = {"system/controlDict": "dummy content"}
    fake_validation = ValidationResult(findings=[])

    with (
        patch("src.agent.graph.CFDRetriever"),
        patch("src.agent.graph.agent_tools.parse_flow_description", return_value=fake_flow),
        patch(
            "src.agent.graph.agent_tools.retrieve_cfd_knowledge",
            return_value={"chunks": [], "quality": 1.0, "aspect": "solver selection"},
        ),
        patch("src.agent.graph.agent_tools.select_solver_and_models", return_value=fake_config),
        patch("src.agent.graph.agent_tools.generate_openfoam_case_files", return_value=fake_files),
        patch("src.agent.graph.agent_tools.validate_case_physics", return_value=fake_validation),
        patch("src.agent.graph.agent_tools.format_final_response", return_value="## done"),
    ):
        seen_nodes: list[str] = []
        final_state = run_agent(
            "Turbulent pipe flow at Re=50000",
            on_step=lambda node_name, _state: seen_nodes.append(node_name),
        )

    assert seen_nodes == [
        "parse_flow_description",
        "retrieve_cfd_knowledge",
        "select_solver_and_models",
        "generate_openfoam_files",
        "validate_physics",
        "format_final_response",
    ]
    assert final_state["final_response"] == "## done"
    assert final_state["error"] is None
    assert final_state["solver_config"]["solver_name"] == "simpleFoam"


def test_run_agent_without_on_step_behaves_like_before():
    """run_agent(query) with no callback still returns the same final state.

    Ensures the .invoke() -> manual .stream() switch didn't change default
    (no-callback) behavior for existing callers (e.g. the evaluation script).
    """
    fake_flow = FlowDescription(
        geometry="pipe", reynolds_number=500, fluid="water", is_steady=True, is_compressible=False
    )
    fake_config = SolverConfiguration(
        solver_name=SolverName.SIMPLE_FOAM,
        turbulence_model=TurbulenceModel.LAMINAR,
        is_compressible=False,
        is_steady=True,
        simulation_type=SimulationType.LAMINAR,
        justification="test",
    )
    fake_files = {"system/controlDict": "dummy content"}
    fake_validation = ValidationResult(findings=[])

    with (
        patch("src.agent.graph.CFDRetriever"),
        patch("src.agent.graph.agent_tools.parse_flow_description", return_value=fake_flow),
        patch(
            "src.agent.graph.agent_tools.retrieve_cfd_knowledge",
            return_value={"chunks": [], "quality": 1.0, "aspect": "solver selection"},
        ),
        patch("src.agent.graph.agent_tools.select_solver_and_models", return_value=fake_config),
        patch("src.agent.graph.agent_tools.generate_openfoam_case_files", return_value=fake_files),
        patch("src.agent.graph.agent_tools.validate_case_physics", return_value=fake_validation),
        patch("src.agent.graph.agent_tools.format_final_response", return_value="## done"),
    ):
        final_state = run_agent("Laminar pipe flow at Re=500")

    assert final_state["final_response"] == "## done"
    assert final_state["error"] is None
    assert final_state["reasoning_steps"]
