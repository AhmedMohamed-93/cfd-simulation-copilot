"""Tests for OpenFOAM file generation, schemas, and physics validation."""

from __future__ import annotations

from src.generation.openfoam_generator import (
    check_openfoam_syntax,
    generate_openfoam_files,
)
from src.generation.physics_validator import (
    _check_reynolds_turbulence_consistency,
    _check_solver_algorithm_consistency,
    _check_turbulence_model_preference,
    _laminar_to_turbulent_threshold,
    _prefers_free_shear_kepsilon,
    _prefers_spalart_allmaras,
    is_external_aerodynamics_geometry,
    validate_physics,
)
from src.generation.schemas import (
    BoundaryCondition,
    ControlDictConfig,
    FieldFile,
    FlowDescription,
    SimulationType,
    SolverConfiguration,
    SolverName,
    TransportProperties,
    TurbulenceModel,
    ValidationSeverity,
)


def test_control_dict_to_openfoam_string_contains_required_entries():
    """ControlDictConfig renders a valid controlDict with the chosen solver."""
    config = ControlDictConfig(solver=SolverName.SIMPLE_FOAM, endTime=500, deltaT=1)
    text = config.to_openfoam_string()
    assert "FoamFile" in text
    assert "application     simpleFoam;" in text
    assert "endTime         500" in text


def test_field_file_to_openfoam_string_renders_boundary_patches():
    """FieldFile renders every boundary patch with its type and value."""
    field = FieldFile(
        field_name="U",
        foam_class="volVectorField",
        dimensions="[0 1 -1 0 0 0 0]",
        internal_field="uniform (0 0 0)",
        boundary_field={
            "inlet": BoundaryCondition(type="fixedValue", value="uniform (10 0 0)"),
            "walls": BoundaryCondition(type="noSlip"),
        },
    )
    text = field.to_openfoam_string()
    assert "inlet" in text
    assert "fixedValue" in text
    assert "walls" in text
    assert "noSlip" in text


def test_transport_properties_renders_scientific_viscosity():
    """TransportProperties renders nu using scientific notation."""
    props = TransportProperties(kinematic_viscosity=1.5e-5)
    text = props.to_openfoam_string()
    assert "nu " in text
    assert "1.500e-05" in text


def test_generate_openfoam_files_laminar_case_has_no_turbulence_fields(laminar_pipe_flow, solver_config_laminar):
    """A laminar solver configuration should not generate k/epsilon/omega files."""
    files = generate_openfoam_files(solver_config_laminar, laminar_pipe_flow)
    assert "0/k" not in files
    assert "0/epsilon" not in files
    assert "0/omega" not in files
    assert "0/U" in files
    assert "0/p" in files


def test_generate_openfoam_files_komega_case_includes_k_and_omega(turbulent_pipe_flow, solver_config_simple_komega):
    """A kOmegaSST configuration must generate 0/k and 0/omega but not 0/epsilon."""
    files = generate_openfoam_files(solver_config_simple_komega, turbulent_pipe_flow)
    assert "0/k" in files
    assert "0/omega" in files
    assert "0/epsilon" not in files


def test_generated_files_all_pass_syntax_check(turbulent_pipe_flow, solver_config_simple_komega):
    """Every generated file must pass the basic OpenFOAM syntax checker."""
    files = generate_openfoam_files(solver_config_simple_komega, turbulent_pipe_flow)
    for path, content in files.items():
        is_valid, problems = check_openfoam_syntax(content)
        assert is_valid, f"{path} failed syntax check: {problems}"


def test_check_openfoam_syntax_flags_unbalanced_braces():
    """check_openfoam_syntax detects unbalanced curly braces."""
    broken = "FoamFile\n{\n    class dictionary;\n"
    is_valid, problems = check_openfoam_syntax(broken)
    assert not is_valid
    assert any("curly braces" in p for p in problems)


def test_check_openfoam_syntax_flags_missing_header():
    """check_openfoam_syntax detects a missing FoamFile header."""
    broken = "just some random text with no header"
    is_valid, problems = check_openfoam_syntax(broken)
    assert not is_valid
    assert any("FoamFile" in p for p in problems)


def test_validate_physics_flags_laminar_re_turbulence_mismatch(turbulent_pipe_flow, solver_config_laminar):
    """A high-Re flow with a laminar model should trigger a warning, not silence."""
    files = generate_openfoam_files(solver_config_laminar, turbulent_pipe_flow)
    result = validate_physics(turbulent_pipe_flow, solver_config_laminar, files)
    reynolds_finding = next(f for f in result.findings if f.rule == "reynolds_turbulence_consistency")
    assert reynolds_finding.severity == ValidationSeverity.WARNING


def test_validate_physics_passes_for_consistent_laminar_case(laminar_pipe_flow, solver_config_laminar):
    """A low-Re flow correctly using a laminar model should pass validation."""
    files = generate_openfoam_files(solver_config_laminar, laminar_pipe_flow)
    result = validate_physics(laminar_pipe_flow, solver_config_laminar, files)
    assert result.passed


def test_validate_physics_detects_missing_pressure_reference(turbulent_pipe_flow, solver_config_simple_komega):
    """Removing the fixedValue pressure BC should raise a critical error for incompressible solvers."""
    files = generate_openfoam_files(solver_config_simple_komega, turbulent_pipe_flow)
    files["0/p"] = files["0/p"].replace("fixedValue", "zeroGradient")
    result = validate_physics(turbulent_pipe_flow, solver_config_simple_komega, files)
    assert not result.passed
    assert any(f.rule == "pressure_reference" and f.severity == ValidationSeverity.ERROR for f in result.findings)


def test_validate_physics_detects_boundary_condition_mismatch(turbulent_pipe_flow, solver_config_simple_komega):
    """Removing a patch from one field file only should be flagged as incomplete."""
    files = generate_openfoam_files(solver_config_simple_komega, turbulent_pipe_flow)
    files["0/p"] = files["0/p"].replace(
        "    frontAndBack\n    {\n        type            empty;\n    }\n", ""
    )
    result = validate_physics(turbulent_pipe_flow, solver_config_simple_komega, files)
    assert not result.passed
    assert any(f.rule == "boundary_condition_completeness" for f in result.errors)


def test_is_external_aerodynamics_geometry_detects_keywords():
    """is_external_aerodynamics_geometry matches airfoil/wing/NACA/etc. case-insensitively."""
    assert is_external_aerodynamics_geometry("External aerodynamics of a NACA0012 airfoil")
    assert is_external_aerodynamics_geometry("flow over a WING")
    assert is_external_aerodynamics_geometry("aircraft fuselage")
    assert is_external_aerodynamics_geometry("turbine blade")


def test_is_external_aerodynamics_geometry_ignores_unrelated_text():
    """is_external_aerodynamics_geometry does not false-positive on unrelated geometry."""
    assert not is_external_aerodynamics_geometry("heated square cavity")
    assert not is_external_aerodynamics_geometry("pipe")
    assert not is_external_aerodynamics_geometry("unspecified")


def test_validate_physics_flags_buoyant_solver_for_external_aerodynamics():
    """A buoyant solver selected for an airfoil case is a critical, physically wrong error.

    Regression test for the reported bug: the agent selected buoyantPimpleFoam
    for a NACA0012 airfoil, which is physically nonsensical: external
    aerodynamics is driven by the free-stream velocity, not buoyancy.
    """
    flow = FlowDescription(
        reynolds_number=500000,
        geometry="External aerodynamics of a NACA0012 airfoil at low speed",
        fluid="air",
        is_compressible=False,
        is_steady=True,
    )
    config = SolverConfiguration(
        solver_name=SolverName.BUOYANT_PIMPLE_FOAM,
        turbulence_model=TurbulenceModel.K_OMEGA_SST,
        is_compressible=False,
        is_steady=False,
        simulation_type=SimulationType.RAS,
        justification="(deliberately wrong, for testing)",
    )
    files = generate_openfoam_files(config, flow)
    result = validate_physics(flow, config, files)

    assert not result.passed
    finding = next(f for f in result.findings if f.rule == "solver_domain_consistency")
    assert finding.severity == ValidationSeverity.ERROR
    assert "buoyant solvers are for buoyancy-driven flows, not external aerodynamics" in finding.message


def test_validate_physics_allows_buoyant_solver_for_natural_convection():
    """A buoyant solver for an actual buoyancy-driven case (heated cavity) passes the check."""
    flow = FlowDescription(
        geometry="Natural convection in a heated square cavity",
        fluid="air",
        temperature_dependent=True,
        is_compressible=True,
        is_steady=True,
    )
    config = SolverConfiguration(
        solver_name=SolverName.BUOYANT_SIMPLE_FOAM,
        turbulence_model=TurbulenceModel.LAMINAR,
        is_compressible=True,
        is_steady=True,
        simulation_type=SimulationType.LAMINAR,
        justification="Natural convection: buoyancy drives the flow.",
    )
    files = generate_openfoam_files(config, flow)
    result = validate_physics(flow, config, files)

    finding = next(f for f in result.findings if f.rule == "solver_domain_consistency")
    assert finding.severity == ValidationSeverity.PASS


def test_validate_physics_flags_compressible_solver_for_low_speed_external_aerodynamics():
    """rhoSimpleFoam for a stated low-speed, incompressible airfoil case is flagged.

    Regression test for a related mistake spotted alongside the buoyant-solver
    bug: a small LLM can conflate a high Reynolds number with a high Mach
    number and pick the compressible solver family even though the case is
    explicitly incompressible/low-speed.
    """
    flow = FlowDescription(
        reynolds_number=500000,
        geometry="External aerodynamics of a NACA0012 airfoil at low speed",
        fluid="air",
        is_compressible=False,
        is_steady=True,
    )
    config = SolverConfiguration(
        solver_name=SolverName.RHO_SIMPLE_FOAM,
        turbulence_model=TurbulenceModel.K_OMEGA_SST,
        is_compressible=False,
        is_steady=True,
        simulation_type=SimulationType.RAS,
        justification="(deliberately wrong, for testing)",
    )
    files = generate_openfoam_files(config, flow)
    result = validate_physics(flow, config, files)

    assert not result.passed
    finding = next(f for f in result.findings if f.rule == "solver_domain_consistency")
    assert finding.severity == ValidationSeverity.ERROR
    assert "low-speed" in finding.message


def test_validate_physics_flags_compressible_solver_for_incompressible_internal_flow():
    """rhoSimpleFoam for a stated incompressible internal pipe flow is also flagged.

    Regression test: evaluation on turbulent pipe flow (Re=50000) showed the
    same "reaches for rho* because Re is high" mistake for ordinary internal
    flow, not just external aerodynamics, so this check must not be gated
    on geometry.
    """
    flow = FlowDescription(
        reynolds_number=50000,
        geometry="circular pipe",
        fluid="air",
        is_compressible=False,
        is_steady=True,
    )
    config = SolverConfiguration(
        solver_name=SolverName.RHO_SIMPLE_FOAM,
        turbulence_model=TurbulenceModel.K_OMEGA_SST,
        is_compressible=False,
        is_steady=True,
        simulation_type=SimulationType.RAS,
        justification="(deliberately wrong, for testing)",
    )
    files = generate_openfoam_files(config, flow)
    result = validate_physics(flow, config, files)

    assert not result.passed
    finding = next(f for f in result.findings if f.rule == "solver_domain_consistency")
    assert finding.severity == ValidationSeverity.ERROR
    assert "low-speed" in finding.message


def test_validate_physics_allows_compressible_solver_for_high_speed_external_aerodynamics():
    """rhoSimpleFoam is fine for a genuinely compressible/high-speed external aero case."""
    flow = FlowDescription(
        reynolds_number=3000000,
        mach_number=0.8,
        geometry="External aerodynamics of a transonic wing",
        fluid="air",
        is_compressible=True,
        is_steady=True,
    )
    config = SolverConfiguration(
        solver_name=SolverName.RHO_SIMPLE_FOAM,
        turbulence_model=TurbulenceModel.K_OMEGA_SST,
        is_compressible=True,
        is_steady=True,
        simulation_type=SimulationType.RAS,
        justification="Transonic external aerodynamics.",
    )
    files = generate_openfoam_files(config, flow)
    result = validate_physics(flow, config, files)

    finding = next(f for f in result.findings if f.rule == "solver_domain_consistency")
    assert finding.severity == ValidationSeverity.PASS


def test_check_solver_algorithm_consistency_flags_pimple_solver_marked_steady():
    """buoyantPimpleFoam (transient-only) with is_steady=True is an internal contradiction.

    Regression test: an LLM selected buoyantPimpleFoam for a case it itself
    marked is_steady=True. Since controlDict generation derives the ddt
    scheme from is_steady independently of solver_name, this mismatch would
    silently produce a broken case (transient solver, steadyState schemes).
    flow.is_steady is set to match config.is_steady here so this test
    isolates the internal-contradiction failure mode specifically.
    """
    flow = FlowDescription(geometry="heated cavity", fluid="air", is_steady=True)
    config = SolverConfiguration(
        solver_name=SolverName.BUOYANT_PIMPLE_FOAM,
        turbulence_model=TurbulenceModel.K_OMEGA_SST,
        is_compressible=False,
        is_steady=True,
        simulation_type=SimulationType.RAS,
        justification="(deliberately inconsistent, for testing)",
    )
    finding = _check_solver_algorithm_consistency(flow, config)
    assert finding.severity == ValidationSeverity.ERROR
    assert finding.rule == "solver_algorithm_consistency"


def test_check_solver_algorithm_consistency_flags_simple_solver_marked_unsteady():
    """simpleFoam (steady-only) with is_steady=False is an internal contradiction."""
    flow = FlowDescription(geometry="pipe", fluid="air", is_steady=False)
    config = SolverConfiguration(
        solver_name=SolverName.SIMPLE_FOAM,
        turbulence_model=TurbulenceModel.K_OMEGA_SST,
        is_compressible=False,
        is_steady=False,
        simulation_type=SimulationType.RAS,
        justification="(deliberately inconsistent, for testing)",
    )
    finding = _check_solver_algorithm_consistency(flow, config)
    assert finding.severity == ValidationSeverity.ERROR
    assert finding.rule == "solver_algorithm_consistency"


def test_check_solver_algorithm_consistency_passes_for_matching_pairs():
    """buoyantSimpleFoam+steady and buoyantPimpleFoam+transient both pass, flow included."""
    steady_flow = FlowDescription(geometry="heated cavity", fluid="air", is_steady=True)
    steady_config = SolverConfiguration(
        solver_name=SolverName.BUOYANT_SIMPLE_FOAM,
        turbulence_model=TurbulenceModel.LAMINAR,
        is_compressible=True,
        is_steady=True,
        simulation_type=SimulationType.LAMINAR,
        justification="Steady natural convection.",
    )
    transient_flow = FlowDescription(geometry="heated cavity", fluid="air", is_steady=False)
    transient_config = SolverConfiguration(
        solver_name=SolverName.BUOYANT_PIMPLE_FOAM,
        turbulence_model=TurbulenceModel.LAMINAR,
        is_compressible=True,
        is_steady=False,
        simulation_type=SimulationType.LAMINAR,
        justification="Transient natural convection.",
    )
    assert _check_solver_algorithm_consistency(steady_flow, steady_config).severity == ValidationSeverity.PASS
    assert (
        _check_solver_algorithm_consistency(transient_flow, transient_config).severity
        == ValidationSeverity.PASS
    )


def test_check_solver_algorithm_consistency_flags_solver_disagreeing_with_flow_steadiness():
    """An internally self-consistent config can still disagree with the flow, and must be flagged.

    Regression test for a real gap exposed by evaluation: the LLM proposed
    simpleFoam + is_steady=True (internally self-consistent, so the old
    config-only check passed it) for a query explicitly describing
    "Unsteady flow past a circular cylinder... vortex shedding"
    (flow.is_steady=False, correctly parsed). The config's own internal
    consistency says nothing about whether it's the RIGHT algorithm for the
    actual flow; this must be checked against flow.is_steady directly.
    """
    flow = FlowDescription(
        reynolds_number=1000,
        geometry="flow past a circular cylinder",
        fluid="air",
        is_compressible=False,
        is_steady=False,
    )
    config = SolverConfiguration(
        solver_name=SolverName.SIMPLE_FOAM,
        turbulence_model=TurbulenceModel.K_OMEGA_SST,
        is_compressible=False,
        is_steady=True,
        simulation_type=SimulationType.RAS,
        justification="(internally consistent but wrong for this flow, for testing)",
    )
    finding = _check_solver_algorithm_consistency(flow, config)
    assert finding.severity == ValidationSeverity.ERROR
    assert finding.rule == "solver_algorithm_consistency"
    assert "does not match the actual flow regime" in finding.message


def test_validate_physics_allows_non_buoyant_solver_for_external_aerodynamics():
    """simpleFoam for an airfoil case passes the solver_domain_consistency check."""
    flow = FlowDescription(
        reynolds_number=3000000,
        geometry="External aerodynamics of a NACA0012 airfoil",
        fluid="air",
        is_compressible=False,
        is_steady=True,
    )
    config = SolverConfiguration(
        solver_name=SolverName.SIMPLE_FOAM,
        turbulence_model=TurbulenceModel.K_OMEGA_SST,
        is_compressible=False,
        is_steady=True,
        simulation_type=SimulationType.RAS,
        justification="Low-speed external aerodynamics.",
    )
    files = generate_openfoam_files(config, flow)
    result = validate_physics(flow, config, files)

    finding = next(f for f in result.findings if f.rule == "solver_domain_consistency")
    assert finding.severity == ValidationSeverity.PASS


def test_laminar_to_turbulent_threshold_is_lower_for_bluff_body_geometry():
    """Cylinder/sphere wakes use a lower laminar/turbulent threshold than pipe flow."""
    assert _laminar_to_turbulent_threshold("flow past a circular cylinder") == 300.0
    assert _laminar_to_turbulent_threshold("circular pipe") == 2300.0


def test_reynolds_turbulence_consistency_allows_turbulent_model_for_cylinder_wake_at_moderate_re():
    """A turbulence model for a Re=1000 cylinder wake passes, unlike the same Re for a pipe.

    Regression test: evaluation on the cylinder vortex-shedding case (tc03)
    showed the generic 2300 pipe-flow threshold incorrectly demanding a
    laminar model for a bluff-body wake that is genuinely transitional well
    below that Reynolds number.
    """
    flow = FlowDescription(
        reynolds_number=1000,
        geometry="flow past a circular cylinder",
        fluid="air",
        is_compressible=False,
        is_steady=False,
    )
    config = SolverConfiguration(
        solver_name=SolverName.PIMPLE_FOAM,
        turbulence_model=TurbulenceModel.K_OMEGA_SST,
        is_compressible=False,
        is_steady=False,
        simulation_type=SimulationType.RAS,
        justification="Vortex shedding behind a cylinder.",
    )
    finding = _check_reynolds_turbulence_consistency(flow, config)
    assert finding.severity == ValidationSeverity.PASS


def test_reynolds_turbulence_consistency_still_flags_laminar_model_for_pipe_at_same_re():
    """The same Re=1000 with a pipe (not bluff-body) geometry still expects laminar."""
    flow = FlowDescription(
        reynolds_number=1000,
        geometry="circular pipe",
        fluid="air",
        is_compressible=False,
        is_steady=False,
    )
    config = SolverConfiguration(
        solver_name=SolverName.PIMPLE_FOAM,
        turbulence_model=TurbulenceModel.K_OMEGA_SST,
        is_compressible=False,
        is_steady=False,
        simulation_type=SimulationType.RAS,
        justification="(deliberately wrong, for testing)",
    )
    finding = _check_reynolds_turbulence_consistency(flow, config)
    assert finding.severity == ValidationSeverity.WARNING


def test_validate_physics_flags_non_icofoam_solver_for_cavity_case():
    """pimpleFoam for a canonical lid-driven-cavity case is flagged in favor of icoFoam.

    Regression test: evaluation on the lid-driven cavity case (tc04) showed
    the fallback (before this fix) choosing pimpleFoam, since it had no
    icoFoam branch at all; icoFoam is OpenFOAM's own bundled tutorial
    solver for exactly this case type.
    """
    flow = FlowDescription(
        reynolds_number=1000,
        geometry="Lid-driven cavity flow",
        fluid="water",
        is_compressible=False,
        is_steady=False,
    )
    config = SolverConfiguration(
        solver_name=SolverName.PIMPLE_FOAM,
        turbulence_model=TurbulenceModel.LAMINAR,
        is_compressible=False,
        is_steady=False,
        simulation_type=SimulationType.LAMINAR,
        justification="(deliberately non-canonical, for testing)",
    )
    files = generate_openfoam_files(config, flow)
    result = validate_physics(flow, config, files)

    assert not result.passed
    finding = next(f for f in result.findings if f.rule == "solver_domain_consistency")
    assert finding.severity == ValidationSeverity.ERROR
    assert "icoFoam" in finding.message


def test_validate_physics_allows_icofoam_solver_for_cavity_case():
    """icoFoam for the same canonical lid-driven-cavity case passes cleanly."""
    flow = FlowDescription(
        reynolds_number=1000,
        geometry="Lid-driven cavity flow",
        fluid="water",
        is_compressible=False,
        is_steady=False,
    )
    config = SolverConfiguration(
        solver_name=SolverName.ICO_FOAM,
        turbulence_model=TurbulenceModel.LAMINAR,
        is_compressible=False,
        is_steady=False,
        simulation_type=SimulationType.LAMINAR,
        justification="Canonical lid-driven cavity case.",
    )
    files = generate_openfoam_files(config, flow)
    result = validate_physics(flow, config, files)

    finding = next(f for f in result.findings if f.rule == "solver_domain_consistency")
    assert finding.severity == ValidationSeverity.PASS


def test_prefers_spalart_allmaras_true_for_low_speed_airfoil():
    """A low-speed, incompressible NACA0012 airfoil prefers SpalartAllmaras."""
    flow = FlowDescription(
        reynolds_number=3000000,
        geometry="External aerodynamics of a NACA0012 airfoil",
        fluid="air",
        is_compressible=False,
        is_steady=True,
    )
    assert _prefers_spalart_allmaras(flow) is True


def test_prefers_spalart_allmaras_false_for_transonic_airfoil():
    """A transonic/compressible airfoil does not prefer SpalartAllmaras.

    Regression test: evaluation suite case tc15 ("Transonic flow over a
    supercritical airfoil at Mach 0.78... shock/boundary-layer interaction")
    expects kOmegaSST, not SpalartAllmaras: a blanket "any airfoil ->
    SpalartAllmaras" rule would get this wrong.
    """
    flow = FlowDescription(
        reynolds_number=5000000,
        mach_number=0.78,
        geometry="Transonic flow over a supercritical airfoil",
        fluid="air",
        is_compressible=True,
        is_steady=True,
    )
    assert _prefers_spalart_allmaras(flow) is False


def test_prefers_spalart_allmaras_false_for_turbine_blade():
    """A turbine blade (external aero, but not airfoil/wing/NACA) does not prefer SpalartAllmaras.

    Regression test: evaluation suite case tc20 ("Turbulent external flow
    over a wind turbine blade section") expects kOmegaSST, not
    SpalartAllmaras: external-aerodynamics keyword matching alone
    (is_external_aerodynamics_geometry) is too broad for this preference;
    it must be scoped to airfoil/wing/NACA specifically.
    """
    flow = FlowDescription(
        reynolds_number=1000000,
        geometry="Turbulent external flow over a wind turbine blade section",
        fluid="air",
        is_compressible=False,
        is_steady=True,
    )
    assert is_external_aerodynamics_geometry(flow.geometry) is True
    assert _prefers_spalart_allmaras(flow) is False


def test_check_turbulence_model_preference_flags_komega_for_low_speed_airfoil():
    """kOmegaSST for a low-speed NACA0012 airfoil is flagged in favor of SpalartAllmaras."""
    flow = FlowDescription(
        reynolds_number=3000000,
        geometry="External aerodynamics of a NACA0012 airfoil",
        fluid="air",
        is_compressible=False,
        is_steady=True,
    )
    config = SolverConfiguration(
        solver_name=SolverName.SIMPLE_FOAM,
        turbulence_model=TurbulenceModel.K_OMEGA_SST,
        is_compressible=False,
        is_steady=True,
        simulation_type=SimulationType.RAS,
        justification="(deliberately non-standard, for testing)",
    )
    finding = _check_turbulence_model_preference(flow, config)
    assert finding.severity == ValidationSeverity.ERROR
    assert finding.rule == "turbulence_model_preference"


def test_check_turbulence_model_preference_passes_spalart_for_low_speed_airfoil():
    """SpalartAllmaras for the same low-speed airfoil case passes cleanly."""
    flow = FlowDescription(
        reynolds_number=3000000,
        geometry="External aerodynamics of a NACA0012 airfoil",
        fluid="air",
        is_compressible=False,
        is_steady=True,
    )
    config = SolverConfiguration(
        solver_name=SolverName.SIMPLE_FOAM,
        turbulence_model=TurbulenceModel.SPALART_ALLMARAS,
        is_compressible=False,
        is_steady=True,
        simulation_type=SimulationType.RAS,
        justification="Standard choice for attached-flow external aerodynamics.",
    )
    finding = _check_turbulence_model_preference(flow, config)
    assert finding.severity == ValidationSeverity.PASS


def test_check_turbulence_model_preference_passes_komega_for_transonic_airfoil():
    """kOmegaSST for a transonic airfoil passes: SpalartAllmaras is not required there."""
    flow = FlowDescription(
        reynolds_number=5000000,
        mach_number=0.78,
        geometry="Transonic flow over a supercritical airfoil",
        fluid="air",
        is_compressible=True,
        is_steady=True,
    )
    config = SolverConfiguration(
        solver_name=SolverName.RHO_SIMPLE_FOAM,
        turbulence_model=TurbulenceModel.K_OMEGA_SST,
        is_compressible=True,
        is_steady=True,
        simulation_type=SimulationType.RAS,
        justification="Transonic shock/boundary-layer interaction.",
    )
    finding = _check_turbulence_model_preference(flow, config)
    assert finding.severity == ValidationSeverity.PASS


def test_check_turbulence_model_preference_passes_komega_for_turbine_blade():
    """kOmegaSST for a turbine blade passes: SpalartAllmaras is not required there."""
    flow = FlowDescription(
        reynolds_number=1000000,
        geometry="Turbulent external flow over a wind turbine blade section",
        fluid="air",
        is_compressible=False,
        is_steady=True,
    )
    config = SolverConfiguration(
        solver_name=SolverName.SIMPLE_FOAM,
        turbulence_model=TurbulenceModel.K_OMEGA_SST,
        is_compressible=False,
        is_steady=True,
        simulation_type=SimulationType.RAS,
        justification="Separation-prone rotating machinery flow.",
    )
    finding = _check_turbulence_model_preference(flow, config)
    assert finding.severity == ValidationSeverity.PASS


def test_check_turbulence_model_preference_not_applicable_in_laminar_regime():
    """The preference check does not apply below the laminar/turbulent threshold."""
    flow = FlowDescription(
        reynolds_number=1000,
        geometry="External aerodynamics of a NACA0012 airfoil",
        fluid="air",
        is_compressible=False,
        is_steady=True,
    )
    config = SolverConfiguration(
        solver_name=SolverName.SIMPLE_FOAM,
        turbulence_model=TurbulenceModel.LAMINAR,
        is_compressible=False,
        is_steady=True,
        simulation_type=SimulationType.LAMINAR,
        justification="Laminar regime.",
    )
    finding = _check_turbulence_model_preference(flow, config)
    assert finding.severity == ValidationSeverity.PASS


def test_prefers_free_shear_kepsilon_true_for_mixing_junction():
    """A T-junction stream-mixing flow prefers kEpsilon.

    Mirrors evaluation suite case tc09 ("Turbulent mixing of two air
    streams... merging in a T-junction"), which expects kEpsilon.
    """
    flow = FlowDescription(
        reynolds_number=20000,
        geometry="Turbulent mixing of two air streams merging in a T-junction",
        fluid="air",
        is_compressible=False,
        is_steady=False,
    )
    assert _prefers_free_shear_kepsilon(flow) is True


def test_prefers_free_shear_kepsilon_true_for_multiphase():
    """A multiphase free-surface flow prefers kEpsilon (OpenFOAM's own damBreak default)."""
    flow = FlowDescription(
        reynolds_number=100000,
        geometry="Dam-break free-surface flow into an air-filled channel",
        fluid="water",
        multiphase=True,
        is_compressible=False,
        is_steady=False,
    )
    assert _prefers_free_shear_kepsilon(flow) is True


def test_prefers_free_shear_kepsilon_false_for_pipe():
    """Ordinary wall-bounded pipe flow does not prefer kEpsilon."""
    flow = FlowDescription(
        reynolds_number=50000,
        geometry="circular pipe",
        fluid="air",
        is_compressible=False,
        is_steady=True,
    )
    assert _prefers_free_shear_kepsilon(flow) is False


def test_prefers_free_shear_kepsilon_false_for_cylinder_wake():
    """A near-body bluff-body wake does not prefer kEpsilon.

    Mirrors evaluation suite case tc03 ("flow past a circular cylinder...
    vortex shedding"), which expects kOmegaSST, not kEpsilon.
    """
    flow = FlowDescription(
        reynolds_number=1000,
        geometry="flow past a circular cylinder",
        fluid="air",
        is_compressible=False,
        is_steady=False,
    )
    assert _prefers_free_shear_kepsilon(flow) is False


def test_check_turbulence_model_preference_flags_kepsilon_for_pipe_flow():
    """kEpsilon for ordinary turbulent pipe flow is flagged in favor of kOmegaSST.

    Regression test: shortening SOLVER_SELECTION_PROMPT removed the prompt's
    soft guidance away from kEpsilon for wall-bounded flow, which must now
    be enforced by the deterministic rule layer instead.
    """
    flow = FlowDescription(
        reynolds_number=50000,
        geometry="circular pipe",
        fluid="air",
        is_compressible=False,
        is_steady=True,
    )
    config = SolverConfiguration(
        solver_name=SolverName.SIMPLE_FOAM,
        turbulence_model=TurbulenceModel.K_EPSILON,
        is_compressible=False,
        is_steady=True,
        simulation_type=SimulationType.RAS,
        justification="(non-standard for this case type, for testing)",
    )
    finding = _check_turbulence_model_preference(flow, config)
    assert finding.severity == ValidationSeverity.ERROR
    assert finding.rule == "turbulence_model_preference"


def test_check_turbulence_model_preference_passes_kepsilon_for_mixing_junction():
    """kEpsilon for a genuinely free-shear mixing flow passes cleanly."""
    flow = FlowDescription(
        reynolds_number=20000,
        geometry="Turbulent mixing of two air streams merging in a T-junction",
        fluid="air",
        is_compressible=False,
        is_steady=False,
    )
    config = SolverConfiguration(
        solver_name=SolverName.PIMPLE_FOAM,
        turbulence_model=TurbulenceModel.K_EPSILON,
        is_compressible=False,
        is_steady=False,
        simulation_type=SimulationType.RAS,
        justification="Free-shear mixing flow.",
    )
    finding = _check_turbulence_model_preference(flow, config)
    assert finding.severity == ValidationSeverity.PASS


def test_check_turbulence_model_preference_passes_kepsilon_for_multiphase():
    """kEpsilon for a multiphase free-surface flow passes cleanly."""
    flow = FlowDescription(
        reynolds_number=100000,
        geometry="Dam-break free-surface flow into an air-filled channel",
        fluid="water",
        multiphase=True,
        is_compressible=False,
        is_steady=False,
    )
    config = SolverConfiguration(
        solver_name=SolverName.INTER_FOAM,
        turbulence_model=TurbulenceModel.K_EPSILON,
        is_compressible=False,
        is_steady=False,
        simulation_type=SimulationType.RAS,
        justification="Standard damBreak-style configuration.",
    )
    finding = _check_turbulence_model_preference(flow, config)
    assert finding.severity == ValidationSeverity.PASS


def test_check_turbulence_model_preference_passes_komega_for_pipe_flow():
    """kOmegaSST for ordinary turbulent pipe flow still passes (unaffected by the new rule)."""
    flow = FlowDescription(
        reynolds_number=50000,
        geometry="circular pipe",
        fluid="air",
        is_compressible=False,
        is_steady=True,
    )
    config = SolverConfiguration(
        solver_name=SolverName.SIMPLE_FOAM,
        turbulence_model=TurbulenceModel.K_OMEGA_SST,
        is_compressible=False,
        is_steady=True,
        simulation_type=SimulationType.RAS,
        justification="Standard wall-bounded internal flow.",
    )
    finding = _check_turbulence_model_preference(flow, config)
    assert finding.severity == ValidationSeverity.PASS
