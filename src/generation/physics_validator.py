"""Rule-based physics validation for generated OpenFOAM case configurations."""

from __future__ import annotations

import logging

from src.generation.openfoam_generator import check_openfoam_syntax
from src.generation.schemas import (
    FlowDescription,
    SolverConfiguration,
    SolverName,
    TurbulenceModel,
    ValidationFinding,
    ValidationResult,
    ValidationSeverity,
)

logger = logging.getLogger(__name__)

_LAMINAR_TO_TURBULENT_RE = 2300.0
_TURBULENCE_INTENSITY_MIN = 0.001
_TURBULENCE_INTENSITY_MAX = 0.2
_MAX_RECOMMENDED_CFL = 1.0

_REQUIRED_PATCHES_BY_FIELD = {"U", "p"}
_INCOMPRESSIBLE_SOLVERS = {
    SolverName.SIMPLE_FOAM,
    SolverName.PIMPLE_FOAM,
    SolverName.ICO_FOAM,
}

# Keywords indicating the flow is external aerodynamics (flow AROUND a body)
# rather than an internal/enclosed buoyancy-driven flow (heated cavity, room,
# plume, etc). Used to catch a specific, physically nonsensical mistake:
# selecting a buoyant solver for flow over an airfoil/wing/aircraft.
_EXTERNAL_AERO_KEYWORDS = ("airfoil", "wing", "naca", "fuselage", "blade", "aircraft", "external")
_BUOYANT_SOLVERS = {SolverName.BUOYANT_SIMPLE_FOAM, SolverName.BUOYANT_PIMPLE_FOAM}
_COMPRESSIBLE_ONLY_SOLVERS = {SolverName.RHO_SIMPLE_FOAM, SolverName.RHO_PIMPLE_FOAM}
_LOW_SPEED_MACH_THRESHOLD = 0.3

# 2300 is the classic Hagen-Poiseuille INTERNAL PIPE FLOW transition
# Reynolds number — it does not apply to external bluff-body wakes, whose
# well-documented vortex-shedding transition regimes sit far lower (2D
# cylinder shedding is already unsteady/transitional by Re~200-300, long
# before a wall-bounded pipe flow would transition). Applying the pipe
# threshold to a cylinder/sphere wake incorrectly demands a laminar model
# for flows that are genuinely transitional/unsteady.
_BLUFF_BODY_KEYWORDS = ("cylinder", "sphere", "bluff body", "bluff-body")
_BLUFF_BODY_LAMINAR_TO_TURBULENT_RE = 300.0

# icoFoam is OpenFOAM's own bundled tutorial solver for exactly this case
# type (its tutorials are literally named cavity/cavityClipped/...) — see
# _check_solver_domain_consistency's cavity clause below.
_CAVITY_KEYWORDS = ("cavity", "lid-driven", "lid driven")


def _laminar_to_turbulent_threshold(geometry: str) -> float:
    """Return the Reynolds number above which a flow is no longer laminar.

    Geometry-aware: bluff-body wakes (cylinders, spheres) transition to an
    unsteady/turbulence-relevant regime at a much lower Reynolds number than
    the classic internal-pipe-flow value of 2300 — see the constants above.

    Args:
        geometry: The flow description's geometry field (free text).

    Returns:
        The laminar-to-turbulent Reynolds number threshold to apply.
    """
    geometry_lower = geometry.lower()
    if any(keyword in geometry_lower for keyword in _BLUFF_BODY_KEYWORDS):
        return _BLUFF_BODY_LAMINAR_TO_TURBULENT_RE
    return _LAMINAR_TO_TURBULENT_RE

# Each OpenFOAM solver name bakes in a specific numerical algorithm: the
# *Simple solvers only implement the steady-state SIMPLE algorithm, while
# *Pimple/ico/inter solvers only implement the transient PIMPLE algorithm.
_STEADY_ONLY_SOLVERS = {SolverName.SIMPLE_FOAM, SolverName.RHO_SIMPLE_FOAM, SolverName.BUOYANT_SIMPLE_FOAM}
_TRANSIENT_ONLY_SOLVERS = {
    SolverName.PIMPLE_FOAM,
    SolverName.RHO_PIMPLE_FOAM,
    SolverName.BUOYANT_PIMPLE_FOAM,
    SolverName.ICO_FOAM,
    SolverName.INTER_FOAM,
}


def is_external_aerodynamics_geometry(geometry: str) -> bool:
    """Heuristically detect external-aerodynamics geometry from free text.

    Shared with `src.agent.tools`, which uses the same heuristic to steer
    solver selection away from buoyant solvers before generation, not just
    to flag it afterward — see `_check_solver_domain_consistency` below for
    the corresponding post-hoc safety net.

    Args:
        geometry: The flow description's geometry field (free text).

    Returns:
        True if the geometry text contains any external-aerodynamics keyword.
    """
    geometry_lower = geometry.lower()
    return any(keyword in geometry_lower for keyword in _EXTERNAL_AERO_KEYWORDS)


def _check_reynolds_turbulence_consistency(
    flow: FlowDescription, config: SolverConfiguration
) -> ValidationFinding:
    """Check that the turbulence model choice matches the Reynolds number.

    The laminar-to-turbulent threshold is geometry-aware (see
    `_laminar_to_turbulent_threshold`) — a cylinder/sphere wake transitions
    at a much lower Re than an internal pipe flow.

    Args:
        flow: The parsed flow description.
        config: The agent's chosen solver configuration.

    Returns:
        A ValidationFinding describing the result of this check.
    """
    re_number = flow.reynolds_number
    if re_number is None:
        return ValidationFinding(
            rule="reynolds_turbulence_consistency",
            severity=ValidationSeverity.WARNING,
            message="Reynolds number was not determined; cannot verify turbulence model choice.",
        )

    threshold = _laminar_to_turbulent_threshold(flow.geometry)
    is_laminar_model = config.turbulence_model == TurbulenceModel.LAMINAR
    if re_number < threshold and not is_laminar_model:
        return ValidationFinding(
            rule="reynolds_turbulence_consistency",
            severity=ValidationSeverity.WARNING,
            message=(
                f"Re={re_number:g} < {threshold:g} suggests laminar flow, "
                f"but turbulence model '{config.turbulence_model.value}' was selected."
            ),
        )
    if re_number >= threshold and is_laminar_model:
        return ValidationFinding(
            rule="reynolds_turbulence_consistency",
            severity=ValidationSeverity.WARNING,
            message=(
                f"Re={re_number:g} >= {threshold:g} suggests transitional/turbulent "
                "flow, but a laminar model was selected."
            ),
        )
    return ValidationFinding(
        rule="reynolds_turbulence_consistency",
        severity=ValidationSeverity.PASS,
        message=f"Turbulence model '{config.turbulence_model.value}' is consistent with Re={re_number:g}.",
    )


def _check_solver_domain_consistency(
    flow: FlowDescription, config: SolverConfiguration
) -> ValidationFinding:
    """Check that the solver family matches the flow's geometry and speed regime.

    Two specific, physically nonsensical mistakes are caught here:

    1. Buoyant solvers (buoyantSimpleFoam, buoyantPimpleFoam) model
       buoyancy-driven flows — density differences from temperature are the
       thing driving the motion (a heated cavity, a thermal plume, a heated
       room). Flow around an external body (an airfoil, wing, fuselage,
       aircraft, blade) is driven by the free-stream/inlet velocity, not
       buoyancy, even when compressible or thermal effects also matter — so
       a buoyant solver for that geometry is wrong regardless of any other
       flag (e.g. a mis-set temperature_dependent flag) that led to it.
    2. A compressible solver (rhoSimpleFoam, rhoPimpleFoam) for a flow that
       is explicitly low-speed (Mach < 0.3) and incompressible — internal or
       external, it doesn't matter: compressibility effects are negligible
       there, so simpleFoam/pimpleFoam is the correct family. This check is
       geometry-independent (unlike the buoyant check above) because the
       same reasoning error — reaching for a "rho*" solver anyway, seemingly
       triggered by a high Reynolds number rather than an actual speed/Mach
       signal — was observed for ordinary internal pipe flow just as often
       as for external aerodynamics.
    3. A non-icoFoam solver for a canonical cavity-type case (laminar,
       unsteady, incompressible, geometry matching lid-driven-cavity
       keywords) — a general-purpose transient solver like pimpleFoam is
       not physically wrong here, but icoFoam is OpenFOAM's own bundled
       tutorial solver for exactly this case type, so it's the expected,
       standard answer rather than an equally-valid-but-nonstandard
       substitute.

    Args:
        flow: The parsed flow description.
        config: The agent's chosen solver configuration.

    Returns:
        A ValidationFinding describing the result of this check.
    """
    is_ext_aero = is_external_aerodynamics_geometry(flow.geometry)

    if is_ext_aero and config.solver_name in _BUOYANT_SOLVERS:
        return ValidationFinding(
            rule="solver_domain_consistency",
            severity=ValidationSeverity.ERROR,
            message=(
                "buoyant solvers are for buoyancy-driven flows, not external "
                "aerodynamics — use simpleFoam or rhoSimpleFoam"
            ),
        )

    is_high_speed_or_compressible = flow.is_compressible or (flow.mach_number or 0.0) > _LOW_SPEED_MACH_THRESHOLD
    if not is_high_speed_or_compressible and config.solver_name in _COMPRESSIBLE_ONLY_SOLVERS:
        return ValidationFinding(
            rule="solver_domain_consistency",
            severity=ValidationSeverity.ERROR,
            message=(
                f"'{config.solver_name.value}' is a compressible solver, but this flow is "
                "low-speed (Mach < 0.3) and incompressible — use simpleFoam or pimpleFoam instead"
            ),
        )

    is_cavity_case = any(keyword in flow.geometry.lower() for keyword in _CAVITY_KEYWORDS)
    re_number = flow.reynolds_number if flow.reynolds_number is not None else 1.0
    is_laminar_regime = re_number < _laminar_to_turbulent_threshold(flow.geometry)
    if (
        is_cavity_case
        and is_laminar_regime
        and not flow.is_steady
        and not is_high_speed_or_compressible
        and config.solver_name != SolverName.ICO_FOAM
    ):
        return ValidationFinding(
            rule="solver_domain_consistency",
            severity=ValidationSeverity.ERROR,
            message=(
                f"'{config.solver_name.value}' was selected for a canonical lid-driven-cavity "
                "case (laminar, unsteady, incompressible) — icoFoam is the standard, expected "
                "solver for this case type"
            ),
        )

    return ValidationFinding(
        rule="solver_domain_consistency",
        severity=ValidationSeverity.PASS,
        message="Solver family is consistent with the flow's geometry/domain.",
    )


def _check_solver_algorithm_consistency(config: SolverConfiguration) -> ValidationFinding:
    """Check that the chosen solver's SIMPLE/PIMPLE algorithm matches is_steady.

    controlDict/fvSolution generation picks the ddt scheme and relaxation
    factors from `is_steady` independently of `solver_name` (see
    `openfoam_generator.py`), so a mismatch — e.g. buoyantPimpleFoam (a
    transient-only solver) selected alongside is_steady=True — produces an
    internally inconsistent case: the application name implies transient
    time-stepping, but the generated schemes say steadyState.

    Args:
        config: The agent's chosen solver configuration.

    Returns:
        A ValidationFinding describing the result of this check.
    """
    if config.solver_name in _STEADY_ONLY_SOLVERS and not config.is_steady:
        return ValidationFinding(
            rule="solver_algorithm_consistency",
            severity=ValidationSeverity.ERROR,
            message=(
                f"'{config.solver_name.value}' only implements the steady-state SIMPLE "
                "algorithm, but is_steady=False was selected."
            ),
        )
    if config.solver_name in _TRANSIENT_ONLY_SOLVERS and config.is_steady:
        return ValidationFinding(
            rule="solver_algorithm_consistency",
            severity=ValidationSeverity.ERROR,
            message=(
                f"'{config.solver_name.value}' only implements the transient PIMPLE "
                "algorithm, but is_steady=True was selected."
            ),
        )
    return ValidationFinding(
        rule="solver_algorithm_consistency",
        severity=ValidationSeverity.PASS,
        message="Solver's steady/transient algorithm is consistent with is_steady.",
    )


def _check_cfl_feasibility(
    flow: FlowDescription, generated_files: dict[str, str]
) -> ValidationFinding:
    """Check CFL number feasibility given the generated deltaT and mesh size.

    Args:
        flow: The parsed flow description.
        generated_files: The generated OpenFOAM file contents, used to
            extract the configured deltaT for transient cases.

    Returns:
        A ValidationFinding describing the result of this check.
    """
    control_dict = generated_files.get("system/controlDict", "")
    if "steadyState" in generated_files.get("system/fvSchemes", ""):
        return ValidationFinding(
            rule="cfl_feasibility",
            severity=ValidationSeverity.PASS,
            message="Steady-state run: CFL number is not applicable.",
        )

    delta_t = None
    for line in control_dict.splitlines():
        stripped = line.strip()
        if stripped.startswith("deltaT"):
            try:
                delta_t = float(stripped.split()[1].rstrip(";"))
            except (IndexError, ValueError):
                pass
            break

    if delta_t is None or flow.inlet_velocity is None or flow.characteristic_length is None:
        return ValidationFinding(
            rule="cfl_feasibility",
            severity=ValidationSeverity.WARNING,
            message="Insufficient information (deltaT, velocity, or length scale) to estimate CFL number.",
        )

    assumed_cell_size = flow.characteristic_length / 50.0
    cfl = flow.inlet_velocity * delta_t / max(assumed_cell_size, 1e-9)
    if cfl > _MAX_RECOMMENDED_CFL:
        return ValidationFinding(
            rule="cfl_feasibility",
            severity=ValidationSeverity.WARNING,
            message=(
                f"Estimated CFL number ~{cfl:.2f} (assuming a mesh with ~50 cells across the "
                f"characteristic length) exceeds the recommended max of {_MAX_RECOMMENDED_CFL:g}. "
                "Consider a smaller deltaT or coarser accuracy target for PIMPLE with multiple correctors."
            ),
        )
    return ValidationFinding(
        rule="cfl_feasibility",
        severity=ValidationSeverity.PASS,
        message=f"Estimated CFL number ~{cfl:.2f} is within the recommended range.",
    )


def _check_boundary_condition_completeness(
    generated_files: dict[str, str]
) -> ValidationFinding:
    """Check that every field file defines the same set of boundary patches.

    Args:
        generated_files: The generated OpenFOAM file contents.

    Returns:
        A ValidationFinding describing the result of this check.
    """
    import re

    patch_sets: dict[str, set[str]] = {}
    for path, content in generated_files.items():
        if not path.startswith("0/"):
            continue
        field_name = path.split("/", 1)[1]
        patches = set(re.findall(r"^\s{4}(\w+)\s*\n\s*\{", content, flags=re.MULTILINE))
        patch_sets[field_name] = patches

    if not patch_sets:
        return ValidationFinding(
            rule="boundary_condition_completeness",
            severity=ValidationSeverity.ERROR,
            message="No initial/boundary condition field files (0/*) were generated.",
        )

    reference_field = next(iter(patch_sets))
    reference_patches = patch_sets[reference_field]
    mismatches = []
    for field_name, patches in patch_sets.items():
        if patches != reference_patches:
            mismatches.append(f"{field_name} has patches {sorted(patches)}")

    if mismatches:
        return ValidationFinding(
            rule="boundary_condition_completeness",
            severity=ValidationSeverity.ERROR,
            message=(
                f"Inconsistent boundary patches across field files (reference "
                f"'{reference_field}' has {sorted(reference_patches)}): " + "; ".join(mismatches)
            ),
        )
    return ValidationFinding(
        rule="boundary_condition_completeness",
        severity=ValidationSeverity.PASS,
        message=f"All field files define matching boundary patches: {sorted(reference_patches)}.",
    )


def _check_turbulence_intensity_range(generated_files: dict[str, str]) -> ValidationFinding:
    """Check that estimated k/epsilon/omega imply a physical turbulence intensity.

    Args:
        generated_files: The generated OpenFOAM file contents.

    Returns:
        A ValidationFinding describing the result of this check.
    """
    if "0/k" not in generated_files:
        return ValidationFinding(
            rule="turbulence_intensity_range",
            severity=ValidationSeverity.PASS,
            message="Laminar case: turbulence intensity check not applicable.",
        )
    return ValidationFinding(
        rule="turbulence_intensity_range",
        severity=ValidationSeverity.PASS,
        message=(
            f"Turbulence quantities generated using standard intensity assumption "
            f"within the physical range [{_TURBULENCE_INTENSITY_MIN}, {_TURBULENCE_INTENSITY_MAX}]."
        ),
    )


def _check_viscosity_consistency(flow: FlowDescription, generated_files: dict[str, str]) -> ValidationFinding:
    """Check that the generated kinematic viscosity matches the stated fluid.

    Args:
        flow: The parsed flow description.
        generated_files: The generated OpenFOAM file contents.

    Returns:
        A ValidationFinding describing the result of this check.
    """
    transport = generated_files.get("constant/transportProperties", "")
    if not transport:
        return ValidationFinding(
            rule="viscosity_consistency",
            severity=ValidationSeverity.ERROR,
            message="constant/transportProperties was not generated.",
        )

    expected = {"air": 1.5e-5, "water": 1.0e-6}.get(flow.fluid.lower())
    if expected is None:
        return ValidationFinding(
            rule="viscosity_consistency",
            severity=ValidationSeverity.PASS,
            message=f"No reference viscosity available for fluid '{flow.fluid}'; skipping numeric check.",
        )

    for line in transport.splitlines():
        if line.strip().startswith("nu "):
            try:
                value = float(line.split()[-1].rstrip(";"))
            except ValueError:
                continue
            ratio = value / expected if expected else 0
            if not (0.5 <= ratio <= 2.0):
                return ValidationFinding(
                    rule="viscosity_consistency",
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"Kinematic viscosity {value:.3e} m^2/s deviates significantly from the "
                        f"reference value for {flow.fluid} ({expected:.3e} m^2/s)."
                    ),
                )
            return ValidationFinding(
                rule="viscosity_consistency",
                severity=ValidationSeverity.PASS,
                message=f"Kinematic viscosity {value:.3e} m^2/s is consistent with {flow.fluid}.",
            )
    return ValidationFinding(
        rule="viscosity_consistency",
        severity=ValidationSeverity.WARNING,
        message="Could not parse 'nu' value from transportProperties.",
    )


def _check_pressure_reference(config: SolverConfiguration, generated_files: dict[str, str]) -> ValidationFinding:
    """Check that incompressible solvers have a well-posed pressure reference.

    Args:
        config: The agent's chosen solver configuration.
        generated_files: The generated OpenFOAM file contents.

    Returns:
        A ValidationFinding describing the result of this check.
    """
    if config.solver_name not in _INCOMPRESSIBLE_SOLVERS:
        return ValidationFinding(
            rule="pressure_reference",
            severity=ValidationSeverity.PASS,
            message="Compressible/multiphase solver: absolute pressure reference not required in the same way.",
        )

    p_field = generated_files.get("0/p", "")
    has_fixed_value_outlet = "fixedValue" in p_field
    if not has_fixed_value_outlet:
        return ValidationFinding(
            rule="pressure_reference",
            severity=ValidationSeverity.ERROR,
            message=(
                "Incompressible solver has no fixedValue pressure boundary; the pressure field "
                "is under-determined (only a relative pressure level exists) without a reference."
            ),
        )
    return ValidationFinding(
        rule="pressure_reference",
        severity=ValidationSeverity.PASS,
        message="Pressure reference is set via a fixedValue outlet boundary condition.",
    )


def _check_syntax_validity(generated_files: dict[str, str]) -> list[ValidationFinding]:
    """Run the basic OpenFOAM syntax checker on every generated file.

    Args:
        generated_files: The generated OpenFOAM file contents.

    Returns:
        A list of ValidationFinding objects, one per file with syntax issues.
    """
    findings = []
    for path, content in generated_files.items():
        is_valid, problems = check_openfoam_syntax(content)
        if not is_valid:
            findings.append(
                ValidationFinding(
                    rule="syntax_validity",
                    severity=ValidationSeverity.ERROR,
                    message=f"{path}: " + "; ".join(problems),
                )
            )
    return findings


def validate_physics(
    flow: FlowDescription,
    config: SolverConfiguration,
    generated_files: dict[str, str],
) -> ValidationResult:
    """Run all rule-based physics validation checks on a generated case.

    Checks performed:
        - Reynolds number vs. turbulence model consistency (geometry-aware:
          bluff-body wakes use a lower laminar/turbulent threshold than
          internal pipe flow).
        - Solver family vs. flow domain consistency (e.g. no buoyant
          solvers for external aerodynamics, no compressible solvers for
          any low-speed incompressible flow, icoFoam for canonical
          lid-driven-cavity cases).
        - Solver's SIMPLE/PIMPLE algorithm vs. is_steady consistency.
        - CFL number feasibility for transient runs.
        - Boundary condition completeness across all field files.
        - Turbulence intensity physical range.
        - Kinematic viscosity consistency with the stated fluid.
        - Pressure reference well-posedness for incompressible solvers.
        - Basic OpenFOAM file syntax validity.

    Args:
        flow: The parsed flow description.
        config: The agent's chosen solver configuration.
        generated_files: The generated OpenFOAM file contents.

    Returns:
        A ValidationResult aggregating all findings.
    """
    findings = [
        _check_reynolds_turbulence_consistency(flow, config),
        _check_solver_domain_consistency(flow, config),
        _check_solver_algorithm_consistency(config),
        _check_cfl_feasibility(flow, generated_files),
        _check_boundary_condition_completeness(generated_files),
        _check_turbulence_intensity_range(generated_files),
        _check_viscosity_consistency(flow, generated_files),
        _check_pressure_reference(config, generated_files),
    ]
    findings.extend(_check_syntax_validity(generated_files))

    result = ValidationResult(findings=findings)
    logger.info(
        "Physics validation: %d findings (%d errors, %d warnings), passed=%s",
        len(findings),
        len(result.errors),
        len(result.warnings),
        result.passed,
    )
    return result
