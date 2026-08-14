"""Generates a complete, ready-to-run OpenFOAM case from structured inputs."""

from __future__ import annotations

import logging
import re

from src.generation.schemas import (
    BoundaryCondition,
    ControlDictConfig,
    FieldFile,
    FieldSolverSettings,
    FlowDescription,
    FvSchemesConfig,
    FvSolutionConfig,
    SolverConfiguration,
    TransportProperties,
    TurbulenceModel,
    TurbulenceProperties,
)

logger = logging.getLogger(__name__)

_DEFAULT_KINEMATIC_VISCOSITY = {
    "air": 1.5e-5,
    "water": 1.0e-6,
}
_TURBULENT_INTENSITY_DEFAULT = 0.05


def _estimate_kinematic_viscosity(flow: FlowDescription) -> float:
    """Estimate kinematic viscosity from the fluid name or explicit value.

    Args:
        flow: The parsed flow description.

    Returns:
        Kinematic viscosity in m^2/s.
    """
    if flow.kinematic_viscosity:
        return flow.kinematic_viscosity
    return _DEFAULT_KINEMATIC_VISCOSITY.get(flow.fluid.lower(), 1.5e-5)


def _estimate_turbulence_quantities(
    flow: FlowDescription, nu: float
) -> dict[str, float]:
    """Estimate initial k, epsilon, and omega from bulk flow quantities.

    Uses standard engineering correlations: turbulence intensity I ~ 5%,
    a turbulent length scale l ~ 0.07 * characteristic length, giving
    k = 1.5*(U*I)^2 and epsilon = C_mu^0.75 * k^1.5 / l,
    omega = epsilon / (C_mu * k).

    Args:
        flow: The parsed flow description.
        nu: Kinematic viscosity in m^2/s, used as a fallback to derive a
            characteristic velocity if none was extracted.

    Returns:
        A dict with keys: k, epsilon, omega.
    """
    velocity = flow.inlet_velocity or 10.0
    length = flow.characteristic_length or 1.0
    intensity = _TURBULENT_INTENSITY_DEFAULT
    length_scale = 0.07 * length
    c_mu = 0.09

    k = 1.5 * (velocity * intensity) ** 2
    epsilon = (c_mu**0.75) * (k**1.5) / max(length_scale, 1e-6)
    omega = epsilon / (c_mu * max(k, 1e-8))
    return {"k": max(k, 1e-6), "epsilon": max(epsilon, 1e-6), "omega": max(omega, 1e-3)}


def _build_control_dict(config: SolverConfiguration, flow: FlowDescription) -> ControlDictConfig:
    """Build a controlDict configuration with realistic time settings.

    Args:
        config: The agent's solver/turbulence selection.
        flow: The parsed flow description.

    Returns:
        A populated ControlDictConfig.
    """
    if config.is_steady:
        return ControlDictConfig(
            solver=config.solver_name,
            startTime=0,
            endTime=1000,
            deltaT=1,
            writeInterval=100,
            writeFormat="ascii",
        )

    velocity = flow.inlet_velocity or 10.0
    length = flow.characteristic_length or 1.0
    flow_through_time = length / velocity if velocity > 0 else 1.0
    end_time = round(flow_through_time * 20, 6)
    delta_t = round(flow_through_time * 0.01, 6)
    write_interval = round(end_time / 20, 6)
    return ControlDictConfig(
        solver=config.solver_name,
        startTime=0,
        endTime=max(end_time, delta_t * 10),
        deltaT=max(delta_t, 1e-6),
        writeInterval=max(write_interval, delta_t),
        writeFormat="ascii",
    )


def _build_fv_schemes(config: SolverConfiguration) -> FvSchemesConfig:
    """Build an fvSchemes configuration appropriate for the chosen solver.

    Args:
        config: The agent's solver/turbulence selection.

    Returns:
        A populated FvSchemesConfig.
    """
    ddt = "steadyState" if config.is_steady else "backward"
    div_schemes = {
        "default": "none",
        "div(phi,U)": "bounded Gauss linearUpwindV grad(U)"
        if not config.is_steady
        else "bounded Gauss upwind",
        "div((nuEff*dev2(T(grad(U)))))": "Gauss linear",
    }
    if config.turbulence_model != TurbulenceModel.LAMINAR:
        div_schemes["div(phi,k)"] = "bounded Gauss upwind"
        if config.turbulence_model == TurbulenceModel.K_OMEGA_SST:
            div_schemes["div(phi,omega)"] = "bounded Gauss upwind"
        else:
            div_schemes["div(phi,epsilon)"] = "bounded Gauss upwind"
    return FvSchemesConfig(ddtSchemes=ddt, divSchemes=div_schemes)


def _build_fv_solution(config: SolverConfiguration) -> FvSolutionConfig:
    """Build an fvSolution configuration with appropriate solvers/tolerances.

    Args:
        config: The agent's solver/turbulence selection.

    Returns:
        A populated FvSolutionConfig.
    """
    solvers = {
        "p": FieldSolverSettings(solver="GAMG", smoother="GaussSeidel", tolerance=1e-6, relTol=0.05),
        "U": FieldSolverSettings(solver="smoothSolver", smoother="GaussSeidel", tolerance=1e-8, relTol=0.1),
    }
    if config.turbulence_model != TurbulenceModel.LAMINAR:
        solvers["k"] = FieldSolverSettings(solver="smoothSolver", smoother="GaussSeidel", tolerance=1e-8, relTol=0.1)
        if config.turbulence_model == TurbulenceModel.K_OMEGA_SST:
            solvers["omega"] = FieldSolverSettings(solver="smoothSolver", smoother="GaussSeidel", tolerance=1e-8, relTol=0.1)
        else:
            solvers["epsilon"] = FieldSolverSettings(solver="smoothSolver", smoother="GaussSeidel", tolerance=1e-8, relTol=0.1)

    relax = {"p": 0.3, "U": 0.7} if config.is_steady else {"p": 1.0, "U": 1.0}
    if config.turbulence_model != TurbulenceModel.LAMINAR:
        relax["k"] = 0.7 if config.is_steady else 1.0
        if config.turbulence_model == TurbulenceModel.K_OMEGA_SST:
            relax["omega"] = 0.7 if config.is_steady else 1.0
        else:
            relax["epsilon"] = 0.7 if config.is_steady else 1.0

    return FvSolutionConfig(
        solvers=solvers,
        is_steady=config.is_steady,
        relaxation_factors=relax,
        n_correctors=2,
        n_non_orthogonal_correctors=1,
    )


def _build_u_field(config: SolverConfiguration, flow: FlowDescription) -> FieldFile:
    """Build the 0/U velocity field file.

    Args:
        config: The agent's solver/turbulence selection.
        flow: The parsed flow description.

    Returns:
        A populated FieldFile for the U field.
    """
    velocity = flow.inlet_velocity or 10.0
    boundary = {
        "inlet": BoundaryCondition(type="fixedValue", value=f"uniform ({velocity:g} 0 0)"),
        "outlet": BoundaryCondition(type="zeroGradient"),
        "walls": BoundaryCondition(type="noSlip"),
        "frontAndBack": BoundaryCondition(type="empty"),
    }
    return FieldFile(
        field_name="U",
        foam_class="volVectorField",
        dimensions="[0 1 -1 0 0 0 0]",
        internal_field="uniform (0 0 0)",
        boundary_field=boundary,
    )


def _build_p_field(config: SolverConfiguration) -> FieldFile:
    """Build the 0/p pressure field file.

    Args:
        config: The agent's solver/turbulence selection.

    Returns:
        A populated FieldFile for the p field.
    """
    boundary = {
        "inlet": BoundaryCondition(type="zeroGradient"),
        "outlet": BoundaryCondition(type="fixedValue", value="uniform 0"),
        "walls": BoundaryCondition(type="zeroGradient"),
        "frontAndBack": BoundaryCondition(type="empty"),
    }
    return FieldFile(
        field_name="p",
        foam_class="volScalarField",
        dimensions="[0 2 -2 0 0 0 0]",
        internal_field="uniform 0",
        boundary_field=boundary,
    )


def _build_turbulence_field(
    field_name: str, dimensions: str, internal_value: float
) -> FieldFile:
    """Build a 0/k, 0/epsilon, or 0/omega turbulence field file.

    Args:
        field_name: "k", "epsilon", or "omega".
        dimensions: The OpenFOAM dimension set string for this field.
        internal_value: The estimated uniform internal field value.

    Returns:
        A populated FieldFile for the given turbulence quantity.
    """
    wall_type = "kqRWallFunction" if field_name == "k" else (
        "omegaWallFunction" if field_name == "omega" else "epsilonWallFunction"
    )
    boundary = {
        "inlet": BoundaryCondition(type="fixedValue", value=f"uniform {internal_value:.6g}"),
        "outlet": BoundaryCondition(type="zeroGradient"),
        "walls": BoundaryCondition(type=wall_type, value=f"uniform {internal_value:.6g}"),
        "frontAndBack": BoundaryCondition(type="empty"),
    }
    return FieldFile(
        field_name=field_name,
        foam_class="volScalarField",
        dimensions=dimensions,
        internal_field=f"uniform {internal_value:.6g}",
        boundary_field=boundary,
    )


def generate_openfoam_files(
    config: SolverConfiguration, flow: FlowDescription
) -> dict[str, str]:
    """Generate the full set of OpenFOAM case files for the given selection.

    Args:
        config: The agent's chosen solver, turbulence model, and rationale.
        flow: The parsed flow description driving numerical values.

    Returns:
        A dict mapping relative OpenFOAM file paths (e.g. "system/controlDict")
        to their full file content strings.
    """
    nu = _estimate_kinematic_viscosity(flow)
    files: dict[str, str] = {}

    files["system/controlDict"] = _build_control_dict(config, flow).to_openfoam_string()
    files["system/fvSchemes"] = _build_fv_schemes(config).to_openfoam_string()
    files["system/fvSolution"] = _build_fv_solution(config).to_openfoam_string()

    files["constant/transportProperties"] = TransportProperties(
        kinematic_viscosity=nu
    ).to_openfoam_string()

    files["constant/turbulenceProperties"] = TurbulenceProperties(
        simulation_type=config.simulation_type, model=config.turbulence_model
    ).to_openfoam_string()

    files["0/U"] = _build_u_field(config, flow).to_openfoam_string()
    files["0/p"] = _build_p_field(config).to_openfoam_string()

    if config.turbulence_model != TurbulenceModel.LAMINAR:
        turb_quantities = _estimate_turbulence_quantities(flow, nu)
        files["0/k"] = _build_turbulence_field(
            "k", "[0 2 -2 0 0 0 0]", turb_quantities["k"]
        ).to_openfoam_string()
        if config.turbulence_model == TurbulenceModel.K_OMEGA_SST:
            files["0/omega"] = _build_turbulence_field(
                "omega", "[0 0 -1 0 0 0 0]", turb_quantities["omega"]
            ).to_openfoam_string()
        elif config.turbulence_model == TurbulenceModel.K_EPSILON:
            files["0/epsilon"] = _build_turbulence_field(
                "epsilon", "[0 2 -3 0 0 0 0]", turb_quantities["epsilon"]
            ).to_openfoam_string()

    logger.info("Generated %d OpenFOAM case files.", len(files))
    return files


def check_openfoam_syntax(file_content: str) -> tuple[bool, list[str]]:
    """Run a basic syntax sanity check on a generated OpenFOAM file.

    Verifies the FoamFile header dictionary is present and that curly
    braces and parentheses are balanced.

    Args:
        file_content: The full text content of an OpenFOAM file.

    Returns:
        A tuple of (is_valid, list of problem descriptions).
    """
    problems: list[str] = []

    if "FoamFile" not in file_content:
        problems.append("Missing FoamFile header block.")

    if re.search(r"\bclass\s+\S+;", file_content) is None:
        problems.append("Missing 'class' entry in FoamFile header.")

    brace_balance = file_content.count("{") - file_content.count("}")
    if brace_balance != 0:
        problems.append(f"Unbalanced curly braces (delta={brace_balance}).")

    paren_balance = file_content.count("(") - file_content.count(")")
    if paren_balance != 0:
        problems.append(f"Unbalanced parentheses (delta={paren_balance}).")

    return len(problems) == 0, problems
