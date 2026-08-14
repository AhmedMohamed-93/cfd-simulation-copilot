"""Pydantic v2 schemas for OpenFOAM case files and agent structured outputs.

Every file-representing schema implements ``to_openfoam_string()``, which
renders a syntactically valid OpenFOAM dictionary file (including the
standard FoamFile header) from the structured data.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

_FOAM_HEADER_TEMPLATE = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox          |
|  \\\\    /   O peration     | Version:  10                                   |
|   \\\\  /    A nd           | Web:      www.openfoam.com                     |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       {foam_class};
    object      {object_name};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
"""


def _render_header(foam_class: str, object_name: str) -> str:
    """Render the standard OpenFOAM FoamFile header block.

    Args:
        foam_class: The OpenFOAM ``class`` entry (e.g. "volVectorField").
        object_name: The OpenFOAM ``object`` entry (e.g. "U").

    Returns:
        The formatted header string, including the FoamFile dictionary.
    """
    return _FOAM_HEADER_TEMPLATE.format(foam_class=foam_class, object_name=object_name)


# --------------------------------------------------------------------------
# Enums
# --------------------------------------------------------------------------


class SolverName(str, Enum):
    """Supported OpenFOAM solver executables."""

    SIMPLE_FOAM = "simpleFoam"
    PIMPLE_FOAM = "pimpleFoam"
    ICO_FOAM = "icoFoam"
    RHO_SIMPLE_FOAM = "rhoSimpleFoam"
    RHO_PIMPLE_FOAM = "rhoPimpleFoam"
    INTER_FOAM = "interFoam"
    BUOYANT_SIMPLE_FOAM = "buoyantSimpleFoam"
    BUOYANT_PIMPLE_FOAM = "buoyantPimpleFoam"


class TurbulenceModel(str, Enum):
    """Supported turbulence closure models."""

    LAMINAR = "laminar"
    K_EPSILON = "kEpsilon"
    K_OMEGA_SST = "kOmegaSST"
    SPALART_ALLMARAS = "SpalartAllmaras"
    SMAGORINSKY = "Smagorinsky"


class SimulationType(str, Enum):
    """OpenFOAM turbulenceProperties simulationType values."""

    LAMINAR = "laminar"
    RAS = "RAS"
    LES = "LES"


# --------------------------------------------------------------------------
# Flow description (agent Tool 1 structured output)
# --------------------------------------------------------------------------


class FlowDescription(BaseModel):
    """Structured extraction of the physical flow problem from user text.

    Attributes:
        reynolds_number: Reynolds number of the flow, if determinable.
        mach_number: Mach number, for compressible flows.
        is_compressible: Whether compressibility effects are significant.
        is_steady: Whether the flow is expected to reach a steady state.
        geometry: Short description of the flow geometry (e.g. "pipe",
            "cylinder in crossflow", "NACA0012 airfoil").
        fluid: Name of the working fluid (e.g. "air", "water").
        kinematic_viscosity: Kinematic viscosity in m^2/s, if known/derivable.
        characteristic_length: Characteristic length scale in meters.
        inlet_velocity: Free-stream / inlet velocity magnitude in m/s.
        desired_outputs: Quantities of interest (e.g. ["drag coefficient",
            "pressure drop"]).
        temperature_dependent: Whether heat transfer / buoyancy matters.
        multiphase: Whether more than one fluid phase is present.
    """

    reynolds_number: float | None = Field(default=None, ge=0)
    mach_number: float | None = Field(default=None, ge=0)
    is_compressible: bool = False
    is_steady: bool = True
    geometry: str = Field(default="unspecified")
    fluid: str = Field(default="air")
    kinematic_viscosity: float | None = Field(default=None, gt=0)
    characteristic_length: float | None = Field(default=None, gt=0)
    inlet_velocity: float | None = Field(default=None, ge=0)
    desired_outputs: list[str] = Field(default_factory=list)
    temperature_dependent: bool = False
    multiphase: bool = False


# --------------------------------------------------------------------------
# Solver configuration (agent Tool 3 structured output)
# --------------------------------------------------------------------------


class SolverConfiguration(BaseModel):
    """The agent's chosen solver, turbulence model, and justification.

    Attributes:
        solver_name: The selected OpenFOAM solver.
        turbulence_model: The selected turbulence closure model.
        is_compressible: Whether the compressible solver family was chosen.
        is_steady: Whether a steady-state solver was chosen.
        simulation_type: RAS, LES, or laminar (drives turbulenceProperties).
        justification: Free-text explanation grounded in retrieved docs.
        numerical_schemes_notes: Short notes on scheme choices.
    """

    solver_name: SolverName
    turbulence_model: TurbulenceModel
    is_compressible: bool
    is_steady: bool
    simulation_type: SimulationType
    justification: str
    numerical_schemes_notes: str = Field(default="")


# --------------------------------------------------------------------------
# controlDict
# --------------------------------------------------------------------------


class ControlDictConfig(BaseModel):
    """Structured representation of system/controlDict.

    Attributes:
        solver: Solver application name.
        startTime: Simulation start time.
        endTime: Simulation end time (or number of iterations for steady).
        deltaT: Time step size.
        writeInterval: Interval (in time units or iterations) between writes.
        writeFormat: "ascii" or "binary".
    """

    solver: SolverName
    startTime: float = 0
    endTime: float = 1000
    deltaT: float = 1
    writeInterval: float = 100
    writeFormat: str = "ascii"

    def to_openfoam_string(self) -> str:
        """Render this config as a valid system/controlDict file.

        Returns:
            The full OpenFOAM controlDict file content as a string.
        """
        header = _render_header("dictionary", "controlDict")
        return f"""{header}
application     {self.solver.value};

startFrom       startTime;
startTime       {self.startTime};

stopAt          endTime;
endTime         {self.endTime};

deltaT          {self.deltaT};

writeControl    timeStep;
writeInterval   {self.writeInterval};

purgeWrite      0;
writeFormat     {self.writeFormat};
writePrecision  6;
writeCompression off;

timeFormat      general;
timePrecision   6;

runTimeModifiable true;

// ************************************************************************* //
"""


# --------------------------------------------------------------------------
# fvSchemes
# --------------------------------------------------------------------------


class FvSchemesConfig(BaseModel):
    """Structured representation of system/fvSchemes.

    Attributes:
        ddtSchemes: Time-derivative discretization scheme.
        gradSchemes: Gradient discretization scheme.
        divSchemes: Mapping of divergence term -> scheme string.
        laplacianSchemes: Laplacian discretization scheme.
        interpolationSchemes: Interpolation scheme.
    """

    ddtSchemes: str = "steadyState"
    gradSchemes: str = "Gauss linear"
    divSchemes: dict[str, str] = Field(
        default_factory=lambda: {
            "default": "none",
            "div(phi,U)": "bounded Gauss upwind",
            "div((nuEff*dev2(T(grad(U)))))": "Gauss linear",
        }
    )
    laplacianSchemes: str = "Gauss linear corrected"
    interpolationSchemes: str = "linear"

    def to_openfoam_string(self) -> str:
        """Render this config as a valid system/fvSchemes file.

        Returns:
            The full OpenFOAM fvSchemes file content as a string.
        """
        header = _render_header("dictionary", "fvSchemes")
        div_lines = "\n".join(
            f"    {term:<40}{scheme};" for term, scheme in self.divSchemes.items()
        )
        return f"""{header}
ddtSchemes
{{
    default         {self.ddtSchemes};
}}

gradSchemes
{{
    default         {self.gradSchemes};
}}

divSchemes
{{
{div_lines}
}}

laplacianSchemes
{{
    default         {self.laplacianSchemes};
}}

interpolationSchemes
{{
    default         {self.interpolationSchemes};
}}

snGradSchemes
{{
    default         corrected;
}}

wallDist
{{
    method          meshWave;
}}

// ************************************************************************* //
"""


# --------------------------------------------------------------------------
# fvSolution
# --------------------------------------------------------------------------


class FieldSolverSettings(BaseModel):
    """Linear solver settings for a single field.

    Attributes:
        solver: Linear solver name (e.g. "GAMG", "smoothSolver").
        smoother: Smoother name, if applicable.
        tolerance: Absolute convergence tolerance.
        relTol: Relative convergence tolerance.
    """

    solver: str = "GAMG"
    smoother: str = "GaussSeidel"
    tolerance: float = 1e-6
    relTol: float = 0.1

    def to_block(self, field_name: str) -> str:
        """Render this field's solver settings as an fvSolution sub-block.

        Args:
            field_name: The field this block applies to (e.g. "p", "U").

        Returns:
            A formatted OpenFOAM dictionary sub-block string.
        """
        return (
            f"    {field_name}\n"
            f"    {{\n"
            f"        solver          {self.solver};\n"
            f"        smoother        {self.smoother};\n"
            f"        tolerance       {self.tolerance:g};\n"
            f"        relTol          {self.relTol:g};\n"
            f"    }}\n"
        )


class FvSolutionConfig(BaseModel):
    """Structured representation of system/fvSolution.

    Attributes:
        solvers: Mapping of field name -> FieldSolverSettings.
        is_steady: Whether SIMPLE (steady) or PIMPLE (transient) is used.
        relaxation_factors: Mapping of field/equation -> under-relaxation
            factor.
        n_correctors: Number of PIMPLE/PISO correctors (transient only).
        n_non_orthogonal_correctors: Number of non-orthogonal correctors.
    """

    solvers: dict[str, FieldSolverSettings] = Field(
        default_factory=lambda: {
            "p": FieldSolverSettings(tolerance=1e-6, relTol=0.05),
            "U": FieldSolverSettings(
                solver="smoothSolver", smoother="GaussSeidel", tolerance=1e-8, relTol=0.1
            ),
        }
    )
    is_steady: bool = True
    relaxation_factors: dict[str, float] = Field(
        default_factory=lambda: {"p": 0.3, "U": 0.7, "k": 0.7, "epsilon": 0.7, "omega": 0.7}
    )
    n_correctors: int = 2
    n_non_orthogonal_correctors: int = 1

    def to_openfoam_string(self) -> str:
        """Render this config as a valid system/fvSolution file.

        Returns:
            The full OpenFOAM fvSolution file content as a string.
        """
        header = _render_header("dictionary", "fvSolution")
        solver_blocks = "\n".join(
            field_settings.to_block(field) for field, field_settings in self.solvers.items()
        )
        relax_lines = "\n".join(
            f"        {field:<10}{factor:g};" for field, factor in self.relaxation_factors.items()
        )
        algo_name = "SIMPLE" if self.is_steady else "PIMPLE"
        algo_block = (
            f"{algo_name}\n{{\n"
            f"    nNonOrthogonalCorrectors {self.n_non_orthogonal_correctors};\n"
        )
        if not self.is_steady:
            algo_block += f"    nCorrectors              {self.n_correctors};\n"
            algo_block += "    nOuterCorrectors         1;\n"
        algo_block += (
            f"    residualControl\n    {{\n        p               1e-4;\n"
            f"        U               1e-4;\n    }}\n\n"
            f"    relaxationFactors\n    {{\n{relax_lines}\n    }}\n}}\n"
        )

        return f"""{header}
solvers
{{
{solver_blocks}
}}

{algo_block}
// ************************************************************************* //
"""


# --------------------------------------------------------------------------
# Boundary conditions / field files
# --------------------------------------------------------------------------


class BoundaryCondition(BaseModel):
    """A single boundary patch condition for one field.

    Attributes:
        type: The OpenFOAM boundary condition type (e.g. "fixedValue",
            "noSlip", "zeroGradient", "inletOutlet").
        value: Optional uniform value string (e.g. "uniform (10 0 0)" or
            "uniform 0"). Omitted for conditions that don't need one
            (e.g. zeroGradient).
    """

    type: str
    value: str | None = None

    def to_block(self, patch_name: str) -> str:
        """Render this boundary condition as a boundaryField sub-block.

        Args:
            patch_name: The name of the mesh patch this applies to.

        Returns:
            A formatted OpenFOAM dictionary sub-block string.
        """
        lines = [f"    {patch_name}", "    {", f"        type            {self.type};"]
        if self.value is not None:
            lines.append(f"        value           {self.value};")
        lines.append("    }")
        return "\n".join(lines)


class FieldFile(BaseModel):
    """Structured representation of a 0/<field> initial/boundary condition file.

    Attributes:
        field_name: The field name (e.g. "U", "p", "k", "omega", "epsilon").
        foam_class: The OpenFOAM class (e.g. "volVectorField", "volScalarField").
        dimensions: The OpenFOAM dimension set string (e.g. "[0 1 -1 0 0 0 0]").
        internal_field: The internalField entry value (e.g. "uniform (0 0 0)").
        boundary_field: Mapping of patch name -> BoundaryCondition.
    """

    field_name: str
    foam_class: str
    dimensions: str
    internal_field: str
    boundary_field: dict[str, BoundaryCondition]

    def to_openfoam_string(self) -> str:
        """Render this config as a valid 0/<field> file.

        Returns:
            The full OpenFOAM field file content as a string.
        """
        header = _render_header(self.foam_class, self.field_name)
        patches = "\n".join(
            bc.to_block(patch) for patch, bc in self.boundary_field.items()
        )
        return f"""{header}
dimensions      {self.dimensions};

internalField   {self.internal_field};

boundaryField
{{
{patches}
}}

// ************************************************************************* //
"""


# --------------------------------------------------------------------------
# turbulenceProperties
# --------------------------------------------------------------------------


class TurbulenceProperties(BaseModel):
    """Structured representation of constant/turbulenceProperties.

    Attributes:
        simulation_type: laminar, RAS, or LES.
        model: The specific turbulence model name (RASModel or LESModel).
        turbulence_on: Whether "turbulence" is switched on (RAS/LES only).
        print_coeffs: Whether to print model coefficients on startup.
    """

    simulation_type: SimulationType
    model: TurbulenceModel
    turbulence_on: bool = True
    print_coeffs: bool = True

    def to_openfoam_string(self) -> str:
        """Render this config as a valid constant/turbulenceProperties file.

        Returns:
            The full OpenFOAM turbulenceProperties file content as a string.
        """
        header = _render_header("dictionary", "turbulenceProperties")
        if self.simulation_type == SimulationType.LAMINAR:
            return f"""{header}
simulationType  laminar;

// ************************************************************************* //
"""
        model_block_key = "RAS" if self.simulation_type == SimulationType.RAS else "LES"
        on_off = "on" if self.turbulence_on else "off"
        print_flag = "on" if self.print_coeffs else "off"
        return f"""{header}
simulationType  {self.simulation_type.value};

{model_block_key}
{{
    {model_block_key}Model       {self.model.value};

    turbulence      {on_off};
    printCoeffs     {print_flag};
}}

// ************************************************************************* //
"""


# --------------------------------------------------------------------------
# transportProperties
# --------------------------------------------------------------------------


class TransportProperties(BaseModel):
    """Structured representation of constant/transportProperties.

    Attributes:
        transport_model: The OpenFOAM transport model, usually "Newtonian".
        kinematic_viscosity: Kinematic viscosity nu in m^2/s.
    """

    transport_model: str = "Newtonian"
    kinematic_viscosity: float = 1.5e-5

    def to_openfoam_string(self) -> str:
        """Render this config as a valid constant/transportProperties file.

        Returns:
            The full OpenFOAM transportProperties file content as a string.
        """
        header = _render_header("dictionary", "transportProperties")
        return f"""{header}
transportModel  {self.transport_model};

nu              [0 2 -1 0 0 0 0] {self.kinematic_viscosity:.3e};

// ************************************************************************* //
"""


# --------------------------------------------------------------------------
# Physics validation results (agent Tool 5 structured output)
# --------------------------------------------------------------------------


class ValidationSeverity(str, Enum):
    """Severity level of a physics validation finding."""

    PASS = "pass"
    WARNING = "warning"
    ERROR = "error"


class ValidationFinding(BaseModel):
    """A single physics validation check result.

    Attributes:
        rule: Short identifier for the rule that produced this finding.
        severity: pass, warning, or error.
        message: Human-readable explanation of the finding.
    """

    rule: str
    severity: ValidationSeverity
    message: str


class ValidationResult(BaseModel):
    """Aggregate physics validation output for a generated case.

    Attributes:
        findings: All individual validation findings.
        passed: True if there are no ERROR-severity findings.
    """

    findings: list[ValidationFinding] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Whether the case has no critical (error-severity) findings.

        Returns:
            True if no finding has ERROR severity, False otherwise.
        """
        return not any(f.severity == ValidationSeverity.ERROR for f in self.findings)

    @property
    def warnings(self) -> list[ValidationFinding]:
        """All warning-severity findings.

        Returns:
            The subset of findings with WARNING severity.
        """
        return [f for f in self.findings if f.severity == ValidationSeverity.WARNING]

    @property
    def errors(self) -> list[ValidationFinding]:
        """All error-severity findings.

        Returns:
            The subset of findings with ERROR severity.
        """
        return [f for f in self.findings if f.severity == ValidationSeverity.ERROR]
