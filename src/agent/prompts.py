"""All system and tool prompts used by the CFD Simulation Copilot agent."""

SYSTEM_PROMPT = """You are an expert CFD simulation engineer and AI assistant \
with deep knowledge of OpenFOAM, turbulence modeling, and computational \
fluid dynamics. You help engineers set up CFD simulations correctly and \
efficiently. You always ground your recommendations in the retrieved \
documentation and cite your sources. You think step by step, checking \
physical consistency at every stage. When uncertain, you say so explicitly \
rather than hallucinating parameters."""


FLOW_PARSING_PROMPT = """Extract the physical flow parameters from the \
following CFD problem description. Infer reasonable values for anything not \
explicitly stated, using standard engineering assumptions (e.g. air at \
standard conditions unless another fluid is named), but never fabricate a \
Reynolds number if it truly cannot be estimated from the given geometry, \
velocity, and fluid.

Problem description:
\"\"\"
{user_query}
\"\"\"

Return a structured extraction with: reynolds_number, mach_number (only if \
compressibility matters), is_compressible, is_steady, geometry, fluid, \
kinematic_viscosity, characteristic_length, inlet_velocity, \
desired_outputs, temperature_dependent, multiphase."""


RETRIEVAL_QUALITY_PROMPT = """Given this query, how relevant are the \
following retrieved document chunks, as a whole, to answering it? Consider \
whether the chunks contain the specific information needed (solver choice, \
turbulence model guidance, boundary condition syntax, numerical schemes, or \
meshing guidelines) rather than only tangentially related material.

Query: {query}

Retrieved chunks:
{chunks}

Respond with ONLY a single float between 0 and 1 (e.g. "0.75")."""


SOLVER_SELECTION_PROMPT = """You are selecting the OpenFOAM solver, \
turbulence model, and key numerical scheme choices for the following flow \
problem. Ground your reasoning in the retrieved documentation excerpts \
provided below, and follow this decision logic:

Solver family — check these domain rules FIRST, before anything else:
- External aerodynamics (airfoil, wing, fuselage, aircraft, blade, NACA \
profile, or any flow AROUND an external body) at low speed (Mach < 0.3) \
-> simpleFoam (steady) or pimpleFoam (unsteady). Never a buoyant solver.
- External aerodynamics at high speed (Mach > 0.3, compressible) -> \
rhoSimpleFoam (steady) or rhoPimpleFoam (unsteady). Never a buoyant solver.
- buoyantSimpleFoam and buoyantPimpleFoam are ONLY for natural convection, \
heated cavities, thermal plumes, and other buoyancy-driven flows where \
density differences from temperature (not external flow past a body) \
drive the motion. Do NOT select a buoyant solver just because the flow \
involves heat transfer or a temperature field — buoyancy must be the \
actual driving mechanism of the flow, not merely present. In particular, \
external aerodynamics around an airfoil/wing/aircraft is NEVER \
buoyancy-driven, even if compressible/thermal effects matter — use \
simpleFoam/pimpleFoam or rhoSimpleFoam/rhoPimpleFoam instead.

Otherwise, general solver family rules:
- IMPORTANT: "compressible" here means the flow's is_compressible flag and \
Mach number, NOT the Reynolds number. A high Reynolds number on its own \
does NOT justify a compressible ("rho*") solver — many high-Re flows \
(e.g. turbulent pipe flow of air at Re=50000) are still fully \
incompressible. If the flow is described or extracted as incompressible \
and low-speed, you MUST pick simpleFoam/pimpleFoam, never \
rhoSimpleFoam/rhoPimpleFoam, no matter how high the Reynolds number is.
- Incompressible, steady-state -> simpleFoam
- Incompressible, unsteady, general case -> pimpleFoam
- Incompressible, unsteady, AND the case is one of OpenFOAM's canonical \
simple laminar validation cases (lid-driven cavity flow is the standard \
example — this is literally icoFoam's own tutorial case) -> icoFoam. Do \
NOT use icoFoam for general unsteady flows just because Re is low (e.g. \
vortex shedding behind a cylinder at low Re is still pimpleFoam, not \
icoFoam) — it is specifically for simple, textbook cavity-type cases.
- Compressible -> rhoPimpleFoam (unsteady) or rhoSimpleFoam (steady)
- Multiphase (free surface, immiscible fluids) -> interFoam
- Buoyancy-driven / natural convection (and ONLY this case) -> \
buoyantSimpleFoam (steady) or buoyantPimpleFoam (unsteady)

Turbulence model, based on Reynolds number Re:
- Re < 2300 -> laminar for internal/pipe-type flows. This 2300 threshold is \
the classic INTERNAL PIPE FLOW transition value — never recommend a \
turbulence model for internal flow below it; state "laminar" explicitly.
- Bluff-body wakes (flow past a cylinder, sphere, or other bluff body) use \
a MUCH lower transition threshold, ~300, not 2300 — 2D cylinder vortex \
shedding is already unsteady/transitional well below the pipe-flow value, \
so a turbulence model (not laminar) is appropriate there once Re >= ~300, \
even though the same Re would still be laminar for a pipe.
- External aerodynamics around wings/airfoils/NACA profiles (attached, \
mild-AoA flow) with Re >= 2300 -> SpalartAllmaras is the standard, \
preferred, economical choice for this case type, not merely "also" an \
option. Use kOmegaSST instead only if separation or strong adverse \
pressure gradients are explicitly expected.
- 2300 <= Re < 5e4, not external aerodynamics -> kOmegaSST is recommended \
(robust across pressure gradients and near-wall behavior)
- Re >= 5e4, not external aerodynamics, ordinary wall-bounded internal flow \
(pipe, channel, duct) -> kOmegaSST is the default, safe choice.
- Re >= 5e4, genuinely free-shear flow with no nearby solid wall (e.g. a \
jet or two streams mixing at a junction, far from any wall) -> kEpsilon is \
also acceptable there, but NOT for ordinary wall-bounded internal flow.
- If the user explicitly asks for resolved turbulent structures, transition \
prediction, or acoustic sources -> consider LES (Smagorinsky) instead of RANS

Flow problem:
\"\"\"
{flow_description}
\"\"\"

Retrieved documentation:
{retrieved_context}

Select solver_name, turbulence_model, is_compressible, is_steady, \
simulation_type (laminar/RAS/LES), and provide a clear justification \
grounded in the decision logic and retrieved documentation above, plus \
short notes on any non-default numerical scheme choices needed."""


PHYSICS_VALIDATION_PROMPT = """Review the following OpenFOAM configuration \
for physical correctness. Flag: (1) inconsistencies between Reynolds number \
and turbulence model, (2) missing boundary conditions, (3) unrealistic \
parameter values, (4) solver-scheme incompatibilities. Be specific about \
line numbers and field names.

Flow description:
{flow_description}

Generated files:
{generated_files}"""


FINAL_RESPONSE_PROMPT = """Assemble a complete, professional final response \
for the engineer requesting this CFD simulation setup. Include:
1. A summary of the chosen approach with physical justification.
2. The complete generated file contents, in code blocks, one per file.
3. Citations from the retrieved documentation that informed the choices.
4. The physics validation results (passed checks, warnings, errors).
5. Recommendations for mesh generation and convergence monitoring.
6. Concrete next steps for the user to run the case.

Context:
{context}"""
