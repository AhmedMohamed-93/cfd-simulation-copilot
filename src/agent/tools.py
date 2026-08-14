"""LangGraph tools implementing each reasoning/action step of the agent."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from config import settings
from src.agent.prompts import (
    FLOW_PARSING_PROMPT,
    RETRIEVAL_QUALITY_PROMPT,
    SOLVER_SELECTION_PROMPT,
    SYSTEM_PROMPT,
)
from src.common.llm_client import LLMClient, get_llm_client
from src.generation.openfoam_generator import generate_openfoam_files as _generate_files
from src.generation.physics_validator import (
    _CAVITY_KEYWORDS,
    _check_reynolds_turbulence_consistency,
    _check_solver_algorithm_consistency,
    _check_solver_domain_consistency,
    _laminar_to_turbulent_threshold,
    is_external_aerodynamics_geometry,
)
from src.generation.physics_validator import validate_physics as _validate_physics
from src.generation.schemas import (
    FlowDescription,
    SolverConfiguration,
    ValidationResult,
    ValidationSeverity,
)
from src.retrieval.retriever import CFDRetriever

logger = logging.getLogger(__name__)

_RETRIEVAL_ASPECTS = (
    "solver selection",
    "turbulence model",
    "boundary conditions",
    "numerical schemes",
    "mesh guidelines",
)


def _extract_json(text: str) -> dict[str, Any]:
    """Extract the first top-level JSON object from an LLM text response.

    Args:
        text: Raw LLM output, possibly wrapped in markdown code fences.

    Returns:
        The parsed JSON object as a dict.

    Raises:
        ValueError: If no valid JSON object could be extracted, or if the
            object looks like an echoed-back JSON Schema rather than actual
            extracted/decided data (a failure mode observed with smaller
            models that misread "respond matching this schema" as "return
            this schema").
    """
    cleaned = text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Could not extract JSON from LLM response: {text[:200]}") from exc
        data = json.loads(match.group())

    if isinstance(data, dict) and {"properties", "type"} & data.keys():
        raise ValueError(
            f"Model echoed a JSON Schema instead of returning data: {text[:200]}"
        )
    return data


def _field_hint(spec: dict[str, Any]) -> str:
    """Build a short human-readable type hint for one JSON Schema field.

    Args:
        spec: The field's JSON Schema definition.

    Returns:
        A short type hint such as "number or null" or "one of a, b, c".
    """
    if "enum" in spec:
        return "one of " + ", ".join(str(v) for v in spec["enum"])
    if "type" in spec:
        return str(spec["type"])
    if "anyOf" in spec:
        return " or ".join(_field_hint(s) for s in spec["anyOf"])
    return "any"


def _response_instruction(schema: dict[str, Any]) -> str:
    """Build a model-friendly instruction asking for data, not the schema itself.

    Smaller/weaker LLMs sometimes misinterpret "respond with JSON matching
    this schema: <raw JSON Schema>" as an instruction to return the schema
    verbatim. Listing plain "field: type hint" pairs instead of the full
    nested JSON Schema, plus an explicit anti-echo instruction, avoids this.

    Args:
        schema: A Pydantic model's ``model_json_schema()`` output.

    Returns:
        An instruction string to append to the user prompt.
    """
    fields = "\n".join(
        f'  "{name}": <{_field_hint(spec)}>' for name, spec in schema.get("properties", {}).items()
    )
    return (
        "\n\nRespond with ONLY a single JSON object containing your extracted or "
        "decided values, using exactly these fields:\n{\n"
        f"{fields}\n"
        "}\n"
        "Fill in real values based on the input above. Do NOT return this field "
        "list, type hints, or any schema/description text — only the filled-in "
        "JSON object."
    )


def parse_flow_description(
    user_query: str, client: LLMClient | None = None
) -> FlowDescription:
    """Tool 1: extract structured flow parameters from a natural-language query.

    Args:
        user_query: The user's natural-language CFD problem description.
        client: Optional LLMClient override (for testing).

    Returns:
        A validated FlowDescription. Falls back to a minimal best-effort
        FlowDescription (geometry only) if structured extraction fails.
    """
    client = client or get_llm_client()
    prompt = FLOW_PARSING_PROMPT.format(user_query=user_query)
    schema = FlowDescription.model_json_schema()

    try:
        raw = client.complete(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": prompt + _response_instruction(schema),
                },
            ],
            temperature=settings.LLM_TEMPERATURE,
            json_mode=True,
        )
        data = _extract_json(raw)
        return FlowDescription.model_validate(data)
    except Exception as exc:  # noqa: BLE001
        logger.warning("parse_flow_description failed, using fallback: %s", exc)
        return FlowDescription(geometry=user_query[:200])


def retrieve_cfd_knowledge(
    flow_description: FlowDescription,
    aspect: str,
    retriever: CFDRetriever | None = None,
    client: LLMClient | None = None,
) -> dict[str, Any]:
    """Tool 2: retrieve and self-grade CFD knowledge for a specific aspect.

    Args:
        flow_description: The parsed flow parameters.
        aspect: The specific aspect to retrieve knowledge about, e.g. one of
            "solver selection", "turbulence model", "boundary conditions",
            "numerical schemes", "mesh guidelines".
        retriever: Optional CFDRetriever override (for testing).
        client: Optional LLMClient override (for testing).

    Returns:
        A dict with keys: chunks (list of chunk dicts), quality (float in
        [0, 1]), note (str, present if quality is low).
    """
    retriever = retriever or CFDRetriever()
    client = client or get_llm_client()

    query = (
        f"{aspect} for a flow with geometry '{flow_description.geometry}', "
        f"Reynolds number {flow_description.reynolds_number}, "
        f"fluid {flow_description.fluid}"
    )
    result = retriever.retrieve(query)
    chunks = result["chunks"]

    chunk_summaries = "\n---\n".join(
        f"[{c.metadata.get('title', 'untitled')}] {c.content[:300]}" for c in chunks
    )
    quality = 0.5
    if chunks:
        try:
            prompt = RETRIEVAL_QUALITY_PROMPT.format(query=query, chunks=chunk_summaries)
            text = client.complete(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            ).strip()
            match = re.search(r"\d+(\.\d+)?", text)
            if match:
                quality = max(0.0, min(1.0, float(match.group())))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Retrieval quality self-grading failed: %s", exc)
    else:
        quality = 0.0

    output: dict[str, Any] = {
        "chunks": [
            {
                "content": c.content,
                "metadata": c.metadata,
                "dense_score": c.dense_score,
                "rerank_score": c.rerank_score,
            }
            for c in chunks
        ],
        "quality": quality,
        "aspect": aspect,
    }
    if quality < settings.RETRIEVAL_QUALITY_THRESHOLD:
        output["note"] = (
            f"Retrieval quality ({quality:.2f}) is below the threshold "
            f"({settings.RETRIEVAL_QUALITY_THRESHOLD}); knowledge on '{aspect}' may be incomplete."
        )
    return output


def select_solver_and_models(
    flow_description: FlowDescription,
    retrieved_chunks: list[dict[str, Any]],
    client: LLMClient | None = None,
) -> SolverConfiguration:
    """Tool 3: reason about solver, turbulence model, and scheme choices.

    Args:
        flow_description: The parsed flow parameters.
        retrieved_chunks: Retrieved knowledge chunks (as dicts) to ground
            the decision in documented OpenFOAM guidance.
        client: Optional LLMClient override (for testing).

    Returns:
        A validated SolverConfiguration. Falls back to a conservative
        rule-based selection (mirroring the documented decision tree) if
        LLM structured output fails.
    """
    client = client or get_llm_client()
    context = "\n---\n".join(
        f"[{c['metadata'].get('title', 'untitled')}] {c['content'][:400]}"
        for c in retrieved_chunks
    )
    schema = SolverConfiguration.model_json_schema()
    prompt = SOLVER_SELECTION_PROMPT.format(
        flow_description=flow_description.model_dump_json(indent=2),
        retrieved_context=context or "(no relevant documentation retrieved)",
    )

    try:
        raw = client.complete(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": prompt + _response_instruction(schema),
                },
            ],
            temperature=settings.LLM_TEMPERATURE,
            json_mode=True,
        )
        data = _extract_json(raw)
        config = SolverConfiguration.model_validate(data)

        domain_check = _check_solver_domain_consistency(flow_description, config)
        algorithm_check = _check_solver_algorithm_consistency(config)
        turbulence_check = _check_reynolds_turbulence_consistency(flow_description, config)

        solver_bad = ValidationSeverity.ERROR in (domain_check.severity, algorithm_check.severity)
        # A missing Reynolds number also reports non-PASS ("cannot verify"),
        # but that's not evidence the LLM's choice is wrong — only override
        # when Re is known and genuinely inconsistent with the model chosen.
        turbulence_bad = (
            flow_description.reynolds_number is not None
            and turbulence_check.severity != ValidationSeverity.PASS
        )

        if not solver_bad and not turbulence_bad:
            return config

        fallback = _rule_based_solver_selection(flow_description)
        bad_rule = domain_check.rule if domain_check.severity == ValidationSeverity.ERROR else (
            algorithm_check.rule if algorithm_check.severity == ValidationSeverity.ERROR else turbulence_check.rule
        )
        logger.warning(
            "select_solver_and_models: LLM chose solver=%s turbulence=%s for geometry %r, "
            "which fails the '%s' sanity check; overriding with rule-based selection "
            "(solver_bad=%s, turbulence_bad=%s).",
            config.solver_name.value,
            config.turbulence_model.value,
            flow_description.geometry,
            bad_rule,
            solver_bad,
            turbulence_bad,
        )

        if solver_bad and turbulence_bad:
            # Neither half of the LLM's answer can be trusted — replace the
            # whole configuration.
            return fallback
        if solver_bad:
            # Only the solver family/algorithm was wrong; the LLM's
            # turbulence model passed its own sanity check, so keep it
            # rather than discarding a correct answer along with a wrong
            # one. The justification is composed (not just fallback.justification
            # copied wholesale) so it doesn't describe a solver that was
            # never actually adopted.
            return config.model_copy(
                update={
                    "solver_name": fallback.solver_name,
                    "is_compressible": fallback.is_compressible,
                    "is_steady": fallback.is_steady,
                    "justification": (
                        f"{config.justification} [Solver overridden: '{bad_rule}' flagged "
                        f"the original choice ({config.solver_name.value}) — corrected to "
                        f"{fallback.solver_name.value} via rule-based fallback.]"
                    ),
                }
            )
        # Only the turbulence model was wrong; keep the LLM's (valid) solver choice.
        return config.model_copy(
            update={
                "turbulence_model": fallback.turbulence_model,
                "simulation_type": fallback.simulation_type,
                "justification": (
                    f"{config.justification} [Turbulence model overridden: "
                    f"'{turbulence_check.rule}' flagged the original choice "
                    f"({config.turbulence_model.value}) — corrected to "
                    f"{fallback.turbulence_model.value} via rule-based fallback.]"
                ),
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("select_solver_and_models failed, using rule-based fallback: %s", exc)
        return _rule_based_solver_selection(flow_description)


def _rule_based_solver_selection(flow: FlowDescription) -> SolverConfiguration:
    """Deterministic fallback solver selection mirroring the decision tree.

    Args:
        flow: The parsed flow parameters.

    Returns:
        A SolverConfiguration chosen via simple, explainable rules, used
        only when the LLM-based selection tool fails.
    """
    from src.generation.schemas import SimulationType, SolverName, TurbulenceModel

    re_number = flow.reynolds_number or 1.0
    is_external_aero = is_external_aerodynamics_geometry(flow.geometry)
    is_high_speed = (flow.mach_number or 0.0) > 0.3
    is_laminar_regime = re_number < _laminar_to_turbulent_threshold(flow.geometry)
    # icoFoam is OpenFOAM's simplest transient incompressible laminar
    # solver, historically bundled with exactly the canonical validation
    # cases this keyword list targets (its own tutorials are named
    # `cavity`/`cavityClipped`/...) — it is not a general-purpose choice
    # for arbitrary laminar unsteady flows (e.g. cylinder vortex shedding
    # at the same Re still wants pimpleFoam), so this is deliberately
    # narrow rather than "any laminar + unsteady flow". Shared with
    # physics_validator's solver_domain_consistency check (clause 3).
    is_cavity_case = any(kw in flow.geometry.lower() for kw in _CAVITY_KEYWORDS)

    if flow.multiphase:
        solver = SolverName.INTER_FOAM
    elif is_external_aero:
        # External aerodynamics (flow around a body) is never
        # buoyancy-driven, regardless of temperature_dependent — that flag
        # can be true for reasons unrelated to what's actually driving the
        # flow (e.g. compressible heating), and buoyant solvers model flows
        # where density differences FROM temperature drive the motion, not
        # flow past an external body. Checked before temperature_dependent
        # so a mis-set flag can't route an airfoil/wing/aircraft case to a
        # buoyant solver. See physics_validator's solver_domain_consistency
        # check for the corresponding post-generation safety net.
        if flow.is_compressible or is_high_speed:
            solver = SolverName.RHO_SIMPLE_FOAM if flow.is_steady else SolverName.RHO_PIMPLE_FOAM
        else:
            solver = SolverName.SIMPLE_FOAM if flow.is_steady else SolverName.PIMPLE_FOAM
    elif flow.temperature_dependent:
        solver = SolverName.BUOYANT_SIMPLE_FOAM if flow.is_steady else SolverName.BUOYANT_PIMPLE_FOAM
    elif flow.is_compressible:
        solver = SolverName.RHO_SIMPLE_FOAM if flow.is_steady else SolverName.RHO_PIMPLE_FOAM
    elif not flow.is_steady and is_laminar_regime and is_cavity_case:
        solver = SolverName.ICO_FOAM
    elif flow.is_steady:
        solver = SolverName.SIMPLE_FOAM
    else:
        solver = SolverName.PIMPLE_FOAM

    if is_laminar_regime:
        turbulence = TurbulenceModel.LAMINAR
        sim_type = SimulationType.LAMINAR
    elif is_external_aero:
        # SpalartAllmaras is the standard, economical one-equation model
        # for attached external aerodynamic boundary layers (airfoils,
        # wings) — the industry-default choice for exactly this case type,
        # not merely "also acceptable" alongside kOmegaSST.
        turbulence = TurbulenceModel.SPALART_ALLMARAS
        sim_type = SimulationType.RAS
    else:
        turbulence = TurbulenceModel.K_OMEGA_SST
        sim_type = SimulationType.RAS

    return SolverConfiguration(
        solver_name=solver,
        turbulence_model=turbulence,
        is_compressible=flow.is_compressible,
        is_steady=flow.is_steady,
        simulation_type=sim_type,
        justification=(
            f"Rule-based fallback: Re={re_number:g} -> {turbulence.value}; "
            f"steady={flow.is_steady}, compressible={flow.is_compressible}, "
            f"multiphase={flow.multiphase}, external_aerodynamics={is_external_aero}, "
            f"buoyant={flow.temperature_dependent and not is_external_aero} -> {solver.value}."
        ),
        numerical_schemes_notes="Default bounded upwind convection scheme for robustness.",
    )


def generate_openfoam_case_files(
    solver_config: SolverConfiguration, flow_description: FlowDescription
) -> dict[str, str]:
    """Tool 4: generate the full OpenFOAM case file set.

    Args:
        solver_config: The selected solver/turbulence configuration.
        flow_description: The parsed flow parameters.

    Returns:
        A dict mapping OpenFOAM file path to file content.
    """
    return _generate_files(solver_config, flow_description)


def validate_case_physics(
    flow_description: FlowDescription,
    solver_config: SolverConfiguration,
    generated_files: dict[str, str],
) -> ValidationResult:
    """Tool 5: run rule-based physics validation on the generated case.

    Args:
        flow_description: The parsed flow parameters.
        solver_config: The selected solver/turbulence configuration.
        generated_files: The generated OpenFOAM file contents.

    Returns:
        A ValidationResult with pass/warning/error findings.
    """
    return _validate_physics(flow_description, solver_config, generated_files)


def format_final_response(
    flow_description: FlowDescription,
    solver_config: SolverConfiguration,
    generated_files: dict[str, str],
    validation_result: ValidationResult,
    citations: list[dict[str, Any]],
) -> str:
    """Tool 6: assemble the final markdown response for the user.

    Args:
        flow_description: The parsed flow parameters.
        solver_config: The selected solver/turbulence configuration.
        generated_files: The generated OpenFOAM file contents.
        validation_result: The physics validation outcome.
        citations: Source citations gathered during retrieval.

    Returns:
        A complete markdown-formatted response string.
    """
    lines: list[str] = []
    lines.append("## Recommended Approach\n")
    lines.append(
        f"**Solver:** `{solver_config.solver_name.value}`  \n"
        f"**Turbulence model:** `{solver_config.turbulence_model.value}`  \n"
        f"**Regime:** {'steady' if solver_config.is_steady else 'transient'}, "
        f"{'compressible' if solver_config.is_compressible else 'incompressible'}\n"
    )
    lines.append(f"\n**Justification:** {solver_config.justification}\n")
    if solver_config.numerical_schemes_notes:
        lines.append(f"\n**Numerical scheme notes:** {solver_config.numerical_schemes_notes}\n")

    lines.append("\n## Generated OpenFOAM Case Files\n")
    for path, content in generated_files.items():
        lines.append(f"\n### `{path}`\n```cpp\n{content}\n```\n")

    lines.append("\n## Physics Validation\n")
    if validation_result.passed:
        lines.append("**Status: PASSED** — no critical physics errors detected.\n")
    else:
        lines.append("**Status: FAILED** — critical physics errors detected.\n")
    for finding in validation_result.findings:
        icon = {"pass": "✅", "warning": "⚠️", "error": "❌"}[finding.severity.value]
        lines.append(f"- {icon} `{finding.rule}`: {finding.message}")

    if citations:
        lines.append("\n## Citations\n")
        for c in citations:
            # rerank_score is 0-10 (min-max normalized within the retrieval
            # batch); display as a 0-100% relevance figure.
            relevance = (
                f" — relevance {c['rerank_score'] * 10:.0f}%"
                if c.get("rerank_score") is not None
                else ""
            )
            lines.append(f"- {c.get('title', 'untitled')} ({c.get('source', 'unknown')}){relevance}")

    lines.append("\n## Recommendations\n")
    lines.append(
        "- Generate the mesh with blockMesh or snappyHexMesh, ensuring wall-adjacent "
        "cells match the y+ target for the selected turbulence model's wall treatment.\n"
        "- Monitor residuals for p and U (and k/epsilon/omega if applicable); target "
        "at least 3-4 orders of magnitude drop for a converged steady-state case.\n"
        "- For transient cases, verify the Courant number stays within the solver's "
        "stable range (see PIMPLE nOuterCorrectors if it must exceed 1).\n"
        "- Run `checkMesh` before the first solve to catch non-orthogonality or "
        "skewness issues.\n"
    )

    lines.append("\n## Next Steps\n")
    lines.append(
        "1. Create the case directory and copy the generated files into their "
        "respective `0/`, `constant/`, and `system/` subdirectories.\n"
        "2. Generate or import a mesh matching the described geometry.\n"
        f"3. Run `{solver_config.solver_name.value}` and monitor convergence.\n"
        "4. Post-process with ParaView or `postProcess` function objects for the "
        f"desired outputs: {', '.join(flow_description.desired_outputs) or 'as specified'}.\n"
    )

    return "\n".join(lines)
