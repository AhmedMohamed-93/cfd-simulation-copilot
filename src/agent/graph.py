"""LangGraph ReAct agent graph wiring together all CFD copilot tools.

Graph shape:

    START
      |
    parse_flow_description
      |
    retrieve_cfd_knowledge  <--- loop (max MAX_RETRIEVAL_ATTEMPTS) if quality low
      |
    select_solver_and_models
      |
    generate_openfoam_files
      |
    validate_physics  <--- loop back to select_solver_and_models (max
      |                    MAX_VALIDATION_RETRIES) if critical errors found
    format_final_response
      |
     END

An iteration counter enforces a hard cap (MAX_AGENT_ITERATIONS) on total
graph steps to guarantee termination regardless of the conditional loops.
"""

from __future__ import annotations

import logging
import time

from langgraph.graph import END, StateGraph

from config import settings
from src.agent import tools as agent_tools
from src.agent.state import AgentState, initial_state
from src.generation.schemas import FlowDescription, SolverConfiguration, ValidationResult
from src.observability.logger import log_agent_trace
from src.retrieval.retriever import CFDRetriever

logger = logging.getLogger(__name__)


def _log_step(state: AgentState, node_name: str, elapsed_s: float) -> None:
    """Append a human-readable trace entry for one node execution.

    Args:
        state: The mutable agent state to append the trace entry to.
        node_name: Name of the node that just ran.
        elapsed_s: Wall-clock time the node took, in seconds.
    """
    entry = f"[{node_name}] completed in {elapsed_s:.2f}s"
    state["reasoning_steps"].append(entry)
    logger.info(entry)


def node_parse_flow_description(state: AgentState) -> AgentState:
    """Graph node wrapping Tool 1: parse_flow_description.

    Args:
        state: The current agent state.

    Returns:
        The updated agent state with flow_description populated.
    """
    start = time.perf_counter()
    flow = agent_tools.parse_flow_description(state["user_query"])
    state["flow_description"] = flow.model_dump(mode="json")
    state["iteration_count"] += 1
    _log_step(state, "parse_flow_description", time.perf_counter() - start)
    return state


def node_retrieve_cfd_knowledge(state: AgentState) -> AgentState:
    """Graph node wrapping Tool 2: retrieve_cfd_knowledge.

    Args:
        state: The current agent state.

    Returns:
        The updated agent state with retrieved_chunks and retrieval_quality
        populated (accumulated across retry attempts).
    """
    start = time.perf_counter()
    flow = FlowDescription.model_validate(state["flow_description"])
    retriever = CFDRetriever()

    aspects = ["solver selection", "turbulence model", "boundary conditions"]
    aspect = aspects[state["retrieval_attempts"] % len(aspects)]

    result = agent_tools.retrieve_cfd_knowledge(flow, aspect, retriever=retriever)
    state["retrieved_chunks"] = result["chunks"]
    state["retrieval_quality"] = result["quality"]
    state["retrieval_attempts"] += 1
    state["iteration_count"] += 1

    state["citations"] = [
        {
            "title": c["metadata"].get("title", "untitled"),
            "source": c["metadata"].get("source", "unknown"),
            "url": c["metadata"].get("url", ""),
            "rerank_score": c.get("rerank_score"),
            "raw_rerank_score": c.get("raw_rerank_score"),
            "match_quality": c.get("match_quality"),
        }
        for c in state["retrieved_chunks"]
    ]

    if "note" in result:
        state["reasoning_steps"].append(result["note"])
    _log_step(state, "retrieve_cfd_knowledge", time.perf_counter() - start)
    return state


def node_select_solver_and_models(state: AgentState) -> AgentState:
    """Graph node wrapping Tool 3: select_solver_and_models.

    Args:
        state: The current agent state.

    Returns:
        The updated agent state with solver_config populated.
    """
    start = time.perf_counter()
    flow = FlowDescription.model_validate(state["flow_description"])
    config = agent_tools.select_solver_and_models(flow, state["retrieved_chunks"])
    state["solver_config"] = config.model_dump(mode="json")
    state["iteration_count"] += 1
    _log_step(state, "select_solver_and_models", time.perf_counter() - start)
    return state


def node_generate_openfoam_files(state: AgentState) -> AgentState:
    """Graph node wrapping Tool 4: generate_openfoam_case_files.

    Args:
        state: The current agent state.

    Returns:
        The updated agent state with generated_files populated.
    """
    start = time.perf_counter()
    flow = FlowDescription.model_validate(state["flow_description"])
    config = SolverConfiguration.model_validate(state["solver_config"])
    state["generated_files"] = agent_tools.generate_openfoam_case_files(config, flow)
    state["iteration_count"] += 1
    _log_step(state, "generate_openfoam_files", time.perf_counter() - start)
    return state


def node_validate_physics(state: AgentState) -> AgentState:
    """Graph node wrapping Tool 5: validate_case_physics.

    Args:
        state: The current agent state.

    Returns:
        The updated agent state with validation_results populated.
    """
    start = time.perf_counter()
    flow = FlowDescription.model_validate(state["flow_description"])
    config = SolverConfiguration.model_validate(state["solver_config"])
    result = agent_tools.validate_case_physics(flow, config, state["generated_files"])
    state["validation_results"] = result.model_dump(mode="json")
    state["iteration_count"] += 1
    _log_step(state, "validate_physics", time.perf_counter() - start)
    return state


def node_format_final_response(state: AgentState) -> AgentState:
    """Graph node wrapping Tool 6: format_final_response.

    Args:
        state: The current agent state.

    Returns:
        The updated agent state with final_response populated.
    """
    start = time.perf_counter()
    flow = FlowDescription.model_validate(state["flow_description"])
    config = SolverConfiguration.model_validate(state["solver_config"])
    validation = ValidationResult.model_validate(state["validation_results"])
    state["final_response"] = agent_tools.format_final_response(
        flow, config, state["generated_files"], validation, state["citations"]
    )
    state["iteration_count"] += 1
    _log_step(state, "format_final_response", time.perf_counter() - start)
    return state


def _route_after_retrieval(state: AgentState) -> str:
    """Decide whether to retry retrieval or proceed to solver selection.

    Args:
        state: The current agent state.

    Returns:
        "retrieve_cfd_knowledge" to retry, or "select_solver_and_models" to
        proceed.
    """
    if state["iteration_count"] >= settings.MAX_AGENT_ITERATIONS:
        return "select_solver_and_models"
    if (
        state["retrieval_quality"] < settings.RETRIEVAL_QUALITY_THRESHOLD
        and state["retrieval_attempts"] < settings.MAX_RETRIEVAL_ATTEMPTS
    ):
        return "retrieve_cfd_knowledge"
    return "select_solver_and_models"


def _route_after_validation(state: AgentState) -> str:
    """Decide whether to retry solver selection or proceed to the final response.

    Args:
        state: The current agent state.

    Returns:
        "select_solver_and_models" to retry, or "format_final_response" to
        proceed.
    """
    if state["iteration_count"] >= settings.MAX_AGENT_ITERATIONS:
        return "format_final_response"
    validation = ValidationResult.model_validate(state["validation_results"])
    if not validation.passed and state["validation_retries"] < settings.MAX_VALIDATION_RETRIES:
        state["validation_retries"] += 1
        return "select_solver_and_models"
    return "format_final_response"


def build_agent_graph():
    """Construct and compile the CFD Simulation Copilot LangGraph agent.

    Returns:
        A compiled LangGraph graph (runnable via ``.invoke(initial_state(...))``).
    """
    graph = StateGraph(AgentState)

    graph.add_node("parse_flow_description", node_parse_flow_description)
    graph.add_node("retrieve_cfd_knowledge", node_retrieve_cfd_knowledge)
    graph.add_node("select_solver_and_models", node_select_solver_and_models)
    graph.add_node("generate_openfoam_files", node_generate_openfoam_files)
    graph.add_node("validate_physics", node_validate_physics)
    graph.add_node("format_final_response", node_format_final_response)

    graph.set_entry_point("parse_flow_description")
    graph.add_edge("parse_flow_description", "retrieve_cfd_knowledge")
    graph.add_conditional_edges(
        "retrieve_cfd_knowledge",
        _route_after_retrieval,
        {
            "retrieve_cfd_knowledge": "retrieve_cfd_knowledge",
            "select_solver_and_models": "select_solver_and_models",
        },
    )
    graph.add_edge("select_solver_and_models", "generate_openfoam_files")
    graph.add_edge("generate_openfoam_files", "validate_physics")
    graph.add_conditional_edges(
        "validate_physics",
        _route_after_validation,
        {
            "select_solver_and_models": "select_solver_and_models",
            "format_final_response": "format_final_response",
        },
    )
    graph.add_edge("format_final_response", END)

    return graph.compile()


def _validation_passed(validation_results: dict) -> bool | None:
    """Summarize whether a serialized ValidationResult has no critical errors.

    Args:
        validation_results: The state's validation_results dict, possibly empty.

    Returns:
        True/False if findings are present, or None if validation never ran.
    """
    findings = validation_results.get("findings")
    if findings is None:
        return None
    return not any(f.get("severity") == "error" for f in findings)


def run_agent(user_query: str) -> AgentState:
    """Run the full agent graph end-to-end for a user query.

    Every run is recorded as one structured entry in the local agent traces
    log (AGENT_TRACES_LOG_PATH), regardless of success or failure.

    Args:
        user_query: The user's natural-language CFD problem description.

    Returns:
        The final AgentState after the graph run completes.
    """
    app = build_agent_graph()
    state = initial_state(user_query)
    start = time.perf_counter()
    try:
        final_state = app.invoke(state, config={"recursion_limit": settings.MAX_AGENT_ITERATIONS * 2})
    except Exception as exc:  # noqa: BLE001
        logger.exception("Agent graph run failed.")
        state["error"] = str(exc)
        final_state = state
    latency_ms = (time.perf_counter() - start) * 1000

    solver_config = final_state.get("solver_config") or {}
    log_agent_trace(
        query=user_query,
        steps=final_state.get("reasoning_steps", []),
        latency_ms=latency_ms,
        result={
            "solver": solver_config.get("solver_name"),
            "turbulence_model": solver_config.get("turbulence_model"),
            "validation_passed": _validation_passed(final_state.get("validation_results") or {}),
            "error": final_state.get("error"),
        },
    )
    return final_state
