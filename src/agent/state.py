"""LangGraph agent state schema for the CFD Simulation Copilot."""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict):
    """Shared mutable state threaded through every node of the agent graph.

    Attributes:
        user_query: The original natural-language question from the user.
        flow_description: Parsed flow parameters (serialized FlowDescription).
        retrieved_chunks: RAG retrieval results, one dict per chunk with
            content, metadata, dense_score, rerank_score (batch-relative,
            0-10), raw_rerank_score (the pre-normalization cross-encoder
            logit, or None), and match_quality ("strong"/"moderate"/"weak"
            absolute band derived from raw_rerank_score, or None).
        retrieval_quality: Self-graded retrieval quality score in [0, 1].
        reasoning_steps: A running trace of the agent's reasoning, one
            human-readable string per step, used for observability/UI.
        generated_files: Mapping of OpenFOAM file path -> file content.
        validation_results: Serialized ValidationResult from physics_validator.
        final_response: The formatted final markdown answer for the user.
        citations: Source citations, one dict per cited chunk (title, source,
            url, rerank_score, raw_rerank_score, match_quality).
        iteration_count: Total number of agent graph steps taken so far.
        retrieval_attempts: Number of retrieval attempts made so far.
        validation_retries: Number of regeneration retries after a failed
            validation, so far.
        solver_config: Serialized SolverConfiguration once selected.
        error: Error message if any node failed unrecoverably, else None.
    """

    user_query: str
    flow_description: dict[str, Any]
    retrieved_chunks: list[dict[str, Any]]
    retrieval_quality: float
    reasoning_steps: list[str]
    generated_files: dict[str, str]
    validation_results: dict[str, Any]
    final_response: str
    citations: list[dict[str, Any]]
    iteration_count: int
    retrieval_attempts: int
    validation_retries: int
    solver_config: dict[str, Any]
    error: str | None


def initial_state(user_query: str) -> AgentState:
    """Build the initial agent state for a new simulation request.

    Args:
        user_query: The user's natural-language simulation description.

    Returns:
        A fully initialized AgentState with empty/default fields.
    """
    return AgentState(
        user_query=user_query,
        flow_description={},
        retrieved_chunks=[],
        retrieval_quality=0.0,
        reasoning_steps=[],
        generated_files={},
        validation_results={},
        final_response="",
        citations=[],
        iteration_count=0,
        retrieval_attempts=0,
        validation_retries=0,
        solver_config={},
        error=None,
    )
