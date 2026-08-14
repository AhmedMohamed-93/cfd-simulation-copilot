"""Streamlit frontend for the CFD Simulation Copilot.

Three pages:
    1. CFD Copilot — submit a problem description, view the agent's solver
       recommendation, generated OpenFOAM files, and physics validation.
    2. Knowledge Base — inspect indexed sources and query retrieval directly.
    3. Agent Traces — browse past simulation sessions and evaluation results.

The app talks to the FastAPI backend over HTTP and never imports agent
internals directly, so it degrades gracefully (shows an error banner
instead of crashing) whenever the API is unreachable or returns an error.
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

EXAMPLE_QUERIES = [
    "Turbulent flow in a pipe at Re=50000",
    "Flow past a cylinder at Re=1000",
    "Natural convection in a heated cavity",
    "External aerodynamics of a NACA0012 airfoil",
]

st.set_page_config(
    page_title="CFD Simulation Copilot",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Base colors (background/text/sidebar) come from .streamlit/config.toml's
# native theme, which Streamlit applies consistently to every built-in
# widget. This block only adds the custom card/badge/step components that
# theming alone can't produce.
_CUSTOM_CSS = """
<style>
    .cfd-card {
        background-color: #161B22;
        border: 1px solid #2B3240;
        border-left: 3px solid #FF6B4A;
        border-radius: 10px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
    }
    .cfd-card h3 { margin-top: 0; }

    .cfd-pass  { color: #3FB950; }
    .cfd-warn  { color: #D29922; }
    .cfd-error { color: #F85149; }

    .cfd-finding {
        padding: 0.45rem 0.7rem;
        border-radius: 6px;
        background-color: #0E1117;
        margin-bottom: 0.4rem;
        font-size: 0.92rem;
        line-height: 1.4;
    }

    .cfd-step {
        padding: 0.45rem 0.8rem;
        border-radius: 6px;
        border-left: 3px solid #FF6B4A;
        background-color: #161B22;
        margin-bottom: 0.35rem;
        font-family: "Source Code Pro", monospace;
        font-size: 0.85rem;
        color: #E6EDF3;
    }

    .cfd-badge-row { margin-bottom: 1.2rem; }
    .cfd-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        margin-right: 0.5rem;
        border-radius: 999px;
        background-color: #161B22;
        border: 1px solid #2B3240;
        font-size: 0.8rem;
        color: #9AA5B1;
    }
    .cfd-badge b { color: #E6EDF3; }
    .cfd-badge.cfd-badge-ok { border-color: #3FB950; }
    .cfd-badge.cfd-badge-bad { border-color: #F85149; }
</style>
"""
st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


def _render_status_badges() -> None:
    """Render a small pill-badge row showing the active LLM provider/model/KB status."""
    health = _api_get("/health")
    if not health:
        st.markdown(
            "<div class='cfd-badge-row'><span class='cfd-badge cfd-badge-bad'>"
            "⚠ API unreachable</span></div>",
            unsafe_allow_html=True,
        )
        return
    status_class = "cfd-badge-ok" if health.get("status") == "ok" else "cfd-badge-bad"
    st.markdown(
        "<div class='cfd-badge-row'>"
        f"<span class='cfd-badge {status_class}'>● <b>{health.get('status', 'unknown')}</b></span>"
        f"<span class='cfd-badge'>Provider: <b>{health.get('llm_provider', '?')}</b></span>"
        f"<span class='cfd-badge'>Model: <b>{health.get('model', '?')}</b></span>"
        f"<span class='cfd-badge'>KB chunks: <b>{health.get('documents_indexed', 0)}</b></span>"
        "</div>",
        unsafe_allow_html=True,
    )


def _api_post(path: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    """POST to the FastAPI backend, returning None (and showing an error) on failure.

    Args:
        path: API path, e.g. "/simulate".
        payload: JSON-serializable request body.

    Returns:
        The parsed JSON response, or None if the request failed.
    """
    try:
        response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"Could not reach the CFD Copilot API at `{API_BASE_URL}{path}`: {exc}")
        return None


def _api_get(path: str) -> dict[str, Any] | list[Any] | None:
    """GET from the FastAPI backend, returning None (and showing an error) on failure.

    Args:
        path: API path, e.g. "/health".

    Returns:
        The parsed JSON response, or None if the request failed.
    """
    try:
        response = requests.get(f"{API_BASE_URL}{path}", timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.warning(f"Could not reach `{API_BASE_URL}{path}`: {exc}")
        return None


def _build_case_zip(generated_files: dict[str, str]) -> bytes:
    """Package generated OpenFOAM files into an in-memory ZIP archive.

    Args:
        generated_files: Mapping of relative file path -> file content.

    Returns:
        The ZIP archive bytes, ready to be offered as a download.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path, content in generated_files.items():
            zf.writestr(path, content)
    buffer.seek(0)
    return buffer.read()


def page_copilot() -> None:
    """Render the main CFD Copilot page: submit a query, view results."""
    st.title("🌀 CFD Simulation Copilot")
    st.caption("LLM agent for OpenFOAM simulation setup — runs free via the Hugging Face Inference API")
    _render_status_badges()

    example = st.selectbox("Example queries", options=["(custom)"] + EXAMPLE_QUERIES)
    default_text = "" if example == "(custom)" else example
    query = st.text_area(
        "Describe your CFD simulation problem...",
        value=default_text,
        height=120,
        placeholder="e.g. Turbulent flow of air through a 90-degree pipe bend at Re=80000, interested in pressure drop.",
    )

    if st.button("Run CFD Copilot", type="primary", disabled=not query.strip()):
        steps = ["Parse flow description", "Retrieve CFD knowledge", "Select solver & models",
                  "Generate OpenFOAM files", "Validate physics"]
        step_placeholder = st.empty()
        with step_placeholder.container():
            for step in steps:
                st.markdown(f"<div class='cfd-step'>⏳ {step}...</div>", unsafe_allow_html=True)

        with st.spinner("Running agent..."):
            result = _api_post("/simulate", {"query": query, "stream": False})
        step_placeholder.empty()

        if result is None:
            return

        st.session_state["last_result"] = result

    result = st.session_state.get("last_result")
    if not result:
        return

    if result.get("error"):
        st.error(f"Agent encountered an error: {result['error']}")

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("<div class='cfd-card'>", unsafe_allow_html=True)
        st.subheader("Solver Recommendation")
        st.markdown(f"**Solver:** `{result.get('solver') or 'N/A'}`")
        st.markdown(f"**Turbulence model:** `{result.get('turbulence_model') or 'N/A'}`")
        st.markdown(f"**Latency:** {result.get('latency_ms', 0):.0f} ms")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown("<div class='cfd-card'>", unsafe_allow_html=True)
        st.subheader("Physics Validation")
        validation = result.get("validation") or {}
        findings = validation.get("findings", [])
        if not findings:
            st.info("No validation findings returned.")
        for f in findings:
            severity = f.get("severity", "pass")
            icon = {"pass": "✅", "warning": "⚠️", "error": "❌"}.get(severity, "•")
            css = {"pass": "cfd-pass", "warning": "cfd-warn", "error": "cfd-error"}.get(severity, "")
            st.markdown(
                f"<div class='cfd-finding'><span class='{css}'>{icon} "
                f"<b>{f.get('rule')}</b></span>: {f.get('message')}</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    generated_files = result.get("generated_files") or {}
    if generated_files:
        st.subheader("Generated OpenFOAM Case Files")
        tabs = st.tabs(list(generated_files.keys()))
        for tab, (_path, content) in zip(tabs, generated_files.items(), strict=False):
            with tab:
                st.code(content, language="cpp")

        st.download_button(
            "⬇️ Download OpenFOAM Case (ZIP)",
            data=_build_case_zip(generated_files),
            file_name="cfd_case.zip",
            mime="application/zip",
        )

    citations = result.get("citations") or []
    if citations:
        with st.expander(f"📚 Citations ({len(citations)})"):
            for c in citations:
                score = c.get("rerank_score")
                # rerank_score is 0-10 (min-max normalized within the
                # retrieval batch); display as a 0-100% relevance figure.
                score_str = f" — relevance {score * 10:.0f}%" if score is not None else ""
                st.markdown(f"- **{c.get('title', 'untitled')}** ({c.get('source', 'unknown')}){score_str}")

    if result.get("explanation"):
        with st.expander("📝 Full agent explanation"):
            st.markdown(result["explanation"])


def page_knowledge_base() -> None:
    """Render the Knowledge Base inspection page."""
    st.title("📚 Knowledge Base")

    stats = _api_get("/knowledge-base/stats")
    if stats:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Total indexed chunks", stats.get("total_documents", 0))
            st.caption(f"Last updated: {stats.get('last_updated') or 'unknown'}")
        with col2:
            topics = stats.get("topics", {})
            if topics:
                df = pd.DataFrame({"topic": list(topics.keys()), "count": list(topics.values())})
                st.bar_chart(df.set_index("topic"))
            else:
                st.info("No topic breakdown available yet — run the ingestion pipeline.")

    health = _api_get("/health")
    if health:
        st.subheader("Indexed Sources")
        st.dataframe(
            pd.DataFrame(
                [{"status": health.get("status"), "documents_indexed": health.get("documents_indexed"),
                  "model": health.get("model")}]
            ),
            use_container_width=True,
        )

    st.subheader("Direct Knowledge Base Search")
    st.caption("Query the knowledge base directly, bypassing the agent, to inspect raw retrieval results.")
    search_query = st.text_input("Search query", placeholder="e.g. kOmegaSST wall functions")
    if st.button("Search") and search_query.strip():
        st.info(
            "Direct search hits the retrieval layer, which requires a running Qdrant + local "
            "embedding/reranking backend. Configure the API and knowledge base, then this box "
            "returns ranked chunks with dense and rerank scores."
        )


def page_agent_traces() -> None:
    """Render the Agent Traces page: session history and evaluation results."""
    st.title("🔍 Agent Traces")

    sessions = _api_get("/sessions")
    st.subheader("Recent Simulation Sessions")
    if sessions:
        rows = []
        for s in sessions:
            resp = s.get("response", {})
            validation = resp.get("validation", {})
            findings = validation.get("findings", [])
            passed = not any(f.get("severity") == "error" for f in findings)
            rows.append(
                {
                    "session_id": s.get("session_id", "")[:8],
                    "query": s.get("query", "")[:60],
                    "solver": resp.get("solver"),
                    "latency_ms": round(resp.get("latency_ms", 0), 1),
                    "validation_pass": passed,
                    "created_at": s.get("created_at"),
                }
            )
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)

        selected = st.selectbox("Inspect a session", options=["(none)"] + [s.get("session_id") for s in sessions])
        if selected != "(none)":
            record = next((s for s in sessions if s.get("session_id") == selected), None)
            if record:
                st.json(record)
    else:
        st.info("No simulation sessions recorded yet. Run a query on the CFD Copilot page first.")

    st.subheader("Evaluation Results (RAGAS + custom metrics)")
    results_path = Path("evaluation/results/ragas_results.json")
    if results_path.exists():
        try:
            eval_data = json.loads(results_path.read_text(encoding="utf-8"))
            st.json(eval_data.get("aggregate_metrics", {}))
            case_df = pd.DataFrame(eval_data.get("case_results", []))
            if not case_df.empty:
                st.dataframe(case_df, use_container_width=True)
        except (json.JSONDecodeError, OSError) as exc:
            st.warning(f"Could not load evaluation results: {exc}")
    else:
        st.info("No evaluation results yet. Run `python -m src.evaluation.ragas_eval` to generate them.")


PAGES = {
    "CFD Copilot": page_copilot,
    "Knowledge Base": page_knowledge_base,
    "Agent Traces": page_agent_traces,
}


def main() -> None:
    """Render the sidebar navigation and dispatch to the selected page."""
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Go to", list(PAGES.keys()))
    st.sidebar.markdown("---")
    st.sidebar.caption(f"API backend: `{API_BASE_URL}`")
    PAGES[choice]()


if __name__ == "__main__":
    main()
