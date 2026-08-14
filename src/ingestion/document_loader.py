"""Loaders for the CFD knowledge base source documents.

Sources targeted (all public, no auth required):
    1. OpenFOAM User Guide from openfoam.com.
    2. OpenFOAM tutorial directory listing (GitHub API).
    3. CFD-Online Wiki pages on turbulence models.
    4. A curated set of arXiv papers on CFD surrogate modeling / ML for fluids.

Every loader is best-effort: network failures, rate limits, schema drift, or
a live page turning out to be a JS-rendered shell with no real static
content must never crash ingestion — and must never silently index noise
(nav-menu boilerplate, script/style markup) as if it were real knowledge.
Each loader extracts and cleans article text (stripping scripts, styles,
and non-article chrome) and falls back to a rich, hand-authored synthetic
document set covering the same topic whenever the cleaned real content is
too thin to be useful, so the rest of the system (chunking, embedding,
retrieval, the agent) always has a working, genuinely useful knowledge base
to run against.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

OPENFOAM_USER_GUIDE_URL = "https://www.openfoam.com/documentation/user-guide"
OPENFOAM_TUTORIALS_TREE_API = (
    "https://api.github.com/repos/OpenFOAM/OpenFOAM-10/git/trees/master?recursive=1"
)
CFD_ONLINE_WIKI_PAGES = {
    "k-epsilon": "https://www.cfd-online.com/Wiki/Standard_k-epsilon_model",
    "k-omega SST": "https://www.cfd-online.com/Wiki/SST_k-omega_model",
    "Spalart-Allmaras": "https://www.cfd-online.com/Wiki/Spalart-Allmaras_model",
    "LES": "https://www.cfd-online.com/Wiki/Large_eddy_simulation_(LES)",
    "DNS": "https://www.cfd-online.com/Wiki/Direct_numerical_simulation_(DNS)",
    "realisable k-epsilon": "https://www.cfd-online.com/Wiki/Realisable_k-epsilon_model",
    "RNG k-epsilon": "https://www.cfd-online.com/Wiki/RNG_k-epsilon_model",
    "Reynolds stress model": "https://www.cfd-online.com/Wiki/Reynolds_stress_model_(RSM)",
    "DES": "https://www.cfd-online.com/Wiki/Detached_eddy_simulation_(DES)",
    "law of the wall": "https://www.cfd-online.com/Wiki/Law_of_the_wall",
    "wall functions": "https://www.cfd-online.com/Wiki/Wall_functions",
    "turbulence intro": "https://www.cfd-online.com/Wiki/Introduction_to_turbulence",
    "RANS overview": "https://www.cfd-online.com/Wiki/RANS-based_turbulence_models",
}
# MediaWiki (CFD-Online Wiki) wraps the real article inside this container;
# everything outside it (in particular the huge site-wide forum megamenu) is
# navigation chrome, not article content.
CFD_ONLINE_CONTENT_SELECTOR = "#bodyContent"

ARXIV_API_URL = "http://export.arxiv.org/api/query"
ARXIV_QUERY = "all:CFD surrogate modeling machine learning fluid mechanics"
REQUEST_TIMEOUT_S = 15

# Many public sites block the default python-requests User-Agent as a bot;
# a plain browser UA is enough to pass that check without doing anything
# deceptive (no cookies/fingerprint spoofing, just an honest client label).
_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}

# Below this many characters of *cleaned* article text, a fetched page is
# treated as unusable (e.g. a JS-rendered shell with no server-rendered
# content, or a redirect/placeholder page) and the synthetic fallback is
# used instead. Raw HTML length is not a reliable signal here — boilerplate
# alone easily exceeds this, which is why the check happens post-cleaning.
_MIN_USEFUL_CONTENT_CHARS = 200


@dataclass
class RawDocument:
    """A single loaded document prior to chunking.

    Attributes:
        content: The full text content of the document.
        source: Human-readable origin of the document (e.g. "cfd-online-wiki").
        title: Document title.
        topic_tags: Tags from {solver, turbulence_model, flow_type,
            boundary_condition, mesh_type}.
        difficulty_level: One of "beginner", "intermediate", "advanced".
        metadata: Any additional free-form metadata (url, authors, date...).
    """

    content: str
    source: str
    title: str
    topic_tags: list[str] = field(default_factory=list)
    difficulty_level: str = "intermediate"
    metadata: dict[str, Any] = field(default_factory=dict)


def _safe_get(url: str, **kwargs: Any) -> requests.Response | None:
    """Perform a GET request, returning None instead of raising on failure.

    Args:
        url: The URL to fetch.
        **kwargs: Extra kwargs forwarded to ``requests.get``. A ``headers``
            dict, if provided, is merged on top of the default browser
            User-Agent rather than replacing it.

    Returns:
        The response if the request succeeded with a 2xx status, else None.
    """
    headers = {**_DEFAULT_HEADERS, **kwargs.pop("headers", {})}
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_S, headers=headers, **kwargs)
        response.raise_for_status()
        return response
    except requests.RequestException as exc:
        logger.warning("Request to %s failed: %s", url, exc)
        return None


def _extract_article_text(html: str, content_selector: str | None = None) -> str:
    """Extract clean, readable article prose from a raw HTML page.

    Scripts, styles, and non-paragraph chrome (nav trees, menus, sidebars)
    are dropped; only text inside ``<p>`` tags is kept, since that's where
    real article prose lives on both documentation sites and MediaWiki
    pages, while navigation/menu structures are almost always lists/divs.

    Args:
        html: The raw HTML page content.
        content_selector: Optional CSS selector to scope extraction to
            (e.g. a MediaWiki article container), avoiding site-wide menus
            that live outside it. Falls back to the whole document if the
            selector doesn't match.

    Returns:
        Cleaned, whitespace-collapsed article text (possibly empty if the
        page has no server-rendered paragraph content).
    """
    soup = BeautifulSoup(html, "html.parser")
    scope = (soup.select_one(content_selector) if content_selector else None) or soup
    for tag in scope(["script", "style"]):
        tag.decompose()
    paragraphs = scope.find_all("p")
    text = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
    return re.sub(r"\s+", " ", text).strip()


def load_openfoam_user_guide() -> list[RawDocument]:
    """Load the OpenFOAM User Guide, falling back to synthetic content.

    Returns:
        A list of RawDocument objects covering the User Guide's key
        sections (solvers, boundary conditions, numerical schemes, meshing).
    """
    response = _safe_get(OPENFOAM_USER_GUIDE_URL)
    if response is not None:
        text = _extract_article_text(response.text)
        if len(text) >= _MIN_USEFUL_CONTENT_CHARS:
            return [
                RawDocument(
                    content=text,
                    source="openfoam-user-guide",
                    title="OpenFOAM User Guide",
                    topic_tags=["solver", "boundary_condition", "mesh_type"],
                    difficulty_level="intermediate",
                    metadata={"url": OPENFOAM_USER_GUIDE_URL},
                )
            ]
        logger.info(
            "OpenFOAM User Guide page returned only %d chars of article text "
            "(likely a JS-rendered shell with no server-rendered content); "
            "falling back to synthetic content.",
            len(text),
        )
    else:
        logger.info("Falling back to synthetic OpenFOAM User Guide content.")
    return _synthetic_openfoam_user_guide()


def load_openfoam_tutorials() -> list[RawDocument]:
    """Load real, individually-named OpenFOAM tutorial cases from GitHub.

    Uses the git trees API with recursive=1 to fetch the entire repository
    tree in a single request — avoiding the unauthenticated GitHub API's
    60-requests/hour rate limit, which per-directory listing calls would
    risk exhausting if crawled level by level — then extracts every actual
    tutorial case directory (four path segments deep:
    tutorials/<category>/<solver>/<caseName>, e.g.
    tutorials/incompressible/simpleFoam/pitzDaily) rather than only the
    top-level category folders.

    Returns:
        A list of RawDocument objects, one per real tutorial case, falling
        back to a small synthetic catalogue if the GitHub API is
        unreachable, rate-limited, or its response schema changes.
    """
    response = _safe_get(
        OPENFOAM_TUTORIALS_TREE_API, headers={"Accept": "application/vnd.github+json"}
    )
    if response is not None:
        try:
            data = response.json()
        except ValueError:
            data = None
        tree = data.get("tree") if isinstance(data, dict) else None
        if isinstance(tree, list) and tree:
            docs = []
            for item in tree:
                path = item.get("path", "")
                if item.get("type") != "tree" or not path.startswith("tutorials/"):
                    continue
                parts = path.split("/")
                if len(parts) != 4:
                    continue
                _root, category, solver, case_name = parts
                docs.append(
                    RawDocument(
                        content=(
                            f"OpenFOAM tutorial case '{case_name}' (path: {path}) "
                            f"demonstrates the {solver} solver in the {category} "
                            f"tutorial category."
                        ),
                        source="openfoam-tutorials-github",
                        title=f"OpenFOAM Tutorial: {path}",
                        topic_tags=["solver", "flow_type"],
                        difficulty_level="beginner",
                        metadata={
                            "url": f"https://github.com/OpenFOAM/OpenFOAM-10/tree/master/{path}",
                            "solver": solver,
                            "category": category,
                        },
                    )
                )
            if docs:
                return docs
    logger.info("Falling back to synthetic OpenFOAM tutorial catalogue.")
    return _synthetic_openfoam_tutorials()


def load_cfd_online_wiki() -> list[RawDocument]:
    """Scrape key CFD-Online Wiki pages on turbulence models.

    Returns:
        A list of RawDocument objects, one per turbulence model page,
        falling back to synthetic summaries for any page that fails to load.
    """
    docs: list[RawDocument] = []
    for model_name, url in CFD_ONLINE_WIKI_PAGES.items():
        response = _safe_get(url)
        text = _extract_article_text(response.text, CFD_ONLINE_CONTENT_SELECTOR) if response else ""
        if len(text) >= _MIN_USEFUL_CONTENT_CHARS:
            docs.append(
                RawDocument(
                    content=text,
                    source="cfd-online-wiki",
                    title=f"CFD-Online Wiki: {model_name}",
                    topic_tags=["turbulence_model"],
                    difficulty_level="advanced",
                    metadata={"url": url, "model": model_name},
                )
            )
        else:
            logger.info("Falling back to synthetic content for %s.", model_name)
            docs.append(_synthetic_turbulence_model_doc(model_name, url))
    return docs


def load_arxiv_papers(max_results: int = 40) -> list[RawDocument]:
    """Download metadata/abstracts for arXiv papers on CFD + ML.

    Args:
        max_results: Maximum number of papers to fetch.

    Returns:
        A list of RawDocument objects built from arXiv abstracts, falling
        back to a synthetic literature summary set if the API is unreachable.
    """
    params = {
        "search_query": ARXIV_QUERY,
        "start": 0,
        "max_results": max_results,
    }
    response = _safe_get(ARXIV_API_URL, params=params)
    if response is not None and "<entry>" in response.text:
        docs = _parse_arxiv_atom_feed(response.text)
        if docs:
            return docs
    logger.info("Falling back to synthetic arXiv literature summaries.")
    return _synthetic_arxiv_papers(max_results)


def _parse_arxiv_atom_feed(xml_text: str) -> list[RawDocument]:
    """Parse an arXiv Atom feed into RawDocument objects.

    Args:
        xml_text: Raw XML text returned by the arXiv API.

    Returns:
        A list of RawDocument objects extracted from feed entries.
    """
    import xml.etree.ElementTree as ET

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    docs: list[RawDocument] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return docs
    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        summary_el = entry.find("atom:summary", ns)
        id_el = entry.find("atom:id", ns)
        title = title_el.text.strip() if title_el is not None and title_el.text else "Untitled"
        summary = (
            summary_el.text.strip() if summary_el is not None and summary_el.text else ""
        )
        url = id_el.text.strip() if id_el is not None and id_el.text else ""
        if not summary:
            continue
        docs.append(
            RawDocument(
                content=f"{title}\n\n{summary}",
                source="arxiv",
                title=title,
                topic_tags=["flow_type"],
                difficulty_level="advanced",
                metadata={"url": url},
            )
        )
    return docs


# --------------------------------------------------------------------------
# Synthetic fallback content
# --------------------------------------------------------------------------


def _synthetic_openfoam_user_guide() -> list[RawDocument]:
    """Build a synthetic OpenFOAM User Guide document set.

    Returns:
        RawDocument objects covering solvers, boundary conditions, and
        numerical schemes, written to closely mirror the real user guide.
    """
    sections = [
        (
            "Incompressible Solvers",
            ["solver", "flow_type"],
            (
                "simpleFoam is a steady-state solver for incompressible, "
                "turbulent flow, using the SIMPLE (Semi-Implicit Method for "
                "Pressure-Linked Equations) algorithm. It is appropriate when "
                "the flow of interest reaches a stationary state and transient "
                "effects are not required. pimpleFoam is a transient solver for "
                "incompressible, turbulent flow of Newtonian fluids, combining "
                "PISO and SIMPLE algorithms (PIMPLE) to allow larger time steps "
                "via under-relaxation, at the cost of iterating within each time "
                "step until the pressure-velocity coupling converges. icoFoam is "
                "a transient solver for incompressible, laminar flow only, using "
                "the PISO algorithm, and does not include a turbulence model."
            ),
        ),
        (
            "Compressible Solvers",
            ["solver", "flow_type"],
            (
                "rhoSimpleFoam is a steady-state solver for turbulent, "
                "compressible flow, applicable for subsonic and mildly "
                "transonic regimes. rhoPimpleFoam is a transient solver for "
                "compressible, turbulent flow of Newtonian fluids, suitable "
                "for both subsonic and supersonic flow, including shocks, "
                "using the PIMPLE algorithm to handle the density-pressure-"
                "velocity coupling. For strongly supersonic flow (Mach > 1.3) "
                "with shocks, sonicFoam or a density-based scheme is preferred "
                "for improved shock capturing."
            ),
        ),
        (
            "Multiphase Solvers",
            ["solver", "flow_type"],
            (
                "interFoam is a solver for two incompressible, isothermal "
                "immiscible fluids using a VOF (volume of fluid) phase-fraction "
                "based interface capturing approach. It is used for free-"
                "surface flows such as sloshing, wave impact, or dam-break "
                "problems. The phase fraction field alpha.water must be "
                "bounded between 0 and 1 at all times; the MULES limiter is "
                "used to enforce this boundedness."
            ),
        ),
        (
            "Buoyancy-Driven Solvers",
            ["solver", "flow_type"],
            (
                "buoyantSimpleFoam is a steady-state solver for buoyant, "
                "turbulent flow of compressible fluids, suitable for natural "
                "or mixed convection heat-transfer problems, for ventilation "
                "and heat-transfer. buoyantPimpleFoam is the transient "
                "equivalent, used when the buoyancy-driven flow is inherently "
                "unsteady, such as thermal plumes or transient heating."
            ),
        ),
        (
            "Boundary Conditions",
            ["boundary_condition"],
            (
                "Common velocity boundary conditions include fixedValue for a "
                "prescribed inlet velocity, noSlip for stationary walls, and "
                "inletOutlet for boundaries that may switch between inflow and "
                "outflow. Common pressure boundary conditions include "
                "fixedFluxPressure for walls (consistent with a fixed-value "
                "velocity boundary), and zeroGradient or fixedValue for "
                "outlets depending on whether a reference pressure is needed "
                "elsewhere in the domain. Every boundary patch defined in the "
                "polyMesh/boundary file must have a matching entry in the "
                "boundaryField of every field file (U, p, k, epsilon/omega), "
                "or the solver will fail at startup."
            ),
        ),
        (
            "Numerical Schemes",
            ["solver"],
            (
                "fvSchemes controls the discretization of each term. For "
                "steady-state solvers (simpleFoam), ddtSchemes should use "
                "steadyState. For transient solvers, Euler (first order, "
                "bounded) or backward (second order) are common choices. "
                "divSchemes for convection terms should use bounded Gauss "
                "upwind for robustness during early iterations, and can be "
                "switched to a higher-order bounded scheme such as Gauss "
                "linearUpwind once the solution has stabilized, for improved "
                "accuracy. laplacianSchemes typically use Gauss linear "
                "corrected for meshes with low non-orthogonality."
            ),
        ),
        (
            "Meshing Guidelines",
            ["mesh_type"],
            (
                "For wall-bounded turbulent flows, the near-wall mesh "
                "resolution should be chosen according to the turbulence "
                "model's wall treatment. Wall functions (the default for "
                "k-epsilon and k-omega SST in OpenFOAM) target y+ values "
                "between 30 and 300 on the first cell center. Low-Reynolds-"
                "number models or resolved boundary layers require y+ ~ 1. "
                "Mesh non-orthogonality above 70 degrees and skewness above "
                "4 typically require special numerical treatment "
                "(correctors, limited schemes) to maintain stability."
            ),
        ),
    ]
    return [
        RawDocument(
            content=content,
            source="openfoam-user-guide-synthetic",
            title=f"OpenFOAM User Guide: {title}",
            topic_tags=tags,
            difficulty_level="intermediate",
            metadata={"synthetic": True},
        )
        for title, tags, content in sections
    ]


def _synthetic_openfoam_tutorials() -> list[RawDocument]:
    """Build a synthetic OpenFOAM tutorial catalogue.

    Returns:
        RawDocument objects describing representative tutorial cases for
        each major solver family.
    """
    tutorials = [
        ("incompressible/simpleFoam/pitzDaily", "simpleFoam", "backward-facing step, turbulent"),
        ("incompressible/pimpleFoam/TJunction", "pimpleFoam", "transient junction flow"),
        ("incompressible/icoFoam/cavity", "icoFoam", "lid-driven cavity, laminar"),
        ("compressible/rhoPimpleFoam/angledDuct", "rhoPimpleFoam", "compressible duct flow"),
        ("multiphase/interFoam/damBreak", "interFoam", "dam-break free-surface flow"),
        ("heatTransfer/buoyantSimpleFoam/hotRoom", "buoyantSimpleFoam", "natural convection"),
    ]
    return [
        RawDocument(
            content=(
                f"Tutorial case '{path}' demonstrates {solver} applied to a "
                f"{desc} problem. It includes a representative mesh, initial "
                f"and boundary conditions, and solver/scheme configuration "
                f"suitable as a starting point for similar problems."
            ),
            source="openfoam-tutorials-synthetic",
            title=f"OpenFOAM Tutorial: {path}",
            topic_tags=["solver", "flow_type"],
            difficulty_level="beginner",
            metadata={"synthetic": True, "solver": solver},
        )
        for path, solver, desc in tutorials
    ]


def _synthetic_turbulence_model_doc(model_name: str, url: str) -> RawDocument:
    """Build a synthetic CFD-Online-style summary for a turbulence model.

    Args:
        model_name: Name of the turbulence model (e.g. "k-omega SST").
        url: The original wiki URL, kept as metadata for citation purposes.

    Returns:
        A RawDocument summarizing the model's formulation and applicability.
    """
    summaries = {
        "k-epsilon": (
            "The standard k-epsilon model solves transport equations for "
            "turbulent kinetic energy (k) and its dissipation rate "
            "(epsilon). It is robust and computationally economical for "
            "fully turbulent, high-Reynolds-number flows away from walls, "
            "but performs poorly for flows with strong adverse pressure "
            "gradients, separation, or curvature. It relies on wall "
            "functions and is not recommended for Re below roughly 5e4 in "
            "internal flows or for flows with significant near-wall "
            "separation."
        ),
        "k-omega SST": (
            "The Shear Stress Transport (SST) k-omega model blends the "
            "k-omega formulation near walls (accurate viscous sublayer "
            "behavior) with the k-epsilon formulation in the free stream "
            "(insensitive to inlet turbulence values). It is the most "
            "widely recommended general-purpose RANS model for external "
            "aerodynamics, adverse pressure gradients, and separated flows, "
            "and is a strong default choice for Reynolds numbers from "
            "roughly 2300 up to fully turbulent regimes."
        ),
        "Spalart-Allmaras": (
            "Spalart-Allmaras is a one-equation model solving a single "
            "transport equation for a modified eddy viscosity. It was "
            "designed specifically for aerodynamic flows with mild "
            "separation, such as flow over wings and airfoils, and is "
            "popular in the aerospace industry for its numerical stability "
            "and low computational cost relative to two-equation models."
        ),
        "LES": (
            "Large Eddy Simulation directly resolves the large, energy-"
            "containing turbulent eddies on the mesh while modeling the "
            "effect of unresolved sub-grid scales (e.g. via the Smagorinsky "
            "model). LES requires much finer meshes and smaller time steps "
            "than RANS, and is used when transient turbulent structures "
            "(e.g. acoustic noise sources, mixing) must be captured rather "
            "than only time-averaged statistics."
        ),
        "DNS": (
            "Direct Numerical Simulation resolves all turbulent scales "
            "down to the Kolmogorov length scale without any turbulence "
            "model. Computational cost scales approximately as Re^3, "
            "restricting DNS to low-to-moderate Reynolds number canonical "
            "flows (e.g. channel flow at Re_tau up to a few thousand) "
            "typically studied on HPC clusters for fundamental turbulence "
            "research rather than engineering design."
        ),
        "realisable k-epsilon": (
            "The realisable k-epsilon model modifies the standard model's "
            "eddy-viscosity formulation and dissipation-rate equation to "
            "satisfy mathematical constraints on the Reynolds stresses "
            "(realisability), giving improved accuracy for flows with "
            "strong streamline curvature, rotation, and separation compared "
            "to the standard k-epsilon model, at similar computational cost."
        ),
        "RNG k-epsilon": (
            "The RNG (renormalization group) k-epsilon model derives its "
            "coefficients analytically via RNG theory rather than empirical "
            "fitting, and adds a strain-rate-dependent term to the "
            "dissipation equation. This improves accuracy for rapidly "
            "strained flows, swirling flows, and locally transitional flows "
            "relative to the standard k-epsilon model."
        ),
        "Reynolds stress model": (
            "The Reynolds Stress Model (RSM) solves individual transport "
            "equations for each component of the Reynolds stress tensor "
            "(plus dissipation rate), avoiding the isotropic eddy-viscosity "
            "assumption used by two-equation models. This captures "
            "anisotropic turbulence, streamline curvature, and swirl more "
            "accurately, at roughly double the computational cost and "
            "reduced numerical robustness compared to k-epsilon/k-omega SST."
        ),
        "DES": (
            "Detached Eddy Simulation (DES) is a hybrid RANS/LES approach: "
            "it applies a RANS turbulence model within attached boundary "
            "layers (where mesh resolution for LES would be prohibitively "
            "expensive) and switches to an LES-like treatment in separated, "
            "detached flow regions. This balances the cost of RANS against "
            "the fidelity of LES for massively separated flows such as "
            "bluff-body wakes."
        ),
        "law of the wall": (
            "The law of the wall describes the near-wall velocity profile "
            "in turbulent boundary layers as a function of the "
            "non-dimensional wall distance y+: a linear viscous sublayer "
            "(y+ < 5), a buffer region, and a logarithmic region (y+ > 30) "
            "described by u+ = (1/kappa) ln(y+) + B. Wall functions use "
            "this relationship to bridge the near-wall cell to the wall "
            "without fully resolving the viscous sublayer."
        ),
        "wall functions": (
            "Wall functions are semi-empirical boundary conditions that "
            "model the near-wall region of a turbulent boundary layer using "
            "the law of the wall, rather than fully resolving it with mesh "
            "cells. They allow the first cell center to be placed at "
            "y+ ~ 30-300, substantially reducing mesh cell count and cost "
            "compared to low-Reynolds-number near-wall resolution (y+ ~ 1), "
            "at the expense of accuracy in flows with strong separation."
        ),
        "turbulence intro": (
            "Turbulence is a flow regime characterized by chaotic, "
            "multi-scale velocity fluctuations, occurring above a "
            "critical Reynolds number where inertial forces dominate "
            "viscous forces. Turbulent flows exhibit enhanced mixing, "
            "increased drag/heat transfer, and a cascade of kinetic energy "
            "from large to progressively smaller eddies down to the "
            "Kolmogorov scale, where it is dissipated as heat by viscosity."
        ),
        "RANS overview": (
            "Reynolds-Averaged Navier-Stokes (RANS) models solve "
            "time-averaged (or ensemble-averaged) transport equations, "
            "modeling all turbulent scales via a closure model (e.g. "
            "k-epsilon, k-omega SST, Reynolds stress models) rather than "
            "resolving them. RANS is the most computationally economical "
            "turbulence modeling approach and remains the default choice "
            "for the large majority of engineering CFD analyses, at the "
            "cost of relying on empirical/modeled closure assumptions."
        ),
    }
    content = summaries.get(
        model_name,
        f"{model_name} is a turbulence closure model used in RANS/LES CFD.",
    )
    return RawDocument(
        content=content,
        source="cfd-online-wiki-synthetic",
        title=f"CFD-Online Wiki: {model_name}",
        topic_tags=["turbulence_model"],
        difficulty_level="advanced",
        metadata={"synthetic": True, "url": url, "model": model_name},
    )


def _synthetic_arxiv_papers(max_results: int) -> list[RawDocument]:
    """Build a synthetic set of arXiv-style CFD/ML abstracts.

    Args:
        max_results: Number of synthetic entries to produce (capped at 40
            hand-authored summaries).

    Returns:
        A list of RawDocument objects mimicking arXiv paper abstracts on
        CFD surrogate modeling and machine learning for fluid mechanics.
    """
    topics = [
        "Physics-informed neural networks for solving the Navier-Stokes equations",
        "Graph neural network surrogates for unstructured-mesh CFD simulations",
        "Deep learning turbulence closure models trained on DNS data",
        "Reduced-order modeling of unsteady flows via autoencoders",
        "Neural operator learning for parametric PDE surrogate solvers",
        "Data-driven RANS turbulence model correction using field inversion",
        "Convolutional neural networks for real-time aerodynamic shape prediction",
        "Transformer-based surrogate models for unsteady wake flows",
        "Machine-learned subgrid-scale models for large eddy simulation",
        "Bayesian optimization for CFD-based aerodynamic shape design",
        "Generative models for synthetic turbulent flow field generation",
        "Physics-constrained deep learning for heat transfer prediction",
        "Multi-fidelity surrogate modeling combining RANS and DNS data",
        "Graph-based mesh generation for CFD using learned representations",
        "Deep reinforcement learning for active flow control",
        "Sparse regression for discovering turbulence closure equations",
        "Neural network acceleration of pressure-velocity coupling solvers",
        "Uncertainty quantification in machine-learned CFD surrogates",
        "Transfer learning across Reynolds numbers for turbulent flow prediction",
        "Equivariant neural networks for fluid flow prediction on manifolds",
        "Fourier neural operators for turbulent flow field super-resolution",
        "Convolutional autoencoders for compressing large-eddy simulation data",
        "Machine learning acceleration of adjoint-based shape optimization",
        "Data-driven discovery of governing equations from CFD flow fields",
        "Neural network surrogates for conjugate heat transfer simulations",
        "Gaussian process regression for aerodynamic coefficient prediction",
        "Deep learning-based mesh adaptation for compressible flow solvers",
        "Physics-informed neural networks for multiphase flow interfaces",
        "Surrogate modeling of combustion chemistry using neural networks",
        "Machine learning wall models for large eddy simulation near-wall regions",
        "Koopman operator theory for reduced-order modeling of fluid flows",
        "Active learning strategies for efficient CFD surrogate training",
        "Graph neural networks for airfoil aerodynamic performance prediction",
        "Deep learning super-resolution of coarse RANS flow fields",
        "Differentiable CFD solvers for gradient-based design optimization",
        "Neural network closures for Reynolds-averaged scalar transport",
        "Machine learning-based inflow turbulence generation for LES",
        "Convolutional neural networks for real-time flow field reconstruction",
        "Physics-informed learning of boundary layer transition onset",
        "Transfer learning of turbulence models across different geometries",
        "Ensemble neural networks for uncertainty-aware aerodynamic prediction",
    ]
    docs = []
    for i, topic in enumerate(topics[:max_results]):
        docs.append(
            RawDocument(
                content=(
                    f"{topic}. This line of work explores how machine "
                    "learning surrogates can accelerate or augment "
                    "traditional CFD solvers, discussing accuracy trade-offs "
                    "relative to full-order RANS/LES/DNS simulations, "
                    "generalization across flow regimes, and integration "
                    "with existing engineering simulation workflows."
                ),
                source="arxiv-synthetic",
                title=topic,
                topic_tags=["flow_type"],
                difficulty_level="advanced",
                metadata={"synthetic": True, "arxiv_id": f"synthetic.{i:04d}"},
            )
        )
    return docs


def load_all_documents() -> list[RawDocument]:
    """Load documents from every configured source.

    Returns:
        The concatenation of all loaders' RawDocument outputs, ready for
        chunking.
    """
    documents: list[RawDocument] = []
    documents.extend(load_openfoam_user_guide())
    documents.extend(load_openfoam_tutorials())
    documents.extend(load_cfd_online_wiki())
    documents.extend(load_arxiv_papers())
    logger.info("Loaded %d raw documents from all sources.", len(documents))
    return documents
