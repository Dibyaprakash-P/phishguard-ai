"""Optional LangGraph security-analysis agent.

This is an **advanced, opt-in** orchestration path. It expresses the same
analysis as ``routes/analyze.py`` as an explicit state machine:

    validate -> extract_features -> ml_predict -> assess_indicators -> explain

Why bother, when a linear function already works? Because a graph makes the
pipeline inspectable and conditionally routable: the ``assess_indicators`` node
decides whether an LLM narrative is worth requesting at all, and low-risk URLs
with no observed indicators short-circuit straight to the deterministic
explanation, saving a network round trip.

Constraints kept from the core product:
  * The agent has **no tools that touch the network**. It cannot browse, fetch
    or resolve the submitted URL. It reasons only over extracted features.
  * The LLM never decides the classification - ``ml_predict`` does, and the
    verdict is fixed before the explanation node runs.
  * The whole module is optional. It is imported lazily and the core API is
    fully functional when ``ENABLE_AGENT=false`` or LangGraph is absent.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from backend.app.core.config import settings
from backend.app.schemas.analysis import (
    AIExplanation,
    Prediction,
    RiskLevel,
    Severity,
    SuspiciousIndicator,
    URLComponents,
)
from backend.app.services import feature_extractor
from backend.app.services.llm_analyzer import llm_analyzer
from backend.app.services.predictor import interpret, predictor, verdict_threshold

logger = logging.getLogger(__name__)


class AnalysisState(TypedDict, total=False):
    """State passed between graph nodes."""

    url: str
    normalized_url: str
    components: URLComponents
    features: dict[str, float]
    probability: float
    prediction: Prediction
    risk_level: RiskLevel
    risk_score: int
    confidence: float
    indicators: list[SuspiciousIndicator]
    needs_llm: bool
    explanation: AIExplanation
    trace: list[str]


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------

def _node_validate(state: AnalysisState) -> AnalysisState:
    """Reject anything that is not an analyzable http(s) URL."""
    normalized, components = feature_extractor.validate_url(state["url"])
    return {
        "normalized_url": normalized,
        "components": components,
        "trace": [*state.get("trace", []), "validate"],
    }


def _node_extract(state: AnalysisState) -> AnalysisState:
    """Derive the static feature vector from the URL string."""
    return {
        "features": feature_extractor.extract(state["normalized_url"]),
        "trace": [*state.get("trace", []), "extract_features"],
    }


def _node_predict(state: AnalysisState) -> AnalysisState:
    """Run the trained classifier. This node, and only this node, decides."""
    probability = predictor.predict_one(state["features"], state["normalized_url"])
    prediction, risk_level, risk_score, confidence = interpret(probability, verdict_threshold())
    return {
        "probability": probability,
        "prediction": prediction,
        "risk_level": risk_level,
        "risk_score": risk_score,
        "confidence": confidence,
        "trace": [*state.get("trace", []), "ml_predict"],
    }


def _node_assess(state: AnalysisState) -> AnalysisState:
    """Collect observable indicators and decide whether an LLM call is warranted."""
    indicators = feature_extractor.build_suspicious_indicators(
        state["normalized_url"], state["features"], state["components"]
    )
    notable = any(i.severity in (Severity.HIGH, Severity.MEDIUM) for i in indicators)
    needs_llm = notable or state["prediction"] is not Prediction.LEGITIMATE
    return {
        "indicators": indicators,
        "needs_llm": needs_llm,
        "trace": [*state.get("trace", []), "assess_indicators"],
    }


async def _node_explain(state: AnalysisState) -> AnalysisState:
    """Produce the narrative, honouring the routing decision from ``assess``."""
    explanation = await llm_analyzer.explain(
        prediction=state["prediction"],
        probability=state["probability"],
        risk_score=state["risk_score"],
        risk_level=state["risk_level"].value,
        components=state["components"],
        features=state["features"],
        indicators=state["indicators"],
        model_name=predictor.model_name,
        model_version=predictor.model_version,
        threshold=verdict_threshold(),
        want_llm=state.get("needs_llm", True),
    )
    return {
        "explanation": explanation,
        "trace": [*state.get("trace", []), "explain"],
    }


# --------------------------------------------------------------------------
# Graph construction
# --------------------------------------------------------------------------

_compiled_graph: Any | None = None
_build_error: str | None = None


def build_graph() -> Any | None:
    """Compile the LangGraph state machine, or return None if unavailable."""
    global _compiled_graph, _build_error
    if _compiled_graph is not None or _build_error is not None:
        return _compiled_graph

    try:
        from langgraph.graph import END, START, StateGraph

        graph = StateGraph(AnalysisState)
        graph.add_node("validate", _node_validate)
        graph.add_node("extract_features", _node_extract)
        graph.add_node("ml_predict", _node_predict)
        graph.add_node("assess_indicators", _node_assess)
        graph.add_node("explain", _node_explain)

        graph.add_edge(START, "validate")
        graph.add_edge("validate", "extract_features")
        graph.add_edge("extract_features", "ml_predict")
        graph.add_edge("ml_predict", "assess_indicators")
        graph.add_edge("assess_indicators", "explain")
        graph.add_edge("explain", END)

        _compiled_graph = graph.compile()
        logger.info("LangGraph security agent compiled")
        return _compiled_graph
    except Exception as exc:
        _build_error = str(exc)
        logger.warning("LangGraph agent unavailable (%s); the core API is unaffected.", exc)
        return None


def is_available() -> bool:
    """True when the agent is enabled in config *and* compiles successfully."""
    return settings.enable_agent and build_graph() is not None


async def run_agent(url: str) -> AnalysisState:
    """Execute the graph for one URL and return the final state."""
    graph = build_graph()
    if graph is None:
        raise RuntimeError(f"LangGraph agent is not available: {_build_error}")
    return await graph.ainvoke({"url": url, "trace": []})
