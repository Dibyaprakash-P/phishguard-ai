"""URL analysis endpoints."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, status

from backend.app.core.config import settings
from backend.app.core.exceptions import AgentUnavailableError, PhishGuardError
from backend.app.core.logging_config import redact_url
from backend.app.schemas.analysis import (
    AnalysisResponse,
    AnalyzeRequest,
    BatchAnalysisItem,
    BatchAnalysisResponse,
    BatchAnalyzeRequest,
    ErrorResponse,
)
from backend.app.services import feature_extractor
from backend.app.services.llm_analyzer import llm_analyzer
from backend.app.services.predictor import interpret, predictor, verdict_threshold

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["analysis"])

_ERROR_RESPONSES = {
    status.HTTP_422_UNPROCESSABLE_ENTITY: {"model": ErrorResponse, "description": "Invalid URL"},
    status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse, "description": "Model unavailable"},
}


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    responses=_ERROR_RESPONSES,
    summary="Analyze a single URL",
    description=(
        "Runs static feature extraction and ML classification on the submitted URL, "
        "then produces a security explanation. The URL is never visited, resolved or "
        "fetched, and it is not persisted anywhere."
    ),
)
async def analyze_url(request: AnalyzeRequest) -> AnalysisResponse:
    """Analyze one URL and return the full security report."""
    started = time.perf_counter()

    normalized, components = feature_extractor.validate_url(request.url)
    features = feature_extractor.extract(normalized)

    probability = predictor.predict_one(features, normalized)
    prediction, risk_level, risk_score, confidence = interpret(probability, verdict_threshold())

    indicators = feature_extractor.build_suspicious_indicators(normalized, features, components)
    highlights = feature_extractor.build_feature_highlights(features, components)

    # The LLM is called only after the ML verdict exists, and only for the
    # explanation. It can never change the classification.
    explanation = await llm_analyzer.explain(
        prediction=prediction,
        probability=probability,
        risk_score=risk_score,
        risk_level=risk_level.value,
        components=components,
        features=features,
        indicators=indicators,
        model_name=predictor.model_name,
        model_version=predictor.model_version,
        threshold=predictor.threshold,
        want_llm=request.include_ai_explanation,
    )

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "Analyzed %s -> %s (p=%.4f, %d ms, explanation=%s)",
        redact_url(normalized), prediction.value, probability,
        elapsed_ms, explanation.source.value,
    )

    return AnalysisResponse(
        url=request.url,
        normalized_url=normalized,
        prediction=prediction,
        risk_level=risk_level,
        confidence=round(confidence, 4),
        phishing_probability=round(probability, 4),
        risk_score=risk_score,
        components=components,
        features={k: round(float(v), 6) for k, v in features.items()},
        feature_highlights=highlights,
        suspicious_indicators=indicators,
        ai_explanation=explanation,
        model_name=predictor.model_name,
        model_version=predictor.model_version,
        decision_threshold=verdict_threshold(),
        analysis_ms=elapsed_ms,
    )


@router.post(
    "/batch-analyze",
    response_model=BatchAnalysisResponse,
    responses=_ERROR_RESPONSES,
    summary="Analyze several URLs at once",
    description=(
        f"Classifies up to {settings.max_batch_size} URLs in a single request. "
        "No LLM explanations are generated for batches; individual failures are "
        "reported per row instead of failing the whole request."
    ),
)
async def batch_analyze(request: BatchAnalyzeRequest) -> BatchAnalysisResponse:
    """Classify a list of URLs, isolating per-URL failures."""
    started = time.perf_counter()
    results: list[BatchAnalysisItem] = []

    # Validate and featurise first so the model runs on one batched matrix.
    valid: list[tuple[int, dict[str, float], str]] = []
    for index, raw in enumerate(request.urls):
        try:
            normalized, _ = feature_extractor.validate_url(raw)
            valid.append((index, feature_extractor.extract(normalized), normalized))
            results.append(BatchAnalysisItem(url=raw))
        except PhishGuardError as exc:
            results.append(BatchAnalysisItem(url=raw, error=exc.message))

    if valid:
        threshold = verdict_threshold()
        probabilities = predictor.predict_proba(
            [features for _, features, _ in valid],
            [url for _, _, url in valid],
        )
        for (index, _, _), probability in zip(valid, probabilities, strict=True):
            prediction, risk_level, risk_score, confidence = interpret(
                float(probability), threshold
            )
            results[index] = BatchAnalysisItem(
                url=request.urls[index],
                prediction=prediction,
                risk_level=risk_level,
                confidence=round(confidence, 4),
                risk_score=risk_score,
            )

    failed = sum(1 for item in results if item.error)
    return BatchAnalysisResponse(
        results=results,
        analyzed=len(results) - failed,
        failed=failed,
        analysis_ms=int((time.perf_counter() - started) * 1000),
    )


@router.post(
    "/agent-analyze",
    response_model=AnalysisResponse,
    responses={
        **_ERROR_RESPONSES,
        status.HTTP_501_NOT_IMPLEMENTED: {
            "model": ErrorResponse, "description": "Agent disabled",
        },
    },
    summary="Analyze a URL through the LangGraph agent (optional)",
    description=(
        "Runs the same analysis as /api/analyze but orchestrated as an explicit "
        "LangGraph state machine, and returns the node trace. Disabled by default; "
        "set ENABLE_AGENT=true to switch it on. Like every other path, the agent "
        "has no network tools and never visits the submitted URL."
    ),
)
async def agent_analyze(request: AnalyzeRequest) -> AnalysisResponse:
    """Analyze a URL via the optional agentic pipeline."""
    from backend.app.services import agent as agent_service

    if not agent_service.is_available():
        raise AgentUnavailableError(
            "The LangGraph analysis agent is not enabled on this deployment. "
            "Use /api/analyze, or start the API with ENABLE_AGENT=true."
        )

    started = time.perf_counter()
    state = await agent_service.run_agent(request.url)
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    logger.info("Agent analyzed %s -> %s via %s",
                redact_url(state["normalized_url"]), state["prediction"].value,
                " -> ".join(state.get("trace", [])))

    features = state["features"]
    return AnalysisResponse(
        url=request.url,
        normalized_url=state["normalized_url"],
        prediction=state["prediction"],
        risk_level=state["risk_level"],
        confidence=round(state["confidence"], 4),
        phishing_probability=round(state["probability"], 4),
        risk_score=state["risk_score"],
        components=state["components"],
        features={k: round(float(v), 6) for k, v in features.items()},
        feature_highlights=feature_extractor.build_feature_highlights(
            features, state["components"]
        ),
        suspicious_indicators=state["indicators"],
        ai_explanation=state["explanation"],
        model_name=predictor.model_name,
        model_version=predictor.model_version,
        decision_threshold=verdict_threshold(),
        analysis_ms=elapsed_ms,
    )
