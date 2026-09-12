"""Pydantic request/response models for the analysis API."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.app.core.config import settings


class Prediction(str, Enum):
    """The model's verdict, mapped to product-facing language."""

    LEGITIMATE = "legitimate"
    SUSPICIOUS = "suspicious"
    PHISHING = "phishing"


class RiskLevel(str, Enum):
    """Coarse risk band derived from the risk score."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ExplanationSource(str, Enum):
    """Where the security narrative came from.

    The UI surfaces this verbatim so a rule-based fallback is never passed off
    as an LLM analysis.
    """

    LLM = "llm"
    RULE_BASED = "rule_based"


class Severity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# --------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    """A single-URL analysis request."""

    model_config = ConfigDict(json_schema_extra={
        "example": {"url": "https://secure-paypa1-login.example.com/verify"}
    })

    url: str = Field(..., min_length=4, description="The URL to analyze.")
    include_ai_explanation: bool = Field(
        True,
        description="Request an LLM-written explanation. Ignored when no LLM is configured.",
    )

    @field_validator("url")
    @classmethod
    def _strip_and_bound(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("URL must not be empty")
        if len(value) > settings.max_url_length:
            raise ValueError(f"URL exceeds the {settings.max_url_length} character limit")
        return value


class BatchAnalyzeRequest(BaseModel):
    """A multi-URL analysis request (no LLM explanations, for throughput)."""

    urls: list[str] = Field(..., min_length=1, description="URLs to analyze.")

    @field_validator("urls")
    @classmethod
    def _bound_batch(cls, values: list[str]) -> list[str]:
        if len(values) > settings.max_batch_size:
            raise ValueError(
                f"Batch size {len(values)} exceeds the limit of {settings.max_batch_size}"
            )
        return [v.strip() for v in values]


# --------------------------------------------------------------------------
# Response building blocks
# --------------------------------------------------------------------------

class SuspiciousIndicator(BaseModel):
    """One concrete, observed property of the URL string.

    Every indicator is a *fact derived from the URL text*, not a model opinion.
    The LLM prompt is built exclusively from these so it cannot invent evidence.
    """

    code: str = Field(..., description="Stable machine-readable identifier.")
    title: str = Field(..., description="Short human-readable label.")
    description: str = Field(..., description="What was observed and why it matters.")
    severity: Severity = Severity.MEDIUM
    evidence: str | None = Field(
        None, description="The literal substring or value observed in the URL."
    )


class FeatureHighlight(BaseModel):
    """A feature formatted for display as a metric card in the UI."""

    key: str
    label: str
    value: str
    raw_value: float
    description: str
    icon: str = Field(..., description="Lucide icon name the frontend should render.")
    tone: Severity = Severity.INFO


class URLComponents(BaseModel):
    """The parsed shape of the URL (string parsing only, no network access)."""

    scheme: str
    host: str
    registrable_domain: str
    port: str | None = None
    path: str
    query: str
    fragment: str
    tld: str


class AIExplanation(BaseModel):
    """The security narrative accompanying the ML verdict."""

    text: str
    source: ExplanationSource
    model: str | None = Field(None, description="LLM model/deployment used, when applicable.")
    available: bool = Field(
        ..., description="False when the LLM could not be reached and a fallback was used."
    )
    notice: str | None = Field(
        None, description="User-facing note explaining why the fallback was used."
    )
    latency_ms: int | None = None


# --------------------------------------------------------------------------
# Responses
# --------------------------------------------------------------------------

class AnalysisResponse(BaseModel):
    """Full result of a single URL analysis."""

    url: str
    normalized_url: str
    prediction: Prediction
    risk_level: RiskLevel
    confidence: float = Field(..., ge=0.0, le=1.0,
                              description="Model confidence in the reported verdict.")
    phishing_probability: float = Field(
        ..., ge=0.0, le=1.0,
        description=(
            "P(phishing) the verdict was derived from. Equal to "
            "`model_probability` unless a reputation rule applied."
        ),
    )
    model_probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Raw P(phishing) from the classifier, before any reputation rule.",
    )
    reputation_domain: str | None = Field(
        None,
        description=(
            "Registrable domain matched on the curated known-good list, or null. "
            "When set, the score was clamped and the difference from "
            "`model_probability` is attributable to that rule alone."
        ),
    )
    risk_score: int = Field(..., ge=0, le=100,
                            description="Model-derived risk indicator, not a universal standard.")
    components: URLComponents
    features: dict[str, float] = Field(..., description="Complete extracted feature vector.")
    feature_highlights: list[FeatureHighlight]
    suspicious_indicators: list[SuspiciousIndicator]
    ai_explanation: AIExplanation
    model_name: str
    model_version: str
    decision_threshold: float
    analysis_ms: int
    static_analysis_only: bool = Field(
        True,
        description="Always true: the submitted URL is never visited or resolved.",
    )


class BatchAnalysisItem(BaseModel):
    """One row of a batch result."""

    url: str
    prediction: Prediction | None = None
    risk_level: RiskLevel | None = None
    confidence: float | None = None
    risk_score: int | None = None
    error: str | None = None


class BatchAnalysisResponse(BaseModel):
    results: list[BatchAnalysisItem]
    analyzed: int
    failed: int
    analysis_ms: int


class ModelMetrics(BaseModel):
    """Measured evaluation metrics. Never populated with placeholder values."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    false_positive_rate: float
    false_negative_rate: float


class ModelComparisonEntry(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    train_seconds: float | None = None


class ExternalHoldout(BaseModel):
    """Result of scoring the independent PhishTank feed."""

    source: str
    rows: int
    unique_domains: int
    detected: int
    recall: float
    note: str


class SanityFailure(BaseModel):
    """One curated case the model currently gets wrong."""

    url: str
    expected: int = Field(..., description="0 = legitimate, 1 = phishing")
    model_probability: float = Field(
        ..., description="Raw classifier output, before any reputation rule."
    )
    served_probability: float = Field(
        ..., description="What the user would see, after the reputation prior."
    )


class SanityGroup(BaseModel):
    """One scored group of the curated set."""

    total: int
    label: int = Field(..., description="0 = the group is legitimate, 1 = phishing")
    model_only_correct: int
    as_served_correct: int


class SanityCheck(BaseModel):
    """Result of the curated post-training smoke test.

    Reported separately from the held-out metrics and never presented as model
    accuracy: the set is hand-written and deliberately easy. It exists to catch
    shortcuts that aggregate metrics cannot, because the shortcut is present in
    the test split too.

    Scored twice throughout: ``model_only`` is the raw classifier, ``as_served``
    includes the reputation prior. Both are published so that a reputation list
    cannot mask a model regression behind a healthier-looking served number.
    """

    total: int
    correct: int
    accuracy: float
    legitimate_accuracy: float
    phishing_accuracy: float
    note: str
    model_only_accuracy: float | None = None
    model_only_legitimate_accuracy: float | None = None
    #: The number that matters most: legitimate URLs the reputation list does
    #: not cover, so nothing shields the model on them.
    unlisted_legitimate_accuracy: float | None = None
    per_group: dict[str, SanityGroup] = Field(default_factory=dict)
    failures: list[SanityFailure] = Field(default_factory=list)


class OperatingPoint(BaseModel):
    """Recall attainable at one precision floor."""

    precision_floor: float
    attainable: bool = True
    threshold: float | None = None
    precision: float | None = None
    recall: float | None = None


class ModelInfoResponse(BaseModel):
    """Everything the Model Intelligence page renders."""

    model_config = ConfigDict(protected_namespaces=())

    trained: bool = Field(..., description="False when no model artifact exists yet.")
    message: str | None = Field(None, description="Guidance shown when untrained.")
    model_name: str | None = None
    model_version: str | None = None
    trained_at: str | None = None
    feature_count: int | None = None
    feature_names: list[str] = Field(default_factory=list)
    decision_threshold: float | None = None
    threshold_policy: str | None = None
    selection_criterion: str | None = None
    test_metrics: ModelMetrics | None = None
    validation_metrics: ModelMetrics | None = None
    comparison: list[ModelComparisonEntry] = Field(default_factory=list)
    external_holdout: ExternalHoldout | None = None
    sanity_check: SanityCheck | None = None
    operating_points: list[OperatingPoint] = Field(default_factory=list)
    precision_floor: float | None = None
    top_features: list[dict[str, float | str]] = Field(default_factory=list)
    dataset: dict[str, object] = Field(default_factory=dict)
    risk_bands: dict[str, list[int]] = Field(default_factory=dict)


class LLMStatus(BaseModel):
    configured: bool
    provider: str
    model: str | None = None


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    app: str
    version: str
    environment: str
    model_loaded: bool
    model_name: str | None = None
    llm: LLMStatus
    agent_enabled: bool


class ErrorResponse(BaseModel):
    error: str
    message: str
    detail: str | None = None
