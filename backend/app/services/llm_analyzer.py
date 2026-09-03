"""LangChain-backed security explanation, with a deterministic fallback.

Design contract
---------------
The LLM is **not** a classifier. It never sees the raw URL as a thing to judge;
it receives an already-final ML verdict plus a list of facts that were
mechanically derived from the URL string, and its only job is to turn that
structured evidence into readable prose.

Two rules follow from that and are enforced in the prompt:

1. The model may only reference evidence it was handed. It has no browsing
   tool, so any claim about page content, WHOIS records, reputation feeds or
   malware would be fabrication.
2. A model prediction must be described as a prediction. "The classifier scored
   this 0.96" is true; "this domain is fraudulent" is not something a URL-string
   classifier can establish.

If no API key is configured, or the provider call fails or times out, the
service returns a rule-based explanation assembled from the same evidence and
flags it as such. The product never claims an AI analysis that did not happen.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from backend.app.core.config import settings
from backend.app.schemas.analysis import (
    AIExplanation,
    ExplanationSource,
    Prediction,
    Severity,
    SuspiciousIndicator,
    URLComponents,
)

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You are a security analyst assistant inside PhishGuard AI, a phishing URL \
detection product. A machine-learning classifier has ALREADY produced the \
verdict. Your only task is to explain that verdict in clear, calm prose for a \
non-expert reader.

You are given:
  - the classifier's prediction, probability and risk score
  - the parsed components of the URL string
  - a list of characteristics that were mechanically extracted from the URL text

STRICT RULES - these are non-negotiable:
1. Use ONLY the evidence provided below. Do not introduce any characteristic \
that is not in the list.
2. The website was NOT visited. No page content, HTML, screenshot, redirect \
chain or downloaded file was inspected. Never imply otherwise.
3. No WHOIS lookup, DNS query, certificate inspection, blocklist check or \
threat-intelligence feed was consulted. Never reference them.
4. Never claim malware was found, downloaded or executed.
5. Never state as established fact that a domain is fraudulent, criminal or \
compromised. The classifier output is a statistical prediction from URL \
structure - describe it as such.
6. Clearly separate two kinds of statement: what was OBSERVED in the URL text \
(factual) and what the MODEL PREDICTED (probabilistic).
7. If the evidence is thin, say so plainly rather than inventing support.
8. Do not give the reader a verdict that contradicts the classifier's.

STYLE:
  - 2 to 4 short paragraphs, 90-160 words total.
  - Plain prose. No markdown headings, no bullet lists, no emoji.
  - Professional and measured. No alarmist language.
  - End with one practical sentence of guidance appropriate to the risk level.
"""

HUMAN_PROMPT = """\
CLASSIFIER OUTPUT (already final - do not re-decide):
  Prediction: {prediction}
  Probability of phishing: {probability:.4f}
  Risk score: {risk_score}/100 ({risk_level} band)
  Model: {model_name} v{model_version}
  Decision threshold: {threshold:.4f}

URL STRUCTURE (parsed from the string; the site was not contacted):
  Scheme: {scheme}
  Hostname: {host}
  Registrable domain: {registrable_domain}
  Path: {path}
  Query string present: {has_query}

MEASURED CHARACTERISTICS:
{measurements}

OBSERVED INDICATORS ({indicator_count} found):
{indicators}

Write the security explanation now, obeying every rule above."""


class LLMAnalyzer:
    """Builds security explanations, preferring the LLM and falling back safely."""

    def __init__(self) -> None:
        self._chain: Any | None = None
        self._model_label: str | None = None
        self._init_error: str | None = None
        self._initialized = False

    # ------------------------------------------------------------- provider
    def _build_chain(self) -> None:
        """Lazily construct the LangChain runnable for the configured provider."""
        if self._initialized:
            return
        self._initialized = True

        if not settings.llm_configured:
            self._init_error = (
                f"No credentials for provider '{settings.llm_provider}'."
                if settings.llm_provider != "none"
                else "LLM_PROVIDER is set to 'none'."
            )
            logger.info("LLM disabled: %s Rule-based explanations will be used.", self._init_error)
            return

        try:
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate

            if settings.llm_provider == "azure":
                from langchain_openai import AzureChatOpenAI

                llm = AzureChatOpenAI(
                    azure_endpoint=settings.azure_openai_endpoint,
                    azure_deployment=settings.azure_openai_deployment,
                    api_key=settings.azure_openai_api_key,
                    api_version=settings.azure_openai_api_version,
                    temperature=settings.llm_temperature,
                    max_tokens=settings.llm_max_tokens,
                    timeout=settings.llm_timeout_seconds,
                    max_retries=1,
                )
                self._model_label = f"azure:{settings.azure_openai_deployment}"
            else:
                from langchain_openai import ChatOpenAI

                kwargs: dict[str, Any] = {
                    "model": settings.openai_model,
                    "api_key": settings.openai_api_key,
                    "temperature": settings.llm_temperature,
                    "max_tokens": settings.llm_max_tokens,
                    "timeout": settings.llm_timeout_seconds,
                    "max_retries": 1,
                }
                if settings.openai_base_url:
                    kwargs["base_url"] = settings.openai_base_url
                llm = ChatOpenAI(**kwargs)
                self._model_label = f"openai:{settings.openai_model}"

            prompt = ChatPromptTemplate.from_messages(
                [("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)]
            )
            self._chain = prompt | llm | StrOutputParser()
            logger.info("LangChain explanation chain ready (%s)", self._model_label)

        except Exception as exc:
            self._init_error = f"LLM initialization failed: {exc}"
            logger.warning("%s Falling back to rule-based explanations.", self._init_error)
            self._chain = None

    @property
    def available(self) -> bool:
        self._build_chain()
        return self._chain is not None

    @property
    def model_label(self) -> str | None:
        self._build_chain()
        return self._model_label

    # ------------------------------------------------------------- prompting
    @staticmethod
    def _format_measurements(features: dict[str, float], components: URLComponents) -> str:
        """Render the handful of features worth putting in the prompt.

        Only interpretable measurements are included. Feeding every raw
        features would dilute the context without helping the narrative.
        """
        rows = [
            ("URL length", f"{int(features['url_length'])} characters"),
            ("Hostname length", f"{int(features['hostname_length'])} characters"),
            ("Subdomain levels (excluding www)", str(int(features["num_subdomains"]))),
            ("Path depth", str(int(features["path_depth"]))),
            ("Digits in URL", str(int(features["num_digits"]))),
            ("Special characters", str(int(features["num_special_chars"]))),
            ("Hyphens in hostname", str(int(features["num_hyphens_in_host"]))),
            ("Query parameters", str(int(features["num_query_params"]))),
            ("HTTPS", "yes" if components.scheme == "https" else "no"),
            ("Hostname is an IP literal", "yes" if features["has_ip_host"] else "no"),
            ("Hostname entropy", f"{features['domain_entropy']:.2f} bits/char"),
            ("Suspicious keyword matches", str(int(features["num_suspicious_keywords"]))),
        ]
        return "\n".join(f"  - {label}: {value}" for label, value in rows)

    @staticmethod
    def _format_indicators(indicators: list[SuspiciousIndicator]) -> str:
        if not indicators:
            return "  (none - no red-flag patterns were found in the URL string)"
        return "\n".join(
            f"  - [{ind.severity.value.upper()}] {ind.title}: {ind.description}"
            + (f" Evidence: {ind.evidence}" if ind.evidence else "")
            for ind in indicators
        )

    def _prompt_variables(
        self,
        prediction: Prediction,
        probability: float,
        risk_score: int,
        risk_level: str,
        components: URLComponents,
        features: dict[str, float],
        indicators: list[SuspiciousIndicator],
        model_name: str,
        model_version: str,
        threshold: float,
    ) -> dict[str, Any]:
        return {
            "prediction": prediction.value,
            "probability": probability,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "model_name": model_name,
            "model_version": model_version,
            "threshold": threshold,
            "scheme": components.scheme,
            "host": components.host,
            "registrable_domain": components.registrable_domain,
            "path": components.path,
            "has_query": "yes" if components.query else "no",
            "measurements": self._format_measurements(features, components),
            "indicators": self._format_indicators(indicators),
            "indicator_count": len(indicators),
        }

    # -------------------------------------------------------------- public
    async def explain(
        self,
        prediction: Prediction,
        probability: float,
        risk_score: int,
        risk_level: str,
        components: URLComponents,
        features: dict[str, float],
        indicators: list[SuspiciousIndicator],
        model_name: str,
        model_version: str,
        threshold: float,
        want_llm: bool = True,
    ) -> AIExplanation:
        """Produce the security narrative, using the LLM when it is available."""
        fallback_text = build_rule_based_explanation(
            prediction, probability, risk_score, components, features, indicators
        )

        if not want_llm:
            return AIExplanation(
                text=fallback_text,
                source=ExplanationSource.RULE_BASED,
                available=False,
                notice="AI explanation was not requested. Showing rule-based security analysis.",
            )

        self._build_chain()
        if self._chain is None:
            return AIExplanation(
                text=fallback_text,
                source=ExplanationSource.RULE_BASED,
                available=False,
                notice=(
                    "AI explanation unavailable. Showing rule-based security analysis. "
                    f"({self._init_error})"
                ),
            )

        variables = self._prompt_variables(
            prediction, probability, risk_score, risk_level, components,
            features, indicators, model_name, model_version, threshold,
        )

        started = time.perf_counter()
        try:
            text = await self._chain.ainvoke(variables)
            latency = int((time.perf_counter() - started) * 1000)
            cleaned = (text or "").strip()
            if not cleaned:
                raise ValueError("LLM returned an empty response")
            return AIExplanation(
                text=cleaned,
                source=ExplanationSource.LLM,
                model=self._model_label,
                available=True,
                latency_ms=latency,
            )
        except Exception as exc:
            # Any provider failure degrades to the deterministic explanation.
            # The URL itself is never logged here.
            logger.warning("LLM explanation failed (%s); using rule-based fallback.",
                           type(exc).__name__)
            return AIExplanation(
                text=fallback_text,
                source=ExplanationSource.RULE_BASED,
                available=False,
                notice=(
                    "AI explanation unavailable. Showing rule-based security analysis. "
                    f"({type(exc).__name__})"
                ),
                latency_ms=int((time.perf_counter() - started) * 1000),
            )


# --------------------------------------------------------------------------
# Deterministic fallback
# --------------------------------------------------------------------------

_VERDICT_OPENING = {
    Prediction.PHISHING: (
        "The classifier assigned this URL a high phishing probability of {prob:.1%}, "
        "placing it in the {level} risk band."
    ),
    Prediction.SUSPICIOUS: (
        "The classifier returned a phishing probability of {prob:.1%} for this URL, which "
        "falls in the uncertain middle band rather than a clear verdict either way."
    ),
    Prediction.LEGITIMATE: (
        "The classifier returned a low phishing probability of {prob:.1%} for this URL, "
        "consistent with a legitimate link."
    ),
}

_GUIDANCE = {
    Prediction.PHISHING: (
        "Treat this link as unsafe: do not enter credentials or payment details, and "
        "verify the destination through a channel you already trust."
    ),
    Prediction.SUSPICIOUS: (
        "Proceed only if you can independently confirm the sender and the destination; "
        "avoid entering credentials until you have."
    ),
    Prediction.LEGITIMATE: (
        "No structural red flags stood out, but a low score is not a guarantee - stay "
        "alert to any request for credentials you did not initiate."
    ),
}


def build_rule_based_explanation(
    prediction: Prediction,
    probability: float,
    risk_score: int,
    components: URLComponents,
    features: dict[str, float],
    indicators: list[SuspiciousIndicator],
) -> str:
    """Compose an explanation from the extracted evidence, with no LLM.

    This is a real analysis, not a placeholder: it cites the same observed
    indicators the LLM would be given. It is deterministic and always
    available, which is what makes the product usable without an API key.
    """
    level = ("high" if risk_score > settings.risk_medium_max
             else "medium" if risk_score > settings.risk_low_max else "low")

    paragraphs: list[str] = [
        _VERDICT_OPENING[prediction].format(prob=probability, level=level)
        + " This score comes from a supervised model that reads only the URL text; "
        "the site itself was not visited, and no registration, reputation or "
        "certificate data was consulted."
    ]

    high = [i for i in indicators if i.severity == Severity.HIGH]
    medium = [i for i in indicators if i.severity == Severity.MEDIUM]
    low = [i for i in indicators if i.severity in (Severity.LOW, Severity.INFO)]

    if high or medium:
        parts: list[str] = []
        if high:
            parts.append(
                "The strongest signals observed in the URL string were: "
                + "; ".join(i.title.lower() for i in high[:4]) + "."
            )
        if medium:
            parts.append(
                "Additional patterns of note: "
                + ", ".join(i.title.lower() for i in medium[:4]) + "."
            )
        paragraphs.append(" ".join(parts))
    elif low:
        paragraphs.append(
            "Only minor structural observations were made: "
            + ", ".join(i.title.lower() for i in low[:4]) + "."
        )
    else:
        paragraphs.append(
            "No red-flag patterns were detected in the URL string. The hostname "
            f"'{components.host}' resolves to the registrable domain "
            f"'{components.registrable_domain}' and the structure is unremarkable."
        )

    paragraphs.append(
        f"Structurally, the URL is {int(features['url_length'])} characters long with "
        f"{int(features['num_subdomains'])} subdomain level(s), a path depth of "
        f"{int(features['path_depth'])}, and "
        f"{'is' if components.scheme == 'https' else 'is not'} served over HTTPS. "
        + _GUIDANCE[prediction]
    )

    return "\n\n".join(paragraphs)


llm_analyzer = LLMAnalyzer()
