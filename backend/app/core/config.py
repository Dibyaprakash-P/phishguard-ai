"""Application settings, loaded from environment variables / ``.env``.

No secret is ever hardcoded. Every value has a safe default so that the API
starts and serves ML predictions even with a completely empty environment.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import List, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# The `ml` package lives at the repository root and is the single source of
# truth for feature extraction. Adding the root to sys.path lets the backend
# import it without duplicating the code (and therefore without train/serve
# skew). The Docker image copies `ml/` alongside `backend/` for the same reason.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class Settings(BaseSettings):
    """Runtime configuration for the PhishGuard AI API."""

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", Path(".env")),
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    # ----------------------------------------------------------------- app
    app_name: str = "PhishGuard AI"
    app_version: str = "1.0.0"
    environment: Literal["development", "production", "test"] = "development"
    debug: bool = False
    log_level: str = "INFO"

    # ------------------------------------------------------------ artifacts
    model_path: Path = PROJECT_ROOT / "models" / "phishing_model.joblib"
    feature_metadata_path: Path = PROJECT_ROOT / "models" / "feature_metadata.json"
    metrics_path: Path = PROJECT_ROOT / "models" / "metrics.json"

    # ---------------------------------------------------------------- CORS
    # NOTE: typing.List, not the builtin generic. On Python 3.10 `list[str]` is
    # an instance of `type`, which makes pydantic-settings' complex-field probe
    # call issubclass() on a parametrised generic and raise TypeError. Ruff's
    # UP006 rule is disabled for this file for exactly that reason.
    cors_origins: List[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173", "http://127.0.0.1:5173",
            "http://localhost:4173", "http://127.0.0.1:4173",
            "http://localhost:3000", "http://127.0.0.1:3000",
        ]
    )

    # --------------------------------------------------------------- limits
    max_url_length: int = 2048
    max_batch_size: int = 25
    request_timeout_seconds: float = 30.0

    # ------------------------------------------------------------------ LLM
    #: ``none`` disables the LLM entirely; the API then always serves the
    #: deterministic rule-based explanation.
    llm_provider: Literal["openai", "azure", "none"] = "none"
    llm_enabled: bool = True
    llm_timeout_seconds: float = 20.0
    llm_temperature: float = 0.1
    llm_max_tokens: int = 400

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = ""

    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_deployment: str = ""
    azure_openai_api_version: str = "2024-08-01-preview"

    #: Opt-in LangGraph multi-step agent. Off by default: the core product
    #: must never depend on it.
    enable_agent: bool = False

    # ------------------------------------------------------- risk thresholds
    #: When true (the default), the phishing boundary is the operating point
    #: chosen on the validation split during training and stored in the model
    #: metadata. That is the threshold whose precision was actually measured,
    #: so using anything else would report a precision the model never had.
    #: Set false to pin the boundary to ``phishing_threshold`` instead.
    use_model_threshold: bool = True

    suspicious_threshold: float = 0.40
    phishing_threshold: float = 0.70
    risk_low_max: int = 30
    risk_medium_max: int = 70

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Allow a comma-separated string in the environment."""
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @property
    def llm_configured(self) -> bool:
        """True when enough credentials are present to attempt an LLM call."""
        if not self.llm_enabled:
            return False
        if self.llm_provider == "openai":
            return bool(self.openai_api_key)
        if self.llm_provider == "azure":
            return bool(
                self.azure_openai_api_key
                and self.azure_openai_endpoint
                and self.azure_openai_deployment
            )
        return False


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


settings = get_settings()
