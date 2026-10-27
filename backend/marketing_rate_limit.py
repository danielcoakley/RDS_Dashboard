from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import os

from .runtime_config import PRODUCTION_RUNTIME, runtime_env_from_env


MARKETING_INQUIRY_RATE_LIMIT_ENABLED_ENV = "SAAS_MARKETING_INQUIRY_RATE_LIMIT_ENABLED"
MARKETING_INQUIRY_RATE_LIMIT_MAX_ENV = "SAAS_MARKETING_INQUIRY_RATE_LIMIT_MAX"
MARKETING_INQUIRY_RATE_LIMIT_WINDOW_MINUTES_ENV = (
    "SAAS_MARKETING_INQUIRY_RATE_LIMIT_WINDOW_MINUTES"
)
MARKETING_INQUIRY_RATE_LIMIT_SECRET_ENV = "SAAS_MARKETING_INQUIRY_RATE_LIMIT_SECRET"
MARKETING_INQUIRY_RATE_LIMIT_SECRET_MIN_LENGTH = 32
DEVELOPMENT_RATE_LIMIT_SECRET = "development-marketing-inquiry-rate-limit"


class MarketingInquiryRateLimitConfigurationError(RuntimeError):
    pass


class MarketingInquiryRateLimitExceeded(RuntimeError):
    def __init__(self, message: str, *, retry_after_seconds: int) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


@dataclass(frozen=True)
class MarketingInquiryRateLimitPolicy:
    enabled: bool
    max_submissions: int
    window_minutes: int
    secret: str


def _env_text(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _bool_env(name: str, default: bool) -> bool:
    value = _env_text(name)
    if value is None:
        return default
    lowered = value.lower()
    if lowered in {"1", "true", "yes"}:
        return True
    if lowered in {"0", "false", "no"}:
        return False
    raise MarketingInquiryRateLimitConfigurationError(f"{name} must be a boolean")


def _int_env(name: str, default: int) -> int:
    value = _env_text(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise MarketingInquiryRateLimitConfigurationError(f"{name} must be an integer") from exc
    if parsed < 1:
        raise MarketingInquiryRateLimitConfigurationError(f"{name} must be greater than zero")
    return parsed


def marketing_inquiry_rate_limit_policy_from_env(
    runtime_env: str | None = None,
) -> MarketingInquiryRateLimitPolicy:
    active_runtime_env = runtime_env or runtime_env_from_env()
    enabled = _bool_env(MARKETING_INQUIRY_RATE_LIMIT_ENABLED_ENV, True)
    max_submissions = _int_env(MARKETING_INQUIRY_RATE_LIMIT_MAX_ENV, 5)
    window_minutes = _int_env(MARKETING_INQUIRY_RATE_LIMIT_WINDOW_MINUTES_ENV, 60)
    secret = _env_text(MARKETING_INQUIRY_RATE_LIMIT_SECRET_ENV)
    if enabled and active_runtime_env == PRODUCTION_RUNTIME and secret is None:
        raise MarketingInquiryRateLimitConfigurationError(
            f"{MARKETING_INQUIRY_RATE_LIMIT_SECRET_ENV} is required when marketing "
            "inquiry rate limiting is enabled in production"
        )
    if (
        enabled
        and active_runtime_env == PRODUCTION_RUNTIME
        and secret is not None
        and len(secret) < MARKETING_INQUIRY_RATE_LIMIT_SECRET_MIN_LENGTH
    ):
        raise MarketingInquiryRateLimitConfigurationError(
            f"{MARKETING_INQUIRY_RATE_LIMIT_SECRET_ENV} must be at least "
            f"{MARKETING_INQUIRY_RATE_LIMIT_SECRET_MIN_LENGTH} characters in production"
        )
    return MarketingInquiryRateLimitPolicy(
        enabled=enabled,
        max_submissions=max_submissions,
        window_minutes=window_minutes,
        secret=secret or DEVELOPMENT_RATE_LIMIT_SECRET,
    )


def marketing_inquiry_rate_limit_key(source_identifier: str, secret: str) -> str:
    normalized = " ".join(source_identifier.strip().split())
    if not normalized:
        normalized = "unknown"
    return hmac.new(
        secret.encode("utf-8"),
        normalized.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def marketing_inquiry_rate_limit_configuration_status(
    runtime_env: str | None = None,
) -> tuple[bool, str]:
    active_runtime_env = runtime_env or runtime_env_from_env()
    try:
        policy = marketing_inquiry_rate_limit_policy_from_env(active_runtime_env)
    except MarketingInquiryRateLimitConfigurationError as exc:
        return False, str(exc)
    if not policy.enabled:
        if active_runtime_env == PRODUCTION_RUNTIME:
            return (
                False,
                f"{MARKETING_INQUIRY_RATE_LIMIT_ENABLED_ENV} must be true in production",
            )
        return True, f"{MARKETING_INQUIRY_RATE_LIMIT_ENABLED_ENV}=false"
    secret_detail = (
        f"{MARKETING_INQUIRY_RATE_LIMIT_SECRET_ENV} configured"
        if _env_text(MARKETING_INQUIRY_RATE_LIMIT_SECRET_ENV) is not None
        else f"{MARKETING_INQUIRY_RATE_LIMIT_SECRET_ENV} optional in development"
    )
    return (
        True,
        f"Marketing inquiry rate limit allows {policy.max_submissions} submissions "
        f"per {policy.window_minutes} minutes; {secret_detail}",
    )
