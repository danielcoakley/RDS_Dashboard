from __future__ import annotations

from datetime import datetime, timezone
import os
from urllib.parse import urlparse

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .runtime_config import PRODUCTION_RUNTIME


ALLOWED_ORIGINS_ENV = "SAAS_ALLOWED_ORIGINS"
HSTS_MAX_AGE_ENV = "SAAS_HSTS_MAX_AGE_SECONDS"
SECURITY_REVIEW_URL_ENV = "SAAS_SECURITY_REVIEW_URL"
SECURITY_REVIEWED_AT_ENV = "SAAS_SECURITY_REVIEWED_AT"
SECURITY_REVIEW_MAX_AGE_DAYS_ENV = "SAAS_SECURITY_REVIEW_MAX_AGE_DAYS"
TENANT_ISOLATION_REVIEW_URL_ENV = "SAAS_TENANT_ISOLATION_REVIEW_URL"
TENANT_ISOLATION_REVIEWED_AT_ENV = "SAAS_TENANT_ISOLATION_REVIEWED_AT"
TENANT_ISOLATION_REVIEW_MAX_AGE_DAYS_ENV = "SAAS_TENANT_ISOLATION_REVIEW_MAX_AGE_DAYS"
DEFAULT_DEVELOPMENT_ORIGINS = ("http://localhost:3000", "http://127.0.0.1:3000")
DEFAULT_HSTS_MAX_AGE_SECONDS = 31_536_000
DEFAULT_SECURITY_REVIEW_MAX_AGE_DAYS = 90
DEFAULT_TENANT_ISOLATION_REVIEW_MAX_AGE_DAYS = 90


class SecurityConfigurationError(ValueError):
    pass


def _configured_value(name: str) -> str:
    return os.getenv(name, "").strip()


def _parse_positive_int_env(name: str, default: int) -> int:
    value = _configured_value(name)
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise SecurityConfigurationError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise SecurityConfigurationError(f"{name} must be positive")
    return parsed


def _require_https_url(name: str, runtime_env: str) -> str | None:
    value = _configured_value(name)
    if not value:
        if runtime_env == PRODUCTION_RUNTIME:
            raise SecurityConfigurationError(f"{name} is required in production")
        return None

    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise SecurityConfigurationError(f"{name} must be a URL")
    if runtime_env == PRODUCTION_RUNTIME and parsed.scheme != "https":
        raise SecurityConfigurationError(f"{name} must use https in production")
    return value


def allowed_origins_from_env(runtime_env: str) -> list[str]:
    raw_value = os.getenv(ALLOWED_ORIGINS_ENV)
    if raw_value is None or not raw_value.strip():
        if runtime_env == PRODUCTION_RUNTIME:
            raise SecurityConfigurationError(
                f"{ALLOWED_ORIGINS_ENV} must list the trusted frontend origins in production"
            )
        return list(DEFAULT_DEVELOPMENT_ORIGINS)

    origins = [part.strip().rstrip("/") for part in raw_value.split(",") if part.strip()]
    if not origins:
        raise SecurityConfigurationError(f"{ALLOWED_ORIGINS_ENV} cannot be empty")
    if "*" in origins and runtime_env == PRODUCTION_RUNTIME:
        raise SecurityConfigurationError(f"{ALLOWED_ORIGINS_ENV} cannot include '*' in production")

    for origin in origins:
        if origin == "*":
            continue
        parsed = urlparse(origin)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path.strip("/"):
            raise SecurityConfigurationError(
                f"{ALLOWED_ORIGINS_ENV} entries must be origins like https://app.example.com"
            )
        if runtime_env == PRODUCTION_RUNTIME and parsed.scheme != "https":
            raise SecurityConfigurationError(
                f"{ALLOWED_ORIGINS_ENV} entries must use https in production"
            )

    return origins


def hsts_max_age_from_env() -> int:
    value = _configured_value(HSTS_MAX_AGE_ENV)
    if not value:
        return DEFAULT_HSTS_MAX_AGE_SECONDS
    try:
        parsed = int(value)
    except ValueError as exc:
        raise SecurityConfigurationError(f"{HSTS_MAX_AGE_ENV} must be an integer") from exc
    if parsed < 0:
        raise SecurityConfigurationError(f"{HSTS_MAX_AGE_ENV} cannot be negative")
    return parsed


def security_review_max_age_days_from_env() -> int:
    return _parse_positive_int_env(
        SECURITY_REVIEW_MAX_AGE_DAYS_ENV,
        DEFAULT_SECURITY_REVIEW_MAX_AGE_DAYS,
    )


def tenant_isolation_review_max_age_days_from_env() -> int:
    return _parse_positive_int_env(
        TENANT_ISOLATION_REVIEW_MAX_AGE_DAYS_ENV,
        DEFAULT_TENANT_ISOLATION_REVIEW_MAX_AGE_DAYS,
    )


def _parse_reviewed_at(value: str, reviewed_at_env: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SecurityConfigurationError(f"{reviewed_at_env} must be an ISO-8601 datetime") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SecurityConfigurationError(f"{reviewed_at_env} must include a timezone")
    return parsed.astimezone(timezone.utc)


def security_configuration_status(runtime_env: str) -> tuple[bool, str]:
    try:
        origins = allowed_origins_from_env(runtime_env)
        hsts_max_age = hsts_max_age_from_env()
    except SecurityConfigurationError as exc:
        return False, str(exc)

    if runtime_env == PRODUCTION_RUNTIME and hsts_max_age <= 0:
        return False, f"{HSTS_MAX_AGE_ENV} must be positive in production"

    return (
        True,
        f"{ALLOWED_ORIGINS_ENV} configured for {len(origins)} origin(s); "
        f"{HSTS_MAX_AGE_ENV}={hsts_max_age}",
    )


def _review_evidence_configuration_status(
    runtime_env: str,
    *,
    url_env: str,
    reviewed_at_env: str,
    max_age_env: str,
    default_max_age_days: int,
    evidence_label: str,
    now: datetime | None = None,
) -> tuple[bool, str]:
    try:
        review_url = _require_https_url(url_env, runtime_env)
        max_age_days = _parse_positive_int_env(max_age_env, default_max_age_days)
    except SecurityConfigurationError as exc:
        return False, str(exc)

    reviewed_at = _configured_value(reviewed_at_env)
    if runtime_env == PRODUCTION_RUNTIME and not reviewed_at:
        return False, f"{reviewed_at_env} is required in production"
    if not reviewed_at:
        return True, f"{evidence_label.capitalize()} evidence is optional in development"

    try:
        observed_at = _parse_reviewed_at(reviewed_at, reviewed_at_env)
    except SecurityConfigurationError as exc:
        return False, str(exc)

    effective_now = now or datetime.now(timezone.utc)
    if observed_at > effective_now:
        return False, f"{reviewed_at_env} cannot be in the future"
    age_days = (effective_now - observed_at).days
    if runtime_env == PRODUCTION_RUNTIME and age_days > max_age_days:
        return False, f"{reviewed_at_env} is older than {max_age_days} days"

    detail = f"{evidence_label} age {age_days} day(s)"
    if review_url:
        detail = f"{url_env} configured; {detail}"
    return True, detail


def security_review_configuration_status(
    runtime_env: str,
    *,
    now: datetime | None = None,
) -> tuple[bool, str]:
    return _review_evidence_configuration_status(
        runtime_env,
        url_env=SECURITY_REVIEW_URL_ENV,
        reviewed_at_env=SECURITY_REVIEWED_AT_ENV,
        max_age_env=SECURITY_REVIEW_MAX_AGE_DAYS_ENV,
        default_max_age_days=DEFAULT_SECURITY_REVIEW_MAX_AGE_DAYS,
        evidence_label="security review",
        now=now,
    )


def tenant_isolation_review_configuration_status(
    runtime_env: str,
    *,
    now: datetime | None = None,
) -> tuple[bool, str]:
    return _review_evidence_configuration_status(
        runtime_env,
        url_env=TENANT_ISOLATION_REVIEW_URL_ENV,
        reviewed_at_env=TENANT_ISOLATION_REVIEWED_AT_ENV,
        max_age_env=TENANT_ISOLATION_REVIEW_MAX_AGE_DAYS_ENV,
        default_max_age_days=DEFAULT_TENANT_ISOLATION_REVIEW_MAX_AGE_DAYS,
        evidence_label="tenant isolation review",
        now=now,
    )


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, runtime_env: str):  # type: ignore[no-untyped-def]
        super().__init__(app)
        self.runtime_env = runtime_env
        self.hsts_max_age = hsts_max_age_from_env()

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        response: Response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Permissions-Policy",
            "camera=(), microphone=(), geolocation=(), payment=()",
        )
        response.headers.setdefault("Cross-Origin-Resource-Policy", "same-site")
        if self.runtime_env == PRODUCTION_RUNTIME:
            response.headers.setdefault(
                "Strict-Transport-Security",
                f"max-age={self.hsts_max_age}; includeSubDomains",
            )
        return response


def configure_security_middleware(app: FastAPI, runtime_env: str) -> None:
    origins = allowed_origins_from_env(runtime_env)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-Request-Id",
            "X-User-Id",
            "traceparent",
        ],
        expose_headers=["X-Request-Id", "traceparent"],
    )
    app.add_middleware(SecurityHeadersMiddleware, runtime_env=runtime_env)
