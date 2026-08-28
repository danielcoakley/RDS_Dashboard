from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
import re
from time import perf_counter
from urllib.parse import urlparse
from uuid import uuid4

from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .runtime_config import PRODUCTION_RUNTIME


REQUEST_ID_HEADER = "X-Request-Id"
TRACEPARENT_HEADER = "traceparent"
API_LOGGER_NAME = "rds.api"
MAX_REQUEST_ID_LENGTH = 128
LOG_AGGREGATION_URL_ENV = "SAAS_LOG_AGGREGATION_URL"
LOG_AGGREGATION_VERIFIED_AT_ENV = "SAAS_LOG_AGGREGATION_VERIFIED_AT"
LOG_AGGREGATION_MAX_AGE_DAYS_ENV = "SAAS_LOG_AGGREGATION_MAX_AGE_DAYS"
LOG_RETENTION_DAYS_ENV = "SAAS_LOG_RETENTION_DAYS"
DEFAULT_LOG_AGGREGATION_MAX_AGE_DAYS = 30
MAX_LOG_RETENTION_DAYS = 3650
TRACEPARENT_PATTERN = re.compile(
    r"^(?P<version>[0-9a-fA-F]{2})-"
    r"(?P<trace_id>[0-9a-fA-F]{32})-"
    r"(?P<span_id>[0-9a-fA-F]{16})-"
    r"(?P<trace_flags>[0-9a-fA-F]{2})$"
)


class ObservabilityConfigurationError(ValueError):
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
        raise ObservabilityConfigurationError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise ObservabilityConfigurationError(f"{name} must be positive")
    return parsed


def _retention_days_from_env(runtime_env: str) -> int | None:
    value = _configured_value(LOG_RETENTION_DAYS_ENV)
    if not value:
        if runtime_env == PRODUCTION_RUNTIME:
            raise ObservabilityConfigurationError(
                f"{LOG_RETENTION_DAYS_ENV} is required in production"
            )
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ObservabilityConfigurationError(
            f"{LOG_RETENTION_DAYS_ENV} must be an integer"
        ) from exc
    if parsed <= 0:
        raise ObservabilityConfigurationError(f"{LOG_RETENTION_DAYS_ENV} must be positive")
    if parsed > MAX_LOG_RETENTION_DAYS:
        raise ObservabilityConfigurationError(
            f"{LOG_RETENTION_DAYS_ENV} cannot exceed {MAX_LOG_RETENTION_DAYS}"
        )
    return parsed


def _require_https_url(name: str, runtime_env: str) -> str | None:
    value = _configured_value(name)
    if not value:
        if runtime_env == PRODUCTION_RUNTIME:
            raise ObservabilityConfigurationError(f"{name} is required in production")
        return None

    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ObservabilityConfigurationError(f"{name} must be a URL")
    if runtime_env == PRODUCTION_RUNTIME and parsed.scheme != "https":
        raise ObservabilityConfigurationError(f"{name} must use https in production")
    return value


def log_aggregation_max_age_days_from_env() -> int:
    return _parse_positive_int_env(
        LOG_AGGREGATION_MAX_AGE_DAYS_ENV,
        DEFAULT_LOG_AGGREGATION_MAX_AGE_DAYS,
    )


def _parse_verified_at(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ObservabilityConfigurationError(
            f"{LOG_AGGREGATION_VERIFIED_AT_ENV} must be an ISO-8601 datetime"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ObservabilityConfigurationError(
            f"{LOG_AGGREGATION_VERIFIED_AT_ENV} must include a timezone"
        )
    return parsed.astimezone(timezone.utc)


def log_aggregation_configuration_status(
    runtime_env: str,
    *,
    now: datetime | None = None,
) -> tuple[bool, str]:
    try:
        aggregation_url = _require_https_url(LOG_AGGREGATION_URL_ENV, runtime_env)
        retention_days = _retention_days_from_env(runtime_env)
        max_age_days = log_aggregation_max_age_days_from_env()
    except ObservabilityConfigurationError as exc:
        return False, str(exc)

    verified_at = _configured_value(LOG_AGGREGATION_VERIFIED_AT_ENV)
    if runtime_env == PRODUCTION_RUNTIME and not verified_at:
        return False, f"{LOG_AGGREGATION_VERIFIED_AT_ENV} is required in production"
    if not verified_at:
        return True, "Log aggregation evidence is optional in development"

    try:
        observed_at = _parse_verified_at(verified_at)
    except ObservabilityConfigurationError as exc:
        return False, str(exc)

    effective_now = now or datetime.now(timezone.utc)
    if observed_at > effective_now:
        return False, f"{LOG_AGGREGATION_VERIFIED_AT_ENV} cannot be in the future"
    age_days = (effective_now - observed_at).days
    if runtime_env == PRODUCTION_RUNTIME and age_days > max_age_days:
        return False, f"Log aggregation verification is older than {max_age_days} days"

    detail = f"log aggregation verified age {age_days} day(s)"
    if aggregation_url:
        detail = f"{LOG_AGGREGATION_URL_ENV} configured; {detail}"
    if retention_days is not None:
        detail = f"{detail}; {LOG_RETENTION_DAYS_ENV}={retention_days} day(s)"
    return True, detail


def _safe_request_id(value: str | None) -> str:
    cleaned = (value or "").strip()
    if (
        not cleaned
        or len(cleaned) > MAX_REQUEST_ID_LENGTH
        or any(not (character.isalnum() or character in {"-", "_", "."}) for character in cleaned)
    ):
        return uuid4().hex
    return cleaned


def _new_traceparent() -> tuple[str, str, str]:
    trace_id = uuid4().hex
    span_id = uuid4().hex[:16]
    return f"00-{trace_id}-{span_id}-01", trace_id, span_id


def _safe_traceparent(value: str | None) -> tuple[str, str, str]:
    cleaned = (value or "").strip()
    match = TRACEPARENT_PATTERN.fullmatch(cleaned)
    if match is None or match.group("version").lower() == "ff":
        return _new_traceparent()

    trace_id = match.group("trace_id").lower()
    span_id = match.group("span_id").lower()
    if trace_id == "0" * 32 or span_id == "0" * 16:
        return _new_traceparent()

    trace_flags = match.group("trace_flags").lower()
    traceparent = f"{match.group('version').lower()}-{trace_id}-{span_id}-{trace_flags}"
    return traceparent, trace_id, span_id


def _route_template(request: Request) -> str:
    route = request.scope.get("route")
    template = getattr(route, "path", None)
    if isinstance(template, str) and template:
        return template
    return "unmatched"


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, logger_name: str = API_LOGGER_NAME):  # type: ignore[no-untyped-def]
        super().__init__(app)
        self.logger = logging.getLogger(logger_name)

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = _safe_request_id(request.headers.get(REQUEST_ID_HEADER))
        traceparent, trace_id, span_id = _safe_traceparent(
            request.headers.get(TRACEPARENT_HEADER)
        )
        request.state.request_id = request_id
        request.state.traceparent = traceparent
        request.state.trace_id = trace_id
        request.state.span_id = span_id
        started_at = perf_counter()
        status_code = 500

        try:
            response: Response = await call_next(request)
            status_code = response.status_code
            response.headers.setdefault(REQUEST_ID_HEADER, request_id)
            response.headers.setdefault(TRACEPARENT_HEADER, traceparent)
            return response
        finally:
            duration_ms = round((perf_counter() - started_at) * 1000, 2)
            self.logger.info(
                json.dumps(
                    {
                        "event": "api_request",
                        "request_id": request_id,
                        "trace_id": trace_id,
                        "span_id": span_id,
                        "method": request.method,
                        "route": _route_template(request),
                        "status_code": status_code,
                        "duration_ms": duration_ms,
                    },
                    sort_keys=True,
                )
            )


def configure_request_logging(app: FastAPI) -> None:
    app.add_middleware(RequestLoggingMiddleware)
