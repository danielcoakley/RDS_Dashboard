from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import os
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from .domain import (
    CspBlockedUriCategory,
    CspViolationReport,
    CspViolationReportDirective,
    CspViolationReportDisposition,
)
from .store import SaaSStore


CSP_REPORT_MAX_BYTES = 64_000
CSP_REPORT_RETENTION_DAYS_ENV = "SAAS_CSP_REPORT_RETENTION_DAYS"
DEFAULT_CSP_REPORT_RETENTION_DAYS = 30
MAX_CSP_REPORT_RETENTION_DAYS = 3650

_DIRECTIVE_ALIASES = {
    "default-src": CspViolationReportDirective.DEFAULT_SRC,
    "script-src": CspViolationReportDirective.SCRIPT_SRC,
    "script-src-elem": CspViolationReportDirective.SCRIPT_SRC,
    "script-src-attr": CspViolationReportDirective.SCRIPT_SRC,
    "style-src": CspViolationReportDirective.STYLE_SRC,
    "style-src-elem": CspViolationReportDirective.STYLE_SRC,
    "style-src-attr": CspViolationReportDirective.STYLE_SRC,
    "img-src": CspViolationReportDirective.IMG_SRC,
    "connect-src": CspViolationReportDirective.CONNECT_SRC,
    "font-src": CspViolationReportDirective.FONT_SRC,
    "frame-src": CspViolationReportDirective.FRAME_SRC,
    "object-src": CspViolationReportDirective.OBJECT_SRC,
    "base-uri": CspViolationReportDirective.BASE_URI,
    "form-action": CspViolationReportDirective.FORM_ACTION,
    "frame-ancestors": CspViolationReportDirective.FRAME_ANCESTORS,
}


class CspViolationReportPayloadError(ValueError):
    pass


class CspViolationReportTooLargeError(ValueError):
    pass


@dataclass(frozen=True)
class CspViolationReportRetentionSummary:
    retention_days: int
    cutoff_created_before: str
    deleted_count: int
    remaining_count: int


def csp_report_retention_days_from_env() -> int:
    raw_value = os.environ.get(CSP_REPORT_RETENTION_DAYS_ENV, "").strip()
    if not raw_value:
        return DEFAULT_CSP_REPORT_RETENTION_DAYS
    try:
        days = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{CSP_REPORT_RETENTION_DAYS_ENV} must be an integer") from exc
    if not 1 <= days <= MAX_CSP_REPORT_RETENTION_DAYS:
        raise ValueError(
            f"{CSP_REPORT_RETENTION_DAYS_ENV} must be between 1 and "
            f"{MAX_CSP_REPORT_RETENTION_DAYS}"
        )
    return days


def _report_object_from_payload(payload: bytes) -> dict[str, Any]:
    if len(payload) > CSP_REPORT_MAX_BYTES:
        raise CspViolationReportTooLargeError("CSP report payload is too large")
    if not payload:
        raise CspViolationReportPayloadError("CSP report payload is required")
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CspViolationReportPayloadError("CSP report payload must be JSON") from exc
    if not isinstance(decoded, dict):
        raise CspViolationReportPayloadError("CSP report payload must be an object")

    legacy_report = decoded.get("csp-report")
    if isinstance(legacy_report, dict):
        return legacy_report

    report_body = decoded.get("body")
    if isinstance(report_body, dict):
        report = dict(report_body)
        if "document-uri" not in report and isinstance(decoded.get("url"), str):
            report["document-uri"] = decoded["url"]
        return report

    return decoded


def _directive_from_value(value: Any) -> CspViolationReportDirective:
    if not isinstance(value, str):
        return CspViolationReportDirective.OTHER
    first_token = value.strip().lower().split(" ", 1)[0].strip("'\";")
    return _DIRECTIVE_ALIASES.get(first_token, CspViolationReportDirective.OTHER)


def _disposition_from_value(value: Any) -> CspViolationReportDisposition:
    if not isinstance(value, str):
        return CspViolationReportDisposition.REPORT
    try:
        return CspViolationReportDisposition(value.strip().lower())
    except ValueError:
        return CspViolationReportDisposition.REPORT


def _blocked_uri_category(value: Any) -> CspBlockedUriCategory:
    if value is None:
        return CspBlockedUriCategory.NONE
    normalized = str(value).strip().lower().strip("'\"")
    if not normalized:
        return CspBlockedUriCategory.NONE
    if normalized == "self":
        return CspBlockedUriCategory.SELF
    if normalized == "inline":
        return CspBlockedUriCategory.INLINE
    if normalized == "eval":
        return CspBlockedUriCategory.EVAL
    if normalized.startswith("data:"):
        return CspBlockedUriCategory.DATA
    if normalized.startswith("blob:"):
        return CspBlockedUriCategory.BLOB

    scheme = urlparse(normalized).scheme
    if scheme == "http":
        return CspBlockedUriCategory.HTTP
    if scheme == "https":
        return CspBlockedUriCategory.HTTPS
    return CspBlockedUriCategory.OTHER


def _document_origin(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    parsed = urlparse(value.strip())
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        return ""
    host = parsed.hostname.lower()
    netloc = f"{host}:{parsed.port}" if parsed.port is not None else host
    return f"{parsed.scheme.lower()}://{netloc}"[:255]


def _status_code(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if 0 <= parsed <= 999:
        return parsed
    return None


def _first_present(payload: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def build_csp_violation_report(
    payload: bytes,
    *,
    now: datetime | None = None,
) -> CspViolationReport:
    report_payload = _report_object_from_payload(payload)
    effective_directive = _directive_from_value(
        _first_present(report_payload, "effective-directive", "effective_directive")
    )
    violated_directive = _directive_from_value(
        _first_present(report_payload, "violated-directive", "violated_directive")
    )
    if (
        effective_directive == CspViolationReportDirective.OTHER
        and violated_directive != CspViolationReportDirective.OTHER
    ):
        effective_directive = violated_directive
    if (
        violated_directive == CspViolationReportDirective.OTHER
        and effective_directive != CspViolationReportDirective.OTHER
    ):
        violated_directive = effective_directive

    observed_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return CspViolationReport(
        id=f"csp_report_{uuid4().hex}",
        disposition=_disposition_from_value(report_payload.get("disposition")),
        effective_directive=effective_directive,
        violated_directive=violated_directive,
        blocked_uri_category=_blocked_uri_category(
            _first_present(report_payload, "blocked-uri", "blocked_uri")
        ),
        document_origin=_document_origin(
            _first_present(report_payload, "document-uri", "document_uri")
        ),
        status_code=_status_code(
            _first_present(report_payload, "status-code", "status_code")
        ),
        created_at=observed_at,
    )


def process_csp_violation_report(
    store: SaaSStore,
    payload: bytes,
    *,
    now: datetime | None = None,
) -> CspViolationReport:
    report = build_csp_violation_report(payload, now=now)
    with store.conn:
        store.create_csp_violation_report(report)
    return report


def apply_csp_violation_report_retention(
    store: SaaSStore,
    *,
    retention_days: int | None = None,
    now: datetime | None = None,
) -> CspViolationReportRetentionSummary:
    days = retention_days if retention_days is not None else csp_report_retention_days_from_env()
    if not 1 <= days <= MAX_CSP_REPORT_RETENTION_DAYS:
        raise ValueError(
            f"CSP report retention days must be between 1 and {MAX_CSP_REPORT_RETENTION_DAYS}"
        )
    observed_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    cutoff = observed_at - timedelta(days=days)
    cutoff_iso = cutoff.isoformat()
    with store.conn:
        deleted_count = store.delete_csp_violation_reports_before(cutoff_iso)
    return CspViolationReportRetentionSummary(
        retention_days=days,
        cutoff_created_before=cutoff_iso,
        deleted_count=deleted_count,
        remaining_count=store.count_csp_violation_reports(),
    )
