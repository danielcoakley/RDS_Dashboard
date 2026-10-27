from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from typing import Any, Protocol
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .domain import MarketingInquiry
from .notifications import (
    MARKETING_INQUIRY_RECIPIENT_EMAIL_ENV,
    marketing_inquiry_recipient_email_from_env,
)
from .runtime_config import PRODUCTION_RUNTIME, runtime_env_from_env


MARKETING_INQUIRY_CRM_WEBHOOK_URL_ENV = "SAAS_MARKETING_INQUIRY_CRM_WEBHOOK_URL"
MARKETING_INQUIRY_CRM_WEBHOOK_SECRET_ENV = "SAAS_MARKETING_INQUIRY_CRM_WEBHOOK_SECRET"
MARKETING_INQUIRY_CRM_WEBHOOK_TIMEOUT_SECONDS_ENV = (
    "SAAS_MARKETING_INQUIRY_CRM_WEBHOOK_TIMEOUT_SECONDS"
)
MARKETING_INQUIRY_CRM_WEBHOOK_SECRET_MIN_LENGTH = 32
CRM_WEBHOOK_SIGNATURE_HEADER = "X-RDS-Signature"
CRM_WEBHOOK_INQUIRY_ID_HEADER = "X-RDS-Marketing-Inquiry-Id"


class MarketingInquiryCrmSender(Protocol):
    def send(self, inquiry: MarketingInquiry) -> None:
        """Deliver a stored public marketing inquiry to the CRM handoff boundary."""


def _env_text(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _float_env(name: str, default: float) -> float:
    value = _env_text(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return parsed


def _normalize_webhook_url(value: str, runtime_env: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{MARKETING_INQUIRY_CRM_WEBHOOK_URL_ENV} must be an HTTP(S) URL")
    if runtime_env == PRODUCTION_RUNTIME and parsed.scheme != "https":
        raise ValueError(
            f"{MARKETING_INQUIRY_CRM_WEBHOOK_URL_ENV} must use https in production"
        )
    return value


def marketing_inquiry_crm_webhook_url_from_env(runtime_env: str | None = None) -> str | None:
    url = _env_text(MARKETING_INQUIRY_CRM_WEBHOOK_URL_ENV)
    if url is None:
        return None
    active_runtime_env = runtime_env or runtime_env_from_env()
    return _normalize_webhook_url(url, active_runtime_env)


def marketing_inquiry_crm_webhook_secret_from_env(runtime_env: str | None = None) -> str | None:
    secret = _env_text(MARKETING_INQUIRY_CRM_WEBHOOK_SECRET_ENV)
    active_runtime_env = runtime_env or runtime_env_from_env()
    webhook_url = marketing_inquiry_crm_webhook_url_from_env(active_runtime_env)
    if (
        active_runtime_env == PRODUCTION_RUNTIME
        and webhook_url is not None
        and secret is None
    ):
        raise ValueError(
            f"{MARKETING_INQUIRY_CRM_WEBHOOK_SECRET_ENV} is required when CRM webhook "
            "handoff is configured in production"
        )
    if (
        active_runtime_env == PRODUCTION_RUNTIME
        and webhook_url is not None
        and secret is not None
        and len(secret) < MARKETING_INQUIRY_CRM_WEBHOOK_SECRET_MIN_LENGTH
    ):
        raise ValueError(
            f"{MARKETING_INQUIRY_CRM_WEBHOOK_SECRET_ENV} must be at least "
            f"{MARKETING_INQUIRY_CRM_WEBHOOK_SECRET_MIN_LENGTH} characters in production"
        )
    return secret


def marketing_inquiry_crm_webhook_timeout_seconds_from_env() -> float:
    return _float_env(MARKETING_INQUIRY_CRM_WEBHOOK_TIMEOUT_SECONDS_ENV, 5.0)


def marketing_inquiry_crm_payload(inquiry: MarketingInquiry) -> dict[str, Any]:
    return {
        "id": inquiry.id,
        "email": inquiry.email,
        "name": inquiry.name,
        "organization_name": inquiry.organization_name,
        "role": inquiry.role,
        "intended_use": inquiry.intended_use,
        "message": inquiry.message,
        "source_page": inquiry.source_page,
        "status": inquiry.status.value,
        "created_at": inquiry.created_at.isoformat(),
    }


@dataclass(frozen=True)
class WebhookMarketingInquiryCrmSender:
    url: str
    secret: str | None = None
    timeout_seconds: float = 5.0
    opener: Any = urlopen

    def send(self, inquiry: MarketingInquiry) -> None:
        body = json.dumps(
            marketing_inquiry_crm_payload(inquiry),
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "rds-saas-api/marketing-handoff",
            CRM_WEBHOOK_INQUIRY_ID_HEADER: inquiry.id,
        }
        if self.secret is not None:
            digest = hmac.new(
                self.secret.encode("utf-8"),
                body,
                hashlib.sha256,
            ).hexdigest()
            headers[CRM_WEBHOOK_SIGNATURE_HEADER] = f"sha256={digest}"

        request = Request(self.url, data=body, headers=headers, method="POST")
        with self.opener(request, timeout=self.timeout_seconds) as response:
            status_code = getattr(response, "status", getattr(response, "code", 200))
            if status_code >= 400:
                raise RuntimeError(f"CRM handoff webhook returned HTTP {status_code}")


def marketing_inquiry_crm_sender_from_env(
    runtime_env: str | None = None,
) -> MarketingInquiryCrmSender | None:
    active_runtime_env = runtime_env or runtime_env_from_env()
    webhook_url = marketing_inquiry_crm_webhook_url_from_env(active_runtime_env)
    if webhook_url is None:
        return None
    return WebhookMarketingInquiryCrmSender(
        url=webhook_url,
        secret=marketing_inquiry_crm_webhook_secret_from_env(active_runtime_env),
        timeout_seconds=marketing_inquiry_crm_webhook_timeout_seconds_from_env(),
    )


def marketing_inquiry_crm_configuration_status(
    runtime_env: str | None = None,
) -> tuple[bool, str]:
    active_runtime_env = runtime_env or runtime_env_from_env()
    try:
        webhook_url = marketing_inquiry_crm_webhook_url_from_env(active_runtime_env)
        secret = marketing_inquiry_crm_webhook_secret_from_env(active_runtime_env)
        marketing_inquiry_crm_webhook_timeout_seconds_from_env()
    except ValueError as exc:
        return False, str(exc)
    if webhook_url is None:
        return False, f"{MARKETING_INQUIRY_CRM_WEBHOOK_URL_ENV} is not configured"
    if secret is None:
        return True, f"{MARKETING_INQUIRY_CRM_WEBHOOK_URL_ENV} configured without signing secret"
    return (
        True,
        f"{MARKETING_INQUIRY_CRM_WEBHOOK_URL_ENV} and "
        f"{MARKETING_INQUIRY_CRM_WEBHOOK_SECRET_ENV} configured",
    )


def marketing_inquiry_handoff_configuration_status(
    runtime_env: str | None = None,
) -> tuple[bool, str]:
    active_runtime_env = runtime_env or runtime_env_from_env()
    try:
        recipient_email = marketing_inquiry_recipient_email_from_env()
    except ValueError as exc:
        return False, str(exc)

    crm_configured, crm_detail = marketing_inquiry_crm_configuration_status(active_runtime_env)
    crm_url_present = _env_text(MARKETING_INQUIRY_CRM_WEBHOOK_URL_ENV) is not None
    if crm_url_present and not crm_configured:
        return False, crm_detail

    configured_channels: list[str] = []
    if recipient_email is not None:
        configured_channels.append(MARKETING_INQUIRY_RECIPIENT_EMAIL_ENV)
    if crm_configured:
        configured_channels.append(MARKETING_INQUIRY_CRM_WEBHOOK_URL_ENV)

    if configured_channels:
        return True, "Marketing inquiry handoff configured via " + ", ".join(
            configured_channels
        )
    return (
        False,
        f"{MARKETING_INQUIRY_RECIPIENT_EMAIL_ENV} or "
        f"{MARKETING_INQUIRY_CRM_WEBHOOK_URL_ENV} must be configured",
    )
