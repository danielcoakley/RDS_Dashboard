from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage as SmtpMessage
import hashlib
import hmac
import json
import os
import re
import smtplib
from typing import Any, Mapping, Protocol
from urllib.parse import urlparse
from uuid import uuid4

from .domain import (
    NotificationProviderEvent,
    NotificationProviderEventStatus,
    NotificationStatus,
)
from .runtime_config import PRODUCTION_RUNTIME, runtime_env_from_env
from .store import SaaSStore


EMAIL_BACKEND_ENV = "SAAS_EMAIL_BACKEND"
EMAIL_BACKEND_MEMORY = "memory"
EMAIL_BACKEND_SMTP = "smtp"
SUPPORTED_EMAIL_BACKENDS = {EMAIL_BACKEND_MEMORY, EMAIL_BACKEND_SMTP}
SMTP_HOST_ENV = "SAAS_SMTP_HOST"
SMTP_PORT_ENV = "SAAS_SMTP_PORT"
SMTP_USERNAME_ENV = "SAAS_SMTP_USERNAME"
SMTP_PASSWORD_ENV = "SAAS_SMTP_PASSWORD"
SMTP_FROM_EMAIL_ENV = "SAAS_SMTP_FROM_EMAIL"
SMTP_USE_TLS_ENV = "SAAS_SMTP_USE_TLS"
SMTP_TIMEOUT_SECONDS_ENV = "SAAS_SMTP_TIMEOUT_SECONDS"
EMAIL_DELIVERY_VERIFICATION_URL_ENV = "SAAS_EMAIL_DELIVERY_VERIFICATION_URL"
EMAIL_DELIVERY_VERIFIED_AT_ENV = "SAAS_EMAIL_DELIVERY_VERIFIED_AT"
EMAIL_DELIVERY_VERIFICATION_MAX_AGE_DAYS_ENV = (
    "SAAS_EMAIL_DELIVERY_VERIFICATION_MAX_AGE_DAYS"
)
DEFAULT_EMAIL_DELIVERY_VERIFICATION_MAX_AGE_DAYS = 30
MARKETING_INQUIRY_RECIPIENT_EMAIL_ENV = "SAAS_MARKETING_INQUIRY_RECIPIENT_EMAIL"
NOTIFICATION_MAX_ATTEMPTS_ENV = "SAAS_NOTIFICATION_MAX_ATTEMPTS"
NOTIFICATION_RETRY_DELAY_SECONDS_ENV = "SAAS_NOTIFICATION_RETRY_DELAY_SECONDS"
NOTIFICATION_PROVIDER_WEBHOOK_SECRET_ENV = "SAAS_NOTIFICATION_PROVIDER_WEBHOOK_SECRET"
NOTIFICATION_PROVIDER_WEBHOOK_SECRET_MIN_LENGTH = 32
NOTIFICATION_PROVIDER_SIGNATURE_HEADER = "X-RDS-Signature"
NOTIFICATION_PROVIDER_TIMESTAMP_HEADER = "X-RDS-Timestamp"
NOTIFICATION_PROVIDER_WEBHOOK_TOLERANCE_SECONDS_ENV = (
    "SAAS_NOTIFICATION_PROVIDER_WEBHOOK_TOLERANCE_SECONDS"
)
DEFAULT_NOTIFICATION_PROVIDER_WEBHOOK_TOLERANCE_SECONDS = 300
MAX_NOTIFICATION_PROVIDER_WEBHOOK_TOLERANCE_SECONDS = 86400
DEFAULT_NOTIFICATION_MAX_ATTEMPTS = 3
DEFAULT_NOTIFICATION_RETRY_DELAY_SECONDS = 300.0
PROVIDER_DETAIL_MAX_LENGTH = 200
PROVIDER_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
PROVIDER_EVENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,160}$")
TERMINAL_PROVIDER_FAILURE_STATUSES = {
    NotificationProviderEventStatus.BOUNCED,
    NotificationProviderEventStatus.COMPLAINED,
    NotificationProviderEventStatus.DROPPED,
}


class NotificationProviderWebhookConfigurationError(RuntimeError):
    pass


class NotificationProviderWebhookVerificationError(RuntimeError):
    pass


class NotificationProviderWebhookPayloadError(ValueError):
    pass


@dataclass(frozen=True)
class EmailMessage:
    notification_id: str
    organization_id: str
    recipient_user_id: str
    to_email: str
    subject: str
    body: str


class EmailSender(Protocol):
    def send(self, message: EmailMessage) -> None:
        """Deliver one email message or raise an exception for retry/failure handling."""


@dataclass
class InMemoryEmailSender:
    sent_messages: list[EmailMessage] = field(default_factory=list)
    fail_notification_ids: set[str] = field(default_factory=set)

    def send(self, message: EmailMessage) -> None:
        if message.notification_id in self.fail_notification_ids:
            raise RuntimeError(f"Simulated email delivery failure for {message.notification_id}")
        self.sent_messages.append(message)


@dataclass(frozen=True)
class SmtpEmailSender:
    host: str
    port: int
    from_email: str
    username: str | None = None
    password: str | None = None
    use_tls: bool = True
    timeout_seconds: float = 10.0
    smtp_factory: Any = smtplib.SMTP

    def send(self, message: EmailMessage) -> None:
        smtp_message = SmtpMessage()
        smtp_message["From"] = self.from_email
        smtp_message["To"] = message.to_email
        smtp_message["Subject"] = message.subject
        smtp_message["X-RDS-Organization-Id"] = message.organization_id
        smtp_message["X-RDS-Notification-Id"] = message.notification_id
        smtp_message.set_content(message.body)

        with self.smtp_factory(
            self.host,
            self.port,
            timeout=self.timeout_seconds,
        ) as smtp:
            if self.use_tls:
                smtp.starttls()
            if self.username is not None:
                smtp.login(self.username, self.password or "")
            smtp.send_message(smtp_message)


@dataclass(frozen=True)
class NotificationDeliverySummary:
    attempted: int
    sent: int
    failed: int
    retry_scheduled: int = 0


@dataclass(frozen=True)
class NotificationProviderWebhookResult:
    provider: str
    provider_event_id: str
    notification_id: str
    organization_id: str
    status: NotificationProviderEventStatus
    duplicate: bool
    notification_status: NotificationStatus


def _env_text(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _email_backend_from_env() -> str:
    backend = os.getenv(EMAIL_BACKEND_ENV, EMAIL_BACKEND_MEMORY).strip().lower()
    if backend not in SUPPORTED_EMAIL_BACKENDS:
        raise ValueError(
            f"{EMAIL_BACKEND_ENV} must be one of: {', '.join(sorted(SUPPORTED_EMAIL_BACKENDS))}"
        )
    return backend


def _bool_env(name: str, default: bool) -> bool:
    value = _env_text(name)
    if value is None:
        return default
    lowered = value.lower()
    if lowered in {"1", "true", "yes"}:
        return True
    if lowered in {"0", "false", "no"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _int_env(name: str, default: int) -> int:
    value = _env_text(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be greater than zero")
    return parsed


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


def _require_https_url(name: str, runtime_env: str) -> str | None:
    value = _env_text(name)
    if value is None:
        if runtime_env == PRODUCTION_RUNTIME:
            raise ValueError(f"{name} is required in production")
        return None

    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{name} must be a URL")
    if runtime_env == PRODUCTION_RUNTIME and parsed.scheme != "https":
        raise ValueError(f"{name} must use https in production")
    return value


def email_delivery_verification_max_age_days_from_env() -> int:
    return _int_env(
        EMAIL_DELIVERY_VERIFICATION_MAX_AGE_DAYS_ENV,
        DEFAULT_EMAIL_DELIVERY_VERIFICATION_MAX_AGE_DAYS,
    )


def _parse_email_delivery_verified_at(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"{EMAIL_DELIVERY_VERIFIED_AT_ENV} must be an ISO-8601 datetime"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{EMAIL_DELIVERY_VERIFIED_AT_ENV} must include a timezone")
    return parsed.astimezone(timezone.utc)


def email_delivery_verification_status(
    runtime_env: str | None = None,
    *,
    now: datetime | None = None,
) -> tuple[bool, str]:
    active_runtime_env = runtime_env or runtime_env_from_env()
    try:
        verification_url = _require_https_url(
            EMAIL_DELIVERY_VERIFICATION_URL_ENV,
            active_runtime_env,
        )
        max_age_days = email_delivery_verification_max_age_days_from_env()
    except ValueError as exc:
        return False, str(exc)

    verified_at = _env_text(EMAIL_DELIVERY_VERIFIED_AT_ENV)
    if active_runtime_env == PRODUCTION_RUNTIME and verified_at is None:
        return False, f"{EMAIL_DELIVERY_VERIFIED_AT_ENV} is required in production"
    if verified_at is None:
        return True, "email delivery verification optional in development"

    try:
        observed_at = _parse_email_delivery_verified_at(verified_at)
    except ValueError as exc:
        return False, str(exc)

    effective_now = _utc_now(now)
    if observed_at > effective_now:
        return False, f"{EMAIL_DELIVERY_VERIFIED_AT_ENV} cannot be in the future"
    age_days = (effective_now - observed_at).days
    if active_runtime_env == PRODUCTION_RUNTIME and age_days > max_age_days:
        return False, f"{EMAIL_DELIVERY_VERIFIED_AT_ENV} is older than {max_age_days} days"

    detail = f"email delivery verified age {age_days} day(s)"
    if verification_url:
        detail = f"{EMAIL_DELIVERY_VERIFICATION_URL_ENV} configured; {detail}"
    return True, detail


def smtp_email_sender_from_env() -> SmtpEmailSender:
    host = _env_text(SMTP_HOST_ENV)
    from_email = _env_text(SMTP_FROM_EMAIL_ENV)
    if host is None:
        raise ValueError(f"{SMTP_HOST_ENV} is required when {EMAIL_BACKEND_ENV}=smtp")
    if from_email is None:
        raise ValueError(f"{SMTP_FROM_EMAIL_ENV} is required when {EMAIL_BACKEND_ENV}=smtp")

    username = _env_text(SMTP_USERNAME_ENV)
    return SmtpEmailSender(
        host=host,
        port=_int_env(SMTP_PORT_ENV, 587),
        from_email=from_email,
        username=username,
        password=_env_text(SMTP_PASSWORD_ENV),
        use_tls=_bool_env(SMTP_USE_TLS_ENV, True),
        timeout_seconds=_float_env(SMTP_TIMEOUT_SECONDS_ENV, 10.0),
    )


def email_sender_from_env() -> EmailSender:
    backend = _email_backend_from_env()
    if backend == EMAIL_BACKEND_MEMORY:
        return InMemoryEmailSender()
    if backend == EMAIL_BACKEND_SMTP:
        return smtp_email_sender_from_env()
    raise ValueError(f"Unsupported email backend: {backend}")


def notification_max_attempts_from_env() -> int:
    return _int_env(NOTIFICATION_MAX_ATTEMPTS_ENV, DEFAULT_NOTIFICATION_MAX_ATTEMPTS)


def notification_retry_delay_seconds_from_env() -> float:
    return _float_env(
        NOTIFICATION_RETRY_DELAY_SECONDS_ENV,
        DEFAULT_NOTIFICATION_RETRY_DELAY_SECONDS,
    )


def notification_provider_webhook_tolerance_seconds_from_env() -> int:
    tolerance_seconds = _int_env(
        NOTIFICATION_PROVIDER_WEBHOOK_TOLERANCE_SECONDS_ENV,
        DEFAULT_NOTIFICATION_PROVIDER_WEBHOOK_TOLERANCE_SECONDS,
    )
    if tolerance_seconds > MAX_NOTIFICATION_PROVIDER_WEBHOOK_TOLERANCE_SECONDS:
        raise ValueError(
            f"{NOTIFICATION_PROVIDER_WEBHOOK_TOLERANCE_SECONDS_ENV} must be less than "
            f"or equal to {MAX_NOTIFICATION_PROVIDER_WEBHOOK_TOLERANCE_SECONDS}"
        )
    return tolerance_seconds


def notification_provider_webhook_secret_from_env(
    runtime_env: str | None = None,
) -> str | None:
    secret = _env_text(NOTIFICATION_PROVIDER_WEBHOOK_SECRET_ENV)
    active_runtime_env = runtime_env or runtime_env_from_env()
    if active_runtime_env == PRODUCTION_RUNTIME and secret is None:
        raise ValueError(
            f"{NOTIFICATION_PROVIDER_WEBHOOK_SECRET_ENV} must be configured "
            "when RDS_RUNTIME_ENV=production"
        )
    if (
        active_runtime_env == PRODUCTION_RUNTIME
        and secret is not None
        and len(secret) < NOTIFICATION_PROVIDER_WEBHOOK_SECRET_MIN_LENGTH
    ):
        raise ValueError(
            f"{NOTIFICATION_PROVIDER_WEBHOOK_SECRET_ENV} must be at least "
            f"{NOTIFICATION_PROVIDER_WEBHOOK_SECRET_MIN_LENGTH} characters in production"
        )
    return secret


def notification_provider_webhook_configuration_status(
    runtime_env: str | None = None,
) -> tuple[bool, str]:
    active_runtime_env = runtime_env or runtime_env_from_env()
    try:
        secret = notification_provider_webhook_secret_from_env(active_runtime_env)
        tolerance_seconds = notification_provider_webhook_tolerance_seconds_from_env()
    except ValueError as exc:
        return False, str(exc)
    tolerance_detail = (
        f"{NOTIFICATION_PROVIDER_WEBHOOK_TOLERANCE_SECONDS_ENV}={tolerance_seconds}"
    )
    if secret is None:
        return (
            True,
            f"{NOTIFICATION_PROVIDER_WEBHOOK_SECRET_ENV} is optional in development; "
            f"{tolerance_detail}",
        )
    return True, f"{NOTIFICATION_PROVIDER_WEBHOOK_SECRET_ENV} configured; {tolerance_detail}"


def marketing_inquiry_recipient_email_from_env() -> str | None:
    email = _env_text(MARKETING_INQUIRY_RECIPIENT_EMAIL_ENV)
    if email is None:
        return None
    normalized = email.lower()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", normalized):
        raise ValueError(f"{MARKETING_INQUIRY_RECIPIENT_EMAIL_ENV} must be a valid email address")
    return normalized


def marketing_inquiry_configuration_status() -> tuple[bool, str]:
    try:
        email = marketing_inquiry_recipient_email_from_env()
    except ValueError as exc:
        return False, str(exc)
    if email is None:
        return False, f"{MARKETING_INQUIRY_RECIPIENT_EMAIL_ENV} is not configured"
    return True, f"{MARKETING_INQUIRY_RECIPIENT_EMAIL_ENV} configured"


def email_delivery_configuration_status(
    runtime_env: str | None = None,
    *,
    now: datetime | None = None,
) -> tuple[bool, str]:
    active_runtime_env = runtime_env or runtime_env_from_env()
    try:
        backend = _email_backend_from_env()
        if backend == EMAIL_BACKEND_SMTP:
            sender = smtp_email_sender_from_env()
            detail = f"{EMAIL_BACKEND_ENV}={EMAIL_BACKEND_SMTP}; {SMTP_HOST_ENV} configured"
            if sender.username is not None:
                detail = f"{detail}; {SMTP_USERNAME_ENV} configured"
            verification_ready, verification_detail = email_delivery_verification_status(
                active_runtime_env,
                now=now,
            )
            if not verification_ready:
                return False, verification_detail
            return True, f"{detail}; {verification_detail}"
    except ValueError as exc:
        return False, str(exc)

    return True, f"{EMAIL_BACKEND_ENV}={EMAIL_BACKEND_MEMORY}"


def _header_value(headers: Mapping[str, str], name: str) -> str | None:
    value = headers.get(name)
    if value is not None:
        return value
    lowered_name = name.lower()
    for header_name, header_value in headers.items():
        if header_name.lower() == lowered_name:
            return header_value
    return None


def _parse_provider_webhook_timestamp(value: str | None) -> tuple[str, datetime]:
    if value is None or not value.strip():
        raise NotificationProviderWebhookVerificationError(
            f"{NOTIFICATION_PROVIDER_TIMESTAMP_HEADER} is required"
        )
    timestamp_text = value.strip()
    try:
        parsed = datetime.fromisoformat(timestamp_text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise NotificationProviderWebhookVerificationError(
            f"{NOTIFICATION_PROVIDER_TIMESTAMP_HEADER} must be ISO-8601"
        ) from exc
    if parsed.tzinfo is None:
        raise NotificationProviderWebhookVerificationError(
            f"{NOTIFICATION_PROVIDER_TIMESTAMP_HEADER} must include a timezone"
        )
    return timestamp_text, parsed.astimezone(timezone.utc)


def _utc_now(now: datetime | None = None) -> datetime:
    effective_now = now or datetime.now(timezone.utc)
    if effective_now.tzinfo is None:
        effective_now = effective_now.replace(tzinfo=timezone.utc)
    return effective_now.astimezone(timezone.utc)


def verify_notification_provider_webhook_signature(
    payload: bytes,
    headers: Mapping[str, str],
    runtime_env: str | None = None,
    now: datetime | None = None,
) -> None:
    try:
        secret = notification_provider_webhook_secret_from_env(runtime_env)
        tolerance_seconds = notification_provider_webhook_tolerance_seconds_from_env()
    except ValueError as exc:
        raise NotificationProviderWebhookConfigurationError(str(exc)) from exc
    if secret is None:
        raise NotificationProviderWebhookConfigurationError(
            f"{NOTIFICATION_PROVIDER_WEBHOOK_SECRET_ENV} must be configured to process "
            "notification provider webhooks"
        )

    timestamp_text, signed_at = _parse_provider_webhook_timestamp(
        _header_value(headers, NOTIFICATION_PROVIDER_TIMESTAMP_HEADER)
    )
    age_seconds = abs((_utc_now(now) - signed_at).total_seconds())
    if age_seconds > tolerance_seconds:
        raise NotificationProviderWebhookVerificationError(
            "Notification provider webhook timestamp is outside tolerance"
        )

    signature = _header_value(headers, NOTIFICATION_PROVIDER_SIGNATURE_HEADER)
    if signature is None or not signature.startswith("sha256="):
        raise NotificationProviderWebhookVerificationError(
            f"{NOTIFICATION_PROVIDER_SIGNATURE_HEADER} must be a sha256 signature"
        )
    signed_payload = timestamp_text.encode("utf-8") + b"." + payload
    expected_digest = hmac.new(
        secret.encode("utf-8"),
        signed_payload,
        hashlib.sha256,
    ).hexdigest()
    presented_digest = signature.removeprefix("sha256=").strip()
    if not hmac.compare_digest(presented_digest, expected_digest):
        raise NotificationProviderWebhookVerificationError(
            "Notification provider webhook signature is invalid"
        )


def _parse_provider_timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise NotificationProviderWebhookPayloadError("occurred_at is required")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise NotificationProviderWebhookPayloadError("occurred_at must be ISO-8601") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_provider_detail(value: Any, status: NotificationProviderEventStatus) -> str:
    fallback = f"provider_status={status.value}"
    if value is None:
        return fallback
    text = str(value).strip()
    if not text:
        return fallback
    safe = re.sub(r"[^A-Za-z0-9 ._:-]", "_", text)
    return safe[:PROVIDER_DETAIL_MAX_LENGTH] or fallback


def _provider_payload_text(
    payload: dict[str, Any],
    field_name: str,
    pattern: re.Pattern[str],
) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise NotificationProviderWebhookPayloadError(f"{field_name} is required")
    normalized = value.strip()
    if not pattern.fullmatch(normalized):
        raise NotificationProviderWebhookPayloadError(f"{field_name} has unsupported characters")
    return normalized


def process_notification_provider_webhook_event(
    store: SaaSStore,
    payload: bytes,
    headers: Mapping[str, str],
    runtime_env: str | None = None,
    now: datetime | None = None,
) -> NotificationProviderWebhookResult:
    verify_notification_provider_webhook_signature(payload, headers, runtime_env, now=now)
    try:
        event_payload = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NotificationProviderWebhookPayloadError("Webhook payload must be JSON") from exc
    if not isinstance(event_payload, dict):
        raise NotificationProviderWebhookPayloadError("Webhook payload must be an object")

    provider = _provider_payload_text(
        event_payload,
        "provider",
        PROVIDER_NAME_PATTERN,
    )
    provider_event_id = _provider_payload_text(
        event_payload,
        "event_id",
        PROVIDER_EVENT_ID_PATTERN,
    )
    notification_id = _provider_payload_text(
        event_payload,
        "notification_id",
        PROVIDER_EVENT_ID_PATTERN,
    )
    try:
        event_status = NotificationProviderEventStatus(str(event_payload.get("status", "")))
    except ValueError as exc:
        raise NotificationProviderWebhookPayloadError("status is unsupported") from exc
    occurred_at = _parse_provider_timestamp(event_payload.get("occurred_at"))
    detail = _safe_provider_detail(event_payload.get("detail"), event_status)

    existing_event = store.get_notification_provider_event(
        provider=provider,
        provider_event_id=provider_event_id,
    )
    notification = store.get_notification(notification_id)
    if notification is None:
        raise NotificationProviderWebhookPayloadError("notification_id was not found")

    current_notification_status = NotificationStatus(notification["status"])
    if existing_event is not None:
        return NotificationProviderWebhookResult(
            provider=provider,
            provider_event_id=provider_event_id,
            notification_id=existing_event["notification_id"],
            organization_id=existing_event["organization_id"],
            status=NotificationProviderEventStatus(existing_event["status"]),
            duplicate=True,
            notification_status=current_notification_status,
        )

    next_notification_status = current_notification_status
    with store.conn:
        store.create_notification_provider_event(
            NotificationProviderEvent(
                id=f"notification_provider_event_{uuid4().hex}",
                provider=provider,
                provider_event_id=provider_event_id,
                notification_id=notification_id,
                organization_id=notification["organization_id"],
                status=event_status,
                detail=detail,
                occurred_at=occurred_at,
            )
        )
        if event_status == NotificationProviderEventStatus.DELIVERED:
            next_notification_status = NotificationStatus.SENT
            store.update_notification_status(
                notification_id,
                NotificationStatus.SENT,
                sent_at=occurred_at.isoformat(),
                error_message=None,
                next_attempt_at=None,
            )
        elif event_status in TERMINAL_PROVIDER_FAILURE_STATUSES:
            next_notification_status = NotificationStatus.FAILED
            store.update_notification_status(
                notification_id,
                NotificationStatus.FAILED,
                sent_at=None,
                error_message=f"provider_delivery_status={event_status.value}",
                next_attempt_at=None,
            )

    return NotificationProviderWebhookResult(
        provider=provider,
        provider_event_id=provider_event_id,
        notification_id=notification_id,
        organization_id=notification["organization_id"],
        status=event_status,
        duplicate=False,
        notification_status=next_notification_status,
    )


def process_pending_notifications(
    store: SaaSStore,
    sender: EmailSender,
    organization_id: str | None = None,
    limit: int | None = None,
    max_attempts: int | None = None,
    retry_delay_seconds: float | None = None,
    now: datetime | None = None,
) -> NotificationDeliverySummary:
    attempt_limit = max_attempts if max_attempts is not None else notification_max_attempts_from_env()
    if attempt_limit < 1:
        raise ValueError("notification max attempts must be greater than zero")
    retry_delay = (
        retry_delay_seconds
        if retry_delay_seconds is not None
        else notification_retry_delay_seconds_from_env()
    )
    if retry_delay <= 0:
        raise ValueError("notification retry delay must be greater than zero")
    effective_now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    attempted_at = effective_now.isoformat()
    pending = store.list_pending_notifications(
        organization_id=organization_id,
        limit=limit,
        due_at=attempted_at,
    )
    sent = 0
    failed = 0
    retry_scheduled = 0

    for notification in pending:
        next_attempt_count = int(notification["attempt_count"] or 0) + 1
        message = EmailMessage(
            notification_id=notification["id"],
            organization_id=notification["organization_id"],
            recipient_user_id=notification["recipient_user_id"],
            to_email=notification["recipient_email"],
            subject=notification["subject"],
            body=notification["body"],
        )

        try:
            sender.send(message)
        except Exception as exc:
            retry_allowed = next_attempt_count < attempt_limit
            status = (
                NotificationStatus.PENDING
                if retry_allowed
                else NotificationStatus.FAILED
            )
            next_attempt_at = (
                (effective_now + timedelta(seconds=retry_delay)).isoformat()
                if retry_allowed
                else None
            )
            with store.conn:
                store.update_notification_status(
                    notification["id"],
                    status,
                    error_message=str(exc),
                    attempt_count=next_attempt_count,
                    last_attempt_at=attempted_at,
                    next_attempt_at=next_attempt_at,
                )
            if retry_allowed:
                retry_scheduled += 1
            else:
                failed += 1
            continue

        with store.conn:
            store.update_notification_status(
                notification["id"],
                NotificationStatus.SENT,
                sent_at=attempted_at,
                error_message=None,
                attempt_count=next_attempt_count,
                last_attempt_at=attempted_at,
                next_attempt_at=None,
            )
        sent += 1

    return NotificationDeliverySummary(
        attempted=len(pending),
        sent=sent,
        failed=failed,
        retry_scheduled=retry_scheduled,
    )
