from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import hmac
import json
import os
from typing import Any, Mapping
from urllib.parse import urlencode, urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from .access_control import require_permission
from .domain import (
    AuditAction,
    AuditEvent,
    OrganizationSubscription,
    SubscriptionStatus,
    User,
)
from .rbac import Action
from .runtime_config import PRODUCTION_RUNTIME
from .store import SaaSStore
from .subscriptions import BILLING_PERIOD_DAYS, ensure_default_subscription


BILLING_PROVIDER_ENV = "SAAS_BILLING_PROVIDER"
BILLING_APP_BASE_URL_ENV = "SAAS_BILLING_APP_BASE_URL"
BILLING_WEBHOOK_SECRET_ENV = "SAAS_BILLING_WEBHOOK_SECRET"
BILLING_WEBHOOK_SECRET_MIN_LENGTH = 32
BILLING_WEBHOOK_TOLERANCE_SECONDS_ENV = "SAAS_BILLING_WEBHOOK_TOLERANCE_SECONDS"
DEFAULT_BILLING_WEBHOOK_TOLERANCE_SECONDS = 300
MAX_BILLING_WEBHOOK_TOLERANCE_SECONDS = 86400
BILLING_OPERATIONS_URL_ENV = "SAAS_BILLING_OPERATIONS_URL"
BILLING_OPERATIONS_VERIFIED_AT_ENV = "SAAS_BILLING_OPERATIONS_VERIFIED_AT"
BILLING_OPERATIONS_MAX_AGE_DAYS_ENV = "SAAS_BILLING_OPERATIONS_MAX_AGE_DAYS"
DEFAULT_BILLING_OPERATIONS_MAX_AGE_DAYS = 30
LOCAL_BILLING_PROVIDER = "local"
STRIPE_BILLING_PROVIDER = "stripe"
STRIPE_SECRET_KEY_ENV = "STRIPE_SECRET_KEY"
STRIPE_PRICE_ID_ENV_PREFIX = "SAAS_STRIPE_PRICE_ID_"
STRIPE_API_BASE_URL_ENV = "SAAS_STRIPE_API_BASE_URL"
SUPPORTED_BILLING_PROVIDERS = {LOCAL_BILLING_PROVIDER, STRIPE_BILLING_PROVIDER}
SYSTEM_BILLING_USER_ID = "system_billing"
SYSTEM_BILLING_EMAIL = "billing-system@rds.local"


class BillingConfigurationError(RuntimeError):
    pass


class BillingWebhookVerificationError(ValueError):
    pass


@dataclass(frozen=True)
class BillingCheckoutSession:
    organization_id: str
    plan_id: str
    provider: str
    checkout_session_id: str
    checkout_url: str


@dataclass(frozen=True)
class BillingPortalSession:
    organization_id: str
    provider: str
    portal_session_id: str
    portal_url: str


@dataclass(frozen=True)
class BillingWebhookResult:
    provider: str
    event_id: str
    event_type: str
    organization_id: str | None
    action: str
    duplicate: bool
    plan_id: str | None = None
    status: str | None = None
    external_customer_id: str | None = None
    external_subscription_id: str | None = None


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def billing_provider_from_env() -> str:
    provider = os.getenv(BILLING_PROVIDER_ENV, LOCAL_BILLING_PROVIDER).strip().lower()
    if provider not in SUPPORTED_BILLING_PROVIDERS:
        raise BillingConfigurationError(
            f"{BILLING_PROVIDER_ENV} must be one of: "
            f"{', '.join(sorted(SUPPORTED_BILLING_PROVIDERS))}"
        )
    return provider


def billing_webhook_tolerance_seconds_from_env() -> int:
    value = os.getenv(BILLING_WEBHOOK_TOLERANCE_SECONDS_ENV)
    if value is None or not value.strip():
        return DEFAULT_BILLING_WEBHOOK_TOLERANCE_SECONDS
    try:
        tolerance_seconds = int(value.strip())
    except ValueError as exc:
        raise BillingConfigurationError(
            f"{BILLING_WEBHOOK_TOLERANCE_SECONDS_ENV} must be an integer"
        ) from exc
    if tolerance_seconds < 1:
        raise BillingConfigurationError(
            f"{BILLING_WEBHOOK_TOLERANCE_SECONDS_ENV} must be greater than zero"
        )
    if tolerance_seconds > MAX_BILLING_WEBHOOK_TOLERANCE_SECONDS:
        raise BillingConfigurationError(
            f"{BILLING_WEBHOOK_TOLERANCE_SECONDS_ENV} must be less than or equal to "
            f"{MAX_BILLING_WEBHOOK_TOLERANCE_SECONDS}"
        )
    return tolerance_seconds


def _configured_value(name: str) -> str:
    return os.getenv(name, "").strip()


def _parse_positive_int_env(name: str, default: int) -> int:
    value = _configured_value(name)
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise BillingConfigurationError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise BillingConfigurationError(f"{name} must be positive")
    return parsed


def billing_operations_max_age_days_from_env() -> int:
    return _parse_positive_int_env(
        BILLING_OPERATIONS_MAX_AGE_DAYS_ENV,
        DEFAULT_BILLING_OPERATIONS_MAX_AGE_DAYS,
    )


def _require_https_url(name: str, runtime_env: str) -> str | None:
    value = _configured_value(name)
    if not value:
        if runtime_env == PRODUCTION_RUNTIME:
            raise BillingConfigurationError(f"{name} is required in production")
        return None

    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise BillingConfigurationError(f"{name} must be a URL")
    if runtime_env == PRODUCTION_RUNTIME and parsed.scheme != "https":
        raise BillingConfigurationError(f"{name} must use https in production")
    return value


def _parse_billing_operations_verified_at(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BillingConfigurationError(
            f"{BILLING_OPERATIONS_VERIFIED_AT_ENV} must be an ISO-8601 datetime"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise BillingConfigurationError(
            f"{BILLING_OPERATIONS_VERIFIED_AT_ENV} must include a timezone"
        )
    return parsed.astimezone(timezone.utc)


def billing_operations_evidence_status(
    runtime_env: str,
    *,
    now: datetime | None = None,
) -> tuple[bool, str]:
    try:
        operations_url = _require_https_url(BILLING_OPERATIONS_URL_ENV, runtime_env)
        max_age_days = billing_operations_max_age_days_from_env()
    except BillingConfigurationError as exc:
        return False, str(exc)

    verified_at = _configured_value(BILLING_OPERATIONS_VERIFIED_AT_ENV)
    if runtime_env == PRODUCTION_RUNTIME and not verified_at:
        return False, f"{BILLING_OPERATIONS_VERIFIED_AT_ENV} is required in production"
    if not verified_at:
        return True, "billing operations evidence optional in development"

    try:
        observed_at = _parse_billing_operations_verified_at(verified_at)
    except BillingConfigurationError as exc:
        return False, str(exc)

    effective_now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if observed_at > effective_now:
        return False, f"{BILLING_OPERATIONS_VERIFIED_AT_ENV} cannot be in the future"
    age_days = (effective_now - observed_at).days
    if runtime_env == PRODUCTION_RUNTIME and age_days > max_age_days:
        return (
            False,
            f"{BILLING_OPERATIONS_VERIFIED_AT_ENV} is older than {max_age_days} days",
        )

    detail = f"billing operations verified age {age_days} day(s)"
    if operations_url:
        detail = f"{BILLING_OPERATIONS_URL_ENV} configured; {detail}"
    return True, detail


def billing_configuration_status(
    runtime_env: str,
    *,
    now: datetime | None = None,
) -> tuple[bool, str]:
    try:
        provider = billing_provider_from_env()
        tolerance_seconds = billing_webhook_tolerance_seconds_from_env()
    except BillingConfigurationError as exc:
        return False, str(exc)

    if runtime_env == PRODUCTION_RUNTIME and provider == LOCAL_BILLING_PROVIDER:
        return (
            False,
            f"{BILLING_PROVIDER_ENV}={LOCAL_BILLING_PROVIDER} is local-only; "
            "configure a payment-provider billing adapter before production",
        )
    if provider == STRIPE_BILLING_PROVIDER and not os.getenv(STRIPE_SECRET_KEY_ENV):
        return False, f"{STRIPE_SECRET_KEY_ENV} is required when {BILLING_PROVIDER_ENV}=stripe"
    if provider == STRIPE_BILLING_PROVIDER and not _stripe_price_id("team"):
        return (
            False,
            f"{STRIPE_PRICE_ID_ENV_PREFIX}TEAM is required when {BILLING_PROVIDER_ENV}=stripe",
        )
    if runtime_env == PRODUCTION_RUNTIME and not os.getenv(BILLING_WEBHOOK_SECRET_ENV):
        return False, f"{BILLING_WEBHOOK_SECRET_ENV} is required for billing webhooks"
    billing_webhook_secret = os.getenv(BILLING_WEBHOOK_SECRET_ENV)
    if (
        runtime_env == PRODUCTION_RUNTIME
        and billing_webhook_secret is not None
        and len(billing_webhook_secret.strip()) < BILLING_WEBHOOK_SECRET_MIN_LENGTH
    ):
        return (
            False,
            f"{BILLING_WEBHOOK_SECRET_ENV} must be at least "
            f"{BILLING_WEBHOOK_SECRET_MIN_LENGTH} characters in production",
        )
    operations_ready, operations_detail = billing_operations_evidence_status(
        runtime_env,
        now=now,
    )
    if not operations_ready:
        return False, operations_detail
    if billing_webhook_secret:
        return (
            True,
            f"{BILLING_PROVIDER_ENV}={provider}; billing webhook signatures enabled; "
            f"{BILLING_WEBHOOK_TOLERANCE_SECONDS_ENV}={tolerance_seconds}; "
            f"{operations_detail}",
        )
    return (
        True,
        f"{BILLING_PROVIDER_ENV}={provider}; billing webhooks unsigned in development; "
        f"{BILLING_WEBHOOK_TOLERANCE_SECONDS_ENV}={tolerance_seconds}; "
        f"{operations_detail}",
    )


def _configured_app_base_url() -> str:
    return (os.getenv(BILLING_APP_BASE_URL_ENV) or "http://localhost:3000").rstrip("/")


def _absolute_return_url(value: str | None, default_path: str) -> str:
    selected = value or default_path
    if selected.startswith("http://") or selected.startswith("https://"):
        return selected
    if selected.startswith("/"):
        return f"{_configured_app_base_url()}{selected}"
    return f"{_configured_app_base_url()}/{selected}"


def _default_return_url(path: str = "/settings") -> str:
    return f"{_configured_app_base_url()}{path}"


def _header_value(headers: Mapping[str, str], name: str) -> str | None:
    target = name.lower()
    for key, value in headers.items():
        if key.lower() == target:
            return value
    return None


def _effective_now(now: datetime | None = None) -> datetime:
    effective_now = now or _now_utc()
    if effective_now.tzinfo is None:
        effective_now = effective_now.replace(tzinfo=timezone.utc)
    return effective_now.astimezone(timezone.utc)


def _verify_stripe_signature(
    raw_body: bytes,
    headers: Mapping[str, str],
    secret: str,
    now: datetime | None = None,
) -> None:
    signature = _header_value(headers, "Stripe-Signature")
    if not signature:
        raise BillingWebhookVerificationError("Missing Stripe billing webhook signature")

    parts: dict[str, list[str]] = {}
    for item in signature.split(","):
        key, _, value = item.partition("=")
        parts.setdefault(key, []).append(value)
    timestamps = parts.get("t", [])
    signatures = parts.get("v1", [])
    if not timestamps or not signatures:
        raise BillingWebhookVerificationError("Invalid Stripe billing webhook signature")
    try:
        signed_at = datetime.fromtimestamp(int(timestamps[0]), timezone.utc)
    except (OSError, OverflowError, ValueError) as exc:
        raise BillingWebhookVerificationError("Invalid Stripe billing webhook signature") from exc
    tolerance_seconds = billing_webhook_tolerance_seconds_from_env()
    age_seconds = abs((_effective_now(now) - signed_at).total_seconds())
    if age_seconds > tolerance_seconds:
        raise BillingWebhookVerificationError(
            "Stripe billing webhook timestamp is outside tolerance"
        )

    signed_payload = f"{timestamps[0]}.".encode("utf-8") + raw_body
    expected = hmac.new(secret.encode("utf-8"), signed_payload, sha256).hexdigest()
    if not any(hmac.compare_digest(candidate, expected) for candidate in signatures):
        raise BillingWebhookVerificationError("Invalid Stripe billing webhook signature")


def _verify_webhook_signature(
    provider: str,
    raw_body: bytes,
    headers: Mapping[str, str],
    now: datetime | None = None,
) -> None:
    secret = os.getenv(BILLING_WEBHOOK_SECRET_ENV)
    if not secret:
        return

    if provider == STRIPE_BILLING_PROVIDER:
        _verify_stripe_signature(raw_body, headers, secret, now=now)
        return

    provided = _header_value(headers, "X-RDS-Billing-Signature")
    if not provided:
        raise BillingWebhookVerificationError("Missing billing webhook signature")

    expected = hmac.new(secret.encode("utf-8"), raw_body, sha256).hexdigest()
    if not hmac.compare_digest(provided, expected):
        raise BillingWebhookVerificationError("Invalid billing webhook signature")


def _stripe_price_id(plan_id: str) -> str | None:
    return os.getenv(f"{STRIPE_PRICE_ID_ENV_PREFIX}{plan_id.upper()}")


def _stripe_api_base_url() -> str:
    return (os.getenv(STRIPE_API_BASE_URL_ENV) or "https://api.stripe.com").rstrip("/")


def _stripe_request(path: str, payload: dict[str, str]) -> dict[str, Any]:
    secret_key = os.getenv(STRIPE_SECRET_KEY_ENV)
    if not secret_key:
        raise BillingConfigurationError(f"{STRIPE_SECRET_KEY_ENV} is required for Stripe billing")

    request = Request(
        f"{_stripe_api_base_url()}{path}",
        data=urlencode(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {secret_key}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=15) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise BillingConfigurationError(
            f"Stripe billing request failed with status {exc.code}: {detail}"
        ) from exc
    except URLError as exc:
        raise BillingConfigurationError(f"Stripe billing request failed: {exc.reason}") from exc

    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise BillingConfigurationError("Stripe billing response was not an object")
    return parsed


def _stripe_checkout_session(
    *,
    organization_id: str,
    plan_id: str,
    external_customer_id: str | None,
    success_url: str | None,
    cancel_url: str | None,
) -> tuple[str, str]:
    price_id = _stripe_price_id(plan_id)
    if not price_id:
        raise BillingConfigurationError(
            f"{STRIPE_PRICE_ID_ENV_PREFIX}{plan_id.upper()} is required for Stripe checkout"
        )

    payload = {
        "mode": "subscription",
        "client_reference_id": organization_id,
        "success_url": _absolute_return_url(success_url, "/settings?status=billing-updated"),
        "cancel_url": _absolute_return_url(cancel_url, "/settings?error=billing-failed"),
        "line_items[0][price]": price_id,
        "line_items[0][quantity]": "1",
        "metadata[organization_id]": organization_id,
        "metadata[plan_id]": plan_id,
        "subscription_data[metadata][organization_id]": organization_id,
        "subscription_data[metadata][plan_id]": plan_id,
    }
    if external_customer_id:
        payload["customer"] = external_customer_id
    else:
        payload["customer_creation"] = "always"

    response = _stripe_request("/v1/checkout/sessions", payload)
    session_id = _id_value(response.get("id"))
    url = _id_value(response.get("url"))
    if session_id is None or url is None:
        raise BillingConfigurationError("Stripe checkout response did not include id and url")
    return session_id, url


def _stripe_portal_session(
    *,
    external_customer_id: str,
    return_url: str | None,
) -> tuple[str, str]:
    response = _stripe_request(
        "/v1/billing_portal/sessions",
        {
            "customer": external_customer_id,
            "return_url": _absolute_return_url(return_url, "/settings"),
        },
    )
    session_id = _id_value(response.get("id"))
    url = _id_value(response.get("url"))
    if session_id is None or url is None:
        raise BillingConfigurationError("Stripe portal response did not include id and url")
    return session_id, url


def _string_metadata(value: dict[str, object]) -> dict[str, str]:
    return {key: str(item) for key, item in value.items() if item is not None}


def _ensure_system_billing_user(store: SaaSStore) -> None:
    store.upsert_user(
        User(
            id=SYSTEM_BILLING_USER_ID,
            email=SYSTEM_BILLING_EMAIL,
            display_name="Billing System",
        )
    )


def create_billing_checkout_session(
    store: SaaSStore,
    user_id: str,
    organization_id: str,
    plan_id: str,
    *,
    success_url: str | None = None,
    cancel_url: str | None = None,
) -> BillingCheckoutSession:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.MANAGE_BILLING,
    )
    provider = billing_provider_from_env()
    plan = store.get_subscription_plan(plan_id)
    if plan is None or not plan.is_active:
        raise ValueError(f"Subscription plan {plan_id} was not found")
    if plan.monthly_price_cents is None:
        raise ValueError(f"Subscription plan {plan_id} requires sales-assisted billing")
    if plan.monthly_price_cents <= 0:
        raise ValueError(f"Subscription plan {plan_id} does not require checkout")

    with store.conn:
        ensure_default_subscription(store, organization_id)
        subscription = store.get_organization_subscription(organization_id)
        external_customer_id = (
            subscription["external_customer_id"] if subscription is not None else None
        )
        if provider == STRIPE_BILLING_PROVIDER:
            session_id, checkout_url = _stripe_checkout_session(
                organization_id=organization_id,
                plan_id=plan_id,
                external_customer_id=external_customer_id,
                success_url=success_url,
                cancel_url=cancel_url,
            )
        else:
            session_id = f"{provider}_checkout_{uuid4().hex}"
            checkout_url = (
                f"{_default_return_url('/settings')}?"
                + urlencode(
                    {
                        "status": "billing-checkout-created",
                        "checkout_session_id": session_id,
                        "organization_id": organization_id,
                        "plan_id": plan_id,
                        "success_url": _absolute_return_url(
                            success_url,
                            "/settings?status=billing-updated",
                        ),
                        "cancel_url": _absolute_return_url(
                            cancel_url,
                            "/settings?error=billing-failed",
                        ),
                    }
                )
            )
        store.create_audit_event(
            AuditEvent(
                id=f"audit_billing_checkout_{uuid4().hex}",
                organization_id=organization_id,
                actor_user_id=user_id,
                action=AuditAction.BILLING_CHECKOUT_CREATED,
                resource_type="billing_checkout",
                resource_id=session_id,
                metadata=_string_metadata(
                    {
                        "provider": provider,
                        "plan_id": plan_id,
                        "success_url": success_url,
                        "cancel_url": cancel_url,
                    }
                ),
            )
        )

    return BillingCheckoutSession(
        organization_id=organization_id,
        plan_id=plan_id,
        provider=provider,
        checkout_session_id=session_id,
        checkout_url=checkout_url,
    )


def create_billing_portal_session(
    store: SaaSStore,
    user_id: str,
    organization_id: str,
    *,
    return_url: str | None = None,
) -> BillingPortalSession:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.MANAGE_BILLING,
    )
    provider = billing_provider_from_env()
    with store.conn:
        ensure_default_subscription(store, organization_id)
    subscription = store.get_organization_subscription(organization_id)
    if subscription is None or not subscription["external_customer_id"]:
        raise ValueError("Billing portal requires a payment-provider customer")

    if provider == STRIPE_BILLING_PROVIDER:
        session_id, portal_url = _stripe_portal_session(
            external_customer_id=subscription["external_customer_id"],
            return_url=return_url,
        )
    else:
        session_id = f"{provider}_portal_{uuid4().hex}"
        portal_url = (
            f"{_default_return_url('/settings')}?"
            + urlencode(
                {
                    "status": "billing-portal-created",
                    "portal_session_id": session_id,
                    "organization_id": organization_id,
                    "return_url": _absolute_return_url(return_url, "/settings"),
                }
            )
        )
    with store.conn:
        store.create_audit_event(
            AuditEvent(
                id=f"audit_billing_portal_{uuid4().hex}",
                organization_id=organization_id,
                actor_user_id=user_id,
                action=AuditAction.BILLING_PORTAL_CREATED,
                resource_type="billing_portal",
                resource_id=session_id,
                metadata=_string_metadata(
                    {
                        "provider": provider,
                        "external_customer_id": subscription["external_customer_id"],
                        "return_url": return_url,
                    }
                ),
            )
        )

    return BillingPortalSession(
        organization_id=organization_id,
        provider=provider,
        portal_session_id=session_id,
        portal_url=portal_url,
    )


def _event_object(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data")
    if isinstance(data, dict):
        event_object = data.get("object")
        if isinstance(event_object, dict):
            return event_object
        return data
    return payload


def _metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _id_value(value: object) -> str | None:
    if isinstance(value, dict):
        nested_id = value.get("id")
        return nested_id if isinstance(nested_id, str) and nested_id.strip() else None
    if isinstance(value, str) and value.strip():
        return value
    return None


def _first_text(*values: object) -> str | None:
    for value in values:
        text = _id_value(value)
        if text is not None:
            return text
    return None


def _parse_provider_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, timezone.utc)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.isdigit():
            return datetime.fromtimestamp(int(stripped), timezone.utc)
        parsed = datetime.fromisoformat(stripped.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


def _row_datetime(row: Mapping[str, Any], key: str) -> datetime | None:
    return _parse_provider_datetime(row[key]) if row[key] is not None else None


def _status_for_event(event_type: str, event: dict[str, Any]) -> SubscriptionStatus | None:
    event_status = event.get("status")
    if event_type == "invoice.payment_failed":
        return SubscriptionStatus.PAST_DUE
    if event_type == "customer.subscription.deleted":
        return SubscriptionStatus.CANCELED
    if isinstance(event_status, str) and event_status:
        try:
            return SubscriptionStatus(event_status)
        except ValueError as exc:
            raise ValueError(f"Unsupported subscription status: {event_status}") from exc
    if event_type in {
        "checkout.session.completed",
        "customer.subscription.created",
        "customer.subscription.updated",
    }:
        return SubscriptionStatus.ACTIVE
    return None


def _resolve_subscription_row(
    store: SaaSStore,
    organization_id: str | None,
    external_customer_id: str | None,
    external_subscription_id: str | None,
) -> tuple[str | None, Mapping[str, Any] | None]:
    if organization_id:
        return organization_id, store.get_organization_subscription(organization_id)
    if external_subscription_id:
        row = store.get_organization_subscription_by_external_subscription_id(
            external_subscription_id
        )
        if row is not None:
            return row["organization_id"], row
    if external_customer_id:
        row = store.get_organization_subscription_by_external_customer_id(external_customer_id)
        if row is not None:
            return row["organization_id"], row
    return None, None


def _provider_event_from_body(
    raw_body: bytes,
    headers: Mapping[str, str],
    now: datetime | None = None,
) -> tuple[str, dict[str, Any]]:
    provider = billing_provider_from_env()
    _verify_webhook_signature(provider, raw_body, headers, now=now)
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise BillingWebhookVerificationError("Billing webhook payload is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise BillingWebhookVerificationError("Billing webhook payload must be an object")
    if not isinstance(payload.get("id"), str) or not payload["id"].strip():
        raise BillingWebhookVerificationError("Billing webhook event id is required")
    if not isinstance(payload.get("type"), str) or not payload["type"].strip():
        raise BillingWebhookVerificationError("Billing webhook event type is required")
    return provider, payload


def process_billing_webhook_event(
    store: SaaSStore,
    raw_body: bytes,
    headers: Mapping[str, str],
    now: datetime | None = None,
) -> BillingWebhookResult:
    provider, payload = _provider_event_from_body(raw_body, headers, now=now)
    event_id = payload["id"]
    event_type = payload["type"]
    event = _event_object(payload)
    metadata = _metadata(event)
    external_customer_id = _first_text(
        event.get("customer_id"),
        event.get("customer"),
        metadata.get("customer_id"),
    )
    external_subscription_id = _first_text(
        event.get("subscription_id"),
        event.get("subscription"),
        metadata.get("subscription_id"),
    )
    organization_id = _first_text(
        event.get("organization_id"),
        metadata.get("organization_id"),
    )
    plan_id = _first_text(event.get("plan_id"), metadata.get("plan_id"))
    status = _status_for_event(event_type, event)

    resolved_organization_id, existing = _resolve_subscription_row(
        store,
        organization_id,
        external_customer_id,
        external_subscription_id,
    )
    organization_id = resolved_organization_id

    if store.has_billing_event(event_id):
        return BillingWebhookResult(
            provider=provider,
            event_id=event_id,
            event_type=event_type,
            organization_id=organization_id,
            action="duplicate_ignored",
            duplicate=True,
            plan_id=plan_id or (existing["plan_id"] if existing is not None else None),
            status=existing["status"] if existing is not None else None,
            external_customer_id=external_customer_id,
            external_subscription_id=external_subscription_id,
        )

    if organization_id is None:
        raise ValueError("Billing webhook could not resolve an organization")

    with store.conn:
        ensure_default_subscription(store, organization_id)
        existing = store.get_organization_subscription(organization_id)
        if existing is None:
            raise ValueError(f"Subscription for organization {organization_id} was not found")

        store.create_billing_event(
            event_id=event_id,
            provider=provider,
            event_type=event_type,
            organization_id=organization_id,
            external_customer_id=external_customer_id,
            external_subscription_id=external_subscription_id,
            payload=payload,
        )

        if status is None:
            return BillingWebhookResult(
                provider=provider,
                event_id=event_id,
                event_type=event_type,
                organization_id=organization_id,
                action="recorded",
                duplicate=False,
                plan_id=plan_id or existing["plan_id"],
                status=existing["status"],
                external_customer_id=external_customer_id,
                external_subscription_id=external_subscription_id,
            )

        effective_plan_id = plan_id or existing["plan_id"]
        plan = store.get_subscription_plan(effective_plan_id)
        if plan is None or not plan.is_active:
            raise ValueError(f"Subscription plan {effective_plan_id} was not found")

        period_start = (
            _parse_provider_datetime(event.get("current_period_start"))
            or _row_datetime(existing, "current_period_start")
            or _now_utc()
        )
        period_end = (
            _parse_provider_datetime(event.get("current_period_end"))
            or _row_datetime(existing, "current_period_end")
            or period_start + timedelta(days=BILLING_PERIOD_DAYS)
        )
        trial_ends_at = (
            _parse_provider_datetime(event.get("trial_end"))
            or _parse_provider_datetime(event.get("trial_ends_at"))
            or (
                _row_datetime(existing, "trial_ends_at")
                if status == SubscriptionStatus.TRIALING
                else None
            )
        )

        subscription = OrganizationSubscription(
            organization_id=organization_id,
            plan_id=effective_plan_id,
            status=status,
            current_period_start=period_start,
            current_period_end=period_end,
            trial_ends_at=trial_ends_at,
            external_customer_id=external_customer_id or existing["external_customer_id"],
            external_subscription_id=external_subscription_id
            or existing["external_subscription_id"],
        )
        store.upsert_organization_subscription(subscription)
        _ensure_system_billing_user(store)
        store.create_audit_event(
            AuditEvent(
                id=f"audit_subscription_webhook_{uuid4().hex}",
                organization_id=organization_id,
                actor_user_id=SYSTEM_BILLING_USER_ID,
                action=AuditAction.SUBSCRIPTION_UPDATED,
                resource_type="subscription",
                resource_id=organization_id,
                metadata=_string_metadata(
                    {
                        "source": "billing_webhook",
                        "provider": provider,
                        "event_id": event_id,
                        "event_type": event_type,
                        "plan_id": effective_plan_id,
                        "status": status.value,
                        "external_customer_id": subscription.external_customer_id,
                        "external_subscription_id": subscription.external_subscription_id,
                    }
                ),
            )
        )

    return BillingWebhookResult(
        provider=provider,
        event_id=event_id,
        event_type=event_type,
        organization_id=organization_id,
        action="subscription_updated",
        duplicate=False,
        plan_id=effective_plan_id,
        status=status.value,
        external_customer_id=external_customer_id,
        external_subscription_id=external_subscription_id,
    )
