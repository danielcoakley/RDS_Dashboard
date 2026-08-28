from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Mapping

from .auth_context import CLERK_AUTH_MODE, AuthenticatedUser
from .clerk_sync import (
    humanize_slug,
    placeholder_email,
    platform_role_from_clerk_role,
    slug_part,
    sync_clerk_authenticated_user,
)
from .domain import Organization, User
from .store import SaaSStore
from .subscriptions import ensure_default_subscription


CLERK_WEBHOOK_SIGNING_SECRET_ENV = "CLERK_WEBHOOK_SIGNING_SECRET"
CLERK_WEBHOOK_SIGNING_SECRET_MIN_LENGTH = 32

USER_CREATED_EVENT = "user.created"
USER_UPDATED_EVENT = "user.updated"
USER_DELETED_EVENT = "user.deleted"
ORGANIZATION_CREATED_EVENT = "organization.created"
ORGANIZATION_UPDATED_EVENT = "organization.updated"
ORGANIZATION_DELETED_EVENT = "organization.deleted"
ORGANIZATION_MEMBERSHIP_CREATED_EVENT = "organizationMembership.created"
ORGANIZATION_MEMBERSHIP_UPDATED_EVENT = "organizationMembership.updated"
ORGANIZATION_MEMBERSHIP_DELETED_EVENT = "organizationMembership.deleted"

SUPPORTED_EVENT_TYPES = {
    USER_CREATED_EVENT,
    USER_UPDATED_EVENT,
    USER_DELETED_EVENT,
    ORGANIZATION_CREATED_EVENT,
    ORGANIZATION_UPDATED_EVENT,
    ORGANIZATION_DELETED_EVENT,
    ORGANIZATION_MEMBERSHIP_CREATED_EVENT,
    ORGANIZATION_MEMBERSHIP_UPDATED_EVENT,
    ORGANIZATION_MEMBERSHIP_DELETED_EVENT,
}


class ClerkWebhookConfigurationError(RuntimeError):
    pass


class ClerkWebhookVerificationError(ValueError):
    pass


@dataclass(frozen=True)
class ClerkWebhookProcessResult:
    event_type: str
    action: str
    user_id: str | None = None
    organization_id: str | None = None
    role: str | None = None


def _env_text(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def clerk_webhook_configuration_status() -> tuple[bool, str]:
    secret = _env_text(CLERK_WEBHOOK_SIGNING_SECRET_ENV)
    if secret is None:
        return False, f"{CLERK_WEBHOOK_SIGNING_SECRET_ENV} is required for Clerk webhooks"
    if len(secret) < CLERK_WEBHOOK_SIGNING_SECRET_MIN_LENGTH:
        return (
            False,
            f"{CLERK_WEBHOOK_SIGNING_SECRET_ENV} must be at least "
            f"{CLERK_WEBHOOK_SIGNING_SECRET_MIN_LENGTH} characters",
        )
    try:
        from svix.webhooks import Webhook
    except ImportError as exc:
        return False, f"svix is required for Clerk webhook verification: {exc}"
    try:
        Webhook(secret)
    except Exception as exc:
        return False, f"{CLERK_WEBHOOK_SIGNING_SECRET_ENV} is invalid: {exc}"
    return True, f"{CLERK_WEBHOOK_SIGNING_SECRET_ENV} configured"


def verify_clerk_webhook_event(
    payload: bytes,
    headers: Mapping[str, str],
    *,
    secret: str | None = None,
) -> dict[str, Any]:
    signing_secret = secret or _env_text(CLERK_WEBHOOK_SIGNING_SECRET_ENV)
    if signing_secret is None:
        raise ClerkWebhookConfigurationError(
            f"{CLERK_WEBHOOK_SIGNING_SECRET_ENV} is required for Clerk webhooks"
        )

    try:
        from svix.webhooks import Webhook, WebhookVerificationError
    except ImportError as exc:
        raise ClerkWebhookConfigurationError(
            "svix is required for Clerk webhook verification"
        ) from exc

    try:
        event = Webhook(signing_secret).verify(payload, dict(headers))
    except WebhookVerificationError as exc:
        raise ClerkWebhookVerificationError("Invalid Clerk webhook signature") from exc

    if not isinstance(event, dict):
        raise ClerkWebhookVerificationError("Clerk webhook payload must be a JSON object")
    return event


def _text(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _first_text(payload: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = _text(payload.get(key))
        if value is not None:
            return value
    return None


def _object(payload: Mapping[str, Any], *keys: str) -> dict[str, Any] | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return None


def _required_text(payload: Mapping[str, Any], *keys: str, field_name: str) -> str:
    value = _first_text(payload, *keys)
    if value is None:
        raise ValueError(f"Clerk webhook {field_name} is required")
    return value


def _email_from_user_payload(payload: Mapping[str, Any]) -> str | None:
    email = _first_text(payload, "email", "email_address", "emailAddress", "identifier")
    if email is not None and "@" in email:
        return email

    primary_email_id = _first_text(
        payload,
        "primary_email_address_id",
        "primaryEmailAddressId",
    )
    email_addresses = payload.get("email_addresses") or payload.get("emailAddresses")
    if isinstance(email_addresses, list):
        first_email: str | None = None
        for item in email_addresses:
            if isinstance(item, str):
                address = _text(item)
                address_id = None
            elif isinstance(item, dict):
                address = _first_text(item, "email_address", "emailAddress", "email")
                address_id = _first_text(item, "id")
            else:
                address = None
                address_id = None

            if address is None:
                continue
            first_email = first_email or address
            if primary_email_id is not None and address_id == primary_email_id:
                return address
        return first_email

    return None


def _display_name_from_user_payload(
    payload: Mapping[str, Any],
    *,
    email: str | None,
    user_id: str,
) -> str:
    first_name = _first_text(payload, "first_name", "firstName")
    last_name = _first_text(payload, "last_name", "lastName")
    combined_name = " ".join(part for part in (first_name, last_name) if part)
    if combined_name:
        return combined_name

    display_name = _first_text(payload, "name", "full_name", "fullName", "username")
    if display_name is not None:
        return display_name
    if email is not None:
        return email.split("@", maxsplit=1)[0]
    return user_id


def _user_from_payload(payload: Mapping[str, Any]) -> User:
    user_id = _required_text(payload, "id", "user_id", "userId", field_name="user id")
    email = _email_from_user_payload(payload) or placeholder_email(user_id)
    return User(
        id=user_id,
        email=email,
        display_name=_display_name_from_user_payload(payload, email=email, user_id=user_id),
    )


def _organization_from_payload(payload: Mapping[str, Any]) -> Organization:
    organization_id = _required_text(
        payload,
        "id",
        "organization_id",
        "organizationId",
        field_name="organization id",
    )
    slug = _first_text(payload, "slug") or organization_id
    name = _first_text(payload, "name") or humanize_slug(slug)
    return Organization(
        id=organization_id,
        name=name,
        slug=slug_part(slug),
    )


def _membership_context(payload: Mapping[str, Any]) -> tuple[AuthenticatedUser, Organization | None]:
    public_user_data = _object(payload, "public_user_data", "publicUserData") or {}
    organization_payload = _object(payload, "organization")

    user_id = _first_text(payload, "user_id", "userId") or _first_text(
        public_user_data,
        "user_id",
        "userId",
    )
    if user_id is None:
        raise ValueError("Clerk webhook membership user id is required")

    organization_id = _first_text(payload, "organization_id", "organizationId")
    organization: Organization | None = None
    if organization_payload is not None:
        organization = _organization_from_payload(organization_payload)
        organization_id = organization_id or organization.id
    if organization_id is None:
        raise ValueError("Clerk webhook membership organization id is required")

    email = _email_from_user_payload(public_user_data)
    if email is None:
        identifier = _first_text(public_user_data, "identifier")
        email = identifier if identifier is not None and "@" in identifier else None

    return (
        AuthenticatedUser(
            id=user_id,
            organization_id=organization_id,
            role=_first_text(payload, "role"),
            auth_mode=CLERK_AUTH_MODE,
            email=email,
            display_name=_display_name_from_user_payload(
                public_user_data,
                email=email,
                user_id=user_id,
            ),
            organization_slug=organization.slug if organization is not None else None,
            organization_name=organization.name if organization is not None else None,
        ),
        organization,
    )


def process_clerk_webhook_event(
    event: Mapping[str, Any],
    store: SaaSStore,
) -> ClerkWebhookProcessResult:
    event_type = _required_text(event, "type", field_name="event type")
    data = event.get("data")
    if not isinstance(data, dict):
        raise ValueError("Clerk webhook data object is required")

    if event_type not in SUPPORTED_EVENT_TYPES:
        return ClerkWebhookProcessResult(event_type=event_type, action="ignored")

    if event_type in {USER_CREATED_EVENT, USER_UPDATED_EVENT}:
        user = _user_from_payload(data)
        with store.conn:
            store.upsert_user(user)
        return ClerkWebhookProcessResult(
            event_type=event_type,
            action="user_upserted",
            user_id=user.id,
        )

    if event_type == USER_DELETED_EVENT:
        user_id = _required_text(data, "id", "user_id", "userId", field_name="user id")
        with store.conn:
            store.set_user_active(user_id, False)
            store.delete_user_memberships(user_id)
        return ClerkWebhookProcessResult(
            event_type=event_type,
            action="user_deactivated",
            user_id=user_id,
        )

    if event_type in {ORGANIZATION_CREATED_EVENT, ORGANIZATION_UPDATED_EVENT}:
        organization = _organization_from_payload(data)
        with store.conn:
            store.upsert_organization(organization)
            ensure_default_subscription(store, organization.id)
        return ClerkWebhookProcessResult(
            event_type=event_type,
            action="organization_upserted",
            organization_id=organization.id,
        )

    if event_type == ORGANIZATION_DELETED_EVENT:
        organization_id = _required_text(
            data,
            "id",
            "organization_id",
            "organizationId",
            field_name="organization id",
        )
        with store.conn:
            store.delete_organization_memberships(organization_id)
        return ClerkWebhookProcessResult(
            event_type=event_type,
            action="organization_memberships_revoked",
            organization_id=organization_id,
        )

    user, organization = _membership_context(data)
    if event_type == ORGANIZATION_MEMBERSHIP_DELETED_EVENT:
        assert user.organization_id is not None
        with store.conn:
            store.delete_membership(user.id, user.organization_id)
        return ClerkWebhookProcessResult(
            event_type=event_type,
            action="membership_deleted",
            user_id=user.id,
            organization_id=user.organization_id,
        )

    result = sync_clerk_authenticated_user(store, user)
    return ClerkWebhookProcessResult(
        event_type=event_type,
        action="membership_upserted",
        user_id=result.user.id,
        organization_id=result.organization.id,
        role=platform_role_from_clerk_role(user.role).value,
    )


def clerk_webhook_event_from_json(payload: str) -> dict[str, Any]:
    event = json.loads(payload)
    if not isinstance(event, dict):
        raise ValueError("Clerk webhook payload must be a JSON object")
    return event
