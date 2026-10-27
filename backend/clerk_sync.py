from __future__ import annotations

from dataclasses import dataclass
import re

from .auth_context import CLERK_AUTH_MODE, AuthenticatedUser
from .domain import Membership, Organization, Role, User
from .store import SaaSStore
from .subscriptions import ensure_default_subscription


PLACEHOLDER_EMAIL_DOMAIN = "clerk.local"


@dataclass(frozen=True)
class ClerkTenantSyncResult:
    user: User
    organization: Organization
    membership: Membership


def platform_role_from_clerk_role(role: str | None) -> Role:
    if role is None:
        return Role.VIEWER

    normalized = role.strip().lower()
    if normalized.startswith("org:"):
        normalized = normalized[4:]

    if normalized in {"owner", "admin"}:
        return Role.OWNER
    if normalized == "manager":
        return Role.MANAGER
    if normalized in {"viewer", "member", "basic_member"}:
        return Role.VIEWER
    return Role.VIEWER


def slug_part(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "clerk"


def placeholder_email(user_id: str) -> str:
    return f"{slug_part(user_id)}@{PLACEHOLDER_EMAIL_DOMAIN}"


def humanize_slug(value: str) -> str:
    words = [word for word in re.split(r"[-_\s]+", value.strip()) if word]
    return " ".join(word.capitalize() for word in words) or value


def _user_from_auth(user: AuthenticatedUser) -> User:
    email = user.email or placeholder_email(user.id)
    display_name = user.display_name or email.split("@", maxsplit=1)[0] or user.id
    return User(id=user.id, email=email, display_name=display_name)


def _organization_from_auth(user: AuthenticatedUser) -> Organization:
    if user.organization_id is None:
        raise ValueError("Verified Clerk token must include an active organization context")

    slug = slug_part(user.organization_slug or user.organization_id)
    name = user.organization_name or humanize_slug(user.organization_slug or user.organization_id)
    return Organization(id=user.organization_id, name=name, slug=slug)


def sync_clerk_authenticated_user(
    store: SaaSStore,
    user: AuthenticatedUser,
) -> ClerkTenantSyncResult:
    if user.auth_mode != CLERK_AUTH_MODE:
        raise ValueError("Clerk tenant sync requires a verified Clerk-authenticated user")

    synced_user = _user_from_auth(user)
    organization = _organization_from_auth(user)
    membership = Membership(
        user_id=synced_user.id,
        organization_id=organization.id,
        role=platform_role_from_clerk_role(user.role),
    )

    with store.conn:
        store.upsert_user(synced_user)
        store.upsert_organization(organization)
        store.upsert_membership(membership)
        ensure_default_subscription(store, organization.id)

    return ClerkTenantSyncResult(
        user=synced_user,
        organization=organization,
        membership=membership,
    )
