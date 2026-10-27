from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .domain import Membership, Role
from .rbac import Action, can_perform


class AccessDenied(PermissionError):
    """Raised when a user cannot access a tenant-scoped resource."""


@dataclass(frozen=True)
class TenantContext:
    user_id: str
    organization_id: str
    role: Role


def require_permission(
    memberships: Iterable[Membership],
    user_id: str,
    organization_id: str,
    action: Action,
) -> TenantContext:
    membership_list = list(memberships)
    if not can_perform(membership_list, user_id, organization_id, action):
        raise AccessDenied(
            f"User {user_id} cannot perform {action.value} in organization {organization_id}"
        )

    for membership in membership_list:
        if membership.user_id == user_id and membership.organization_id == organization_id:
            return TenantContext(
                user_id=user_id,
                organization_id=organization_id,
                role=membership.role,
            )

    raise AccessDenied(f"User {user_id} has no membership in organization {organization_id}")


def require_resource_scope(
    context: TenantContext,
    resource_organization_id: str,
    resource_name: str = "resource",
) -> None:
    if context.organization_id != resource_organization_id:
        raise AccessDenied(
            f"{resource_name} belongs to organization {resource_organization_id}, "
            f"not {context.organization_id}"
        )
