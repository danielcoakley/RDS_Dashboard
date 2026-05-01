from __future__ import annotations

from enum import Enum
from typing import Iterable

from .domain import Membership, Role


class Action(str, Enum):
    READ = "read"
    MANAGE_UPLOADS = "manage_uploads"
    MANAGE_RUNS = "manage_runs"
    MANAGE_REPORTS = "manage_reports"
    MANAGE_MEMBERS = "manage_members"
    MANAGE_ORGANIZATION = "manage_organization"
    VIEW_AUDIT = "view_audit"


_ROLE_ACTIONS: dict[Role, set[Action]] = {
    Role.OWNER: {
        Action.READ,
        Action.MANAGE_UPLOADS,
        Action.MANAGE_RUNS,
        Action.MANAGE_REPORTS,
        Action.MANAGE_MEMBERS,
        Action.MANAGE_ORGANIZATION,
        Action.VIEW_AUDIT,
    },
    Role.MANAGER: {
        Action.READ,
        Action.MANAGE_UPLOADS,
        Action.MANAGE_RUNS,
        Action.MANAGE_REPORTS,
        Action.VIEW_AUDIT,
    },
    Role.VIEWER: {Action.READ},
}


def _membership_for(
    memberships: Iterable[Membership],
    user_id: str,
    organization_id: str,
) -> Membership | None:
    for membership in memberships:
        if membership.user_id == user_id and membership.organization_id == organization_id:
            return membership
    return None


def can_access_organization(
    memberships: Iterable[Membership],
    user_id: str,
    organization_id: str,
) -> bool:
    return _membership_for(memberships, user_id, organization_id) is not None


def can_perform(
    memberships: Iterable[Membership],
    user_id: str,
    organization_id: str,
    action: Action,
) -> bool:
    membership = _membership_for(memberships, user_id, organization_id)
    if membership is None:
        return False
    return action in _ROLE_ACTIONS[membership.role]
