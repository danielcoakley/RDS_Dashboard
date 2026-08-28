from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from .access_control import require_permission
from .domain import AuditAction, AuditEvent, Membership, Role
from .rbac import Action
from .store import SaaSStore


@dataclass(frozen=True)
class MembershipAdminRecord:
    user_id: str
    organization_id: str
    role: Role
    invited_by_user_id: str | None
    email: str
    display_name: str
    is_active: bool


def _member_record_from_row(row) -> MembershipAdminRecord:
    return MembershipAdminRecord(
        user_id=row["user_id"],
        organization_id=row["organization_id"],
        role=Role(row["role"]),
        invited_by_user_id=row["invited_by_user_id"],
        email=row["email"],
        display_name=row["display_name"],
        is_active=bool(row["is_active"]),
    )


def _get_member_record(
    store: SaaSStore,
    organization_id: str,
    member_user_id: str,
) -> MembershipAdminRecord:
    for row in store.list_organization_members(organization_id):
        if row["user_id"] == member_user_id:
            return _member_record_from_row(row)
    raise ValueError(f"User {member_user_id} is not a member of organization {organization_id}")


def _owner_count(store: SaaSStore, organization_id: str) -> int:
    return sum(
        1
        for membership in store.list_memberships(organization_id=organization_id)
        if membership.role == Role.OWNER
    )


def update_membership_role(
    store: SaaSStore,
    *,
    actor_user_id: str,
    organization_id: str,
    member_user_id: str,
    role: Role,
) -> MembershipAdminRecord:
    require_permission(
        store.list_memberships(user_id=actor_user_id, organization_id=organization_id),
        user_id=actor_user_id,
        organization_id=organization_id,
        action=Action.MANAGE_MEMBERS,
    )
    current = _get_member_record(store, organization_id, member_user_id)
    if actor_user_id == member_user_id and current.role != role:
        raise ValueError("Owners cannot change their own role")
    if current.role == Role.OWNER and role != Role.OWNER and _owner_count(store, organization_id) <= 1:
        raise ValueError("Cannot remove the last owner from an organization")

    membership = Membership(
        user_id=member_user_id,
        organization_id=organization_id,
        role=role,
        invited_by_user_id=current.invited_by_user_id,
    )
    audit_event = AuditEvent(
        id=f"audit_membership_role_{uuid4().hex}",
        organization_id=organization_id,
        actor_user_id=actor_user_id,
        action=AuditAction.MEMBERSHIP_ROLE_UPDATED,
        resource_type="membership",
        resource_id=member_user_id,
        metadata={
            "target_user_id": member_user_id,
            "previous_role": current.role.value,
            "role": role.value,
        },
    )
    with store.conn:
        store.upsert_membership(membership)
        store.create_audit_event(audit_event)
    return _get_member_record(store, organization_id, member_user_id)


def remove_membership(
    store: SaaSStore,
    *,
    actor_user_id: str,
    organization_id: str,
    member_user_id: str,
) -> MembershipAdminRecord:
    require_permission(
        store.list_memberships(user_id=actor_user_id, organization_id=organization_id),
        user_id=actor_user_id,
        organization_id=organization_id,
        action=Action.MANAGE_MEMBERS,
    )
    current = _get_member_record(store, organization_id, member_user_id)
    if actor_user_id == member_user_id:
        raise ValueError("Owners cannot remove their own membership")
    if current.role == Role.OWNER and _owner_count(store, organization_id) <= 1:
        raise ValueError("Cannot remove the last owner from an organization")

    actor = _get_member_record(store, organization_id, actor_user_id)
    owned_schedules = store.list_report_schedules_by_creator(organization_id, member_user_id)
    transferred_schedule_count = len(owned_schedules)
    reassigned_recipient_count = sum(
        1
        for schedule in owned_schedules
        if schedule["recipient_email"].strip().lower() == current.email.strip().lower()
    )
    audit_event = AuditEvent(
        id=f"audit_membership_remove_{uuid4().hex}",
        organization_id=organization_id,
        actor_user_id=actor_user_id,
        action=AuditAction.MEMBERSHIP_REMOVED,
        resource_type="membership",
        resource_id=member_user_id,
        metadata={
            "target_user_id": member_user_id,
            "previous_role": current.role.value,
            "report_schedule_transfer_count": transferred_schedule_count,
            "report_schedule_recipient_reassignment_count": reassigned_recipient_count,
        },
    )
    with store.conn:
        if transferred_schedule_count:
            store.transfer_report_schedule_ownership(
                organization_id,
                from_user_id=member_user_id,
                to_user_id=actor_user_id,
                from_recipient_email=current.email,
                to_recipient_email=actor.email,
                updated_at=datetime.now(timezone.utc).isoformat(),
            )
        store.delete_membership(member_user_id, organization_id)
        store.create_audit_event(audit_event)
    return current
