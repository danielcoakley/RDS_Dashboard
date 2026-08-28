from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .access_control import require_permission
from .domain import AuditAction, AuditEvent, InviteStatus, Membership, OrganizationInvite, Role, User
from .rbac import Action
from .store import SaaSStore


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class InviteAcceptanceResult:
    invite: OrganizationInvite
    membership: Membership
    user: User


def create_organization_invite(
    store: SaaSStore,
    *,
    actor_user_id: str,
    organization_id: str,
    invite_id: str,
    email: str,
    role: Role,
) -> OrganizationInvite:
    require_permission(
        store.list_memberships(user_id=actor_user_id, organization_id=organization_id),
        user_id=actor_user_id,
        organization_id=organization_id,
        action=Action.MANAGE_MEMBERS,
    )
    invite = OrganizationInvite(
        id=invite_id,
        organization_id=organization_id,
        email=email,
        role=role,
        invited_by_user_id=actor_user_id,
    )
    audit_event = AuditEvent(
        id=f"audit_{invite_id}",
        organization_id=organization_id,
        actor_user_id=actor_user_id,
        action=AuditAction.INVITE_CREATED,
        resource_type="organization_invite",
        resource_id=invite.id,
        metadata={"email": invite.email, "role": invite.role.value},
    )
    with store.conn:
        store.create_organization_invite(invite)
        store.create_audit_event(audit_event)
    return invite


def accept_organization_invite(
    store: SaaSStore,
    *,
    invite_id: str,
    user: User,
) -> InviteAcceptanceResult:
    invite_row = store.get_organization_invite(invite_id)
    if invite_row is None:
        raise ValueError(f"Invite {invite_id} was not found")
    if invite_row["status"] != InviteStatus.PENDING.value:
        raise ValueError(f"Invite {invite_id} is not pending")
    if invite_row["email"].strip().lower() != user.email.strip().lower():
        raise ValueError("Invite email does not match accepting user email")

    invite = OrganizationInvite(
        id=invite_row["id"],
        organization_id=invite_row["organization_id"],
        email=invite_row["email"],
        role=Role(invite_row["role"]),
        invited_by_user_id=invite_row["invited_by_user_id"],
        status=InviteStatus(invite_row["status"]),
        accepted_by_user_id=invite_row["accepted_by_user_id"],
        accepted_at=datetime.fromisoformat(invite_row["accepted_at"])
        if invite_row["accepted_at"]
        else None,
    )
    membership = Membership(
        user_id=user.id,
        organization_id=invite.organization_id,
        role=invite.role,
        invited_by_user_id=invite.invited_by_user_id,
    )
    accepted_at = _utc_now_iso()
    audit_event = AuditEvent(
        id=f"audit_accept_{invite_id}",
        organization_id=invite.organization_id,
        actor_user_id=user.id,
        action=AuditAction.INVITE_ACCEPTED,
        resource_type="organization_invite",
        resource_id=invite.id,
        metadata={"email": user.email, "role": invite.role.value},
    )
    with store.conn:
        existing_user = store.conn.execute(
            "SELECT id FROM users WHERE id = ?",
            (user.id,),
        ).fetchone()
        if existing_user is None:
            store.create_user(user)
        store.add_membership(membership)
        store.accept_organization_invite(invite_id, user.id, accepted_at)
        store.create_audit_event(audit_event)

    accepted_invite = OrganizationInvite(
        id=invite.id,
        organization_id=invite.organization_id,
        email=invite.email,
        role=invite.role,
        invited_by_user_id=invite.invited_by_user_id,
        status=InviteStatus.ACCEPTED,
        accepted_by_user_id=user.id,
        accepted_at=datetime.fromisoformat(accepted_at),
    )
    return InviteAcceptanceResult(
        invite=accepted_invite,
        membership=membership,
        user=user,
    )


def revoke_organization_invite(
    store: SaaSStore,
    *,
    actor_user_id: str,
    organization_id: str,
    invite_id: str,
) -> OrganizationInvite:
    require_permission(
        store.list_memberships(user_id=actor_user_id, organization_id=organization_id),
        user_id=actor_user_id,
        organization_id=organization_id,
        action=Action.MANAGE_MEMBERS,
    )
    invite_row = store.get_organization_invite(invite_id)
    if invite_row is None:
        raise ValueError(f"Invite {invite_id} was not found")
    if invite_row["organization_id"] != organization_id:
        raise ValueError(f"Invite {invite_id} does not belong to organization {organization_id}")
    if invite_row["status"] != InviteStatus.PENDING.value:
        raise ValueError(f"Invite {invite_id} is not pending")

    revoked_invite = OrganizationInvite(
        id=invite_row["id"],
        organization_id=invite_row["organization_id"],
        email=invite_row["email"],
        role=Role(invite_row["role"]),
        invited_by_user_id=invite_row["invited_by_user_id"],
        status=InviteStatus.REVOKED,
        accepted_by_user_id=None,
        accepted_at=None,
    )
    audit_event = AuditEvent(
        id=f"audit_revoke_{invite_id}",
        organization_id=organization_id,
        actor_user_id=actor_user_id,
        action=AuditAction.INVITE_REVOKED,
        resource_type="organization_invite",
        resource_id=invite_id,
        metadata={"email": revoked_invite.email, "role": revoked_invite.role.value},
    )
    with store.conn:
        store.revoke_organization_invite(invite_id)
        store.create_audit_event(audit_event)
    return revoked_invite
