from backend.access_control import AccessDenied
from backend.database import initialize_database
from backend.domain import InviteStatus, Organization, Role, User
from backend.invitations import (
    accept_organization_invite,
    create_organization_invite,
    revoke_organization_invite,
)
from backend.onboarding import create_owner_organization
from backend.store import SaaSStore


def test_create_organization_invite_requires_membership_management_permission():
    store = SaaSStore(initialize_database())
    create_owner_organization(
        store,
        User(id="owner_1", email="owner@example.com", display_name="Owner"),
        Organization(id="org_1", name="Example Energy", slug="example-energy"),
    )

    invite = create_organization_invite(
        store,
        actor_user_id="owner_1",
        organization_id="org_1",
        invite_id="invite_1",
        email="invitee@example.com",
        role=Role.MANAGER,
    )

    assert invite.status == InviteStatus.PENDING
    assert invite.role == Role.MANAGER

    try:
        create_organization_invite(
            store,
            actor_user_id="unknown",
            organization_id="org_1",
            invite_id="invite_2",
            email="blocked@example.com",
            role=Role.VIEWER,
        )
        assert False, "Expected unauthorized invite creation to fail"
    except AccessDenied:
        pass


def test_accept_organization_invite_creates_user_membership_and_audit_state():
    store = SaaSStore(initialize_database())
    create_owner_organization(
        store,
        User(id="owner_1", email="owner@example.com", display_name="Owner"),
        Organization(id="org_1", name="Example Energy", slug="example-energy"),
    )
    create_organization_invite(
        store,
        actor_user_id="owner_1",
        organization_id="org_1",
        invite_id="invite_1",
        email="invitee@example.com",
        role=Role.VIEWER,
    )

    result = accept_organization_invite(
        store,
        invite_id="invite_1",
        user=User(id="user_2", email="invitee@example.com", display_name="Invitee"),
    )

    memberships = store.list_memberships(user_id="user_2", organization_id="org_1")
    invite_row = store.get_organization_invite("invite_1")

    assert result.membership.role == Role.VIEWER
    assert [membership.role for membership in memberships] == [Role.VIEWER]
    assert invite_row["status"] == InviteStatus.ACCEPTED.value
    assert invite_row["accepted_by_user_id"] == "user_2"


def test_accept_organization_invite_requires_matching_email():
    store = SaaSStore(initialize_database())
    create_owner_organization(
        store,
        User(id="owner_1", email="owner@example.com", display_name="Owner"),
        Organization(id="org_1", name="Example Energy", slug="example-energy"),
    )
    create_organization_invite(
        store,
        actor_user_id="owner_1",
        organization_id="org_1",
        invite_id="invite_1",
        email="invitee@example.com",
        role=Role.VIEWER,
    )

    try:
        accept_organization_invite(
            store,
            invite_id="invite_1",
            user=User(id="user_2", email="mismatch@example.com", display_name="Invitee"),
        )
        assert False, "Expected invite acceptance to require matching email"
    except ValueError as exc:
        assert "email" in str(exc)


def test_revoke_organization_invite_marks_pending_invite_as_revoked():
    store = SaaSStore(initialize_database())
    create_owner_organization(
        store,
        User(id="owner_1", email="owner@example.com", display_name="Owner"),
        Organization(id="org_1", name="Example Energy", slug="example-energy"),
    )
    create_organization_invite(
        store,
        actor_user_id="owner_1",
        organization_id="org_1",
        invite_id="invite_1",
        email="invitee@example.com",
        role=Role.VIEWER,
    )

    invite = revoke_organization_invite(
        store,
        actor_user_id="owner_1",
        organization_id="org_1",
        invite_id="invite_1",
    )

    invite_row = store.get_organization_invite("invite_1")

    assert invite.status == InviteStatus.REVOKED
    assert invite_row["status"] == InviteStatus.REVOKED.value
