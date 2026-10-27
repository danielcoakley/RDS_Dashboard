from backend.auth_context import AuthenticatedUser
from backend.clerk_sync import platform_role_from_clerk_role, sync_clerk_authenticated_user
from backend.database import initialize_database
from backend.domain import Role
from backend.store import SaaSStore


def test_platform_role_from_clerk_role_maps_known_roles_and_defaults_to_viewer():
    assert platform_role_from_clerk_role("admin") == Role.OWNER
    assert platform_role_from_clerk_role("org:admin") == Role.OWNER
    assert platform_role_from_clerk_role("manager") == Role.MANAGER
    assert platform_role_from_clerk_role("org:member") == Role.VIEWER
    assert platform_role_from_clerk_role("custom-analyst") == Role.VIEWER
    assert platform_role_from_clerk_role(None) == Role.VIEWER


def test_sync_clerk_authenticated_user_upserts_tenant_metadata():
    store = SaaSStore(initialize_database())

    result = sync_clerk_authenticated_user(
        store,
        AuthenticatedUser(
            id="user_123",
            organization_id="org_123",
            role="admin",
            auth_mode="clerk",
            email="owner@example.com",
            display_name="Owner User",
            organization_slug="example-energy",
            organization_name="Example Energy",
        ),
    )

    assert result.user.id == "user_123"
    assert result.organization.id == "org_123"
    assert result.membership.role == Role.OWNER

    user_row = store.get_user("user_123")
    organization_row = store.get_organization("org_123")
    memberships = store.list_memberships(user_id="user_123", organization_id="org_123")

    assert user_row is not None
    assert user_row["email"] == "owner@example.com"
    assert organization_row is not None
    assert organization_row["slug"] == "example-energy"
    assert memberships[0].role == Role.OWNER

    sync_clerk_authenticated_user(
        store,
        AuthenticatedUser(
            id="user_123",
            organization_id="org_123",
            role="member",
            auth_mode="clerk",
            email="updated@example.com",
            display_name="Updated Owner",
            organization_slug="example-energy",
            organization_name="Example Energy Updated",
        ),
    )

    user_row = store.get_user("user_123")
    organization_row = store.get_organization("org_123")
    memberships = store.list_memberships(user_id="user_123", organization_id="org_123")

    assert user_row is not None
    assert user_row["email"] == "updated@example.com"
    assert user_row["display_name"] == "Updated Owner"
    assert organization_row is not None
    assert organization_row["name"] == "Example Energy Updated"
    assert memberships[0].role == Role.VIEWER


def test_sync_clerk_authenticated_user_requires_clerk_auth_and_active_org():
    store = SaaSStore(initialize_database())

    try:
        sync_clerk_authenticated_user(
            store,
            AuthenticatedUser(id="user_123", organization_id="org_123", auth_mode="dev"),
        )
        assert False, "Expected non-Clerk sync to fail"
    except ValueError as exc:
        assert "verified Clerk" in str(exc)

    try:
        sync_clerk_authenticated_user(
            store,
            AuthenticatedUser(id="user_123", auth_mode="clerk"),
        )
        assert False, "Expected missing active organization to fail"
    except ValueError as exc:
        assert "active organization" in str(exc)
