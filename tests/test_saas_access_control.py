from backend.access_control import AccessDenied, require_permission, require_resource_scope
from backend.domain import Membership, Role
from backend.rbac import Action


def test_require_permission_returns_tenant_context_for_allowed_action():
    memberships = [Membership(user_id="user_1", organization_id="org_1", role=Role.MANAGER)]

    context = require_permission(
        memberships,
        user_id="user_1",
        organization_id="org_1",
        action=Action.MANAGE_RUNS,
    )

    assert context.user_id == "user_1"
    assert context.organization_id == "org_1"
    assert context.role == Role.MANAGER


def test_audit_permission_is_limited_to_owner_and_manager():
    assert require_permission(
        [Membership(user_id="owner", organization_id="org_1", role=Role.OWNER)],
        user_id="owner",
        organization_id="org_1",
        action=Action.VIEW_AUDIT,
    )
    assert require_permission(
        [Membership(user_id="manager", organization_id="org_1", role=Role.MANAGER)],
        user_id="manager",
        organization_id="org_1",
        action=Action.VIEW_AUDIT,
    )

    try:
        require_permission(
            [Membership(user_id="viewer", organization_id="org_1", role=Role.VIEWER)],
            user_id="viewer",
            organization_id="org_1",
            action=Action.VIEW_AUDIT,
        )
        assert False, "Expected viewer audit access to be denied"
    except AccessDenied:
        pass


def test_require_permission_rejects_role_without_action():
    memberships = [Membership(user_id="user_1", organization_id="org_1", role=Role.VIEWER)]

    try:
        require_permission(
            memberships,
            user_id="user_1",
            organization_id="org_1",
            action=Action.MANAGE_UPLOADS,
        )
        assert False, "Expected viewer upload management to be denied"
    except AccessDenied as exc:
        assert "manage_uploads" in str(exc)


def test_require_permission_rejects_missing_tenant_membership():
    memberships = [Membership(user_id="user_1", organization_id="org_1", role=Role.OWNER)]

    try:
        require_permission(
            memberships,
            user_id="user_1",
            organization_id="org_2",
            action=Action.READ,
        )
        assert False, "Expected cross-tenant access to be denied"
    except AccessDenied as exc:
        assert "org_2" in str(exc)


def test_require_resource_scope_allows_matching_tenant():
    context = require_permission(
        [Membership(user_id="user_1", organization_id="org_1", role=Role.OWNER)],
        user_id="user_1",
        organization_id="org_1",
        action=Action.READ,
    )

    require_resource_scope(context, "org_1", "site")


def test_require_resource_scope_rejects_cross_tenant_resource():
    context = require_permission(
        [Membership(user_id="user_1", organization_id="org_1", role=Role.OWNER)],
        user_id="user_1",
        organization_id="org_1",
        action=Action.READ,
    )

    try:
        require_resource_scope(context, "org_2", "site")
        assert False, "Expected cross-tenant resource to be denied"
    except AccessDenied as exc:
        assert "org_2" in str(exc)
