from backend.domain import Membership, Organization, Role, Run, RunStatus, Upload, UploadStatus, User
from backend.rbac import Action, can_access_organization, can_perform


def test_membership_enforces_tenant_access():
    user = User(id="user_1", email="owner@example.com", display_name="Owner")
    org = Organization(id="org_1", name="Example Energy", slug="example-energy")
    other_org = Organization(id="org_2", name="Other Energy", slug="other-energy")
    memberships = [Membership(user_id=user.id, organization_id=org.id, role=Role.OWNER)]

    assert can_access_organization(memberships, user.id, org.id)
    assert not can_access_organization(memberships, user.id, other_org.id)


def test_rbac_owner_manager_viewer_permissions():
    memberships = [
        Membership(user_id="owner", organization_id="org_1", role=Role.OWNER),
        Membership(user_id="manager", organization_id="org_1", role=Role.MANAGER),
        Membership(user_id="viewer", organization_id="org_1", role=Role.VIEWER),
    ]

    assert can_perform(memberships, "owner", "org_1", Action.MANAGE_ORGANIZATION)
    assert can_perform(memberships, "manager", "org_1", Action.MANAGE_UPLOADS)
    assert not can_perform(memberships, "manager", "org_1", Action.MANAGE_MEMBERS)
    assert can_perform(memberships, "viewer", "org_1", Action.READ)
    assert not can_perform(memberships, "viewer", "org_1", Action.MANAGE_RUNS)


def test_upload_and_run_are_organization_scoped():
    upload = Upload(
        id="upload_1",
        organization_id="org_1",
        site_id="site_1",
        uploaded_by_user_id="user_1",
        category="energy",
        storage_key="org_1/site_1/upload_1/energy.csv",
        checksum="abc123",
        status=UploadStatus.STORED,
    )
    run = Run(
        id="run_1",
        organization_id=upload.organization_id,
        site_id=upload.site_id,
        requested_by_user_id="user_1",
        upload_ids=[upload.id],
        status=RunStatus.QUEUED,
    )

    assert upload.organization_id == "org_1"
    assert run.organization_id == upload.organization_id
    assert run.upload_ids == [upload.id]


def test_run_requires_upload_ids():
    try:
        Run(
            id="run_1",
            organization_id="org_1",
            site_id="site_1",
            requested_by_user_id="user_1",
            upload_ids=[],
        )
        assert False, "Expected ValueError for missing upload ids"
    except ValueError as exc:
        assert "upload_ids" in str(exc)
