import sqlite3

from backend.database import initialize_database
from backend.domain import (
    Membership,
    Organization,
    Report,
    Role,
    Run,
    RunStatus,
    Site,
    Upload,
    UploadStatus,
    User,
)
from backend.store import SaaSStore


def seed_tenant(store: SaaSStore) -> None:
    store.create_user(User(id="user_1", email="owner@example.com", display_name="Owner"))
    store.create_organization(Organization(id="org_1", name="Example Energy", slug="example-energy"))
    store.add_membership(Membership(user_id="user_1", organization_id="org_1", role=Role.OWNER))
    store.create_site(Site(id="site_1", organization_id="org_1", name="Main Site", timezone="Europe/London"))


def test_upload_run_and_report_metadata_are_tenant_scoped():
    store = SaaSStore(initialize_database())
    seed_tenant(store)
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
        organization_id="org_1",
        site_id="site_1",
        requested_by_user_id="user_1",
        upload_ids=["upload_1"],
        status=RunStatus.QUEUED,
    )
    report = Report(
        id="report_1",
        organization_id="org_1",
        run_id="run_1",
        report_type="iso_summary",
        storage_key="org_1/site_1/run_1/report.json",
        is_published=True,
    )

    store.create_upload(upload)
    store.create_run(run)
    store.update_run_status("run_1", RunStatus.SUCCEEDED, completed_at="2026-05-01T22:00:00Z")
    store.create_report(report)

    uploads = store.list_uploads("org_1", site_id="site_1")
    runs = store.list_runs("org_1", site_id="site_1")
    reports = store.list_reports("org_1", run_id="run_1")

    assert [row["id"] for row in uploads] == ["upload_1"]
    assert [(row["id"], row["status"]) for row in runs] == [("run_1", "succeeded")]
    assert [(row["id"], row["is_published"]) for row in reports] == [("report_1", 1)]


def test_run_cannot_use_upload_from_another_tenant():
    store = SaaSStore(initialize_database())
    seed_tenant(store)
    store.create_organization(Organization(id="org_2", name="Other Energy", slug="other-energy"))
    store.create_site(Site(id="site_2", organization_id="org_2", name="Other Site", timezone="Europe/London"))
    store.create_user(User(id="user_2", email="other@example.com", display_name="Other"))
    store.create_upload(
        Upload(
            id="upload_2",
            organization_id="org_2",
            site_id="site_2",
            uploaded_by_user_id="user_2",
            category="energy",
            storage_key="org_2/site_2/upload_2/energy.csv",
            checksum="def456",
            status=UploadStatus.STORED,
        )
    )

    try:
        store.create_run(
            Run(
                id="run_1",
                organization_id="org_1",
                site_id="site_1",
                requested_by_user_id="user_1",
                upload_ids=["upload_2"],
                status=RunStatus.QUEUED,
            )
        )
        assert False, "Expected cross-tenant run upload reference to fail"
    except sqlite3.IntegrityError:
        pass
