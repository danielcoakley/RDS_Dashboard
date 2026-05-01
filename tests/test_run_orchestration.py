import pandas as pd

from backend.access_control import AccessDenied
from backend.database import initialize_database
from backend.domain import Membership, Organization, Role, Site, User
from backend.run_orchestration import AnalysisRunRequest, execute_analysis_run
from backend.store import SaaSStore
from src.config_loader import load_client_config


def seed_tenant(store: SaaSStore, role: Role = Role.MANAGER) -> None:
    store.create_user(User(id="user_1", email="owner@example.com", display_name="Owner"))
    store.create_organization(Organization(id="org_1", name="Example Energy", slug="example-energy"))
    store.add_membership(Membership(user_id="user_1", organization_id="org_1", role=role))
    store.create_site(Site(id="site_1", organization_id="org_1", name="Main Site", timezone="Europe/London"))


def sample_request() -> AnalysisRunRequest:
    return AnalysisRunRequest(
        user_id="user_1",
        organization_id="org_1",
        site_id="site_1",
        upload_id="upload_1",
        run_id="run_1",
        report_id="report_1",
        filename="energy.csv",
        raw_meter_data=pd.DataFrame(
            {
                "Date": ["2025-01-01", "2025-02-01"],
                "Main Electricity": [100, 120],
                "Main Gas": [80, 70],
            }
        ),
        client_config=load_client_config("config/clients/example_client.yaml"),
    )


def test_execute_analysis_run_creates_metadata_and_report():
    store = SaaSStore(initialize_database())
    seed_tenant(store)

    outcome = execute_analysis_run(store, sample_request())

    runs = store.list_runs("org_1")
    reports = store.list_reports("org_1", run_id="run_1")
    uploads = store.list_uploads("org_1")

    assert outcome.upload.storage_key == "tenants/org_1/sites/site_1/uploads/upload_1/energy.csv"
    assert outcome.report.storage_key == "tenants/org_1/sites/site_1/runs/run_1/reports/iso-summary.json"
    assert runs[0]["status"] == "succeeded"
    assert reports[0]["id"] == "report_1"
    assert uploads[0]["checksum"]
    assert outcome.analytics.iso_summary["total_records"] == 4
    assert [event["action"] for event in store.list_audit_events("org_1")] == [
        "upload_stored",
        "run_started",
        "run_succeeded",
        "report_created",
    ]


def test_execute_analysis_run_requires_run_permission():
    store = SaaSStore(initialize_database())
    seed_tenant(store, role=Role.VIEWER)

    try:
        execute_analysis_run(store, sample_request())
        assert False, "Expected viewer run execution to be denied"
    except AccessDenied:
        pass


def test_execute_analysis_run_marks_failed_when_analytics_fails():
    store = SaaSStore(initialize_database())
    seed_tenant(store)
    request = sample_request()
    bad_request = AnalysisRunRequest(
        user_id=request.user_id,
        organization_id=request.organization_id,
        site_id=request.site_id,
        upload_id=request.upload_id,
        run_id=request.run_id,
        report_id=request.report_id,
        filename=request.filename,
        raw_meter_data=pd.DataFrame({"Date": ["2025-01-01"], "Main Electricity": [100]}),
        client_config=request.client_config,
    )

    try:
        execute_analysis_run(store, bad_request)
        assert False, "Expected analytics failure"
    except ValueError:
        pass

    runs = store.list_runs("org_1")
    reports = store.list_reports("org_1")
    assert runs[0]["status"] == "failed"
    assert "missing" in runs[0]["error_message"].lower()
    assert reports == []
    assert [event["action"] for event in store.list_audit_events("org_1")] == [
        "upload_stored",
        "run_started",
        "run_failed",
    ]
