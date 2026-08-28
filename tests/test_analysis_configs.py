import base64
import json

from backend.access_control import AccessDenied
from backend.analysis_configs import (
    build_analysis_config_from_tenant_assets,
    get_tenant_analysis_config,
    normalize_analysis_config,
    update_tenant_analysis_config,
    weather_model_settings_from_config,
)
from backend.api import (
    OwnerOrganizationCreate,
    StoredUploadAnalysisRunCreate,
    UploadContentCreate,
    create_upload_content_summary,
    execute_stored_upload_analysis_run,
    onboard_owner_organization,
    queue_stored_upload_analysis_run,
)
from backend.artifacts import InMemoryTenantArtifactStore
from backend.database import initialize_database
from backend.domain import Membership, Meter, Role, Site, User
from backend.store import SaaSStore
from analytics.config_loader import load_client_config


def seed_config_tenant(store: SaaSStore, source_column: str = "Tenant Electricity") -> None:
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_2",
            email="other@example.com",
            display_name="Other Owner",
            organization_id="org_2",
            organization_name="Other Energy",
            organization_slug="other-energy",
        ),
        store,
    )
    store.create_user(User(id="manager_1", email="manager@example.com", display_name="Manager"))
    store.create_user(User(id="viewer_1", email="viewer@example.com", display_name="Viewer"))
    store.add_membership(
        Membership(
            user_id="manager_1",
            organization_id="org_1",
            role=Role.MANAGER,
            invited_by_user_id="owner_1",
        )
    )
    store.add_membership(
        Membership(
            user_id="viewer_1",
            organization_id="org_1",
            role=Role.VIEWER,
            invited_by_user_id="owner_1",
        )
    )
    store.create_site(
        Site(id="site_1", organization_id="org_1", name="Main Site", timezone="Europe/London")
    )
    store.create_meter(
        Meter(
            id="meter_1",
            organization_id="org_1",
            site_id="site_1",
            display_name="Tenant Electricity",
            commodity="electricity",
            unit="kWh",
            source_column=source_column,
            is_seu=True,
        )
    )


def test_tenant_analysis_config_defaults_to_tenant_assets():
    store = SaaSStore(initialize_database())
    seed_config_tenant(store)

    config = get_tenant_analysis_config(store, "owner_1", "org_1", site_id="site_1")

    assert config.source == "tenant_assets"
    assert config.updated_by_user_id is None
    assert config.config["client"]["id"] == "org_1"
    assert config.config["client"]["name"] == "Example Energy"
    assert config.config["site"]["name"] == "Main Site"
    assert config.config["meters"][0]["source_column"] == "Tenant Electricity"
    assert config.config["analysis"]["weather_normalisation"] is True
    assert config.config["analysis"]["weather_model_min_baseline_records"] == 10
    assert config.config["analysis"]["weather_model_min_comparison_records"] == 1

    try:
        get_tenant_analysis_config(store, "owner_2", "org_1", site_id="site_1")
        assert False, "Expected cross-tenant analysis config read to be denied"
    except AccessDenied:
        pass


def test_update_tenant_analysis_config_persists_validated_config_and_audits():
    store = SaaSStore(initialize_database())
    seed_config_tenant(store)
    config = build_analysis_config_from_tenant_assets(store, "org_1", site_id="site_1")
    config["analysis"]["hdd_base_temperature"] = 16.0
    config["analysis"]["weather_normalisation"] = False
    config["analysis"]["weather_model_min_baseline_records"] = 24
    config["analysis"]["weather_model_min_comparison_records"] = 6

    saved = update_tenant_analysis_config(store, "manager_1", "org_1", config)
    loaded = get_tenant_analysis_config(store, "owner_1", "org_1")

    assert saved.source == "saved"
    assert saved.updated_by_user_id == "manager_1"
    assert loaded.config["analysis"]["hdd_base_temperature"] == 16.0
    assert loaded.config["analysis"]["weather_normalisation"] is False
    assert loaded.config["analysis"]["weather_model_min_baseline_records"] == 24
    assert loaded.config["analysis"]["weather_model_min_comparison_records"] == 6
    assert weather_model_settings_from_config(loaded.config).enabled is False
    assert json.loads(store.get_analysis_config("org_1")["config_json"])["client"]["id"] == "org_1"
    audit_events = store.list_audit_events("org_1")
    assert [event["action"] for event in audit_events] == ["analysis_config_updated"]
    assert json.loads(audit_events[0]["metadata_json"]) == {
        "client_id": "org_1",
        "meter_count": "1",
        "weather_model_min_baseline_records": "24",
        "weather_model_min_comparison_records": "6",
        "weather_normalisation_enabled": False,
    }

    invalid = load_client_config("config/clients/example_client.yaml")
    invalid.pop("meters")
    try:
        update_tenant_analysis_config(store, "manager_1", "org_1", invalid)
        assert False, "Expected invalid analysis config to be rejected"
    except ValueError as exc:
        assert "Missing top-level config sections" in str(exc)

    invalid_threshold = build_analysis_config_from_tenant_assets(store, "org_1", site_id="site_1")
    invalid_threshold["analysis"]["weather_model_min_baseline_records"] = 1
    try:
        normalize_analysis_config(invalid_threshold)
        assert False, "Expected invalid weather model threshold to be rejected"
    except ValueError as exc:
        assert "analysis.weather_model_min_baseline_records" in str(exc)

    try:
        update_tenant_analysis_config(store, "viewer_1", "org_1", config)
        assert False, "Expected viewer analysis config update to be denied"
    except AccessDenied:
        pass


def test_queued_run_snapshots_saved_tenant_analysis_config_when_payload_omits_config():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    seed_config_tenant(store, source_column="Tenant Electricity")
    config = build_analysis_config_from_tenant_assets(store, "org_1", site_id="site_1")
    config["meters"][0]["display_name"] = "Saved Electricity"
    update_tenant_analysis_config(store, "owner_1", "org_1", config)
    upload = create_upload_content_summary(
        "owner_1",
        "org_1",
        UploadContentCreate(
            upload_id="upload_1",
            site_id="site_1",
            category="energy",
            filename="tenant-energy.csv",
            content_base64=base64.b64encode(
                b"Date,Tenant Electricity\n2025-01-01,100\n2025-02-01,120\n"
            ).decode("ascii"),
        ),
        store,
        artifact_store,
    )

    response = queue_stored_upload_analysis_run(
        "owner_1",
        "org_1",
        StoredUploadAnalysisRunCreate(
            upload_id=upload.id,
            run_id="run_1",
            report_id="report_1",
        ),
        store,
    )

    job = store.get_analysis_job(response.job_id)
    job_config = json.loads(job["client_config_json"])
    assert job_config["meters"][0]["display_name"] == "Saved Electricity"
    assert job_config["meters"][0]["source_column"] == "Tenant Electricity"


def test_stored_upload_run_can_use_generated_tenant_asset_config():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    seed_config_tenant(store, source_column="Tenant Electricity")
    upload = create_upload_content_summary(
        "owner_1",
        "org_1",
        UploadContentCreate(
            upload_id="upload_1",
            site_id="site_1",
            category="energy",
            filename="tenant-energy.csv",
            content_base64=base64.b64encode(
                b"Date,Tenant Electricity\n2025-01-01,100\n2025-02-01,120\n"
            ).decode("ascii"),
        ),
        store,
        artifact_store,
    )

    response = execute_stored_upload_analysis_run(
        "owner_1",
        "org_1",
        StoredUploadAnalysisRunCreate(
            upload_id=upload.id,
            run_id="run_1",
            report_id="report_1",
        ),
        store,
        artifact_store,
    )

    assert response.run_status == "succeeded"
    assert response.iso_summary["total_records"] == 2
    report_row = store.get_report("report_1")
    assert "report_storage_key" not in response.model_dump()
    assert report_row is not None
    assert report_row["storage_key"] == "tenants/org_1/sites/site_1/runs/run_1/reports/iso-summary.json"
