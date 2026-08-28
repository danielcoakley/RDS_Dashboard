import json

import pandas as pd
import pytest

from backend.access_control import AccessDenied
from backend.artifacts import InMemoryTenantArtifactStore
from backend.database import initialize_database
from backend.domain import Membership, Organization, Role, Site, Upload, UploadStatus, User
from backend.run_orchestration import (
    AnalysisRunRequest,
    StoredUploadAnalysisRunRequest,
    _report_artifact_content,
    execute_analysis_run,
    execute_analysis_run_from_upload,
    process_analysis_job,
    queue_analysis_run_from_upload,
)
from backend.store import SaaSStore
from analytics.config_loader import load_client_config
from analytics.service import AnalyticsRunResult


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


def test_report_artifact_source_package_keeps_only_public_lineage_metadata():
    primary_upload = Upload(
        id="upload_1",
        organization_id="org_1",
        site_id="site_1",
        uploaded_by_user_id="user_1",
        category="energy",
        storage_key="tenants/org_1/sites/site_1/uploads/upload_1/energy.csv",
        checksum="abc123",
        status=UploadStatus.STORED,
    )
    analytics = AnalyticsRunResult(
        standardised_data=pd.DataFrame(),
        consumption_summary=pd.DataFrame(),
        enpi_summary=pd.DataFrame(),
        data_gaps=pd.DataFrame(),
        outliers=pd.DataFrame(),
        iso_summary={"total_records": 0},
        baseline_model={},
    )

    content = _report_artifact_content(
        analytics,
        primary_upload,
        [primary_upload],
        {
            "execution_input": "saas_meter_csv",
            "analysis_input_path": primary_upload.storage_key,
            "storage_key": primary_upload.storage_key,
            "resolved_meter_columns": ["Main Electricity"],
            "resolved_record_count": 2,
            "sources": [
                {
                    "upload_id": primary_upload.id,
                    "category": primary_upload.category,
                    "is_primary": True,
                    "parse_status": "parsed",
                    "shape": "saas_meter_csv",
                    "row_count": 2,
                    "meter_columns": ["Main Electricity"],
                    "storage_key": primary_upload.storage_key,
                    "analysis_input_path": primary_upload.storage_key,
                }
            ],
        },
    )

    source_package = content["source_package"]
    analysis_context = source_package["analysis_context"]
    source_summary = analysis_context["sources"][0]
    serialized_content = json.dumps(content)

    assert analysis_context["execution_input"] == "saas_meter_csv"
    assert analysis_context["resolved_record_count"] == 2
    assert source_summary["upload_id"] == "upload_1"
    assert "analysis_input_path" not in analysis_context
    assert "storage_key" not in analysis_context
    assert "analysis_input_path" not in source_summary
    assert "storage_key" not in source_summary
    assert primary_upload.storage_key not in serialized_content


def modeled_legacy_source_files() -> dict[str, tuple[str, str, str]]:
    baseline_dates = pd.date_range("2024-01-01", periods=12, freq="D")
    comparison_dates = pd.date_range("2025-01-01", periods=12, freq="D")
    dates = [*baseline_dates, *comparison_dates]
    date_headers = [date.strftime("%d/%m/%Y") for date in dates]
    hdd_values = [float(index + 1) for index, _date in enumerate(dates)]
    cdd_values = [float((index + 1) * 2) for index, _date in enumerate(dates)]
    gas_values = [100 + 2 * value for value in hdd_values[:12]]
    gas_values.extend([100 + 2 * value - 10 for value in hdd_values[12:]])
    electricity_values = [50 + 3 * value for value in cdd_values[:12]]
    electricity_values.extend([50 + 3 * value - 5 for value in cdd_values[12:]])

    def _csv_values(values: list[float]) -> str:
        return ",".join(str(round(value, 2)) for value in values)

    energy_csv = "\n".join(
        [
            "Metered Sector,Utility,Units,Period," + ",".join(date_headers),
            "Boiler gas,Gas,kWh,Total," + _csv_values(gas_values),
            ",,,Day," + _csv_values(gas_values),
            "Office power,Electrical,kWh,Total," + _csv_values(electricity_values),
            ",,,Day," + _csv_values(electricity_values),
            "",
        ]
    )
    hdd_csv = "\n".join(
        [
            "Description:,Celsius-based heating degree days",
            "Source:,degreedays.net",
            "",
            "Date,HDD 15.5",
            *[
                f"{date.strftime('%d/%m/%Y')},{value}"
                for date, value in zip(dates, hdd_values)
            ],
            "",
        ]
    )
    cdd_csv = "\n".join(
        [
            "Description:,Celsius-based cooling degree days",
            "Source:,degreedays.net",
            "",
            "Date,CDD 15.5",
            *[
                f"{date.strftime('%d/%m/%Y')},{value}"
                for date, value in zip(dates, cdd_values)
            ],
            "",
        ]
    )
    return {
        "upload_energy": (
            "energy",
            "tenants/org_1/sites/site_1/uploads/upload_energy/legacy-energy.csv",
            energy_csv,
        ),
        "upload_hdd": (
            "hdd",
            "tenants/org_1/sites/site_1/uploads/upload_hdd/hdd.csv",
            hdd_csv,
        ),
        "upload_cdd": (
            "cdd",
            "tenants/org_1/sites/site_1/uploads/upload_cdd/cdd.csv",
            cdd_csv,
        ),
        "upload_mapping": (
            "seu_mapping",
            "tenants/org_1/sites/site_1/uploads/upload_mapping/mapping.csv",
            "Meter,SEU_Category\nBoiler gas,Boiler\nOffice power,Lighting\n",
        ),
    }


def test_execute_analysis_run_creates_metadata_and_report():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    seed_tenant(store)

    outcome = execute_analysis_run(store, sample_request(), artifact_store=artifact_store)

    runs = store.list_runs("org_1")
    reports = store.list_reports("org_1", run_id="run_1")
    uploads = store.list_uploads("org_1")

    assert outcome.upload.storage_key == "tenants/org_1/sites/site_1/uploads/upload_1/energy.csv"
    assert outcome.report.storage_key == "tenants/org_1/sites/site_1/runs/run_1/reports/iso-summary.json"
    assert runs[0]["status"] == "succeeded"
    assert reports[0]["id"] == "report_1"
    assert uploads[0]["checksum"]
    assert outcome.analytics.iso_summary["total_records"] == 4
    assert artifact_store.get_text("org_1", outcome.upload.storage_key).startswith(
        "Date,Main Electricity,Main Gas"
    )
    assert "\"total_records\": 4" in artifact_store.get_text("org_1", outcome.report.storage_key)
    report_artifact = json.loads(artifact_store.get_text("org_1", outcome.report.storage_key))
    source_package = report_artifact["source_package"]
    assert source_package["primary_upload_id"] == "upload_1"
    assert source_package["site_id"] == "site_1"
    assert source_package["upload_count"] == 1
    assert source_package["categories"] == ["energy"]
    assert source_package["uploads"] == [
        {
            "id": "upload_1",
            "category": "energy",
            "checksum": outcome.upload.checksum,
            "status": "stored",
            "is_primary": True,
        }
    ]
    assert source_package["analysis_context"]["execution_input"] == "local_dataframe"
    assert source_package["analysis_context"]["resolved_record_count"] == 2
    assert [event["action"] for event in store.list_audit_events("org_1")] == [
        "upload_stored",
        "run_started",
        "run_succeeded",
        "report_created",
    ]


def test_execute_analysis_run_from_upload_uses_existing_tenant_artifact():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    seed_tenant(store)
    upload_key = "tenants/org_1/sites/site_1/uploads/upload_1/energy.csv"
    artifact_store.put_text(
        "org_1",
        upload_key,
        "Date,Main Electricity,Main Gas\n2025-01-01,100,80\n2025-02-01,120,70\n",
    )
    store.create_upload(
        Upload(
            id="upload_1",
            organization_id="org_1",
            site_id="site_1",
            uploaded_by_user_id="user_1",
            category="energy",
            storage_key=upload_key,
            checksum="abc123",
            status=UploadStatus.STORED,
        )
    )

    outcome = execute_analysis_run_from_upload(
        store,
        StoredUploadAnalysisRunRequest(
            user_id="user_1",
            organization_id="org_1",
            upload_id="upload_1",
            run_id="run_1",
            report_id="report_1",
            client_config=load_client_config("config/clients/example_client.yaml"),
        ),
        artifact_store,
    )

    assert outcome.upload.id == "upload_1"
    assert outcome.run.site_id == "site_1"
    assert outcome.analytics.iso_summary["total_records"] == 4
    assert store.list_runs("org_1")[0]["status"] == "succeeded"
    assert "\"total_records\": 4" in artifact_store.get_text("org_1", outcome.report.storage_key)
    report_artifact = json.loads(artifact_store.get_text("org_1", outcome.report.storage_key))
    assert report_artifact["source_package"]["analysis_context"]["execution_input"] == "saas_meter_csv"
    assert report_artifact["source_package"]["analysis_context"]["resolved_record_count"] == 2
    assert [event["action"] for event in store.list_audit_events("org_1")] == [
        "run_started",
        "run_succeeded",
        "report_created",
    ]


def test_execute_analysis_run_from_upload_converts_legacy_source_package():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    seed_tenant(store)
    config = load_client_config("config/clients/example_client.yaml")
    config["meters"][0]["meter_id"] = "boiler_gas"
    config["meters"][0]["display_name"] = "Boiler gas"
    config["meters"][0]["commodity"] = "gas"
    config["meters"][0]["source_column"] = "Boiler gas"
    config["meters"][0]["is_seu"] = True
    config["meters"][1]["meter_id"] = "office_power"
    config["meters"][1]["display_name"] = "Office power"
    config["meters"][1]["commodity"] = "electricity"
    config["meters"][1]["source_column"] = "Office power"
    config["meters"][1]["is_seu"] = True
    source_files = {
        "upload_energy": (
            "energy",
            "tenants/org_1/sites/site_1/uploads/upload_energy/legacy-energy.csv",
            (
                "Metered Sector,Utility,Units,Period,01/01/2025,02/01/2025\n"
                "Boiler gas,Gas,kWh,Total,21,25\n"
                ",,,Day,10,12\n"
                "Office power,Electrical,kWh,Total,41,45\n"
                ",,,Day,20,22\n"
            ),
        ),
        "upload_hdd": (
            "hdd",
            "tenants/org_1/sites/site_1/uploads/upload_hdd/hdd.csv",
            (
                "Description:,Celsius-based heating degree days\n"
                "Source:,degreedays.net\n"
                "\n"
                "Date,HDD 15.5\n01/01/2025,6\n02/01/2025,8\n"
            ),
        ),
        "upload_cdd": (
            "cdd",
            "tenants/org_1/sites/site_1/uploads/upload_cdd/cdd.csv",
            (
                "Description:,Celsius-based cooling degree days\n"
                "Source:,degreedays.net\n"
                "\n"
                "Date,CDD 15.5\n01/01/2025,0\n02/01/2025,1\n"
            ),
        ),
        "upload_mapping": (
            "seu_mapping",
            "tenants/org_1/sites/site_1/uploads/upload_mapping/mapping.csv",
            "Meter,SEU_Category\nBoiler gas,Boiler\nOffice power,Lighting\n",
        ),
    }
    for upload_id, (category, storage_key, content) in source_files.items():
        artifact_store.put_text("org_1", storage_key, content)
        store.create_upload(
            Upload(
                id=upload_id,
                organization_id="org_1",
                site_id="site_1",
                uploaded_by_user_id="user_1",
                category=category,
                storage_key=storage_key,
                checksum=f"{upload_id}_checksum",
                status=UploadStatus.STORED,
            )
        )

    outcome = execute_analysis_run_from_upload(
        store,
        StoredUploadAnalysisRunRequest(
            user_id="user_1",
            organization_id="org_1",
            upload_id="upload_energy",
            run_id="run_1",
            report_id="report_1",
            source_upload_ids=("upload_hdd", "upload_cdd", "upload_mapping"),
            client_config=config,
        ),
        artifact_store,
    )

    report_artifact = json.loads(artifact_store.get_text("org_1", outcome.report.storage_key))

    assert outcome.analytics.iso_summary["total_records"] == 4
    assert report_artifact["source_package"]["analysis_context"]["execution_input"] == "legacy_energy_export"
    assert report_artifact["source_package"]["analysis_context"]["resolved_meter_columns"] == [
        "Boiler gas",
        "Office power",
    ]
    assert report_artifact["source_package"]["analysis_context"]["resolved_record_count"] == 2
    assert report_artifact["source_package"]["categories"] == [
        "energy",
        "hdd",
        "cdd",
        "seu_mapping",
    ]
    assert [source["category"] for source in report_artifact["source_package"]["analysis_context"]["sources"]] == [
        "energy",
        "hdd",
        "cdd",
        "seu_mapping",
    ]
    legacy_baseline = report_artifact["legacy_baseline"]
    assert legacy_baseline["source"] == "legacy_source_package"
    assert legacy_baseline["energy_record_count"] == 4
    assert legacy_baseline["active_energy_record_count"] == 4
    assert legacy_baseline["meter_count"] == 2
    assert legacy_baseline["active_meter_count"] == 2
    assert legacy_baseline["gas_record_count"] == 2
    assert legacy_baseline["electricity_record_count"] == 2
    assert legacy_baseline["mapped_meter_count"] == 2
    assert legacy_baseline["ignored_meter_count"] == 0
    assert legacy_baseline["seu_categories"] == ["Boiler", "Lighting"]
    assert legacy_baseline["date_range"] == {
        "start_date": "2025-01-01",
        "end_date": "2025-01-02",
    }
    assert legacy_baseline["weather_coverage"]["hdd"]["record_count"] == 2
    assert legacy_baseline["weather_coverage"]["hdd"]["matched_energy_record_count"] == 2
    assert legacy_baseline["weather_coverage"]["cdd"]["record_count"] == 2
    assert legacy_baseline["weather_coverage"]["cdd"]["matched_energy_record_count"] == 2
    assert legacy_baseline["total_consumption"] == 64.0
    assert {
        item["utility"]: item["consumption"]
        for item in legacy_baseline["total_consumption_by_utility"]
    } == {"electricity": 42.0, "gas": 22.0}
    assert {
        item["seu_category"]: item["consumption"]
        for item in legacy_baseline["total_consumption_by_seu_category"]
    } == {"Boiler": 22.0, "Lighting": 42.0}


def test_execute_analysis_run_from_upload_includes_legacy_weather_normalization():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    seed_tenant(store)
    config = load_client_config("config/clients/example_client.yaml")
    config["meters"][0]["meter_id"] = "boiler_gas"
    config["meters"][0]["display_name"] = "Boiler gas"
    config["meters"][0]["commodity"] = "gas"
    config["meters"][0]["source_column"] = "Boiler gas"
    config["meters"][0]["is_seu"] = True
    config["meters"][1]["meter_id"] = "office_power"
    config["meters"][1]["display_name"] = "Office power"
    config["meters"][1]["commodity"] = "electricity"
    config["meters"][1]["source_column"] = "Office power"
    config["meters"][1]["is_seu"] = True

    for upload_id, (category, storage_key, content) in modeled_legacy_source_files().items():
        artifact_store.put_text("org_1", storage_key, content)
        store.create_upload(
            Upload(
                id=upload_id,
                organization_id="org_1",
                site_id="site_1",
                uploaded_by_user_id="user_1",
                category=category,
                storage_key=storage_key,
                checksum=f"{upload_id}_checksum",
                status=UploadStatus.STORED,
            )
        )

    outcome = execute_analysis_run_from_upload(
        store,
        StoredUploadAnalysisRunRequest(
            user_id="user_1",
            organization_id="org_1",
            upload_id="upload_energy",
            run_id="run_1",
            report_id="report_1",
            source_upload_ids=("upload_hdd", "upload_cdd", "upload_mapping"),
            client_config=config,
        ),
        artifact_store,
    )

    report_artifact = json.loads(artifact_store.get_text("org_1", outcome.report.storage_key))
    weather_normalization = report_artifact["legacy_baseline"]["weather_normalization"]

    assert outcome.analytics.iso_summary["total_records"] == 48
    assert weather_normalization["status"] == "modeled"
    assert weather_normalization["baseline_year"] == 2024
    assert weather_normalization["comparison_year"] == 2025
    assert weather_normalization["model_count"] == 2
    assert weather_normalization["skipped_meter_count"] == 0
    assert weather_normalization["model_settings"] == {
        "enabled": True,
        "min_baseline_records": 10,
        "min_comparison_records": 1,
    }
    assert weather_normalization["total_actual_consumption"] == pytest.approx(3396.0)
    assert weather_normalization["total_predicted_consumption"] == pytest.approx(3576.0)
    assert weather_normalization["estimated_savings"] == pytest.approx(180.0)
    models_by_meter = {model["meter"]: model for model in weather_normalization["models"]}
    assert models_by_meter["Boiler gas"]["normalization"] == "hdd"
    assert models_by_meter["Boiler gas"]["estimated_savings"] == pytest.approx(120.0)
    assert models_by_meter["Office power"]["normalization"] == "cdd"
    assert models_by_meter["Office power"]["estimated_savings"] == pytest.approx(60.0)


def test_stored_upload_weather_model_thresholds_are_tenant_configurable():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    seed_tenant(store)
    config = load_client_config("config/clients/example_client.yaml")
    config["analysis"]["weather_model_min_baseline_records"] = 20
    config["analysis"]["weather_model_min_comparison_records"] = 1
    config["meters"][0]["meter_id"] = "boiler_gas"
    config["meters"][0]["display_name"] = "Boiler gas"
    config["meters"][0]["commodity"] = "gas"
    config["meters"][0]["source_column"] = "Boiler gas"
    config["meters"][0]["is_seu"] = True
    config["meters"][1]["meter_id"] = "office_power"
    config["meters"][1]["display_name"] = "Office power"
    config["meters"][1]["commodity"] = "electricity"
    config["meters"][1]["source_column"] = "Office power"
    config["meters"][1]["is_seu"] = True

    for upload_id, (category, storage_key, content) in modeled_legacy_source_files().items():
        artifact_store.put_text("org_1", storage_key, content)
        store.create_upload(
            Upload(
                id=upload_id,
                organization_id="org_1",
                site_id="site_1",
                uploaded_by_user_id="user_1",
                category=category,
                storage_key=storage_key,
                checksum=f"{upload_id}_checksum",
                status=UploadStatus.STORED,
            )
        )

    outcome = execute_analysis_run_from_upload(
        store,
        StoredUploadAnalysisRunRequest(
            user_id="user_1",
            organization_id="org_1",
            upload_id="upload_energy",
            run_id="run_1",
            report_id="report_1",
            source_upload_ids=("upload_hdd", "upload_cdd", "upload_mapping"),
            client_config=config,
        ),
        artifact_store,
    )

    report_artifact = json.loads(artifact_store.get_text("org_1", outcome.report.storage_key))
    weather_normalization = report_artifact["legacy_baseline"]["weather_normalization"]

    assert weather_normalization["status"] == "not_modeled"
    assert weather_normalization["model_count"] == 0
    assert weather_normalization["skipped_meter_count"] == 2
    assert weather_normalization["model_settings"] == {
        "enabled": True,
        "min_baseline_records": 20,
        "min_comparison_records": 1,
    }
    assert {
        skipped["reason"]
        for skipped in weather_normalization["skipped_meters"]
    } == {"insufficient_baseline_records"}


def test_queue_then_process_analysis_job_from_upload():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    seed_tenant(store)
    upload_key = "tenants/org_1/sites/site_1/uploads/upload_1/energy.csv"
    artifact_store.put_text(
        "org_1",
        upload_key,
        "Date,Main Electricity,Main Gas\n2025-01-01,100,80\n2025-02-01,120,70\n",
    )
    store.create_upload(
        Upload(
            id="upload_1",
            organization_id="org_1",
            site_id="site_1",
            uploaded_by_user_id="user_1",
            category="energy",
            storage_key=upload_key,
            checksum="abc123",
            status=UploadStatus.STORED,
        )
    )
    supporting_key = "tenants/org_1/sites/site_1/uploads/upload_hdd/hdd.csv"
    artifact_store.put_text(
        "org_1",
        supporting_key,
        "Date,HDD\n2025-01-01,12\n2025-02-01,18\n",
    )
    store.create_upload(
        Upload(
            id="upload_hdd",
            organization_id="org_1",
            site_id="site_1",
            uploaded_by_user_id="user_1",
            category="hdd",
            storage_key=supporting_key,
            checksum="def456",
            status=UploadStatus.STORED,
        )
    )
    request = StoredUploadAnalysisRunRequest(
        user_id="user_1",
        organization_id="org_1",
        upload_id="upload_1",
        run_id="run_1",
        report_id="report_1",
        source_upload_ids=("upload_hdd",),
        client_config=load_client_config("config/clients/example_client.yaml"),
    )

    queued = queue_analysis_run_from_upload(store, request)

    assert queued.job.id == "job_run_1"
    assert queued.job.upload_id == "upload_1"
    assert list(queued.run.upload_ids) == ["upload_1", "upload_hdd"]
    assert [upload["id"] for upload in store.list_run_uploads("org_1", "run_1")] == [
        "upload_1",
        "upload_hdd",
    ]
    assert store.list_runs("org_1")[0]["status"] == "queued"
    assert store.list_analysis_jobs("org_1")[0]["status"] == "queued"
    assert store.list_reports("org_1") == []

    outcome = process_analysis_job(store, queued.job.id, artifact_store)

    assert outcome.run.id == "run_1"
    assert list(outcome.run.upload_ids) == ["upload_1", "upload_hdd"]
    assert store.list_runs("org_1")[0]["status"] == "succeeded"
    assert store.list_analysis_jobs("org_1")[0]["status"] == "succeeded"
    assert [report["id"] for report in store.list_reports("org_1")] == ["report_1"]
    report_artifact = json.loads(artifact_store.get_text("org_1", outcome.report.storage_key))
    assert report_artifact["source_package"]["primary_upload_id"] == "upload_1"
    assert report_artifact["source_package"]["upload_count"] == 2
    assert report_artifact["source_package"]["categories"] == ["energy", "hdd"]
    assert [upload["id"] for upload in report_artifact["source_package"]["uploads"]] == [
        "upload_1",
        "upload_hdd",
    ]
    assert report_artifact["source_package"]["uploads"][0]["is_primary"] is True
    assert report_artifact["source_package"]["uploads"][1]["is_primary"] is False
    assert report_artifact["source_package"]["analysis_context"]["execution_input"] == "saas_meter_csv"
    assert [source["category"] for source in report_artifact["source_package"]["analysis_context"]["sources"]] == [
        "energy",
        "hdd",
    ]
    notifications = store.list_notifications("org_1", recipient_user_id="user_1")
    assert [notification["event_type"] for notification in notifications] == ["run_succeeded"]
    assert notifications[0]["channel"] == "email"
    assert notifications[0]["status"] == "pending"
    assert notifications[0]["resource_id"] == "run_1"


def test_queue_analysis_run_from_upload_requires_energy_primary_upload():
    store = SaaSStore(initialize_database())
    seed_tenant(store)
    store.create_upload(
        Upload(
            id="upload_hdd",
            organization_id="org_1",
            site_id="site_1",
            uploaded_by_user_id="user_1",
            category="hdd",
            storage_key="tenants/org_1/sites/site_1/uploads/upload_hdd/hdd.csv",
            checksum="abc123",
            status=UploadStatus.STORED,
        )
    )

    try:
        queue_analysis_run_from_upload(
            store,
            StoredUploadAnalysisRunRequest(
                user_id="user_1",
                organization_id="org_1",
                upload_id="upload_hdd",
                run_id="run_1",
                report_id="report_1",
                client_config=load_client_config("config/clients/example_client.yaml"),
            ),
        )
        assert False, "Expected non-energy primary upload to be rejected"
    except ValueError as exc:
        assert "energy upload" in str(exc)

    assert store.list_runs("org_1") == []


def test_queue_analysis_run_from_upload_rejects_cross_tenant_supporting_upload():
    store = SaaSStore(initialize_database())
    seed_tenant(store)
    store.create_upload(
        Upload(
            id="upload_1",
            organization_id="org_1",
            site_id="site_1",
            uploaded_by_user_id="user_1",
            category="energy",
            storage_key="tenants/org_1/sites/site_1/uploads/upload_1/energy.csv",
            checksum="abc123",
            status=UploadStatus.STORED,
        )
    )
    store.create_user(User(id="owner_2", email="other@example.com", display_name="Other Owner"))
    store.create_organization(Organization(id="org_2", name="Other Energy", slug="other-energy"))
    store.add_membership(Membership(user_id="owner_2", organization_id="org_2", role=Role.OWNER))
    store.create_site(Site(id="site_2", organization_id="org_2", name="Other Site", timezone="Europe/London"))
    store.create_upload(
        Upload(
            id="upload_2",
            organization_id="org_2",
            site_id="site_2",
            uploaded_by_user_id="owner_2",
            category="hdd",
            storage_key="tenants/org_2/sites/site_2/uploads/upload_2/hdd.csv",
            checksum="def456",
            status=UploadStatus.STORED,
        )
    )

    try:
        queue_analysis_run_from_upload(
            store,
            StoredUploadAnalysisRunRequest(
                user_id="user_1",
                organization_id="org_1",
                upload_id="upload_1",
                run_id="run_1",
                report_id="report_1",
                source_upload_ids=("upload_2",),
                client_config=load_client_config("config/clients/example_client.yaml"),
            ),
        )
        assert False, "Expected cross-tenant supporting upload to be denied"
    except AccessDenied:
        pass

    assert store.list_runs("org_1") == []


def test_execute_analysis_run_from_upload_rejects_cross_tenant_upload():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    seed_tenant(store)
    store.create_user(User(id="owner_2", email="other@example.com", display_name="Other Owner"))
    store.create_organization(Organization(id="org_2", name="Other Energy", slug="other-energy"))
    store.add_membership(Membership(user_id="owner_2", organization_id="org_2", role=Role.OWNER))
    store.create_site(Site(id="site_2", organization_id="org_2", name="Other Site", timezone="Europe/London"))
    upload_key = "tenants/org_2/sites/site_2/uploads/upload_2/energy.csv"
    artifact_store.put_text(
        "org_2",
        upload_key,
        "Date,Main Electricity,Main Gas\n2025-01-01,100,80\n",
    )
    store.create_upload(
        Upload(
            id="upload_2",
            organization_id="org_2",
            site_id="site_2",
            uploaded_by_user_id="owner_2",
            category="energy",
            storage_key=upload_key,
            checksum="abc123",
            status=UploadStatus.STORED,
        )
    )

    try:
        execute_analysis_run_from_upload(
            store,
            StoredUploadAnalysisRunRequest(
                user_id="user_1",
                organization_id="org_1",
                upload_id="upload_2",
                run_id="run_1",
                report_id="report_1",
                client_config=load_client_config("config/clients/example_client.yaml"),
            ),
            artifact_store,
        )
        assert False, "Expected cross-tenant upload execution to be denied"
    except AccessDenied:
        pass

    assert store.list_runs("org_1") == []
    assert store.list_audit_events("org_1") == []


def test_execute_analysis_run_requires_run_permission():
    store = SaaSStore(initialize_database())
    seed_tenant(store, role=Role.VIEWER)

    try:
        execute_analysis_run(store, sample_request())
        assert False, "Expected viewer run execution to be denied"
    except AccessDenied:
        pass


def test_execute_analysis_run_rejects_cross_tenant_site_id_before_persisting():
    store = SaaSStore(initialize_database())
    seed_tenant(store)
    store.create_user(User(id="owner_2", email="other@example.com", display_name="Other Owner"))
    store.create_organization(Organization(id="org_2", name="Other Energy", slug="other-energy"))
    store.add_membership(Membership(user_id="owner_2", organization_id="org_2", role=Role.OWNER))
    store.create_site(Site(id="site_2", organization_id="org_2", name="Other Site", timezone="Europe/London"))

    request = sample_request()
    cross_tenant_request = AnalysisRunRequest(
        user_id=request.user_id,
        organization_id=request.organization_id,
        site_id="site_2",
        upload_id=request.upload_id,
        run_id=request.run_id,
        report_id=request.report_id,
        filename=request.filename,
        raw_meter_data=request.raw_meter_data,
        client_config=request.client_config,
    )

    try:
        execute_analysis_run(store, cross_tenant_request)
        assert False, "Expected cross-tenant site run execution to be denied"
    except AccessDenied:
        pass

    assert store.list_uploads("org_1") == []
    assert store.list_runs("org_1") == []
    assert store.list_audit_events("org_1") == []


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
    notifications = store.list_notifications("org_1", recipient_user_id="user_1")
    assert [notification["event_type"] for notification in notifications] == ["run_failed"]
    assert "could not complete" in notifications[0]["body"]
