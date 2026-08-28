import json
from datetime import datetime, timezone

from backend.artifacts import ARTIFACT_STORE_BACKEND_ENV, ARTIFACT_STORE_DIR_ENV
from backend.artifacts import InMemoryTenantArtifactStore
from backend.database import DATABASE_URL_ENV
from backend.database import initialize_database
from backend.domain import (
    Membership,
    Organization,
    Report,
    ReportSchedule,
    ReportScheduleFrequency,
    Role,
    Run,
    RunStatus,
    Site,
    Upload,
    UploadStatus,
    User,
)
from backend.notifications import InMemoryEmailSender
from backend.run_orchestration import StoredUploadAnalysisRunRequest, queue_analysis_run_from_upload
from backend.store import SaaSStore
from backend.worker_app import (
    AnalysisWorkerConfig,
    REPORT_SCHEDULE_DISPATCH_EVENT_TYPE,
    WORKER_BATCH_SIZE_ENV,
    WORKER_HEARTBEAT_MAX_AGE_SECONDS_ENV,
    WORKER_ID_ENV,
    WORKER_ORGANIZATION_ID_ENV,
    WORKER_POLL_SECONDS_ENV,
    WORKER_RUN_ONCE_ENV,
    WORKER_SUPERVISOR_MAX_AGE_DAYS_ENV,
    WORKER_SUPERVISOR_URL_ENV,
    WORKER_SUPERVISOR_VERIFIED_AT_ENV,
    process_worker_batch,
    record_report_schedule_dispatch_event,
    run_once_from_env,
    worker_config_from_env,
    worker_heartbeat_max_age_seconds_from_env,
    worker_supervision_status,
)
from backend.workers import AnalysisWorkerSummary
from analytics.config_loader import load_client_config


def test_worker_config_from_env_uses_defaults(monkeypatch):
    monkeypatch.delenv(WORKER_POLL_SECONDS_ENV, raising=False)
    monkeypatch.delenv(WORKER_BATCH_SIZE_ENV, raising=False)
    monkeypatch.delenv(WORKER_ORGANIZATION_ID_ENV, raising=False)
    monkeypatch.delenv(WORKER_ID_ENV, raising=False)
    monkeypatch.delenv(WORKER_HEARTBEAT_MAX_AGE_SECONDS_ENV, raising=False)

    config = worker_config_from_env()

    assert config.poll_seconds == 10.0
    assert config.batch_size == 10
    assert config.organization_id is None
    assert config.worker_id == "default"


def test_worker_config_from_env_accepts_overrides(monkeypatch):
    monkeypatch.setenv(WORKER_POLL_SECONDS_ENV, "0.5")
    monkeypatch.setenv(WORKER_BATCH_SIZE_ENV, "3")
    monkeypatch.setenv(WORKER_ORGANIZATION_ID_ENV, " org_123 ")
    monkeypatch.setenv(WORKER_ID_ENV, " worker-a ")

    config = worker_config_from_env()

    assert config.poll_seconds == 0.5
    assert config.batch_size == 3
    assert config.organization_id == "org_123"
    assert config.worker_id == "worker-a"


def test_run_once_from_env_accepts_boolean_values(monkeypatch):
    monkeypatch.delenv(WORKER_RUN_ONCE_ENV, raising=False)
    assert run_once_from_env() is False

    monkeypatch.setenv(WORKER_RUN_ONCE_ENV, "true")
    assert run_once_from_env() is True

    monkeypatch.setenv(WORKER_RUN_ONCE_ENV, "0")
    assert run_once_from_env() is False


def test_worker_config_from_env_rejects_invalid_values(monkeypatch):
    monkeypatch.setenv(WORKER_BATCH_SIZE_ENV, "0")
    try:
        worker_config_from_env()
        assert False, "Expected invalid worker batch size to fail"
    except ValueError as exc:
        assert WORKER_BATCH_SIZE_ENV in str(exc)

    monkeypatch.setenv(WORKER_BATCH_SIZE_ENV, "2")
    monkeypatch.setenv(WORKER_POLL_SECONDS_ENV, "-1")
    try:
        worker_config_from_env()
        assert False, "Expected invalid worker poll interval to fail"
    except ValueError as exc:
        assert WORKER_POLL_SECONDS_ENV in str(exc)

    monkeypatch.setenv(WORKER_POLL_SECONDS_ENV, "1")
    monkeypatch.setenv(WORKER_HEARTBEAT_MAX_AGE_SECONDS_ENV, "0")
    try:
        worker_heartbeat_max_age_seconds_from_env()
        assert False, "Expected invalid worker heartbeat age to fail"
    except ValueError as exc:
        assert WORKER_HEARTBEAT_MAX_AGE_SECONDS_ENV in str(exc)


def test_worker_supervision_status_is_optional_in_development(monkeypatch):
    monkeypatch.delenv(WORKER_SUPERVISOR_URL_ENV, raising=False)
    monkeypatch.delenv(WORKER_SUPERVISOR_VERIFIED_AT_ENV, raising=False)
    monkeypatch.delenv(WORKER_SUPERVISOR_MAX_AGE_DAYS_ENV, raising=False)

    configured, detail = worker_supervision_status("development")

    assert configured is True
    assert "optional" in detail


def test_worker_supervision_status_validates_production_evidence(monkeypatch):
    monkeypatch.setenv(WORKER_SUPERVISOR_URL_ENV, "https://ops.example.com/workers/rds")
    monkeypatch.setenv(WORKER_SUPERVISOR_VERIFIED_AT_ENV, "2026-07-20T12:00:00+00:00")
    monkeypatch.setenv(WORKER_SUPERVISOR_MAX_AGE_DAYS_ENV, "7")

    configured, detail = worker_supervision_status(
        "production",
        now=datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc),
    )

    assert configured is True
    assert WORKER_SUPERVISOR_URL_ENV in detail
    assert "1 day(s)" in detail


def test_process_worker_batch_boots_from_runtime_env(monkeypatch, tmp_path):
    monkeypatch.setenv(DATABASE_URL_ENV, f"sqlite:///{tmp_path / 'metadata.db'}")
    monkeypatch.setenv(ARTIFACT_STORE_BACKEND_ENV, "local")
    monkeypatch.setenv(ARTIFACT_STORE_DIR_ENV, str(tmp_path / "artifacts"))
    monkeypatch.setenv(WORKER_BATCH_SIZE_ENV, "5")
    monkeypatch.setenv(WORKER_POLL_SECONDS_ENV, "0")
    monkeypatch.delenv(WORKER_ORGANIZATION_ID_ENV, raising=False)

    summary = process_worker_batch()

    assert summary.attempted == 0
    assert summary.succeeded == 0
    assert summary.failed == 0
    assert summary.skipped == 0
    assert summary.scheduled_attempted == 0
    assert summary.scheduled_sent == 0
    assert summary.scheduled_failed == 0


def test_process_worker_batch_sends_due_report_schedules():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    sender = InMemoryEmailSender()
    now = datetime.now(timezone.utc)

    store.create_user(User(id="user_1", email="owner@example.com", display_name="Owner"))
    store.create_organization(Organization(id="org_1", name="Example Energy", slug="example"))
    store.add_membership(Membership(user_id="user_1", organization_id="org_1", role=Role.OWNER))
    store.create_site(Site(id="site_1", organization_id="org_1", name="Main Site", timezone="UTC"))
    upload_key = "tenants/org_1/sites/site_1/uploads/upload_1/energy.csv"
    artifact_store.put_text("org_1", upload_key, "Date,kWh\n2026-01-01,100\n")
    store.create_upload(
        Upload(
            id="upload_1",
            organization_id="org_1",
            site_id="site_1",
            uploaded_by_user_id="user_1",
            category="energy",
            storage_key=upload_key,
            checksum="checksum-upload-1",
            status=UploadStatus.STORED,
        )
    )
    store.create_run(
        Run(
            id="run_1",
            organization_id="org_1",
            site_id="site_1",
            requested_by_user_id="user_1",
            upload_ids=["upload_1"],
            status=RunStatus.SUCCEEDED,
        )
    )
    report_key = "tenants/org_1/sites/site_1/runs/run_1/reports/iso-summary.json"
    artifact_store.put_text("org_1", report_key, json.dumps({"total_records": 1}))
    store.create_report(
        Report(
            id="report_1",
            organization_id="org_1",
            run_id="run_1",
            report_type="iso_summary",
            storage_key=report_key,
            is_published=True,
        )
    )
    store.create_report_schedule(
        ReportSchedule(
            id="schedule_1",
            organization_id="org_1",
            report_id="report_1",
            created_by_user_id="user_1",
            frequency=ReportScheduleFrequency.MONTHLY,
            recipient_email="manager@example.com",
            next_run_at=now,
        )
    )

    summary = process_worker_batch(
        AnalysisWorkerConfig(batch_size=5, poll_seconds=0),
        store=store,
        artifact_store=artifact_store,
        email_sender=sender,
    )

    assert summary.attempted == 0
    assert summary.scheduled_attempted == 1
    assert summary.scheduled_sent == 1
    assert summary.scheduled_failed == 0
    assert [message.to_email for message in sender.sent_messages] == ["manager@example.com"]
    heartbeat = store.get_latest_worker_heartbeat()
    assert heartbeat is not None
    assert heartbeat["worker_id"] == "default"
    assert heartbeat["scope"] == "all"
    assert heartbeat["last_scheduled_attempted"] == 1
    assert heartbeat["last_scheduled_sent"] == 1
    events = store.list_platform_operation_events(
        event_type=REPORT_SCHEDULE_DISPATCH_EVENT_TYPE,
    )
    assert len(events) == 1
    assert events[0]["status"] == "succeeded"
    assert events[0]["request_id"] is None
    assert events[0]["detail"] == (
        "scope=all scheduled_attempted=1 scheduled_sent=1 "
        "scheduled_failed=0 scheduled_skipped=0"
    )
    serialized_event = json.dumps(dict(events[0]))
    assert "org_1" not in serialized_event
    assert "schedule_1" not in serialized_event
    assert "report_1" not in serialized_event
    assert "manager@example.com" not in serialized_event


def test_report_schedule_dispatch_event_records_failed_counts_without_tenant_ids():
    store = SaaSStore(initialize_database())
    summary = AnalysisWorkerSummary(
        attempted=0,
        succeeded=0,
        failed=0,
        skipped=0,
        scheduled_attempted=2,
        scheduled_sent=1,
        scheduled_failed=1,
    )

    event_id = record_report_schedule_dispatch_event(
        store,
        AnalysisWorkerConfig(
            batch_size=10,
            poll_seconds=0,
            organization_id="org_sensitive",
            worker_id="worker_sensitive",
        ),
        summary,
        observed_at=datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc),
    )

    assert event_id is not None
    event = store.list_platform_operation_events(
        event_type=REPORT_SCHEDULE_DISPATCH_EVENT_TYPE,
    )[0]
    assert event["id"] == event_id
    assert event["status"] == "failed"
    assert event["request_id"] is None
    assert event["created_at"] == "2026-07-20T12:00:00+00:00"
    assert event["detail"] == (
        "scope=organization scheduled_attempted=2 scheduled_sent=1 "
        "scheduled_failed=1 scheduled_skipped=0"
    )
    serialized_event = json.dumps(dict(event))
    assert "org_sensitive" not in serialized_event
    assert "worker_sensitive" not in serialized_event


def test_report_schedule_dispatch_event_skips_empty_batches():
    store = SaaSStore(initialize_database())

    event_id = record_report_schedule_dispatch_event(
        store,
        AnalysisWorkerConfig(batch_size=10, poll_seconds=0),
        AnalysisWorkerSummary(attempted=0, succeeded=0, failed=0, skipped=0),
        observed_at=datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc),
    )

    assert event_id is None
    assert store.list_platform_operation_events(
        event_type=REPORT_SCHEDULE_DISPATCH_EVENT_TYPE,
    ) == []


def test_process_worker_batch_scopes_jobs_schedules_and_email_to_selected_tenant():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    sender = InMemoryEmailSender()
    now = datetime.now(timezone.utc)

    def seed_tenant_with_job_and_schedule(
        *,
        user_id: str,
        organization_id: str,
        site_id: str,
        upload_id: str,
        run_id: str,
        report_id: str,
        scheduled_run_id: str,
        scheduled_report_id: str,
        schedule_id: str,
    ) -> None:
        store.create_user(User(id=user_id, email=f"{user_id}@example.com", display_name=user_id))
        store.create_organization(
            Organization(id=organization_id, name=f"{organization_id} Energy", slug=organization_id)
        )
        store.add_membership(Membership(user_id=user_id, organization_id=organization_id, role=Role.OWNER))
        store.create_site(Site(id=site_id, organization_id=organization_id, name=site_id, timezone="UTC"))

        upload_key = f"tenants/{organization_id}/sites/{site_id}/uploads/{upload_id}/energy.csv"
        artifact_store.put_text(
            organization_id,
            upload_key,
            "Date,Main Electricity,Main Gas\n2026-01-01,100,80\n2026-02-01,110,70\n",
        )
        store.create_upload(
            Upload(
                id=upload_id,
                organization_id=organization_id,
                site_id=site_id,
                uploaded_by_user_id=user_id,
                category="energy",
                storage_key=upload_key,
                checksum=f"checksum-{upload_id}",
                status=UploadStatus.STORED,
            )
        )
        queue_analysis_run_from_upload(
            store,
            StoredUploadAnalysisRunRequest(
                user_id=user_id,
                organization_id=organization_id,
                upload_id=upload_id,
                run_id=run_id,
                report_id=report_id,
                client_config=load_client_config("config/clients/example_client.yaml"),
            ),
        )

        store.create_run(
            Run(
                id=scheduled_run_id,
                organization_id=organization_id,
                site_id=site_id,
                requested_by_user_id=user_id,
                upload_ids=[upload_id],
                status=RunStatus.SUCCEEDED,
            )
        )
        scheduled_report_key = (
            f"tenants/{organization_id}/sites/{site_id}/runs/{scheduled_run_id}/reports/iso-summary.json"
        )
        artifact_store.put_text(
            organization_id,
            scheduled_report_key,
            json.dumps({"total_records": 1, "source_package": {"upload_count": 1, "categories": ["energy"]}}),
        )
        store.create_report(
            Report(
                id=scheduled_report_id,
                organization_id=organization_id,
                run_id=scheduled_run_id,
                report_type="iso_summary",
                storage_key=scheduled_report_key,
                is_published=True,
            )
        )
        store.create_report_schedule(
            ReportSchedule(
                id=schedule_id,
                organization_id=organization_id,
                report_id=scheduled_report_id,
                created_by_user_id=user_id,
                frequency=ReportScheduleFrequency.MONTHLY,
                recipient_email=f"{organization_id}.manager@example.com",
                next_run_at=now,
            )
        )

    seed_tenant_with_job_and_schedule(
        user_id="user_1",
        organization_id="org_1",
        site_id="site_1",
        upload_id="upload_1",
        run_id="run_1",
        report_id="report_1",
        scheduled_run_id="scheduled_run_1",
        scheduled_report_id="scheduled_report_1",
        schedule_id="schedule_1",
    )
    seed_tenant_with_job_and_schedule(
        user_id="user_2",
        organization_id="org_2",
        site_id="site_2",
        upload_id="upload_2",
        run_id="run_2",
        report_id="report_2",
        scheduled_run_id="scheduled_run_2",
        scheduled_report_id="scheduled_report_2",
        schedule_id="schedule_2",
    )

    summary = process_worker_batch(
        AnalysisWorkerConfig(batch_size=10, poll_seconds=0, organization_id="org_1"),
        store=store,
        artifact_store=artifact_store,
        email_sender=sender,
    )

    assert summary.attempted == 1
    assert summary.succeeded == 1
    assert summary.failed == 0
    assert summary.scheduled_attempted == 1
    assert summary.scheduled_sent == 1
    assert summary.scheduled_failed == 0
    assert store.list_analysis_jobs("org_1")[0]["status"] == "succeeded"
    assert store.list_analysis_jobs("org_2")[0]["status"] == "queued"
    assert store.get_report("report_1") is not None
    assert store.get_report("report_2") is None
    assert store.get_report_schedule("schedule_1")["last_delivery_status"] == "sent"
    assert store.get_report_schedule("schedule_2")["last_delivery_status"] is None
    assert {message.organization_id for message in sender.sent_messages} == {"org_1"}
    assert "org_1.manager@example.com" in {message.to_email for message in sender.sent_messages}
    assert "org_2.manager@example.com" not in {message.to_email for message in sender.sent_messages}
    scoped_heartbeat = store.get_latest_worker_heartbeat("org_1")
    all_tenant_heartbeat = store.get_latest_worker_heartbeat()
    assert scoped_heartbeat is not None
    assert scoped_heartbeat["scope"] == "organization"
    assert scoped_heartbeat["organization_id"] == "org_1"
    assert scoped_heartbeat["last_attempted"] == 1
    assert scoped_heartbeat["last_scheduled_sent"] == 1
    assert all_tenant_heartbeat is None
