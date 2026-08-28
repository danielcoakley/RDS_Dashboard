import json
from datetime import datetime, timedelta, timezone

from backend import report_scheduling
from backend.artifacts import InMemoryTenantArtifactStore
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
from backend.report_scheduling import (
    SYSTEM_REPORT_SCHEDULER_USER_ID,
    process_due_report_schedules,
)
from backend.store import SaaSStore


def seed_scheduled_report(
    store: SaaSStore,
    artifact_store: InMemoryTenantArtifactStore,
    *,
    user_id: str,
    organization_id: str,
    site_id: str,
    upload_id: str,
    run_id: str,
    report_id: str,
    schedule_id: str,
    next_run_at: datetime,
    write_artifact: bool = True,
) -> str:
    store.create_user(User(id=user_id, email=f"{user_id}@example.com", display_name=user_id))
    store.create_organization(
        Organization(id=organization_id, name=f"{organization_id} Energy", slug=organization_id)
    )
    store.add_membership(Membership(user_id=user_id, organization_id=organization_id, role=Role.OWNER))
    store.create_site(Site(id=site_id, organization_id=organization_id, name=site_id, timezone="UTC"))

    upload_key = f"tenants/{organization_id}/sites/{site_id}/uploads/{upload_id}/energy.csv"
    artifact_store.put_text(organization_id, upload_key, "Date,kWh\n2026-01-01,100\n")
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
    store.create_run(
        Run(
            id=run_id,
            organization_id=organization_id,
            site_id=site_id,
            requested_by_user_id=user_id,
            upload_ids=[upload_id],
            status=RunStatus.SUCCEEDED,
            completed_at=next_run_at - timedelta(days=1),
        )
    )

    report_key = f"tenants/{organization_id}/sites/{site_id}/runs/{run_id}/reports/iso-summary.json"
    if write_artifact:
        artifact_store.put_text(
            organization_id,
            report_key,
            json.dumps(
                {
                    "annual_kwh": 12345,
                    "legacy_baseline": {
                        "weather_normalization": {
                            "estimated_savings": 42.5,
                            "model_count": 1,
                            "skipped_meter_count": 1,
                            "status": "modeled",
                        }
                    },
                    "source_package": {
                        "categories": ["energy", "hdd", "cdd", "seu_mapping"],
                        "upload_count": 4,
                    },
                    "tenant_sensitive_note": f"{organization_id} management review",
                    "total_records": 4,
                },
                sort_keys=True,
            ),
        )
    store.create_report(
        Report(
            id=report_id,
            organization_id=organization_id,
            run_id=run_id,
            report_type="iso_summary",
            storage_key=report_key,
            is_published=True,
        )
    )
    store.create_report_schedule(
        ReportSchedule(
            id=schedule_id,
            organization_id=organization_id,
            report_id=report_id,
            created_by_user_id=user_id,
            frequency=ReportScheduleFrequency.MONTHLY,
            recipient_email=f"{organization_id}.manager@example.com",
            next_run_at=next_run_at,
        )
    )
    return report_key


def test_due_report_schedule_sends_notification_without_artifact_content_and_audits():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    sender = InMemoryEmailSender()
    now = datetime(2026, 1, 20, 9, 0, tzinfo=timezone.utc)
    report_key = seed_scheduled_report(
        store,
        artifact_store,
        user_id="user_1",
        organization_id="org_1",
        site_id="site_1",
        upload_id="upload_1",
        run_id="run_1",
        report_id="report_1",
        schedule_id="schedule_1",
        next_run_at=datetime(2026, 1, 10, 9, 0, tzinfo=timezone.utc),
    )

    summary = process_due_report_schedules(store, artifact_store, sender, now=now)

    assert summary.attempted == 1
    assert summary.sent == 1
    assert summary.failed == 0
    assert summary.skipped == 0
    assert len(sender.sent_messages) == 1
    message = sender.sent_messages[0]
    assert message.organization_id == "org_1"
    assert message.to_email == "org_1.manager@example.com"
    assert "report_1" in message.body
    assert "Report digest:" in message.body
    assert "Total records: 4" in message.body
    assert "Source files: 4" in message.body
    assert "Source categories: energy, hdd, cdd, seu_mapping" in message.body
    assert "Weather normalization: modeled" in message.body
    assert "Weather models: 1" in message.body
    assert "Skipped weather models: 1" in message.body
    assert "Estimated normalized savings: 42.5" in message.body
    assert "Sign in" in message.body
    assert "annual_kwh" not in message.body
    assert "12345" not in message.body
    assert "tenant_sensitive_note" not in message.body
    assert "org_1 management review" not in message.body
    assert report_key not in message.body

    schedule = store.get_report_schedule("schedule_1")
    assert schedule["next_run_at"] == "2026-02-10T09:00:00+00:00"
    assert schedule["last_delivery_at"] == now.isoformat()
    assert schedule["last_delivery_status"] == "sent"
    assert schedule["last_delivery_error"] is None

    deliveries = store.list_report_schedule_deliveries("org_1", "schedule_1")
    assert len(deliveries) == 1
    assert deliveries[0]["status"] == "sent"
    assert deliveries[0]["sent_at"] == now.isoformat()
    assert deliveries[0]["recipient_email"] == "org_1.manager@example.com"

    audit_events = store.list_audit_events("org_1")
    audit_metadata = json.loads(audit_events[-1]["metadata_json"])
    assert store.get_user(SYSTEM_REPORT_SCHEDULER_USER_ID) is not None
    assert audit_events[-1]["action"] == "report_schedule_delivered"
    assert audit_events[-1]["actor_user_id"] == SYSTEM_REPORT_SCHEDULER_USER_ID
    assert audit_metadata["delivery_status"] == "sent"
    assert audit_metadata["report_id"] == "report_1"
    assert audit_metadata["next_run_at"] == "2026-02-10T09:00:00+00:00"
    assert "recipient_email" not in audit_metadata
    assert "org_1.manager@example.com" not in audit_events[-1]["metadata_json"]
    assert "storage_key" not in audit_events[-1]["metadata_json"]
    assert "annual_kwh" not in audit_events[-1]["metadata_json"]
    assert "tenant_sensitive_note" not in audit_events[-1]["metadata_json"]


def test_due_report_schedule_sanitizes_legacy_artifact_before_digest(monkeypatch):
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    sender = InMemoryEmailSender()
    now = datetime(2026, 1, 20, 9, 0, tzinfo=timezone.utc)
    report_key = seed_scheduled_report(
        store,
        artifact_store,
        user_id="user_1",
        organization_id="org_1",
        site_id="site_1",
        upload_id="upload_1",
        run_id="run_1",
        report_id="report_1",
        schedule_id="schedule_1",
        next_run_at=datetime(2026, 1, 10, 9, 0, tzinfo=timezone.utc),
    )
    legacy_artifact = json.loads(artifact_store.get_text("org_1", report_key))
    legacy_artifact["storage_key"] = report_key
    legacy_artifact["source_package"]["storage_key"] = report_key
    legacy_artifact["source_package"]["analysis_context"] = {
        "execution_input": "saas_meter_csv",
        "resolved_record_count": 4,
        "analysis_input_path": report_key,
        "storage_key": report_key,
        "sources": [
            {
                "upload_id": "upload_1",
                "category": "energy",
                "storage_key": report_key,
                "analysis_input_path": report_key,
            }
        ],
    }
    artifact_store.put_text("org_1", report_key, json.dumps(legacy_artifact))

    captured_digest_content: list[dict[str, object]] = []

    def capture_digest_content(content: dict[str, object]) -> list[str]:
        captured_digest_content.append(content)
        return ["Captured safe digest"]

    monkeypatch.setattr(
        report_scheduling,
        "report_artifact_digest_lines",
        capture_digest_content,
    )

    summary = process_due_report_schedules(store, artifact_store, sender, now=now)

    assert summary.sent == 1
    assert len(captured_digest_content) == 1
    sanitized_digest_content = json.dumps(captured_digest_content[0])
    assert "storage_key" not in sanitized_digest_content
    assert "analysis_input_path" not in sanitized_digest_content
    assert report_key not in sanitized_digest_content
    assert "Captured safe digest" in sender.sent_messages[0].body
    assert report_key not in sender.sent_messages[0].body


def test_due_report_schedule_failure_is_tenant_scoped_and_advances_schedule():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    sender = InMemoryEmailSender()
    now = datetime(2026, 1, 20, 9, 0, tzinfo=timezone.utc)
    seed_scheduled_report(
        store,
        artifact_store,
        user_id="user_1",
        organization_id="org_1",
        site_id="site_1",
        upload_id="upload_1",
        run_id="run_1",
        report_id="report_1",
        schedule_id="schedule_1",
        next_run_at=datetime(2026, 1, 10, 9, 0, tzinfo=timezone.utc),
        write_artifact=False,
    )
    seed_scheduled_report(
        store,
        artifact_store,
        user_id="user_2",
        organization_id="org_2",
        site_id="site_2",
        upload_id="upload_2",
        run_id="run_2",
        report_id="report_2",
        schedule_id="schedule_2",
        next_run_at=datetime(2026, 1, 10, 9, 0, tzinfo=timezone.utc),
    )

    summary = process_due_report_schedules(
        store,
        artifact_store,
        sender,
        organization_id="org_1",
        now=now,
    )

    assert summary.attempted == 1
    assert summary.sent == 0
    assert summary.failed == 1
    assert sender.sent_messages == []

    org_1_schedule = store.get_report_schedule("schedule_1")
    org_2_schedule = store.get_report_schedule("schedule_2")
    assert org_1_schedule["next_run_at"] == "2026-02-10T09:00:00+00:00"
    assert org_1_schedule["last_delivery_status"] == "failed"
    assert org_1_schedule["last_delivery_error"] == "Report artifact is unavailable"
    assert org_2_schedule["next_run_at"] == "2026-01-10T09:00:00+00:00"
    assert org_2_schedule["last_delivery_status"] is None

    deliveries = store.list_report_schedule_deliveries("org_1", "schedule_1")
    assert len(deliveries) == 1
    assert deliveries[0]["status"] == "failed"
    assert deliveries[0]["error_message"] == "Report artifact is unavailable"

    org_1_audit = store.list_audit_events("org_1")[-1]
    org_1_metadata = json.loads(org_1_audit["metadata_json"])
    assert org_1_audit["action"] == "report_schedule_failed"
    assert org_1_metadata["delivery_status"] == "failed"
    assert "recipient_email" not in org_1_metadata
    assert "org_1.manager@example.com" not in org_1_audit["metadata_json"]
    assert "storage_key" not in org_1_audit["metadata_json"]
    assert store.list_audit_events("org_2") == []
