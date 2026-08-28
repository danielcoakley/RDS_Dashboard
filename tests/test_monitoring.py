from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from backend.api import create_app
from backend.artifacts import InMemoryTenantArtifactStore
from backend.database import initialize_database
from backend.domain import (
    AnalysisJob,
    CspBlockedUriCategory,
    CspViolationReport,
    CspViolationReportDirective,
    CspViolationReportDisposition,
    MarketingInquiry,
    MarketingInquiryStatus,
    Membership,
    Meter,
    Notification,
    NotificationChannel,
    NotificationProviderEvent,
    NotificationProviderEventStatus,
    NotificationStatus,
    Organization,
    Report,
    ReportSchedule,
    ReportScheduleDelivery,
    ReportScheduleDeliveryStatus,
    ReportScheduleFrequency,
    Role,
    Run,
    RunStatus,
    Site,
    Upload,
    UploadStatus,
    User,
)
from backend.monitoring import (
    METRICS_TOKEN_ENV,
    MetricsAuthenticationError,
    MetricsConfigurationError,
    collect_operational_metrics,
    format_prometheus_metrics,
    verify_metrics_request,
)
from backend.operations import (
    ARTIFACT_STORE_TEST_EVENT_TYPE,
    BILLING_WEBHOOK_REPLAY_EVENT_TYPE,
    CLERK_WEBHOOK_REPLAY_EVENT_TYPE,
    DATABASE_TEST_EVENT_TYPE,
    NOTIFICATION_PROVIDER_WEBHOOK_REPLAY_EVENT_TYPE,
    RUN_LIFECYCLE_TEST_EVENT_TYPE,
    TENANT_ISOLATION_TEST_EVENT_TYPE,
    WORKER_RUN_LIFECYCLE_TEST_EVENT_TYPE,
)
from backend.runtime_config import RUNTIME_ENV_ENV
from backend.security import ALLOWED_ORIGINS_ENV
from backend.store import SaaSStore

PRODUCTION_METRICS_TOKEN = "metrics-token-for-production-tests-123456"


def seeded_metrics_store() -> SaaSStore:
    store = SaaSStore(initialize_database())

    store.create_user(User(id="user_1", email="owner@example.com", display_name="Owner"))
    store.create_user(
        User(
            id="user_2",
            email="inactive@example.com",
            display_name="Inactive",
            is_active=False,
        )
    )
    store.create_organization(
        Organization(id="org_1", name="Example Energy", slug="example-energy")
    )
    store.create_organization(Organization(id="org_2", name="Other Energy", slug="other-energy"))
    store.add_membership(
        Membership(user_id="user_1", organization_id="org_1", role=Role.OWNER)
    )
    store.add_membership(
        Membership(user_id="user_2", organization_id="org_2", role=Role.VIEWER)
    )

    store.create_site(
        Site(id="site_1", organization_id="org_1", name="Factory", timezone="Europe/London")
    )
    store.create_meter(
        Meter(
            id="meter_1",
            organization_id="org_1",
            site_id="site_1",
            display_name="Main meter",
            commodity="electricity",
            unit="kWh",
            source_column="Main kWh",
        )
    )

    upload_statuses = [
        UploadStatus.PENDING,
        UploadStatus.STORED,
        UploadStatus.REJECTED,
    ]
    for status in upload_statuses:
        store.create_upload(
            Upload(
                id=f"upload_{status.value}",
                organization_id="org_1",
                site_id="site_1",
                uploaded_by_user_id="user_1",
                category="energy",
                storage_key=f"org_1/uploads/{status.value}.csv",
                checksum=f"checksum-{status.value}",
                status=status,
            )
        )

    for status in RunStatus:
        run_id = f"run_{status.value}"
        store.create_run(
            Run(
                id=run_id,
                organization_id="org_1",
                site_id="site_1",
                requested_by_user_id="user_1",
                upload_ids=["upload_stored"],
                status=status,
            )
        )
        store.create_analysis_job(
            AnalysisJob(
                id=f"job_{status.value}",
                organization_id="org_1",
                run_id=run_id,
                upload_id="upload_stored",
                report_id=f"report_{status.value}",
                requested_by_user_id="user_1",
                client_config={"client": "example"},
                status=status,
            )
        )

    store.create_report(
        Report(
            id="report_schedule_metrics",
            organization_id="org_1",
            run_id="run_succeeded",
            report_type="iso_summary",
            storage_key="tenants/org_1/sites/site_1/runs/run_succeeded/reports/iso-summary.json",
            is_published=True,
        )
    )
    schedule_specs = [
        (
            "schedule_due",
            datetime(2026, 7, 20, 8, 0, tzinfo=timezone.utc),
            True,
        ),
        (
            "schedule_future",
            datetime(2099, 1, 1, 8, 10, tzinfo=timezone.utc),
            True,
        ),
        (
            "schedule_paused",
            datetime(2026, 7, 20, 7, 55, tzinfo=timezone.utc),
            False,
        ),
        (
            "schedule_failed",
            datetime(2099, 1, 1, 8, 20, tzinfo=timezone.utc),
            True,
        ),
    ]
    for schedule_id, next_run_at, is_active in schedule_specs:
        store.create_report_schedule(
            ReportSchedule(
                id=schedule_id,
                organization_id="org_1",
                report_id="report_schedule_metrics",
                created_by_user_id="user_1",
                frequency=ReportScheduleFrequency.MONTHLY,
                recipient_email="scheduler.manager@example.com",
                next_run_at=next_run_at,
                is_active=is_active,
            )
        )
    store.create_report_schedule_delivery(
        ReportScheduleDelivery(
            id="schedule_delivery_sent",
            organization_id="org_1",
            schedule_id="schedule_due",
            report_id="report_schedule_metrics",
            recipient_email="scheduler.manager@example.com",
            status=ReportScheduleDeliveryStatus.SENT,
            attempted_at=datetime(2026, 7, 20, 8, 0, tzinfo=timezone.utc),
            sent_at=datetime(2026, 7, 20, 8, 0, tzinfo=timezone.utc),
        )
    )
    store.create_report_schedule_delivery(
        ReportScheduleDelivery(
            id="schedule_delivery_failed",
            organization_id="org_1",
            schedule_id="schedule_failed",
            report_id="report_schedule_metrics",
            recipient_email="scheduler.manager@example.com",
            status=ReportScheduleDeliveryStatus.FAILED,
            error_message="smtp unavailable",
            attempted_at=datetime(2026, 7, 20, 8, 1, tzinfo=timezone.utc),
        )
    )
    store.update_report_schedule_delivery_state(
        "schedule_failed",
        next_run_at="2099-01-01T08:20:00+00:00",
        last_delivery_at="2026-07-20T08:01:00+00:00",
        last_delivery_status=ReportScheduleDeliveryStatus.FAILED,
        last_delivery_error="smtp unavailable",
    )

    for status in NotificationStatus:
        store.create_notification(
            Notification(
                id=f"notification_{status.value}",
                organization_id="org_1",
                recipient_user_id="user_1",
                channel=NotificationChannel.EMAIL,
                event_type="run_completed",
                subject=f"Run {status.value}",
                body="The run changed status.",
                resource_type="run",
                resource_id="run_succeeded",
                status=status,
            )
        )
    store.conn.execute(
        """
        UPDATE notifications
        SET created_at = ?
        WHERE id = ?
        """,
        ("2026-07-20T08:00:00+00:00", "notification_pending"),
    )
    store.create_notification(
        Notification(
            id="notification_retry_scheduled",
            organization_id="org_1",
            recipient_user_id="user_1",
            channel=NotificationChannel.EMAIL,
            event_type="report_approval_assigned",
            subject="Approval retry scheduled",
            body="The approval notification will be retried.",
            resource_type="report_approval_assignment",
            resource_id="approval_1",
            status=NotificationStatus.PENDING,
            attempt_count=1,
            last_attempt_at=datetime(2026, 7, 20, 8, 3, tzinfo=timezone.utc),
            next_attempt_at=datetime(2030, 1, 1, 8, 10, tzinfo=timezone.utc),
        )
    )

    for status in MarketingInquiryStatus:
        store.create_marketing_inquiry(
            MarketingInquiry(
                id=f"inquiry_{status.value}",
                email=f"{status.value}@example.com",
                name=f"{status.value.title()} Contact",
                organization_name="Inquiry Co",
                role="Energy Lead",
                intended_use="audit_readiness",
                message="Need a pilot.",
                source_page="/",
                status=status,
            )
        )

    store.create_marketing_inquiry_submission_event(
        event_id="marketing_submission_accepted",
        source_key_hash="source-hash-accepted",
        inquiry_id="inquiry_new",
        outcome="accepted",
        created_at_iso="2026-07-20T08:00:00+00:00",
    )
    store.create_marketing_inquiry_submission_event(
        event_id="marketing_submission_rate_limited",
        source_key_hash="source-hash-rate-limited",
        inquiry_id=None,
        outcome="rate_limited",
        created_at_iso="2026-07-20T08:01:00+00:00",
    )
    store.upsert_worker_heartbeat(
        worker_id="worker_all",
        organization_id=None,
        poll_seconds=10,
        batch_size=5,
        last_attempted=2,
        last_succeeded=1,
        last_failed=1,
        last_skipped=0,
        last_scheduled_attempted=3,
        last_scheduled_sent=2,
        last_scheduled_failed=1,
        last_scheduled_skipped=0,
        last_seen_at="2026-07-20T08:02:00+00:00",
    )
    store.upsert_worker_heartbeat(
        worker_id="worker_org_1",
        organization_id="org_1",
        poll_seconds=10,
        batch_size=5,
        last_attempted=1,
        last_succeeded=1,
        last_failed=0,
        last_skipped=0,
        last_scheduled_attempted=1,
        last_scheduled_sent=1,
        last_scheduled_failed=0,
        last_scheduled_skipped=0,
        last_seen_at="2026-07-20T08:03:00+00:00",
    )
    store.create_platform_operation_event(
        event_id="platform_operation_alert_success",
        event_type="rds.alert_delivery_test",
        event_status="succeeded",
        request_id="req_alert_success",
        detail="delivered http_status=204",
        created_at="2026-07-20T08:01:00+00:00",
    )
    store.create_platform_operation_event(
        event_id="platform_operation_alert_failure",
        event_type="rds.alert_delivery_test",
        event_status="failed",
        request_id="req_alert_failure",
        detail="webhook_error=http_status_503",
        created_at="2026-07-20T08:02:00+00:00",
    )
    store.create_platform_operation_event(
        event_id="platform_operation_schedule_dispatch_success",
        event_type="rds.report_schedule_dispatch",
        event_status="succeeded",
        request_id=None,
        detail=(
            "scope=all scheduled_attempted=3 scheduled_sent=2 "
            "scheduled_failed=0 scheduled_skipped=1"
        ),
        created_at="2026-07-20T08:03:00+00:00",
    )
    store.create_platform_operation_event(
        event_id="platform_operation_schedule_dispatch_failure",
        event_type="rds.report_schedule_dispatch",
        event_status="failed",
        request_id=None,
        detail=(
            "scope=organization scheduled_attempted=2 scheduled_sent=1 "
            "scheduled_failed=1 scheduled_skipped=0"
        ),
        created_at="2026-07-20T08:03:30+00:00",
    )
    store.create_platform_operation_event(
        event_id="platform_operation_clerk_replay_success",
        event_type=CLERK_WEBHOOK_REPLAY_EVENT_TYPE,
        event_status="succeeded",
        request_id="req_clerk_replay",
        detail="action=membership_upserted",
        created_at="2026-07-20T08:03:40+00:00",
    )
    store.create_platform_operation_event(
        event_id="platform_operation_billing_replay_success",
        event_type=BILLING_WEBHOOK_REPLAY_EVENT_TYPE,
        event_status="succeeded",
        request_id="req_billing_replay",
        detail="provider=stripe action=subscription_updated duplicate=false",
        created_at="2026-07-20T08:03:45+00:00",
    )
    store.create_platform_operation_event(
        event_id="platform_operation_notification_replay_failure",
        event_type=NOTIFICATION_PROVIDER_WEBHOOK_REPLAY_EVENT_TYPE,
        event_status="failed",
        request_id="req_notification_replay",
        detail="status=unsupported",
        created_at="2026-07-20T08:03:50+00:00",
    )
    store.create_platform_operation_event(
        event_id="platform_operation_artifact_store_test_success",
        event_type=ARTIFACT_STORE_TEST_EVENT_TYPE,
        event_status="succeeded",
        request_id="req_artifact_store_test",
        detail="bytes_written=64",
        created_at="2026-07-20T08:03:55+00:00",
    )
    store.create_platform_operation_event(
        event_id="platform_operation_artifact_store_test_failure",
        event_type=ARTIFACT_STORE_TEST_EVENT_TYPE,
        event_status="failed",
        request_id="req_artifact_store_test_failed",
        detail="Artifact store smoke test failed during write",
        created_at="2026-07-20T08:03:56+00:00",
    )
    store.create_platform_operation_event(
        event_id="platform_operation_database_test_success",
        event_type=DATABASE_TEST_EVENT_TYPE,
        event_status="succeeded",
        request_id="req_database_test",
        detail="write_read_verified=true",
        created_at="2026-07-20T08:03:58+00:00",
    )
    store.create_platform_operation_event(
        event_id="platform_operation_database_test_failure",
        event_type=DATABASE_TEST_EVENT_TYPE,
        event_status="failed",
        request_id="req_database_test_failed",
        detail="Database smoke test failed during write/read",
        created_at="2026-07-20T08:03:59+00:00",
    )
    store.create_platform_operation_event(
        event_id="platform_operation_tenant_isolation_test_success",
        event_type=TENANT_ISOLATION_TEST_EVENT_TYPE,
        event_status="succeeded",
        request_id="req_tenant_isolation_test",
        detail="checks_passed=6 synthetic_records=12",
        created_at="2026-07-20T08:03:57+00:00",
    )
    store.create_platform_operation_event(
        event_id="platform_operation_tenant_isolation_test_failure",
        event_type=TENANT_ISOLATION_TEST_EVENT_TYPE,
        event_status="failed",
        request_id="req_tenant_isolation_test_failed",
        detail="Tenant isolation smoke test failed for site listing",
        created_at="2026-07-20T08:03:59+00:00",
    )
    store.create_platform_operation_event(
        event_id="platform_operation_run_lifecycle_test_success",
        event_type=RUN_LIFECYCLE_TEST_EVENT_TYPE,
        event_status="succeeded",
        request_id="req_run_lifecycle_test",
        detail="checks_passed=6 artifacts_verified=2 synthetic_records=15",
        created_at="2026-07-20T08:03:54+00:00",
    )
    store.create_platform_operation_event(
        event_id="platform_operation_run_lifecycle_test_failure",
        event_type=RUN_LIFECYCLE_TEST_EVENT_TYPE,
        event_status="failed",
        request_id="req_run_lifecycle_test_failed",
        detail="Run lifecycle smoke test failed during upload-to-report execution",
        created_at="2026-07-20T08:03:59+00:00",
    )
    store.create_platform_operation_event(
        event_id="platform_operation_worker_run_lifecycle_test_success",
        event_type=WORKER_RUN_LIFECYCLE_TEST_EVENT_TYPE,
        event_status="succeeded",
        request_id="req_worker_run_lifecycle_test",
        detail=(
            "checks_passed=8 jobs_attempted=1 jobs_succeeded=1 "
            "artifacts_verified=2 synthetic_records=15"
        ),
        created_at="2026-07-20T08:03:53+00:00",
    )
    store.create_platform_operation_event(
        event_id="platform_operation_worker_run_lifecycle_test_failure",
        event_type=WORKER_RUN_LIFECYCLE_TEST_EVENT_TYPE,
        event_status="failed",
        request_id="req_worker_run_lifecycle_test_failed",
        detail="Worker run lifecycle smoke test failed during queued job processing",
        created_at="2026-07-20T08:03:59+00:00",
    )
    store.create_notification_provider_event(
        NotificationProviderEvent(
            id="provider_event_delivered",
            provider="smtp-relay",
            provider_event_id="provider_evt_delivered",
            notification_id="notification_sent",
            organization_id="org_1",
            status=NotificationProviderEventStatus.DELIVERED,
            detail="provider_status=delivered",
            occurred_at=datetime(2026, 7, 20, 8, 2, tzinfo=timezone.utc),
        )
    )
    store.create_notification_provider_event(
        NotificationProviderEvent(
            id="provider_event_bounced",
            provider="smtp-relay",
            provider_event_id="provider_evt_bounced",
            notification_id="notification_failed",
            organization_id="org_1",
            status=NotificationProviderEventStatus.BOUNCED,
            detail="provider_status=bounced",
            occurred_at=datetime(2026, 7, 20, 8, 3, tzinfo=timezone.utc),
        )
    )
    store.create_csp_violation_report(
        CspViolationReport(
            id="csp_report_script",
            disposition=CspViolationReportDisposition.REPORT,
            effective_directive=CspViolationReportDirective.SCRIPT_SRC,
            violated_directive=CspViolationReportDirective.SCRIPT_SRC,
            blocked_uri_category=CspBlockedUriCategory.HTTPS,
            document_origin="https://tenant.example.com",
            status_code=200,
            created_at=datetime(2026, 7, 20, 8, 2, tzinfo=timezone.utc),
        )
    )
    store.create_csp_violation_report(
        CspViolationReport(
            id="csp_report_connect",
            disposition=CspViolationReportDisposition.ENFORCE,
            effective_directive=CspViolationReportDirective.CONNECT_SRC,
            violated_directive=CspViolationReportDirective.CONNECT_SRC,
            blocked_uri_category=CspBlockedUriCategory.HTTP,
            document_origin="https://tenant.example.com",
            status_code=0,
            created_at=datetime(2026, 7, 20, 8, 3, tzinfo=timezone.utc),
        )
    )

    return store


def test_collect_operational_metrics_returns_safe_aggregate_counts():
    store = seeded_metrics_store()

    metrics = collect_operational_metrics(
        store,
        now=datetime(2026, 7, 20, 8, 4, tzinfo=timezone.utc),
    )
    output = format_prometheus_metrics(metrics)

    assert metrics.organizations_total == 2
    assert metrics.users_total == 2
    assert metrics.users_active_total == 1
    assert metrics.sites_total == 1
    assert metrics.meters_total == 1
    assert metrics.uploads_by_status == {"pending": 1, "stored": 1, "rejected": 1}
    assert metrics.runs_by_status == {
        "queued": 1,
        "running": 1,
        "succeeded": 1,
        "failed": 1,
    }
    assert metrics.analysis_jobs_by_status == {
        "queued": 1,
        "running": 1,
        "succeeded": 1,
        "failed": 1,
    }
    assert metrics.notifications_by_status == {"pending": 2, "sent": 1, "failed": 1}
    assert metrics.notification_provider_events_by_status == {
        "delivered": 1,
        "bounced": 1,
        "complained": 0,
        "deferred": 0,
        "dropped": 0,
    }
    assert metrics.csp_violation_reports_by_disposition == {
        "enforce": 1,
        "report": 1,
    }
    assert metrics.csp_violation_reports_by_effective_directive["script-src"] == 1
    assert metrics.csp_violation_reports_by_effective_directive["connect-src"] == 1
    assert metrics.csp_violation_reports_by_effective_directive["default-src"] == 0
    assert metrics.notification_outbox_by_phase == {
        "pending_due": 1,
        "pending_retry_scheduled": 1,
        "failed": 1,
    }
    assert metrics.oldest_due_notification_age_seconds == 240
    assert metrics.report_schedules_by_phase == {
        "active_due": 1,
        "active_future": 2,
        "paused": 1,
        "failed_last_delivery": 1,
    }
    assert metrics.oldest_due_report_schedule_age_seconds == 240
    assert metrics.report_schedule_deliveries_by_status == {
        "failed": 1,
        "sent": 1,
    }
    assert metrics.marketing_inquiries_by_status == {
        "new": 1,
        "contacted": 1,
        "qualified": 1,
        "closed": 1,
    }
    assert metrics.marketing_inquiry_submissions_by_outcome == {
        "accepted": 1,
        "rate_limited": 1,
    }
    assert metrics.marketing_inquiry_rejections_by_reason == {"source_rate_limit": 1}
    assert metrics.worker_heartbeats_by_scope == {"all": 1, "organization": 1}
    assert metrics.latest_all_tenant_worker_heartbeat_age_seconds == 120
    assert metrics.alert_delivery_test_events_by_status == {"succeeded": 1, "failed": 1}
    assert metrics.latest_alert_delivery_test_success_age_seconds == 180
    assert metrics.report_schedule_dispatch_events_by_status == {
        "succeeded": 1,
        "failed": 1,
    }
    assert metrics.clerk_webhook_replay_events_by_status == {"succeeded": 1, "failed": 0}
    assert metrics.billing_webhook_replay_events_by_status == {"succeeded": 1, "failed": 0}
    assert metrics.notification_provider_webhook_replay_events_by_status == {
        "succeeded": 0,
        "failed": 1,
    }
    assert metrics.artifact_store_test_events_by_status == {
        "succeeded": 1,
        "failed": 1,
    }
    assert metrics.database_test_events_by_status == {
        "succeeded": 1,
        "failed": 1,
    }
    assert metrics.tenant_isolation_test_events_by_status == {
        "succeeded": 1,
        "failed": 1,
    }
    assert metrics.run_lifecycle_test_events_by_status == {
        "succeeded": 1,
        "failed": 1,
    }
    assert metrics.worker_run_lifecycle_test_events_by_status == {
        "succeeded": 1,
        "failed": 1,
    }
    assert metrics.latest_report_schedule_dispatch_success_age_seconds == 60
    assert metrics.latest_artifact_store_test_success_age_seconds == 5
    assert metrics.latest_database_test_success_age_seconds == 2
    assert metrics.latest_tenant_isolation_test_success_age_seconds == 3
    assert metrics.latest_run_lifecycle_test_success_age_seconds == 6
    assert metrics.latest_worker_run_lifecycle_test_success_age_seconds == 7
    assert "rds_organizations_total 2" in output
    assert 'rds_runs_total{status="succeeded"} 1' in output
    assert 'rds_notifications_total{status="failed"} 1' in output
    assert 'rds_notification_provider_events_total{status="delivered"} 1' in output
    assert 'rds_notification_provider_events_total{status="bounced"} 1' in output
    assert 'rds_notification_provider_events_total{status="complained"} 0' in output
    assert 'rds_csp_violation_reports_total{disposition="report"} 1' in output
    assert 'rds_csp_violation_reports_total{disposition="enforce"} 1' in output
    assert (
        'rds_csp_violation_reports_by_directive_total{directive="script-src"} 1'
    ) in output
    assert (
        'rds_csp_violation_reports_by_directive_total{directive="connect-src"} 1'
    ) in output
    assert (
        'rds_csp_violation_reports_by_directive_total{directive="default-src"} 0'
    ) in output
    assert 'rds_notification_outbox_total{phase="pending_due"} 1' in output
    assert 'rds_notification_outbox_total{phase="pending_retry_scheduled"} 1' in output
    assert "rds_notification_oldest_due_present 1" in output
    assert "rds_notification_oldest_due_age_seconds 240" in output
    assert 'rds_report_schedules_total{phase="active_due"} 1' in output
    assert 'rds_report_schedules_total{phase="active_future"} 2' in output
    assert 'rds_report_schedules_total{phase="paused"} 1' in output
    assert 'rds_report_schedules_total{phase="failed_last_delivery"} 1' in output
    assert "rds_report_schedule_oldest_due_present 1" in output
    assert "rds_report_schedule_oldest_due_age_seconds 240" in output
    assert 'rds_report_schedule_deliveries_total{status="sent"} 1' in output
    assert 'rds_report_schedule_deliveries_total{status="failed"} 1' in output
    assert 'rds_marketing_inquiries_total{status="qualified"} 1' in output
    assert 'rds_marketing_inquiry_submissions_total{outcome="accepted"} 1' in output
    assert 'rds_marketing_inquiry_submissions_total{outcome="rate_limited"} 1' in output
    assert 'rds_marketing_inquiry_rejections_total{reason="source_rate_limit"} 1' in output
    assert 'rds_worker_heartbeats_total{scope="all"} 1' in output
    assert 'rds_worker_heartbeats_total{scope="organization"} 1' in output
    assert "rds_worker_latest_all_tenant_heartbeat_present 1" in output
    assert "rds_worker_latest_all_tenant_heartbeat_age_seconds 120" in output
    assert (
        'rds_platform_operation_events_total{event_type="alert_delivery_test",'
        'status="succeeded"} 1'
    ) in output
    assert (
        'rds_platform_operation_events_total{event_type="alert_delivery_test",'
        'status="failed"} 1'
    ) in output
    assert (
        'rds_platform_operation_events_total{event_type="report_schedule_dispatch",'
        'status="succeeded"} 1'
    ) in output
    assert (
        'rds_platform_operation_events_total{event_type="report_schedule_dispatch",'
        'status="failed"} 1'
    ) in output
    assert (
        'rds_platform_operation_events_total{event_type="clerk_webhook_replay",'
        'status="succeeded"} 1'
    ) in output
    assert (
        'rds_platform_operation_events_total{event_type="billing_webhook_replay",'
        'status="succeeded"} 1'
    ) in output
    assert (
        'rds_platform_operation_events_total{event_type="notification_provider_webhook_replay",'
        'status="failed"} 1'
    ) in output
    assert (
        'rds_platform_operation_events_total{event_type="artifact_store_test",'
        'status="succeeded"} 1'
    ) in output
    assert (
        'rds_platform_operation_events_total{event_type="artifact_store_test",'
        'status="failed"} 1'
    ) in output
    assert (
        'rds_platform_operation_events_total{event_type="database_test",'
        'status="succeeded"} 1'
    ) in output
    assert (
        'rds_platform_operation_events_total{event_type="database_test",'
        'status="failed"} 1'
    ) in output
    assert (
        'rds_platform_operation_events_total{event_type="tenant_isolation_test",'
        'status="succeeded"} 1'
    ) in output
    assert (
        'rds_platform_operation_events_total{event_type="tenant_isolation_test",'
        'status="failed"} 1'
    ) in output
    assert (
        'rds_platform_operation_events_total{event_type="run_lifecycle_smoke_test",'
        'status="succeeded"} 1'
    ) in output
    assert (
        'rds_platform_operation_events_total{event_type="run_lifecycle_smoke_test",'
        'status="failed"} 1'
    ) in output
    assert (
        'rds_platform_operation_events_total{event_type="worker_run_lifecycle_smoke_test",'
        'status="succeeded"} 1'
    ) in output
    assert (
        'rds_platform_operation_events_total{event_type="worker_run_lifecycle_smoke_test",'
        'status="failed"} 1'
    ) in output
    assert "rds_platform_alert_delivery_latest_success_present 1" in output
    assert "rds_platform_alert_delivery_latest_success_age_seconds 180" in output
    assert (
        "rds_platform_report_schedule_dispatch_latest_success_present 1"
    ) in output
    assert (
        "rds_platform_report_schedule_dispatch_latest_success_age_seconds 60"
    ) in output
    assert "rds_platform_artifact_store_test_latest_success_present 1" in output
    assert "rds_platform_artifact_store_test_latest_success_age_seconds 5" in output
    assert "rds_platform_database_test_latest_success_present 1" in output
    assert "rds_platform_database_test_latest_success_age_seconds 2" in output
    assert "rds_platform_tenant_isolation_test_latest_success_present 1" in output
    assert "rds_platform_tenant_isolation_test_latest_success_age_seconds 3" in output
    assert "rds_platform_run_lifecycle_test_latest_success_present 1" in output
    assert "rds_platform_run_lifecycle_test_latest_success_age_seconds 6" in output
    assert "rds_platform_worker_run_lifecycle_test_latest_success_present 1" in output
    assert "rds_platform_worker_run_lifecycle_test_latest_success_age_seconds 7" in output
    assert "org_1" not in output
    assert "org_2" not in output
    assert "worker_all" not in output
    assert "worker_org_1" not in output
    assert "smtp-relay" not in output
    assert "provider_evt" not in output
    assert "new@example.com" not in output
    assert "Inquiry Co" not in output
    assert "source-hash" not in output
    assert "Approval retry scheduled" not in output
    assert "approval_1" not in output
    assert "schedule_due" not in output
    assert "report_schedule_metrics" not in output
    assert "platform_operation_schedule_dispatch" not in output
    assert "scheduled_attempted" not in output
    assert "scheduler.manager@example.com" not in output
    assert "smtp unavailable" not in output
    assert "req_alert" not in output
    assert "req_clerk_replay" not in output
    assert "req_billing_replay" not in output
    assert "req_notification_replay" not in output
    assert "req_artifact_store_test" not in output
    assert "req_database_test" not in output
    assert "req_tenant_isolation_test" not in output
    assert "req_run_lifecycle_test" not in output
    assert "req_worker_run_lifecycle_test" not in output
    assert "platform_artifact_probe" not in output
    assert "artifact-store-tests" not in output
    assert "tenants/" not in output
    assert "bytes_written" not in output
    assert "write_read_verified" not in output
    assert "checks_passed" not in output
    assert "synthetic_records" not in output
    assert "platform_operation_clerk_replay" not in output
    assert "platform_operation_billing_replay" not in output
    assert "platform_operation_notification_replay" not in output
    assert "platform_operation_artifact_store_test" not in output
    assert "platform_operation_database_test" not in output
    assert "platform_operation_tenant_isolation_test" not in output
    assert "platform_operation_run_lifecycle_test" not in output
    assert "platform_operation_worker_run_lifecycle_test" not in output
    assert "http_status" not in output
    assert "tenant.example.com" not in output
    assert "csp_report" not in output


def test_verify_metrics_request_enforces_runtime_token_rules(monkeypatch):
    monkeypatch.delenv(METRICS_TOKEN_ENV, raising=False)

    verify_metrics_request(None, runtime_env="development")

    with pytest.raises(MetricsConfigurationError):
        verify_metrics_request(None, runtime_env="production")

    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    verify_metrics_request(f"Bearer {PRODUCTION_METRICS_TOKEN}", runtime_env="production")

    with pytest.raises(MetricsAuthenticationError):
        verify_metrics_request(None, runtime_env="production")

    with pytest.raises(MetricsAuthenticationError):
        verify_metrics_request("Bearer wrong-token", runtime_env="production")

    monkeypatch.setenv(METRICS_TOKEN_ENV, "short-token")
    with pytest.raises(MetricsConfigurationError):
        verify_metrics_request("Bearer short-token", runtime_env="production")


def test_metrics_route_exposes_prometheus_text_when_authorized(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(ALLOWED_ORIGINS_ENV, "https://app.example.com")
    app = create_app(
        store=seeded_metrics_store(),
        artifact_store=InMemoryTenantArtifactStore(),
    )
    client = TestClient(app)

    unauthorized = client.get("/metrics")
    response = client.get(
        "/metrics",
        headers={"Authorization": f"Bearer {PRODUCTION_METRICS_TOKEN}"},
    )

    assert unauthorized.status_code == 401
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "rds_users_active_total 1" in response.text
    assert 'rds_uploads_total{status="stored"} 1' in response.text
    assert 'rds_notification_provider_events_total{status="delivered"} 1' in response.text
    assert 'rds_notification_provider_events_total{status="bounced"} 1' in response.text
    assert 'rds_csp_violation_reports_total{disposition="report"} 1' in response.text
    assert (
        'rds_csp_violation_reports_by_directive_total{directive="script-src"} 1'
    ) in response.text
    assert 'rds_marketing_inquiries_total{status="new"} 1' in response.text
    assert 'rds_marketing_inquiry_submissions_total{outcome="rate_limited"} 1' in response.text
    assert 'rds_marketing_inquiry_rejections_total{reason="source_rate_limit"} 1' in response.text
    assert 'rds_notification_outbox_total{phase="pending_due"} 1' in response.text
    assert 'rds_notification_outbox_total{phase="pending_retry_scheduled"} 1' in response.text
    assert "rds_notification_oldest_due_present 1" in response.text
    assert 'rds_report_schedules_total{phase="active_due"} 1' in response.text
    assert "rds_report_schedule_oldest_due_present 1" in response.text
    assert 'rds_report_schedule_deliveries_total{status="failed"} 1' in response.text
    assert 'rds_worker_heartbeats_total{scope="all"} 1' in response.text
    assert "rds_worker_latest_all_tenant_heartbeat_present 1" in response.text
    assert (
        'rds_platform_operation_events_total{event_type="alert_delivery_test",'
        'status="succeeded"} 1'
    ) in response.text
    assert (
        'rds_platform_operation_events_total{event_type="report_schedule_dispatch",'
        'status="failed"} 1'
    ) in response.text
    assert (
        'rds_platform_operation_events_total{event_type="clerk_webhook_replay",'
        'status="succeeded"} 1'
    ) in response.text
    assert (
        'rds_platform_operation_events_total{event_type="notification_provider_webhook_replay",'
        'status="failed"} 1'
    ) in response.text
    assert (
        'rds_platform_operation_events_total{event_type="artifact_store_test",'
        'status="succeeded"} 1'
    ) in response.text
    assert (
        'rds_platform_operation_events_total{event_type="database_test",'
        'status="succeeded"} 1'
    ) in response.text
    assert (
        'rds_platform_operation_events_total{event_type="tenant_isolation_test",'
        'status="succeeded"} 1'
    ) in response.text
    assert (
        'rds_platform_operation_events_total{event_type="run_lifecycle_smoke_test",'
        'status="succeeded"} 1'
    ) in response.text
    assert (
        'rds_platform_operation_events_total{event_type="worker_run_lifecycle_smoke_test",'
        'status="succeeded"} 1'
    ) in response.text
    assert "rds_platform_alert_delivery_latest_success_present 1" in response.text
    assert (
        "rds_platform_report_schedule_dispatch_latest_success_present 1"
    ) in response.text
    assert (
        "rds_platform_artifact_store_test_latest_success_present 1"
    ) in response.text
    assert "rds_platform_database_test_latest_success_present 1" in response.text
    assert (
        "rds_platform_tenant_isolation_test_latest_success_present 1"
    ) in response.text
    assert (
        "rds_platform_run_lifecycle_test_latest_success_present 1"
    ) in response.text
    assert (
        "rds_platform_worker_run_lifecycle_test_latest_success_present 1"
    ) in response.text
    assert "org_1" not in response.text
    assert "worker_all" not in response.text
    assert "smtp-relay" not in response.text
    assert "provider_evt" not in response.text
    assert "source-hash" not in response.text
    assert "scheduler.manager@example.com" not in response.text
    assert "schedule_due" not in response.text
    assert "req_alert" not in response.text
    assert "req_tenant_isolation_test" not in response.text
    assert "req_run_lifecycle_test" not in response.text
    assert "req_worker_run_lifecycle_test" not in response.text
    assert "req_clerk_replay" not in response.text
    assert "req_billing_replay" not in response.text
    assert "req_notification_replay" not in response.text
    assert "req_artifact_store_test" not in response.text
    assert "req_database_test" not in response.text
    assert "platform_operation_run_lifecycle_test" not in response.text
    assert "platform_operation_worker_run_lifecycle_test" not in response.text
    assert "platform_artifact_probe" not in response.text
    assert "artifact-store-tests" not in response.text
    assert "tenants/" not in response.text
    assert "bytes_written" not in response.text
    assert "write_read_verified" not in response.text
    assert "http_status" not in response.text
    assert "tenant.example.com" not in response.text
    assert "csp_report" not in response.text


def test_metrics_route_allows_development_without_token(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "development")
    monkeypatch.delenv(METRICS_TOKEN_ENV, raising=False)
    app = create_app(
        store=seeded_metrics_store(),
        artifact_store=InMemoryTenantArtifactStore(),
    )
    client = TestClient(app)

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "rds_organizations_total 2" in response.text


def test_metrics_route_reports_missing_production_token(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.delenv(METRICS_TOKEN_ENV, raising=False)
    monkeypatch.setenv(ALLOWED_ORIGINS_ENV, "https://app.example.com")
    app = create_app(
        store=seeded_metrics_store(),
        artifact_store=InMemoryTenantArtifactStore(),
    )
    client = TestClient(app)

    response = client.get("/metrics")

    assert response.status_code == 503
    assert METRICS_TOKEN_ENV in response.json()["detail"]


def test_metrics_route_rejects_short_production_token(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv(METRICS_TOKEN_ENV, "short-token")
    monkeypatch.setenv(ALLOWED_ORIGINS_ENV, "https://app.example.com")
    app = create_app(
        store=seeded_metrics_store(),
        artifact_store=InMemoryTenantArtifactStore(),
    )
    client = TestClient(app)

    response = client.get("/metrics", headers={"Authorization": "Bearer short-token"})

    assert response.status_code == 503
    assert "at least 32 characters" in response.json()["detail"]
