import json
from datetime import datetime, timedelta, timezone
from hashlib import sha256

from fastapi.testclient import TestClient
import pytest

from backend.api import create_app
from backend.artifacts import InMemoryTenantArtifactStore
from backend.database import initialize_database
from backend.domain import Site
from backend.observability import REQUEST_ID_HEADER
from backend.operations import (
    ALERT_DELIVERY_TEST_EVENT_TYPE,
    ALERT_NOTIFICATION_DUE_AGE_SECONDS_ENV,
    ALERT_NOTIFICATION_FAILED_COUNT_ENV,
    ALERT_REPORT_SCHEDULE_DUE_AGE_SECONDS_ENV,
    ALERT_REPORT_SCHEDULE_FAILED_COUNT_ENV,
    ALERT_RULES_MAX_AGE_DAYS_ENV,
    ALERT_RULES_URL_ENV,
    ALERT_RULES_VERIFIED_AT_ENV,
    ALERT_RUNBOOK_URL_ENV,
    ALERT_WEBHOOK_TIMEOUT_SECONDS_ENV,
    ALERT_WEBHOOK_URL_ENV,
    ARTIFACT_STORE_TEST_EVENT_TYPE,
    BILLING_WEBHOOK_REPLAY_EVENT_TYPE,
    BACKUP_JOB_URL_ENV,
    BACKUP_JOB_VERIFIED_AT_ENV,
    BACKUP_MAX_JOB_AGE_DAYS_ENV,
    BACKUP_MAX_RESTORE_TEST_AGE_DAYS_ENV,
    BACKUP_POLICY_URL_ENV,
    BACKUP_RESTORE_TESTED_AT_ENV,
    CLERK_WEBHOOK_REPLAY_EVENT_TYPE,
    DATABASE_TEST_EVENT_TYPE,
    NOTIFICATION_PROVIDER_WEBHOOK_REPLAY_EVENT_TYPE,
    REPORT_SCHEDULE_DISPATCH_EVENT_TYPE,
    RUN_LIFECYCLE_TEST_EVENT_TYPE,
    TENANT_ISOLATION_TEST_EVENT_TYPE,
    WORKER_RUN_LIFECYCLE_TEST_EVENT_TYPE,
    AlertDeliveryError,
    OperationsConfigurationError,
    RunLifecycleSmokeTestError,
    TenantIsolationSmokeTestError,
    WorkerRunLifecycleSmokeTestError,
    run_lifecycle_smoke_test,
    alerting_configuration_status,
    backup_configuration_status,
    run_tenant_isolation_smoke_test,
    send_alert_delivery_test,
    worker_run_lifecycle_smoke_test,
)
from backend.store import SaaSStore
from backend.support import SUPPORT_ACCESS_TOKEN_ENV


def table_count(store: SaaSStore, table_name: str) -> int:
    return int(store.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])


def test_send_alert_delivery_test_posts_sanitized_payload(monkeypatch):
    monkeypatch.setenv(ALERT_WEBHOOK_URL_ENV, "https://alerts.example.com/rds")
    monkeypatch.setenv(ALERT_WEBHOOK_TIMEOUT_SECONDS_ENV, "2.5")
    calls = []

    def sender(webhook_url, payload, timeout_seconds):
        calls.append((webhook_url, payload, timeout_seconds))
        return 202

    result = send_alert_delivery_test(
        request_id="req_alert_1",
        runtime_env="production",
        webhook_sender=sender,
        now=datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc),
    )

    assert result.event_type == ALERT_DELIVERY_TEST_EVENT_TYPE
    assert result.status_code == 202
    assert result.delivered_at == "2026-07-20T12:00:00+00:00"
    assert len(calls) == 1
    webhook_url, payload, timeout_seconds = calls[0]
    assert webhook_url == "https://alerts.example.com/rds"
    assert timeout_seconds == 2.5
    assert payload == {
        "event_type": ALERT_DELIVERY_TEST_EVENT_TYPE,
        "service": "rds-saas-api",
        "severity": "info",
        "message": "RDS SaaS alert delivery test",
        "created_at": "2026-07-20T12:00:00+00:00",
        "request_id": "req_alert_1",
    }
    serialized_payload = json.dumps(payload)
    assert "organization" not in serialized_payload
    assert "tenant" not in serialized_payload
    assert "authorization" not in serialized_payload


def test_send_alert_delivery_test_requires_configured_webhook(monkeypatch):
    monkeypatch.delenv(ALERT_WEBHOOK_URL_ENV, raising=False)

    with pytest.raises(OperationsConfigurationError) as exc:
        send_alert_delivery_test(runtime_env="development", webhook_sender=lambda *_: 202)

    assert ALERT_WEBHOOK_URL_ENV in str(exc.value)


def test_send_alert_delivery_test_rejects_non_success_response(monkeypatch):
    monkeypatch.setenv(ALERT_WEBHOOK_URL_ENV, "https://alerts.example.com/rds")

    with pytest.raises(AlertDeliveryError) as exc:
        send_alert_delivery_test(
            runtime_env="production",
            webhook_sender=lambda *_: 500,
        )

    assert "HTTP 500" in str(exc.value)


def test_backup_configuration_status_accepts_fresh_job_and_restore_evidence(monkeypatch):
    monkeypatch.setenv(BACKUP_POLICY_URL_ENV, "https://ops.example.com/backups")
    monkeypatch.setenv(BACKUP_JOB_URL_ENV, "https://ops.example.com/backups/latest")
    monkeypatch.setenv(BACKUP_JOB_VERIFIED_AT_ENV, "2026-07-20T12:00:00+00:00")
    monkeypatch.setenv(BACKUP_RESTORE_TESTED_AT_ENV, "2026-07-19T12:00:00+00:00")
    monkeypatch.setenv(BACKUP_MAX_JOB_AGE_DAYS_ENV, "7")
    monkeypatch.setenv(BACKUP_MAX_RESTORE_TEST_AGE_DAYS_ENV, "30")

    configured, detail = backup_configuration_status(
        "production",
        now=datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc),
    )

    assert configured is True
    assert BACKUP_POLICY_URL_ENV in detail
    assert BACKUP_JOB_URL_ENV in detail
    assert "backup job verified age 1 day(s)" in detail
    assert "restore test age 2 day(s)" in detail


def test_backup_configuration_status_rejects_stale_backup_job_evidence(monkeypatch):
    monkeypatch.setenv(BACKUP_POLICY_URL_ENV, "https://ops.example.com/backups")
    monkeypatch.setenv(BACKUP_JOB_URL_ENV, "https://ops.example.com/backups/latest")
    monkeypatch.setenv(BACKUP_JOB_VERIFIED_AT_ENV, "2026-07-18T12:00:00+00:00")
    monkeypatch.setenv(BACKUP_RESTORE_TESTED_AT_ENV, "2026-07-21T12:00:00+00:00")
    monkeypatch.setenv(BACKUP_MAX_JOB_AGE_DAYS_ENV, "1")
    monkeypatch.setenv(BACKUP_MAX_RESTORE_TEST_AGE_DAYS_ENV, "30")

    configured, detail = backup_configuration_status(
        "production",
        now=datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc),
    )

    assert configured is False
    assert BACKUP_JOB_VERIFIED_AT_ENV in detail
    assert "older than 1 days" in detail


def test_alerting_configuration_status_accepts_alert_rule_evidence(monkeypatch):
    monkeypatch.setenv(ALERT_WEBHOOK_URL_ENV, "https://alerts.example.com/rds")
    monkeypatch.setenv(ALERT_RUNBOOK_URL_ENV, "https://ops.example.com/rds-runbook")
    monkeypatch.setenv(ALERT_RULES_URL_ENV, "https://monitoring.example.com/rds/alerts")
    monkeypatch.setenv(ALERT_RULES_VERIFIED_AT_ENV, datetime.now(timezone.utc).isoformat())
    monkeypatch.setenv(ALERT_RULES_MAX_AGE_DAYS_ENV, "30")
    monkeypatch.setenv(ALERT_NOTIFICATION_DUE_AGE_SECONDS_ENV, "900")
    monkeypatch.setenv(ALERT_NOTIFICATION_FAILED_COUNT_ENV, "1")
    monkeypatch.setenv(ALERT_REPORT_SCHEDULE_DUE_AGE_SECONDS_ENV, "900")
    monkeypatch.setenv(ALERT_REPORT_SCHEDULE_FAILED_COUNT_ENV, "1")

    configured, detail = alerting_configuration_status("production")

    assert configured is True
    assert ALERT_RULES_URL_ENV in detail
    assert "alert rules verified age" in detail


def test_alerting_configuration_status_rejects_stale_alert_rule_evidence(monkeypatch):
    monkeypatch.setenv(ALERT_WEBHOOK_URL_ENV, "https://alerts.example.com/rds")
    monkeypatch.setenv(ALERT_RUNBOOK_URL_ENV, "https://ops.example.com/rds-runbook")
    monkeypatch.setenv(ALERT_RULES_URL_ENV, "https://monitoring.example.com/rds/alerts")
    monkeypatch.setenv(
        ALERT_RULES_VERIFIED_AT_ENV,
        (datetime.now(timezone.utc) - timedelta(days=3)).isoformat(),
    )
    monkeypatch.setenv(ALERT_RULES_MAX_AGE_DAYS_ENV, "1")
    monkeypatch.setenv(ALERT_NOTIFICATION_DUE_AGE_SECONDS_ENV, "900")
    monkeypatch.setenv(ALERT_NOTIFICATION_FAILED_COUNT_ENV, "1")
    monkeypatch.setenv(ALERT_REPORT_SCHEDULE_DUE_AGE_SECONDS_ENV, "900")
    monkeypatch.setenv(ALERT_REPORT_SCHEDULE_FAILED_COUNT_ENV, "1")

    configured, detail = alerting_configuration_status("production")

    assert configured is False
    assert ALERT_RULES_VERIFIED_AT_ENV in detail
    assert "older than 1 days" in detail


def test_support_alert_test_route_requires_token_and_sends(monkeypatch):
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, "support-secret")
    monkeypatch.setenv(ALERT_WEBHOOK_URL_ENV, "https://alerts.example.com/rds")
    calls = []
    app = create_app(
        store=SaaSStore(initialize_database()),
        artifact_store=InMemoryTenantArtifactStore(),
    )
    app.state.alert_webhook_sender = lambda webhook_url, payload, timeout: (
        calls.append((webhook_url, payload, timeout)) or 204
    )
    client = TestClient(app)

    unauthorized = client.post("/support/alert-test")
    response = client.post(
        "/support/alert-test",
        headers={
            "Authorization": "Bearer support-secret",
            REQUEST_ID_HEADER: "req_route_alert",
        },
    )

    assert unauthorized.status_code == 401
    assert response.status_code == 200
    assert response.json()["event_id"].startswith("platform_operation_")
    assert response.json()["event_type"] == ALERT_DELIVERY_TEST_EVENT_TYPE
    assert response.json()["status_code"] == 204
    assert len(calls) == 1
    assert calls[0][0] == "https://alerts.example.com/rds"
    assert calls[0][1]["request_id"] == "req_route_alert"
    serialized_payload = json.dumps(calls[0][1])
    assert "support-secret" not in serialized_payload
    assert "organization" not in serialized_payload

    events_response = client.get(
        "/support/operation-events?event_type=rds.alert_delivery_test",
        headers={"Authorization": "Bearer support-secret"},
    )

    assert events_response.status_code == 200
    events = events_response.json()
    assert len(events) == 1
    assert events[0]["id"] == response.json()["event_id"]
    assert events[0]["event_type"] == ALERT_DELIVERY_TEST_EVENT_TYPE
    assert events[0]["status"] == "succeeded"
    assert events[0]["request_id"] == "req_route_alert"
    assert events[0]["detail"] == "http_status=204"
    serialized_event = json.dumps(events[0])
    assert "alerts.example.com" not in serialized_event
    assert "support-secret" not in serialized_event
    assert "organization" not in serialized_event


def test_support_alert_test_route_records_failed_delivery(monkeypatch):
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, "support-secret")
    monkeypatch.setenv(ALERT_WEBHOOK_URL_ENV, "https://alerts.example.com/rds")
    app = create_app(
        store=SaaSStore(initialize_database()),
        artifact_store=InMemoryTenantArtifactStore(),
    )
    app.state.alert_webhook_sender = lambda *_: 500
    client = TestClient(app)

    response = client.post(
        "/support/alert-test",
        headers={
            "Authorization": "Bearer support-secret",
            REQUEST_ID_HEADER: "req_failed_alert",
        },
    )
    events_response = client.get(
        "/support/operation-events?status=failed",
        headers={"Authorization": "Bearer support-secret"},
    )

    assert response.status_code == 502
    assert "HTTP 500" in response.json()["detail"]
    assert events_response.status_code == 200
    events = events_response.json()
    assert len(events) == 1
    assert events[0]["event_type"] == ALERT_DELIVERY_TEST_EVENT_TYPE
    assert events[0]["status"] == "failed"
    assert events[0]["request_id"] == "req_failed_alert"
    assert "HTTP 500" in events[0]["detail"]
    assert "alerts.example.com" not in json.dumps(events[0])


def test_support_artifact_store_test_route_requires_token_and_records_success(
    monkeypatch,
):
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, "support-secret")
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    app = create_app(
        store=store,
        artifact_store=artifact_store,
    )
    client = TestClient(app)

    unauthorized = client.post("/support/artifact-store-test")
    response = client.post(
        "/support/artifact-store-test",
        headers={
            "Authorization": "Bearer support-secret",
            REQUEST_ID_HEADER: "req_artifact_store_test",
        },
    )

    assert unauthorized.status_code == 401
    assert response.status_code == 200
    payload = response.json()
    assert payload["event_id"].startswith("platform_operation_")
    assert payload["event_type"] == ARTIFACT_STORE_TEST_EVENT_TYPE
    assert payload["bytes_written"] > 0
    assert artifact_store._objects == {}

    events_response = client.get(
        f"/support/operation-events?event_type={ARTIFACT_STORE_TEST_EVENT_TYPE}",
        headers={"Authorization": "Bearer support-secret"},
    )

    assert events_response.status_code == 200
    events = events_response.json()
    assert len(events) == 1
    assert events[0]["id"] == payload["event_id"]
    assert events[0]["event_type"] == ARTIFACT_STORE_TEST_EVENT_TYPE
    assert events[0]["status"] == "succeeded"
    assert events[0]["request_id"] == "req_artifact_store_test"
    assert events[0]["detail"] == f"bytes_written={payload['bytes_written']}"
    serialized_event = json.dumps(events[0])
    assert "platform_artifact_probe" not in serialized_event
    assert "artifact-store-tests" not in serialized_event
    assert "tenants/" not in serialized_event
    assert "storage_key" not in serialized_event
    assert "support-secret" not in serialized_event


def test_support_artifact_store_test_route_records_failed_probe(monkeypatch):
    class FailingArtifactStore(InMemoryTenantArtifactStore):
        def put_bytes(self, organization_id: str, storage_key: str, payload: bytes) -> None:
            raise RuntimeError("bucket policy secret should not leak")

    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, "support-secret")
    app = create_app(
        store=SaaSStore(initialize_database()),
        artifact_store=FailingArtifactStore(),
    )
    client = TestClient(app)

    response = client.post(
        "/support/artifact-store-test",
        headers={
            "Authorization": "Bearer support-secret",
            REQUEST_ID_HEADER: "req_failed_artifact_store_test",
        },
    )
    events_response = client.get(
        f"/support/operation-events?event_type={ARTIFACT_STORE_TEST_EVENT_TYPE}&status=failed",
        headers={"Authorization": "Bearer support-secret"},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == "Artifact store smoke test failed during write"
    assert "secret" not in response.json()["detail"]
    assert events_response.status_code == 200
    events = events_response.json()
    assert len(events) == 1
    assert events[0]["event_type"] == ARTIFACT_STORE_TEST_EVENT_TYPE
    assert events[0]["status"] == "failed"
    assert events[0]["request_id"] == "req_failed_artifact_store_test"
    assert events[0]["detail"] == "Artifact store smoke test failed during write"
    serialized_event = json.dumps(events[0])
    assert "bucket policy" not in serialized_event
    assert "platform_artifact_probe" not in serialized_event
    assert "tenants/" not in serialized_event


def test_support_database_test_route_requires_token_and_records_success(monkeypatch):
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, "support-secret")
    store = SaaSStore(initialize_database())
    app = create_app(
        store=store,
        artifact_store=InMemoryTenantArtifactStore(),
    )
    client = TestClient(app)

    unauthorized = client.post("/support/database-test")
    response = client.post(
        "/support/database-test",
        headers={
            "Authorization": "Bearer support-secret",
            REQUEST_ID_HEADER: "req_database_test",
        },
    )

    assert unauthorized.status_code == 401
    assert response.status_code == 200
    payload = response.json()
    assert payload["event_id"].startswith("platform_operation_")
    assert payload["event_type"] == DATABASE_TEST_EVENT_TYPE

    event = store.get_platform_operation_event(payload["event_id"])
    assert event is not None
    assert event["event_type"] == DATABASE_TEST_EVENT_TYPE
    assert event["status"] == "succeeded"
    assert event["request_id"] == "req_database_test"
    assert event["detail"] == "write_read_verified=true"

    events_response = client.get(
        f"/support/operation-events?event_type={DATABASE_TEST_EVENT_TYPE}",
        headers={"Authorization": "Bearer support-secret"},
    )

    assert events_response.status_code == 200
    events = events_response.json()
    assert len(events) == 1
    assert events[0]["id"] == payload["event_id"]
    assert events[0]["event_type"] == DATABASE_TEST_EVENT_TYPE
    assert events[0]["status"] == "succeeded"
    serialized_event = json.dumps(events[0])
    assert "organization_id" not in serialized_event
    assert "postgresql://" not in serialized_event
    assert "password" not in serialized_event
    assert "support-secret" not in serialized_event


def test_support_database_test_route_records_failed_readback(monkeypatch):
    class ReadbackFailingStore(SaaSStore):
        def get_platform_operation_event(self, event_id: str):
            return None

    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, "support-secret")
    store = ReadbackFailingStore(initialize_database())
    app = create_app(
        store=store,
        artifact_store=InMemoryTenantArtifactStore(),
    )
    client = TestClient(app)

    response = client.post(
        "/support/database-test",
        headers={
            "Authorization": "Bearer support-secret",
            REQUEST_ID_HEADER: "req_failed_database_test",
        },
    )
    events_response = client.get(
        f"/support/operation-events?event_type={DATABASE_TEST_EVENT_TYPE}&status=failed",
        headers={"Authorization": "Bearer support-secret"},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == "Database smoke test failed during write/read"
    assert events_response.status_code == 200
    events = events_response.json()
    assert len(events) == 1
    assert events[0]["event_type"] == DATABASE_TEST_EVENT_TYPE
    assert events[0]["status"] == "failed"
    assert events[0]["request_id"] == "req_failed_database_test"
    assert events[0]["detail"] == "Database smoke test failed during write/read"
    serialized_event = json.dumps(events[0])
    assert "postgresql://" not in serialized_event
    assert "password" not in serialized_event


def test_tenant_isolation_smoke_test_rolls_back_synthetic_metadata():
    store = SaaSStore(initialize_database())

    result = run_tenant_isolation_smoke_test(
        store,
        now=datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc),
    )

    assert result.event_type == TENANT_ISOLATION_TEST_EVENT_TYPE
    assert result.tested_at == "2026-07-20T12:00:00+00:00"
    assert result.checks_passed == 6
    assert result.synthetic_records == 12
    assert store.count_organizations() == 0
    assert store.count_users() == 0
    assert store.count_sites() == 0
    assert store.count_meters() == 0


def test_tenant_isolation_smoke_test_detects_cross_tenant_leakage():
    class LeakySiteStore(SaaSStore):
        def list_sites(self, organization_id: str) -> list[Site]:
            rows = self.conn.execute(
                """
                SELECT id, organization_id, name, timezone
                FROM sites
                ORDER BY name
                """
            ).fetchall()
            return [
                Site(
                    id=row["id"],
                    organization_id=row["organization_id"],
                    name=row["name"],
                    timezone=row["timezone"],
                )
                for row in rows
            ]

    store = LeakySiteStore(initialize_database())

    with pytest.raises(TenantIsolationSmokeTestError) as exc:
        run_tenant_isolation_smoke_test(store)

    assert "site listing" in str(exc.value)
    assert store.count_organizations() == 0
    assert store.count_users() == 0
    assert store.count_sites() == 0


def test_run_lifecycle_smoke_test_executes_analysis_and_cleans_up_artifacts():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()

    result = run_lifecycle_smoke_test(
        store,
        artifact_store,
        now=datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc),
    )

    assert result.event_type == RUN_LIFECYCLE_TEST_EVENT_TYPE
    assert result.tested_at == "2026-07-20T12:00:00+00:00"
    assert result.checks_passed == 6
    assert result.artifacts_verified == 2
    assert result.synthetic_records == 15
    assert store.count_organizations() == 0
    assert store.count_users() == 0
    assert store.count_sites() == 0
    assert store.count_meters() == 0
    for table_name in (
        "uploads",
        "runs",
        "run_uploads",
        "reports",
        "analysis_jobs",
        "audit_events",
        "notifications",
        "organization_subscriptions",
    ):
        assert table_count(store, table_name) == 0
    assert artifact_store._objects == {}


def test_run_lifecycle_smoke_test_records_generic_failures_and_cleans_up():
    class FailingReportArtifactStore(InMemoryTenantArtifactStore):
        def put_text(self, organization_id: str, storage_key: str, payload: str) -> None:
            if "/runs/" in storage_key:
                raise RuntimeError(f"do not leak {organization_id} {storage_key}")
            super().put_text(organization_id, storage_key, payload)

    store = SaaSStore(initialize_database())
    artifact_store = FailingReportArtifactStore()

    with pytest.raises(RunLifecycleSmokeTestError) as exc:
        run_lifecycle_smoke_test(store, artifact_store)

    assert str(exc.value) == "Run lifecycle smoke test failed during upload-to-report execution"
    assert store.count_organizations() == 0
    assert store.count_users() == 0
    assert table_count(store, "runs") == 0
    assert table_count(store, "analysis_jobs") == 0
    assert table_count(store, "reports") == 0
    assert artifact_store._objects == {}


def test_worker_run_lifecycle_smoke_test_drains_queue_and_cleans_up_artifacts():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()

    result = worker_run_lifecycle_smoke_test(
        store,
        artifact_store,
        now=datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc),
    )

    assert result.event_type == WORKER_RUN_LIFECYCLE_TEST_EVENT_TYPE
    assert result.tested_at == "2026-07-20T12:00:00+00:00"
    assert result.checks_passed == 8
    assert result.jobs_attempted == 1
    assert result.jobs_succeeded == 1
    assert result.artifacts_verified == 2
    assert result.synthetic_records == 15
    assert store.count_organizations() == 0
    assert store.count_users() == 0
    assert store.count_sites() == 0
    assert store.count_meters() == 0
    for table_name in (
        "uploads",
        "runs",
        "run_uploads",
        "reports",
        "analysis_jobs",
        "audit_events",
        "notifications",
        "organization_subscriptions",
    ):
        assert table_count(store, table_name) == 0
    assert artifact_store._objects == {}


def test_worker_run_lifecycle_smoke_test_records_generic_failures_and_cleans_up():
    class FailingReportArtifactStore(InMemoryTenantArtifactStore):
        def put_text(self, organization_id: str, storage_key: str, payload: str) -> None:
            if "/runs/" in storage_key:
                raise RuntimeError(f"do not leak {organization_id} {storage_key}")
            super().put_text(organization_id, storage_key, payload)

    store = SaaSStore(initialize_database())
    artifact_store = FailingReportArtifactStore()

    with pytest.raises(WorkerRunLifecycleSmokeTestError) as exc:
        worker_run_lifecycle_smoke_test(store, artifact_store)

    assert str(exc.value) == "Worker run lifecycle smoke test failed during queued job processing"
    assert store.count_organizations() == 0
    assert store.count_users() == 0
    assert table_count(store, "runs") == 0
    assert table_count(store, "analysis_jobs") == 0
    assert table_count(store, "reports") == 0
    assert artifact_store._objects == {}


def test_support_tenant_isolation_test_route_requires_token_and_records_success(
    monkeypatch,
):
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, "support-secret")
    store = SaaSStore(initialize_database())
    app = create_app(
        store=store,
        artifact_store=InMemoryTenantArtifactStore(),
    )
    client = TestClient(app)

    unauthorized = client.post("/support/tenant-isolation-test")
    response = client.post(
        "/support/tenant-isolation-test",
        headers={
            "Authorization": "Bearer support-secret",
            REQUEST_ID_HEADER: "req_tenant_isolation_test",
        },
    )

    assert unauthorized.status_code == 401
    assert response.status_code == 200
    payload = response.json()
    assert payload["event_id"].startswith("platform_operation_")
    assert payload["event_type"] == TENANT_ISOLATION_TEST_EVENT_TYPE
    assert payload["checks_passed"] == 6
    assert payload["synthetic_records"] == 12
    assert store.count_organizations() == 0
    assert store.count_users() == 0

    event = store.get_platform_operation_event(payload["event_id"])
    assert event is not None
    assert event["event_type"] == TENANT_ISOLATION_TEST_EVENT_TYPE
    assert event["status"] == "succeeded"
    assert event["request_id"] == "req_tenant_isolation_test"
    assert event["detail"] == "checks_passed=6 synthetic_records=12"

    events_response = client.get(
        f"/support/operation-events?event_type={TENANT_ISOLATION_TEST_EVENT_TYPE}",
        headers={"Authorization": "Bearer support-secret"},
    )

    assert events_response.status_code == 200
    events = events_response.json()
    assert len(events) == 1
    assert events[0]["id"] == payload["event_id"]
    assert events[0]["event_type"] == TENANT_ISOLATION_TEST_EVENT_TYPE
    assert events[0]["status"] == "succeeded"
    serialized_event = json.dumps(events[0])
    assert "platform_tenant_probe" not in serialized_event
    assert "example.invalid" not in serialized_event
    assert "tenants/" not in serialized_event
    assert "organization_id" not in serialized_event
    assert "user_id" not in serialized_event
    assert "support-secret" not in serialized_event


def test_support_tenant_isolation_test_route_records_failed_probe(monkeypatch):
    class LeakySiteStore(SaaSStore):
        def list_sites(self, organization_id: str) -> list[Site]:
            rows = self.conn.execute(
                """
                SELECT id, organization_id, name, timezone
                FROM sites
                ORDER BY name
                """
            ).fetchall()
            return [
                Site(
                    id=row["id"],
                    organization_id=row["organization_id"],
                    name=row["name"],
                    timezone=row["timezone"],
                )
                for row in rows
            ]

    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, "support-secret")
    store = LeakySiteStore(initialize_database())
    app = create_app(
        store=store,
        artifact_store=InMemoryTenantArtifactStore(),
    )
    client = TestClient(app)

    response = client.post(
        "/support/tenant-isolation-test",
        headers={
            "Authorization": "Bearer support-secret",
            REQUEST_ID_HEADER: "req_failed_tenant_isolation_test",
        },
    )
    events_response = client.get(
        f"/support/operation-events?event_type={TENANT_ISOLATION_TEST_EVENT_TYPE}&status=failed",
        headers={"Authorization": "Bearer support-secret"},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == "Tenant isolation smoke test failed for site listing"
    assert store.count_organizations() == 0
    assert store.count_users() == 0
    assert events_response.status_code == 200
    events = events_response.json()
    assert len(events) == 1
    assert events[0]["event_type"] == TENANT_ISOLATION_TEST_EVENT_TYPE
    assert events[0]["status"] == "failed"
    assert events[0]["request_id"] == "req_failed_tenant_isolation_test"
    assert events[0]["detail"] == "Tenant isolation smoke test failed for site listing"
    serialized_event = json.dumps(events[0])
    assert "platform_tenant_probe" not in serialized_event
    assert "example.invalid" not in serialized_event
    assert "tenants/" not in serialized_event


def test_support_run_lifecycle_test_route_requires_token_and_records_success(
    monkeypatch,
):
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, "support-secret")
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    app = create_app(
        store=store,
        artifact_store=artifact_store,
    )
    client = TestClient(app)

    unauthorized = client.post("/support/run-lifecycle-test")
    response = client.post(
        "/support/run-lifecycle-test",
        headers={
            "Authorization": "Bearer support-secret",
            REQUEST_ID_HEADER: "req_run_lifecycle_test",
        },
    )

    assert unauthorized.status_code == 401
    assert response.status_code == 200
    payload = response.json()
    assert payload["event_id"].startswith("platform_operation_")
    assert payload["event_type"] == RUN_LIFECYCLE_TEST_EVENT_TYPE
    assert payload["checks_passed"] == 6
    assert payload["artifacts_verified"] == 2
    assert payload["synthetic_records"] == 15
    assert store.count_organizations() == 0
    assert store.count_users() == 0
    assert artifact_store._objects == {}

    event = store.get_platform_operation_event(payload["event_id"])
    assert event is not None
    assert event["event_type"] == RUN_LIFECYCLE_TEST_EVENT_TYPE
    assert event["status"] == "succeeded"
    assert event["request_id"] == "req_run_lifecycle_test"
    assert event["detail"] == "checks_passed=6 artifacts_verified=2 synthetic_records=15"

    events_response = client.get(
        f"/support/operation-events?event_type={RUN_LIFECYCLE_TEST_EVENT_TYPE}",
        headers={"Authorization": "Bearer support-secret"},
    )

    assert events_response.status_code == 200
    events = events_response.json()
    assert len(events) == 1
    assert events[0]["id"] == payload["event_id"]
    assert events[0]["event_type"] == RUN_LIFECYCLE_TEST_EVENT_TYPE
    assert events[0]["status"] == "succeeded"
    serialized_event = json.dumps(events[0])
    assert "platform_run_probe" not in serialized_event
    assert "example.invalid" not in serialized_event
    assert "tenants/" not in serialized_event
    assert "storage_key" not in serialized_event
    assert "organization_id" not in serialized_event
    assert "user_id" not in serialized_event
    assert "support-secret" not in serialized_event


def test_support_run_lifecycle_test_route_records_failed_probe(monkeypatch):
    class FailingReportArtifactStore(InMemoryTenantArtifactStore):
        def put_text(self, organization_id: str, storage_key: str, payload: str) -> None:
            if "/runs/" in storage_key:
                raise RuntimeError(f"do not leak {organization_id} {storage_key}")
            super().put_text(organization_id, storage_key, payload)

    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, "support-secret")
    store = SaaSStore(initialize_database())
    artifact_store = FailingReportArtifactStore()
    app = create_app(
        store=store,
        artifact_store=artifact_store,
    )
    client = TestClient(app)

    response = client.post(
        "/support/run-lifecycle-test",
        headers={
            "Authorization": "Bearer support-secret",
            REQUEST_ID_HEADER: "req_failed_run_lifecycle_test",
        },
    )
    events_response = client.get(
        f"/support/operation-events?event_type={RUN_LIFECYCLE_TEST_EVENT_TYPE}&status=failed",
        headers={"Authorization": "Bearer support-secret"},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "Run lifecycle smoke test failed during upload-to-report execution"
    )
    assert store.count_organizations() == 0
    assert store.count_users() == 0
    assert artifact_store._objects == {}
    assert events_response.status_code == 200
    events = events_response.json()
    assert len(events) == 1
    assert events[0]["event_type"] == RUN_LIFECYCLE_TEST_EVENT_TYPE
    assert events[0]["status"] == "failed"
    assert events[0]["request_id"] == "req_failed_run_lifecycle_test"
    assert events[0]["detail"] == (
        "Run lifecycle smoke test failed during upload-to-report execution"
    )
    serialized_event = json.dumps(events[0])
    assert "platform_run_probe" not in serialized_event
    assert "example.invalid" not in serialized_event
    assert "tenants/" not in serialized_event
    assert "storage_key" not in serialized_event


def test_support_worker_run_lifecycle_test_route_requires_token_and_records_success(
    monkeypatch,
):
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, "support-secret")
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    app = create_app(
        store=store,
        artifact_store=artifact_store,
    )
    client = TestClient(app)

    unauthorized = client.post("/support/worker-run-lifecycle-test")
    response = client.post(
        "/support/worker-run-lifecycle-test",
        headers={
            "Authorization": "Bearer support-secret",
            REQUEST_ID_HEADER: "req_worker_run_lifecycle_test",
        },
    )

    assert unauthorized.status_code == 401
    assert response.status_code == 200
    payload = response.json()
    assert payload["event_id"].startswith("platform_operation_")
    assert payload["event_type"] == WORKER_RUN_LIFECYCLE_TEST_EVENT_TYPE
    assert payload["checks_passed"] == 8
    assert payload["jobs_attempted"] == 1
    assert payload["jobs_succeeded"] == 1
    assert payload["artifacts_verified"] == 2
    assert payload["synthetic_records"] == 15
    assert store.count_organizations() == 0
    assert store.count_users() == 0
    assert artifact_store._objects == {}

    event = store.get_platform_operation_event(payload["event_id"])
    assert event is not None
    assert event["event_type"] == WORKER_RUN_LIFECYCLE_TEST_EVENT_TYPE
    assert event["status"] == "succeeded"
    assert event["request_id"] == "req_worker_run_lifecycle_test"
    assert event["detail"] == (
        "checks_passed=8 jobs_attempted=1 jobs_succeeded=1 "
        "artifacts_verified=2 synthetic_records=15"
    )

    events_response = client.get(
        f"/support/operation-events?event_type={WORKER_RUN_LIFECYCLE_TEST_EVENT_TYPE}",
        headers={"Authorization": "Bearer support-secret"},
    )

    assert events_response.status_code == 200
    events = events_response.json()
    assert len(events) == 1
    assert events[0]["id"] == payload["event_id"]
    assert events[0]["event_type"] == WORKER_RUN_LIFECYCLE_TEST_EVENT_TYPE
    assert events[0]["status"] == "succeeded"
    serialized_event = json.dumps(events[0])
    assert "platform_worker_probe" not in serialized_event
    assert "example.invalid" not in serialized_event
    assert "tenants/" not in serialized_event
    assert "storage_key" not in serialized_event
    assert "organization_id" not in serialized_event
    assert "user_id" not in serialized_event
    assert "support-secret" not in serialized_event


def test_support_worker_run_lifecycle_test_route_records_failed_probe(monkeypatch):
    class FailingReportArtifactStore(InMemoryTenantArtifactStore):
        def put_text(self, organization_id: str, storage_key: str, payload: str) -> None:
            if "/runs/" in storage_key:
                raise RuntimeError(f"do not leak {organization_id} {storage_key}")
            super().put_text(organization_id, storage_key, payload)

    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, "support-secret")
    store = SaaSStore(initialize_database())
    artifact_store = FailingReportArtifactStore()
    app = create_app(
        store=store,
        artifact_store=artifact_store,
    )
    client = TestClient(app)

    response = client.post(
        "/support/worker-run-lifecycle-test",
        headers={
            "Authorization": "Bearer support-secret",
            REQUEST_ID_HEADER: "req_failed_worker_run_lifecycle_test",
        },
    )
    events_response = client.get(
        (
            "/support/operation-events?"
            f"event_type={WORKER_RUN_LIFECYCLE_TEST_EVENT_TYPE}&status=failed"
        ),
        headers={"Authorization": "Bearer support-secret"},
    )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "Worker run lifecycle smoke test failed during queued job processing"
    )
    assert store.count_organizations() == 0
    assert store.count_users() == 0
    assert artifact_store._objects == {}
    assert events_response.status_code == 200
    events = events_response.json()
    assert len(events) == 1
    assert events[0]["event_type"] == WORKER_RUN_LIFECYCLE_TEST_EVENT_TYPE
    assert events[0]["status"] == "failed"
    assert events[0]["request_id"] == "req_failed_worker_run_lifecycle_test"
    assert events[0]["detail"] == (
        "Worker run lifecycle smoke test failed during queued job processing"
    )
    serialized_event = json.dumps(events[0])
    assert "platform_worker_probe" not in serialized_event
    assert "example.invalid" not in serialized_event
    assert "tenants/" not in serialized_event
    assert "storage_key" not in serialized_event


def test_support_operation_events_route_rejects_unsupported_filters(monkeypatch):
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, "support-secret")
    app = create_app(
        store=SaaSStore(initialize_database()),
        artifact_store=InMemoryTenantArtifactStore(),
    )
    client = TestClient(app)

    unsupported_type = client.get(
        "/support/operation-events?event_type=tenant.created",
        headers={"Authorization": "Bearer support-secret"},
    )
    unsupported_status = client.get(
        "/support/operation-events?status=queued",
        headers={"Authorization": "Bearer support-secret"},
    )

    assert unsupported_type.status_code == 400
    assert "Unsupported platform operation event type" in unsupported_type.json()["detail"]
    assert unsupported_status.status_code == 400
    assert "Unsupported platform operation event status" in unsupported_status.json()["detail"]


def test_support_operation_events_route_allows_report_schedule_dispatch_filter(monkeypatch):
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, "support-secret")
    store = SaaSStore(initialize_database())
    store.create_platform_operation_event(
        event_id="platform_operation_schedule_dispatch",
        event_type=REPORT_SCHEDULE_DISPATCH_EVENT_TYPE,
        event_status="failed",
        request_id=None,
        detail=(
            "scope=all scheduled_attempted=2 scheduled_sent=1 "
            "scheduled_failed=1 scheduled_skipped=0"
        ),
        created_at="2026-07-20T12:05:00+00:00",
    )
    app = create_app(
        store=store,
        artifact_store=InMemoryTenantArtifactStore(),
    )
    client = TestClient(app)

    response = client.get(
        f"/support/operation-events?event_type={REPORT_SCHEDULE_DISPATCH_EVENT_TYPE}",
        headers={"Authorization": "Bearer support-secret"},
    )

    assert response.status_code == 200
    events = response.json()
    assert len(events) == 1
    assert events[0]["id"] == "platform_operation_schedule_dispatch"
    assert events[0]["event_type"] == REPORT_SCHEDULE_DISPATCH_EVENT_TYPE
    assert events[0]["status"] == "failed"
    assert events[0]["request_id"] is None
    serialized_event = json.dumps(events[0])
    assert "organization_id" not in serialized_event
    assert "schedule_123" not in serialized_event
    assert "report_123" not in serialized_event
    assert "manager@example.com" not in serialized_event


def test_support_operation_events_route_allows_provider_replay_filters(monkeypatch):
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, "support-secret")
    store = SaaSStore(initialize_database())
    for index, event_type in enumerate(
        (
            CLERK_WEBHOOK_REPLAY_EVENT_TYPE,
            BILLING_WEBHOOK_REPLAY_EVENT_TYPE,
            NOTIFICATION_PROVIDER_WEBHOOK_REPLAY_EVENT_TYPE,
        ),
        start=1,
    ):
        store.create_platform_operation_event(
            event_id=f"platform_operation_provider_replay_{index}",
            event_type=event_type,
            event_status="succeeded",
            request_id=f"req_provider_replay_{index}",
            detail="action=processed duplicate=false",
            created_at=f"2026-07-20T12:0{index}:00+00:00",
        )
    app = create_app(
        store=store,
        artifact_store=InMemoryTenantArtifactStore(),
    )
    client = TestClient(app)

    for event_type in (
        CLERK_WEBHOOK_REPLAY_EVENT_TYPE,
        BILLING_WEBHOOK_REPLAY_EVENT_TYPE,
        NOTIFICATION_PROVIDER_WEBHOOK_REPLAY_EVENT_TYPE,
    ):
        response = client.get(
            f"/support/operation-events?event_type={event_type}",
            headers={"Authorization": "Bearer support-secret"},
        )

        assert response.status_code == 200
        events = response.json()
        assert len(events) == 1
        assert events[0]["event_type"] == event_type
        serialized_event = json.dumps(events[0])
        assert "organization_id" not in serialized_event
        assert "user_123" not in serialized_event
        assert "evt_" not in serialized_event
        assert "owner@example.com" not in serialized_event


def test_support_operation_events_route_allows_artifact_store_test_filter(
    monkeypatch,
):
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, "support-secret")
    store = SaaSStore(initialize_database())
    store.create_platform_operation_event(
        event_id="platform_operation_artifact_store_test",
        event_type=ARTIFACT_STORE_TEST_EVENT_TYPE,
        event_status="succeeded",
        request_id="req_artifact_store_test",
        detail="bytes_written=64",
        created_at="2026-07-20T12:10:00+00:00",
    )
    app = create_app(
        store=store,
        artifact_store=InMemoryTenantArtifactStore(),
    )
    client = TestClient(app)

    response = client.get(
        f"/support/operation-events?event_type={ARTIFACT_STORE_TEST_EVENT_TYPE}",
        headers={"Authorization": "Bearer support-secret"},
    )

    assert response.status_code == 200
    events = response.json()
    assert len(events) == 1
    assert events[0]["event_type"] == ARTIFACT_STORE_TEST_EVENT_TYPE
    assert events[0]["detail"] == "bytes_written=64"
    serialized_event = json.dumps(events[0])
    assert "platform_artifact_probe" not in serialized_event
    assert "artifact-store-tests" not in serialized_event
    assert "tenants/" not in serialized_event


def test_support_operation_events_route_allows_database_test_filter(monkeypatch):
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, "support-secret")
    store = SaaSStore(initialize_database())
    store.create_platform_operation_event(
        event_id="platform_operation_database_test",
        event_type=DATABASE_TEST_EVENT_TYPE,
        event_status="succeeded",
        request_id="req_database_test",
        detail="write_read_verified=true",
        created_at="2026-07-20T12:11:00+00:00",
    )
    app = create_app(
        store=store,
        artifact_store=InMemoryTenantArtifactStore(),
    )
    client = TestClient(app)

    response = client.get(
        f"/support/operation-events?event_type={DATABASE_TEST_EVENT_TYPE}",
        headers={"Authorization": "Bearer support-secret"},
    )

    assert response.status_code == 200
    events = response.json()
    assert len(events) == 1
    assert events[0]["event_type"] == DATABASE_TEST_EVENT_TYPE
    assert events[0]["detail"] == "write_read_verified=true"
    serialized_event = json.dumps(events[0])
    assert "organization_id" not in serialized_event
    assert "postgresql://" not in serialized_event


def test_support_worker_heartbeats_route_lists_sanitized_records(monkeypatch):
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, "support-secret")
    store = SaaSStore(initialize_database())
    store.upsert_worker_heartbeat(
        worker_id="worker_org_secret",
        organization_id="org_secret",
        poll_seconds=10,
        batch_size=5,
        last_attempted=1,
        last_succeeded=1,
        last_failed=0,
        last_skipped=0,
        last_scheduled_attempted=2,
        last_scheduled_sent=1,
        last_scheduled_failed=1,
        last_scheduled_skipped=0,
        last_seen_at="2026-07-20T12:00:00+00:00",
    )
    store.upsert_worker_heartbeat(
        worker_id="worker_all_primary",
        organization_id=None,
        poll_seconds=5,
        batch_size=10,
        last_attempted=3,
        last_succeeded=2,
        last_failed=1,
        last_skipped=0,
        last_scheduled_attempted=0,
        last_scheduled_sent=0,
        last_scheduled_failed=0,
        last_scheduled_skipped=0,
        last_seen_at="2026-07-20T12:01:00+00:00",
    )
    app = create_app(
        store=store,
        artifact_store=InMemoryTenantArtifactStore(),
    )
    client = TestClient(app)

    unauthorized = client.get("/support/worker-heartbeats")
    unsupported_scope = client.get(
        "/support/worker-heartbeats?scope=tenant",
        headers={"Authorization": "Bearer support-secret"},
    )
    response = client.get(
        "/support/worker-heartbeats?scope=organization",
        headers={"Authorization": "Bearer support-secret"},
    )

    assert unauthorized.status_code == 401
    assert unsupported_scope.status_code == 400
    assert "Unsupported worker heartbeat scope" in unsupported_scope.json()["detail"]
    assert response.status_code == 200
    heartbeats = response.json()
    assert len(heartbeats) == 1
    heartbeat = heartbeats[0]
    assert heartbeat["worker_fingerprint"] == sha256(
        b"worker_org_secret"
    ).hexdigest()[:12]
    assert heartbeat["scope"] == "organization"
    assert heartbeat["poll_seconds"] == 10
    assert heartbeat["batch_size"] == 5
    assert heartbeat["last_attempted"] == 1
    assert heartbeat["last_scheduled_failed"] == 1
    assert heartbeat["last_seen_at"] == "2026-07-20T12:00:00+00:00"
    assert heartbeat["age_seconds"] is not None
    serialized_response = json.dumps(heartbeats)
    assert "worker_org_secret" not in serialized_response
    assert "worker_all_primary" not in serialized_response
    assert "org_secret" not in serialized_response
    assert "organization_id" not in serialized_response


def test_support_worker_heartbeat_summary_returns_scope_counts_without_ids(monkeypatch):
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, "support-secret")
    store = SaaSStore(initialize_database())
    store.upsert_worker_heartbeat(
        worker_id="worker_all_primary",
        organization_id=None,
        poll_seconds=5,
        batch_size=10,
        last_attempted=3,
        last_succeeded=2,
        last_failed=1,
        last_skipped=0,
        last_scheduled_attempted=0,
        last_scheduled_sent=0,
        last_scheduled_failed=0,
        last_scheduled_skipped=0,
        last_seen_at="2026-07-20T12:01:00+00:00",
    )
    store.upsert_worker_heartbeat(
        worker_id="worker_org_secret",
        organization_id="org_secret",
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
        last_seen_at="2026-07-20T12:00:00+00:00",
    )
    app = create_app(
        store=store,
        artifact_store=InMemoryTenantArtifactStore(),
    )
    client = TestClient(app)

    response = client.get(
        "/support/worker-heartbeats/summary",
        headers={"Authorization": "Bearer support-secret"},
    )

    assert response.status_code == 200
    summary = response.json()
    assert summary["total_count"] == 2
    assert summary["latest_all_tenant_last_seen_at"] == "2026-07-20T12:01:00+00:00"
    assert summary["latest_all_tenant_age_seconds"] is not None
    assert summary["by_scope"] == [
        {"scope": "all", "count": 1},
        {"scope": "organization", "count": 1},
    ]
    serialized_summary = json.dumps(summary)
    assert "worker_all_primary" not in serialized_summary
    assert "worker_org_secret" not in serialized_summary
    assert "org_secret" not in serialized_summary
    assert "organization_id" not in serialized_summary
