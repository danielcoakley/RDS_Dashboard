from datetime import datetime, timedelta, timezone

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from backend.artifacts import (
    ARTIFACT_STORE_BACKEND_ENV,
    ARTIFACT_STORE_GOVERNANCE_MAX_AGE_DAYS_ENV,
    ARTIFACT_STORE_GOVERNANCE_URL_ENV,
    ARTIFACT_STORE_GOVERNANCE_VERIFIED_AT_ENV,
    S3_BUCKET_ENV,
    InMemoryTenantArtifactStore,
    LocalTenantArtifactStore,
    S3TenantArtifactStore,
)
from backend.auth_context import (
    AUTH_FLOW_VERIFICATION_MAX_AGE_DAYS_ENV,
    AUTH_FLOW_VERIFICATION_URL_ENV,
    AUTH_FLOW_VERIFIED_AT_ENV,
    CLERK_ISSUER_ENV,
    CLERK_JWT_KEY_ENV,
)
from backend.billing import (
    BILLING_OPERATIONS_MAX_AGE_DAYS_ENV,
    BILLING_OPERATIONS_URL_ENV,
    BILLING_OPERATIONS_VERIFIED_AT_ENV,
    BILLING_PROVIDER_ENV,
    BILLING_WEBHOOK_SECRET_ENV,
    BILLING_WEBHOOK_TOLERANCE_SECONDS_ENV,
    STRIPE_BILLING_PROVIDER,
    STRIPE_PRICE_ID_ENV_PREFIX,
    STRIPE_SECRET_KEY_ENV,
)
from backend.clerk_webhooks import CLERK_WEBHOOK_SIGNING_SECRET_ENV
from backend.csp_reports import CSP_REPORT_RETENTION_DAYS_ENV
from backend.database import connect_database, initialize_database
from backend.database import DATABASE_URL_ENV
from backend.marketing_handoff import (
    MARKETING_INQUIRY_CRM_WEBHOOK_SECRET_ENV,
    MARKETING_INQUIRY_CRM_WEBHOOK_URL_ENV,
)
from backend.marketing_rate_limit import MARKETING_INQUIRY_RATE_LIMIT_SECRET_ENV
from backend.monitoring import METRICS_TOKEN_ENV
from backend.notifications import (
    EMAIL_BACKEND_ENV,
    EMAIL_DELIVERY_VERIFICATION_MAX_AGE_DAYS_ENV,
    EMAIL_DELIVERY_VERIFICATION_URL_ENV,
    EMAIL_DELIVERY_VERIFIED_AT_ENV,
    MARKETING_INQUIRY_RECIPIENT_EMAIL_ENV,
    NOTIFICATION_PROVIDER_WEBHOOK_SECRET_ENV,
    NOTIFICATION_PROVIDER_WEBHOOK_TOLERANCE_SECONDS_ENV,
    SMTP_FROM_EMAIL_ENV,
    SMTP_HOST_ENV,
)
from backend.observability import (
    LOG_AGGREGATION_MAX_AGE_DAYS_ENV,
    LOG_AGGREGATION_URL_ENV,
    LOG_AGGREGATION_VERIFIED_AT_ENV,
    LOG_RETENTION_DAYS_ENV,
)
from backend.operations import (
    ALERT_DELIVERY_TEST_EVENT_TYPE,
    ALERT_DELIVERY_TEST_MAX_AGE_DAYS_ENV,
    ALERT_NOTIFICATION_DUE_AGE_SECONDS_ENV,
    ALERT_NOTIFICATION_FAILED_COUNT_ENV,
    ALERT_REPORT_SCHEDULE_DUE_AGE_SECONDS_ENV,
    ALERT_REPORT_SCHEDULE_FAILED_COUNT_ENV,
    ALERT_RULES_MAX_AGE_DAYS_ENV,
    ALERT_RULES_URL_ENV,
    ALERT_RULES_VERIFIED_AT_ENV,
    ALERT_RUNBOOK_URL_ENV,
    ALERT_WEBHOOK_URL_ENV,
    ARTIFACT_STORE_TEST_EVENT_TYPE,
    ARTIFACT_STORE_TEST_MAX_AGE_DAYS_ENV,
    BILLING_WEBHOOK_REPLAY_EVENT_TYPE,
    BACKUP_JOB_URL_ENV,
    BACKUP_JOB_VERIFIED_AT_ENV,
    BACKUP_MAX_RESTORE_TEST_AGE_DAYS_ENV,
    BACKUP_MAX_JOB_AGE_DAYS_ENV,
    BACKUP_POLICY_URL_ENV,
    BACKUP_RESTORE_TESTED_AT_ENV,
    CLERK_WEBHOOK_REPLAY_EVENT_TYPE,
    DATABASE_TEST_EVENT_TYPE,
    DATABASE_TEST_MAX_AGE_DAYS_ENV,
    NOTIFICATION_PROVIDER_WEBHOOK_REPLAY_EVENT_TYPE,
    PROVIDER_WEBHOOK_REPLAY_MAX_AGE_DAYS_ENV,
    RUN_LIFECYCLE_TEST_EVENT_TYPE,
    RUN_LIFECYCLE_TEST_MAX_AGE_DAYS_ENV,
    TENANT_ISOLATION_TEST_EVENT_TYPE,
    TENANT_ISOLATION_TEST_MAX_AGE_DAYS_ENV,
    WORKER_RUN_LIFECYCLE_TEST_EVENT_TYPE,
    WORKER_RUN_LIFECYCLE_TEST_MAX_AGE_DAYS_ENV,
)
from backend.readiness import readiness_report
from backend.runtime_config import RUNTIME_ENV_ENV, runtime_env_from_env
from backend.security import (
    ALLOWED_ORIGINS_ENV,
    HSTS_MAX_AGE_ENV,
    SECURITY_REVIEW_MAX_AGE_DAYS_ENV,
    SECURITY_REVIEW_URL_ENV,
    SECURITY_REVIEWED_AT_ENV,
    TENANT_ISOLATION_REVIEW_MAX_AGE_DAYS_ENV,
    TENANT_ISOLATION_REVIEW_URL_ENV,
    TENANT_ISOLATION_REVIEWED_AT_ENV,
)
from backend.store import SaaSStore
from backend.support import SUPPORT_ACCESS_TOKEN_ENV, SUPPORT_OWNER_EMAIL_ENV
from backend.worker_app import (
    WORKER_ENABLED_ENV,
    WORKER_HEARTBEAT_MAX_AGE_SECONDS_ENV,
    WORKER_ORGANIZATION_ID_ENV,
    WORKER_POLL_SECONDS_ENV,
    WORKER_RUN_ONCE_ENV,
    WORKER_SUPERVISOR_MAX_AGE_DAYS_ENV,
    WORKER_SUPERVISOR_URL_ENV,
    WORKER_SUPERVISOR_VERIFIED_AT_ENV,
)

PRODUCTION_MARKETING_CRM_WEBHOOK_SECRET = "crm-webhook-secret-for-production-tests-123456"
PRODUCTION_MARKETING_RATE_LIMIT_SECRET = "rate-limit-secret-for-production-tests-123456"
PRODUCTION_BILLING_WEBHOOK_SECRET = "whsec_billing_secret_for_production_tests_123456"
PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET = "whsec_test_clerk_webhook_secret_123456"
PRODUCTION_METRICS_TOKEN = "metrics-token-for-production-tests-123456"
PRODUCTION_SUPPORT_TOKEN = "support-token-for-production-tests-123456"
PRODUCTION_SUPPORT_OWNER_EMAIL = "support.owner@example.com"
PRODUCTION_NOTIFICATION_PROVIDER_WEBHOOK_SECRET = (
    "notification-provider-webhook-secret-for-production-tests"
)


class ProductionArtifactStore:
    def put_bytes(self, organization_id: str, storage_key: str, payload: bytes) -> None:
        pass

    def get_bytes(self, organization_id: str, storage_key: str) -> bytes:
        return b""

    def exists(self, organization_id: str, storage_key: str) -> bool:
        return False

    def delete(self, organization_id: str, storage_key: str) -> bool:
        return False


def public_key_pem() -> str:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return (
        private_key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("utf-8")
    )


def configure_production_auth_flow(monkeypatch) -> None:
    monkeypatch.setenv(
        AUTH_FLOW_VERIFICATION_URL_ENV,
        "https://ops.example.com/auth/flow-verification",
    )
    monkeypatch.setenv(AUTH_FLOW_VERIFIED_AT_ENV, datetime.now(timezone.utc).isoformat())
    monkeypatch.setenv(AUTH_FLOW_VERIFICATION_MAX_AGE_DAYS_ENV, "30")


def configure_production_email(monkeypatch) -> None:
    monkeypatch.setenv(EMAIL_BACKEND_ENV, "smtp")
    monkeypatch.setenv(SMTP_HOST_ENV, "smtp.example.com")
    monkeypatch.setenv(SMTP_FROM_EMAIL_ENV, "reports@example.com")
    monkeypatch.setenv(
        EMAIL_DELIVERY_VERIFICATION_URL_ENV,
        "https://ops.example.com/email/delivery",
    )
    monkeypatch.setenv(EMAIL_DELIVERY_VERIFIED_AT_ENV, datetime.now(timezone.utc).isoformat())
    monkeypatch.setenv(EMAIL_DELIVERY_VERIFICATION_MAX_AGE_DAYS_ENV, "30")
    monkeypatch.setenv(
        NOTIFICATION_PROVIDER_WEBHOOK_SECRET_ENV,
        PRODUCTION_NOTIFICATION_PROVIDER_WEBHOOK_SECRET,
    )
    monkeypatch.setenv(NOTIFICATION_PROVIDER_WEBHOOK_TOLERANCE_SECONDS_ENV, "300")
    monkeypatch.setenv(MARKETING_INQUIRY_RECIPIENT_EMAIL_ENV, "sales@example.com")
    monkeypatch.setenv(
        MARKETING_INQUIRY_RATE_LIMIT_SECRET_ENV,
        PRODUCTION_MARKETING_RATE_LIMIT_SECRET,
    )


def configure_production_billing(monkeypatch) -> None:
    monkeypatch.setenv(BILLING_PROVIDER_ENV, STRIPE_BILLING_PROVIDER)
    monkeypatch.setenv(STRIPE_SECRET_KEY_ENV, "sk_test_rds")
    monkeypatch.setenv(f"{STRIPE_PRICE_ID_ENV_PREFIX}TEAM", "price_team")
    monkeypatch.setenv(BILLING_WEBHOOK_SECRET_ENV, PRODUCTION_BILLING_WEBHOOK_SECRET)
    monkeypatch.setenv(BILLING_WEBHOOK_TOLERANCE_SECONDS_ENV, "300")
    monkeypatch.setenv(
        BILLING_OPERATIONS_URL_ENV,
        "https://ops.example.com/billing/lifecycle",
    )
    monkeypatch.setenv(
        BILLING_OPERATIONS_VERIFIED_AT_ENV,
        datetime.now(timezone.utc).isoformat(),
    )
    monkeypatch.setenv(BILLING_OPERATIONS_MAX_AGE_DAYS_ENV, "30")


def configure_production_security(monkeypatch) -> None:
    monkeypatch.setenv(ALLOWED_ORIGINS_ENV, "https://app.example.com")
    monkeypatch.setenv(HSTS_MAX_AGE_ENV, "31536000")
    monkeypatch.setenv(SECURITY_REVIEW_URL_ENV, "https://ops.example.com/security-review")
    monkeypatch.setenv(SECURITY_REVIEWED_AT_ENV, datetime.now(timezone.utc).isoformat())
    monkeypatch.setenv(SECURITY_REVIEW_MAX_AGE_DAYS_ENV, "90")
    monkeypatch.setenv(
        TENANT_ISOLATION_REVIEW_URL_ENV,
        "https://ops.example.com/tenant-isolation-review",
    )
    monkeypatch.setenv(TENANT_ISOLATION_REVIEWED_AT_ENV, datetime.now(timezone.utc).isoformat())
    monkeypatch.setenv(TENANT_ISOLATION_REVIEW_MAX_AGE_DAYS_ENV, "90")


def configure_production_backups(monkeypatch) -> None:
    monkeypatch.setenv(BACKUP_POLICY_URL_ENV, "https://ops.example.com/backups")
    monkeypatch.setenv(BACKUP_JOB_URL_ENV, "https://ops.example.com/backups/latest")
    monkeypatch.setenv(
        BACKUP_JOB_VERIFIED_AT_ENV,
        datetime.now(timezone.utc).isoformat(),
    )
    monkeypatch.setenv(BACKUP_MAX_JOB_AGE_DAYS_ENV, "7")
    monkeypatch.setenv(
        BACKUP_RESTORE_TESTED_AT_ENV,
        datetime.now(timezone.utc).isoformat(),
    )
    monkeypatch.setenv(BACKUP_MAX_RESTORE_TEST_AGE_DAYS_ENV, "90")


def configure_production_alerting(monkeypatch) -> None:
    monkeypatch.setenv(ALERT_WEBHOOK_URL_ENV, "https://alerts.example.com/rds")
    monkeypatch.setenv(ALERT_RUNBOOK_URL_ENV, "https://ops.example.com/rds-runbook")
    monkeypatch.setenv(ALERT_RULES_URL_ENV, "https://monitoring.example.com/rds/alerts")
    monkeypatch.setenv(ALERT_RULES_VERIFIED_AT_ENV, datetime.now(timezone.utc).isoformat())
    monkeypatch.setenv(ALERT_RULES_MAX_AGE_DAYS_ENV, "30")
    monkeypatch.setenv(ALERT_NOTIFICATION_DUE_AGE_SECONDS_ENV, "900")
    monkeypatch.setenv(ALERT_NOTIFICATION_FAILED_COUNT_ENV, "1")
    monkeypatch.setenv(ALERT_REPORT_SCHEDULE_DUE_AGE_SECONDS_ENV, "900")
    monkeypatch.setenv(ALERT_REPORT_SCHEDULE_FAILED_COUNT_ENV, "1")
    monkeypatch.setenv(ALERT_DELIVERY_TEST_MAX_AGE_DAYS_ENV, "30")


def configure_production_artifact_store_governance(monkeypatch) -> None:
    monkeypatch.setenv(
        ARTIFACT_STORE_GOVERNANCE_URL_ENV,
        "https://ops.example.com/artifacts/governance",
    )
    monkeypatch.setenv(
        ARTIFACT_STORE_GOVERNANCE_VERIFIED_AT_ENV,
        datetime.now(timezone.utc).isoformat(),
    )
    monkeypatch.setenv(ARTIFACT_STORE_GOVERNANCE_MAX_AGE_DAYS_ENV, "30")


def configure_production_worker(monkeypatch) -> None:
    monkeypatch.setenv(WORKER_ENABLED_ENV, "true")
    monkeypatch.setenv(WORKER_RUN_ONCE_ENV, "false")
    monkeypatch.delenv(WORKER_ORGANIZATION_ID_ENV, raising=False)
    monkeypatch.setenv(WORKER_POLL_SECONDS_ENV, "10")
    monkeypatch.setenv(WORKER_SUPERVISOR_URL_ENV, "https://ops.example.com/workers/rds")
    monkeypatch.setenv(WORKER_SUPERVISOR_VERIFIED_AT_ENV, datetime.now(timezone.utc).isoformat())
    monkeypatch.setenv(WORKER_SUPERVISOR_MAX_AGE_DAYS_ENV, "30")


def configure_production_support(monkeypatch) -> None:
    monkeypatch.setenv(SUPPORT_OWNER_EMAIL_ENV, PRODUCTION_SUPPORT_OWNER_EMAIL)


def configure_production_operations(monkeypatch) -> None:
    configure_production_artifact_store_governance(monkeypatch)
    configure_production_backups(monkeypatch)
    configure_production_alerting(monkeypatch)
    configure_production_worker(monkeypatch)
    configure_production_support(monkeypatch)
    monkeypatch.setenv(LOG_AGGREGATION_URL_ENV, "https://logs.example.com/rds")
    monkeypatch.setenv(LOG_AGGREGATION_VERIFIED_AT_ENV, datetime.now(timezone.utc).isoformat())
    monkeypatch.setenv(LOG_AGGREGATION_MAX_AGE_DAYS_ENV, "30")
    monkeypatch.setenv(LOG_RETENTION_DAYS_ENV, "90")
    monkeypatch.setenv(PROVIDER_WEBHOOK_REPLAY_MAX_AGE_DAYS_ENV, "30")
    monkeypatch.setenv(ARTIFACT_STORE_TEST_MAX_AGE_DAYS_ENV, "30")
    monkeypatch.setenv(DATABASE_TEST_MAX_AGE_DAYS_ENV, "30")
    monkeypatch.setenv(RUN_LIFECYCLE_TEST_MAX_AGE_DAYS_ENV, "30")
    monkeypatch.setenv(WORKER_RUN_LIFECYCLE_TEST_MAX_AGE_DAYS_ENV, "30")


def record_recent_worker_heartbeat(store: SaaSStore) -> None:
    store.upsert_worker_heartbeat(
        worker_id="worker_production_1",
        organization_id=None,
        poll_seconds=10,
        batch_size=10,
        last_attempted=0,
        last_succeeded=0,
        last_failed=0,
        last_skipped=0,
        last_scheduled_attempted=0,
        last_scheduled_sent=0,
        last_scheduled_failed=0,
        last_scheduled_skipped=0,
        last_seen_at=datetime.now(timezone.utc).isoformat(),
    )


def record_recent_alert_delivery_evidence(store: SaaSStore) -> None:
    store.create_platform_operation_event(
        event_id="platform_operation_alert_test",
        event_type=ALERT_DELIVERY_TEST_EVENT_TYPE,
        event_status="succeeded",
        request_id="readiness-alert-test",
        detail="http_status=204",
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def record_recent_provider_webhook_replay_evidence(store: SaaSStore) -> None:
    for event_id, event_type, detail in (
        (
            "platform_operation_clerk_webhook_replay",
            CLERK_WEBHOOK_REPLAY_EVENT_TYPE,
            "action=membership_upserted",
        ),
        (
            "platform_operation_billing_webhook_replay",
            BILLING_WEBHOOK_REPLAY_EVENT_TYPE,
            "provider=stripe action=subscription_updated duplicate=false",
        ),
        (
            "platform_operation_notification_provider_webhook_replay",
            NOTIFICATION_PROVIDER_WEBHOOK_REPLAY_EVENT_TYPE,
            "status=delivered duplicate=false",
        ),
    ):
        store.create_platform_operation_event(
            event_id=event_id,
            event_type=event_type,
            event_status="succeeded",
            request_id=None,
            detail=detail,
            created_at=datetime.now(timezone.utc).isoformat(),
        )


def record_recent_artifact_store_test_evidence(store: SaaSStore) -> None:
    store.create_platform_operation_event(
        event_id="platform_operation_artifact_store_test",
        event_type=ARTIFACT_STORE_TEST_EVENT_TYPE,
        event_status="succeeded",
        request_id="readiness-artifact-store-test",
        detail="bytes_written=64",
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def record_recent_database_test_evidence(store: SaaSStore) -> None:
    store.create_platform_operation_event(
        event_id="platform_operation_database_test",
        event_type=DATABASE_TEST_EVENT_TYPE,
        event_status="succeeded",
        request_id="readiness-database-test",
        detail="write_read_verified=true",
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def record_recent_tenant_isolation_test_evidence(store: SaaSStore) -> None:
    store.create_platform_operation_event(
        event_id="platform_operation_tenant_isolation_test",
        event_type=TENANT_ISOLATION_TEST_EVENT_TYPE,
        event_status="succeeded",
        request_id="readiness-tenant-isolation-test",
        detail="checks_passed=6 synthetic_records=12",
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def record_recent_run_lifecycle_test_evidence(store: SaaSStore) -> None:
    store.create_platform_operation_event(
        event_id="platform_operation_run_lifecycle_test",
        event_type=RUN_LIFECYCLE_TEST_EVENT_TYPE,
        event_status="succeeded",
        request_id="readiness-run-lifecycle-test",
        detail="checks_passed=6 artifacts_verified=2 synthetic_records=15",
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def record_recent_worker_run_lifecycle_test_evidence(store: SaaSStore) -> None:
    store.create_platform_operation_event(
        event_id="platform_operation_worker_run_lifecycle_test",
        event_type=WORKER_RUN_LIFECYCLE_TEST_EVENT_TYPE,
        event_status="succeeded",
        request_id="readiness-worker-run-lifecycle-test",
        detail=(
            "checks_passed=8 jobs_attempted=1 jobs_succeeded=1 "
            "artifacts_verified=2 synthetic_records=15"
        ),
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def configure_ready_production_runtime(monkeypatch, store: SaaSStore) -> None:
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    configure_production_auth_flow(monkeypatch)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    record_recent_worker_heartbeat(store)
    record_recent_alert_delivery_evidence(store)
    record_recent_provider_webhook_replay_evidence(store)
    record_recent_artifact_store_test_evidence(store)
    record_recent_database_test_evidence(store)
    record_recent_tenant_isolation_test_evidence(store)
    record_recent_run_lifecycle_test_evidence(store)
    record_recent_worker_run_lifecycle_test_evidence(store)


def check_status(report: dict, name: str) -> str:
    for check in report["checks"]:
        if check["name"] == name:
            return check["status"]
    raise AssertionError(f"Check {name} was not present")


def check_detail(report: dict, name: str) -> str:
    for check in report["checks"]:
        if check["name"] == name:
            return check["detail"]
    raise AssertionError(f"Check {name} was not present")


def test_runtime_env_from_env_defaults_to_development(monkeypatch):
    monkeypatch.delenv(RUNTIME_ENV_ENV, raising=False)

    assert runtime_env_from_env() == "development"


def test_runtime_env_from_env_rejects_unknown_mode(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "somewhere-between")

    try:
        runtime_env_from_env()
        assert False, "Expected unsupported runtime env to fail"
    except ValueError as exc:
        assert RUNTIME_ENV_ENV in str(exc)


def test_readiness_report_passes_for_default_development_configuration(monkeypatch, tmp_path):
    monkeypatch.delenv(RUNTIME_ENV_ENV, raising=False)
    monkeypatch.delenv("RDS_AUTH_MODE", raising=False)
    monkeypatch.delenv(DATABASE_URL_ENV, raising=False)
    monkeypatch.delenv(BILLING_PROVIDER_ENV, raising=False)
    monkeypatch.delenv(BILLING_WEBHOOK_SECRET_ENV, raising=False)
    monkeypatch.delenv(BILLING_WEBHOOK_TOLERANCE_SECONDS_ENV, raising=False)
    monkeypatch.delenv(BILLING_OPERATIONS_URL_ENV, raising=False)
    monkeypatch.delenv(BILLING_OPERATIONS_VERIFIED_AT_ENV, raising=False)
    monkeypatch.delenv(BILLING_OPERATIONS_MAX_AGE_DAYS_ENV, raising=False)
    monkeypatch.delenv(STRIPE_SECRET_KEY_ENV, raising=False)
    monkeypatch.delenv(f"{STRIPE_PRICE_ID_ENV_PREFIX}TEAM", raising=False)
    monkeypatch.delenv(ALLOWED_ORIGINS_ENV, raising=False)
    monkeypatch.delenv(HSTS_MAX_AGE_ENV, raising=False)
    monkeypatch.delenv(AUTH_FLOW_VERIFICATION_URL_ENV, raising=False)
    monkeypatch.delenv(AUTH_FLOW_VERIFIED_AT_ENV, raising=False)
    monkeypatch.delenv(AUTH_FLOW_VERIFICATION_MAX_AGE_DAYS_ENV, raising=False)
    monkeypatch.delenv(SECURITY_REVIEW_URL_ENV, raising=False)
    monkeypatch.delenv(SECURITY_REVIEWED_AT_ENV, raising=False)
    monkeypatch.delenv(SECURITY_REVIEW_MAX_AGE_DAYS_ENV, raising=False)
    monkeypatch.delenv(TENANT_ISOLATION_REVIEW_URL_ENV, raising=False)
    monkeypatch.delenv(TENANT_ISOLATION_REVIEWED_AT_ENV, raising=False)
    monkeypatch.delenv(TENANT_ISOLATION_REVIEW_MAX_AGE_DAYS_ENV, raising=False)
    monkeypatch.delenv(BACKUP_POLICY_URL_ENV, raising=False)
    monkeypatch.delenv(BACKUP_JOB_URL_ENV, raising=False)
    monkeypatch.delenv(BACKUP_JOB_VERIFIED_AT_ENV, raising=False)
    monkeypatch.delenv(BACKUP_MAX_JOB_AGE_DAYS_ENV, raising=False)
    monkeypatch.delenv(BACKUP_RESTORE_TESTED_AT_ENV, raising=False)
    monkeypatch.delenv(ALERT_WEBHOOK_URL_ENV, raising=False)
    monkeypatch.delenv(ALERT_RUNBOOK_URL_ENV, raising=False)
    monkeypatch.delenv(ALERT_RULES_URL_ENV, raising=False)
    monkeypatch.delenv(ALERT_RULES_VERIFIED_AT_ENV, raising=False)
    monkeypatch.delenv(ALERT_RULES_MAX_AGE_DAYS_ENV, raising=False)
    monkeypatch.delenv(ALERT_NOTIFICATION_DUE_AGE_SECONDS_ENV, raising=False)
    monkeypatch.delenv(ALERT_NOTIFICATION_FAILED_COUNT_ENV, raising=False)
    monkeypatch.delenv(ALERT_REPORT_SCHEDULE_DUE_AGE_SECONDS_ENV, raising=False)
    monkeypatch.delenv(ALERT_REPORT_SCHEDULE_FAILED_COUNT_ENV, raising=False)
    monkeypatch.delenv(ALERT_DELIVERY_TEST_MAX_AGE_DAYS_ENV, raising=False)
    monkeypatch.delenv(PROVIDER_WEBHOOK_REPLAY_MAX_AGE_DAYS_ENV, raising=False)
    monkeypatch.delenv(ARTIFACT_STORE_TEST_MAX_AGE_DAYS_ENV, raising=False)
    monkeypatch.delenv(ARTIFACT_STORE_GOVERNANCE_URL_ENV, raising=False)
    monkeypatch.delenv(ARTIFACT_STORE_GOVERNANCE_VERIFIED_AT_ENV, raising=False)
    monkeypatch.delenv(ARTIFACT_STORE_GOVERNANCE_MAX_AGE_DAYS_ENV, raising=False)
    monkeypatch.delenv(DATABASE_TEST_MAX_AGE_DAYS_ENV, raising=False)
    monkeypatch.delenv(TENANT_ISOLATION_TEST_MAX_AGE_DAYS_ENV, raising=False)
    monkeypatch.delenv(RUN_LIFECYCLE_TEST_MAX_AGE_DAYS_ENV, raising=False)
    monkeypatch.delenv(WORKER_RUN_LIFECYCLE_TEST_MAX_AGE_DAYS_ENV, raising=False)
    monkeypatch.delenv(LOG_AGGREGATION_URL_ENV, raising=False)
    monkeypatch.delenv(LOG_AGGREGATION_VERIFIED_AT_ENV, raising=False)
    monkeypatch.delenv(LOG_AGGREGATION_MAX_AGE_DAYS_ENV, raising=False)
    monkeypatch.delenv(LOG_RETENTION_DAYS_ENV, raising=False)
    monkeypatch.delenv(SUPPORT_ACCESS_TOKEN_ENV, raising=False)
    monkeypatch.delenv(SUPPORT_OWNER_EMAIL_ENV, raising=False)
    monkeypatch.delenv(CSP_REPORT_RETENTION_DAYS_ENV, raising=False)
    monkeypatch.delenv(NOTIFICATION_PROVIDER_WEBHOOK_SECRET_ENV, raising=False)
    monkeypatch.delenv(NOTIFICATION_PROVIDER_WEBHOOK_TOLERANCE_SECONDS_ENV, raising=False)
    monkeypatch.delenv(EMAIL_DELIVERY_VERIFICATION_URL_ENV, raising=False)
    monkeypatch.delenv(EMAIL_DELIVERY_VERIFIED_AT_ENV, raising=False)
    monkeypatch.delenv(EMAIL_DELIVERY_VERIFICATION_MAX_AGE_DAYS_ENV, raising=False)
    monkeypatch.delenv(WORKER_HEARTBEAT_MAX_AGE_SECONDS_ENV, raising=False)
    monkeypatch.delenv(WORKER_SUPERVISOR_URL_ENV, raising=False)
    monkeypatch.delenv(WORKER_SUPERVISOR_VERIFIED_AT_ENV, raising=False)
    monkeypatch.delenv(WORKER_SUPERVISOR_MAX_AGE_DAYS_ENV, raising=False)
    store = SaaSStore(initialize_database())

    report = readiness_report(store=store, artifact_store=LocalTenantArtifactStore(tmp_path))

    assert report["status"] == "ready"
    assert report["runtime_env"] == "development"
    assert check_status(report, "database") == "pass"
    assert check_status(report, "database_test_evidence") == "pass"
    assert check_status(report, "tenant_isolation_test_evidence") == "pass"
    assert check_status(report, "auth") == "pass"
    assert check_status(report, "email_delivery") == "pass"
    assert check_status(report, "notification_provider_webhooks") == "pass"
    assert check_status(report, "billing") == "pass"
    assert check_status(report, "security") == "pass"
    assert check_status(report, "security_review") == "pass"
    assert check_status(report, "tenant_isolation_review") == "pass"
    assert check_status(report, "backups") == "pass"
    assert check_status(report, "run_lifecycle_test_evidence") == "pass"
    assert check_status(report, "worker_run_lifecycle_test_evidence") == "pass"
    assert check_status(report, "alerting") == "pass"
    assert check_status(report, "alert_delivery_evidence") == "pass"
    assert check_status(report, "clerk_webhook_replay_evidence") == "pass"
    assert check_status(report, "billing_webhook_replay_evidence") == "pass"
    assert check_status(report, "notification_provider_replay_evidence") == "pass"
    assert check_status(report, "tenant_isolation_test_evidence") == "pass"
    assert check_status(report, "metrics") == "pass"
    assert check_status(report, "log_aggregation") == "pass"
    assert check_status(report, "support_access") == "pass"
    assert check_status(report, "csp_report_retention") == "pass"
    assert check_status(report, "worker") == "pass"
    assert check_status(report, "worker_supervision") == "pass"
    assert check_status(report, "artifact_store_test_evidence") == "pass"
    assert check_status(report, "artifact_store") == "pass"


def test_readiness_report_fails_for_local_artifacts_in_production(monkeypatch, tmp_path):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    store = SaaSStore(initialize_database())

    report = readiness_report(store=store, artifact_store=LocalTenantArtifactStore(tmp_path))

    assert report["status"] == "not_ready"
    assert check_status(report, "artifact_store") == "fail"
    assert "local-only" in check_detail(report, "artifact_store")


def test_readiness_report_fails_for_in_memory_artifacts_in_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    store = SaaSStore(initialize_database())

    report = readiness_report(store=store, artifact_store=InMemoryTenantArtifactStore())

    assert report["status"] == "not_ready"
    assert check_status(report, "artifact_store") == "fail"


def test_readiness_report_reports_artifact_store_configuration_errors(monkeypatch):
    monkeypatch.delenv(RUNTIME_ENV_ENV, raising=False)
    monkeypatch.delenv("RDS_AUTH_MODE", raising=False)
    monkeypatch.setenv(ARTIFACT_STORE_BACKEND_ENV, "s3")
    monkeypatch.delenv(S3_BUCKET_ENV, raising=False)
    store = SaaSStore(initialize_database())

    report = readiness_report(store=store)

    assert report["status"] == "not_ready"
    assert check_status(report, "artifact_store") == "fail"
    assert "Artifact store configuration failed" in check_detail(report, "artifact_store")
    assert "bucket" in check_detail(report, "artifact_store")


def test_readiness_report_requires_artifact_store_governance_url_in_production(
    monkeypatch,
):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.delenv(ARTIFACT_STORE_GOVERNANCE_URL_ENV, raising=False)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "artifact_store") == "fail"
    assert ARTIFACT_STORE_GOVERNANCE_URL_ENV in check_detail(report, "artifact_store")


def test_readiness_report_rejects_insecure_artifact_store_governance_url(
    monkeypatch,
):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.setenv(
        ARTIFACT_STORE_GOVERNANCE_URL_ENV,
        "http://ops.example.com/artifacts/governance",
    )

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "artifact_store") == "fail"
    assert "must use https" in check_detail(report, "artifact_store")


def test_readiness_report_requires_artifact_store_governance_timestamp_in_production(
    monkeypatch,
):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.delenv(ARTIFACT_STORE_GOVERNANCE_VERIFIED_AT_ENV, raising=False)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "artifact_store") == "fail"
    assert ARTIFACT_STORE_GOVERNANCE_VERIFIED_AT_ENV in check_detail(report, "artifact_store")


def test_readiness_report_rejects_timezone_less_artifact_store_governance_timestamp(
    monkeypatch,
):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.setenv(ARTIFACT_STORE_GOVERNANCE_VERIFIED_AT_ENV, "2026-07-20T12:00:00")

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "artifact_store") == "fail"
    assert "timezone" in check_detail(report, "artifact_store")


def test_readiness_report_rejects_future_artifact_store_governance_timestamp(
    monkeypatch,
):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.setenv(
        ARTIFACT_STORE_GOVERNANCE_VERIFIED_AT_ENV,
        (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
    )

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "artifact_store") == "fail"
    assert "future" in check_detail(report, "artifact_store")


def test_readiness_report_rejects_stale_artifact_store_governance_evidence(
    monkeypatch,
):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.setenv(ARTIFACT_STORE_GOVERNANCE_MAX_AGE_DAYS_ENV, "7")
    monkeypatch.setenv(
        ARTIFACT_STORE_GOVERNANCE_VERIFIED_AT_ENV,
        (datetime.now(timezone.utc) - timedelta(days=8)).isoformat(),
    )

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "artifact_store") == "fail"
    assert "older than 7 days" in check_detail(report, "artifact_store")


def test_readiness_report_rejects_invalid_artifact_store_governance_max_age(
    monkeypatch,
):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.setenv(ARTIFACT_STORE_GOVERNANCE_MAX_AGE_DAYS_ENV, "0")

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "artifact_store") == "fail"
    assert ARTIFACT_STORE_GOVERNANCE_MAX_AGE_DAYS_ENV in check_detail(
        report,
        "artifact_store",
    )


def test_readiness_report_fails_for_dev_auth_in_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "dev")
    store = SaaSStore(initialize_database())

    report = readiness_report(store=store, artifact_store=ProductionArtifactStore())

    assert report["status"] == "not_ready"
    assert check_status(report, "auth") == "fail"
    assert "RDS_AUTH_MODE" in check_detail(report, "auth")


def test_readiness_report_fails_for_sqlite_database_in_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.delenv(DATABASE_URL_ENV, raising=False)
    store = SaaSStore(initialize_database())

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "database") == "fail"
    assert "SQLite metadata storage" in check_detail(report, "database")


def test_readiness_report_fails_for_missing_clerk_config_in_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.delenv(CLERK_ISSUER_ENV, raising=False)
    monkeypatch.delenv(CLERK_JWT_KEY_ENV, raising=False)
    configure_production_artifact_store_governance(monkeypatch)
    store = SaaSStore(initialize_database())

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "auth") == "fail"
    assert check_status(report, "artifact_store") == "pass"
    assert CLERK_ISSUER_ENV in check_detail(report, "auth")


def test_readiness_report_requires_auth_flow_verification_url_in_production(monkeypatch):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.delenv(AUTH_FLOW_VERIFICATION_URL_ENV, raising=False)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "auth") == "fail"
    assert AUTH_FLOW_VERIFICATION_URL_ENV in check_detail(report, "auth")


def test_readiness_report_rejects_insecure_auth_flow_verification_url(monkeypatch):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.setenv(
        AUTH_FLOW_VERIFICATION_URL_ENV,
        "http://ops.example.com/auth/flow-verification",
    )

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "auth") == "fail"
    assert "must use https" in check_detail(report, "auth")


def test_readiness_report_requires_auth_flow_verification_timestamp_in_production(
    monkeypatch,
):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.delenv(AUTH_FLOW_VERIFIED_AT_ENV, raising=False)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "auth") == "fail"
    assert AUTH_FLOW_VERIFIED_AT_ENV in check_detail(report, "auth")


def test_readiness_report_rejects_timezone_less_auth_flow_verification_timestamp(
    monkeypatch,
):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.setenv(AUTH_FLOW_VERIFIED_AT_ENV, "2026-07-20T12:00:00")

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "auth") == "fail"
    assert "timezone" in check_detail(report, "auth")


def test_readiness_report_rejects_future_auth_flow_verification_timestamp(monkeypatch):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.setenv(
        AUTH_FLOW_VERIFIED_AT_ENV,
        (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
    )

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "auth") == "fail"
    assert "future" in check_detail(report, "auth")


def test_readiness_report_rejects_stale_auth_flow_verification_evidence(monkeypatch):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.setenv(AUTH_FLOW_VERIFICATION_MAX_AGE_DAYS_ENV, "7")
    monkeypatch.setenv(
        AUTH_FLOW_VERIFIED_AT_ENV,
        (datetime.now(timezone.utc) - timedelta(days=8)).isoformat(),
    )

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "auth") == "fail"
    assert "older than 7 days" in check_detail(report, "auth")


def test_readiness_report_rejects_invalid_auth_flow_verification_max_age(monkeypatch):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.setenv(AUTH_FLOW_VERIFICATION_MAX_AGE_DAYS_ENV, "0")

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "auth") == "fail"
    assert AUTH_FLOW_VERIFICATION_MAX_AGE_DAYS_ENV in check_detail(report, "auth")


def test_readiness_report_accepts_clerk_jwt_config_in_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    configure_production_auth_flow(monkeypatch)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    store = SaaSStore(initialize_database())
    record_recent_worker_heartbeat(store)
    record_recent_alert_delivery_evidence(store)
    record_recent_provider_webhook_replay_evidence(store)
    record_recent_artifact_store_test_evidence(store)
    record_recent_database_test_evidence(store)
    record_recent_tenant_isolation_test_evidence(store)
    record_recent_run_lifecycle_test_evidence(store)
    record_recent_worker_run_lifecycle_test_evidence(store)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "ready"
    assert check_status(report, "auth") == "pass"
    assert check_status(report, "database_test_evidence") == "pass"
    assert check_status(report, "clerk_webhooks") == "pass"
    assert check_status(report, "email_delivery") == "pass"
    assert check_status(report, "notification_provider_webhooks") == "pass"
    assert check_status(report, "marketing_inquiries") == "pass"
    assert check_status(report, "billing") == "pass"
    assert check_status(report, "security") == "pass"
    assert check_status(report, "security_review") == "pass"
    assert check_status(report, "tenant_isolation_review") == "pass"
    assert check_status(report, "backups") == "pass"
    assert check_status(report, "run_lifecycle_test_evidence") == "pass"
    assert check_status(report, "worker_run_lifecycle_test_evidence") == "pass"
    assert check_status(report, "alerting") == "pass"
    assert check_status(report, "alert_delivery_evidence") == "pass"
    assert check_status(report, "clerk_webhook_replay_evidence") == "pass"
    assert check_status(report, "billing_webhook_replay_evidence") == "pass"
    assert check_status(report, "notification_provider_replay_evidence") == "pass"
    assert check_status(report, "metrics") == "pass"
    assert check_status(report, "log_aggregation") == "pass"
    assert check_status(report, "support_access") == "pass"
    assert check_status(report, "csp_report_retention") == "pass"
    assert check_status(report, "worker") == "pass"
    assert check_status(report, "worker_supervision") == "pass"
    assert check_status(report, "artifact_store_test_evidence") == "pass"
    assert "CLERK_JWT_KEY" in check_detail(report, "auth")


def test_readiness_report_rejects_local_billing_in_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    configure_production_auth_flow(monkeypatch)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(BILLING_PROVIDER_ENV, "local")
    configure_production_email(monkeypatch)
    configure_production_security(monkeypatch)
    store = SaaSStore(initialize_database())

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "billing") == "fail"
    assert "local-only" in check_detail(report, "billing")


def test_readiness_report_rejects_short_billing_webhook_secret_in_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    configure_production_auth_flow(monkeypatch)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    monkeypatch.setenv(BILLING_WEBHOOK_SECRET_ENV, "whsec_test_billing")
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    store = SaaSStore(initialize_database())
    record_recent_worker_heartbeat(store)
    record_recent_alert_delivery_evidence(store)
    record_recent_provider_webhook_replay_evidence(store)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "billing") == "fail"
    assert "at least 32 characters" in check_detail(report, "billing")


def test_readiness_report_rejects_invalid_billing_webhook_tolerance_in_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    monkeypatch.setenv(BILLING_WEBHOOK_TOLERANCE_SECONDS_ENV, "0")
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    store = SaaSStore(initialize_database())
    record_recent_worker_heartbeat(store)
    record_recent_alert_delivery_evidence(store)
    record_recent_provider_webhook_replay_evidence(store)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "billing") == "fail"
    assert BILLING_WEBHOOK_TOLERANCE_SECONDS_ENV in check_detail(report, "billing")


def test_readiness_report_requires_billing_operations_url_in_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    monkeypatch.delenv(BILLING_OPERATIONS_URL_ENV, raising=False)
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    store = SaaSStore(initialize_database())
    record_recent_worker_heartbeat(store)
    record_recent_alert_delivery_evidence(store)
    record_recent_provider_webhook_replay_evidence(store)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "billing") == "fail"
    assert BILLING_OPERATIONS_URL_ENV in check_detail(report, "billing")


def test_readiness_report_rejects_insecure_billing_operations_url_in_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    monkeypatch.setenv(BILLING_OPERATIONS_URL_ENV, "http://ops.example.com/billing/lifecycle")
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    store = SaaSStore(initialize_database())
    record_recent_worker_heartbeat(store)
    record_recent_alert_delivery_evidence(store)
    record_recent_provider_webhook_replay_evidence(store)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "billing") == "fail"
    assert "must use https" in check_detail(report, "billing")


def test_readiness_report_requires_billing_operations_timestamp_in_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    monkeypatch.delenv(BILLING_OPERATIONS_VERIFIED_AT_ENV, raising=False)
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    store = SaaSStore(initialize_database())
    record_recent_worker_heartbeat(store)
    record_recent_alert_delivery_evidence(store)
    record_recent_provider_webhook_replay_evidence(store)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "billing") == "fail"
    assert BILLING_OPERATIONS_VERIFIED_AT_ENV in check_detail(report, "billing")


def test_readiness_report_rejects_timezone_less_billing_operations_timestamp(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    monkeypatch.setenv(BILLING_OPERATIONS_VERIFIED_AT_ENV, "2026-07-20T12:00:00")
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    store = SaaSStore(initialize_database())
    record_recent_worker_heartbeat(store)
    record_recent_alert_delivery_evidence(store)
    record_recent_provider_webhook_replay_evidence(store)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "billing") == "fail"
    assert "timezone" in check_detail(report, "billing")


def test_readiness_report_rejects_future_billing_operations_timestamp(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    monkeypatch.setenv(
        BILLING_OPERATIONS_VERIFIED_AT_ENV,
        (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
    )
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    store = SaaSStore(initialize_database())
    record_recent_worker_heartbeat(store)
    record_recent_alert_delivery_evidence(store)
    record_recent_provider_webhook_replay_evidence(store)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "billing") == "fail"
    assert "future" in check_detail(report, "billing")


def test_readiness_report_rejects_stale_billing_operations_evidence(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    monkeypatch.setenv(BILLING_OPERATIONS_MAX_AGE_DAYS_ENV, "7")
    monkeypatch.setenv(
        BILLING_OPERATIONS_VERIFIED_AT_ENV,
        (datetime.now(timezone.utc) - timedelta(days=8)).isoformat(),
    )
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    store = SaaSStore(initialize_database())
    record_recent_worker_heartbeat(store)
    record_recent_alert_delivery_evidence(store)
    record_recent_provider_webhook_replay_evidence(store)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "billing") == "fail"
    assert "older than 7 days" in check_detail(report, "billing")


def test_readiness_report_rejects_invalid_billing_operations_max_age(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    monkeypatch.setenv(BILLING_OPERATIONS_MAX_AGE_DAYS_ENV, "0")
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    store = SaaSStore(initialize_database())
    record_recent_worker_heartbeat(store)
    record_recent_alert_delivery_evidence(store)
    record_recent_provider_webhook_replay_evidence(store)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "billing") == "fail"
    assert BILLING_OPERATIONS_MAX_AGE_DAYS_ENV in check_detail(report, "billing")


def test_readiness_report_requires_allowed_origins_in_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.delenv(ALLOWED_ORIGINS_ENV, raising=False)
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    store = SaaSStore(initialize_database())

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "security") == "fail"
    assert ALLOWED_ORIGINS_ENV in check_detail(report, "security")


def test_readiness_report_rejects_insecure_allowed_origin_in_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(ALLOWED_ORIGINS_ENV, "http://app.example.com")
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    store = SaaSStore(initialize_database())

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "security") == "fail"
    assert "https" in check_detail(report, "security")


def test_readiness_report_requires_hsts_in_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(ALLOWED_ORIGINS_ENV, "https://app.example.com")
    monkeypatch.setenv(HSTS_MAX_AGE_ENV, "0")
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    store = SaaSStore(initialize_database())

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "security") == "fail"
    assert HSTS_MAX_AGE_ENV in check_detail(report, "security")


def test_readiness_report_requires_security_review_url_in_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    monkeypatch.delenv(SECURITY_REVIEW_URL_ENV, raising=False)
    store = SaaSStore(initialize_database())
    record_recent_worker_heartbeat(store)
    record_recent_alert_delivery_evidence(store)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "security_review") == "fail"
    assert SECURITY_REVIEW_URL_ENV in check_detail(report, "security_review")


def test_readiness_report_requires_security_review_timestamp_in_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    monkeypatch.delenv(SECURITY_REVIEWED_AT_ENV, raising=False)
    store = SaaSStore(initialize_database())
    record_recent_worker_heartbeat(store)
    record_recent_alert_delivery_evidence(store)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "security_review") == "fail"
    assert SECURITY_REVIEWED_AT_ENV in check_detail(report, "security_review")


def test_readiness_report_rejects_insecure_security_review_url_in_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    monkeypatch.setenv(SECURITY_REVIEW_URL_ENV, "http://ops.example.com/security-review")
    store = SaaSStore(initialize_database())
    record_recent_worker_heartbeat(store)
    record_recent_alert_delivery_evidence(store)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "security_review") == "fail"
    assert "https" in check_detail(report, "security_review")


def test_readiness_report_rejects_stale_security_review_in_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    monkeypatch.setenv(SECURITY_REVIEW_MAX_AGE_DAYS_ENV, "30")
    monkeypatch.setenv(
        SECURITY_REVIEWED_AT_ENV,
        (datetime.now(timezone.utc) - timedelta(days=45)).isoformat(),
    )
    store = SaaSStore(initialize_database())
    record_recent_worker_heartbeat(store)
    record_recent_alert_delivery_evidence(store)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "security_review") == "fail"
    assert "older than 30 days" in check_detail(report, "security_review")


def test_readiness_report_rejects_naive_security_review_timestamp_in_production(
    monkeypatch,
):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    monkeypatch.setenv(SECURITY_REVIEWED_AT_ENV, "2026-07-20T12:00:00")
    store = SaaSStore(initialize_database())
    record_recent_worker_heartbeat(store)
    record_recent_alert_delivery_evidence(store)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "security_review") == "fail"
    assert "timezone" in check_detail(report, "security_review")


def test_readiness_report_requires_tenant_isolation_review_url_in_production(
    monkeypatch,
):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    monkeypatch.delenv(TENANT_ISOLATION_REVIEW_URL_ENV, raising=False)
    store = SaaSStore(initialize_database())
    record_recent_worker_heartbeat(store)
    record_recent_alert_delivery_evidence(store)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "tenant_isolation_review") == "fail"
    assert TENANT_ISOLATION_REVIEW_URL_ENV in check_detail(report, "tenant_isolation_review")


def test_readiness_report_requires_tenant_isolation_review_timestamp_in_production(
    monkeypatch,
):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    monkeypatch.delenv(TENANT_ISOLATION_REVIEWED_AT_ENV, raising=False)
    store = SaaSStore(initialize_database())
    record_recent_worker_heartbeat(store)
    record_recent_alert_delivery_evidence(store)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "tenant_isolation_review") == "fail"
    assert TENANT_ISOLATION_REVIEWED_AT_ENV in check_detail(
        report,
        "tenant_isolation_review",
    )


def test_readiness_report_rejects_stale_tenant_isolation_review_in_production(
    monkeypatch,
):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    monkeypatch.setenv(TENANT_ISOLATION_REVIEW_MAX_AGE_DAYS_ENV, "30")
    monkeypatch.setenv(
        TENANT_ISOLATION_REVIEWED_AT_ENV,
        (datetime.now(timezone.utc) - timedelta(days=45)).isoformat(),
    )
    store = SaaSStore(initialize_database())
    record_recent_worker_heartbeat(store)
    record_recent_alert_delivery_evidence(store)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "tenant_isolation_review") == "fail"
    assert "older than 30 days" in check_detail(report, "tenant_isolation_review")


def test_readiness_report_requires_backup_policy_in_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.delenv(BACKUP_POLICY_URL_ENV, raising=False)
    monkeypatch.delenv(BACKUP_RESTORE_TESTED_AT_ENV, raising=False)
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_alerting(monkeypatch)
    store = SaaSStore(initialize_database())

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "backups") == "fail"
    assert BACKUP_POLICY_URL_ENV in check_detail(report, "backups")


def test_readiness_report_rejects_stale_restore_test_in_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(BACKUP_POLICY_URL_ENV, "https://ops.example.com/backups")
    monkeypatch.setenv(BACKUP_JOB_URL_ENV, "https://ops.example.com/backups/latest")
    monkeypatch.setenv(BACKUP_JOB_VERIFIED_AT_ENV, datetime.now(timezone.utc).isoformat())
    monkeypatch.setenv(BACKUP_MAX_JOB_AGE_DAYS_ENV, "7")
    monkeypatch.setenv(BACKUP_MAX_RESTORE_TEST_AGE_DAYS_ENV, "30")
    monkeypatch.setenv(
        BACKUP_RESTORE_TESTED_AT_ENV,
        (datetime.now(timezone.utc) - timedelta(days=45)).isoformat(),
    )
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_alerting(monkeypatch)
    store = SaaSStore(initialize_database())

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "backups") == "fail"
    assert "older than 30 days" in check_detail(report, "backups")


def test_readiness_report_requires_backup_job_url_in_production(monkeypatch):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.delenv(BACKUP_JOB_URL_ENV, raising=False)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "backups") == "fail"
    assert BACKUP_JOB_URL_ENV in check_detail(report, "backups")


def test_readiness_report_requires_https_backup_job_url_in_production(monkeypatch):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.setenv(BACKUP_JOB_URL_ENV, "http://ops.example.com/backups/latest")

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "backups") == "fail"
    assert "https" in check_detail(report, "backups")


def test_readiness_report_requires_backup_job_timestamp_in_production(monkeypatch):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.delenv(BACKUP_JOB_VERIFIED_AT_ENV, raising=False)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "backups") == "fail"
    assert BACKUP_JOB_VERIFIED_AT_ENV in check_detail(report, "backups")


def test_readiness_report_rejects_naive_backup_job_timestamp_in_production(
    monkeypatch,
):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.setenv(BACKUP_JOB_VERIFIED_AT_ENV, "2026-01-01T00:00:00")

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "backups") == "fail"
    assert "timezone" in check_detail(report, "backups")


def test_readiness_report_rejects_stale_backup_job_evidence_in_production(
    monkeypatch,
):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.setenv(BACKUP_MAX_JOB_AGE_DAYS_ENV, "1")
    monkeypatch.setenv(
        BACKUP_JOB_VERIFIED_AT_ENV,
        (datetime.now(timezone.utc) - timedelta(days=3)).isoformat(),
    )

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "backups") == "fail"
    assert "older than 1 days" in check_detail(report, "backups")


def test_readiness_report_rejects_invalid_backup_job_max_age(monkeypatch):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.setenv(BACKUP_MAX_JOB_AGE_DAYS_ENV, "0")

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "backups") == "fail"
    assert BACKUP_MAX_JOB_AGE_DAYS_ENV in check_detail(report, "backups")
    assert "positive" in check_detail(report, "backups")


def test_readiness_report_requires_https_alerting_in_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(ALERT_WEBHOOK_URL_ENV, "http://alerts.example.com/rds")
    monkeypatch.setenv(ALERT_RUNBOOK_URL_ENV, "https://ops.example.com/rds-runbook")
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_backups(monkeypatch)
    store = SaaSStore(initialize_database())

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "alerting") == "fail"
    assert "https" in check_detail(report, "alerting")


def test_readiness_report_requires_alert_rules_url_in_production(monkeypatch):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.delenv(ALERT_RULES_URL_ENV, raising=False)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "alerting") == "fail"
    assert ALERT_RULES_URL_ENV in check_detail(report, "alerting")


def test_readiness_report_requires_https_alert_rules_url_in_production(monkeypatch):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.setenv(ALERT_RULES_URL_ENV, "http://monitoring.example.com/rds/alerts")

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "alerting") == "fail"
    assert "https" in check_detail(report, "alerting")


def test_readiness_report_requires_alert_rules_timestamp_in_production(monkeypatch):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.delenv(ALERT_RULES_VERIFIED_AT_ENV, raising=False)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "alerting") == "fail"
    assert ALERT_RULES_VERIFIED_AT_ENV in check_detail(report, "alerting")


def test_readiness_report_rejects_naive_alert_rules_timestamp_in_production(
    monkeypatch,
):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.setenv(ALERT_RULES_VERIFIED_AT_ENV, "2026-01-01T00:00:00")

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "alerting") == "fail"
    assert "timezone" in check_detail(report, "alerting")


def test_readiness_report_rejects_stale_alert_rules_evidence_in_production(
    monkeypatch,
):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.setenv(ALERT_RULES_MAX_AGE_DAYS_ENV, "7")
    monkeypatch.setenv(
        ALERT_RULES_VERIFIED_AT_ENV,
        (datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
    )

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "alerting") == "fail"
    assert "older than 7 days" in check_detail(report, "alerting")


def test_readiness_report_rejects_invalid_alert_rules_max_age(monkeypatch):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.setenv(ALERT_RULES_MAX_AGE_DAYS_ENV, "0")

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "alerting") == "fail"
    assert ALERT_RULES_MAX_AGE_DAYS_ENV in check_detail(report, "alerting")
    assert "positive" in check_detail(report, "alerting")


def test_readiness_report_requires_notification_alert_thresholds_in_production(
    monkeypatch,
):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(ALERT_WEBHOOK_URL_ENV, "https://alerts.example.com/rds")
    monkeypatch.setenv(ALERT_RUNBOOK_URL_ENV, "https://ops.example.com/rds-runbook")
    monkeypatch.setenv(ALERT_RULES_URL_ENV, "https://monitoring.example.com/rds/alerts")
    monkeypatch.setenv(ALERT_RULES_VERIFIED_AT_ENV, datetime.now(timezone.utc).isoformat())
    monkeypatch.setenv(ALERT_RULES_MAX_AGE_DAYS_ENV, "30")
    monkeypatch.delenv(ALERT_NOTIFICATION_DUE_AGE_SECONDS_ENV, raising=False)
    monkeypatch.setenv(ALERT_NOTIFICATION_FAILED_COUNT_ENV, "1")
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_backups(monkeypatch)
    store = SaaSStore(initialize_database())

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "alerting") == "fail"
    assert ALERT_NOTIFICATION_DUE_AGE_SECONDS_ENV in check_detail(report, "alerting")


def test_readiness_report_rejects_invalid_notification_alert_threshold(
    monkeypatch,
):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(ALERT_WEBHOOK_URL_ENV, "https://alerts.example.com/rds")
    monkeypatch.setenv(ALERT_RUNBOOK_URL_ENV, "https://ops.example.com/rds-runbook")
    monkeypatch.setenv(ALERT_RULES_URL_ENV, "https://monitoring.example.com/rds/alerts")
    monkeypatch.setenv(ALERT_RULES_VERIFIED_AT_ENV, datetime.now(timezone.utc).isoformat())
    monkeypatch.setenv(ALERT_RULES_MAX_AGE_DAYS_ENV, "30")
    monkeypatch.setenv(ALERT_NOTIFICATION_DUE_AGE_SECONDS_ENV, "900")
    monkeypatch.setenv(ALERT_NOTIFICATION_FAILED_COUNT_ENV, "0")
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_backups(monkeypatch)
    store = SaaSStore(initialize_database())

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "alerting") == "fail"
    assert ALERT_NOTIFICATION_FAILED_COUNT_ENV in check_detail(report, "alerting")
    assert "positive" in check_detail(report, "alerting")


def test_readiness_report_requires_report_schedule_alert_thresholds_in_production(
    monkeypatch,
):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(ALERT_WEBHOOK_URL_ENV, "https://alerts.example.com/rds")
    monkeypatch.setenv(ALERT_RUNBOOK_URL_ENV, "https://ops.example.com/rds-runbook")
    monkeypatch.setenv(ALERT_RULES_URL_ENV, "https://monitoring.example.com/rds/alerts")
    monkeypatch.setenv(ALERT_RULES_VERIFIED_AT_ENV, datetime.now(timezone.utc).isoformat())
    monkeypatch.setenv(ALERT_RULES_MAX_AGE_DAYS_ENV, "30")
    monkeypatch.setenv(ALERT_NOTIFICATION_DUE_AGE_SECONDS_ENV, "900")
    monkeypatch.setenv(ALERT_NOTIFICATION_FAILED_COUNT_ENV, "1")
    monkeypatch.delenv(ALERT_REPORT_SCHEDULE_DUE_AGE_SECONDS_ENV, raising=False)
    monkeypatch.setenv(ALERT_REPORT_SCHEDULE_FAILED_COUNT_ENV, "1")
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_backups(monkeypatch)
    store = SaaSStore(initialize_database())

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "alerting") == "fail"
    assert ALERT_REPORT_SCHEDULE_DUE_AGE_SECONDS_ENV in check_detail(report, "alerting")


def test_readiness_report_requires_alert_delivery_evidence_in_production(
    monkeypatch,
):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    store = SaaSStore(initialize_database())
    record_recent_worker_heartbeat(store)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "alert_delivery_evidence") == "fail"
    assert "No successful alert delivery smoke test" in check_detail(
        report,
        "alert_delivery_evidence",
    )


def test_readiness_report_rejects_stale_alert_delivery_evidence_in_production(
    monkeypatch,
):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    monkeypatch.setenv(ALERT_DELIVERY_TEST_MAX_AGE_DAYS_ENV, "1")
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    monkeypatch.setenv(ALERT_DELIVERY_TEST_MAX_AGE_DAYS_ENV, "1")
    store = SaaSStore(initialize_database())
    record_recent_worker_heartbeat(store)
    store.create_platform_operation_event(
        event_id="platform_operation_stale_alert_test",
        event_type=ALERT_DELIVERY_TEST_EVENT_TYPE,
        event_status="succeeded",
        request_id="stale-alert-test",
        detail="http_status=204",
        created_at=(datetime.now(timezone.utc) - timedelta(days=3)).isoformat(),
    )

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "alert_delivery_evidence") == "fail"
    assert "older than 1 days" in check_detail(report, "alert_delivery_evidence")


def test_readiness_report_rejects_invalid_alert_delivery_evidence_max_age(
    monkeypatch,
):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    monkeypatch.setenv(ALERT_DELIVERY_TEST_MAX_AGE_DAYS_ENV, "0")
    store = SaaSStore(initialize_database())
    record_recent_worker_heartbeat(store)
    record_recent_alert_delivery_evidence(store)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "alert_delivery_evidence") == "fail"
    assert ALERT_DELIVERY_TEST_MAX_AGE_DAYS_ENV in check_detail(
        report,
        "alert_delivery_evidence",
    )
    assert "positive" in check_detail(report, "alert_delivery_evidence")


def test_readiness_report_requires_database_test_evidence_in_production(
    monkeypatch,
):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    store = SaaSStore(initialize_database())
    record_recent_worker_heartbeat(store)
    record_recent_alert_delivery_evidence(store)
    record_recent_provider_webhook_replay_evidence(store)
    record_recent_artifact_store_test_evidence(store)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "database_test_evidence") == "fail"
    assert "No successful database smoke test" in check_detail(
        report,
        "database_test_evidence",
    )


def test_readiness_report_rejects_stale_database_test_evidence(
    monkeypatch,
):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    monkeypatch.setenv(DATABASE_TEST_MAX_AGE_DAYS_ENV, "7")
    store = SaaSStore(initialize_database())
    record_recent_worker_heartbeat(store)
    record_recent_alert_delivery_evidence(store)
    record_recent_provider_webhook_replay_evidence(store)
    record_recent_artifact_store_test_evidence(store)
    store.create_platform_operation_event(
        event_id="platform_operation_stale_database_test",
        event_type=DATABASE_TEST_EVENT_TYPE,
        event_status="succeeded",
        request_id=None,
        detail="write_read_verified=true",
        created_at=(datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
    )

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "database_test_evidence") == "fail"
    assert "older than 7 days" in check_detail(report, "database_test_evidence")


def test_readiness_report_rejects_invalid_database_test_max_age(
    monkeypatch,
):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    monkeypatch.setenv(DATABASE_TEST_MAX_AGE_DAYS_ENV, "0")
    store = SaaSStore(initialize_database())
    record_recent_worker_heartbeat(store)
    record_recent_alert_delivery_evidence(store)
    record_recent_provider_webhook_replay_evidence(store)
    record_recent_artifact_store_test_evidence(store)
    record_recent_database_test_evidence(store)
    record_recent_tenant_isolation_test_evidence(store)
    record_recent_run_lifecycle_test_evidence(store)
    record_recent_worker_run_lifecycle_test_evidence(store)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "database_test_evidence") == "fail"
    assert DATABASE_TEST_MAX_AGE_DAYS_ENV in check_detail(
        report,
        "database_test_evidence",
    )
    assert "positive" in check_detail(report, "database_test_evidence")


def test_readiness_report_requires_tenant_isolation_test_evidence_in_production(
    monkeypatch,
):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    store.conn.execute(
        "DELETE FROM platform_operation_events WHERE event_type = ?",
        (TENANT_ISOLATION_TEST_EVENT_TYPE,),
    )

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "tenant_isolation_test_evidence") == "fail"
    assert "No successful tenant isolation smoke test" in check_detail(
        report,
        "tenant_isolation_test_evidence",
    )


def test_readiness_report_rejects_stale_tenant_isolation_test_evidence(
    monkeypatch,
):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.setenv(TENANT_ISOLATION_TEST_MAX_AGE_DAYS_ENV, "7")
    store.conn.execute(
        """
        UPDATE platform_operation_events
        SET created_at = ?
        WHERE event_type = ?
        """,
        (
            (datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
            TENANT_ISOLATION_TEST_EVENT_TYPE,
        ),
    )

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "tenant_isolation_test_evidence") == "fail"
    assert "older than 7 days" in check_detail(
        report,
        "tenant_isolation_test_evidence",
    )


def test_readiness_report_rejects_invalid_tenant_isolation_test_max_age(
    monkeypatch,
):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.setenv(TENANT_ISOLATION_TEST_MAX_AGE_DAYS_ENV, "0")

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "tenant_isolation_test_evidence") == "fail"
    assert TENANT_ISOLATION_TEST_MAX_AGE_DAYS_ENV in check_detail(
        report,
        "tenant_isolation_test_evidence",
    )
    assert "positive" in check_detail(report, "tenant_isolation_test_evidence")


def test_readiness_report_requires_run_lifecycle_test_evidence_in_production(
    monkeypatch,
):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    store.conn.execute(
        "DELETE FROM platform_operation_events WHERE event_type = ?",
        (RUN_LIFECYCLE_TEST_EVENT_TYPE,),
    )

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "run_lifecycle_test_evidence") == "fail"
    assert "No successful run lifecycle smoke test" in check_detail(
        report,
        "run_lifecycle_test_evidence",
    )


def test_readiness_report_rejects_stale_run_lifecycle_test_evidence(
    monkeypatch,
):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.setenv(RUN_LIFECYCLE_TEST_MAX_AGE_DAYS_ENV, "7")
    store.conn.execute(
        """
        UPDATE platform_operation_events
        SET created_at = ?
        WHERE event_type = ?
        """,
        (
            (datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
            RUN_LIFECYCLE_TEST_EVENT_TYPE,
        ),
    )

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "run_lifecycle_test_evidence") == "fail"
    assert "older than 7 days" in check_detail(
        report,
        "run_lifecycle_test_evidence",
    )


def test_readiness_report_rejects_invalid_run_lifecycle_test_max_age(
    monkeypatch,
):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.setenv(RUN_LIFECYCLE_TEST_MAX_AGE_DAYS_ENV, "0")

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "run_lifecycle_test_evidence") == "fail"
    assert RUN_LIFECYCLE_TEST_MAX_AGE_DAYS_ENV in check_detail(
        report,
        "run_lifecycle_test_evidence",
    )
    assert "positive" in check_detail(report, "run_lifecycle_test_evidence")


def test_readiness_report_requires_worker_run_lifecycle_test_evidence_in_production(
    monkeypatch,
):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    store.conn.execute(
        "DELETE FROM platform_operation_events WHERE event_type = ?",
        (WORKER_RUN_LIFECYCLE_TEST_EVENT_TYPE,),
    )

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "worker_run_lifecycle_test_evidence") == "fail"
    assert "No successful worker run lifecycle smoke test" in check_detail(
        report,
        "worker_run_lifecycle_test_evidence",
    )


def test_readiness_report_rejects_stale_worker_run_lifecycle_test_evidence(
    monkeypatch,
):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.setenv(WORKER_RUN_LIFECYCLE_TEST_MAX_AGE_DAYS_ENV, "7")
    store.conn.execute(
        """
        UPDATE platform_operation_events
        SET created_at = ?
        WHERE event_type = ?
        """,
        (
            (datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
            WORKER_RUN_LIFECYCLE_TEST_EVENT_TYPE,
        ),
    )

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "worker_run_lifecycle_test_evidence") == "fail"
    assert "older than 7 days" in check_detail(
        report,
        "worker_run_lifecycle_test_evidence",
    )


def test_readiness_report_rejects_invalid_worker_run_lifecycle_test_max_age(
    monkeypatch,
):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.setenv(WORKER_RUN_LIFECYCLE_TEST_MAX_AGE_DAYS_ENV, "0")

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "worker_run_lifecycle_test_evidence") == "fail"
    assert WORKER_RUN_LIFECYCLE_TEST_MAX_AGE_DAYS_ENV in check_detail(
        report,
        "worker_run_lifecycle_test_evidence",
    )
    assert "positive" in check_detail(report, "worker_run_lifecycle_test_evidence")


def test_readiness_report_requires_artifact_store_test_evidence_in_production(
    monkeypatch,
):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    store = SaaSStore(initialize_database())
    record_recent_worker_heartbeat(store)
    record_recent_alert_delivery_evidence(store)
    record_recent_provider_webhook_replay_evidence(store)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "artifact_store_test_evidence") == "fail"
    assert "No successful artifact store smoke test" in check_detail(
        report,
        "artifact_store_test_evidence",
    )


def test_readiness_report_rejects_stale_artifact_store_test_evidence(
    monkeypatch,
):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    monkeypatch.setenv(ARTIFACT_STORE_TEST_MAX_AGE_DAYS_ENV, "7")
    store = SaaSStore(initialize_database())
    record_recent_worker_heartbeat(store)
    record_recent_alert_delivery_evidence(store)
    record_recent_provider_webhook_replay_evidence(store)
    store.create_platform_operation_event(
        event_id="platform_operation_stale_artifact_store_test",
        event_type=ARTIFACT_STORE_TEST_EVENT_TYPE,
        event_status="succeeded",
        request_id=None,
        detail="bytes_written=64",
        created_at=(datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
    )

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "artifact_store_test_evidence") == "fail"
    assert "older than 7 days" in check_detail(report, "artifact_store_test_evidence")


def test_readiness_report_rejects_invalid_artifact_store_test_max_age(
    monkeypatch,
):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    monkeypatch.setenv(ARTIFACT_STORE_TEST_MAX_AGE_DAYS_ENV, "0")
    store = SaaSStore(initialize_database())
    record_recent_worker_heartbeat(store)
    record_recent_alert_delivery_evidence(store)
    record_recent_provider_webhook_replay_evidence(store)
    record_recent_artifact_store_test_evidence(store)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "artifact_store_test_evidence") == "fail"
    assert ARTIFACT_STORE_TEST_MAX_AGE_DAYS_ENV in check_detail(
        report,
        "artifact_store_test_evidence",
    )
    assert "positive" in check_detail(report, "artifact_store_test_evidence")


def test_readiness_report_requires_provider_webhook_replay_evidence_in_production(
    monkeypatch,
):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    store = SaaSStore(initialize_database())
    record_recent_worker_heartbeat(store)
    record_recent_alert_delivery_evidence(store)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "clerk_webhook_replay_evidence") == "fail"
    assert check_status(report, "billing_webhook_replay_evidence") == "fail"
    assert check_status(report, "notification_provider_replay_evidence") == "fail"
    assert "No successful Clerk webhook replay evidence" in check_detail(
        report,
        "clerk_webhook_replay_evidence",
    )


def test_readiness_report_rejects_stale_provider_webhook_replay_evidence(
    monkeypatch,
):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    monkeypatch.setenv(PROVIDER_WEBHOOK_REPLAY_MAX_AGE_DAYS_ENV, "7")
    store = SaaSStore(initialize_database())
    record_recent_worker_heartbeat(store)
    record_recent_alert_delivery_evidence(store)
    store.create_platform_operation_event(
        event_id="platform_operation_clerk_webhook_replay",
        event_type=CLERK_WEBHOOK_REPLAY_EVENT_TYPE,
        event_status="succeeded",
        request_id=None,
        detail="action=membership_upserted",
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    store.create_platform_operation_event(
        event_id="platform_operation_billing_webhook_replay_stale",
        event_type=BILLING_WEBHOOK_REPLAY_EVENT_TYPE,
        event_status="succeeded",
        request_id=None,
        detail="provider=stripe action=subscription_updated duplicate=false",
        created_at=(datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
    )
    store.create_platform_operation_event(
        event_id="platform_operation_notification_provider_webhook_replay",
        event_type=NOTIFICATION_PROVIDER_WEBHOOK_REPLAY_EVENT_TYPE,
        event_status="succeeded",
        request_id=None,
        detail="status=delivered duplicate=false",
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "clerk_webhook_replay_evidence") == "pass"
    assert check_status(report, "billing_webhook_replay_evidence") == "fail"
    assert check_status(report, "notification_provider_replay_evidence") == "pass"
    assert "older than 7 days" in check_detail(report, "billing_webhook_replay_evidence")


def test_readiness_report_rejects_invalid_provider_webhook_replay_max_age(
    monkeypatch,
):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    monkeypatch.setenv(PROVIDER_WEBHOOK_REPLAY_MAX_AGE_DAYS_ENV, "0")
    store = SaaSStore(initialize_database())
    record_recent_worker_heartbeat(store)
    record_recent_alert_delivery_evidence(store)
    record_recent_provider_webhook_replay_evidence(store)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "clerk_webhook_replay_evidence") == "fail"
    assert PROVIDER_WEBHOOK_REPLAY_MAX_AGE_DAYS_ENV in check_detail(
        report,
        "clerk_webhook_replay_evidence",
    )
    assert "positive" in check_detail(report, "clerk_webhook_replay_evidence")


def test_readiness_report_requires_smtp_email_delivery_in_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(EMAIL_BACKEND_ENV, "memory")
    store = SaaSStore(initialize_database())

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "email_delivery") == "fail"
    assert EMAIL_BACKEND_ENV in check_detail(report, "email_delivery")


def test_readiness_report_requires_email_delivery_verification_url_in_production(
    monkeypatch,
):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.delenv(EMAIL_DELIVERY_VERIFICATION_URL_ENV, raising=False)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "email_delivery") == "fail"
    assert EMAIL_DELIVERY_VERIFICATION_URL_ENV in check_detail(report, "email_delivery")


def test_readiness_report_rejects_insecure_email_delivery_verification_url(
    monkeypatch,
):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.setenv(
        EMAIL_DELIVERY_VERIFICATION_URL_ENV,
        "http://ops.example.com/email/delivery",
    )

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "email_delivery") == "fail"
    assert "must use https" in check_detail(report, "email_delivery")


def test_readiness_report_requires_email_delivery_verification_timestamp_in_production(
    monkeypatch,
):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.delenv(EMAIL_DELIVERY_VERIFIED_AT_ENV, raising=False)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "email_delivery") == "fail"
    assert EMAIL_DELIVERY_VERIFIED_AT_ENV in check_detail(report, "email_delivery")


def test_readiness_report_rejects_timezone_less_email_delivery_verification_timestamp(
    monkeypatch,
):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.setenv(EMAIL_DELIVERY_VERIFIED_AT_ENV, "2026-07-20T12:00:00")

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "email_delivery") == "fail"
    assert "timezone" in check_detail(report, "email_delivery")


def test_readiness_report_rejects_future_email_delivery_verification_timestamp(
    monkeypatch,
):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.setenv(
        EMAIL_DELIVERY_VERIFIED_AT_ENV,
        (datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
    )

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "email_delivery") == "fail"
    assert "future" in check_detail(report, "email_delivery")


def test_readiness_report_rejects_stale_email_delivery_verification(
    monkeypatch,
):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.setenv(EMAIL_DELIVERY_VERIFICATION_MAX_AGE_DAYS_ENV, "7")
    monkeypatch.setenv(
        EMAIL_DELIVERY_VERIFIED_AT_ENV,
        (datetime.now(timezone.utc) - timedelta(days=8)).isoformat(),
    )

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "email_delivery") == "fail"
    assert "older than 7 days" in check_detail(report, "email_delivery")


def test_readiness_report_rejects_invalid_email_delivery_verification_max_age(
    monkeypatch,
):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.setenv(EMAIL_DELIVERY_VERIFICATION_MAX_AGE_DAYS_ENV, "0")

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "email_delivery") == "fail"
    assert EMAIL_DELIVERY_VERIFICATION_MAX_AGE_DAYS_ENV in check_detail(
        report,
        "email_delivery",
    )


def test_readiness_report_requires_notification_provider_webhook_secret_in_production(
    monkeypatch,
):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(EMAIL_BACKEND_ENV, "smtp")
    monkeypatch.setenv(SMTP_HOST_ENV, "smtp.example.com")
    monkeypatch.setenv(SMTP_FROM_EMAIL_ENV, "reports@example.com")
    monkeypatch.delenv(NOTIFICATION_PROVIDER_WEBHOOK_SECRET_ENV, raising=False)
    store = SaaSStore(initialize_database())

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "notification_provider_webhooks") == "fail"
    assert NOTIFICATION_PROVIDER_WEBHOOK_SECRET_ENV in check_detail(
        report,
        "notification_provider_webhooks",
    )


def test_readiness_report_rejects_short_notification_provider_webhook_secret_in_production(
    monkeypatch,
):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(EMAIL_BACKEND_ENV, "smtp")
    monkeypatch.setenv(SMTP_HOST_ENV, "smtp.example.com")
    monkeypatch.setenv(SMTP_FROM_EMAIL_ENV, "reports@example.com")
    monkeypatch.setenv(NOTIFICATION_PROVIDER_WEBHOOK_SECRET_ENV, "short")
    store = SaaSStore(initialize_database())

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "notification_provider_webhooks") == "fail"
    assert "at least 32 characters" in check_detail(
        report,
        "notification_provider_webhooks",
    )


def test_readiness_report_rejects_invalid_notification_provider_webhook_tolerance_in_production(
    monkeypatch,
):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(EMAIL_BACKEND_ENV, "smtp")
    monkeypatch.setenv(SMTP_HOST_ENV, "smtp.example.com")
    monkeypatch.setenv(SMTP_FROM_EMAIL_ENV, "reports@example.com")
    monkeypatch.setenv(
        NOTIFICATION_PROVIDER_WEBHOOK_SECRET_ENV,
        PRODUCTION_NOTIFICATION_PROVIDER_WEBHOOK_SECRET,
    )
    monkeypatch.setenv(NOTIFICATION_PROVIDER_WEBHOOK_TOLERANCE_SECONDS_ENV, "0")
    store = SaaSStore(initialize_database())

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "notification_provider_webhooks") == "fail"
    assert NOTIFICATION_PROVIDER_WEBHOOK_TOLERANCE_SECONDS_ENV in check_detail(
        report,
        "notification_provider_webhooks",
    )


def test_readiness_report_requires_marketing_inquiry_handoff_in_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(EMAIL_BACKEND_ENV, "smtp")
    monkeypatch.setenv(SMTP_HOST_ENV, "smtp.example.com")
    monkeypatch.setenv(SMTP_FROM_EMAIL_ENV, "reports@example.com")
    monkeypatch.delenv(MARKETING_INQUIRY_RECIPIENT_EMAIL_ENV, raising=False)
    monkeypatch.delenv(MARKETING_INQUIRY_CRM_WEBHOOK_URL_ENV, raising=False)
    monkeypatch.delenv(MARKETING_INQUIRY_CRM_WEBHOOK_SECRET_ENV, raising=False)
    monkeypatch.setenv(
        MARKETING_INQUIRY_RATE_LIMIT_SECRET_ENV,
        PRODUCTION_MARKETING_RATE_LIMIT_SECRET,
    )
    store = SaaSStore(initialize_database())

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "marketing_inquiries") == "fail"
    assert MARKETING_INQUIRY_RECIPIENT_EMAIL_ENV in check_detail(report, "marketing_inquiries")
    assert MARKETING_INQUIRY_CRM_WEBHOOK_URL_ENV in check_detail(report, "marketing_inquiries")


def test_readiness_report_requires_marketing_inquiry_rate_limit_secret_in_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(EMAIL_BACKEND_ENV, "smtp")
    monkeypatch.setenv(SMTP_HOST_ENV, "smtp.example.com")
    monkeypatch.setenv(SMTP_FROM_EMAIL_ENV, "reports@example.com")
    monkeypatch.setenv(MARKETING_INQUIRY_RECIPIENT_EMAIL_ENV, "sales@example.com")
    monkeypatch.delenv(MARKETING_INQUIRY_RATE_LIMIT_SECRET_ENV, raising=False)
    store = SaaSStore(initialize_database())

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "marketing_inquiries") == "fail"
    assert MARKETING_INQUIRY_RATE_LIMIT_SECRET_ENV in check_detail(
        report,
        "marketing_inquiries",
    )


def test_readiness_report_rejects_short_marketing_inquiry_rate_limit_secret_in_production(
    monkeypatch,
):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(EMAIL_BACKEND_ENV, "smtp")
    monkeypatch.setenv(SMTP_HOST_ENV, "smtp.example.com")
    monkeypatch.setenv(SMTP_FROM_EMAIL_ENV, "reports@example.com")
    monkeypatch.setenv(MARKETING_INQUIRY_RECIPIENT_EMAIL_ENV, "sales@example.com")
    monkeypatch.setenv(MARKETING_INQUIRY_RATE_LIMIT_SECRET_ENV, "rate-secret")
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    store = SaaSStore(initialize_database())
    record_recent_worker_heartbeat(store)
    record_recent_alert_delivery_evidence(store)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "marketing_inquiries") == "fail"
    assert "at least 32 characters" in check_detail(report, "marketing_inquiries")


def test_readiness_report_rejects_short_marketing_inquiry_crm_secret_in_production(
    monkeypatch,
):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    configure_production_email(monkeypatch)
    monkeypatch.delenv(MARKETING_INQUIRY_RECIPIENT_EMAIL_ENV, raising=False)
    monkeypatch.setenv(MARKETING_INQUIRY_CRM_WEBHOOK_URL_ENV, "https://crm.example.com/rds")
    monkeypatch.setenv(MARKETING_INQUIRY_CRM_WEBHOOK_SECRET_ENV, "crm-secret")
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    store = SaaSStore(initialize_database())

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "marketing_inquiries") == "fail"
    assert "at least 32 characters" in check_detail(report, "marketing_inquiries")


def test_readiness_report_accepts_crm_marketing_handoff_in_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    configure_production_auth_flow(monkeypatch)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    monkeypatch.delenv(MARKETING_INQUIRY_RECIPIENT_EMAIL_ENV, raising=False)
    monkeypatch.setenv(MARKETING_INQUIRY_CRM_WEBHOOK_URL_ENV, "https://crm.example.com/rds")
    monkeypatch.setenv(
        MARKETING_INQUIRY_CRM_WEBHOOK_SECRET_ENV,
        PRODUCTION_MARKETING_CRM_WEBHOOK_SECRET,
    )
    configure_production_email(monkeypatch)
    monkeypatch.delenv(MARKETING_INQUIRY_RECIPIENT_EMAIL_ENV, raising=False)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    store = SaaSStore(initialize_database())
    record_recent_worker_heartbeat(store)
    record_recent_alert_delivery_evidence(store)
    record_recent_provider_webhook_replay_evidence(store)
    record_recent_artifact_store_test_evidence(store)
    record_recent_database_test_evidence(store)
    record_recent_tenant_isolation_test_evidence(store)
    record_recent_run_lifecycle_test_evidence(store)
    record_recent_worker_run_lifecycle_test_evidence(store)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "ready"
    assert check_status(report, "marketing_inquiries") == "pass"
    assert check_status(report, "database_test_evidence") == "pass"
    assert check_status(report, "alert_delivery_evidence") == "pass"
    assert check_status(report, "clerk_webhook_replay_evidence") == "pass"
    assert check_status(report, "billing_webhook_replay_evidence") == "pass"
    assert check_status(report, "notification_provider_replay_evidence") == "pass"
    assert check_status(report, "artifact_store_test_evidence") == "pass"
    assert check_status(report, "tenant_isolation_test_evidence") == "pass"
    assert check_status(report, "run_lifecycle_test_evidence") == "pass"
    assert check_status(report, "worker_run_lifecycle_test_evidence") == "pass"
    assert check_status(report, "log_aggregation") == "pass"
    assert check_status(report, "worker_supervision") == "pass"
    assert MARKETING_INQUIRY_CRM_WEBHOOK_URL_ENV in check_detail(
        report,
        "marketing_inquiries",
    )


def test_readiness_report_requires_metrics_token_in_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.delenv(METRICS_TOKEN_ENV, raising=False)
    configure_production_email(monkeypatch)
    store = SaaSStore(initialize_database())

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "metrics") == "fail"
    assert METRICS_TOKEN_ENV in check_detail(report, "metrics")


def test_readiness_report_rejects_short_metrics_token_in_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, "short-token")
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    store = SaaSStore(initialize_database())

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "metrics") == "fail"
    assert "at least 32 characters" in check_detail(report, "metrics")


def test_readiness_report_requires_log_aggregation_url_in_production(monkeypatch):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.delenv(LOG_AGGREGATION_URL_ENV, raising=False)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "log_aggregation") == "fail"
    assert LOG_AGGREGATION_URL_ENV in check_detail(report, "log_aggregation")


def test_readiness_report_requires_https_log_aggregation_url_in_production(monkeypatch):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.setenv(LOG_AGGREGATION_URL_ENV, "http://logs.example.com/rds")

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "log_aggregation") == "fail"
    assert "https" in check_detail(report, "log_aggregation")


def test_readiness_report_requires_log_aggregation_verification_timestamp(
    monkeypatch,
):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.delenv(LOG_AGGREGATION_VERIFIED_AT_ENV, raising=False)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "log_aggregation") == "fail"
    assert LOG_AGGREGATION_VERIFIED_AT_ENV in check_detail(report, "log_aggregation")


def test_readiness_report_rejects_naive_log_aggregation_verification_timestamp(
    monkeypatch,
):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.setenv(LOG_AGGREGATION_VERIFIED_AT_ENV, "2026-01-01T00:00:00")

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "log_aggregation") == "fail"
    assert "timezone" in check_detail(report, "log_aggregation")


def test_readiness_report_rejects_stale_log_aggregation_verification(
    monkeypatch,
):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.setenv(LOG_AGGREGATION_MAX_AGE_DAYS_ENV, "7")
    monkeypatch.setenv(
        LOG_AGGREGATION_VERIFIED_AT_ENV,
        (datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
    )

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "log_aggregation") == "fail"
    assert "older than 7 days" in check_detail(report, "log_aggregation")


def test_readiness_report_rejects_invalid_log_aggregation_max_age(
    monkeypatch,
):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.setenv(LOG_AGGREGATION_MAX_AGE_DAYS_ENV, "0")

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "log_aggregation") == "fail"
    assert LOG_AGGREGATION_MAX_AGE_DAYS_ENV in check_detail(report, "log_aggregation")
    assert "positive" in check_detail(report, "log_aggregation")


def test_readiness_report_requires_log_retention_days_in_production(monkeypatch):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.delenv(LOG_RETENTION_DAYS_ENV, raising=False)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "log_aggregation") == "fail"
    assert LOG_RETENTION_DAYS_ENV in check_detail(report, "log_aggregation")


def test_readiness_report_rejects_invalid_log_retention_days(monkeypatch):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.setenv(LOG_RETENTION_DAYS_ENV, "0")

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "log_aggregation") == "fail"
    assert LOG_RETENTION_DAYS_ENV in check_detail(report, "log_aggregation")
    assert "positive" in check_detail(report, "log_aggregation")


def test_readiness_report_requires_support_access_token_in_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.delenv(SUPPORT_ACCESS_TOKEN_ENV, raising=False)
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    store = SaaSStore(initialize_database())

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "support_access") == "fail"
    assert SUPPORT_ACCESS_TOKEN_ENV in check_detail(report, "support_access")


def test_readiness_report_rejects_short_support_access_token_in_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, "short-support-token")
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    store = SaaSStore(initialize_database())

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "support_access") == "fail"
    assert "at least 32 characters" in check_detail(report, "support_access")


def test_readiness_report_requires_support_owner_email_in_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    monkeypatch.delenv(SUPPORT_OWNER_EMAIL_ENV, raising=False)
    store = SaaSStore(initialize_database())

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "support_access") == "fail"
    assert SUPPORT_OWNER_EMAIL_ENV in check_detail(report, "support_access")


def test_readiness_report_rejects_invalid_support_owner_email_in_production(
    monkeypatch,
):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    monkeypatch.setenv(SUPPORT_OWNER_EMAIL_ENV, "support-owner")
    store = SaaSStore(initialize_database())

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "support_access") == "fail"
    assert "email address" in check_detail(report, "support_access")


def test_readiness_report_rejects_invalid_csp_report_retention_in_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    monkeypatch.setenv(CSP_REPORT_RETENTION_DAYS_ENV, "0")
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    store = SaaSStore(initialize_database())
    record_recent_worker_heartbeat(store)
    record_recent_alert_delivery_evidence(store)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "csp_report_retention") == "fail"
    assert CSP_REPORT_RETENTION_DAYS_ENV in check_detail(report, "csp_report_retention")
    assert "between 1 and 3650" in check_detail(report, "csp_report_retention")


def test_readiness_report_requires_worker_enabled_in_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    monkeypatch.delenv(WORKER_ENABLED_ENV, raising=False)
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_backups(monkeypatch)
    configure_production_alerting(monkeypatch)
    store = SaaSStore(initialize_database())

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "worker") == "fail"
    assert WORKER_ENABLED_ENV in check_detail(report, "worker")


def test_readiness_report_rejects_tenant_filtered_worker_in_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    monkeypatch.setenv(WORKER_ORGANIZATION_ID_ENV, "org_1")
    store = SaaSStore(initialize_database())

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "worker") == "fail"
    assert WORKER_ORGANIZATION_ID_ENV in check_detail(report, "worker")


def test_readiness_report_rejects_run_once_worker_in_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    monkeypatch.setenv(WORKER_RUN_ONCE_ENV, "true")
    store = SaaSStore(initialize_database())

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "worker") == "fail"
    assert WORKER_RUN_ONCE_ENV in check_detail(report, "worker")


def test_readiness_report_requires_positive_worker_poll_interval_in_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    monkeypatch.setenv(WORKER_POLL_SECONDS_ENV, "0")
    store = SaaSStore(initialize_database())

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "worker") == "fail"
    assert WORKER_POLL_SECONDS_ENV in check_detail(report, "worker")


def test_readiness_report_requires_recent_worker_heartbeat_in_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    store = SaaSStore(initialize_database())

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "worker") == "fail"
    assert "No all-tenant worker heartbeat" in check_detail(report, "worker")


def test_readiness_report_requires_worker_supervisor_url_in_production(monkeypatch):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.delenv(WORKER_SUPERVISOR_URL_ENV, raising=False)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "worker_supervision") == "fail"
    assert WORKER_SUPERVISOR_URL_ENV in check_detail(report, "worker_supervision")


def test_readiness_report_requires_https_worker_supervisor_url_in_production(monkeypatch):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.setenv(WORKER_SUPERVISOR_URL_ENV, "http://ops.example.com/workers/rds")

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "worker_supervision") == "fail"
    assert "https" in check_detail(report, "worker_supervision")


def test_readiness_report_requires_worker_supervisor_verification_timestamp(
    monkeypatch,
):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.delenv(WORKER_SUPERVISOR_VERIFIED_AT_ENV, raising=False)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "worker_supervision") == "fail"
    assert WORKER_SUPERVISOR_VERIFIED_AT_ENV in check_detail(report, "worker_supervision")


def test_readiness_report_rejects_naive_worker_supervisor_verification_timestamp(
    monkeypatch,
):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.setenv(WORKER_SUPERVISOR_VERIFIED_AT_ENV, "2026-01-01T00:00:00")

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "worker_supervision") == "fail"
    assert "timezone" in check_detail(report, "worker_supervision")


def test_readiness_report_rejects_stale_worker_supervisor_verification(
    monkeypatch,
):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.setenv(WORKER_SUPERVISOR_MAX_AGE_DAYS_ENV, "7")
    monkeypatch.setenv(
        WORKER_SUPERVISOR_VERIFIED_AT_ENV,
        (datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
    )

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "worker_supervision") == "fail"
    assert "older than 7 days" in check_detail(report, "worker_supervision")


def test_readiness_report_rejects_invalid_worker_supervisor_max_age(
    monkeypatch,
):
    store = SaaSStore(initialize_database())
    configure_ready_production_runtime(monkeypatch, store)
    monkeypatch.setenv(WORKER_SUPERVISOR_MAX_AGE_DAYS_ENV, "0")

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "worker_supervision") == "fail"
    assert WORKER_SUPERVISOR_MAX_AGE_DAYS_ENV in check_detail(report, "worker_supervision")
    assert "positive" in check_detail(report, "worker_supervision")


def test_readiness_report_rejects_stale_worker_heartbeat_in_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, PRODUCTION_CLERK_WEBHOOK_SIGNING_SECRET)
    monkeypatch.setenv(METRICS_TOKEN_ENV, PRODUCTION_METRICS_TOKEN)
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, PRODUCTION_SUPPORT_TOKEN)
    monkeypatch.setenv(WORKER_HEARTBEAT_MAX_AGE_SECONDS_ENV, "60")
    configure_production_email(monkeypatch)
    configure_production_billing(monkeypatch)
    configure_production_security(monkeypatch)
    configure_production_operations(monkeypatch)
    store = SaaSStore(initialize_database())
    stale_seen_at = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    store.upsert_worker_heartbeat(
        worker_id="worker_stale",
        organization_id=None,
        poll_seconds=10,
        batch_size=10,
        last_attempted=0,
        last_succeeded=0,
        last_failed=0,
        last_skipped=0,
        last_scheduled_attempted=0,
        last_scheduled_sent=0,
        last_scheduled_failed=0,
        last_scheduled_skipped=0,
        last_seen_at=stale_seen_at,
    )

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "worker") == "fail"
    assert "stale" in check_detail(report, "worker")


def test_readiness_report_requires_clerk_webhook_secret_in_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.delenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, raising=False)
    configure_production_auth_flow(monkeypatch)
    configure_production_email(monkeypatch)
    store = SaaSStore(initialize_database())

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "auth") == "pass"
    assert check_status(report, "clerk_webhooks") == "fail"
    assert CLERK_WEBHOOK_SIGNING_SECRET_ENV in check_detail(report, "clerk_webhooks")


def test_readiness_report_rejects_short_clerk_webhook_secret_in_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv("RDS_AUTH_MODE", "clerk")
    monkeypatch.setenv(DATABASE_URL_ENV, "postgresql://rds:secret@db.example.com/rds")
    monkeypatch.setenv(CLERK_ISSUER_ENV, "https://example.clerk.accounts.dev")
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem())
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, "whsec_test_clerk_webhook_secret")
    configure_production_auth_flow(monkeypatch)
    configure_production_email(monkeypatch)
    store = SaaSStore(initialize_database())

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "auth") == "pass"
    assert check_status(report, "clerk_webhooks") == "fail"
    assert "at least 32 characters" in check_detail(report, "clerk_webhooks")


def test_readiness_report_fails_when_migrations_are_missing(monkeypatch):
    monkeypatch.delenv(RUNTIME_ENV_ENV, raising=False)
    monkeypatch.delenv("RDS_AUTH_MODE", raising=False)
    conn = connect_database()
    store = SaaSStore(conn)

    report = readiness_report(
        store=store,
        artifact_store=S3TenantArtifactStore("rds-artifacts", client=ProductionArtifactStore()),
    )

    assert report["status"] == "not_ready"
    assert check_status(report, "database") == "fail"
    assert "Database check failed" in check_detail(report, "database")
