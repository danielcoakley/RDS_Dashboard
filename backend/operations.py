from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from typing import Any, Callable
from urllib.parse import urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from .domain import (
    Membership,
    Meter,
    Organization,
    Role,
    Site,
    Upload,
    UploadStatus,
    User,
)
from .run_orchestration import (
    StoredUploadAnalysisRunRequest,
    execute_analysis_run_from_upload,
    queue_analysis_run_from_upload,
)
from .runtime_config import PRODUCTION_RUNTIME
from .storage_keys import report_storage_key, upload_storage_key
from .workers import process_queued_analysis_jobs


BACKUP_POLICY_URL_ENV = "SAAS_BACKUP_POLICY_URL"
BACKUP_JOB_URL_ENV = "SAAS_BACKUP_JOB_URL"
BACKUP_JOB_VERIFIED_AT_ENV = "SAAS_BACKUP_JOB_VERIFIED_AT"
BACKUP_MAX_JOB_AGE_DAYS_ENV = "SAAS_BACKUP_MAX_JOB_AGE_DAYS"
BACKUP_RESTORE_TESTED_AT_ENV = "SAAS_BACKUP_RESTORE_TESTED_AT"
BACKUP_MAX_RESTORE_TEST_AGE_DAYS_ENV = "SAAS_BACKUP_MAX_RESTORE_TEST_AGE_DAYS"
ALERT_WEBHOOK_URL_ENV = "SAAS_ALERT_WEBHOOK_URL"
ALERT_RUNBOOK_URL_ENV = "SAAS_ALERT_RUNBOOK_URL"
ALERT_RULES_URL_ENV = "SAAS_ALERT_RULES_URL"
ALERT_RULES_VERIFIED_AT_ENV = "SAAS_ALERT_RULES_VERIFIED_AT"
ALERT_RULES_MAX_AGE_DAYS_ENV = "SAAS_ALERT_RULES_MAX_AGE_DAYS"
ALERT_NOTIFICATION_DUE_AGE_SECONDS_ENV = "SAAS_ALERT_NOTIFICATION_DUE_AGE_SECONDS"
ALERT_NOTIFICATION_FAILED_COUNT_ENV = "SAAS_ALERT_NOTIFICATION_FAILED_COUNT"
ALERT_REPORT_SCHEDULE_DUE_AGE_SECONDS_ENV = "SAAS_ALERT_REPORT_SCHEDULE_DUE_AGE_SECONDS"
ALERT_REPORT_SCHEDULE_FAILED_COUNT_ENV = "SAAS_ALERT_REPORT_SCHEDULE_FAILED_COUNT"
ALERT_WEBHOOK_TIMEOUT_SECONDS_ENV = "SAAS_ALERT_WEBHOOK_TIMEOUT_SECONDS"
ALERT_DELIVERY_TEST_MAX_AGE_DAYS_ENV = "SAAS_ALERT_DELIVERY_TEST_MAX_AGE_DAYS"
PROVIDER_WEBHOOK_REPLAY_MAX_AGE_DAYS_ENV = "SAAS_PROVIDER_WEBHOOK_REPLAY_MAX_AGE_DAYS"
ARTIFACT_STORE_TEST_MAX_AGE_DAYS_ENV = "SAAS_ARTIFACT_STORE_TEST_MAX_AGE_DAYS"
DATABASE_TEST_MAX_AGE_DAYS_ENV = "SAAS_DATABASE_TEST_MAX_AGE_DAYS"
TENANT_ISOLATION_TEST_MAX_AGE_DAYS_ENV = "SAAS_TENANT_ISOLATION_TEST_MAX_AGE_DAYS"
RUN_LIFECYCLE_TEST_MAX_AGE_DAYS_ENV = "SAAS_RUN_LIFECYCLE_TEST_MAX_AGE_DAYS"
WORKER_RUN_LIFECYCLE_TEST_MAX_AGE_DAYS_ENV = (
    "SAAS_WORKER_RUN_LIFECYCLE_TEST_MAX_AGE_DAYS"
)
DEFAULT_MAX_RESTORE_TEST_AGE_DAYS = 90
DEFAULT_MAX_BACKUP_JOB_AGE_DAYS = 7
DEFAULT_ALERT_NOTIFICATION_DUE_AGE_SECONDS = 900
DEFAULT_ALERT_NOTIFICATION_FAILED_COUNT = 1
DEFAULT_ALERT_REPORT_SCHEDULE_DUE_AGE_SECONDS = 900
DEFAULT_ALERT_REPORT_SCHEDULE_FAILED_COUNT = 1
DEFAULT_ALERT_RULES_MAX_AGE_DAYS = 30
DEFAULT_ALERT_WEBHOOK_TIMEOUT_SECONDS = 5.0
DEFAULT_ALERT_DELIVERY_TEST_MAX_AGE_DAYS = 30
DEFAULT_PROVIDER_WEBHOOK_REPLAY_MAX_AGE_DAYS = 30
DEFAULT_ARTIFACT_STORE_TEST_MAX_AGE_DAYS = 30
DEFAULT_DATABASE_TEST_MAX_AGE_DAYS = 30
DEFAULT_TENANT_ISOLATION_TEST_MAX_AGE_DAYS = 30
DEFAULT_RUN_LIFECYCLE_TEST_MAX_AGE_DAYS = 30
DEFAULT_WORKER_RUN_LIFECYCLE_TEST_MAX_AGE_DAYS = 30
ALERT_DELIVERY_TEST_EVENT_TYPE = "rds.alert_delivery_test"
REPORT_SCHEDULE_DISPATCH_EVENT_TYPE = "rds.report_schedule_dispatch"
CLERK_WEBHOOK_REPLAY_EVENT_TYPE = "rds.clerk_webhook_replay"
BILLING_WEBHOOK_REPLAY_EVENT_TYPE = "rds.billing_webhook_replay"
NOTIFICATION_PROVIDER_WEBHOOK_REPLAY_EVENT_TYPE = "rds.notification_provider_webhook_replay"
ARTIFACT_STORE_TEST_EVENT_TYPE = "rds.artifact_store_test"
DATABASE_TEST_EVENT_TYPE = "rds.database_test"
TENANT_ISOLATION_TEST_EVENT_TYPE = "rds.tenant_isolation_test"
RUN_LIFECYCLE_TEST_EVENT_TYPE = "rds.run_lifecycle_smoke_test"
WORKER_RUN_LIFECYCLE_TEST_EVENT_TYPE = "rds.worker_run_lifecycle_smoke_test"
SUPPORTED_PLATFORM_OPERATION_EVENT_TYPES = frozenset(
    {
        ALERT_DELIVERY_TEST_EVENT_TYPE,
        REPORT_SCHEDULE_DISPATCH_EVENT_TYPE,
        CLERK_WEBHOOK_REPLAY_EVENT_TYPE,
        BILLING_WEBHOOK_REPLAY_EVENT_TYPE,
        NOTIFICATION_PROVIDER_WEBHOOK_REPLAY_EVENT_TYPE,
        ARTIFACT_STORE_TEST_EVENT_TYPE,
        DATABASE_TEST_EVENT_TYPE,
        TENANT_ISOLATION_TEST_EVENT_TYPE,
        RUN_LIFECYCLE_TEST_EVENT_TYPE,
        WORKER_RUN_LIFECYCLE_TEST_EVENT_TYPE,
    }
)
SUPPORTED_PLATFORM_OPERATION_EVENT_STATUSES = frozenset({"succeeded", "failed"})


class OperationsConfigurationError(ValueError):
    pass


class AlertDeliveryError(RuntimeError):
    pass


class TenantIsolationSmokeTestError(RuntimeError):
    pass


class RunLifecycleSmokeTestError(RuntimeError):
    pass


class WorkerRunLifecycleSmokeTestError(RuntimeError):
    pass


AlertWebhookSender = Callable[[str, dict[str, Any], float], int]


@dataclass(frozen=True)
class AlertDeliveryTestResult:
    event_type: str
    delivered_at: str
    status_code: int


@dataclass(frozen=True)
class TenantIsolationSmokeTestResult:
    event_type: str
    tested_at: str
    checks_passed: int
    synthetic_records: int


@dataclass(frozen=True)
class RunLifecycleSmokeTestResult:
    event_type: str
    tested_at: str
    checks_passed: int
    artifacts_verified: int
    synthetic_records: int


@dataclass(frozen=True)
class WorkerRunLifecycleSmokeTestResult:
    event_type: str
    tested_at: str
    checks_passed: int
    jobs_attempted: int
    jobs_succeeded: int
    artifacts_verified: int
    synthetic_records: int


def _configured_value(name: str) -> str:
    return os.getenv(name, "").strip()


def _parse_positive_int_env(name: str, default: int) -> int:
    value = _configured_value(name)
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise OperationsConfigurationError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise OperationsConfigurationError(f"{name} must be positive")
    return parsed


def _parse_positive_float_env(name: str, default: float) -> float:
    value = _configured_value(name)
    if not value:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise OperationsConfigurationError(f"{name} must be a number") from exc
    if parsed <= 0:
        raise OperationsConfigurationError(f"{name} must be positive")
    return parsed


def _parse_alert_threshold_env(name: str, default: int, runtime_env: str) -> int:
    value = _configured_value(name)
    if not value and runtime_env == PRODUCTION_RUNTIME:
        raise OperationsConfigurationError(f"{name} is required in production")
    return _parse_positive_int_env(name, default)


def _require_https_url(name: str, runtime_env: str) -> str | None:
    value = _configured_value(name)
    if not value:
        if runtime_env == PRODUCTION_RUNTIME:
            raise OperationsConfigurationError(f"{name} is required in production")
        return None

    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise OperationsConfigurationError(f"{name} must be a URL")
    if runtime_env == PRODUCTION_RUNTIME and parsed.scheme != "https":
        raise OperationsConfigurationError(f"{name} must use https in production")
    return value


def alert_webhook_timeout_seconds_from_env() -> float:
    return _parse_positive_float_env(
        ALERT_WEBHOOK_TIMEOUT_SECONDS_ENV,
        DEFAULT_ALERT_WEBHOOK_TIMEOUT_SECONDS,
    )


def alert_delivery_test_max_age_days_from_env() -> int:
    return _parse_positive_int_env(
        ALERT_DELIVERY_TEST_MAX_AGE_DAYS_ENV,
        DEFAULT_ALERT_DELIVERY_TEST_MAX_AGE_DAYS,
    )


def alert_rules_max_age_days_from_env() -> int:
    return _parse_positive_int_env(
        ALERT_RULES_MAX_AGE_DAYS_ENV,
        DEFAULT_ALERT_RULES_MAX_AGE_DAYS,
    )


def provider_webhook_replay_max_age_days_from_env() -> int:
    return _parse_positive_int_env(
        PROVIDER_WEBHOOK_REPLAY_MAX_AGE_DAYS_ENV,
        DEFAULT_PROVIDER_WEBHOOK_REPLAY_MAX_AGE_DAYS,
    )


def artifact_store_test_max_age_days_from_env() -> int:
    return _parse_positive_int_env(
        ARTIFACT_STORE_TEST_MAX_AGE_DAYS_ENV,
        DEFAULT_ARTIFACT_STORE_TEST_MAX_AGE_DAYS,
    )


def database_test_max_age_days_from_env() -> int:
    return _parse_positive_int_env(
        DATABASE_TEST_MAX_AGE_DAYS_ENV,
        DEFAULT_DATABASE_TEST_MAX_AGE_DAYS,
    )


def tenant_isolation_test_max_age_days_from_env() -> int:
    return _parse_positive_int_env(
        TENANT_ISOLATION_TEST_MAX_AGE_DAYS_ENV,
        DEFAULT_TENANT_ISOLATION_TEST_MAX_AGE_DAYS,
    )


def run_lifecycle_test_max_age_days_from_env() -> int:
    return _parse_positive_int_env(
        RUN_LIFECYCLE_TEST_MAX_AGE_DAYS_ENV,
        DEFAULT_RUN_LIFECYCLE_TEST_MAX_AGE_DAYS,
    )


def worker_run_lifecycle_test_max_age_days_from_env() -> int:
    return _parse_positive_int_env(
        WORKER_RUN_LIFECYCLE_TEST_MAX_AGE_DAYS_ENV,
        DEFAULT_WORKER_RUN_LIFECYCLE_TEST_MAX_AGE_DAYS,
    )


def backup_job_max_age_days_from_env() -> int:
    return _parse_positive_int_env(
        BACKUP_MAX_JOB_AGE_DAYS_ENV,
        DEFAULT_MAX_BACKUP_JOB_AGE_DAYS,
    )


def _parse_restore_tested_at(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise OperationsConfigurationError(
            f"{BACKUP_RESTORE_TESTED_AT_ENV} must be an ISO-8601 datetime"
        ) from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_backup_job_verified_at(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise OperationsConfigurationError(
            f"{BACKUP_JOB_VERIFIED_AT_ENV} must be an ISO-8601 datetime"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise OperationsConfigurationError(f"{BACKUP_JOB_VERIFIED_AT_ENV} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _parse_alert_rules_verified_at(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise OperationsConfigurationError(
            f"{ALERT_RULES_VERIFIED_AT_ENV} must be an ISO-8601 datetime"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise OperationsConfigurationError(f"{ALERT_RULES_VERIFIED_AT_ENV} must include a timezone")
    return parsed.astimezone(timezone.utc)


def backup_configuration_status(
    runtime_env: str,
    *,
    now: datetime | None = None,
) -> tuple[bool, str]:
    try:
        policy_url = _require_https_url(BACKUP_POLICY_URL_ENV, runtime_env)
        job_url = _require_https_url(BACKUP_JOB_URL_ENV, runtime_env)
        max_age_days = _parse_positive_int_env(
            BACKUP_MAX_RESTORE_TEST_AGE_DAYS_ENV,
            DEFAULT_MAX_RESTORE_TEST_AGE_DAYS,
        )
        max_job_age_days = backup_job_max_age_days_from_env()
    except OperationsConfigurationError as exc:
        return False, str(exc)

    restore_tested_at = _configured_value(BACKUP_RESTORE_TESTED_AT_ENV)
    backup_job_verified_at = _configured_value(BACKUP_JOB_VERIFIED_AT_ENV)
    if runtime_env == PRODUCTION_RUNTIME and not restore_tested_at:
        return False, f"{BACKUP_RESTORE_TESTED_AT_ENV} is required in production"
    if runtime_env == PRODUCTION_RUNTIME and not backup_job_verified_at:
        return False, f"{BACKUP_JOB_VERIFIED_AT_ENV} is required in production"
    if not restore_tested_at and not backup_job_verified_at:
        return True, "Backup job and restore-test evidence are optional in development"

    effective_now = now or datetime.now(timezone.utc)
    detail_parts: list[str] = []

    if policy_url:
        detail_parts.append(f"{BACKUP_POLICY_URL_ENV} configured")
    if job_url:
        detail_parts.append(f"{BACKUP_JOB_URL_ENV} configured")

    if restore_tested_at:
        try:
            tested_at = _parse_restore_tested_at(restore_tested_at)
        except OperationsConfigurationError as exc:
            return False, str(exc)
        if tested_at > effective_now:
            return False, f"{BACKUP_RESTORE_TESTED_AT_ENV} cannot be in the future"
        age_days = (effective_now - tested_at).days
        if runtime_env == PRODUCTION_RUNTIME and age_days > max_age_days:
            return (
                False,
                f"{BACKUP_RESTORE_TESTED_AT_ENV} is older than {max_age_days} days",
            )
        detail_parts.append(f"restore test age {age_days} day(s)")

    if backup_job_verified_at:
        try:
            verified_at = _parse_backup_job_verified_at(backup_job_verified_at)
        except OperationsConfigurationError as exc:
            return False, str(exc)
        if verified_at > effective_now:
            return False, f"{BACKUP_JOB_VERIFIED_AT_ENV} cannot be in the future"
        job_age_days = (effective_now - verified_at).days
        if runtime_env == PRODUCTION_RUNTIME and job_age_days > max_job_age_days:
            return False, f"{BACKUP_JOB_VERIFIED_AT_ENV} is older than {max_job_age_days} days"
        detail_parts.append(f"backup job verified age {job_age_days} day(s)")

    return True, "; ".join(detail_parts)


def alerting_configuration_status(runtime_env: str) -> tuple[bool, str]:
    try:
        webhook_url = _require_https_url(ALERT_WEBHOOK_URL_ENV, runtime_env)
        runbook_url = _require_https_url(ALERT_RUNBOOK_URL_ENV, runtime_env)
        alert_rules_url = _require_https_url(ALERT_RULES_URL_ENV, runtime_env)
        alert_rules_max_age_days = alert_rules_max_age_days_from_env()
        webhook_timeout_seconds = alert_webhook_timeout_seconds_from_env()
        notification_due_age_seconds = _parse_alert_threshold_env(
            ALERT_NOTIFICATION_DUE_AGE_SECONDS_ENV,
            DEFAULT_ALERT_NOTIFICATION_DUE_AGE_SECONDS,
            runtime_env,
        )
        notification_failed_count = _parse_alert_threshold_env(
            ALERT_NOTIFICATION_FAILED_COUNT_ENV,
            DEFAULT_ALERT_NOTIFICATION_FAILED_COUNT,
            runtime_env,
        )
        report_schedule_due_age_seconds = _parse_alert_threshold_env(
            ALERT_REPORT_SCHEDULE_DUE_AGE_SECONDS_ENV,
            DEFAULT_ALERT_REPORT_SCHEDULE_DUE_AGE_SECONDS,
            runtime_env,
        )
        report_schedule_failed_count = _parse_alert_threshold_env(
            ALERT_REPORT_SCHEDULE_FAILED_COUNT_ENV,
            DEFAULT_ALERT_REPORT_SCHEDULE_FAILED_COUNT,
            runtime_env,
        )
    except OperationsConfigurationError as exc:
        return False, str(exc)

    alert_rules_verified_at = _configured_value(ALERT_RULES_VERIFIED_AT_ENV)
    if runtime_env == PRODUCTION_RUNTIME and not alert_rules_verified_at:
        return False, f"{ALERT_RULES_VERIFIED_AT_ENV} is required in production"

    alert_rules_detail = ""
    if alert_rules_verified_at:
        try:
            verified_at = _parse_alert_rules_verified_at(alert_rules_verified_at)
        except OperationsConfigurationError as exc:
            return False, str(exc)
        effective_now = datetime.now(timezone.utc)
        if verified_at > effective_now:
            return False, f"{ALERT_RULES_VERIFIED_AT_ENV} cannot be in the future"
        age_days = (effective_now - verified_at).days
        if runtime_env == PRODUCTION_RUNTIME and age_days > alert_rules_max_age_days:
            return False, f"{ALERT_RULES_VERIFIED_AT_ENV} is older than {alert_rules_max_age_days} days"
        alert_rules_detail = f"; alert rules verified age {age_days} day(s)"

    if webhook_url and runbook_url:
        alert_rules_prefix = f" and {ALERT_RULES_URL_ENV}" if alert_rules_url else ""
        return (
            True,
            (
                f"{ALERT_WEBHOOK_URL_ENV} and {ALERT_RUNBOOK_URL_ENV}{alert_rules_prefix} configured; "
                f"{ALERT_WEBHOOK_TIMEOUT_SECONDS_ENV}="
                f"{webhook_timeout_seconds:g}; "
                f"{ALERT_NOTIFICATION_DUE_AGE_SECONDS_ENV}="
                f"{notification_due_age_seconds}; "
                f"{ALERT_NOTIFICATION_FAILED_COUNT_ENV}={notification_failed_count}; "
                f"{ALERT_REPORT_SCHEDULE_DUE_AGE_SECONDS_ENV}="
                f"{report_schedule_due_age_seconds}; "
                f"{ALERT_REPORT_SCHEDULE_FAILED_COUNT_ENV}="
                f"{report_schedule_failed_count}"
                f"{alert_rules_detail}"
            ),
        )
    return (
        True,
        (
            "Alerting webhook, runbook, and alert-rule evidence are optional in development; "
            f"notification alert webhook timeout uses {ALERT_WEBHOOK_TIMEOUT_SECONDS_ENV}="
            f"{webhook_timeout_seconds:g}; "
            f"notification alert defaults use {ALERT_NOTIFICATION_DUE_AGE_SECONDS_ENV}="
            f"{notification_due_age_seconds}; "
            f"{ALERT_NOTIFICATION_FAILED_COUNT_ENV}={notification_failed_count}; "
            f"report schedule alert defaults use {ALERT_REPORT_SCHEDULE_DUE_AGE_SECONDS_ENV}="
            f"{report_schedule_due_age_seconds}; "
            f"{ALERT_REPORT_SCHEDULE_FAILED_COUNT_ENV}={report_schedule_failed_count}"
            f"{alert_rules_detail}"
        ),
    )


def _default_alert_webhook_sender(
    webhook_url: str,
    payload: dict[str, Any],
    timeout_seconds: float,
) -> int:
    encoded_payload = json.dumps(payload, sort_keys=True).encode("utf-8")
    request = Request(
        webhook_url,
        data=encoded_payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "User-Agent": "rds-saas-alert-test",
        },
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return int(response.status)
    except HTTPError as exc:
        return int(exc.code)
    except (OSError, TimeoutError, URLError) as exc:
        raise AlertDeliveryError("Alert webhook delivery failed") from exc


def send_alert_delivery_test(
    *,
    request_id: str | None = None,
    runtime_env: str = PRODUCTION_RUNTIME,
    webhook_sender: AlertWebhookSender | None = None,
    now: datetime | None = None,
) -> AlertDeliveryTestResult:
    webhook_url = _require_https_url(ALERT_WEBHOOK_URL_ENV, runtime_env)
    if webhook_url is None:
        raise OperationsConfigurationError(
            f"{ALERT_WEBHOOK_URL_ENV} must be configured for alert delivery tests"
        )

    delivered_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    timeout_seconds = alert_webhook_timeout_seconds_from_env()
    payload: dict[str, Any] = {
        "event_type": ALERT_DELIVERY_TEST_EVENT_TYPE,
        "service": "rds-saas-api",
        "severity": "info",
        "message": "RDS SaaS alert delivery test",
        "created_at": delivered_at,
    }
    if request_id:
        payload["request_id"] = request_id

    sender = webhook_sender or _default_alert_webhook_sender
    status_code = sender(webhook_url, payload, timeout_seconds)
    if status_code < 200 or status_code >= 300:
        raise AlertDeliveryError(f"Alert webhook returned HTTP {status_code}")

    return AlertDeliveryTestResult(
        event_type=ALERT_DELIVERY_TEST_EVENT_TYPE,
        delivered_at=delivered_at,
        status_code=status_code,
    )


def _assert_exact_ids(
    actual_ids: set[str],
    expected_ids: set[str],
    check_label: str,
) -> None:
    if actual_ids != expected_ids:
        raise TenantIsolationSmokeTestError(
            f"Tenant isolation smoke test failed for {check_label}"
        )


def _rollback_tenant_isolation_savepoint(conn: Any) -> None:
    conn.execute("ROLLBACK TO SAVEPOINT tenant_isolation_smoke_test")
    conn.execute("RELEASE SAVEPOINT tenant_isolation_smoke_test")


def run_tenant_isolation_smoke_test(
    store: Any,
    *,
    now: datetime | None = None,
) -> TenantIsolationSmokeTestResult:
    suffix = uuid4().hex
    user_a_id = f"platform_tenant_probe_user_a_{suffix}"
    user_b_id = f"platform_tenant_probe_user_b_{suffix}"
    org_a_id = f"platform_tenant_probe_org_a_{suffix}"
    org_b_id = f"platform_tenant_probe_org_b_{suffix}"
    site_a_id = f"platform_tenant_probe_site_a_{suffix}"
    site_b_id = f"platform_tenant_probe_site_b_{suffix}"
    meter_a_id = f"platform_tenant_probe_meter_a_{suffix}"
    meter_b_id = f"platform_tenant_probe_meter_b_{suffix}"
    upload_a_id = f"platform_tenant_probe_upload_a_{suffix}"
    upload_b_id = f"platform_tenant_probe_upload_b_{suffix}"
    tested_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    checks_passed = 0

    store.conn.execute("SAVEPOINT tenant_isolation_smoke_test")
    try:
        store.create_user(
            User(
                id=user_a_id,
                email=f"{user_a_id}@example.invalid",
                display_name="Tenant Probe User A",
            )
        )
        store.create_user(
            User(
                id=user_b_id,
                email=f"{user_b_id}@example.invalid",
                display_name="Tenant Probe User B",
            )
        )
        store.create_organization(
            Organization(
                id=org_a_id,
                name="Tenant Probe A",
                slug=f"tenant-probe-a-{suffix}",
            )
        )
        store.create_organization(
            Organization(
                id=org_b_id,
                name="Tenant Probe B",
                slug=f"tenant-probe-b-{suffix}",
            )
        )
        store.add_membership(Membership(user_id=user_a_id, organization_id=org_a_id, role=Role.OWNER))
        store.add_membership(Membership(user_id=user_b_id, organization_id=org_b_id, role=Role.OWNER))
        store.create_site(
            Site(
                id=site_a_id,
                organization_id=org_a_id,
                name="Tenant Probe Site A",
                timezone="UTC",
            )
        )
        store.create_site(
            Site(
                id=site_b_id,
                organization_id=org_b_id,
                name="Tenant Probe Site B",
                timezone="UTC",
            )
        )
        store.create_meter(
            Meter(
                id=meter_a_id,
                organization_id=org_a_id,
                site_id=site_a_id,
                display_name="Tenant Probe Meter A",
                commodity="electricity",
                unit="kWh",
                source_column="meter_a_kwh",
            )
        )
        store.create_meter(
            Meter(
                id=meter_b_id,
                organization_id=org_b_id,
                site_id=site_b_id,
                display_name="Tenant Probe Meter B",
                commodity="electricity",
                unit="kWh",
                source_column="meter_b_kwh",
            )
        )
        store.create_upload(
            Upload(
                id=upload_a_id,
                organization_id=org_a_id,
                site_id=site_a_id,
                uploaded_by_user_id=user_a_id,
                category="energy",
                storage_key=(
                    f"tenants/{org_a_id}/sites/{site_a_id}/uploads/{upload_a_id}/energy.csv"
                ),
                checksum=f"checksum-{suffix}-a",
                status=UploadStatus.STORED,
            )
        )
        store.create_upload(
            Upload(
                id=upload_b_id,
                organization_id=org_b_id,
                site_id=site_b_id,
                uploaded_by_user_id=user_b_id,
                category="energy",
                storage_key=(
                    f"tenants/{org_b_id}/sites/{site_b_id}/uploads/{upload_b_id}/energy.csv"
                ),
                checksum=f"checksum-{suffix}-b",
                status=UploadStatus.STORED,
            )
        )

        _assert_exact_ids(
            {organization.id for organization in store.list_user_organizations(user_a_id)},
            {org_a_id},
            "user organization listing",
        )
        checks_passed += 1
        _assert_exact_ids(
            {site.id for site in store.list_sites(org_a_id)},
            {site_a_id},
            "site listing",
        )
        checks_passed += 1
        _assert_exact_ids(
            {meter.id for meter in store.list_meters(org_a_id)},
            {meter_a_id},
            "meter listing",
        )
        checks_passed += 1
        _assert_exact_ids(
            {row["id"] for row in store.list_uploads(org_a_id)},
            {upload_a_id},
            "upload listing",
        )
        checks_passed += 1
        _assert_exact_ids(
            {meter.id for meter in store.list_meters(org_a_id, site_id=site_b_id)},
            set(),
            "cross-tenant meter site filter",
        )
        checks_passed += 1
        _assert_exact_ids(
            {row["id"] for row in store.list_uploads(org_a_id, site_id=site_b_id)},
            set(),
            "cross-tenant upload site filter",
        )
        checks_passed += 1
    except TenantIsolationSmokeTestError:
        _rollback_tenant_isolation_savepoint(store.conn)
        raise
    except Exception as exc:
        _rollback_tenant_isolation_savepoint(store.conn)
        raise TenantIsolationSmokeTestError(
            "Tenant isolation smoke test failed during metadata write/read"
        ) from exc
    else:
        _rollback_tenant_isolation_savepoint(store.conn)

    if (
        store.get_organization(org_a_id) is not None
        or store.get_organization(org_b_id) is not None
        or store.get_user(user_a_id) is not None
        or store.get_user(user_b_id) is not None
    ):
        raise TenantIsolationSmokeTestError(
            "Tenant isolation smoke test failed during synthetic cleanup"
        )

    return TenantIsolationSmokeTestResult(
        event_type=TENANT_ISOLATION_TEST_EVENT_TYPE,
        tested_at=tested_at,
        checks_passed=checks_passed,
        synthetic_records=12,
    )


def _cleanup_run_lifecycle_smoke_test(
    store: Any,
    artifact_store: Any,
    *,
    organization_id: str,
    user_id: str,
    storage_keys: tuple[str, ...],
) -> None:
    for storage_key in storage_keys:
        try:
            artifact_store.delete(organization_id, storage_key)
        except FileNotFoundError:
            continue

    organization_tables = (
        "notification_provider_events",
        "organization_report_schedule_deliveries",
        "organization_report_schedules",
        "organization_report_approval_assignments",
        "organization_report_evidence_packages",
        "organization_report_reviews",
        "organization_report_comments",
        "notifications",
        "analysis_jobs",
        "reports",
        "run_uploads",
        "runs",
        "uploads",
        "audit_events",
        "analysis_configs",
        "organization_pilot_readiness_items",
        "organization_privacy_requests",
        "organization_privacy_settings",
        "organization_report_branding",
        "organization_subscriptions",
        "meters",
        "sites",
        "organization_invites",
        "memberships",
    )
    with store.conn:
        for table_name in organization_tables:
            store.conn.execute(
                f"DELETE FROM {table_name} WHERE organization_id = ?",
                (organization_id,),
            )
        store.conn.execute("DELETE FROM organizations WHERE id = ?", (organization_id,))
        store.conn.execute("DELETE FROM users WHERE id = ?", (user_id,))


def run_lifecycle_smoke_test(
    store: Any,
    artifact_store: Any,
    *,
    now: datetime | None = None,
) -> RunLifecycleSmokeTestResult:
    suffix = uuid4().hex
    user_id = f"platform_run_probe_user_{suffix}"
    org_id = f"platform_run_probe_org_{suffix}"
    site_id = f"platform_run_probe_site_{suffix}"
    meter_id = f"platform_run_probe_meter_{suffix}"
    upload_id = f"platform_run_probe_upload_{suffix}"
    run_id = f"platform_run_probe_run_{suffix}"
    report_id = f"platform_run_probe_report_{suffix}"
    meter_source_column = "Platform Probe kWh"
    tested_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    upload_key = upload_storage_key(org_id, site_id, upload_id, "energy.csv")
    expected_report_key = report_storage_key(org_id, site_id, run_id, "iso-summary.json")
    csv_payload = (
        f"Date,{meter_source_column}\n"
        "2026-01-01,100\n"
        "2026-02-01,120\n"
    )
    storage_keys = (upload_key, expected_report_key)
    checks_passed = 0
    result: RunLifecycleSmokeTestResult | None = None
    failure: Exception | None = None

    try:
        with store.conn:
            store.create_user(
                User(
                    id=user_id,
                    email=f"{user_id}@example.invalid",
                    display_name="Run Lifecycle Probe User",
                )
            )
            store.create_organization(
                Organization(
                    id=org_id,
                    name="Run Lifecycle Probe",
                    slug=f"run-lifecycle-probe-{suffix}",
                )
            )
            store.add_membership(Membership(user_id=user_id, organization_id=org_id, role=Role.OWNER))
            store.create_site(
                Site(
                    id=site_id,
                    organization_id=org_id,
                    name="Run Lifecycle Probe Site",
                    timezone="UTC",
                )
            )
            store.create_meter(
                Meter(
                    id=meter_id,
                    organization_id=org_id,
                    site_id=site_id,
                    display_name="Run Lifecycle Probe Meter",
                    commodity="electricity",
                    unit="kWh",
                    source_column=meter_source_column,
                    is_seu=True,
                )
            )
            store.create_upload(
                Upload(
                    id=upload_id,
                    organization_id=org_id,
                    site_id=site_id,
                    uploaded_by_user_id=user_id,
                    category="energy",
                    storage_key=upload_key,
                    checksum=sha256(csv_payload.encode("utf-8")).hexdigest(),
                    status=UploadStatus.STORED,
                )
            )
        artifact_store.put_text(org_id, upload_key, csv_payload)
        if not artifact_store.exists(org_id, upload_key):
            raise RunLifecycleSmokeTestError(
                "Run lifecycle smoke test failed during upload artifact verification"
            )
        checks_passed += 1

        execute_analysis_run_from_upload(
            store,
            StoredUploadAnalysisRunRequest(
                user_id=user_id,
                organization_id=org_id,
                upload_id=upload_id,
                run_id=run_id,
                report_id=report_id,
            ),
            artifact_store,
        )

        run_row = store.get_run(run_id)
        if run_row is None or run_row["status"] != "succeeded":
            raise RunLifecycleSmokeTestError(
                "Run lifecycle smoke test failed during run status verification"
            )
        checks_passed += 1

        job_row = store.get_analysis_job(f"job_{run_id}")
        if job_row is None or job_row["status"] != "succeeded":
            raise RunLifecycleSmokeTestError(
                "Run lifecycle smoke test failed during job status verification"
            )
        checks_passed += 1

        report_row = store.get_report(report_id)
        if report_row is None or report_row["storage_key"] != expected_report_key:
            raise RunLifecycleSmokeTestError(
                "Run lifecycle smoke test failed during report metadata verification"
            )
        checks_passed += 1

        if not artifact_store.exists(org_id, expected_report_key):
            raise RunLifecycleSmokeTestError(
                "Run lifecycle smoke test failed during report artifact verification"
            )
        report_content = json.loads(artifact_store.get_text(org_id, expected_report_key))
        if report_content.get("total_records") != 2:
            raise RunLifecycleSmokeTestError(
                "Run lifecycle smoke test failed during report content verification"
            )
        checks_passed += 1

        serialized_report = json.dumps(report_content, sort_keys=True)
        if "tenants/" in serialized_report or "storage_key" in serialized_report:
            raise RunLifecycleSmokeTestError(
                "Run lifecycle smoke test failed during report lineage sanitization"
            )
        checks_passed += 1

        result = RunLifecycleSmokeTestResult(
            event_type=RUN_LIFECYCLE_TEST_EVENT_TYPE,
            tested_at=tested_at,
            checks_passed=checks_passed,
            artifacts_verified=2,
            synthetic_records=15,
        )
    except Exception as exc:
        failure = exc

    try:
        _cleanup_run_lifecycle_smoke_test(
            store,
            artifact_store,
            organization_id=org_id,
            user_id=user_id,
            storage_keys=storage_keys,
        )
        if (
            store.get_organization(org_id) is not None
            or store.get_user(user_id) is not None
            or store.get_upload(upload_id) is not None
            or store.get_run(run_id) is not None
            or store.get_analysis_job(f"job_{run_id}") is not None
            or store.get_report(report_id) is not None
            or artifact_store.exists(org_id, upload_key)
            or artifact_store.exists(org_id, expected_report_key)
        ):
            raise RuntimeError("synthetic rows or artifacts remained after cleanup")
    except Exception as cleanup_exc:
        raise RunLifecycleSmokeTestError(
            "Run lifecycle smoke test failed during synthetic cleanup"
        ) from cleanup_exc

    if failure is not None:
        if isinstance(failure, RunLifecycleSmokeTestError):
            raise failure
        raise RunLifecycleSmokeTestError(
            "Run lifecycle smoke test failed during upload-to-report execution"
        ) from failure
    if result is None:
        raise RunLifecycleSmokeTestError(
            "Run lifecycle smoke test failed before producing a result"
        )
    return result


def worker_run_lifecycle_smoke_test(
    store: Any,
    artifact_store: Any,
    *,
    now: datetime | None = None,
) -> WorkerRunLifecycleSmokeTestResult:
    suffix = uuid4().hex
    user_id = f"platform_worker_probe_user_{suffix}"
    org_id = f"platform_worker_probe_org_{suffix}"
    site_id = f"platform_worker_probe_site_{suffix}"
    meter_id = f"platform_worker_probe_meter_{suffix}"
    upload_id = f"platform_worker_probe_upload_{suffix}"
    run_id = f"platform_worker_probe_run_{suffix}"
    report_id = f"platform_worker_probe_report_{suffix}"
    meter_source_column = "Platform Worker Probe kWh"
    tested_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    upload_key = upload_storage_key(org_id, site_id, upload_id, "energy.csv")
    expected_report_key = report_storage_key(org_id, site_id, run_id, "iso-summary.json")
    csv_payload = (
        f"Date,{meter_source_column}\n"
        "2026-01-01,100\n"
        "2026-02-01,120\n"
    )
    storage_keys = (upload_key, expected_report_key)
    checks_passed = 0
    result: WorkerRunLifecycleSmokeTestResult | None = None
    failure: Exception | None = None

    try:
        with store.conn:
            store.create_user(
                User(
                    id=user_id,
                    email=f"{user_id}@example.invalid",
                    display_name="Worker Run Lifecycle Probe User",
                )
            )
            store.create_organization(
                Organization(
                    id=org_id,
                    name="Worker Run Lifecycle Probe",
                    slug=f"worker-run-lifecycle-probe-{suffix}",
                )
            )
            store.add_membership(Membership(user_id=user_id, organization_id=org_id, role=Role.OWNER))
            store.create_site(
                Site(
                    id=site_id,
                    organization_id=org_id,
                    name="Worker Run Lifecycle Probe Site",
                    timezone="UTC",
                )
            )
            store.create_meter(
                Meter(
                    id=meter_id,
                    organization_id=org_id,
                    site_id=site_id,
                    display_name="Worker Run Lifecycle Probe Meter",
                    commodity="electricity",
                    unit="kWh",
                    source_column=meter_source_column,
                    is_seu=True,
                )
            )
            store.create_upload(
                Upload(
                    id=upload_id,
                    organization_id=org_id,
                    site_id=site_id,
                    uploaded_by_user_id=user_id,
                    category="energy",
                    storage_key=upload_key,
                    checksum=sha256(csv_payload.encode("utf-8")).hexdigest(),
                    status=UploadStatus.STORED,
                )
            )
        artifact_store.put_text(org_id, upload_key, csv_payload)
        if not artifact_store.exists(org_id, upload_key):
            raise WorkerRunLifecycleSmokeTestError(
                "Worker run lifecycle smoke test failed during upload artifact verification"
            )
        checks_passed += 1

        queue_analysis_run_from_upload(
            store,
            StoredUploadAnalysisRunRequest(
                user_id=user_id,
                organization_id=org_id,
                upload_id=upload_id,
                run_id=run_id,
                report_id=report_id,
            ),
        )

        queued_job = store.get_analysis_job(f"job_{run_id}")
        if queued_job is None or queued_job["status"] != "queued":
            raise WorkerRunLifecycleSmokeTestError(
                "Worker run lifecycle smoke test failed during queued job verification"
            )
        checks_passed += 1

        summary = process_queued_analysis_jobs(
            store,
            artifact_store,
            organization_id=org_id,
            limit=1,
        )
        if summary.attempted != 1 or summary.succeeded != 1 or summary.failed != 0:
            raise WorkerRunLifecycleSmokeTestError(
                "Worker run lifecycle smoke test failed during queued job processing"
            )
        checks_passed += 1

        run_row = store.get_run(run_id)
        if run_row is None or run_row["status"] != "succeeded":
            raise WorkerRunLifecycleSmokeTestError(
                "Worker run lifecycle smoke test failed during run status verification"
            )
        checks_passed += 1

        job_row = store.get_analysis_job(f"job_{run_id}")
        if job_row is None or job_row["status"] != "succeeded":
            raise WorkerRunLifecycleSmokeTestError(
                "Worker run lifecycle smoke test failed during job status verification"
            )
        checks_passed += 1

        report_row = store.get_report(report_id)
        if report_row is None or report_row["storage_key"] != expected_report_key:
            raise WorkerRunLifecycleSmokeTestError(
                "Worker run lifecycle smoke test failed during report metadata verification"
            )
        checks_passed += 1

        if not artifact_store.exists(org_id, expected_report_key):
            raise WorkerRunLifecycleSmokeTestError(
                "Worker run lifecycle smoke test failed during report artifact verification"
            )
        report_content = json.loads(artifact_store.get_text(org_id, expected_report_key))
        if report_content.get("total_records") != 2:
            raise WorkerRunLifecycleSmokeTestError(
                "Worker run lifecycle smoke test failed during report content verification"
            )
        checks_passed += 1

        serialized_report = json.dumps(report_content, sort_keys=True)
        if "tenants/" in serialized_report or "storage_key" in serialized_report:
            raise WorkerRunLifecycleSmokeTestError(
                "Worker run lifecycle smoke test failed during report lineage sanitization"
            )
        checks_passed += 1

        result = WorkerRunLifecycleSmokeTestResult(
            event_type=WORKER_RUN_LIFECYCLE_TEST_EVENT_TYPE,
            tested_at=tested_at,
            checks_passed=checks_passed,
            jobs_attempted=summary.attempted,
            jobs_succeeded=summary.succeeded,
            artifacts_verified=2,
            synthetic_records=15,
        )
    except Exception as exc:
        failure = exc

    try:
        _cleanup_run_lifecycle_smoke_test(
            store,
            artifact_store,
            organization_id=org_id,
            user_id=user_id,
            storage_keys=storage_keys,
        )
        if (
            store.get_organization(org_id) is not None
            or store.get_user(user_id) is not None
            or store.get_upload(upload_id) is not None
            or store.get_run(run_id) is not None
            or store.get_analysis_job(f"job_{run_id}") is not None
            or store.get_report(report_id) is not None
            or artifact_store.exists(org_id, upload_key)
            or artifact_store.exists(org_id, expected_report_key)
        ):
            raise RuntimeError("synthetic rows or artifacts remained after cleanup")
    except Exception as cleanup_exc:
        raise WorkerRunLifecycleSmokeTestError(
            "Worker run lifecycle smoke test failed during synthetic cleanup"
        ) from cleanup_exc

    if failure is not None:
        if isinstance(failure, WorkerRunLifecycleSmokeTestError):
            raise failure
        raise WorkerRunLifecycleSmokeTestError(
            "Worker run lifecycle smoke test failed during queued job execution"
        ) from failure
    if result is None:
        raise WorkerRunLifecycleSmokeTestError(
            "Worker run lifecycle smoke test failed before producing a result"
        )
    return result
