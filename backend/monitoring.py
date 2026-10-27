from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hmac
import os

from .domain import (
    CspViolationReportDirective,
    CspViolationReportDisposition,
    MarketingInquiryStatus,
    NotificationProviderEventStatus,
    NotificationStatus,
    ReportScheduleDeliveryStatus,
    RunStatus,
    UploadStatus,
)
from .operations import (
    ALERT_DELIVERY_TEST_EVENT_TYPE,
    ARTIFACT_STORE_TEST_EVENT_TYPE,
    BILLING_WEBHOOK_REPLAY_EVENT_TYPE,
    CLERK_WEBHOOK_REPLAY_EVENT_TYPE,
    DATABASE_TEST_EVENT_TYPE,
    NOTIFICATION_PROVIDER_WEBHOOK_REPLAY_EVENT_TYPE,
    REPORT_SCHEDULE_DISPATCH_EVENT_TYPE,
    RUN_LIFECYCLE_TEST_EVENT_TYPE,
    TENANT_ISOLATION_TEST_EVENT_TYPE,
    WORKER_RUN_LIFECYCLE_TEST_EVENT_TYPE,
)
from .runtime_config import PRODUCTION_RUNTIME, runtime_env_from_env
from .store import SaaSStore


METRICS_TOKEN_ENV = "RDS_METRICS_TOKEN"
METRICS_TOKEN_MIN_LENGTH = 32
PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"
ALERT_DELIVERY_TEST_METRIC_EVENT_TYPE = "alert_delivery_test"
REPORT_SCHEDULE_DISPATCH_METRIC_EVENT_TYPE = "report_schedule_dispatch"
CLERK_WEBHOOK_REPLAY_METRIC_EVENT_TYPE = "clerk_webhook_replay"
BILLING_WEBHOOK_REPLAY_METRIC_EVENT_TYPE = "billing_webhook_replay"
NOTIFICATION_PROVIDER_WEBHOOK_REPLAY_METRIC_EVENT_TYPE = "notification_provider_webhook_replay"
ARTIFACT_STORE_TEST_METRIC_EVENT_TYPE = "artifact_store_test"
DATABASE_TEST_METRIC_EVENT_TYPE = "database_test"
TENANT_ISOLATION_TEST_METRIC_EVENT_TYPE = "tenant_isolation_test"
RUN_LIFECYCLE_TEST_METRIC_EVENT_TYPE = "run_lifecycle_smoke_test"
WORKER_RUN_LIFECYCLE_TEST_METRIC_EVENT_TYPE = "worker_run_lifecycle_smoke_test"


class MetricsConfigurationError(RuntimeError):
    pass


class MetricsAuthenticationError(RuntimeError):
    pass


@dataclass(frozen=True)
class OperationalMetrics:
    organizations_total: int
    users_total: int
    users_active_total: int
    sites_total: int
    meters_total: int
    uploads_by_status: dict[str, int]
    runs_by_status: dict[str, int]
    analysis_jobs_by_status: dict[str, int]
    notifications_by_status: dict[str, int]
    notification_provider_events_by_status: dict[str, int]
    csp_violation_reports_by_disposition: dict[str, int]
    csp_violation_reports_by_effective_directive: dict[str, int]
    notification_outbox_by_phase: dict[str, int]
    oldest_due_notification_age_seconds: float | None
    report_schedules_by_phase: dict[str, int]
    oldest_due_report_schedule_age_seconds: float | None
    report_schedule_deliveries_by_status: dict[str, int]
    marketing_inquiries_by_status: dict[str, int]
    marketing_inquiry_submissions_by_outcome: dict[str, int]
    marketing_inquiry_rejections_by_reason: dict[str, int]
    worker_heartbeats_by_scope: dict[str, int]
    latest_all_tenant_worker_heartbeat_age_seconds: float | None
    alert_delivery_test_events_by_status: dict[str, int]
    latest_alert_delivery_test_success_age_seconds: float | None
    report_schedule_dispatch_events_by_status: dict[str, int]
    latest_report_schedule_dispatch_success_age_seconds: float | None
    clerk_webhook_replay_events_by_status: dict[str, int]
    billing_webhook_replay_events_by_status: dict[str, int]
    notification_provider_webhook_replay_events_by_status: dict[str, int]
    artifact_store_test_events_by_status: dict[str, int]
    latest_artifact_store_test_success_age_seconds: float | None
    database_test_events_by_status: dict[str, int]
    latest_database_test_success_age_seconds: float | None
    tenant_isolation_test_events_by_status: dict[str, int]
    latest_tenant_isolation_test_success_age_seconds: float | None
    run_lifecycle_test_events_by_status: dict[str, int]
    latest_run_lifecycle_test_success_age_seconds: float | None
    worker_run_lifecycle_test_events_by_status: dict[str, int]
    latest_worker_run_lifecycle_test_success_age_seconds: float | None


def _configured_metrics_token() -> str:
    return os.environ.get(METRICS_TOKEN_ENV, "").strip()


def metrics_configuration_status(runtime_env: str | None = None) -> tuple[bool, str]:
    active_runtime_env = runtime_env or runtime_env_from_env()
    token = _configured_metrics_token()
    if token:
        if active_runtime_env == PRODUCTION_RUNTIME and len(token) < METRICS_TOKEN_MIN_LENGTH:
            return (
                False,
                f"{METRICS_TOKEN_ENV} must be at least {METRICS_TOKEN_MIN_LENGTH} "
                "characters in production",
            )
        return True, f"{METRICS_TOKEN_ENV} configured"
    if active_runtime_env == PRODUCTION_RUNTIME:
        return (
            False,
            f"{METRICS_TOKEN_ENV} must be configured when RDS_RUNTIME_ENV=production",
        )
    return True, f"{METRICS_TOKEN_ENV} is optional in development"


def verify_metrics_request(
    authorization: str | None,
    runtime_env: str | None = None,
) -> None:
    configured, detail = metrics_configuration_status(runtime_env)
    if not configured:
        raise MetricsConfigurationError(detail)

    token = _configured_metrics_token()
    if not token:
        return

    scheme, separator, presented_token = (authorization or "").partition(" ")
    if (
        separator != " "
        or scheme.lower() != "bearer"
        or not hmac.compare_digest(presented_token.strip(), token)
    ):
        raise MetricsAuthenticationError("A valid metrics bearer token is required")


def _parse_observed_at(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _latest_all_tenant_worker_heartbeat_age_seconds(
    store: SaaSStore,
    now: datetime,
) -> float | None:
    heartbeat = store.get_latest_worker_heartbeat()
    if heartbeat is None:
        return None
    try:
        observed_at = _parse_observed_at(heartbeat["last_seen_at"])
    except (TypeError, ValueError):
        return None
    return max(0.0, (now - observed_at).total_seconds())


def _oldest_due_notification_age_seconds(
    store: SaaSStore,
    now: datetime,
) -> float | None:
    created_at = store.get_oldest_due_notification_created_at(now.isoformat())
    if created_at is None:
        return None
    try:
        observed_at = _parse_observed_at(created_at)
    except (TypeError, ValueError):
        return None
    return max(0.0, (now - observed_at).total_seconds())


def _oldest_due_report_schedule_age_seconds(
    store: SaaSStore,
    now: datetime,
) -> float | None:
    next_run_at = store.get_oldest_due_report_schedule_next_run_at(now.isoformat())
    if next_run_at is None:
        return None
    try:
        observed_at = _parse_observed_at(next_run_at)
    except (TypeError, ValueError):
        return None
    return max(0.0, (now - observed_at).total_seconds())


def _latest_alert_delivery_test_success_age_seconds(
    store: SaaSStore,
    now: datetime,
) -> float | None:
    created_at = store.get_latest_platform_operation_event_created_at(
        event_type=ALERT_DELIVERY_TEST_EVENT_TYPE,
        event_status="succeeded",
    )
    if created_at is None:
        return None
    try:
        observed_at = _parse_observed_at(created_at)
    except (TypeError, ValueError):
        return None
    return max(0.0, (now - observed_at).total_seconds())


def _latest_report_schedule_dispatch_success_age_seconds(
    store: SaaSStore,
    now: datetime,
) -> float | None:
    created_at = store.get_latest_platform_operation_event_created_at(
        event_type=REPORT_SCHEDULE_DISPATCH_EVENT_TYPE,
        event_status="succeeded",
    )
    if created_at is None:
        return None
    try:
        observed_at = _parse_observed_at(created_at)
    except (TypeError, ValueError):
        return None
    return max(0.0, (now - observed_at).total_seconds())


def _latest_artifact_store_test_success_age_seconds(
    store: SaaSStore,
    now: datetime,
) -> float | None:
    created_at = store.get_latest_platform_operation_event_created_at(
        event_type=ARTIFACT_STORE_TEST_EVENT_TYPE,
        event_status="succeeded",
    )
    if created_at is None:
        return None
    try:
        observed_at = _parse_observed_at(created_at)
    except (TypeError, ValueError):
        return None
    return max(0.0, (now - observed_at).total_seconds())


def _latest_database_test_success_age_seconds(
    store: SaaSStore,
    now: datetime,
) -> float | None:
    created_at = store.get_latest_platform_operation_event_created_at(
        event_type=DATABASE_TEST_EVENT_TYPE,
        event_status="succeeded",
    )
    if created_at is None:
        return None
    try:
        observed_at = _parse_observed_at(created_at)
    except (TypeError, ValueError):
        return None
    return max(0.0, (now - observed_at).total_seconds())


def _latest_tenant_isolation_test_success_age_seconds(
    store: SaaSStore,
    now: datetime,
) -> float | None:
    created_at = store.get_latest_platform_operation_event_created_at(
        event_type=TENANT_ISOLATION_TEST_EVENT_TYPE,
        event_status="succeeded",
    )
    if created_at is None:
        return None
    try:
        observed_at = _parse_observed_at(created_at)
    except (TypeError, ValueError):
        return None
    return max(0.0, (now - observed_at).total_seconds())


def _latest_run_lifecycle_test_success_age_seconds(
    store: SaaSStore,
    now: datetime,
) -> float | None:
    created_at = store.get_latest_platform_operation_event_created_at(
        event_type=RUN_LIFECYCLE_TEST_EVENT_TYPE,
        event_status="succeeded",
    )
    if created_at is None:
        return None
    try:
        observed_at = _parse_observed_at(created_at)
    except (TypeError, ValueError):
        return None
    return max(0.0, (now - observed_at).total_seconds())


def _latest_worker_run_lifecycle_test_success_age_seconds(
    store: SaaSStore,
    now: datetime,
) -> float | None:
    created_at = store.get_latest_platform_operation_event_created_at(
        event_type=WORKER_RUN_LIFECYCLE_TEST_EVENT_TYPE,
        event_status="succeeded",
    )
    if created_at is None:
        return None
    try:
        observed_at = _parse_observed_at(created_at)
    except (TypeError, ValueError):
        return None
    return max(0.0, (now - observed_at).total_seconds())


def collect_operational_metrics(
    store: SaaSStore,
    *,
    now: datetime | None = None,
) -> OperationalMetrics:
    observed_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return OperationalMetrics(
        organizations_total=store.count_organizations(),
        users_total=store.count_users(),
        users_active_total=store.count_users(active=True),
        sites_total=store.count_sites(),
        meters_total=store.count_meters(),
        uploads_by_status=store.count_uploads_by_status(),
        runs_by_status=store.count_runs_by_status(),
        analysis_jobs_by_status=store.count_analysis_jobs_by_status(),
        notifications_by_status=store.count_notifications_by_status(),
        notification_provider_events_by_status=(
            store.count_notification_provider_events_by_status()
        ),
        csp_violation_reports_by_disposition=(
            store.count_csp_violation_reports_by_disposition()
        ),
        csp_violation_reports_by_effective_directive=(
            store.count_csp_violation_reports_by_effective_directive()
        ),
        notification_outbox_by_phase=store.count_notification_outbox_by_phase(
            observed_at.isoformat(),
        ),
        oldest_due_notification_age_seconds=_oldest_due_notification_age_seconds(
            store,
            observed_at,
        ),
        report_schedules_by_phase=store.count_report_schedules_by_phase(
            observed_at.isoformat(),
        ),
        oldest_due_report_schedule_age_seconds=_oldest_due_report_schedule_age_seconds(
            store,
            observed_at,
        ),
        report_schedule_deliveries_by_status=(
            store.count_report_schedule_deliveries_by_status()
        ),
        marketing_inquiries_by_status=store.count_marketing_inquiries_by_status(),
        marketing_inquiry_submissions_by_outcome=(
            store.count_marketing_inquiry_submissions_by_outcome()
        ),
        marketing_inquiry_rejections_by_reason=(
            store.count_marketing_inquiry_submissions_by_rejection_reason()
        ),
        worker_heartbeats_by_scope=store.count_worker_heartbeats_by_scope(),
        latest_all_tenant_worker_heartbeat_age_seconds=(
            _latest_all_tenant_worker_heartbeat_age_seconds(store, observed_at)
        ),
        alert_delivery_test_events_by_status=store.count_platform_operation_events_by_status(
            ALERT_DELIVERY_TEST_EVENT_TYPE,
        ),
        latest_alert_delivery_test_success_age_seconds=(
            _latest_alert_delivery_test_success_age_seconds(store, observed_at)
        ),
        report_schedule_dispatch_events_by_status=(
            store.count_platform_operation_events_by_status(
                REPORT_SCHEDULE_DISPATCH_EVENT_TYPE,
            )
        ),
        latest_report_schedule_dispatch_success_age_seconds=(
            _latest_report_schedule_dispatch_success_age_seconds(store, observed_at)
        ),
        clerk_webhook_replay_events_by_status=store.count_platform_operation_events_by_status(
            CLERK_WEBHOOK_REPLAY_EVENT_TYPE,
        ),
        billing_webhook_replay_events_by_status=store.count_platform_operation_events_by_status(
            BILLING_WEBHOOK_REPLAY_EVENT_TYPE,
        ),
        notification_provider_webhook_replay_events_by_status=(
            store.count_platform_operation_events_by_status(
                NOTIFICATION_PROVIDER_WEBHOOK_REPLAY_EVENT_TYPE,
            )
        ),
        artifact_store_test_events_by_status=store.count_platform_operation_events_by_status(
            ARTIFACT_STORE_TEST_EVENT_TYPE,
        ),
        latest_artifact_store_test_success_age_seconds=(
            _latest_artifact_store_test_success_age_seconds(store, observed_at)
        ),
        database_test_events_by_status=store.count_platform_operation_events_by_status(
            DATABASE_TEST_EVENT_TYPE,
        ),
        latest_database_test_success_age_seconds=(
            _latest_database_test_success_age_seconds(store, observed_at)
        ),
        tenant_isolation_test_events_by_status=store.count_platform_operation_events_by_status(
            TENANT_ISOLATION_TEST_EVENT_TYPE,
        ),
        latest_tenant_isolation_test_success_age_seconds=(
            _latest_tenant_isolation_test_success_age_seconds(store, observed_at)
        ),
        run_lifecycle_test_events_by_status=store.count_platform_operation_events_by_status(
            RUN_LIFECYCLE_TEST_EVENT_TYPE,
        ),
        latest_run_lifecycle_test_success_age_seconds=(
            _latest_run_lifecycle_test_success_age_seconds(store, observed_at)
        ),
        worker_run_lifecycle_test_events_by_status=(
            store.count_platform_operation_events_by_status(
                WORKER_RUN_LIFECYCLE_TEST_EVENT_TYPE,
            )
        ),
        latest_worker_run_lifecycle_test_success_age_seconds=(
            _latest_worker_run_lifecycle_test_success_age_seconds(store, observed_at)
        ),
    )


def _metric_header(name: str, help_text: str) -> list[str]:
    return [
        f"# HELP {name} {help_text}",
        f"# TYPE {name} gauge",
    ]


def _status_metric_lines(
    metric_name: str,
    help_text: str,
    statuses: tuple[str, ...],
    counts: dict[str, int],
) -> list[str]:
    lines = _metric_header(metric_name, help_text)
    for status in statuses:
        lines.append(f'{metric_name}{{status="{status}"}} {counts.get(status, 0)}')
    return lines


def _outcome_metric_lines(
    metric_name: str,
    help_text: str,
    outcomes: tuple[str, ...],
    counts: dict[str, int],
) -> list[str]:
    lines = _metric_header(metric_name, help_text)
    for outcome in outcomes:
        lines.append(f'{metric_name}{{outcome="{outcome}"}} {counts.get(outcome, 0)}')
    return lines


def _reason_metric_lines(
    metric_name: str,
    help_text: str,
    reasons: tuple[str, ...],
    counts: dict[str, int],
) -> list[str]:
    lines = _metric_header(metric_name, help_text)
    for reason in reasons:
        lines.append(f'{metric_name}{{reason="{reason}"}} {counts.get(reason, 0)}')
    return lines


def _labeled_metric_lines(
    metric_name: str,
    help_text: str,
    label_name: str,
    label_values: tuple[str, ...],
    counts: dict[str, int],
) -> list[str]:
    lines = _metric_header(metric_name, help_text)
    for label_value in label_values:
        lines.append(f'{metric_name}{{{label_name}="{label_value}"}} {counts.get(label_value, 0)}')
    return lines


def format_prometheus_metrics(metrics: OperationalMetrics) -> str:
    lines: list[str] = []
    scalar_metrics = [
        (
            "rds_organizations_total",
            "Total tenant organizations.",
            metrics.organizations_total,
        ),
        ("rds_users_total", "Total user accounts.", metrics.users_total),
        ("rds_users_active_total", "Total active user accounts.", metrics.users_active_total),
        ("rds_sites_total", "Total tenant sites.", metrics.sites_total),
        ("rds_meters_total", "Total tenant meters.", metrics.meters_total),
    ]

    for metric_name, help_text, value in scalar_metrics:
        lines.extend(_metric_header(metric_name, help_text))
        lines.append(f"{metric_name} {value}")

    # Labels intentionally use finite lifecycle enums only; tenant identifiers are never emitted.
    lines.extend(
        _status_metric_lines(
            "rds_uploads_total",
            "Tenant uploads by storage status.",
            tuple(status.value for status in UploadStatus),
            metrics.uploads_by_status,
        )
    )
    lines.extend(
        _status_metric_lines(
            "rds_runs_total",
            "Analysis runs by lifecycle status.",
            tuple(status.value for status in RunStatus),
            metrics.runs_by_status,
        )
    )
    lines.extend(
        _status_metric_lines(
            "rds_analysis_jobs_total",
            "Background analysis jobs by lifecycle status.",
            tuple(status.value for status in RunStatus),
            metrics.analysis_jobs_by_status,
        )
    )
    lines.extend(
        _status_metric_lines(
            "rds_notifications_total",
            "Notification outbox records by delivery status.",
            tuple(status.value for status in NotificationStatus),
            metrics.notifications_by_status,
        )
    )
    lines.extend(
        _status_metric_lines(
            "rds_notification_provider_events_total",
            "External notification provider events by delivery status.",
            tuple(status.value for status in NotificationProviderEventStatus),
            metrics.notification_provider_events_by_status,
        )
    )
    lines.extend(
        _labeled_metric_lines(
            "rds_csp_violation_reports_total",
            "Content Security Policy violation reports by enforcement disposition.",
            "disposition",
            tuple(disposition.value for disposition in CspViolationReportDisposition),
            metrics.csp_violation_reports_by_disposition,
        )
    )
    lines.extend(
        _labeled_metric_lines(
            "rds_csp_violation_reports_by_directive_total",
            "Content Security Policy violation reports by sanitized effective directive.",
            "directive",
            tuple(directive.value for directive in CspViolationReportDirective),
            metrics.csp_violation_reports_by_effective_directive,
        )
    )
    lines.extend(
        _labeled_metric_lines(
            "rds_notification_outbox_total",
            "Notification outbox records by tenant-safe delivery phase.",
            "phase",
            ("pending_due", "pending_retry_scheduled", "failed"),
            metrics.notification_outbox_by_phase,
        )
    )
    oldest_due_notification_present = (
        1 if metrics.oldest_due_notification_age_seconds is not None else 0
    )
    oldest_due_notification_age = metrics.oldest_due_notification_age_seconds or 0
    lines.extend(
        _metric_header(
            "rds_notification_oldest_due_present",
            "Whether at least one due notification is waiting for delivery.",
        )
    )
    lines.append(f"rds_notification_oldest_due_present {oldest_due_notification_present}")
    lines.extend(
        _metric_header(
            "rds_notification_oldest_due_age_seconds",
            "Age in seconds of the oldest due notification, or zero when absent.",
        )
    )
    lines.append(f"rds_notification_oldest_due_age_seconds {oldest_due_notification_age:g}")
    lines.extend(
        _labeled_metric_lines(
            "rds_report_schedules_total",
            "Report schedules by tenant-safe execution phase.",
            "phase",
            ("active_due", "active_future", "paused", "failed_last_delivery"),
            metrics.report_schedules_by_phase,
        )
    )
    oldest_due_schedule_present = (
        1 if metrics.oldest_due_report_schedule_age_seconds is not None else 0
    )
    oldest_due_schedule_age = metrics.oldest_due_report_schedule_age_seconds or 0
    lines.extend(
        _metric_header(
            "rds_report_schedule_oldest_due_present",
            "Whether at least one active report schedule is due for delivery.",
        )
    )
    lines.append(f"rds_report_schedule_oldest_due_present {oldest_due_schedule_present}")
    lines.extend(
        _metric_header(
            "rds_report_schedule_oldest_due_age_seconds",
            "Age in seconds of the oldest due report schedule, or zero when absent.",
        )
    )
    lines.append(f"rds_report_schedule_oldest_due_age_seconds {oldest_due_schedule_age:g}")
    lines.extend(
        _status_metric_lines(
            "rds_report_schedule_deliveries_total",
            "Report schedule delivery attempts by status.",
            tuple(status.value for status in ReportScheduleDeliveryStatus),
            metrics.report_schedule_deliveries_by_status,
        )
    )
    lines.extend(
        _status_metric_lines(
            "rds_marketing_inquiries_total",
            "Public pilot inquiries by support lifecycle status.",
            tuple(status.value for status in MarketingInquiryStatus),
            metrics.marketing_inquiries_by_status,
        )
    )
    lines.extend(
        _outcome_metric_lines(
            "rds_marketing_inquiry_submissions_total",
            "Public pilot inquiry submissions by sanitized rate-limit outcome.",
            ("accepted", "rate_limited"),
            metrics.marketing_inquiry_submissions_by_outcome,
        )
    )
    lines.extend(
        _reason_metric_lines(
            "rds_marketing_inquiry_rejections_total",
            "Public pilot inquiry rejections by sanitized reason.",
            ("source_rate_limit",),
            metrics.marketing_inquiry_rejections_by_reason,
        )
    )
    lines.extend(
        _labeled_metric_lines(
            "rds_worker_heartbeats_total",
            "Worker heartbeat records by tenant-safe processing scope.",
            "scope",
            ("all", "organization"),
            metrics.worker_heartbeats_by_scope,
        )
    )
    worker_heartbeat_present = (
        1 if metrics.latest_all_tenant_worker_heartbeat_age_seconds is not None else 0
    )
    worker_heartbeat_age = metrics.latest_all_tenant_worker_heartbeat_age_seconds or 0
    lines.extend(
        _metric_header(
            "rds_worker_latest_all_tenant_heartbeat_present",
            "Whether an all-tenant worker heartbeat has been recorded.",
        )
    )
    lines.append(f"rds_worker_latest_all_tenant_heartbeat_present {worker_heartbeat_present}")
    lines.extend(
        _metric_header(
            "rds_worker_latest_all_tenant_heartbeat_age_seconds",
            "Age in seconds of the latest all-tenant worker heartbeat, or zero when absent.",
        )
    )
    lines.append(f"rds_worker_latest_all_tenant_heartbeat_age_seconds {worker_heartbeat_age:g}")
    lines.extend(
        _metric_header(
            "rds_platform_operation_events_total",
            "Platform operation evidence events by finite type and status.",
        )
    )
    platform_operation_counts = (
        (
            ALERT_DELIVERY_TEST_METRIC_EVENT_TYPE,
            metrics.alert_delivery_test_events_by_status,
        ),
        (
            REPORT_SCHEDULE_DISPATCH_METRIC_EVENT_TYPE,
            metrics.report_schedule_dispatch_events_by_status,
        ),
        (
            CLERK_WEBHOOK_REPLAY_METRIC_EVENT_TYPE,
            metrics.clerk_webhook_replay_events_by_status,
        ),
        (
            BILLING_WEBHOOK_REPLAY_METRIC_EVENT_TYPE,
            metrics.billing_webhook_replay_events_by_status,
        ),
        (
            NOTIFICATION_PROVIDER_WEBHOOK_REPLAY_METRIC_EVENT_TYPE,
            metrics.notification_provider_webhook_replay_events_by_status,
        ),
        (
            ARTIFACT_STORE_TEST_METRIC_EVENT_TYPE,
            metrics.artifact_store_test_events_by_status,
        ),
        (
            DATABASE_TEST_METRIC_EVENT_TYPE,
            metrics.database_test_events_by_status,
        ),
        (
            TENANT_ISOLATION_TEST_METRIC_EVENT_TYPE,
            metrics.tenant_isolation_test_events_by_status,
        ),
        (
            RUN_LIFECYCLE_TEST_METRIC_EVENT_TYPE,
            metrics.run_lifecycle_test_events_by_status,
        ),
        (
            WORKER_RUN_LIFECYCLE_TEST_METRIC_EVENT_TYPE,
            metrics.worker_run_lifecycle_test_events_by_status,
        ),
    )
    for event_type, counts in platform_operation_counts:
        for status in ("succeeded", "failed"):
            lines.append(
                "rds_platform_operation_events_total"
                f'{{event_type="{event_type}",status="{status}"}} {counts.get(status, 0)}'
            )
    latest_alert_success_present = (
        1 if metrics.latest_alert_delivery_test_success_age_seconds is not None else 0
    )
    latest_alert_success_age = metrics.latest_alert_delivery_test_success_age_seconds or 0
    lines.extend(
        _metric_header(
            "rds_platform_alert_delivery_latest_success_present",
            "Whether a successful alert delivery smoke test has been recorded.",
        )
    )
    lines.append(
        "rds_platform_alert_delivery_latest_success_present "
        f"{latest_alert_success_present}"
    )
    lines.extend(
        _metric_header(
            "rds_platform_alert_delivery_latest_success_age_seconds",
            "Age in seconds of the latest successful alert delivery smoke test, or zero when absent.",
        )
    )
    lines.append(
        "rds_platform_alert_delivery_latest_success_age_seconds "
        f"{latest_alert_success_age:g}"
    )
    latest_schedule_dispatch_success_present = (
        1 if metrics.latest_report_schedule_dispatch_success_age_seconds is not None else 0
    )
    latest_schedule_dispatch_success_age = (
        metrics.latest_report_schedule_dispatch_success_age_seconds or 0
    )
    lines.extend(
        _metric_header(
            "rds_platform_report_schedule_dispatch_latest_success_present",
            "Whether a successful scheduled report dispatch batch has been recorded.",
        )
    )
    lines.append(
        "rds_platform_report_schedule_dispatch_latest_success_present "
        f"{latest_schedule_dispatch_success_present}"
    )
    lines.extend(
        _metric_header(
            "rds_platform_report_schedule_dispatch_latest_success_age_seconds",
            "Age in seconds of the latest successful scheduled report dispatch batch, or zero when absent.",
        )
    )
    lines.append(
        "rds_platform_report_schedule_dispatch_latest_success_age_seconds "
        f"{latest_schedule_dispatch_success_age:g}"
    )
    latest_artifact_store_success_present = (
        1 if metrics.latest_artifact_store_test_success_age_seconds is not None else 0
    )
    latest_artifact_store_success_age = (
        metrics.latest_artifact_store_test_success_age_seconds or 0
    )
    lines.extend(
        _metric_header(
            "rds_platform_artifact_store_test_latest_success_present",
            "Whether a successful artifact store smoke test has been recorded.",
        )
    )
    lines.append(
        "rds_platform_artifact_store_test_latest_success_present "
        f"{latest_artifact_store_success_present}"
    )
    lines.extend(
        _metric_header(
            "rds_platform_artifact_store_test_latest_success_age_seconds",
            "Age in seconds of the latest successful artifact store smoke test, or zero when absent.",
        )
    )
    lines.append(
        "rds_platform_artifact_store_test_latest_success_age_seconds "
        f"{latest_artifact_store_success_age:g}"
    )
    latest_database_success_present = (
        1 if metrics.latest_database_test_success_age_seconds is not None else 0
    )
    latest_database_success_age = metrics.latest_database_test_success_age_seconds or 0
    lines.extend(
        _metric_header(
            "rds_platform_database_test_latest_success_present",
            "Whether a successful database smoke test has been recorded.",
        )
    )
    lines.append(
        "rds_platform_database_test_latest_success_present "
        f"{latest_database_success_present}"
    )
    lines.extend(
        _metric_header(
            "rds_platform_database_test_latest_success_age_seconds",
            "Age in seconds of the latest successful database smoke test, or zero when absent.",
        )
    )
    lines.append(
        "rds_platform_database_test_latest_success_age_seconds "
        f"{latest_database_success_age:g}"
    )
    latest_tenant_isolation_success_present = (
        1 if metrics.latest_tenant_isolation_test_success_age_seconds is not None else 0
    )
    latest_tenant_isolation_success_age = (
        metrics.latest_tenant_isolation_test_success_age_seconds or 0
    )
    lines.extend(
        _metric_header(
            "rds_platform_tenant_isolation_test_latest_success_present",
            "Whether a successful tenant isolation smoke test has been recorded.",
        )
    )
    lines.append(
        "rds_platform_tenant_isolation_test_latest_success_present "
        f"{latest_tenant_isolation_success_present}"
    )
    lines.extend(
        _metric_header(
            "rds_platform_tenant_isolation_test_latest_success_age_seconds",
            "Age in seconds of the latest successful tenant isolation smoke test, or zero when absent.",
        )
    )
    lines.append(
        "rds_platform_tenant_isolation_test_latest_success_age_seconds "
        f"{latest_tenant_isolation_success_age:g}"
    )
    latest_run_lifecycle_success_present = (
        1 if metrics.latest_run_lifecycle_test_success_age_seconds is not None else 0
    )
    latest_run_lifecycle_success_age = (
        metrics.latest_run_lifecycle_test_success_age_seconds or 0
    )
    lines.extend(
        _metric_header(
            "rds_platform_run_lifecycle_test_latest_success_present",
            "Whether a successful run lifecycle smoke test has been recorded.",
        )
    )
    lines.append(
        "rds_platform_run_lifecycle_test_latest_success_present "
        f"{latest_run_lifecycle_success_present}"
    )
    lines.extend(
        _metric_header(
            "rds_platform_run_lifecycle_test_latest_success_age_seconds",
            "Age in seconds of the latest successful run lifecycle smoke test, or zero when absent.",
        )
    )
    lines.append(
        "rds_platform_run_lifecycle_test_latest_success_age_seconds "
        f"{latest_run_lifecycle_success_age:g}"
    )
    latest_worker_run_lifecycle_success_present = (
        1 if metrics.latest_worker_run_lifecycle_test_success_age_seconds is not None else 0
    )
    latest_worker_run_lifecycle_success_age = (
        metrics.latest_worker_run_lifecycle_test_success_age_seconds or 0
    )
    lines.extend(
        _metric_header(
            "rds_platform_worker_run_lifecycle_test_latest_success_present",
            "Whether a successful worker run lifecycle smoke test has been recorded.",
        )
    )
    lines.append(
        "rds_platform_worker_run_lifecycle_test_latest_success_present "
        f"{latest_worker_run_lifecycle_success_present}"
    )
    lines.extend(
        _metric_header(
            "rds_platform_worker_run_lifecycle_test_latest_success_age_seconds",
            (
                "Age in seconds of the latest successful worker run lifecycle smoke "
                "test, or zero when absent."
            ),
        )
    )
    lines.append(
        "rds_platform_worker_run_lifecycle_test_latest_success_age_seconds "
        f"{latest_worker_run_lifecycle_success_age:g}"
    )
    return "\n".join(lines) + "\n"
