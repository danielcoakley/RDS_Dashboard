from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
import time
from typing import TYPE_CHECKING
from urllib.parse import urlparse
from uuid import uuid4

from .artifacts import artifact_store_from_env
from .artifacts import TenantArtifactStore
from .database import initialize_database_from_env
from .notifications import EmailSender, email_sender_from_env
from .operations import REPORT_SCHEDULE_DISPATCH_EVENT_TYPE
from .report_scheduling import process_due_report_schedules
from .runtime_config import PRODUCTION_RUNTIME, RUNTIME_ENV_ENV
from .store import SaaSStore
from .workers import (
    AnalysisWorkerSummary,
    process_due_data_connector_imports,
    process_queued_analysis_jobs,
)

if TYPE_CHECKING:
    from .data_connector_sync import DataConnectorProvider


WORKER_POLL_SECONDS_ENV = "SAAS_WORKER_POLL_SECONDS"
WORKER_BATCH_SIZE_ENV = "SAAS_WORKER_BATCH_SIZE"
WORKER_ORGANIZATION_ID_ENV = "SAAS_WORKER_ORGANIZATION_ID"
WORKER_RUN_ONCE_ENV = "SAAS_WORKER_RUN_ONCE"
WORKER_ENABLED_ENV = "SAAS_WORKER_ENABLED"
WORKER_ID_ENV = "SAAS_WORKER_ID"
WORKER_HEARTBEAT_MAX_AGE_SECONDS_ENV = "SAAS_WORKER_HEARTBEAT_MAX_AGE_SECONDS"
WORKER_SUPERVISOR_URL_ENV = "SAAS_WORKER_SUPERVISOR_URL"
WORKER_SUPERVISOR_VERIFIED_AT_ENV = "SAAS_WORKER_SUPERVISOR_VERIFIED_AT"
WORKER_SUPERVISOR_MAX_AGE_DAYS_ENV = "SAAS_WORKER_SUPERVISOR_MAX_AGE_DAYS"
DEFAULT_WORKER_ID = "default"
DEFAULT_WORKER_HEARTBEAT_MAX_AGE_SECONDS = 120.0
DEFAULT_WORKER_SUPERVISOR_MAX_AGE_DAYS = 30


@dataclass(frozen=True)
class AnalysisWorkerConfig:
    poll_seconds: float = 10.0
    batch_size: int = 10
    organization_id: str | None = None
    worker_id: str = DEFAULT_WORKER_ID


def _optional_env_text(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _float_env(name: str, default: float) -> float:
    value = _optional_env_text(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


def _int_env(name: str, default: int) -> int:
    value = _optional_env_text(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be greater than zero")
    return parsed


def _optional_bool_env(name: str) -> bool | None:
    value = _optional_env_text(name)
    if value is None:
        return None
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def _positive_int_env(name: str, default: int) -> int:
    value = _optional_env_text(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _require_https_url(name: str, runtime_env: str) -> str | None:
    value = _optional_env_text(name)
    if value is None:
        if runtime_env == PRODUCTION_RUNTIME:
            raise ValueError(f"{name} is required in production")
        return None

    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{name} must be a URL")
    if runtime_env == PRODUCTION_RUNTIME and parsed.scheme != "https":
        raise ValueError(f"{name} must use https in production")
    return value


def worker_config_from_env() -> AnalysisWorkerConfig:
    return AnalysisWorkerConfig(
        poll_seconds=_float_env(WORKER_POLL_SECONDS_ENV, 10.0),
        batch_size=_int_env(WORKER_BATCH_SIZE_ENV, 10),
        organization_id=_optional_env_text(WORKER_ORGANIZATION_ID_ENV),
        worker_id=_optional_env_text(WORKER_ID_ENV) or DEFAULT_WORKER_ID,
    )


def run_once_from_env() -> bool:
    return _optional_bool_env(WORKER_RUN_ONCE_ENV) is True


def worker_heartbeat_max_age_seconds_from_env() -> float:
    parsed = _float_env(
        WORKER_HEARTBEAT_MAX_AGE_SECONDS_ENV,
        DEFAULT_WORKER_HEARTBEAT_MAX_AGE_SECONDS,
    )
    if parsed <= 0:
        raise ValueError(f"{WORKER_HEARTBEAT_MAX_AGE_SECONDS_ENV} must be greater than zero")
    return parsed


def worker_supervisor_max_age_days_from_env() -> int:
    return _positive_int_env(
        WORKER_SUPERVISOR_MAX_AGE_DAYS_ENV,
        DEFAULT_WORKER_SUPERVISOR_MAX_AGE_DAYS,
    )


def _parse_supervisor_verified_at(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"{WORKER_SUPERVISOR_VERIFIED_AT_ENV} must be an ISO-8601 datetime"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{WORKER_SUPERVISOR_VERIFIED_AT_ENV} must include a timezone")
    return parsed.astimezone(timezone.utc)


def worker_configuration_status(runtime_env: str) -> tuple[bool, str]:
    try:
        config = worker_config_from_env()
        enabled = _optional_bool_env(WORKER_ENABLED_ENV)
        run_once = run_once_from_env()
        heartbeat_max_age_seconds = worker_heartbeat_max_age_seconds_from_env()
    except ValueError as exc:
        return False, str(exc)

    if runtime_env == PRODUCTION_RUNTIME:
        if enabled is not True:
            return (
                False,
                f"{WORKER_ENABLED_ENV}=true is required when {RUNTIME_ENV_ENV}=production",
            )
        if run_once:
            return (
                False,
                f"{WORKER_RUN_ONCE_ENV} must be false or unset for production worker supervision",
            )
        if config.organization_id is not None:
            return (
                False,
                f"{WORKER_ORGANIZATION_ID_ENV} must be unset in production so all tenants are processed",
            )
        if config.poll_seconds <= 0:
            return (
                False,
                f"{WORKER_POLL_SECONDS_ENV} must be greater than zero in production",
            )

    scope = config.organization_id if config.organization_id is not None else "all organizations"
    enabled_detail = "enabled" if enabled is True else "not asserted"
    return (
        True,
        "worker configuration valid: "
        f"{enabled_detail}, worker_id={config.worker_id}, "
        f"poll_seconds={config.poll_seconds:g}, batch_size={config.batch_size}, "
        f"scope={scope}, heartbeat_max_age_seconds={heartbeat_max_age_seconds:g}",
    )


def worker_supervision_status(
    runtime_env: str,
    *,
    now: datetime | None = None,
) -> tuple[bool, str]:
    try:
        supervisor_url = _require_https_url(WORKER_SUPERVISOR_URL_ENV, runtime_env)
        max_age_days = worker_supervisor_max_age_days_from_env()
    except ValueError as exc:
        return False, str(exc)

    verified_at = _optional_env_text(WORKER_SUPERVISOR_VERIFIED_AT_ENV)
    if runtime_env == PRODUCTION_RUNTIME and verified_at is None:
        return False, f"{WORKER_SUPERVISOR_VERIFIED_AT_ENV} is required in production"
    if verified_at is None:
        return True, "Worker supervisor evidence is optional in development"

    try:
        observed_at = _parse_supervisor_verified_at(verified_at)
    except ValueError as exc:
        return False, str(exc)

    checked_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if observed_at > checked_at:
        return False, f"{WORKER_SUPERVISOR_VERIFIED_AT_ENV} cannot be in the future"
    age_days = (checked_at - observed_at).days
    if runtime_env == PRODUCTION_RUNTIME and age_days > max_age_days:
        return False, f"Worker supervisor verification is older than {max_age_days} days"

    detail = f"worker supervisor verified age {age_days} day(s)"
    if supervisor_url:
        detail = f"{WORKER_SUPERVISOR_URL_ENV} configured; {detail}"
    return True, detail


def _parse_heartbeat_timestamp(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def record_worker_heartbeat(
    store: SaaSStore,
    config: AnalysisWorkerConfig,
    summary: AnalysisWorkerSummary,
    *,
    observed_at: datetime | None = None,
) -> None:
    timestamp = (observed_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    store.upsert_worker_heartbeat(
        worker_id=config.worker_id,
        organization_id=config.organization_id,
        poll_seconds=config.poll_seconds,
        batch_size=config.batch_size,
        last_attempted=summary.attempted,
        last_succeeded=summary.succeeded,
        last_failed=summary.failed,
        last_skipped=summary.skipped,
        last_scheduled_attempted=summary.scheduled_attempted,
        last_scheduled_sent=summary.scheduled_sent,
        last_scheduled_failed=summary.scheduled_failed,
        last_scheduled_skipped=summary.scheduled_skipped,
        last_connector_attempted=summary.connector_attempted,
        last_connector_succeeded=summary.connector_succeeded,
        last_connector_failed=summary.connector_failed,
        last_connector_skipped=summary.connector_skipped,
        last_seen_at=timestamp.isoformat(),
    )


def record_report_schedule_dispatch_event(
    store: SaaSStore,
    config: AnalysisWorkerConfig,
    summary: AnalysisWorkerSummary,
    *,
    observed_at: datetime | None = None,
) -> str | None:
    if summary.scheduled_attempted <= 0:
        return None

    timestamp = (observed_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    event_status = "failed" if summary.scheduled_failed > 0 else "succeeded"
    scope = "organization" if config.organization_id else "all"
    detail = (
        f"scope={scope} "
        f"scheduled_attempted={summary.scheduled_attempted} "
        f"scheduled_sent={summary.scheduled_sent} "
        f"scheduled_failed={summary.scheduled_failed} "
        f"scheduled_skipped={summary.scheduled_skipped}"
    )
    event_id = f"platform_operation_report_schedule_dispatch_{uuid4().hex}"
    store.create_platform_operation_event(
        event_id=event_id,
        event_type=REPORT_SCHEDULE_DISPATCH_EVENT_TYPE,
        event_status=event_status,
        request_id=None,
        detail=detail,
        created_at=timestamp.isoformat(),
    )
    return event_id


def worker_liveness_status(
    store: SaaSStore,
    runtime_env: str,
    *,
    now: datetime | None = None,
) -> tuple[bool, str]:
    configured, configuration_detail = worker_configuration_status(runtime_env)
    if not configured:
        return False, configuration_detail
    if runtime_env != PRODUCTION_RUNTIME:
        return True, configuration_detail

    try:
        heartbeat_max_age_seconds = worker_heartbeat_max_age_seconds_from_env()
        heartbeat = store.get_latest_worker_heartbeat()
    except Exception as exc:
        return False, f"Worker heartbeat check failed: {exc}"

    if heartbeat is None:
        return (
            False,
            "No all-tenant worker heartbeat recorded; start the long-running worker "
            "against the production metadata database",
        )

    checked_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    try:
        last_seen_at = _parse_heartbeat_timestamp(heartbeat["last_seen_at"])
    except (TypeError, ValueError) as exc:
        return False, f"Worker heartbeat timestamp is invalid: {exc}"
    age_seconds = max(0.0, (checked_at - last_seen_at).total_seconds())
    if age_seconds > heartbeat_max_age_seconds:
        return (
            False,
            f"Latest all-tenant worker heartbeat is stale: {age_seconds:.0f}s old "
            f"(max {heartbeat_max_age_seconds:g}s)",
        )

    return (
        True,
        f"{configuration_detail}; latest all-tenant heartbeat from "
        f"{heartbeat['worker_id']} {age_seconds:.0f}s ago",
    )


def process_worker_batch(
    config: AnalysisWorkerConfig | None = None,
    *,
    store: SaaSStore | None = None,
    artifact_store: TenantArtifactStore | None = None,
    email_sender: EmailSender | None = None,
    data_connector_provider: DataConnectorProvider | None = None,
) -> AnalysisWorkerSummary:
    active_config = config or worker_config_from_env()
    active_store = store or SaaSStore(initialize_database_from_env())
    active_artifact_store = artifact_store or artifact_store_from_env()
    active_email_sender = email_sender or email_sender_from_env()
    analysis_summary = process_queued_analysis_jobs(
        active_store,
        active_artifact_store,
        email_sender=active_email_sender,
        organization_id=active_config.organization_id,
        limit=active_config.batch_size,
    )
    schedule_summary = process_due_report_schedules(
        active_store,
        active_artifact_store,
        active_email_sender,
        organization_id=active_config.organization_id,
        limit=active_config.batch_size,
    )
    connector_summary = process_due_data_connector_imports(
        active_store,
        active_artifact_store,
        provider=data_connector_provider,
        organization_id=active_config.organization_id,
        limit=active_config.batch_size,
    )
    summary = AnalysisWorkerSummary(
        attempted=analysis_summary.attempted,
        succeeded=analysis_summary.succeeded,
        failed=analysis_summary.failed,
        skipped=analysis_summary.skipped,
        scheduled_attempted=schedule_summary.attempted,
        scheduled_sent=schedule_summary.sent,
        scheduled_failed=schedule_summary.failed,
        scheduled_skipped=schedule_summary.skipped,
        connector_attempted=connector_summary.connector_attempted,
        connector_succeeded=connector_summary.connector_succeeded,
        connector_failed=connector_summary.connector_failed,
        connector_skipped=connector_summary.connector_skipped,
    )
    with active_store.conn:
        record_worker_heartbeat(active_store, active_config, summary)
        record_report_schedule_dispatch_event(active_store, active_config, summary)
    return summary


def _print_worker_summary(summary: AnalysisWorkerSummary) -> None:
    print(
        "analysis_worker "
        f"attempted={summary.attempted} "
        f"succeeded={summary.succeeded} "
        f"failed={summary.failed} "
        f"skipped={summary.skipped} "
        f"scheduled_attempted={summary.scheduled_attempted} "
        f"scheduled_sent={summary.scheduled_sent} "
        f"scheduled_failed={summary.scheduled_failed} "
        f"scheduled_skipped={summary.scheduled_skipped} "
        f"connector_attempted={summary.connector_attempted} "
        f"connector_succeeded={summary.connector_succeeded} "
        f"connector_failed={summary.connector_failed} "
        f"connector_skipped={summary.connector_skipped}",
        flush=True,
    )


def run_worker_loop(config: AnalysisWorkerConfig | None = None) -> None:
    active_config = config or worker_config_from_env()
    store = SaaSStore(initialize_database_from_env())
    artifact_store = artifact_store_from_env()
    email_sender = email_sender_from_env()
    while True:
        summary = process_worker_batch(
            active_config,
            store=store,
            artifact_store=artifact_store,
            email_sender=email_sender,
        )
        _print_worker_summary(summary)
        time.sleep(active_config.poll_seconds)


def main() -> None:
    if run_once_from_env():
        _print_worker_summary(process_worker_batch())
        return
    run_worker_loop()


if __name__ == "__main__":
    main()
