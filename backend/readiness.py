from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .artifacts import (
    InMemoryTenantArtifactStore,
    LocalTenantArtifactStore,
    TenantArtifactStore,
    artifact_store_from_env,
    artifact_store_governance_status,
)
from .auth_context import (
    AUTH_MODE_ENV,
    CLERK_AUTH_MODE,
    auth_mode_from_env,
    clerk_auth_configuration_status,
)
from .billing import billing_configuration_status
from .clerk_webhooks import clerk_webhook_configuration_status
from .csp_reports import (
    CSP_REPORT_RETENTION_DAYS_ENV,
    csp_report_retention_days_from_env,
)
from .database import (
    POSTGRES_BACKEND,
    DatabaseConfig,
    database_config_from_env,
    initialize_database_from_config,
)
from .database import MIGRATIONS_DIR
from .marketing_handoff import marketing_inquiry_handoff_configuration_status
from .marketing_rate_limit import marketing_inquiry_rate_limit_configuration_status
from .monitoring import metrics_configuration_status
from .notifications import (
    EMAIL_BACKEND_ENV,
    EMAIL_BACKEND_SMTP,
    email_delivery_configuration_status,
    notification_provider_webhook_configuration_status,
)
from .observability import log_aggregation_configuration_status
from .operations import (
    ALERT_DELIVERY_TEST_EVENT_TYPE,
    ALERT_DELIVERY_TEST_MAX_AGE_DAYS_ENV,
    ARTIFACT_STORE_TEST_EVENT_TYPE,
    BILLING_WEBHOOK_REPLAY_EVENT_TYPE,
    CLERK_WEBHOOK_REPLAY_EVENT_TYPE,
    DATABASE_TEST_EVENT_TYPE,
    NOTIFICATION_PROVIDER_WEBHOOK_REPLAY_EVENT_TYPE,
    OperationsConfigurationError,
    PROVIDER_WEBHOOK_REPLAY_MAX_AGE_DAYS_ENV,
    RUN_LIFECYCLE_TEST_EVENT_TYPE,
    TENANT_ISOLATION_TEST_EVENT_TYPE,
    WORKER_RUN_LIFECYCLE_TEST_EVENT_TYPE,
    alert_delivery_test_max_age_days_from_env,
    alerting_configuration_status,
    artifact_store_test_max_age_days_from_env,
    backup_configuration_status,
    database_test_max_age_days_from_env,
    provider_webhook_replay_max_age_days_from_env,
    run_lifecycle_test_max_age_days_from_env,
    tenant_isolation_test_max_age_days_from_env,
    worker_run_lifecycle_test_max_age_days_from_env,
)
from .runtime_config import PRODUCTION_RUNTIME, RUNTIME_ENV_ENV, runtime_env_from_env
from .security import (
    security_configuration_status,
    security_review_configuration_status,
    tenant_isolation_review_configuration_status,
)
from .store import SaaSStore
from .support import support_access_configuration_status
from .worker_app import worker_liveness_status, worker_supervision_status


SERVICE_NAME = "rds-saas-api"


def _check(name: str, passed: bool, detail: str) -> dict[str, str]:
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "detail": detail,
    }


def _expected_migration_versions() -> list[str]:
    return [path.stem for path in sorted(MIGRATIONS_DIR.glob("*.sql"))]


def _database_check(
    store: SaaSStore,
    runtime_env: str,
    database_config: DatabaseConfig,
) -> dict[str, str]:
    if runtime_env == PRODUCTION_RUNTIME and database_config.backend != POSTGRES_BACKEND:
        return _check(
            "database",
            False,
            "SQLite metadata storage is local-only; configure PostgreSQL with SAAS_DATABASE_URL",
        )

    try:
        store.conn.execute("SELECT 1").fetchone()
        applied = {
            row["version"]
            for row in store.conn.execute("SELECT version FROM schema_migrations").fetchall()
        }
    except Exception as exc:
        return _check("database", False, f"Database check failed: {exc}")

    expected = set(_expected_migration_versions())
    missing = sorted(expected - applied)
    if missing:
        return _check("database", False, f"Missing migrations: {', '.join(missing)}")
    return _check(
        "database",
        True,
        f"{database_config.backend} metadata store has {len(applied)} migrations applied",
    )


def _auth_check(runtime_env: str) -> dict[str, str]:
    try:
        auth_mode = auth_mode_from_env()
    except ValueError as exc:
        return _check("auth", False, str(exc))

    if runtime_env == PRODUCTION_RUNTIME and auth_mode != CLERK_AUTH_MODE:
        return _check(
            "auth",
            False,
            f"{AUTH_MODE_ENV} must be {CLERK_AUTH_MODE!r} when {RUNTIME_ENV_ENV}=production",
        )
    if auth_mode == CLERK_AUTH_MODE:
        configured, detail = clerk_auth_configuration_status(runtime_env)
        return _check("auth", configured, detail)
    return _check("auth", True, f"{AUTH_MODE_ENV}={auth_mode}")


def _artifact_store_check(runtime_env: str, artifact_store: TenantArtifactStore) -> dict[str, str]:
    store_type = type(artifact_store).__name__
    if runtime_env == PRODUCTION_RUNTIME and isinstance(
        artifact_store,
        (InMemoryTenantArtifactStore, LocalTenantArtifactStore),
    ):
        return _check(
            "artifact_store",
            False,
            f"{store_type} is local-only; configure production object storage",
        )
    governance_configured, governance_detail = artifact_store_governance_status(runtime_env)
    if not governance_configured:
        return _check("artifact_store", False, governance_detail)
    return _check("artifact_store", True, f"{store_type} configured; {governance_detail}")


def _webhook_check(runtime_env: str) -> dict[str, str]:
    configured, detail = clerk_webhook_configuration_status()
    if runtime_env == PRODUCTION_RUNTIME:
        return _check("clerk_webhooks", configured, detail)
    if configured:
        return _check("clerk_webhooks", True, detail)
    return _check("clerk_webhooks", True, f"{detail}; webhooks optional in development")


def _email_delivery_check(runtime_env: str) -> dict[str, str]:
    configured, detail = email_delivery_configuration_status(runtime_env)
    if not configured:
        return _check("email_delivery", False, detail)
    if runtime_env == PRODUCTION_RUNTIME and f"{EMAIL_BACKEND_ENV}={EMAIL_BACKEND_SMTP}" not in detail:
        return _check(
            "email_delivery",
            False,
            f"{EMAIL_BACKEND_ENV} must be {EMAIL_BACKEND_SMTP!r} when {RUNTIME_ENV_ENV}=production",
        )
    return _check("email_delivery", True, detail)


def _notification_provider_webhook_check(runtime_env: str) -> dict[str, str]:
    configured, detail = notification_provider_webhook_configuration_status(runtime_env)
    return _check("notification_provider_webhooks", configured, detail)


def _marketing_inquiry_check(runtime_env: str) -> dict[str, str]:
    handoff_configured, handoff_detail = marketing_inquiry_handoff_configuration_status(
        runtime_env
    )
    rate_limit_configured, rate_limit_detail = marketing_inquiry_rate_limit_configuration_status(
        runtime_env
    )
    configured = handoff_configured and rate_limit_configured
    detail = f"{handoff_detail}; {rate_limit_detail}"
    if runtime_env == PRODUCTION_RUNTIME:
        return _check("marketing_inquiries", configured, detail)
    if not rate_limit_configured:
        return _check("marketing_inquiries", False, detail)
    if handoff_configured:
        return _check("marketing_inquiries", True, detail)
    if "must be configured" not in handoff_detail:
        return _check("marketing_inquiries", False, detail)
    return _check("marketing_inquiries", True, f"{detail}; handoff optional in development")


def _metrics_check(runtime_env: str) -> dict[str, str]:
    configured, detail = metrics_configuration_status(runtime_env)
    return _check("metrics", configured, detail)


def _log_aggregation_check(runtime_env: str) -> dict[str, str]:
    configured, detail = log_aggregation_configuration_status(runtime_env)
    return _check("log_aggregation", configured, detail)


def _support_access_check(runtime_env: str) -> dict[str, str]:
    configured, detail = support_access_configuration_status(runtime_env)
    return _check("support_access", configured, detail)


def _csp_report_retention_check() -> dict[str, str]:
    try:
        retention_days = csp_report_retention_days_from_env()
    except ValueError as exc:
        return _check("csp_report_retention", False, str(exc))
    return _check(
        "csp_report_retention",
        True,
        f"{CSP_REPORT_RETENTION_DAYS_ENV}={retention_days} day(s)",
    )


def _billing_check(runtime_env: str) -> dict[str, str]:
    configured, detail = billing_configuration_status(runtime_env)
    return _check("billing", configured, detail)


def _security_check(runtime_env: str) -> dict[str, str]:
    configured, detail = security_configuration_status(runtime_env)
    return _check("security", configured, detail)


def _security_review_check(runtime_env: str) -> dict[str, str]:
    configured, detail = security_review_configuration_status(runtime_env)
    return _check("security_review", configured, detail)


def _tenant_isolation_review_check(runtime_env: str) -> dict[str, str]:
    configured, detail = tenant_isolation_review_configuration_status(runtime_env)
    return _check("tenant_isolation_review", configured, detail)


def _tenant_isolation_test_evidence_check(
    store: SaaSStore | None,
    runtime_env: str,
) -> dict[str, str]:
    try:
        max_age_days = tenant_isolation_test_max_age_days_from_env()
    except OperationsConfigurationError as exc:
        return _check("tenant_isolation_test_evidence", False, str(exc))

    if store is None:
        return _check(
            "tenant_isolation_test_evidence",
            False,
            "Tenant isolation smoke-test evidence check requires metadata store access",
        )

    try:
        rows = store.list_platform_operation_events(
            event_type=TENANT_ISOLATION_TEST_EVENT_TYPE,
            event_status="succeeded",
            limit=1,
        )
    except Exception as exc:
        return _check(
            "tenant_isolation_test_evidence",
            False,
            f"Tenant isolation smoke-test evidence check failed: {exc}",
        )
    if not rows:
        if runtime_env == PRODUCTION_RUNTIME:
            return _check(
                "tenant_isolation_test_evidence",
                False,
                "No successful tenant isolation smoke test has been recorded",
            )
        return _check(
            "tenant_isolation_test_evidence",
            True,
            "Tenant isolation smoke-test evidence is optional in development",
        )

    latest = rows[0]
    try:
        observed_at = _parse_observed_at(latest["created_at"])
    except (TypeError, ValueError) as exc:
        return _check(
            "tenant_isolation_test_evidence",
            False,
            f"Latest tenant isolation smoke-test timestamp is invalid: {exc}",
        )
    now = datetime.now(timezone.utc)
    if observed_at > now:
        return _check(
            "tenant_isolation_test_evidence",
            False,
            "Latest tenant isolation smoke-test evidence cannot be in the future",
        )
    age_days = (now - observed_at).days
    if runtime_env == PRODUCTION_RUNTIME and age_days > max_age_days:
        return _check(
            "tenant_isolation_test_evidence",
            False,
            f"Latest tenant isolation smoke-test evidence is older than {max_age_days} days",
        )
    return _check(
        "tenant_isolation_test_evidence",
        True,
        f"latest successful tenant isolation smoke test age {age_days} day(s)",
    )


def _run_lifecycle_test_evidence_check(
    store: SaaSStore | None,
    runtime_env: str,
) -> dict[str, str]:
    try:
        max_age_days = run_lifecycle_test_max_age_days_from_env()
    except OperationsConfigurationError as exc:
        return _check("run_lifecycle_test_evidence", False, str(exc))

    if store is None:
        return _check(
            "run_lifecycle_test_evidence",
            False,
            "Run lifecycle smoke-test evidence check requires metadata store access",
        )

    try:
        rows = store.list_platform_operation_events(
            event_type=RUN_LIFECYCLE_TEST_EVENT_TYPE,
            event_status="succeeded",
            limit=1,
        )
    except Exception as exc:
        return _check(
            "run_lifecycle_test_evidence",
            False,
            f"Run lifecycle smoke-test evidence check failed: {exc}",
        )
    if not rows:
        if runtime_env == PRODUCTION_RUNTIME:
            return _check(
                "run_lifecycle_test_evidence",
                False,
                "No successful run lifecycle smoke test has been recorded",
            )
        return _check(
            "run_lifecycle_test_evidence",
            True,
            "Run lifecycle smoke-test evidence is optional in development",
        )

    latest = rows[0]
    try:
        observed_at = _parse_observed_at(latest["created_at"])
    except (TypeError, ValueError) as exc:
        return _check(
            "run_lifecycle_test_evidence",
            False,
            f"Latest run lifecycle smoke-test timestamp is invalid: {exc}",
        )
    now = datetime.now(timezone.utc)
    if observed_at > now:
        return _check(
            "run_lifecycle_test_evidence",
            False,
            "Latest run lifecycle smoke-test evidence cannot be in the future",
        )
    age_days = (now - observed_at).days
    if runtime_env == PRODUCTION_RUNTIME and age_days > max_age_days:
        return _check(
            "run_lifecycle_test_evidence",
            False,
            f"Latest run lifecycle smoke-test evidence is older than {max_age_days} days",
        )
    return _check(
        "run_lifecycle_test_evidence",
        True,
        f"latest successful run lifecycle smoke test age {age_days} day(s)",
    )


def _worker_run_lifecycle_test_evidence_check(
    store: SaaSStore | None,
    runtime_env: str,
) -> dict[str, str]:
    try:
        max_age_days = worker_run_lifecycle_test_max_age_days_from_env()
    except OperationsConfigurationError as exc:
        return _check("worker_run_lifecycle_test_evidence", False, str(exc))

    if store is None:
        return _check(
            "worker_run_lifecycle_test_evidence",
            False,
            "Worker run lifecycle smoke-test evidence check requires metadata store access",
        )

    try:
        rows = store.list_platform_operation_events(
            event_type=WORKER_RUN_LIFECYCLE_TEST_EVENT_TYPE,
            event_status="succeeded",
            limit=1,
        )
    except Exception as exc:
        return _check(
            "worker_run_lifecycle_test_evidence",
            False,
            f"Worker run lifecycle smoke-test evidence check failed: {exc}",
        )
    if not rows:
        if runtime_env == PRODUCTION_RUNTIME:
            return _check(
                "worker_run_lifecycle_test_evidence",
                False,
                "No successful worker run lifecycle smoke test has been recorded",
            )
        return _check(
            "worker_run_lifecycle_test_evidence",
            True,
            "Worker run lifecycle smoke-test evidence is optional in development",
        )

    latest = rows[0]
    try:
        observed_at = _parse_observed_at(latest["created_at"])
    except (TypeError, ValueError) as exc:
        return _check(
            "worker_run_lifecycle_test_evidence",
            False,
            f"Latest worker run lifecycle smoke-test timestamp is invalid: {exc}",
        )
    now = datetime.now(timezone.utc)
    if observed_at > now:
        return _check(
            "worker_run_lifecycle_test_evidence",
            False,
            "Latest worker run lifecycle smoke-test evidence cannot be in the future",
        )
    age_days = (now - observed_at).days
    if runtime_env == PRODUCTION_RUNTIME and age_days > max_age_days:
        return _check(
            "worker_run_lifecycle_test_evidence",
            False,
            f"Latest worker run lifecycle smoke-test evidence is older than {max_age_days} days",
        )
    return _check(
        "worker_run_lifecycle_test_evidence",
        True,
        f"latest successful worker run lifecycle smoke test age {age_days} day(s)",
    )


def _backup_check(runtime_env: str) -> dict[str, str]:
    configured, detail = backup_configuration_status(runtime_env)
    return _check("backups", configured, detail)


def _alerting_check(runtime_env: str) -> dict[str, str]:
    configured, detail = alerting_configuration_status(runtime_env)
    return _check("alerting", configured, detail)


def _parse_observed_at(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _alert_delivery_evidence_check(
    store: SaaSStore | None,
    runtime_env: str,
) -> dict[str, str]:
    try:
        max_age_days = alert_delivery_test_max_age_days_from_env()
    except OperationsConfigurationError as exc:
        return _check("alert_delivery_evidence", False, str(exc))

    if store is None:
        return _check(
            "alert_delivery_evidence",
            False,
            "Alert delivery evidence check requires metadata store access",
        )

    try:
        rows = store.list_platform_operation_events(
            event_type=ALERT_DELIVERY_TEST_EVENT_TYPE,
            event_status="succeeded",
            limit=1,
        )
    except Exception as exc:
        return _check(
            "alert_delivery_evidence",
            False,
            f"Alert delivery evidence check failed: {exc}",
        )
    if not rows:
        if runtime_env == PRODUCTION_RUNTIME:
            return _check(
                "alert_delivery_evidence",
                False,
                "No successful alert delivery smoke test has been recorded",
            )
        return _check(
            "alert_delivery_evidence",
            True,
            "Alert delivery smoke-test evidence is optional in development",
        )

    latest = rows[0]
    try:
        observed_at = _parse_observed_at(latest["created_at"])
    except (TypeError, ValueError) as exc:
        return _check(
            "alert_delivery_evidence",
            False,
            f"Latest alert delivery smoke-test timestamp is invalid: {exc}",
        )
    now = datetime.now(timezone.utc)
    if observed_at > now:
        return _check(
            "alert_delivery_evidence",
            False,
            "Latest alert delivery smoke-test evidence cannot be in the future",
        )
    age_days = (now - observed_at).days
    if runtime_env == PRODUCTION_RUNTIME and age_days > max_age_days:
        return _check(
            "alert_delivery_evidence",
            False,
            (
                "Latest alert delivery smoke-test evidence is older than "
                f"{max_age_days} days"
            ),
        )
    return _check(
        "alert_delivery_evidence",
        True,
        f"latest successful alert delivery smoke test age {age_days} day(s)",
    )


def _provider_webhook_replay_evidence_check(
    store: SaaSStore | None,
    runtime_env: str,
    *,
    name: str,
    event_type: str,
    evidence_label: str,
) -> dict[str, str]:
    try:
        max_age_days = provider_webhook_replay_max_age_days_from_env()
    except OperationsConfigurationError as exc:
        return _check(name, False, str(exc))

    if store is None:
        return _check(name, False, f"{evidence_label} check requires metadata store access")

    try:
        rows = store.list_platform_operation_events(
            event_type=event_type,
            event_status="succeeded",
            limit=1,
        )
    except Exception as exc:
        return _check(name, False, f"{evidence_label} check failed: {exc}")

    if not rows:
        if runtime_env == PRODUCTION_RUNTIME:
            return _check(name, False, f"No successful {evidence_label} has been recorded")
        return _check(name, True, f"{evidence_label} is optional in development")

    latest = rows[0]
    try:
        observed_at = _parse_observed_at(latest["created_at"])
    except (TypeError, ValueError) as exc:
        return _check(name, False, f"Latest {evidence_label} timestamp is invalid: {exc}")

    now = datetime.now(timezone.utc)
    if observed_at > now:
        return _check(name, False, f"Latest {evidence_label} cannot be in the future")
    age_days = (now - observed_at).days
    if runtime_env == PRODUCTION_RUNTIME and age_days > max_age_days:
        return _check(
            name,
            False,
            f"Latest {evidence_label} is older than {max_age_days} days",
        )
    return _check(name, True, f"latest successful {evidence_label} age {age_days} day(s)")


def _artifact_store_test_evidence_check(
    store: SaaSStore | None,
    runtime_env: str,
) -> dict[str, str]:
    try:
        max_age_days = artifact_store_test_max_age_days_from_env()
    except OperationsConfigurationError as exc:
        return _check("artifact_store_test_evidence", False, str(exc))

    if store is None:
        return _check(
            "artifact_store_test_evidence",
            False,
            "Artifact store smoke-test evidence check requires metadata store access",
        )

    try:
        rows = store.list_platform_operation_events(
            event_type=ARTIFACT_STORE_TEST_EVENT_TYPE,
            event_status="succeeded",
            limit=1,
        )
    except Exception as exc:
        return _check(
            "artifact_store_test_evidence",
            False,
            f"Artifact store smoke-test evidence check failed: {exc}",
        )
    if not rows:
        if runtime_env == PRODUCTION_RUNTIME:
            return _check(
                "artifact_store_test_evidence",
                False,
                "No successful artifact store smoke test has been recorded",
            )
        return _check(
            "artifact_store_test_evidence",
            True,
            "Artifact store smoke-test evidence is optional in development",
        )

    latest = rows[0]
    try:
        observed_at = _parse_observed_at(latest["created_at"])
    except (TypeError, ValueError) as exc:
        return _check(
            "artifact_store_test_evidence",
            False,
            f"Latest artifact store smoke-test timestamp is invalid: {exc}",
        )
    now = datetime.now(timezone.utc)
    if observed_at > now:
        return _check(
            "artifact_store_test_evidence",
            False,
            "Latest artifact store smoke-test evidence cannot be in the future",
        )
    age_days = (now - observed_at).days
    if runtime_env == PRODUCTION_RUNTIME and age_days > max_age_days:
        return _check(
            "artifact_store_test_evidence",
            False,
            f"Latest artifact store smoke-test evidence is older than {max_age_days} days",
        )
    return _check(
        "artifact_store_test_evidence",
        True,
        f"latest successful artifact store smoke test age {age_days} day(s)",
    )


def _database_test_evidence_check(
    store: SaaSStore | None,
    runtime_env: str,
) -> dict[str, str]:
    try:
        max_age_days = database_test_max_age_days_from_env()
    except OperationsConfigurationError as exc:
        return _check("database_test_evidence", False, str(exc))

    if store is None:
        return _check(
            "database_test_evidence",
            False,
            "Database smoke-test evidence check requires metadata store access",
        )

    try:
        rows = store.list_platform_operation_events(
            event_type=DATABASE_TEST_EVENT_TYPE,
            event_status="succeeded",
            limit=1,
        )
    except Exception as exc:
        return _check(
            "database_test_evidence",
            False,
            f"Database smoke-test evidence check failed: {exc}",
        )
    if not rows:
        if runtime_env == PRODUCTION_RUNTIME:
            return _check(
                "database_test_evidence",
                False,
                "No successful database smoke test has been recorded",
            )
        return _check(
            "database_test_evidence",
            True,
            "Database smoke-test evidence is optional in development",
        )

    latest = rows[0]
    try:
        observed_at = _parse_observed_at(latest["created_at"])
    except (TypeError, ValueError) as exc:
        return _check(
            "database_test_evidence",
            False,
            f"Latest database smoke-test timestamp is invalid: {exc}",
        )
    now = datetime.now(timezone.utc)
    if observed_at > now:
        return _check(
            "database_test_evidence",
            False,
            "Latest database smoke-test evidence cannot be in the future",
        )
    age_days = (now - observed_at).days
    if runtime_env == PRODUCTION_RUNTIME and age_days > max_age_days:
        return _check(
            "database_test_evidence",
            False,
            f"Latest database smoke-test evidence is older than {max_age_days} days",
        )
    return _check(
        "database_test_evidence",
        True,
        f"latest successful database smoke test age {age_days} day(s)",
    )


def _worker_check(store: SaaSStore | None, runtime_env: str) -> dict[str, str]:
    if store is None:
        return _check("worker", False, "Worker heartbeat check requires metadata store access")
    configured, detail = worker_liveness_status(store, runtime_env)
    return _check("worker", configured, detail)


def _worker_supervision_check(runtime_env: str) -> dict[str, str]:
    configured, detail = worker_supervision_status(runtime_env)
    return _check("worker_supervision", configured, detail)


def readiness_report(
    store: SaaSStore | None = None,
    artifact_store: TenantArtifactStore | None = None,
) -> dict[str, Any]:
    runtime_env = runtime_env_from_env()
    database_config_error: str | None = None
    database_connection_error: str | None = None
    try:
        database_config = database_config_from_env()
    except ValueError as exc:
        database_config = None
        database_config_error = str(exc)

    active_store = store
    if active_store is None and database_config is not None:
        try:
            active_store = SaaSStore(initialize_database_from_config(database_config))
        except Exception as exc:
            database_connection_error = str(exc)

    artifact_store_error: str | None = None
    active_artifact_store: TenantArtifactStore | None = artifact_store
    if active_artifact_store is None:
        try:
            active_artifact_store = artifact_store_from_env()
        except (RuntimeError, ValueError) as exc:
            artifact_store_error = str(exc)

    artifact_store_check = (
        _check(
            "artifact_store",
            False,
            f"Artifact store configuration failed: {artifact_store_error}",
        )
        if active_artifact_store is None
        else _artifact_store_check(runtime_env, active_artifact_store)
    )
    if database_config_error is not None:
        database_check = _check("database", False, database_config_error)
    elif active_store is None or database_config is None:
        database_check = _check(
            "database",
            False,
            f"Database connection failed: {database_connection_error}",
        )
    else:
        database_check = _database_check(active_store, runtime_env, database_config)

    checks = [
        database_check,
        _database_test_evidence_check(active_store, runtime_env),
        _auth_check(runtime_env),
        _webhook_check(runtime_env),
        _email_delivery_check(runtime_env),
        _notification_provider_webhook_check(runtime_env),
        _marketing_inquiry_check(runtime_env),
        _billing_check(runtime_env),
        _security_check(runtime_env),
        _security_review_check(runtime_env),
        _tenant_isolation_review_check(runtime_env),
        _tenant_isolation_test_evidence_check(active_store, runtime_env),
        _run_lifecycle_test_evidence_check(active_store, runtime_env),
        _backup_check(runtime_env),
        _alerting_check(runtime_env),
        _alert_delivery_evidence_check(active_store, runtime_env),
        _provider_webhook_replay_evidence_check(
            active_store,
            runtime_env,
            name="clerk_webhook_replay_evidence",
            event_type=CLERK_WEBHOOK_REPLAY_EVENT_TYPE,
            evidence_label="Clerk webhook replay evidence",
        ),
        _provider_webhook_replay_evidence_check(
            active_store,
            runtime_env,
            name="billing_webhook_replay_evidence",
            event_type=BILLING_WEBHOOK_REPLAY_EVENT_TYPE,
            evidence_label="billing webhook replay evidence",
        ),
        _provider_webhook_replay_evidence_check(
            active_store,
            runtime_env,
            name="notification_provider_replay_evidence",
            event_type=NOTIFICATION_PROVIDER_WEBHOOK_REPLAY_EVENT_TYPE,
            evidence_label="notification provider replay evidence",
        ),
        _metrics_check(runtime_env),
        _log_aggregation_check(runtime_env),
        _support_access_check(runtime_env),
        _csp_report_retention_check(),
        _worker_check(active_store, runtime_env),
        _worker_supervision_check(runtime_env),
        _worker_run_lifecycle_test_evidence_check(active_store, runtime_env),
        _artifact_store_test_evidence_check(active_store, runtime_env),
        artifact_store_check,
    ]
    is_ready = all(check["status"] == "pass" for check in checks)
    return {
        "status": "ready" if is_ready else "not_ready",
        "service": SERVICE_NAME,
        "runtime_env": runtime_env,
        "checks": checks,
    }
