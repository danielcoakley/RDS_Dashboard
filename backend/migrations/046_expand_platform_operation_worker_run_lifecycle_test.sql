ALTER TABLE platform_operation_events RENAME TO platform_operation_events_old;

DROP INDEX IF EXISTS idx_platform_operation_events_type_created;

CREATE TABLE platform_operation_events (
    id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL CHECK (
        event_type IN (
            'rds.alert_delivery_test',
            'rds.report_schedule_dispatch',
            'rds.clerk_webhook_replay',
            'rds.billing_webhook_replay',
            'rds.notification_provider_webhook_replay',
            'rds.artifact_store_test',
            'rds.database_test',
            'rds.tenant_isolation_test',
            'rds.run_lifecycle_smoke_test',
            'rds.worker_run_lifecycle_smoke_test'
        )
    ),
    status TEXT NOT NULL CHECK (status IN ('succeeded', 'failed')),
    request_id TEXT,
    detail TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO platform_operation_events (
    id,
    event_type,
    status,
    request_id,
    detail,
    created_at
)
SELECT
    id,
    event_type,
    status,
    request_id,
    detail,
    created_at
FROM platform_operation_events_old;

DROP TABLE platform_operation_events_old;

CREATE INDEX idx_platform_operation_events_type_created
    ON platform_operation_events (event_type, created_at);
