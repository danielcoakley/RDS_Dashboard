CREATE TABLE platform_operation_events (
    id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL CHECK (
        event_type IN (
            'rds.alert_delivery_test'
        )
    ),
    status TEXT NOT NULL CHECK (status IN ('succeeded', 'failed')),
    request_id TEXT,
    detail TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_platform_operation_events_type_created
    ON platform_operation_events (event_type, created_at);

