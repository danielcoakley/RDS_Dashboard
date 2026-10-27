CREATE TABLE notification_provider_events (
    id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    provider_event_id TEXT NOT NULL,
    notification_id TEXT NOT NULL,
    organization_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (
        status IN (
            'delivered',
            'bounced',
            'complained',
            'deferred',
            'dropped'
        )
    ),
    detail TEXT NOT NULL DEFAULT '',
    occurred_at TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (provider, provider_event_id),
    FOREIGN KEY (notification_id) REFERENCES notifications(id) ON DELETE CASCADE,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE
);

CREATE INDEX idx_notification_provider_events_status
    ON notification_provider_events (status, occurred_at);

CREATE INDEX idx_notification_provider_events_notification
    ON notification_provider_events (organization_id, notification_id, occurred_at);
