CREATE TABLE worker_heartbeats (
    worker_id TEXT PRIMARY KEY,
    organization_id TEXT,
    scope TEXT NOT NULL CHECK (scope IN ('all', 'organization')),
    poll_seconds REAL NOT NULL CHECK (poll_seconds >= 0),
    batch_size INTEGER NOT NULL CHECK (batch_size > 0),
    last_attempted INTEGER NOT NULL DEFAULT 0,
    last_succeeded INTEGER NOT NULL DEFAULT 0,
    last_failed INTEGER NOT NULL DEFAULT 0,
    last_skipped INTEGER NOT NULL DEFAULT 0,
    last_scheduled_attempted INTEGER NOT NULL DEFAULT 0,
    last_scheduled_sent INTEGER NOT NULL DEFAULT 0,
    last_scheduled_failed INTEGER NOT NULL DEFAULT 0,
    last_scheduled_skipped INTEGER NOT NULL DEFAULT 0,
    last_seen_at TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_worker_heartbeats_scope_seen
    ON worker_heartbeats (scope, organization_id, last_seen_at);
