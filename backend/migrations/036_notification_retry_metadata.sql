ALTER TABLE notifications
    ADD COLUMN attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0);

ALTER TABLE notifications
    ADD COLUMN last_attempt_at TEXT;

ALTER TABLE notifications
    ADD COLUMN next_attempt_at TEXT;

CREATE INDEX idx_notifications_status_next_attempt
    ON notifications (status, next_attempt_at, created_at);
