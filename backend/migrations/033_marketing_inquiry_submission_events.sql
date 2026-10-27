CREATE TABLE marketing_inquiry_submission_events (
    id TEXT PRIMARY KEY,
    source_key_hash TEXT NOT NULL,
    inquiry_id TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (inquiry_id) REFERENCES marketing_inquiries(id) ON DELETE SET NULL
);

CREATE INDEX idx_marketing_inquiry_submission_events_source_created
    ON marketing_inquiry_submission_events (source_key_hash, created_at);

CREATE INDEX idx_marketing_inquiry_submission_events_inquiry
    ON marketing_inquiry_submission_events (inquiry_id);
