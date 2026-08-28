CREATE TABLE marketing_inquiry_followup_events (
    id TEXT PRIMARY KEY,
    inquiry_id TEXT NOT NULL,
    actor_label TEXT NOT NULL DEFAULT 'support',
    action TEXT NOT NULL CHECK (
        action IN ('created', 'updated')
    ),
    status TEXT CHECK (
        status IS NULL OR status IN ('new', 'contacted', 'qualified', 'closed')
    ),
    assigned_to TEXT,
    changed_fields_json TEXT NOT NULL DEFAULT '[]',
    notes_changed INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (inquiry_id) REFERENCES marketing_inquiries(id) ON DELETE CASCADE
);

CREATE INDEX idx_marketing_inquiry_followup_events_inquiry_created
    ON marketing_inquiry_followup_events (inquiry_id, created_at);
