CREATE TABLE marketing_inquiries (
    id TEXT PRIMARY KEY,
    email TEXT NOT NULL,
    name TEXT NOT NULL,
    organization_name TEXT,
    role TEXT,
    intended_use TEXT NOT NULL,
    message TEXT NOT NULL,
    source_page TEXT NOT NULL DEFAULT 'landing',
    status TEXT NOT NULL DEFAULT 'new' CHECK (
        status IN ('new', 'contacted', 'qualified', 'closed')
    ),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_marketing_inquiries_status_created
    ON marketing_inquiries (status, created_at);

CREATE INDEX idx_marketing_inquiries_email
    ON marketing_inquiries (email);
