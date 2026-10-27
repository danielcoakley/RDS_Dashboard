ALTER TABLE marketing_inquiries
    ADD COLUMN assigned_to TEXT;

ALTER TABLE marketing_inquiries
    ADD COLUMN follow_up_notes TEXT;

ALTER TABLE marketing_inquiries
    ADD COLUMN updated_at TEXT;

CREATE INDEX idx_marketing_inquiries_assignee_status
    ON marketing_inquiries (assigned_to, status, created_at);
