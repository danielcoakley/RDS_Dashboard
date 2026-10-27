CREATE TABLE notifications (
    id TEXT PRIMARY KEY,
    organization_id TEXT NOT NULL,
    recipient_user_id TEXT NOT NULL,
    channel TEXT NOT NULL CHECK (channel IN ('email')),
    event_type TEXT NOT NULL,
    subject TEXT NOT NULL,
    body TEXT NOT NULL,
    resource_type TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'sent', 'failed')) DEFAULT 'pending',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    sent_at TEXT,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (recipient_user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (recipient_user_id, organization_id)
        REFERENCES memberships(user_id, organization_id)
        ON DELETE CASCADE
);

CREATE INDEX idx_notifications_recipient_status
    ON notifications (organization_id, recipient_user_id, status);
CREATE INDEX idx_notifications_resource
    ON notifications (organization_id, resource_type, resource_id);
