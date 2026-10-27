ALTER TABLE organization_report_schedules ADD COLUMN last_delivery_at TEXT;
ALTER TABLE organization_report_schedules ADD COLUMN last_delivery_status TEXT;
ALTER TABLE organization_report_schedules ADD COLUMN last_delivery_error TEXT;

CREATE TABLE organization_report_schedule_deliveries (
    id TEXT PRIMARY KEY,
    organization_id TEXT NOT NULL,
    schedule_id TEXT NOT NULL,
    report_id TEXT NOT NULL,
    recipient_email TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('sent', 'failed')),
    error_message TEXT,
    attempted_at TEXT NOT NULL,
    sent_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (schedule_id) REFERENCES organization_report_schedules(id) ON DELETE CASCADE,
    FOREIGN KEY (report_id) REFERENCES reports(id) ON DELETE CASCADE
);

CREATE INDEX idx_report_schedule_deliveries_schedule_created
    ON organization_report_schedule_deliveries (organization_id, schedule_id, created_at);

ALTER TABLE audit_events RENAME TO audit_events_old;

CREATE TABLE audit_events (
    id TEXT PRIMARY KEY,
    organization_id TEXT NOT NULL,
    actor_user_id TEXT NOT NULL,
    action TEXT NOT NULL CHECK (
        action IN (
            'upload_stored',
            'run_started',
            'run_succeeded',
            'run_failed',
            'report_created',
            'report_accessed',
            'report_retention_applied',
            'report_comment_created',
            'report_exported',
            'report_schedule_created',
            'report_schedule_updated',
            'report_schedule_delivered',
            'report_schedule_failed',
            'privacy_settings_updated',
            'privacy_request_created',
            'privacy_request_updated',
            'pilot_readiness_updated',
            'organization_updated',
            'analysis_config_updated',
            'membership_role_updated',
            'membership_removed',
            'subscription_updated',
            'billing_checkout_created',
            'billing_portal_created',
            'site_created',
            'meter_created',
            'invite_created',
            'invite_accepted',
            'invite_revoked'
        )
    ),
    resource_type TEXT NOT NULL,
    resource_id TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (actor_user_id) REFERENCES users(id) ON DELETE RESTRICT
);

INSERT INTO audit_events (
    id,
    organization_id,
    actor_user_id,
    action,
    resource_type,
    resource_id,
    metadata_json,
    created_at
)
SELECT
    id,
    organization_id,
    actor_user_id,
    action,
    resource_type,
    resource_id,
    metadata_json,
    created_at
FROM audit_events_old;

DROP TABLE audit_events_old;

CREATE INDEX idx_audit_events_organization_created ON audit_events (organization_id, created_at);
CREATE INDEX idx_audit_events_resource ON audit_events (resource_type, resource_id);
