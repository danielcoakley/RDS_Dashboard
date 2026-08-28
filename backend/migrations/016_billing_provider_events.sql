CREATE TABLE billing_events (
    id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,
    event_type TEXT NOT NULL,
    organization_id TEXT,
    external_customer_id TEXT,
    external_subscription_id TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}',
    processed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE SET NULL
);

CREATE INDEX idx_billing_events_organization_processed
    ON billing_events (organization_id, processed_at);
CREATE INDEX idx_billing_events_external_subscription
    ON billing_events (external_subscription_id);

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
