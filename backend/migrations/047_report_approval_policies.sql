CREATE TABLE organization_report_approval_policies (
    organization_id TEXT PRIMARY KEY,
    auto_lock_evidence_on_final_approval INTEGER NOT NULL DEFAULT 1 CHECK (
        auto_lock_evidence_on_final_approval IN (0, 1)
    ),
    minimum_approval_assignments_for_lock INTEGER NOT NULL DEFAULT 0 CHECK (
        minimum_approval_assignments_for_lock >= 0
    ),
    updated_by_user_id TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (updated_by_user_id) REFERENCES users(id) ON DELETE SET NULL
);

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
            'report_review_recorded',
            'report_exported',
            'report_evidence_exported',
            'report_evidence_package_created',
            'report_evidence_package_downloaded',
            'report_approval_assigned',
            'report_approval_updated',
            'report_approval_policy_updated',
            'report_branding_updated',
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
