CREATE TABLE subscription_plans (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    monthly_price_cents INTEGER CHECK (monthly_price_cents IS NULL OR monthly_price_cents >= 0),
    included_users INTEGER CHECK (included_users IS NULL OR included_users >= 0),
    included_sites INTEGER CHECK (included_sites IS NULL OR included_sites >= 0),
    monthly_run_limit INTEGER CHECK (monthly_run_limit IS NULL OR monthly_run_limit >= 0),
    report_retention_days INTEGER CHECK (report_retention_days IS NULL OR report_retention_days >= 0),
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE organization_subscriptions (
    organization_id TEXT PRIMARY KEY,
    plan_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('trialing', 'active', 'past_due', 'canceled')),
    current_period_start TEXT,
    current_period_end TEXT,
    trial_ends_at TEXT,
    external_customer_id TEXT,
    external_subscription_id TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (plan_id) REFERENCES subscription_plans(id) ON DELETE RESTRICT
);

CREATE INDEX idx_organization_subscriptions_plan ON organization_subscriptions (plan_id);
CREATE INDEX idx_organization_subscriptions_status ON organization_subscriptions (status);

INSERT INTO subscription_plans (
    id,
    name,
    description,
    monthly_price_cents,
    included_users,
    included_sites,
    monthly_run_limit,
    report_retention_days
)
VALUES
    (
        'trial',
        'Pilot trial',
        'Self-serve ISO 50001 pilot workspace',
        0,
        5,
        3,
        25,
        90
    ),
    (
        'team',
        'Team',
        'Multi-client energy management workspace for a single organization',
        49000,
        15,
        10,
        250,
        365
    ),
    (
        'enterprise',
        'Enterprise',
        'Portfolio workspace with custom limits and support',
        NULL,
        NULL,
        NULL,
        NULL,
        NULL
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
            'organization_updated',
            'analysis_config_updated',
            'membership_role_updated',
            'membership_removed',
            'subscription_updated',
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
