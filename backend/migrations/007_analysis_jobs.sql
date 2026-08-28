CREATE TABLE analysis_jobs (
    id TEXT PRIMARY KEY,
    organization_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    upload_id TEXT NOT NULL,
    report_id TEXT NOT NULL,
    requested_by_user_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'succeeded', 'failed')),
    client_config_json TEXT NOT NULL,
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT,
    UNIQUE (run_id, organization_id),
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (run_id, organization_id) REFERENCES runs(id, organization_id) ON DELETE CASCADE,
    FOREIGN KEY (upload_id, organization_id) REFERENCES uploads(id, organization_id) ON DELETE RESTRICT,
    FOREIGN KEY (requested_by_user_id) REFERENCES users(id) ON DELETE RESTRICT
);

CREATE INDEX idx_analysis_jobs_organization_status
    ON analysis_jobs (organization_id, status);
CREATE INDEX idx_analysis_jobs_run
    ON analysis_jobs (organization_id, run_id);
