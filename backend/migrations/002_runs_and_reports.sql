CREATE TABLE uploads (
    id TEXT PRIMARY KEY,
    organization_id TEXT NOT NULL,
    site_id TEXT NOT NULL,
    uploaded_by_user_id TEXT NOT NULL,
    category TEXT NOT NULL,
    storage_key TEXT NOT NULL UNIQUE,
    checksum TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'stored', 'rejected')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (id, organization_id),
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (site_id, organization_id) REFERENCES sites(id, organization_id) ON DELETE CASCADE,
    FOREIGN KEY (uploaded_by_user_id) REFERENCES users(id) ON DELETE RESTRICT
);

CREATE TABLE runs (
    id TEXT PRIMARY KEY,
    organization_id TEXT NOT NULL,
    site_id TEXT NOT NULL,
    requested_by_user_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'succeeded', 'failed')),
    error_message TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT,
    UNIQUE (id, organization_id),
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (site_id, organization_id) REFERENCES sites(id, organization_id) ON DELETE CASCADE,
    FOREIGN KEY (requested_by_user_id) REFERENCES users(id) ON DELETE RESTRICT
);

CREATE TABLE run_uploads (
    run_id TEXT NOT NULL,
    organization_id TEXT NOT NULL,
    upload_id TEXT NOT NULL,
    PRIMARY KEY (run_id, upload_id),
    FOREIGN KEY (run_id, organization_id) REFERENCES runs(id, organization_id) ON DELETE CASCADE,
    FOREIGN KEY (upload_id, organization_id) REFERENCES uploads(id, organization_id) ON DELETE RESTRICT
);

CREATE TABLE reports (
    id TEXT PRIMARY KEY,
    organization_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    report_type TEXT NOT NULL,
    storage_key TEXT NOT NULL UNIQUE,
    is_published INTEGER NOT NULL DEFAULT 0 CHECK (is_published IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    FOREIGN KEY (run_id, organization_id) REFERENCES runs(id, organization_id) ON DELETE CASCADE
);

CREATE INDEX idx_uploads_organization_site ON uploads (organization_id, site_id);
CREATE INDEX idx_runs_organization_site ON runs (organization_id, site_id);
CREATE INDEX idx_reports_organization_run ON reports (organization_id, run_id);
