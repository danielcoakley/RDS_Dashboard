CREATE TABLE csp_violation_reports (
    id TEXT PRIMARY KEY,
    disposition TEXT NOT NULL CHECK (disposition IN ('enforce', 'report')),
    effective_directive TEXT NOT NULL CHECK (
        effective_directive IN (
            'default-src',
            'script-src',
            'style-src',
            'img-src',
            'connect-src',
            'font-src',
            'frame-src',
            'object-src',
            'base-uri',
            'form-action',
            'frame-ancestors',
            'other'
        )
    ),
    violated_directive TEXT NOT NULL CHECK (
        violated_directive IN (
            'default-src',
            'script-src',
            'style-src',
            'img-src',
            'connect-src',
            'font-src',
            'frame-src',
            'object-src',
            'base-uri',
            'form-action',
            'frame-ancestors',
            'other'
        )
    ),
    blocked_uri_category TEXT NOT NULL CHECK (
        blocked_uri_category IN (
            'self',
            'inline',
            'eval',
            'data',
            'blob',
            'http',
            'https',
            'none',
            'other'
        )
    ),
    document_origin TEXT NOT NULL DEFAULT '',
    status_code INTEGER CHECK (status_code IS NULL OR (status_code >= 0 AND status_code <= 999)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_csp_violation_reports_disposition
    ON csp_violation_reports (disposition, created_at);

CREATE INDEX idx_csp_violation_reports_effective_directive
    ON csp_violation_reports (effective_directive, created_at);
