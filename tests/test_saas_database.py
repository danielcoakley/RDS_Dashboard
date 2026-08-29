import json
import sqlite3

from backend.database import (
    DATABASE_URL_ENV,
    POSTGRES_BACKEND,
    SQLITE_BACKEND,
    PostgresConnectionAdapter,
    apply_migrations,
    database_config_from_env,
    database_config_from_url,
    initialize_database,
    initialize_database_from_config,
)


class FakePostgresConnection:
    def __init__(self):
        self.executed: list[tuple[str, tuple]] = []

    def execute(self, sql: str, params: tuple = ()):
        self.executed.append((sql, params))
        return self


def test_database_config_defaults_to_sqlite_memory(monkeypatch):
    monkeypatch.delenv(DATABASE_URL_ENV, raising=False)

    config = database_config_from_env()

    assert config.backend == SQLITE_BACKEND
    assert config.sqlite_path == ":memory:"


def test_database_config_parses_postgres_url():
    config = database_config_from_url("postgresql://rds:secret@db.example.com:5432/rds")

    assert config.backend == POSTGRES_BACKEND
    assert config.url == "postgresql://rds:secret@db.example.com:5432/rds"
    assert config.sqlite_path is None


def test_database_config_parses_sqlite_relative_path():
    config = database_config_from_url("sqlite:///./.saas/metadata.db")

    assert config.backend == SQLITE_BACKEND
    assert config.sqlite_path == "./.saas/metadata.db"


def test_database_config_rejects_unsupported_backend():
    try:
        database_config_from_url("mysql://example")
        assert False, "Expected unsupported database URL to fail"
    except ValueError as exc:
        assert DATABASE_URL_ENV in str(exc)


def test_initialize_database_from_sqlite_config_creates_schema_file(tmp_path):
    db_path = tmp_path / "metadata.db"
    config = database_config_from_url(f"sqlite:///{db_path.as_posix()}")

    conn = initialize_database_from_config(config)

    assert db_path.exists()
    assert conn.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] == 53


def test_postgres_connection_adapter_translates_sqlite_placeholders():
    raw_conn = FakePostgresConnection()
    adapter = PostgresConnectionAdapter(raw_conn)

    adapter.execute("SELECT * FROM users WHERE id = ? AND email = ?", ("user_1", "a@example.com"))
    adapter.executescript("SELECT 1;\n\nSELECT 2;")

    assert raw_conn.executed == [
        (
            "SELECT * FROM users WHERE id = %s AND email = %s",
            ("user_1", "a@example.com"),
        ),
        ("SELECT 1", ()),
        ("SELECT 2", ()),
    ]


def test_initialize_database_creates_tenant_skeleton_tables():
    conn = initialize_database()

    tables = {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    }

    assert {
        "schema_migrations",
        "users",
        "organizations",
        "memberships",
        "organization_invites",
        "sites",
        "meters",
        "analysis_jobs",
        "analysis_configs",
        "organization_data_connectors",
        "organization_data_connector_syncs",
        "notifications",
        "subscription_plans",
        "organization_subscriptions",
        "billing_events",
        "organization_report_branding",
        "organization_report_approval_policies",
        "organization_evidence_package_templates",
        "organization_privacy_settings",
        "organization_privacy_requests",
        "organization_pilot_readiness_items",
        "organization_report_comments",
        "organization_report_reviews",
        "organization_report_approval_assignments",
        "organization_report_schedules",
        "organization_report_schedule_deliveries",
        "marketing_inquiries",
        "marketing_inquiry_followup_events",
        "marketing_inquiry_submission_events",
        "worker_heartbeats",
        "platform_operation_events",
        "notification_provider_events",
        "csp_violation_reports",
    }.issubset(tables)


def test_migrations_are_idempotent():
    conn = initialize_database()

    assert apply_migrations(conn) == []
    versions = conn.execute("SELECT version FROM schema_migrations").fetchall()
    assert [row["version"] for row in versions] == [
        "001_tenant_skeleton",
        "002_runs_and_reports",
        "003_audit_events",
        "004_invites",
        "005_expand_audit_actions",
        "006_expand_audit_actions_for_invite_revoke",
        "007_analysis_jobs",
        "008_expand_audit_actions_for_report_access",
        "009_expand_audit_actions_for_config_changes",
        "010_notifications",
        "011_notification_delivery_metadata",
        "012_expand_audit_actions_for_organization_profile",
        "013_analysis_configs",
        "014_expand_audit_actions_for_membership_admin",
        "015_subscriptions",
        "016_billing_provider_events",
        "017_report_retention",
        "018_privacy_settings",
        "019_privacy_requests",
        "020_pilot_readiness",
        "021_report_comments",
        "022_report_exports",
        "023_report_schedules",
        "024_report_schedule_deliveries",
        "025_report_reviews",
        "026_report_evidence_packages",
        "027_report_evidence_package_snapshots",
        "028_report_approval_assignments",
        "029_organization_report_branding",
        "030_marketing_inquiries",
        "031_marketing_inquiry_followup",
        "032_marketing_inquiry_followup_events",
        "033_marketing_inquiry_submission_events",
        "034_marketing_inquiry_submission_outcomes",
        "035_worker_heartbeats",
        "036_notification_retry_metadata",
        "037_platform_operation_events",
        "038_notification_provider_events",
        "039_csp_violation_reports",
        "040_expand_platform_operation_event_types",
        "041_expand_platform_operation_provider_replay_events",
        "042_expand_platform_operation_artifact_store_test",
        "043_expand_platform_operation_database_test",
        "044_expand_platform_operation_tenant_isolation_test",
        "045_expand_platform_operation_run_lifecycle_test",
        "046_expand_platform_operation_worker_run_lifecycle_test",
        "047_report_approval_policies",
        "048_evidence_package_templates",
        "049_evidence_package_template_layout",
        "050_data_connectors",
        "051_data_connector_syncs",
        "052_worker_connector_heartbeat_counts",
        "053_site_location",
    ]


def test_csp_violation_reports_schema_tracks_sanitized_browser_reports():
    conn = initialize_database()
    conn.execute(
        """
        INSERT INTO csp_violation_reports (
            id,
            disposition,
            effective_directive,
            violated_directive,
            blocked_uri_category,
            document_origin,
            status_code,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "csp_report_1",
            "report",
            "script-src",
            "script-src",
            "https",
            "https://tenant.example.com",
            200,
            "2026-07-20T12:00:00+00:00",
        ),
    )

    report = conn.execute(
        """
        SELECT
            disposition,
            effective_directive,
            violated_directive,
            blocked_uri_category,
            document_origin,
            status_code,
            created_at
        FROM csp_violation_reports
        WHERE id = ?
        """,
        ("csp_report_1",),
    ).fetchone()

    assert report["disposition"] == "report"
    assert report["effective_directive"] == "script-src"
    assert report["violated_directive"] == "script-src"
    assert report["blocked_uri_category"] == "https"
    assert report["document_origin"] == "https://tenant.example.com"
    assert report["status_code"] == 200
    assert report["created_at"] == "2026-07-20T12:00:00+00:00"


def test_platform_operation_events_schema_tracks_sanitized_evidence():
    conn = initialize_database()
    conn.execute(
        """
        INSERT INTO platform_operation_events (
            id,
            event_type,
            status,
            request_id,
            detail,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "platform_operation_1",
            "rds.alert_delivery_test",
            "succeeded",
            "req_alert_1",
            "http_status=204",
            "2026-07-20T12:00:00+00:00",
        ),
    )

    event = conn.execute(
        """
        SELECT event_type, status, request_id, detail, created_at
        FROM platform_operation_events
        WHERE id = ?
        """,
        ("platform_operation_1",),
    ).fetchone()

    assert event["event_type"] == "rds.alert_delivery_test"
    assert event["status"] == "succeeded"
    assert event["request_id"] == "req_alert_1"
    assert event["detail"] == "http_status=204"
    assert event["created_at"] == "2026-07-20T12:00:00+00:00"

    conn.execute(
        """
        INSERT INTO platform_operation_events (
            id,
            event_type,
            status,
            request_id,
            detail,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "platform_operation_2",
            "rds.report_schedule_dispatch",
            "failed",
            None,
            "scope=all scheduled_attempted=2 scheduled_sent=1 scheduled_failed=1 scheduled_skipped=0",
            "2026-07-20T12:05:00+00:00",
        ),
    )

    dispatch_event = conn.execute(
        """
        SELECT event_type, status, request_id, detail, created_at
        FROM platform_operation_events
        WHERE id = ?
        """,
        ("platform_operation_2",),
    ).fetchone()

    assert dispatch_event["event_type"] == "rds.report_schedule_dispatch"
    assert dispatch_event["status"] == "failed"
    assert dispatch_event["request_id"] is None
    assert dispatch_event["detail"].startswith("scope=all")

    conn.execute(
        """
        INSERT INTO platform_operation_events (
            id,
            event_type,
            status,
            request_id,
            detail,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "platform_operation_3",
            "rds.clerk_webhook_replay",
            "succeeded",
            "req_clerk_replay",
            "action=membership_upserted",
            "2026-07-20T12:06:00+00:00",
        ),
    )

    provider_event = conn.execute(
        """
        SELECT event_type, status, request_id, detail
        FROM platform_operation_events
        WHERE id = ?
        """,
        ("platform_operation_3",),
    ).fetchone()

    assert provider_event["event_type"] == "rds.clerk_webhook_replay"
    assert provider_event["status"] == "succeeded"
    assert provider_event["request_id"] == "req_clerk_replay"
    assert provider_event["detail"] == "action=membership_upserted"

    conn.execute(
        """
        INSERT INTO platform_operation_events (
            id,
            event_type,
            status,
            request_id,
            detail,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "platform_operation_4",
            "rds.artifact_store_test",
            "succeeded",
            "req_artifact_store_test",
            "bytes_written=64",
            "2026-07-20T12:07:00+00:00",
        ),
    )

    artifact_event = conn.execute(
        """
        SELECT event_type, status, request_id, detail
        FROM platform_operation_events
        WHERE id = ?
        """,
        ("platform_operation_4",),
    ).fetchone()

    assert artifact_event["event_type"] == "rds.artifact_store_test"
    assert artifact_event["status"] == "succeeded"
    assert artifact_event["request_id"] == "req_artifact_store_test"
    assert artifact_event["detail"] == "bytes_written=64"

    conn.execute(
        """
        INSERT INTO platform_operation_events (
            id,
            event_type,
            status,
            request_id,
            detail,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "platform_operation_5",
            "rds.database_test",
            "succeeded",
            "req_database_test",
            "write_read_verified=true",
            "2026-07-20T12:08:00+00:00",
        ),
    )

    database_event = conn.execute(
        """
        SELECT event_type, status, request_id, detail
        FROM platform_operation_events
        WHERE id = ?
        """,
        ("platform_operation_5",),
    ).fetchone()

    assert database_event["event_type"] == "rds.database_test"
    assert database_event["status"] == "succeeded"
    assert database_event["request_id"] == "req_database_test"
    assert database_event["detail"] == "write_read_verified=true"

    conn.execute(
        """
        INSERT INTO platform_operation_events (
            id,
            event_type,
            status,
            request_id,
            detail,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "platform_operation_6",
            "rds.run_lifecycle_smoke_test",
            "succeeded",
            "req_run_lifecycle_test",
            "checks_passed=6 artifacts_verified=2 synthetic_records=15",
            "2026-07-20T12:09:00+00:00",
        ),
    )

    run_lifecycle_event = conn.execute(
        """
        SELECT event_type, status, request_id, detail
        FROM platform_operation_events
        WHERE id = ?
        """,
        ("platform_operation_6",),
    ).fetchone()

    assert run_lifecycle_event["event_type"] == "rds.run_lifecycle_smoke_test"
    assert run_lifecycle_event["status"] == "succeeded"
    assert run_lifecycle_event["request_id"] == "req_run_lifecycle_test"
    assert run_lifecycle_event["detail"] == (
        "checks_passed=6 artifacts_verified=2 synthetic_records=15"
    )

    conn.execute(
        """
        INSERT INTO platform_operation_events (
            id,
            event_type,
            status,
            request_id,
            detail,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "platform_operation_7",
            "rds.worker_run_lifecycle_smoke_test",
            "succeeded",
            "req_worker_run_lifecycle_test",
            (
                "checks_passed=8 jobs_attempted=1 jobs_succeeded=1 "
                "artifacts_verified=2 synthetic_records=15"
            ),
            "2026-07-20T12:10:00+00:00",
        ),
    )

    worker_run_lifecycle_event = conn.execute(
        """
        SELECT event_type, status, request_id, detail
        FROM platform_operation_events
        WHERE id = ?
        """,
        ("platform_operation_7",),
    ).fetchone()

    assert worker_run_lifecycle_event["event_type"] == "rds.worker_run_lifecycle_smoke_test"
    assert worker_run_lifecycle_event["status"] == "succeeded"
    assert worker_run_lifecycle_event["request_id"] == "req_worker_run_lifecycle_test"
    assert worker_run_lifecycle_event["detail"] == (
        "checks_passed=8 jobs_attempted=1 jobs_succeeded=1 "
        "artifacts_verified=2 synthetic_records=15"
    )


def test_notification_retry_metadata_schema_defaults_and_updates():
    conn = initialize_database()
    conn.execute(
        "INSERT INTO users (id, email, display_name) VALUES (?, ?, ?)",
        ("user_1", "owner@example.com", "Owner"),
    )
    conn.execute(
        "INSERT INTO organizations (id, name, slug) VALUES (?, ?, ?)",
        ("org_1", "Example Energy", "example-energy"),
    )
    conn.execute(
        "INSERT INTO memberships (user_id, organization_id, role) VALUES (?, ?, ?)",
        ("user_1", "org_1", "owner"),
    )
    conn.execute(
        """
        INSERT INTO notifications (
            id,
            organization_id,
            recipient_user_id,
            channel,
            event_type,
            subject,
            body,
            resource_type,
            resource_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "notification_1",
            "org_1",
            "user_1",
            "email",
            "run_succeeded",
            "Run complete",
            "Run run_1 is ready.",
            "run",
            "run_1",
        ),
    )

    notification = conn.execute(
        """
        SELECT attempt_count, last_attempt_at, next_attempt_at
        FROM notifications
        WHERE id = ?
        """,
        ("notification_1",),
    ).fetchone()

    assert notification["attempt_count"] == 0
    assert notification["last_attempt_at"] is None
    assert notification["next_attempt_at"] is None


def test_organization_profile_audit_action_is_allowed():
    conn = initialize_database()
    conn.execute(
        "INSERT INTO users (id, email, display_name) VALUES (?, ?, ?)",
        ("user_1", "owner@example.com", "Owner"),
    )
    conn.execute(
        "INSERT INTO organizations (id, name, slug) VALUES (?, ?, ?)",
        ("org_1", "Example Energy", "example-energy"),
    )

    conn.execute(
        """
        INSERT INTO audit_events (
            id,
            organization_id,
            actor_user_id,
            action,
            resource_type,
            resource_id
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "audit_org_update",
            "org_1",
            "user_1",
            "organization_updated",
            "organization",
            "org_1",
        ),
    )

    event = conn.execute("SELECT action FROM audit_events WHERE id = ?", ("audit_org_update",)).fetchone()
    assert event["action"] == "organization_updated"


def test_report_branding_table_is_tenant_scoped_and_auditable():
    conn = initialize_database()
    conn.execute(
        "INSERT INTO users (id, email, display_name) VALUES (?, ?, ?)",
        ("user_1", "owner@example.com", "Owner"),
    )
    conn.execute(
        "INSERT INTO organizations (id, name, slug) VALUES (?, ?, ?)",
        ("org_1", "Example Energy", "example-energy"),
    )
    conn.execute(
        """
        INSERT INTO organization_report_branding (
            organization_id,
            evidence_pack_title,
            primary_color,
            logo_url,
            support_email
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            "org_1",
            "Example Energy Audit Pack",
            "#123abc",
            "https://cdn.example.com/logo.png",
            "support@example.com",
        ),
    )
    conn.execute(
        """
        INSERT INTO audit_events (
            id,
            organization_id,
            actor_user_id,
            action,
            resource_type,
            resource_id
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "audit_report_branding_update",
            "org_1",
            "user_1",
            "report_branding_updated",
            "report_branding",
            "org_1",
        ),
    )

    branding = conn.execute(
        "SELECT evidence_pack_title, primary_color FROM organization_report_branding WHERE organization_id = ?",
        ("org_1",),
    ).fetchone()
    event = conn.execute(
        "SELECT action FROM audit_events WHERE id = ?",
        ("audit_report_branding_update",),
    ).fetchone()
    assert branding["evidence_pack_title"] == "Example Energy Audit Pack"
    assert branding["primary_color"] == "#123abc"
    assert event["action"] == "report_branding_updated"


def test_report_approval_policy_table_is_tenant_scoped_and_auditable():
    conn = initialize_database()
    conn.execute(
        "INSERT INTO users (id, email, display_name) VALUES (?, ?, ?)",
        ("user_1", "owner@example.com", "Owner"),
    )
    conn.execute(
        "INSERT INTO organizations (id, name, slug) VALUES (?, ?, ?)",
        ("org_1", "Example Energy", "example-energy"),
    )
    conn.execute(
        """
        INSERT INTO organization_report_approval_policies (
            organization_id,
            auto_lock_evidence_on_final_approval,
            minimum_approval_assignments_for_lock,
            updated_by_user_id
        )
        VALUES (?, ?, ?, ?)
        """,
        ("org_1", 0, 2, "user_1"),
    )
    conn.execute(
        """
        INSERT INTO audit_events (
            id,
            organization_id,
            actor_user_id,
            action,
            resource_type,
            resource_id
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "audit_report_approval_policy_update",
            "org_1",
            "user_1",
            "report_approval_policy_updated",
            "report_approval_policy",
            "org_1",
        ),
    )

    policy = conn.execute(
        """
        SELECT
            auto_lock_evidence_on_final_approval,
            minimum_approval_assignments_for_lock,
            updated_by_user_id
        FROM organization_report_approval_policies
        WHERE organization_id = ?
        """,
        ("org_1",),
    ).fetchone()
    event = conn.execute(
        "SELECT action FROM audit_events WHERE id = ?",
        ("audit_report_approval_policy_update",),
    ).fetchone()
    assert policy["auto_lock_evidence_on_final_approval"] == 0
    assert policy["minimum_approval_assignments_for_lock"] == 2
    assert policy["updated_by_user_id"] == "user_1"
    assert event["action"] == "report_approval_policy_updated"


def test_evidence_package_template_table_is_tenant_scoped_and_auditable():
    conn = initialize_database()
    conn.execute(
        "INSERT INTO users (id, email, display_name) VALUES (?, ?, ?)",
        ("user_1", "owner@example.com", "Owner"),
    )
    conn.execute(
        "INSERT INTO organizations (id, name, slug) VALUES (?, ?, ?)",
        ("org_1", "Example Energy", "example-energy"),
    )
    conn.execute(
        """
        INSERT INTO organization_evidence_package_templates (
            organization_id,
            template_name,
            include_report_artifact,
            include_uploads,
            include_approval_assignments,
            include_reviews,
            include_comments,
            include_schedules,
            section_labels_json,
            section_order_json,
            updated_by_user_id
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "org_1",
            "Board summary pack",
            0,
            1,
            1,
            1,
            0,
            0,
            '{"approval_assignments":"Board sign-off"}',
            '["approval_assignments","uploads"]',
            "user_1",
        ),
    )
    conn.execute(
        """
        INSERT INTO audit_events (
            id,
            organization_id,
            actor_user_id,
            action,
            resource_type,
            resource_id
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "audit_report_evidence_template_update",
            "org_1",
            "user_1",
            "report_evidence_template_updated",
            "report_evidence_template",
            "org_1",
        ),
    )

    template = conn.execute(
        """
        SELECT
            template_name,
            include_report_artifact,
            include_comments,
            include_schedules,
            section_labels_json,
            section_order_json,
            updated_by_user_id
        FROM organization_evidence_package_templates
        WHERE organization_id = ?
        """,
        ("org_1",),
    ).fetchone()
    event = conn.execute(
        "SELECT action FROM audit_events WHERE id = ?",
        ("audit_report_evidence_template_update",),
    ).fetchone()
    assert template["template_name"] == "Board summary pack"
    assert template["include_report_artifact"] == 0
    assert template["include_comments"] == 0
    assert template["include_schedules"] == 0
    assert template["section_labels_json"] == '{"approval_assignments":"Board sign-off"}'
    assert template["section_order_json"] == '["approval_assignments","uploads"]'
    assert template["updated_by_user_id"] == "user_1"
    assert event["action"] == "report_evidence_template_updated"


def test_data_connectors_table_is_tenant_scoped_and_auditable():
    conn = initialize_database()
    conn.execute(
        "INSERT INTO users (id, email, display_name) VALUES (?, ?, ?)",
        ("user_1", "owner@example.com", "Owner"),
    )
    conn.execute(
        "INSERT INTO organizations (id, name, slug) VALUES (?, ?, ?)",
        ("org_1", "Example Energy", "example-energy"),
    )
    conn.execute(
        "INSERT INTO sites (id, organization_id, name, timezone) VALUES (?, ?, ?, ?)",
        ("site_1", "org_1", "Main Site", "Europe/London"),
    )
    conn.execute(
        """
        INSERT INTO organization_data_connectors (
            id,
            organization_id,
            site_id,
            name,
            connector_type,
            source_category,
            status,
            schedule,
            external_reference,
            credential_reference,
            created_by_user_id,
            updated_by_user_id,
            next_sync_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "connector_1",
            "org_1",
            "site_1",
            "Main utility SFTP",
            "sftp",
            "energy",
            "draft",
            "weekly",
            "sftp://example/path",
            "secret/data-connectors/org_1/connector_1",
            "user_1",
            "user_1",
            "2026-08-01T09:00:00+00:00",
        ),
    )
    conn.execute(
        """
        INSERT INTO audit_events (
            id,
            organization_id,
            actor_user_id,
            action,
            resource_type,
            resource_id
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "audit_data_connector_created",
            "org_1",
            "user_1",
            "data_connector_created",
            "data_connector",
            "connector_1",
        ),
    )

    connector = conn.execute(
        """
        SELECT
            organization_id,
            site_id,
            connector_type,
            source_category,
            status,
            schedule,
            credential_reference,
            next_sync_at
        FROM organization_data_connectors
        WHERE id = ?
        """,
        ("connector_1",),
    ).fetchone()
    event = conn.execute(
        "SELECT action FROM audit_events WHERE id = ?",
        ("audit_data_connector_created",),
    ).fetchone()

    assert connector["organization_id"] == "org_1"
    assert connector["site_id"] == "site_1"
    assert connector["connector_type"] == "sftp"
    assert connector["source_category"] == "energy"
    assert connector["status"] == "draft"
    assert connector["schedule"] == "weekly"
    assert connector["credential_reference"] == "secret/data-connectors/org_1/connector_1"
    assert connector["next_sync_at"] == "2026-08-01T09:00:00+00:00"
    assert event["action"] == "data_connector_created"


def test_data_connector_syncs_table_tracks_import_lifecycle_without_secret_metadata():
    conn = initialize_database()
    conn.execute(
        "INSERT INTO users (id, email, display_name) VALUES (?, ?, ?)",
        ("user_1", "owner@example.com", "Owner"),
    )
    conn.execute(
        "INSERT INTO organizations (id, name, slug) VALUES (?, ?, ?)",
        ("org_1", "Example Energy", "example-energy"),
    )
    conn.execute(
        "INSERT INTO sites (id, organization_id, name, timezone) VALUES (?, ?, ?, ?)",
        ("site_1", "org_1", "Main Site", "Europe/London"),
    )
    conn.execute(
        """
        INSERT INTO uploads (
            id,
            organization_id,
            site_id,
            uploaded_by_user_id,
            category,
            storage_key,
            checksum,
            status
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "upload_1",
            "org_1",
            "site_1",
            "user_1",
            "energy",
            "tenants/org_1/sites/site_1/uploads/upload_1/energy.csv",
            "abc123",
            "stored",
        ),
    )
    conn.execute(
        """
        INSERT INTO organization_data_connectors (
            id,
            organization_id,
            site_id,
            name,
            connector_type,
            source_category,
            status,
            schedule,
            created_by_user_id,
            next_sync_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "connector_1",
            "org_1",
            "site_1",
            "Utility API",
            "api",
            "energy",
            "active",
            "daily",
            "user_1",
            "2026-08-01T09:00:00+00:00",
        ),
    )
    conn.execute(
        """
        INSERT INTO organization_data_connector_syncs (
            id,
            organization_id,
            connector_id,
            site_id,
            source_category,
            status,
            upload_id,
            records_processed,
            status_reason,
            triggered_by_user_id,
            started_at,
            completed_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "sync_1",
            "org_1",
            "connector_1",
            "site_1",
            "energy",
            "succeeded",
            "upload_1",
            42,
            "Imported interval data",
            "user_1",
            "2026-08-01T09:00:00+00:00",
            "2026-08-01T09:01:00+00:00",
        ),
    )
    conn.execute(
        """
        UPDATE organization_data_connectors
        SET last_sync_at = ?,
            last_sync_status = ?,
            next_sync_at = ?
        WHERE id = ?
        """,
        (
            "2026-08-01T09:01:00+00:00",
            "succeeded",
            "2026-08-02T09:01:00+00:00",
            "connector_1",
        ),
    )
    conn.execute(
        """
        INSERT INTO audit_events (
            id,
            organization_id,
            actor_user_id,
            action,
            resource_type,
            resource_id,
            metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "audit_data_connector_sync",
            "org_1",
            "user_1",
            "data_connector_sync_recorded",
            "data_connector",
            "connector_1",
            json.dumps(
                {
                    "connector_type": "api",
                    "records_processed": 42,
                    "schedule": "daily",
                    "source_category": "energy",
                    "status": "succeeded",
                    "status_reason_configured": True,
                    "upload_configured": True,
                },
                sort_keys=True,
            ),
        ),
    )

    sync = conn.execute(
        """
        SELECT status, upload_id, records_processed, status_reason
        FROM organization_data_connector_syncs
        WHERE id = ?
        """,
        ("sync_1",),
    ).fetchone()
    connector = conn.execute(
        """
        SELECT last_sync_status, last_sync_at, next_sync_at
        FROM organization_data_connectors
        WHERE id = ?
        """,
        ("connector_1",),
    ).fetchone()
    event = conn.execute(
        "SELECT action, metadata_json FROM audit_events WHERE id = ?",
        ("audit_data_connector_sync",),
    ).fetchone()

    assert sync["status"] == "succeeded"
    assert sync["upload_id"] == "upload_1"
    assert sync["records_processed"] == 42
    assert sync["status_reason"] == "Imported interval data"
    assert connector["last_sync_status"] == "succeeded"
    assert connector["last_sync_at"] == "2026-08-01T09:01:00+00:00"
    assert connector["next_sync_at"] == "2026-08-02T09:01:00+00:00"
    assert event["action"] == "data_connector_sync_recorded"
    assert "tenants/org_1" not in event["metadata_json"]
    assert "Utility API" not in event["metadata_json"]


def test_analysis_config_table_is_tenant_scoped_and_auditable():
    conn = initialize_database()
    conn.execute(
        "INSERT INTO users (id, email, display_name) VALUES (?, ?, ?)",
        ("user_1", "owner@example.com", "Owner"),
    )
    conn.execute(
        "INSERT INTO organizations (id, name, slug) VALUES (?, ?, ?)",
        ("org_1", "Example Energy", "example-energy"),
    )
    conn.execute(
        """
        INSERT INTO analysis_configs (organization_id, config_json, updated_by_user_id)
        VALUES (?, ?, ?)
        """,
        ("org_1", '{"client":{"id":"org_1"}}', "user_1"),
    )
    conn.execute(
        """
        INSERT INTO audit_events (
            id,
            organization_id,
            actor_user_id,
            action,
            resource_type,
            resource_id
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "audit_analysis_config_update",
            "org_1",
            "user_1",
            "analysis_config_updated",
            "analysis_config",
            "org_1",
        ),
    )

    config = conn.execute(
        "SELECT config_json FROM analysis_configs WHERE organization_id = ?",
        ("org_1",),
    ).fetchone()
    event = conn.execute(
        "SELECT action FROM audit_events WHERE id = ?",
        ("audit_analysis_config_update",),
    ).fetchone()
    assert config["config_json"] == '{"client":{"id":"org_1"}}'
    assert event["action"] == "analysis_config_updated"


def test_membership_admin_audit_actions_are_allowed():
    conn = initialize_database()
    conn.execute(
        "INSERT INTO users (id, email, display_name) VALUES (?, ?, ?)",
        ("user_1", "owner@example.com", "Owner"),
    )
    conn.execute(
        "INSERT INTO organizations (id, name, slug) VALUES (?, ?, ?)",
        ("org_1", "Example Energy", "example-energy"),
    )

    for action in ["membership_role_updated", "membership_removed"]:
        conn.execute(
            """
            INSERT INTO audit_events (
                id,
                organization_id,
                actor_user_id,
                action,
                resource_type,
                resource_id
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                f"audit_{action}",
                "org_1",
                "user_1",
                action,
                "membership",
                "user_2",
            ),
        )

    actions = [
        row["action"]
        for row in conn.execute(
            "SELECT action FROM audit_events ORDER BY id",
        ).fetchall()
    ]
    assert actions == ["membership_removed", "membership_role_updated"]


def test_subscription_schema_has_seeded_plans_and_auditable_updates():
    conn = initialize_database()
    conn.execute(
        "INSERT INTO users (id, email, display_name) VALUES (?, ?, ?)",
        ("user_1", "owner@example.com", "Owner"),
    )
    conn.execute(
        "INSERT INTO organizations (id, name, slug) VALUES (?, ?, ?)",
        ("org_1", "Example Energy", "example-energy"),
    )
    conn.execute(
        """
        INSERT INTO organization_subscriptions (
            organization_id,
            plan_id,
            status,
            current_period_start
        )
        VALUES (?, ?, ?, ?)
        """,
        ("org_1", "trial", "trialing", "2026-07-20T00:00:00+00:00"),
    )
    conn.execute(
        """
        INSERT INTO audit_events (
            id,
            organization_id,
            actor_user_id,
            action,
            resource_type,
            resource_id
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "audit_subscription_update",
            "org_1",
            "user_1",
            "subscription_updated",
            "subscription",
            "org_1",
        ),
    )

    plans = conn.execute("SELECT id FROM subscription_plans ORDER BY id").fetchall()
    subscription = conn.execute(
        "SELECT plan_id, status FROM organization_subscriptions WHERE organization_id = ?",
        ("org_1",),
    ).fetchone()
    event = conn.execute(
        "SELECT action FROM audit_events WHERE id = ?",
        ("audit_subscription_update",),
    ).fetchone()

    assert [plan["id"] for plan in plans] == ["enterprise", "team", "trial"]
    assert subscription["plan_id"] == "trial"
    assert subscription["status"] == "trialing"
    assert event["action"] == "subscription_updated"


def test_report_retention_schema_and_audit_action_are_allowed():
    conn = initialize_database()
    conn.execute(
        "INSERT INTO users (id, email, display_name) VALUES (?, ?, ?)",
        ("user_1", "owner@example.com", "Owner"),
    )
    conn.execute(
        "INSERT INTO organizations (id, name, slug) VALUES (?, ?, ?)",
        ("org_1", "Example Energy", "example-energy"),
    )
    conn.execute(
        "INSERT INTO sites (id, organization_id, name, timezone) VALUES (?, ?, ?, ?)",
        ("site_1", "org_1", "Main Site", "Europe/London"),
    )
    conn.execute(
        """
        INSERT INTO runs (
            id,
            organization_id,
            site_id,
            requested_by_user_id,
            status
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        ("run_1", "org_1", "site_1", "user_1", "succeeded"),
    )
    conn.execute(
        """
        INSERT INTO reports (
            id,
            organization_id,
            run_id,
            report_type,
            storage_key,
            is_published,
            expires_at,
            retention_applied_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "report_1",
            "org_1",
            "run_1",
            "iso_summary",
            "tenants/org_1/sites/site_1/runs/run_1/reports/iso-summary.json",
            0,
            "2026-08-19T00:00:00+00:00",
            "2026-08-20T00:00:00+00:00",
        ),
    )
    conn.execute(
        """
        INSERT INTO audit_events (
            id,
            organization_id,
            actor_user_id,
            action,
            resource_type,
            resource_id
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "audit_report_retention",
            "org_1",
            "user_1",
            "report_retention_applied",
            "report",
            "report_1",
        ),
    )

    report = conn.execute(
        """
        SELECT expires_at, retention_applied_at
        FROM reports
        WHERE id = ?
        """,
        ("report_1",),
    ).fetchone()
    event = conn.execute(
        "SELECT action FROM audit_events WHERE id = ?",
        ("audit_report_retention",),
    ).fetchone()

    assert report["expires_at"] == "2026-08-19T00:00:00+00:00"
    assert report["retention_applied_at"] == "2026-08-20T00:00:00+00:00"
    assert event["action"] == "report_retention_applied"


def test_privacy_settings_schema_and_audit_action_are_allowed():
    conn = initialize_database()
    conn.execute(
        "INSERT INTO users (id, email, display_name) VALUES (?, ?, ?)",
        ("user_1", "owner@example.com", "Owner"),
    )
    conn.execute(
        "INSERT INTO organizations (id, name, slug) VALUES (?, ?, ?)",
        ("org_1", "Example Energy", "example-energy"),
    )
    conn.execute(
        """
        INSERT INTO organization_privacy_settings (
            organization_id,
            data_processing_acknowledged_at,
            data_processing_acknowledged_by_user_id,
            data_processing_contact_email,
            subprocessor_notifications_enabled,
            analytics_data_sharing_enabled
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "org_1",
            "2026-07-20T00:00:00+00:00",
            "user_1",
            "privacy@example.com",
            1,
            0,
        ),
    )
    conn.execute(
        """
        INSERT INTO audit_events (
            id,
            organization_id,
            actor_user_id,
            action,
            resource_type,
            resource_id
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "audit_privacy_settings",
            "org_1",
            "user_1",
            "privacy_settings_updated",
            "privacy_settings",
            "org_1",
        ),
    )

    settings = conn.execute(
        """
        SELECT
            data_processing_acknowledged_at,
            data_processing_acknowledged_by_user_id,
            data_processing_contact_email,
            subprocessor_notifications_enabled,
            analytics_data_sharing_enabled
        FROM organization_privacy_settings
        WHERE organization_id = ?
        """,
        ("org_1",),
    ).fetchone()
    event = conn.execute(
        "SELECT action FROM audit_events WHERE id = ?",
        ("audit_privacy_settings",),
    ).fetchone()

    assert settings["data_processing_acknowledged_at"] == "2026-07-20T00:00:00+00:00"
    assert settings["data_processing_acknowledged_by_user_id"] == "user_1"
    assert settings["data_processing_contact_email"] == "privacy@example.com"
    assert settings["subprocessor_notifications_enabled"] == 1
    assert settings["analytics_data_sharing_enabled"] == 0
    assert event["action"] == "privacy_settings_updated"


def test_privacy_request_schema_and_audit_actions_are_allowed():
    conn = initialize_database()
    conn.execute(
        "INSERT INTO users (id, email, display_name) VALUES (?, ?, ?)",
        ("user_1", "owner@example.com", "Owner"),
    )
    conn.execute(
        "INSERT INTO users (id, email, display_name) VALUES (?, ?, ?)",
        ("user_2", "member@example.com", "Member"),
    )
    conn.execute(
        "INSERT INTO organizations (id, name, slug) VALUES (?, ?, ?)",
        ("org_1", "Example Energy", "example-energy"),
    )
    conn.execute(
        """
        INSERT INTO organization_privacy_requests (
            id,
            organization_id,
            requested_by_user_id,
            subject_user_id,
            request_type,
            status,
            description,
            resolution_notes,
            completed_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "privacy_request_1",
            "org_1",
            "user_1",
            "user_2",
            "deletion",
            "completed",
            "Delete member data",
            "Handled manually",
            "2026-07-20T00:00:00+00:00",
        ),
    )

    for action in ["privacy_request_created", "privacy_request_updated"]:
        conn.execute(
            """
            INSERT INTO audit_events (
                id,
                organization_id,
                actor_user_id,
                action,
                resource_type,
                resource_id
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                f"audit_{action}",
                "org_1",
                "user_1",
                action,
                "privacy_request",
                "privacy_request_1",
            ),
        )

    request = conn.execute(
        """
        SELECT request_type, status, subject_user_id, completed_at
        FROM organization_privacy_requests
        WHERE id = ?
        """,
        ("privacy_request_1",),
    ).fetchone()
    actions = [
        row["action"]
        for row in conn.execute(
            "SELECT action FROM audit_events ORDER BY id",
        ).fetchall()
    ]

    assert request["request_type"] == "deletion"
    assert request["status"] == "completed"
    assert request["subject_user_id"] == "user_2"
    assert request["completed_at"] == "2026-07-20T00:00:00+00:00"
    assert actions == ["privacy_request_created", "privacy_request_updated"]


def test_pilot_readiness_schema_and_audit_action_are_allowed():
    conn = initialize_database()
    conn.execute(
        "INSERT INTO users (id, email, display_name) VALUES (?, ?, ?)",
        ("user_1", "owner@example.com", "Owner"),
    )
    conn.execute(
        "INSERT INTO organizations (id, name, slug) VALUES (?, ?, ?)",
        ("org_1", "Example Energy", "example-energy"),
    )
    conn.execute(
        """
        INSERT INTO organization_pilot_readiness_items (
            organization_id,
            item_key,
            status,
            evidence_url,
            notes,
            updated_by_user_id
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "org_1",
            "storage_backup_verified",
            "verified",
            "https://ops.example.com/restore-test",
            "Restore tested",
            "user_1",
        ),
    )
    conn.execute(
        """
        INSERT INTO audit_events (
            id,
            organization_id,
            actor_user_id,
            action,
            resource_type,
            resource_id
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "audit_pilot_readiness",
            "org_1",
            "user_1",
            "pilot_readiness_updated",
            "pilot_readiness",
            "storage_backup_verified",
        ),
    )

    item = conn.execute(
        """
        SELECT status, evidence_url, notes, updated_by_user_id
        FROM organization_pilot_readiness_items
        WHERE organization_id = ? AND item_key = ?
        """,
        ("org_1", "storage_backup_verified"),
    ).fetchone()
    event = conn.execute(
        "SELECT action FROM audit_events WHERE id = ?",
        ("audit_pilot_readiness",),
    ).fetchone()

    assert item["status"] == "verified"
    assert item["evidence_url"] == "https://ops.example.com/restore-test"
    assert item["notes"] == "Restore tested"
    assert item["updated_by_user_id"] == "user_1"
    assert event["action"] == "pilot_readiness_updated"


def test_worker_heartbeats_schema_tracks_all_tenant_and_scoped_checkins():
    conn = initialize_database()
    conn.execute(
        """
        INSERT INTO worker_heartbeats (
            worker_id,
            organization_id,
            scope,
            poll_seconds,
            batch_size,
            last_attempted,
            last_succeeded,
            last_failed,
            last_skipped,
            last_scheduled_attempted,
            last_scheduled_sent,
            last_scheduled_failed,
            last_scheduled_skipped,
            last_connector_attempted,
            last_connector_succeeded,
            last_connector_failed,
            last_connector_skipped,
            last_seen_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "worker_all",
            None,
            "all",
            10,
            5,
            2,
            1,
            1,
            0,
            3,
            2,
            1,
            0,
            4,
            3,
            1,
            0,
            "2026-07-20T12:00:00+00:00",
        ),
    )
    conn.execute(
        """
        INSERT INTO worker_heartbeats (
            worker_id,
            organization_id,
            scope,
            poll_seconds,
            batch_size,
            last_seen_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "worker_org_1",
            "org_1",
            "organization",
            0,
            1,
            "2026-07-20T12:01:00+00:00",
        ),
    )

    rows = conn.execute(
        """
        SELECT
            worker_id,
            organization_id,
            scope,
            last_attempted,
            last_scheduled_sent,
            last_connector_attempted,
            last_connector_succeeded
        FROM worker_heartbeats
        ORDER BY worker_id
        """
    ).fetchall()

    assert [row["worker_id"] for row in rows] == ["worker_all", "worker_org_1"]
    assert rows[0]["scope"] == "all"
    assert rows[0]["organization_id"] is None
    assert rows[0]["last_attempted"] == 2
    assert rows[0]["last_scheduled_sent"] == 2
    assert rows[0]["last_connector_attempted"] == 4
    assert rows[0]["last_connector_succeeded"] == 3
    assert rows[1]["scope"] == "organization"
    assert rows[1]["organization_id"] == "org_1"
    assert rows[1]["last_connector_attempted"] == 0


def test_report_comments_schema_and_audit_action_are_allowed():
    conn = initialize_database()
    conn.execute(
        "INSERT INTO users (id, email, display_name) VALUES (?, ?, ?)",
        ("user_1", "owner@example.com", "Owner"),
    )
    conn.execute(
        "INSERT INTO users (id, email, display_name) VALUES (?, ?, ?)",
        ("user_2", "viewer@example.com", "Viewer"),
    )
    conn.execute(
        "INSERT INTO organizations (id, name, slug) VALUES (?, ?, ?)",
        ("org_1", "Example Energy", "example-energy"),
    )
    conn.execute(
        "INSERT INTO sites (id, organization_id, name, timezone) VALUES (?, ?, ?, ?)",
        ("site_1", "org_1", "Main Site", "Europe/London"),
    )
    conn.execute(
        """
        INSERT INTO runs (id, organization_id, site_id, requested_by_user_id, status)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("run_1", "org_1", "site_1", "user_1", "succeeded"),
    )
    conn.execute(
        """
        INSERT INTO reports (
            id,
            organization_id,
            run_id,
            report_type,
            storage_key,
            is_published
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "report_1",
            "org_1",
            "run_1",
            "iso_summary",
            "tenants/org_1/sites/site_1/runs/run_1/reports/iso-summary.json",
            1,
        ),
    )
    conn.execute(
        """
        INSERT INTO organization_report_comments (
            id,
            organization_id,
            report_id,
            author_user_id,
            body
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            "report_comment_1",
            "org_1",
            "report_1",
            "user_2",
            "Ready for management review",
        ),
    )
    conn.execute(
        """
        INSERT INTO audit_events (
            id,
            organization_id,
            actor_user_id,
            action,
            resource_type,
            resource_id
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "audit_report_comment",
            "org_1",
            "user_2",
            "report_comment_created",
            "report_comment",
            "report_comment_1",
        ),
    )

    comment = conn.execute(
        """
        SELECT report_id, author_user_id, body
        FROM organization_report_comments
        WHERE id = ?
        """,
        ("report_comment_1",),
    ).fetchone()
    event = conn.execute(
        "SELECT action FROM audit_events WHERE id = ?",
        ("audit_report_comment",),
    ).fetchone()

    assert comment["report_id"] == "report_1"
    assert comment["author_user_id"] == "user_2"
    assert comment["body"] == "Ready for management review"
    assert event["action"] == "report_comment_created"


def test_report_export_audit_actions_are_allowed():
    conn = initialize_database()
    conn.execute(
        "INSERT INTO users (id, email, display_name) VALUES (?, ?, ?)",
        ("user_1", "owner@example.com", "Owner"),
    )
    conn.execute(
        "INSERT INTO organizations (id, name, slug) VALUES (?, ?, ?)",
        ("org_1", "Example Energy", "example-energy"),
    )
    for action in [
        "report_exported",
        "report_evidence_exported",
        "report_evidence_package_created",
        "report_evidence_package_downloaded",
    ]:
        conn.execute(
            """
            INSERT INTO audit_events (
                id,
                organization_id,
                actor_user_id,
                action,
                resource_type,
                resource_id
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                f"audit_{action}",
                "org_1",
                "user_1",
                action,
                "report",
                "report_1",
            ),
        )

    actions = [
        row["action"]
        for row in conn.execute(
            "SELECT action FROM audit_events ORDER BY id",
        ).fetchall()
    ]

    assert actions == [
        "report_evidence_exported",
        "report_evidence_package_created",
        "report_evidence_package_downloaded",
        "report_exported",
    ]


def test_report_evidence_package_snapshots_are_tenant_and_report_scoped():
    conn = initialize_database()
    conn.execute(
        "INSERT INTO users (id, email, display_name) VALUES (?, ?, ?)",
        ("user_1", "owner@example.com", "Owner"),
    )
    conn.execute(
        "INSERT INTO organizations (id, name, slug) VALUES (?, ?, ?)",
        ("org_1", "Example Energy", "example-energy"),
    )
    conn.execute(
        "INSERT INTO sites (id, organization_id, name, timezone) VALUES (?, ?, ?, ?)",
        ("site_1", "org_1", "Main Site", "Europe/London"),
    )
    conn.execute(
        """
        INSERT INTO runs (id, organization_id, site_id, requested_by_user_id, status)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("run_1", "org_1", "site_1", "user_1", "succeeded"),
    )
    conn.execute(
        """
        INSERT INTO reports (
            id,
            organization_id,
            run_id,
            report_type,
            storage_key,
            is_published
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "report_1",
            "org_1",
            "run_1",
            "iso_summary",
            "tenants/org_1/sites/site_1/runs/run_1/reports/iso-summary.json",
            1,
        ),
    )
    conn.execute(
        """
        INSERT INTO organization_report_evidence_packages (
            id,
            organization_id,
            report_id,
            created_by_user_id,
            storage_key,
            filename,
            content_type,
            content_length,
            package_type,
            format,
            comment_count,
            review_count,
            schedule_count,
            delivery_count,
            upload_count
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "report_evidence_package_1",
            "org_1",
            "report_1",
            "user_1",
            (
                "tenants/org_1/reports/report_1/evidence-packages/"
                "report_evidence_package_1/report_1-iso-summary-evidence-package.json"
            ),
            "report_1-iso-summary-evidence-package.json",
            "application/json",
            1200,
            "iso50001_report_evidence",
            "json",
            2,
            1,
            1,
            3,
            1,
        ),
    )

    package = conn.execute(
        """
        SELECT *
        FROM organization_report_evidence_packages
        WHERE id = ?
        """,
        ("report_evidence_package_1",),
    ).fetchone()

    assert package["organization_id"] == "org_1"
    assert package["report_id"] == "report_1"
    assert package["created_by_user_id"] == "user_1"
    assert package["storage_key"].startswith("tenants/org_1/reports/report_1/")
    assert package["content_length"] == 1200
    assert package["comment_count"] == 2
    assert package["review_count"] == 1
    assert package["schedule_count"] == 1
    assert package["delivery_count"] == 3
    assert package["upload_count"] == 1


def test_report_schedules_schema_and_audit_actions_are_allowed():
    conn = initialize_database()
    conn.execute(
        "INSERT INTO users (id, email, display_name) VALUES (?, ?, ?)",
        ("user_1", "owner@example.com", "Owner"),
    )
    conn.execute(
        "INSERT INTO organizations (id, name, slug) VALUES (?, ?, ?)",
        ("org_1", "Example Energy", "example-energy"),
    )
    conn.execute(
        "INSERT INTO sites (id, organization_id, name, timezone) VALUES (?, ?, ?, ?)",
        ("site_1", "org_1", "Main Site", "Europe/London"),
    )
    conn.execute(
        """
        INSERT INTO runs (id, organization_id, site_id, requested_by_user_id, status)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("run_1", "org_1", "site_1", "user_1", "succeeded"),
    )
    conn.execute(
        """
        INSERT INTO reports (
            id,
            organization_id,
            run_id,
            report_type,
            storage_key,
            is_published
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "report_1",
            "org_1",
            "run_1",
            "iso_summary",
            "tenants/org_1/sites/site_1/runs/run_1/reports/iso-summary.json",
            1,
        ),
    )
    conn.execute(
        """
        INSERT INTO organization_report_schedules (
            id,
            organization_id,
            report_id,
            created_by_user_id,
            frequency,
            recipient_email,
            next_run_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "report_schedule_1",
            "org_1",
            "report_1",
            "user_1",
            "monthly",
            "energy.manager@example.com",
            "2030-01-01T09:00:00+00:00",
        ),
    )

    conn.execute(
        """
        INSERT INTO organization_report_schedule_deliveries (
            id,
            organization_id,
            schedule_id,
            report_id,
            recipient_email,
            status,
            attempted_at,
            sent_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "report_schedule_delivery_1",
            "org_1",
            "report_schedule_1",
            "report_1",
            "energy.manager@example.com",
            "sent",
            "2030-01-01T09:00:00+00:00",
            "2030-01-01T09:00:00+00:00",
        ),
    )
    conn.execute(
        """
        UPDATE organization_report_schedules
        SET last_delivery_at = ?,
            last_delivery_status = ?,
            last_delivery_error = ?
        WHERE id = ?
        """,
        (
            "2030-01-01T09:00:00+00:00",
            "sent",
            None,
            "report_schedule_1",
        ),
    )

    for action in [
        "report_schedule_created",
        "report_schedule_updated",
        "report_schedule_delivered",
        "report_schedule_failed",
    ]:
        conn.execute(
            """
            INSERT INTO audit_events (
                id,
                organization_id,
                actor_user_id,
                action,
                resource_type,
                resource_id
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                f"audit_{action}",
                "org_1",
                "user_1",
                action,
                "report_schedule",
                "report_schedule_1",
            ),
        )

    schedule = conn.execute(
        """
        SELECT
            frequency,
            recipient_email,
            next_run_at,
            is_active,
            last_delivery_at,
            last_delivery_status,
            last_delivery_error
        FROM organization_report_schedules
        WHERE id = ?
        """,
        ("report_schedule_1",),
    ).fetchone()
    delivery = conn.execute(
        """
        SELECT status, error_message, sent_at
        FROM organization_report_schedule_deliveries
        WHERE id = ?
        """,
        ("report_schedule_delivery_1",),
    ).fetchone()
    actions = [
        row["action"]
        for row in conn.execute(
            "SELECT action FROM audit_events ORDER BY id",
        ).fetchall()
    ]

    assert schedule["frequency"] == "monthly"
    assert schedule["recipient_email"] == "energy.manager@example.com"
    assert schedule["next_run_at"] == "2030-01-01T09:00:00+00:00"
    assert schedule["is_active"] == 1
    assert schedule["last_delivery_at"] == "2030-01-01T09:00:00+00:00"
    assert schedule["last_delivery_status"] == "sent"
    assert schedule["last_delivery_error"] is None
    assert delivery["status"] == "sent"
    assert delivery["error_message"] is None
    assert delivery["sent_at"] == "2030-01-01T09:00:00+00:00"
    assert actions == [
        "report_schedule_created",
        "report_schedule_delivered",
        "report_schedule_failed",
        "report_schedule_updated",
    ]


def test_report_reviews_schema_and_audit_action_are_allowed():
    conn = initialize_database()
    conn.execute(
        "INSERT INTO users (id, email, display_name) VALUES (?, ?, ?)",
        ("user_1", "owner@example.com", "Owner"),
    )
    conn.execute(
        "INSERT INTO organizations (id, name, slug) VALUES (?, ?, ?)",
        ("org_1", "Example Energy", "example-energy"),
    )
    conn.execute(
        "INSERT INTO sites (id, organization_id, name, timezone) VALUES (?, ?, ?, ?)",
        ("site_1", "org_1", "Main Site", "Europe/London"),
    )
    conn.execute(
        """
        INSERT INTO runs (id, organization_id, site_id, requested_by_user_id, status)
        VALUES (?, ?, ?, ?, ?)
        """,
        ("run_1", "org_1", "site_1", "user_1", "succeeded"),
    )
    conn.execute(
        """
        INSERT INTO reports (
            id,
            organization_id,
            run_id,
            report_type,
            storage_key,
            is_published
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "report_1",
            "org_1",
            "run_1",
            "iso_summary",
            "tenants/org_1/sites/site_1/runs/run_1/reports/iso-summary.json",
            1,
        ),
    )
    conn.execute(
        """
        INSERT INTO organization_report_reviews (
            id,
            organization_id,
            report_id,
            reviewer_user_id,
            status,
            notes
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "report_review_1",
            "org_1",
            "report_1",
            "user_1",
            "approved",
            "Approved for the July evidence package.",
        ),
    )
    conn.execute(
        """
        INSERT INTO audit_events (
            id,
            organization_id,
            actor_user_id,
            action,
            resource_type,
            resource_id
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            "audit_report_review",
            "org_1",
            "user_1",
            "report_review_recorded",
            "report_review",
            "report_review_1",
        ),
    )

    review = conn.execute(
        """
        SELECT status, notes
        FROM organization_report_reviews
        WHERE id = ?
        """,
        ("report_review_1",),
    ).fetchone()
    event = conn.execute(
        "SELECT action FROM audit_events WHERE id = ?",
        ("audit_report_review",),
    ).fetchone()

    assert review["status"] == "approved"
    assert review["notes"] == "Approved for the July evidence package."
    assert event["action"] == "report_review_recorded"


def test_membership_and_site_meter_records_are_tenant_scoped():
    conn = initialize_database()
    conn.execute(
        "INSERT INTO users (id, email, display_name) VALUES (?, ?, ?)",
        ("user_1", "owner@example.com", "Owner"),
    )
    conn.execute(
        "INSERT INTO organizations (id, name, slug) VALUES (?, ?, ?)",
        ("org_1", "Example Energy", "example-energy"),
    )
    conn.execute(
        "INSERT INTO memberships (user_id, organization_id, role) VALUES (?, ?, ?)",
        ("user_1", "org_1", "owner"),
    )
    conn.execute(
        "INSERT INTO sites (id, organization_id, name, timezone) VALUES (?, ?, ?, ?)",
        ("site_1", "org_1", "Main Site", "Europe/London"),
    )
    conn.execute(
        """
        INSERT INTO meters (
            id,
            organization_id,
            site_id,
            display_name,
            commodity,
            unit,
            source_column,
            is_seu
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ("meter_1", "org_1", "site_1", "Main Electricity", "electricity", "kWh", "Main Electricity", 1),
    )

    meter = conn.execute("SELECT * FROM meters WHERE id = ?", ("meter_1",)).fetchone()
    assert meter["organization_id"] == "org_1"
    assert meter["site_id"] == "site_1"
    assert meter["is_seu"] == 1


def test_meter_cannot_reference_site_from_another_tenant():
    conn = initialize_database()
    conn.execute(
        "INSERT INTO organizations (id, name, slug) VALUES (?, ?, ?)",
        ("org_1", "Example Energy", "example-energy"),
    )
    conn.execute(
        "INSERT INTO organizations (id, name, slug) VALUES (?, ?, ?)",
        ("org_2", "Other Energy", "other-energy"),
    )
    conn.execute(
        "INSERT INTO sites (id, organization_id, name, timezone) VALUES (?, ?, ?, ?)",
        ("site_1", "org_1", "Main Site", "Europe/London"),
    )

    try:
        conn.execute(
            """
            INSERT INTO meters (
                id,
                organization_id,
                site_id,
                display_name,
                commodity,
                unit,
                source_column
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("meter_1", "org_2", "site_1", "Leaked Meter", "electricity", "kWh", "Main Electricity"),
        )
        assert False, "Expected cross-tenant meter/site reference to fail"
    except sqlite3.IntegrityError:
        pass


def test_invalid_membership_role_is_rejected():
    conn = initialize_database()
    conn.execute(
        "INSERT INTO users (id, email, display_name) VALUES (?, ?, ?)",
        ("user_1", "owner@example.com", "Owner"),
    )
    conn.execute(
        "INSERT INTO organizations (id, name, slug) VALUES (?, ?, ?)",
        ("org_1", "Example Energy", "example-energy"),
    )

    try:
        conn.execute(
            "INSERT INTO memberships (user_id, organization_id, role) VALUES (?, ?, ?)",
            ("user_1", "org_1", "admin"),
        )
        assert False, "Expected invalid role to fail"
    except sqlite3.IntegrityError:
        pass
