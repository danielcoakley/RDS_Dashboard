import sqlite3

from backend.database import apply_migrations, initialize_database


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
    ]


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
