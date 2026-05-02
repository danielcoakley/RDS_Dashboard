from __future__ import annotations

import sqlite3
import json

from .domain import (
    AuditEvent,
    InviteStatus,
    Membership,
    Meter,
    OrganizationInvite,
    Organization,
    Report,
    Role,
    Run,
    RunStatus,
    Site,
    Upload,
    User,
)


class SaaSStore:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def get_user(self, user_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            """
            SELECT id, email, display_name, is_active
            FROM users
            WHERE id = ?
            """,
            (user_id,),
        ).fetchone()

    def create_user(self, user: User) -> None:
        self.conn.execute(
            """
            INSERT INTO users (id, email, display_name, is_active)
            VALUES (?, ?, ?, ?)
            """,
            (user.id, user.email, user.display_name, int(user.is_active)),
        )

    def create_organization(self, organization: Organization) -> None:
        self.conn.execute(
            """
            INSERT INTO organizations (id, name, slug)
            VALUES (?, ?, ?)
            """,
            (organization.id, organization.name, organization.slug),
        )

    def add_membership(self, membership: Membership) -> None:
        self.conn.execute(
            """
            INSERT INTO memberships (user_id, organization_id, role, invited_by_user_id)
            VALUES (?, ?, ?, ?)
            """,
            (
                membership.user_id,
                membership.organization_id,
                membership.role.value,
                membership.invited_by_user_id,
            ),
        )

    def create_organization_invite(self, invite: OrganizationInvite) -> None:
        self.conn.execute(
            """
            INSERT INTO organization_invites (
                id,
                organization_id,
                email,
                role,
                invited_by_user_id,
                status,
                accepted_by_user_id,
                accepted_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                invite.id,
                invite.organization_id,
                invite.email,
                invite.role.value,
                invite.invited_by_user_id,
                invite.status.value,
                invite.accepted_by_user_id,
                invite.accepted_at.isoformat() if invite.accepted_at else None,
            ),
        )

    def create_site(self, site: Site) -> None:
        self.conn.execute(
            """
            INSERT INTO sites (id, organization_id, name, timezone)
            VALUES (?, ?, ?, ?)
            """,
            (site.id, site.organization_id, site.name, site.timezone),
        )

    def create_meter(self, meter: Meter) -> None:
        self.conn.execute(
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
            (
                meter.id,
                meter.organization_id,
                meter.site_id,
                meter.display_name,
                meter.commodity,
                meter.unit,
                meter.source_column,
                int(meter.is_seu),
            ),
        )

    def create_upload(self, upload: Upload) -> None:
        self.conn.execute(
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
                upload.id,
                upload.organization_id,
                upload.site_id,
                upload.uploaded_by_user_id,
                upload.category,
                upload.storage_key,
                upload.checksum,
                upload.status.value,
            ),
        )

    def create_run(self, run: Run) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO runs (
                    id,
                    organization_id,
                    site_id,
                    requested_by_user_id,
                    status,
                    error_message,
                    completed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.id,
                    run.organization_id,
                    run.site_id,
                    run.requested_by_user_id,
                    run.status.value,
                    run.error_message,
                    run.completed_at.isoformat() if run.completed_at else None,
                ),
            )
            for upload_id in run.upload_ids:
                self.conn.execute(
                    "INSERT INTO run_uploads (run_id, organization_id, upload_id) VALUES (?, ?, ?)",
                    (run.id, run.organization_id, upload_id),
                )

    def update_run_status(
        self,
        run_id: str,
        status: RunStatus,
        error_message: str | None = None,
        completed_at: str | None = None,
    ) -> None:
        self.conn.execute(
            """
            UPDATE runs
            SET status = ?, error_message = ?, completed_at = ?
            WHERE id = ?
            """,
            (status.value, error_message, completed_at, run_id),
        )

    def create_report(self, report: Report) -> None:
        self.conn.execute(
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
                report.id,
                report.organization_id,
                report.run_id,
                report.report_type,
                report.storage_key,
                int(report.is_published),
            ),
        )

    def create_audit_event(self, event: AuditEvent) -> None:
        self.conn.execute(
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
                event.id,
                event.organization_id,
                event.actor_user_id,
                event.action.value,
                event.resource_type,
                event.resource_id,
                json.dumps(event.metadata, sort_keys=True),
            ),
        )

    def accept_organization_invite(
        self,
        invite_id: str,
        accepted_by_user_id: str,
        accepted_at: str,
    ) -> None:
        self.conn.execute(
            """
            UPDATE organization_invites
            SET status = ?, accepted_by_user_id = ?, accepted_at = ?
            WHERE id = ?
            """,
            (InviteStatus.ACCEPTED.value, accepted_by_user_id, accepted_at, invite_id),
        )

    def revoke_organization_invite(self, invite_id: str) -> None:
        self.conn.execute(
            """
            UPDATE organization_invites
            SET status = ?, accepted_by_user_id = NULL, accepted_at = NULL
            WHERE id = ?
            """,
            (InviteStatus.REVOKED.value, invite_id),
        )

    def list_user_organizations(self, user_id: str) -> list[Organization]:
        rows = self.conn.execute(
            """
            SELECT organizations.id, organizations.name, organizations.slug
            FROM organizations
            JOIN memberships ON memberships.organization_id = organizations.id
            WHERE memberships.user_id = ?
            ORDER BY organizations.name
            """,
            (user_id,),
        ).fetchall()
        return [
            Organization(id=row["id"], name=row["name"], slug=row["slug"])
            for row in rows
        ]

    def list_memberships(
        self,
        user_id: str | None = None,
        organization_id: str | None = None,
    ) -> list[Membership]:
        clauses: list[str] = []
        params: list[str] = []
        if user_id is not None:
            clauses.append("user_id = ?")
            params.append(user_id)
        if organization_id is not None:
            clauses.append("organization_id = ?")
            params.append(organization_id)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.conn.execute(
            f"""
            SELECT user_id, organization_id, role, invited_by_user_id
            FROM memberships
            {where}
            ORDER BY organization_id, user_id
            """,
            params,
        ).fetchall()
        return [
            Membership(
                user_id=row["user_id"],
                organization_id=row["organization_id"],
                role=Role(row["role"]),
                invited_by_user_id=row["invited_by_user_id"],
            )
            for row in rows
        ]

    def list_organization_members(self, organization_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT
                memberships.user_id,
                memberships.organization_id,
                memberships.role,
                memberships.invited_by_user_id,
                users.email,
                users.display_name,
                users.is_active
            FROM memberships
            JOIN users ON users.id = memberships.user_id
            WHERE memberships.organization_id = ?
            ORDER BY memberships.role, users.display_name, memberships.user_id
            """,
            (organization_id,),
        ).fetchall()

    def list_sites(self, organization_id: str) -> list[Site]:
        rows = self.conn.execute(
            """
            SELECT id, organization_id, name, timezone
            FROM sites
            WHERE organization_id = ?
            ORDER BY name
            """,
            (organization_id,),
        ).fetchall()
        return [
            Site(
                id=row["id"],
                organization_id=row["organization_id"],
                name=row["name"],
                timezone=row["timezone"],
            )
            for row in rows
        ]

    def list_organization_invites(
        self,
        organization_id: str,
        status: InviteStatus | None = None,
    ) -> list[sqlite3.Row]:
        clauses = ["organization_id = ?"]
        params: list[str] = [organization_id]
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        return self.conn.execute(
            f"""
            SELECT *
            FROM organization_invites
            WHERE {' AND '.join(clauses)}
            ORDER BY created_at DESC, id
            """,
            params,
        ).fetchall()

    def get_organization_invite(self, invite_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            """
            SELECT *
            FROM organization_invites
            WHERE id = ?
            """,
            (invite_id,),
        ).fetchone()

    def list_meters(self, organization_id: str, site_id: str | None = None) -> list[Meter]:
        clauses = ["organization_id = ?"]
        params = [organization_id]
        if site_id is not None:
            clauses.append("site_id = ?")
            params.append(site_id)

        rows = self.conn.execute(
            f"""
            SELECT id, organization_id, site_id, display_name, commodity, unit, source_column, is_seu
            FROM meters
            WHERE {' AND '.join(clauses)}
            ORDER BY display_name
            """,
            params,
        ).fetchall()
        return [
            Meter(
                id=row["id"],
                organization_id=row["organization_id"],
                site_id=row["site_id"],
                display_name=row["display_name"],
                commodity=row["commodity"],
                unit=row["unit"],
                source_column=row["source_column"],
                is_seu=bool(row["is_seu"]),
            )
            for row in rows
        ]

    def list_uploads(self, organization_id: str, site_id: str | None = None) -> list[sqlite3.Row]:
        clauses = ["organization_id = ?"]
        params = [organization_id]
        if site_id is not None:
            clauses.append("site_id = ?")
            params.append(site_id)
        return self.conn.execute(
            f"""
            SELECT *
            FROM uploads
            WHERE {' AND '.join(clauses)}
            ORDER BY created_at DESC, id
            """,
            params,
        ).fetchall()

    def list_runs(self, organization_id: str, site_id: str | None = None) -> list[sqlite3.Row]:
        clauses = ["organization_id = ?"]
        params = [organization_id]
        if site_id is not None:
            clauses.append("site_id = ?")
            params.append(site_id)
        return self.conn.execute(
            f"""
            SELECT *
            FROM runs
            WHERE {' AND '.join(clauses)}
            ORDER BY created_at DESC, id
            """,
            params,
        ).fetchall()

    def list_reports(self, organization_id: str, run_id: str | None = None) -> list[sqlite3.Row]:
        clauses = ["organization_id = ?"]
        params = [organization_id]
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        return self.conn.execute(
            f"""
            SELECT *
            FROM reports
            WHERE {' AND '.join(clauses)}
            ORDER BY created_at DESC, id
            """,
            params,
        ).fetchall()

    def list_audit_events(self, organization_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT *
            FROM audit_events
            WHERE organization_id = ?
            ORDER BY rowid
            """,
            (organization_id,),
        ).fetchall()
