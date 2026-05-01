from __future__ import annotations

import sqlite3

from .domain import Membership, Meter, Organization, Role, Site, User


class SaaSStore:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

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
