from __future__ import annotations

from datetime import datetime, timezone
import sqlite3
import json

from .domain import (
    AnalysisConfig,
    AnalysisJob,
    AuditEvent,
    CspViolationReport,
    CspViolationReportDirective,
    CspViolationReportDisposition,
    DataConnector,
    DataConnectorSync,
    InviteStatus,
    MarketingInquiry,
    MarketingInquiryStatus,
    Membership,
    Meter,
    Notification,
    NotificationProviderEvent,
    NotificationProviderEventStatus,
    NotificationStatus,
    OrganizationEvidencePackageTemplate,
    OrganizationInvite,
    Organization,
    OrganizationReportApprovalPolicy,
    OrganizationReportBranding,
    OrganizationSubscription,
    PilotReadinessStatus,
    PrivacyRequestStatus,
    PrivacyRequestType,
    Report,
    ReportApprovalAssignment,
    ReportApprovalAssignmentStatus,
    ReportComment,
    ReportEvidencePackage,
    ReportReview,
    ReportSchedule,
    ReportScheduleDelivery,
    ReportScheduleDeliveryStatus,
    Role,
    Run,
    RunStatus,
    Site,
    SubscriptionPlan,
    Upload,
    UploadStatus,
    User,
)


class SaaSStore:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def _count_all(self, table_name: str) -> int:
        allowed_tables = {"organizations", "users", "sites", "meters"}
        if table_name not in allowed_tables:
            raise ValueError(f"Unsupported count table: {table_name}")
        row = self.conn.execute(f"SELECT COUNT(*) AS total FROM {table_name}").fetchone()
        return int(row["total"])

    def _count_by_status(self, table_name: str, statuses: tuple[str, ...]) -> dict[str, int]:
        allowed_tables = {
            "uploads",
            "runs",
            "analysis_jobs",
            "notifications",
            "marketing_inquiries",
        }
        if table_name not in allowed_tables:
            raise ValueError(f"Unsupported status count table: {table_name}")

        counts = {status: 0 for status in statuses}
        rows = self.conn.execute(
            f"""
            SELECT status, COUNT(*) AS total
            FROM {table_name}
            GROUP BY status
            """
        ).fetchall()
        for row in rows:
            status = row["status"]
            if status in counts:
                counts[status] = int(row["total"])
        return counts

    def count_organizations(self) -> int:
        return self._count_all("organizations")

    def count_users(self, active: bool | None = None) -> int:
        if active is None:
            return self._count_all("users")
        row = self.conn.execute(
            """
            SELECT COUNT(*) AS total
            FROM users
            WHERE is_active = ?
            """,
            (int(active),),
        ).fetchone()
        return int(row["total"])

    def count_sites(self) -> int:
        return self._count_all("sites")

    def count_meters(self) -> int:
        return self._count_all("meters")

    def count_uploads_by_status(self) -> dict[str, int]:
        return self._count_by_status("uploads", tuple(status.value for status in UploadStatus))

    def count_runs_by_status(self) -> dict[str, int]:
        return self._count_by_status("runs", tuple(status.value for status in RunStatus))

    def count_analysis_jobs_by_status(self) -> dict[str, int]:
        return self._count_by_status("analysis_jobs", tuple(status.value for status in RunStatus))

    def count_notifications_by_status(self) -> dict[str, int]:
        return self._count_by_status(
            "notifications",
            tuple(status.value for status in NotificationStatus),
        )

    def count_notification_provider_events_by_status(self) -> dict[str, int]:
        counts = {status.value: 0 for status in NotificationProviderEventStatus}
        rows = self.conn.execute(
            """
            SELECT status, COUNT(*) AS total
            FROM notification_provider_events
            GROUP BY status
            """
        ).fetchall()
        for row in rows:
            status = row["status"]
            if status in counts:
                counts[status] = int(row["total"])
        return counts

    def count_csp_violation_reports_by_disposition(self) -> dict[str, int]:
        counts = {disposition.value: 0 for disposition in CspViolationReportDisposition}
        rows = self.conn.execute(
            """
            SELECT disposition, COUNT(*) AS total
            FROM csp_violation_reports
            GROUP BY disposition
            """
        ).fetchall()
        for row in rows:
            disposition = row["disposition"]
            if disposition in counts:
                counts[disposition] = int(row["total"])
        return counts

    def count_csp_violation_reports(self) -> int:
        row = self.conn.execute(
            """
            SELECT COUNT(*) AS total
            FROM csp_violation_reports
            """
        ).fetchone()
        return int(row["total"] or 0)

    def count_csp_violation_reports_by_effective_directive(self) -> dict[str, int]:
        counts = {directive.value: 0 for directive in CspViolationReportDirective}
        rows = self.conn.execute(
            """
            SELECT effective_directive, COUNT(*) AS total
            FROM csp_violation_reports
            GROUP BY effective_directive
            """
        ).fetchall()
        for row in rows:
            directive = row["effective_directive"]
            if directive in counts:
                counts[directive] = int(row["total"])
        return counts

    def get_latest_csp_violation_report_created_at(self) -> str | None:
        row = self.conn.execute(
            """
            SELECT created_at
            FROM csp_violation_reports
            ORDER BY created_at DESC, id
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None
        return row["created_at"]

    def delete_csp_violation_reports_before(self, cutoff_created_before: str) -> int:
        cursor = self.conn.execute(
            """
            DELETE FROM csp_violation_reports
            WHERE created_at < ?
            """,
            (cutoff_created_before,),
        )
        if cursor.rowcount is None or cursor.rowcount < 0:
            return 0
        return int(cursor.rowcount)

    def list_csp_violation_reports(
        self,
        *,
        disposition: str | None = None,
        effective_directive: str | None = None,
        limit: int = 100,
    ) -> list[sqlite3.Row]:
        clauses: list[str] = []
        params: list[str | int] = []
        if disposition is not None:
            clauses.append("disposition = ?")
            params.append(disposition)
        if effective_directive is not None:
            clauses.append("effective_directive = ?")
            params.append(effective_directive)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        return self.conn.execute(
            f"""
            SELECT
                id,
                disposition,
                effective_directive,
                violated_directive,
                blocked_uri_category,
                document_origin,
                status_code,
                created_at
            FROM csp_violation_reports
            {where}
            ORDER BY created_at DESC, id
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()

    def count_notification_outbox_by_phase(self, due_at: str) -> dict[str, int]:
        due_row = self.conn.execute(
            """
            SELECT COUNT(*) AS total
            FROM notifications
            WHERE status = ?
              AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
            """,
            (NotificationStatus.PENDING.value, due_at),
        ).fetchone()
        scheduled_retry_row = self.conn.execute(
            """
            SELECT COUNT(*) AS total
            FROM notifications
            WHERE status = ?
              AND next_attempt_at > ?
            """,
            (NotificationStatus.PENDING.value, due_at),
        ).fetchone()
        failed_row = self.conn.execute(
            """
            SELECT COUNT(*) AS total
            FROM notifications
            WHERE status = ?
            """,
            (NotificationStatus.FAILED.value,),
        ).fetchone()

        return {
            "pending_due": int(due_row["total"] or 0),
            "pending_retry_scheduled": int(scheduled_retry_row["total"] or 0),
            "failed": int(failed_row["total"] or 0),
        }

    def get_oldest_due_notification_created_at(self, due_at: str) -> str | None:
        row = self.conn.execute(
            """
            SELECT MIN(created_at) AS created_at
            FROM notifications
            WHERE status = ?
              AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
            """,
            (NotificationStatus.PENDING.value, due_at),
        ).fetchone()
        if row is None or row["created_at"] is None:
            return None
        return str(row["created_at"])

    def count_report_schedules_by_phase(self, due_at: str) -> dict[str, int]:
        active_due_row = self.conn.execute(
            """
            SELECT COUNT(*) AS total
            FROM organization_report_schedules
            WHERE is_active = 1
              AND next_run_at <= ?
            """,
            (due_at,),
        ).fetchone()
        active_future_row = self.conn.execute(
            """
            SELECT COUNT(*) AS total
            FROM organization_report_schedules
            WHERE is_active = 1
              AND next_run_at > ?
            """,
            (due_at,),
        ).fetchone()
        paused_row = self.conn.execute(
            """
            SELECT COUNT(*) AS total
            FROM organization_report_schedules
            WHERE is_active = 0
            """,
        ).fetchone()
        failed_last_delivery_row = self.conn.execute(
            """
            SELECT COUNT(*) AS total
            FROM organization_report_schedules
            WHERE last_delivery_status = ?
            """,
            (ReportScheduleDeliveryStatus.FAILED.value,),
        ).fetchone()
        return {
            "active_due": int(active_due_row["total"] or 0),
            "active_future": int(active_future_row["total"] or 0),
            "paused": int(paused_row["total"] or 0),
            "failed_last_delivery": int(failed_last_delivery_row["total"] or 0),
        }

    def get_oldest_due_report_schedule_next_run_at(self, due_at: str) -> str | None:
        row = self.conn.execute(
            """
            SELECT MIN(next_run_at) AS next_run_at
            FROM organization_report_schedules
            WHERE is_active = 1
              AND next_run_at <= ?
            """,
            (due_at,),
        ).fetchone()
        if row is None or row["next_run_at"] is None:
            return None
        return str(row["next_run_at"])

    def count_report_schedule_deliveries_by_status(self) -> dict[str, int]:
        counts = {status.value: 0 for status in ReportScheduleDeliveryStatus}
        rows = self.conn.execute(
            """
            SELECT status, COUNT(*) AS total
            FROM organization_report_schedule_deliveries
            GROUP BY status
            """
        ).fetchall()
        for row in rows:
            status = row["status"]
            if status in counts:
                counts[status] = int(row["total"])
        return counts

    def count_marketing_inquiries_by_status(self) -> dict[str, int]:
        return self._count_by_status(
            "marketing_inquiries",
            tuple(status.value for status in MarketingInquiryStatus),
        )

    def count_marketing_inquiries_by_assignee_status(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT
                assigned_to,
                status,
                COUNT(*) AS total
            FROM marketing_inquiries
            GROUP BY assigned_to, status
            ORDER BY assigned_to, status
            """
        ).fetchall()

    def count_marketing_inquiry_submissions_by_outcome(self) -> dict[str, int]:
        counts = {"accepted": 0, "rate_limited": 0}
        rows = self.conn.execute(
            """
            SELECT outcome, COUNT(*) AS total
            FROM marketing_inquiry_submission_events
            GROUP BY outcome
            """
        ).fetchall()
        for row in rows:
            outcome = row["outcome"]
            if outcome in counts:
                counts[outcome] = int(row["total"])
        return counts

    def count_marketing_inquiry_submissions_by_rejection_reason(self) -> dict[str, int]:
        row = self.conn.execute(
            """
            SELECT COUNT(*) AS total
            FROM marketing_inquiry_submission_events
            WHERE outcome = 'rate_limited'
            """
        ).fetchone()
        if row is None:
            return {"source_rate_limit": 0}
        return {"source_rate_limit": int(row["total"] or 0)}

    def count_worker_heartbeats_by_scope(self) -> dict[str, int]:
        counts = {"all": 0, "organization": 0}
        rows = self.conn.execute(
            """
            SELECT scope, COUNT(*) AS total
            FROM worker_heartbeats
            GROUP BY scope
            """
        ).fetchall()
        for row in rows:
            scope = row["scope"]
            if scope in counts:
                counts[scope] = int(row["total"])
        return counts

    def list_worker_heartbeats(
        self,
        *,
        scope: str | None = None,
        limit: int = 100,
    ) -> list[sqlite3.Row]:
        if limit < 1:
            raise ValueError("worker heartbeat limit must be greater than zero")
        if scope is not None and scope not in {"all", "organization"}:
            raise ValueError(f"Unsupported worker heartbeat scope: {scope}")
        params: list[str | int] = []
        where_clause = ""
        if scope is not None:
            where_clause = "WHERE scope = ?"
            params.append(scope)
        params.append(limit)
        return self.conn.execute(
            f"""
            SELECT
                worker_id,
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
                last_seen_at,
                created_at,
                updated_at
            FROM worker_heartbeats
            {where_clause}
            ORDER BY last_seen_at DESC, updated_at DESC, worker_id ASC
            LIMIT ?
            """,
            params,
        ).fetchall()

    def upsert_worker_heartbeat(
        self,
        *,
        worker_id: str,
        organization_id: str | None,
        poll_seconds: float,
        batch_size: int,
        last_attempted: int,
        last_succeeded: int,
        last_failed: int,
        last_skipped: int,
        last_scheduled_attempted: int,
        last_scheduled_sent: int,
        last_scheduled_failed: int,
        last_scheduled_skipped: int,
        last_connector_attempted: int = 0,
        last_connector_succeeded: int = 0,
        last_connector_failed: int = 0,
        last_connector_skipped: int = 0,
        last_seen_at: str | None = None,
    ) -> None:
        observed_at = last_seen_at or datetime.now(timezone.utc).isoformat()
        scope = "organization" if organization_id is not None else "all"
        self.conn.execute(
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
            ON CONFLICT(worker_id) DO UPDATE SET
                organization_id = excluded.organization_id,
                scope = excluded.scope,
                poll_seconds = excluded.poll_seconds,
                batch_size = excluded.batch_size,
                last_attempted = excluded.last_attempted,
                last_succeeded = excluded.last_succeeded,
                last_failed = excluded.last_failed,
                last_skipped = excluded.last_skipped,
                last_scheduled_attempted = excluded.last_scheduled_attempted,
                last_scheduled_sent = excluded.last_scheduled_sent,
                last_scheduled_failed = excluded.last_scheduled_failed,
                last_scheduled_skipped = excluded.last_scheduled_skipped,
                last_connector_attempted = excluded.last_connector_attempted,
                last_connector_succeeded = excluded.last_connector_succeeded,
                last_connector_failed = excluded.last_connector_failed,
                last_connector_skipped = excluded.last_connector_skipped,
                last_seen_at = excluded.last_seen_at,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
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
                observed_at,
            ),
        )

    def get_latest_worker_heartbeat(
        self,
        organization_id: str | None = None,
    ) -> sqlite3.Row | None:
        if organization_id is None:
            return self.conn.execute(
                """
                SELECT *
                FROM worker_heartbeats
                WHERE scope = 'all'
                  AND organization_id IS NULL
                ORDER BY last_seen_at DESC, updated_at DESC
                LIMIT 1
                """
            ).fetchone()
        return self.conn.execute(
            """
            SELECT *
            FROM worker_heartbeats
            WHERE scope = 'organization'
              AND organization_id = ?
            ORDER BY last_seen_at DESC, updated_at DESC
            LIMIT 1
            """,
            (organization_id,),
        ).fetchone()

    def create_platform_operation_event(
        self,
        *,
        event_id: str,
        event_type: str,
        event_status: str,
        request_id: str | None,
        detail: str,
        created_at: str,
    ) -> None:
        self.conn.execute(
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
                event_id,
                event_type,
                event_status,
                request_id,
                detail,
                created_at,
            ),
        )

    def count_platform_operation_events_by_status(self, event_type: str) -> dict[str, int]:
        counts = {"succeeded": 0, "failed": 0}
        rows = self.conn.execute(
            """
            SELECT status, COUNT(*) AS total
            FROM platform_operation_events
            WHERE event_type = ?
            GROUP BY status
            """,
            (event_type,),
        ).fetchall()
        for row in rows:
            status = row["status"]
            if status in counts:
                counts[status] = int(row["total"])
        return counts

    def get_latest_platform_operation_event_created_at(
        self,
        *,
        event_type: str,
        event_status: str,
    ) -> str | None:
        row = self.conn.execute(
            """
            SELECT created_at
            FROM platform_operation_events
            WHERE event_type = ?
              AND status = ?
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            (event_type, event_status),
        ).fetchone()
        if row is None or row["created_at"] is None:
            return None
        return str(row["created_at"])

    def get_platform_operation_event(self, event_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            """
            SELECT *
            FROM platform_operation_events
            WHERE id = ?
            """,
            (event_id,),
        ).fetchone()

    def list_platform_operation_events(
        self,
        *,
        event_type: str | None = None,
        event_status: str | None = None,
        limit: int = 100,
    ) -> list[sqlite3.Row]:
        if limit < 1:
            raise ValueError("platform operation event limit must be greater than zero")
        clauses: list[str] = []
        params: list[str | int] = []
        if event_type is not None:
            clauses.append("event_type = ?")
            params.append(event_type)
        if event_status is not None:
            clauses.append("status = ?")
            params.append(event_status)

        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        return self.conn.execute(
            f"""
            SELECT *
            FROM platform_operation_events
            {where_clause}
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            params,
        ).fetchall()

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

    def upsert_user(self, user: User) -> None:
        self.conn.execute(
            """
            INSERT INTO users (id, email, display_name, is_active)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                email = excluded.email,
                display_name = excluded.display_name,
                is_active = excluded.is_active
            """,
            (user.id, user.email, user.display_name, int(user.is_active)),
        )

    def set_user_active(self, user_id: str, is_active: bool) -> None:
        self.conn.execute(
            """
            UPDATE users
            SET is_active = ?
            WHERE id = ?
            """,
            (int(is_active), user_id),
        )

    def create_organization(self, organization: Organization) -> None:
        self.conn.execute(
            """
            INSERT INTO organizations (id, name, slug)
            VALUES (?, ?, ?)
            """,
            (organization.id, organization.name, organization.slug),
        )

    def create_marketing_inquiry(self, inquiry: MarketingInquiry) -> None:
        self.conn.execute(
            """
            INSERT INTO marketing_inquiries (
                id,
                email,
                name,
                organization_name,
                role,
                intended_use,
                message,
                source_page,
                status,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                inquiry.id,
                inquiry.email,
                inquiry.name,
                inquiry.organization_name,
                inquiry.role,
                inquiry.intended_use,
                inquiry.message,
                inquiry.source_page,
                inquiry.status.value,
                inquiry.created_at.isoformat(),
            ),
        )

    def get_marketing_inquiry(self, inquiry_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            """
            SELECT
                id,
                email,
                name,
                organization_name,
                role,
                intended_use,
                message,
                source_page,
                status,
                created_at,
                assigned_to,
                follow_up_notes,
                updated_at
            FROM marketing_inquiries
            WHERE id = ?
            """,
            (inquiry_id,),
        ).fetchone()

    def count_recent_marketing_inquiries_for_email(self, email: str, since_iso: str) -> int:
        row = self.conn.execute(
            """
            SELECT COUNT(*) AS total
            FROM marketing_inquiries
            WHERE email = ?
              AND created_at >= ?
            """,
            (email, since_iso),
        ).fetchone()
        if row is None:
            return 0
        return int(row["total"])

    def count_recent_marketing_inquiry_submissions_for_source(
        self,
        source_key_hash: str,
        since_iso: str,
    ) -> int:
        row = self.conn.execute(
            """
            SELECT COUNT(*) AS total
            FROM marketing_inquiry_submission_events
            WHERE source_key_hash = ?
              AND created_at >= ?
            """,
            (source_key_hash, since_iso),
        ).fetchone()
        if row is None:
            return 0
        return int(row["total"])

    def create_marketing_inquiry_submission_event(
        self,
        *,
        event_id: str,
        source_key_hash: str,
        inquiry_id: str | None,
        outcome: str,
        created_at_iso: str,
    ) -> None:
        if outcome not in {"accepted", "rate_limited"}:
            raise ValueError("Marketing inquiry submission outcome is unsupported")
        self.conn.execute(
            """
            INSERT INTO marketing_inquiry_submission_events (
                id,
                source_key_hash,
                inquiry_id,
                outcome,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (event_id, source_key_hash, inquiry_id, outcome, created_at_iso),
        )

    def create_marketing_inquiry_followup_event(
        self,
        *,
        event_id: str,
        inquiry_id: str,
        actor_label: str,
        action: str,
        status: MarketingInquiryStatus | None,
        assigned_to: str | None,
        changed_fields: list[str],
        notes_changed: bool,
        created_at_iso: str,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO marketing_inquiry_followup_events (
                id,
                inquiry_id,
                actor_label,
                action,
                status,
                assigned_to,
                changed_fields_json,
                notes_changed,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                inquiry_id,
                actor_label,
                action,
                status.value if status is not None else None,
                assigned_to,
                json.dumps(changed_fields),
                int(notes_changed),
                created_at_iso,
            ),
        )

    def list_marketing_inquiry_followup_events(self, inquiry_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT
                id,
                inquiry_id,
                actor_label,
                action,
                status,
                assigned_to,
                changed_fields_json,
                notes_changed,
                created_at
            FROM marketing_inquiry_followup_events
            WHERE inquiry_id = ?
            ORDER BY created_at DESC, id DESC
            """,
            (inquiry_id,),
        ).fetchall()

    def list_marketing_inquiries(
        self,
        status: MarketingInquiryStatus | None = None,
        limit: int | None = None,
        assigned_to: str | None = None,
        unassigned: bool = False,
    ) -> list[sqlite3.Row]:
        clauses: list[str] = []
        params: list[str | int] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        if assigned_to is not None:
            clauses.append("assigned_to = ?")
            params.append(assigned_to)
        if unassigned:
            clauses.append("assigned_to IS NULL")
        where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        limit_clause = ""
        if limit is not None:
            if limit <= 0:
                raise ValueError("Marketing inquiry list limit must be positive")
            limit_clause = "LIMIT ?"
            params.append(limit)
        return self.conn.execute(
            f"""
            SELECT
                id,
                email,
                name,
                organization_name,
                role,
                intended_use,
                message,
                source_page,
                status,
                created_at,
                assigned_to,
                follow_up_notes,
                updated_at
            FROM marketing_inquiries
            {where_clause}
            ORDER BY created_at DESC, id DESC
            {limit_clause}
            """,
            params,
        ).fetchall()

    def update_marketing_inquiry_status(
        self,
        inquiry_id: str,
        status: MarketingInquiryStatus,
    ) -> sqlite3.Row | None:
        return self.update_marketing_inquiry_followup(
            inquiry_id,
            status=status,
            updated_at_iso=datetime.now(timezone.utc).isoformat(),
        )

    def update_marketing_inquiry_followup(
        self,
        inquiry_id: str,
        *,
        status: MarketingInquiryStatus | None = None,
        assigned_to_set: bool = False,
        assigned_to: str | None = None,
        follow_up_notes_set: bool = False,
        follow_up_notes: str | None = None,
        updated_at_iso: str,
    ) -> sqlite3.Row | None:
        assignments: list[str] = []
        params: list[str | None] = []
        if status is not None:
            assignments.append("status = ?")
            params.append(status.value)
        if assigned_to_set:
            assignments.append("assigned_to = ?")
            params.append(assigned_to)
        if follow_up_notes_set:
            assignments.append("follow_up_notes = ?")
            params.append(follow_up_notes)
        if not assignments:
            raise ValueError("Marketing inquiry follow-up update must include a change")
        assignments.append("updated_at = ?")
        params.append(updated_at_iso)
        params.append(inquiry_id)
        self.conn.execute(
            f"""
            UPDATE marketing_inquiries
            SET {', '.join(assignments)}
            WHERE id = ?
            """,
            tuple(params),
        )
        return self.get_marketing_inquiry(inquiry_id)

    def get_organization(self, organization_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            """
            SELECT id, name, slug
            FROM organizations
            WHERE id = ?
            """,
            (organization_id,),
        ).fetchone()

    def get_organization_by_slug(self, slug: str) -> sqlite3.Row | None:
        return self.conn.execute(
            """
            SELECT id, name, slug
            FROM organizations
            WHERE slug = ?
            """,
            (slug,),
        ).fetchone()

    def upsert_organization(self, organization: Organization) -> None:
        self.conn.execute(
            """
            INSERT INTO organizations (id, name, slug)
            VALUES (?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                slug = excluded.slug
            """,
            (organization.id, organization.name, organization.slug),
        )

    def update_organization(self, organization: Organization) -> None:
        self.conn.execute(
            """
            UPDATE organizations
            SET name = ?, slug = ?
            WHERE id = ?
            """,
            (organization.name, organization.slug, organization.id),
        )

    def _subscription_plan_from_row(self, row: sqlite3.Row) -> SubscriptionPlan:
        return SubscriptionPlan(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            monthly_price_cents=row["monthly_price_cents"],
            included_users=row["included_users"],
            included_sites=row["included_sites"],
            monthly_run_limit=row["monthly_run_limit"],
            report_retention_days=row["report_retention_days"],
            is_active=bool(row["is_active"]),
        )

    def list_subscription_plans(self, active_only: bool = True) -> list[SubscriptionPlan]:
        where = "WHERE is_active = 1" if active_only else ""
        rows = self.conn.execute(
            f"""
            SELECT
                id,
                name,
                description,
                monthly_price_cents,
                included_users,
                included_sites,
                monthly_run_limit,
                report_retention_days,
                is_active
            FROM subscription_plans
            {where}
            ORDER BY
                CASE id
                    WHEN 'trial' THEN 1
                    WHEN 'team' THEN 2
                    WHEN 'enterprise' THEN 3
                    ELSE 99
                END,
                name
            """
        ).fetchall()
        return [self._subscription_plan_from_row(row) for row in rows]

    def get_subscription_plan(self, plan_id: str) -> SubscriptionPlan | None:
        row = self.conn.execute(
            """
            SELECT
                id,
                name,
                description,
                monthly_price_cents,
                included_users,
                included_sites,
                monthly_run_limit,
                report_retention_days,
                is_active
            FROM subscription_plans
            WHERE id = ?
            """,
            (plan_id,),
        ).fetchone()
        if row is None:
            return None
        return self._subscription_plan_from_row(row)

    def get_organization_subscription(self, organization_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            """
            SELECT
                organization_subscriptions.organization_id,
                organization_subscriptions.plan_id,
                organization_subscriptions.status,
                organization_subscriptions.current_period_start,
                organization_subscriptions.current_period_end,
                organization_subscriptions.trial_ends_at,
                organization_subscriptions.external_customer_id,
                organization_subscriptions.external_subscription_id,
                organization_subscriptions.created_at,
                organization_subscriptions.updated_at,
                subscription_plans.name AS plan_name,
                subscription_plans.description AS plan_description,
                subscription_plans.monthly_price_cents,
                subscription_plans.included_users,
                subscription_plans.included_sites,
                subscription_plans.monthly_run_limit,
                subscription_plans.report_retention_days,
                subscription_plans.is_active AS plan_is_active
            FROM organization_subscriptions
            JOIN subscription_plans ON subscription_plans.id = organization_subscriptions.plan_id
            WHERE organization_subscriptions.organization_id = ?
            """,
            (organization_id,),
        ).fetchone()

    def get_organization_subscription_by_external_subscription_id(
        self,
        external_subscription_id: str,
    ) -> sqlite3.Row | None:
        return self.conn.execute(
            """
            SELECT
                organization_subscriptions.organization_id,
                organization_subscriptions.plan_id,
                organization_subscriptions.status,
                organization_subscriptions.current_period_start,
                organization_subscriptions.current_period_end,
                organization_subscriptions.trial_ends_at,
                organization_subscriptions.external_customer_id,
                organization_subscriptions.external_subscription_id,
                organization_subscriptions.created_at,
                organization_subscriptions.updated_at,
                subscription_plans.name AS plan_name,
                subscription_plans.description AS plan_description,
                subscription_plans.monthly_price_cents,
                subscription_plans.included_users,
                subscription_plans.included_sites,
                subscription_plans.monthly_run_limit,
                subscription_plans.report_retention_days,
                subscription_plans.is_active AS plan_is_active
            FROM organization_subscriptions
            JOIN subscription_plans ON subscription_plans.id = organization_subscriptions.plan_id
            WHERE organization_subscriptions.external_subscription_id = ?
            """,
            (external_subscription_id,),
        ).fetchone()

    def get_organization_subscription_by_external_customer_id(
        self,
        external_customer_id: str,
    ) -> sqlite3.Row | None:
        return self.conn.execute(
            """
            SELECT
                organization_subscriptions.organization_id,
                organization_subscriptions.plan_id,
                organization_subscriptions.status,
                organization_subscriptions.current_period_start,
                organization_subscriptions.current_period_end,
                organization_subscriptions.trial_ends_at,
                organization_subscriptions.external_customer_id,
                organization_subscriptions.external_subscription_id,
                organization_subscriptions.created_at,
                organization_subscriptions.updated_at,
                subscription_plans.name AS plan_name,
                subscription_plans.description AS plan_description,
                subscription_plans.monthly_price_cents,
                subscription_plans.included_users,
                subscription_plans.included_sites,
                subscription_plans.monthly_run_limit,
                subscription_plans.report_retention_days,
                subscription_plans.is_active AS plan_is_active
            FROM organization_subscriptions
            JOIN subscription_plans ON subscription_plans.id = organization_subscriptions.plan_id
            WHERE organization_subscriptions.external_customer_id = ?
            """,
            (external_customer_id,),
        ).fetchone()

    def upsert_organization_subscription(self, subscription: OrganizationSubscription) -> None:
        self.conn.execute(
            """
            INSERT INTO organization_subscriptions (
                organization_id,
                plan_id,
                status,
                current_period_start,
                current_period_end,
                trial_ends_at,
                external_customer_id,
                external_subscription_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(organization_id) DO UPDATE SET
                plan_id = excluded.plan_id,
                status = excluded.status,
                current_period_start = excluded.current_period_start,
                current_period_end = excluded.current_period_end,
                trial_ends_at = excluded.trial_ends_at,
                external_customer_id = excluded.external_customer_id,
                external_subscription_id = excluded.external_subscription_id,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                subscription.organization_id,
                subscription.plan_id,
                subscription.status.value,
                subscription.current_period_start.isoformat()
                if subscription.current_period_start
                else None,
                subscription.current_period_end.isoformat()
                if subscription.current_period_end
                else None,
                subscription.trial_ends_at.isoformat() if subscription.trial_ends_at else None,
                subscription.external_customer_id,
                subscription.external_subscription_id,
            ),
        )

    def has_billing_event(self, event_id: str) -> bool:
        row = self.conn.execute(
            """
            SELECT id
            FROM billing_events
            WHERE id = ?
            """,
            (event_id,),
        ).fetchone()
        return row is not None

    def create_billing_event(
        self,
        *,
        event_id: str,
        provider: str,
        event_type: str,
        organization_id: str | None,
        external_customer_id: str | None,
        external_subscription_id: str | None,
        payload: dict[str, object],
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO billing_events (
                id,
                provider,
                event_type,
                organization_id,
                external_customer_id,
                external_subscription_id,
                payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                provider,
                event_type,
                organization_id,
                external_customer_id,
                external_subscription_id,
                json.dumps(payload, sort_keys=True),
            ),
        )

    def get_analysis_config(self, organization_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            """
            SELECT organization_id, config_json, updated_by_user_id, created_at, updated_at
            FROM analysis_configs
            WHERE organization_id = ?
            """,
            (organization_id,),
        ).fetchone()

    def upsert_analysis_config(self, config: AnalysisConfig) -> None:
        self.conn.execute(
            """
            INSERT INTO analysis_configs (organization_id, config_json, updated_by_user_id)
            VALUES (?, ?, ?)
            ON CONFLICT(organization_id) DO UPDATE SET
                config_json = excluded.config_json,
                updated_by_user_id = excluded.updated_by_user_id,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                config.organization_id,
                json.dumps(config.config, sort_keys=True),
                config.updated_by_user_id,
            ),
        )

    def list_data_connectors(
        self,
        organization_id: str,
        site_id: str | None = None,
    ) -> list[sqlite3.Row]:
        if site_id is None:
            return self.conn.execute(
                """
                SELECT *
                FROM organization_data_connectors
                WHERE organization_id = ?
                ORDER BY updated_at DESC, created_at DESC, id DESC
                """,
                (organization_id,),
            ).fetchall()
        return self.conn.execute(
            """
            SELECT *
            FROM organization_data_connectors
            WHERE organization_id = ? AND site_id = ?
            ORDER BY updated_at DESC, created_at DESC, id DESC
            """,
            (organization_id, site_id),
        ).fetchall()

    def list_due_data_connectors(
        self,
        due_at: str,
        organization_id: str | None = None,
        limit: int | None = None,
    ) -> list[sqlite3.Row]:
        query = """
            SELECT *
            FROM organization_data_connectors
            WHERE status = 'active'
              AND schedule != 'manual'
              AND next_sync_at IS NOT NULL
              AND next_sync_at <= ?
        """
        params: list[object] = [due_at]
        if organization_id is not None:
            query += " AND organization_id = ?"
            params.append(organization_id)
        query += " ORDER BY next_sync_at ASC, id ASC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        return self.conn.execute(query, tuple(params)).fetchall()

    def get_data_connector(self, connector_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            """
            SELECT *
            FROM organization_data_connectors
            WHERE id = ?
            """,
            (connector_id,),
        ).fetchone()

    def create_data_connector(self, connector: DataConnector) -> None:
        self.conn.execute(
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
                last_sync_at,
                next_sync_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                connector.id,
                connector.organization_id,
                connector.site_id,
                connector.name,
                connector.connector_type.value,
                connector.source_category,
                connector.status.value,
                connector.schedule.value,
                connector.external_reference,
                connector.credential_reference,
                connector.created_by_user_id,
                connector.updated_by_user_id,
                connector.last_sync_at.isoformat() if connector.last_sync_at else None,
                connector.next_sync_at.isoformat() if connector.next_sync_at else None,
            ),
        )

    def update_data_connector(self, connector: DataConnector) -> None:
        self.conn.execute(
            """
            UPDATE organization_data_connectors
            SET
                name = ?,
                status = ?,
                schedule = ?,
                external_reference = ?,
                credential_reference = ?,
                updated_by_user_id = ?,
                last_sync_at = ?,
                next_sync_at = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND organization_id = ?
            """,
            (
                connector.name,
                connector.status.value,
                connector.schedule.value,
                connector.external_reference,
                connector.credential_reference,
                connector.updated_by_user_id,
                connector.last_sync_at.isoformat() if connector.last_sync_at else None,
                connector.next_sync_at.isoformat() if connector.next_sync_at else None,
                connector.id,
                connector.organization_id,
            ),
        )

    def create_data_connector_sync(self, sync: DataConnectorSync) -> None:
        self.conn.execute(
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
                sync.id,
                sync.organization_id,
                sync.connector_id,
                sync.site_id,
                sync.source_category,
                sync.status.value,
                sync.upload_id,
                sync.records_processed,
                sync.status_reason,
                sync.triggered_by_user_id,
                sync.started_at.isoformat(),
                sync.completed_at.isoformat() if sync.completed_at else None,
            ),
        )

    def list_data_connector_syncs(
        self,
        organization_id: str,
        connector_id: str,
        limit: int | None = None,
    ) -> list[sqlite3.Row]:
        query = """
            SELECT *
            FROM organization_data_connector_syncs
            WHERE organization_id = ? AND connector_id = ?
            ORDER BY COALESCE(completed_at, started_at) DESC, created_at DESC, id DESC
        """
        params: list[object] = [organization_id, connector_id]
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        return self.conn.execute(query, tuple(params)).fetchall()

    def update_data_connector_sync_state(
        self,
        connector_id: str,
        organization_id: str,
        *,
        last_sync_at: str,
        last_sync_status: str,
        last_sync_error: str | None,
        next_sync_at: str | None,
    ) -> None:
        self.conn.execute(
            """
            UPDATE organization_data_connectors
            SET
                last_sync_at = ?,
                last_sync_status = ?,
                last_sync_error = ?,
                next_sync_at = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND organization_id = ?
            """,
            (
                last_sync_at,
                last_sync_status,
                last_sync_error,
                next_sync_at,
                connector_id,
                organization_id,
            ),
        )

    def get_organization_report_branding(self, organization_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            """
            SELECT
                organization_id,
                evidence_pack_title,
                primary_color,
                logo_url,
                support_email,
                created_at,
                updated_at
            FROM organization_report_branding
            WHERE organization_id = ?
            """,
            (organization_id,),
        ).fetchone()

    def upsert_organization_report_branding(
        self,
        branding: OrganizationReportBranding,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO organization_report_branding (
                organization_id,
                evidence_pack_title,
                primary_color,
                logo_url,
                support_email
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(organization_id) DO UPDATE SET
                evidence_pack_title = excluded.evidence_pack_title,
                primary_color = excluded.primary_color,
                logo_url = excluded.logo_url,
                support_email = excluded.support_email,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                branding.organization_id,
                branding.evidence_pack_title,
                branding.primary_color,
                branding.logo_url,
                branding.support_email,
            ),
        )

    def get_organization_report_approval_policy(
        self,
        organization_id: str,
    ) -> sqlite3.Row | None:
        return self.conn.execute(
            """
            SELECT
                organization_id,
                auto_lock_evidence_on_final_approval,
                minimum_approval_assignments_for_lock,
                updated_by_user_id,
                created_at,
                updated_at
            FROM organization_report_approval_policies
            WHERE organization_id = ?
            """,
            (organization_id,),
        ).fetchone()

    def upsert_organization_report_approval_policy(
        self,
        policy: OrganizationReportApprovalPolicy,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO organization_report_approval_policies (
                organization_id,
                auto_lock_evidence_on_final_approval,
                minimum_approval_assignments_for_lock,
                updated_by_user_id
            )
            VALUES (?, ?, ?, ?)
            ON CONFLICT(organization_id) DO UPDATE SET
                auto_lock_evidence_on_final_approval =
                    excluded.auto_lock_evidence_on_final_approval,
                minimum_approval_assignments_for_lock =
                    excluded.minimum_approval_assignments_for_lock,
                updated_by_user_id = excluded.updated_by_user_id,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                policy.organization_id,
                int(policy.auto_lock_evidence_on_final_approval),
                policy.minimum_approval_assignments_for_lock,
                policy.updated_by_user_id,
            ),
        )

    def get_organization_evidence_package_template(
        self,
        organization_id: str,
    ) -> sqlite3.Row | None:
        return self.conn.execute(
            """
            SELECT
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
                updated_by_user_id,
                created_at,
                updated_at
            FROM organization_evidence_package_templates
            WHERE organization_id = ?
            """,
            (organization_id,),
        ).fetchone()

    def upsert_organization_evidence_package_template(
        self,
        template: OrganizationEvidencePackageTemplate,
    ) -> None:
        self.conn.execute(
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
            ON CONFLICT(organization_id) DO UPDATE SET
                template_name = excluded.template_name,
                include_report_artifact = excluded.include_report_artifact,
                include_uploads = excluded.include_uploads,
                include_approval_assignments = excluded.include_approval_assignments,
                include_reviews = excluded.include_reviews,
                include_comments = excluded.include_comments,
                include_schedules = excluded.include_schedules,
                section_labels_json = excluded.section_labels_json,
                section_order_json = excluded.section_order_json,
                updated_by_user_id = excluded.updated_by_user_id,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                template.organization_id,
                template.template_name,
                int(template.include_report_artifact),
                int(template.include_uploads),
                int(template.include_approval_assignments),
                int(template.include_reviews),
                int(template.include_comments),
                int(template.include_schedules),
                template.section_labels_json,
                template.section_order_json,
                template.updated_by_user_id,
            ),
        )

    def get_organization_privacy_settings(self, organization_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            """
            SELECT
                organization_id,
                data_processing_acknowledged_at,
                data_processing_acknowledged_by_user_id,
                data_processing_contact_email,
                subprocessor_notifications_enabled,
                analytics_data_sharing_enabled,
                created_at,
                updated_at
            FROM organization_privacy_settings
            WHERE organization_id = ?
            """,
            (organization_id,),
        ).fetchone()

    def upsert_organization_privacy_settings(
        self,
        *,
        organization_id: str,
        data_processing_acknowledged_at: str | None,
        data_processing_acknowledged_by_user_id: str | None,
        data_processing_contact_email: str | None,
        subprocessor_notifications_enabled: bool,
        analytics_data_sharing_enabled: bool,
    ) -> None:
        self.conn.execute(
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
            ON CONFLICT(organization_id) DO UPDATE SET
                data_processing_acknowledged_at = excluded.data_processing_acknowledged_at,
                data_processing_acknowledged_by_user_id = excluded.data_processing_acknowledged_by_user_id,
                data_processing_contact_email = excluded.data_processing_contact_email,
                subprocessor_notifications_enabled = excluded.subprocessor_notifications_enabled,
                analytics_data_sharing_enabled = excluded.analytics_data_sharing_enabled,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                organization_id,
                data_processing_acknowledged_at,
                data_processing_acknowledged_by_user_id,
                data_processing_contact_email,
                int(subprocessor_notifications_enabled),
                int(analytics_data_sharing_enabled),
            ),
        )

    def create_organization_privacy_request(
        self,
        *,
        request_id: str,
        organization_id: str,
        requested_by_user_id: str,
        subject_user_id: str | None,
        request_type: PrivacyRequestType,
        status: PrivacyRequestStatus,
        description: str | None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO organization_privacy_requests (
                id,
                organization_id,
                requested_by_user_id,
                subject_user_id,
                request_type,
                status,
                description
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                organization_id,
                requested_by_user_id,
                subject_user_id,
                request_type.value,
                status.value,
                description,
            ),
        )

    def get_organization_privacy_request(self, request_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            """
            SELECT *
            FROM organization_privacy_requests
            WHERE id = ?
            """,
            (request_id,),
        ).fetchone()

    def list_organization_privacy_requests(
        self,
        organization_id: str,
        status: PrivacyRequestStatus | None = None,
    ) -> list[sqlite3.Row]:
        if status is None:
            return self.conn.execute(
                """
                SELECT *
                FROM organization_privacy_requests
                WHERE organization_id = ?
                ORDER BY created_at DESC, id DESC
                """,
                (organization_id,),
            ).fetchall()
        return self.conn.execute(
            """
            SELECT *
            FROM organization_privacy_requests
            WHERE organization_id = ?
              AND status = ?
            ORDER BY created_at DESC, id DESC
            """,
            (organization_id, status.value),
        ).fetchall()

    def update_organization_privacy_request_status(
        self,
        *,
        request_id: str,
        status: PrivacyRequestStatus,
        resolution_notes: str | None,
        completed_at: str | None,
    ) -> None:
        self.conn.execute(
            """
            UPDATE organization_privacy_requests
            SET status = ?,
                resolution_notes = ?,
                completed_at = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                status.value,
                resolution_notes,
                completed_at,
                request_id,
            ),
        )

    def list_organization_pilot_readiness_items(self, organization_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT *
            FROM organization_pilot_readiness_items
            WHERE organization_id = ?
            ORDER BY item_key
            """,
            (organization_id,),
        ).fetchall()

    def get_organization_pilot_readiness_item(
        self,
        organization_id: str,
        item_key: str,
    ) -> sqlite3.Row | None:
        return self.conn.execute(
            """
            SELECT *
            FROM organization_pilot_readiness_items
            WHERE organization_id = ?
              AND item_key = ?
            """,
            (organization_id, item_key),
        ).fetchone()

    def upsert_organization_pilot_readiness_item(
        self,
        *,
        organization_id: str,
        item_key: str,
        status: PilotReadinessStatus,
        evidence_url: str | None,
        notes: str | None,
        updated_by_user_id: str,
    ) -> None:
        self.conn.execute(
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
            ON CONFLICT(organization_id, item_key) DO UPDATE SET
                status = excluded.status,
                evidence_url = excluded.evidence_url,
                notes = excluded.notes,
                updated_by_user_id = excluded.updated_by_user_id,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                organization_id,
                item_key,
                status.value,
                evidence_url,
                notes,
                updated_by_user_id,
            ),
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

    def upsert_membership(self, membership: Membership) -> None:
        self.conn.execute(
            """
            INSERT INTO memberships (user_id, organization_id, role, invited_by_user_id)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id, organization_id) DO UPDATE SET
                role = excluded.role,
                invited_by_user_id = COALESCE(
                    excluded.invited_by_user_id,
                    memberships.invited_by_user_id
                )
            """,
            (
                membership.user_id,
                membership.organization_id,
                membership.role.value,
                membership.invited_by_user_id,
            ),
        )

    def delete_membership(self, user_id: str, organization_id: str) -> None:
        self.conn.execute(
            """
            DELETE FROM memberships
            WHERE user_id = ? AND organization_id = ?
            """,
            (user_id, organization_id),
        )

    def delete_user_memberships(self, user_id: str) -> None:
        self.conn.execute(
            """
            DELETE FROM memberships
            WHERE user_id = ?
            """,
            (user_id,),
        )

    def delete_organization_memberships(self, organization_id: str) -> None:
        self.conn.execute(
            """
            DELETE FROM memberships
            WHERE organization_id = ?
            """,
            (organization_id,),
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

    def _site_from_row(self, row: Any) -> Site:
        return Site(
            id=row["id"],
            organization_id=row["organization_id"],
            name=row["name"],
            timezone=row["timezone"],
            location_label=row["location_label"],
            latitude=row["latitude"],
            longitude=row["longitude"],
        )

    def create_site(self, site: Site) -> None:
        self.conn.execute(
            """
            INSERT INTO sites (id, organization_id, name, timezone, location_label, latitude, longitude)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                site.id,
                site.organization_id,
                site.name,
                site.timezone,
                site.location_label,
                site.latitude,
                site.longitude,
            ),
        )

    def update_site_location(
        self,
        site_id: str,
        location_label: str,
        latitude: float,
        longitude: float,
    ) -> None:
        self.conn.execute(
            """
            UPDATE sites
            SET location_label = ?, latitude = ?, longitude = ?
            WHERE id = ?
            """,
            (location_label, latitude, longitude, site_id),
        )

    def get_site(self, site_id: str) -> Site | None:
        row = self.conn.execute(
            """
            SELECT id, organization_id, name, timezone, location_label, latitude, longitude
            FROM sites
            WHERE id = ?
            """,
            (site_id,),
        ).fetchone()
        if row is None:
            return None
        return self._site_from_row(row)

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

    def get_upload(self, upload_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            """
            SELECT *
            FROM uploads
            WHERE id = ?
            """,
            (upload_id,),
        ).fetchone()

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

    def get_run(self, run_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            """
            SELECT *
            FROM runs
            WHERE id = ?
            """,
            (run_id,),
        ).fetchone()

    def list_run_uploads(self, organization_id: str, run_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT uploads.*
            FROM run_uploads
            JOIN uploads ON uploads.id = run_uploads.upload_id
            WHERE run_uploads.organization_id = ?
              AND run_uploads.run_id = ?
            ORDER BY uploads.created_at ASC, uploads.id
            """,
            (organization_id, run_id),
        ).fetchall()

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

    def create_analysis_job(self, job: AnalysisJob) -> None:
        self.conn.execute(
            """
            INSERT INTO analysis_jobs (
                id,
                organization_id,
                run_id,
                upload_id,
                report_id,
                requested_by_user_id,
                status,
                client_config_json,
                error_message,
                completed_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job.id,
                job.organization_id,
                job.run_id,
                job.upload_id,
                job.report_id,
                job.requested_by_user_id,
                job.status.value,
                json.dumps(job.client_config, sort_keys=True),
                job.error_message,
                job.completed_at.isoformat() if job.completed_at else None,
            ),
        )

    def get_analysis_job(self, job_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            """
            SELECT *
            FROM analysis_jobs
            WHERE id = ?
            """,
            (job_id,),
        ).fetchone()

    def update_analysis_job_status(
        self,
        job_id: str,
        status: RunStatus,
        error_message: str | None = None,
        completed_at: str | None = None,
    ) -> None:
        self.conn.execute(
            """
            UPDATE analysis_jobs
            SET status = ?, error_message = ?, completed_at = ?
            WHERE id = ?
            """,
            (status.value, error_message, completed_at, job_id),
        )

    def claim_analysis_job(self, job_id: str) -> bool:
        cursor = self.conn.execute(
            """
            UPDATE analysis_jobs
            SET status = ?
            WHERE id = ? AND status = ?
            """,
            (RunStatus.RUNNING.value, job_id, RunStatus.QUEUED.value),
        )
        return cursor.rowcount == 1

    def list_analysis_jobs(self, organization_id: str) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT *
            FROM analysis_jobs
            WHERE organization_id = ?
            ORDER BY created_at DESC, id
            """,
            (organization_id,),
        ).fetchall()

    def list_queued_analysis_jobs(
        self,
        organization_id: str | None = None,
        limit: int | None = None,
    ) -> list[sqlite3.Row]:
        clauses = ["status = ?"]
        params: list[str | int] = [RunStatus.QUEUED.value]
        if organization_id is not None:
            clauses.append("organization_id = ?")
            params.append(organization_id)

        limit_clause = ""
        if limit is not None:
            if limit < 1:
                raise ValueError("analysis job limit must be greater than zero")
            limit_clause = "LIMIT ?"
            params.append(limit)

        return self.conn.execute(
            f"""
            SELECT *
            FROM analysis_jobs
            WHERE {' AND '.join(clauses)}
            ORDER BY created_at ASC, id
            {limit_clause}
            """,
            params,
        ).fetchall()

    def create_report(self, report: Report) -> None:
        self.conn.execute(
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
                report.id,
                report.organization_id,
                report.run_id,
                report.report_type,
                report.storage_key,
                int(report.is_published),
                report.expires_at.isoformat() if report.expires_at else None,
                report.retention_applied_at.isoformat() if report.retention_applied_at else None,
            ),
        )

    def mark_report_retention_applied(self, report_id: str, applied_at: str) -> None:
        self.conn.execute(
            """
            UPDATE reports
            SET is_published = 0,
                retention_applied_at = ?
            WHERE id = ?
            """,
            (applied_at, report_id),
        )

    def get_report(self, report_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            """
            SELECT *
            FROM reports
            WHERE id = ?
            """,
            (report_id,),
        ).fetchone()

    def create_report_comment(self, comment: ReportComment) -> None:
        self.conn.execute(
            """
            INSERT INTO organization_report_comments (
                id,
                organization_id,
                report_id,
                author_user_id,
                body,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                comment.id,
                comment.organization_id,
                comment.report_id,
                comment.author_user_id,
                comment.body,
                comment.created_at.isoformat(),
            ),
        )

    def get_report_comment(self, comment_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            """
            SELECT *
            FROM organization_report_comments
            WHERE id = ?
            """,
            (comment_id,),
        ).fetchone()

    def list_report_comments(
        self,
        organization_id: str,
        report_id: str,
    ) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT *
            FROM organization_report_comments
            WHERE organization_id = ?
              AND report_id = ?
            ORDER BY created_at ASC, id
            """,
            (organization_id, report_id),
        ).fetchall()

    def create_report_review(self, review: ReportReview) -> None:
        self.conn.execute(
            """
            INSERT INTO organization_report_reviews (
                id,
                organization_id,
                report_id,
                reviewer_user_id,
                status,
                notes,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                review.id,
                review.organization_id,
                review.report_id,
                review.reviewer_user_id,
                review.status.value,
                review.notes,
                review.created_at.isoformat(),
            ),
        )

    def get_report_review(self, review_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            """
            SELECT *
            FROM organization_report_reviews
            WHERE id = ?
            """,
            (review_id,),
        ).fetchone()

    def list_report_reviews(
        self,
        organization_id: str,
        report_id: str,
    ) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT *
            FROM organization_report_reviews
            WHERE organization_id = ?
              AND report_id = ?
            ORDER BY created_at DESC, id DESC
            """,
            (organization_id, report_id),
        ).fetchall()

    def create_report_approval_assignment(
        self,
        assignment: ReportApprovalAssignment,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO organization_report_approval_assignments (
                id,
                organization_id,
                report_id,
                assigned_to_user_id,
                assigned_by_user_id,
                status,
                due_at,
                notes,
                completed_at,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                assignment.id,
                assignment.organization_id,
                assignment.report_id,
                assignment.assigned_to_user_id,
                assignment.assigned_by_user_id,
                assignment.status.value,
                assignment.due_at.isoformat() if assignment.due_at else None,
                assignment.notes,
                assignment.completed_at.isoformat() if assignment.completed_at else None,
                assignment.created_at.isoformat(),
                assignment.updated_at.isoformat(),
            ),
        )

    def get_report_approval_assignment(
        self,
        assignment_id: str,
    ) -> sqlite3.Row | None:
        return self.conn.execute(
            """
            SELECT *
            FROM organization_report_approval_assignments
            WHERE id = ?
            """,
            (assignment_id,),
        ).fetchone()

    def list_report_approval_assignments(
        self,
        organization_id: str,
        report_id: str,
    ) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT *
            FROM organization_report_approval_assignments
            WHERE organization_id = ?
              AND report_id = ?
            ORDER BY
                CASE status
                    WHEN 'pending' THEN 0
                    ELSE 1
                END,
                COALESCE(due_at, created_at) ASC,
                created_at ASC,
                id
            """,
            (organization_id, report_id),
        ).fetchall()

    def update_report_approval_assignment(
        self,
        assignment_id: str,
        *,
        status: ReportApprovalAssignmentStatus,
        notes: str | None,
        completed_at: str | None,
        updated_at: str,
    ) -> None:
        self.conn.execute(
            """
            UPDATE organization_report_approval_assignments
            SET status = ?,
                notes = ?,
                completed_at = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                status.value,
                notes,
                completed_at,
                updated_at,
                assignment_id,
            ),
        )

    def create_report_evidence_package(self, package: ReportEvidencePackage) -> None:
        self.conn.execute(
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
                upload_count,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                package.id,
                package.organization_id,
                package.report_id,
                package.created_by_user_id,
                package.storage_key,
                package.filename,
                package.content_type,
                package.content_length,
                package.package_type,
                package.format,
                package.comment_count,
                package.review_count,
                package.schedule_count,
                package.delivery_count,
                package.upload_count,
                package.created_at.isoformat(),
            ),
        )

    def get_report_evidence_package(self, package_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            """
            SELECT *
            FROM organization_report_evidence_packages
            WHERE id = ?
            """,
            (package_id,),
        ).fetchone()

    def list_report_evidence_packages(
        self,
        organization_id: str,
        report_id: str,
    ) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT *
            FROM organization_report_evidence_packages
            WHERE organization_id = ?
              AND report_id = ?
            ORDER BY created_at DESC, id DESC
            """,
            (organization_id, report_id),
        ).fetchall()

    def delete_report_evidence_packages(
        self,
        organization_id: str,
        report_id: str,
    ) -> int:
        cursor = self.conn.execute(
            """
            DELETE FROM organization_report_evidence_packages
            WHERE organization_id = ?
              AND report_id = ?
            """,
            (organization_id, report_id),
        )
        return cursor.rowcount

    def create_report_schedule(self, schedule: ReportSchedule) -> None:
        self.conn.execute(
            """
            INSERT INTO organization_report_schedules (
                id,
                organization_id,
                report_id,
                created_by_user_id,
                frequency,
                recipient_email,
                next_run_at,
                is_active,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                schedule.id,
                schedule.organization_id,
                schedule.report_id,
                schedule.created_by_user_id,
                schedule.frequency.value,
                schedule.recipient_email,
                schedule.next_run_at.isoformat(),
                int(schedule.is_active),
                schedule.created_at.isoformat(),
                schedule.updated_at.isoformat(),
            ),
        )

    def get_report_schedule(self, schedule_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            """
            SELECT *
            FROM organization_report_schedules
            WHERE id = ?
            """,
            (schedule_id,),
        ).fetchone()

    def list_report_schedules(
        self,
        organization_id: str,
        report_id: str,
    ) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT *
            FROM organization_report_schedules
            WHERE organization_id = ?
              AND report_id = ?
            ORDER BY is_active DESC, next_run_at ASC, id
            """,
            (organization_id, report_id),
        ).fetchall()

    def list_report_schedules_by_creator(
        self,
        organization_id: str,
        created_by_user_id: str,
    ) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT *
            FROM organization_report_schedules
            WHERE organization_id = ?
              AND created_by_user_id = ?
            ORDER BY is_active DESC, next_run_at ASC, id
            """,
            (organization_id, created_by_user_id),
        ).fetchall()

    def transfer_report_schedule_ownership(
        self,
        organization_id: str,
        *,
        from_user_id: str,
        to_user_id: str,
        from_recipient_email: str,
        to_recipient_email: str,
        updated_at: str,
    ) -> int:
        cursor = self.conn.execute(
            """
            UPDATE organization_report_schedules
            SET created_by_user_id = ?,
                recipient_email = CASE
                    WHEN lower(recipient_email) = lower(?) THEN ?
                    ELSE recipient_email
                END,
                updated_at = ?
            WHERE organization_id = ?
              AND created_by_user_id = ?
            """,
            (
                to_user_id,
                from_recipient_email,
                to_recipient_email,
                updated_at,
                organization_id,
                from_user_id,
            ),
        )
        return cursor.rowcount

    def list_due_report_schedules(
        self,
        due_at: str,
        organization_id: str | None = None,
        limit: int | None = None,
    ) -> list[sqlite3.Row]:
        clauses = ["is_active = 1", "next_run_at <= ?"]
        params: list[str | int] = [due_at]
        if organization_id is not None:
            clauses.append("organization_id = ?")
            params.append(organization_id)

        limit_clause = ""
        if limit is not None:
            if limit < 1:
                raise ValueError("report schedule limit must be greater than zero")
            limit_clause = "LIMIT ?"
            params.append(limit)

        return self.conn.execute(
            f"""
            SELECT *
            FROM organization_report_schedules
            WHERE {' AND '.join(clauses)}
            ORDER BY next_run_at ASC, id
            {limit_clause}
            """,
            params,
        ).fetchall()

    def update_report_schedule(
        self,
        schedule_id: str,
        *,
        frequency: str,
        recipient_email: str,
        next_run_at: str,
        is_active: bool,
        updated_at: str,
    ) -> None:
        self.conn.execute(
            """
            UPDATE organization_report_schedules
            SET frequency = ?,
                recipient_email = ?,
                next_run_at = ?,
                is_active = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                frequency,
                recipient_email,
                next_run_at,
                int(is_active),
                updated_at,
                schedule_id,
            ),
        )

    def create_report_schedule_delivery(self, delivery: ReportScheduleDelivery) -> None:
        self.conn.execute(
            """
            INSERT INTO organization_report_schedule_deliveries (
                id,
                organization_id,
                schedule_id,
                report_id,
                recipient_email,
                status,
                error_message,
                attempted_at,
                sent_at,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                delivery.id,
                delivery.organization_id,
                delivery.schedule_id,
                delivery.report_id,
                delivery.recipient_email,
                delivery.status.value,
                delivery.error_message,
                delivery.attempted_at.isoformat(),
                delivery.sent_at.isoformat() if delivery.sent_at else None,
                delivery.created_at.isoformat(),
            ),
        )

    def update_report_schedule_delivery_state(
        self,
        schedule_id: str,
        *,
        next_run_at: str,
        last_delivery_at: str,
        last_delivery_status: ReportScheduleDeliveryStatus,
        last_delivery_error: str | None,
    ) -> None:
        self.conn.execute(
            """
            UPDATE organization_report_schedules
            SET next_run_at = ?,
                last_delivery_at = ?,
                last_delivery_status = ?,
                last_delivery_error = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (
                next_run_at,
                last_delivery_at,
                last_delivery_status.value,
                last_delivery_error,
                schedule_id,
            ),
        )

    def list_report_schedule_deliveries(
        self,
        organization_id: str,
        schedule_id: str,
    ) -> list[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT *
            FROM organization_report_schedule_deliveries
            WHERE organization_id = ?
              AND schedule_id = ?
            ORDER BY created_at ASC, id
            """,
            (organization_id, schedule_id),
        ).fetchall()

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

    def create_notification(self, notification: Notification) -> None:
        self.conn.execute(
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
                resource_id,
                status,
                error_message,
                sent_at,
                attempt_count,
                last_attempt_at,
                next_attempt_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                notification.id,
                notification.organization_id,
                notification.recipient_user_id,
                notification.channel.value,
                notification.event_type,
                notification.subject,
                notification.body,
                notification.resource_type,
                notification.resource_id,
                notification.status.value,
                notification.error_message,
                notification.sent_at.isoformat() if notification.sent_at else None,
                notification.attempt_count,
                (
                    notification.last_attempt_at.isoformat()
                    if notification.last_attempt_at
                    else None
                ),
                (
                    notification.next_attempt_at.isoformat()
                    if notification.next_attempt_at
                    else None
                ),
            ),
        )

    def get_notification(self, notification_id: str) -> sqlite3.Row | None:
        return self.conn.execute(
            """
            SELECT *
            FROM notifications
            WHERE id = ?
            """,
            (notification_id,),
        ).fetchone()

    def update_notification_status(
        self,
        notification_id: str,
        status: NotificationStatus,
        sent_at: str | None = None,
        error_message: str | None = None,
        attempt_count: int | None = None,
        last_attempt_at: str | None = None,
        next_attempt_at: str | None = None,
    ) -> None:
        assignments = ["status = ?", "sent_at = ?", "error_message = ?"]
        params: list[str | int | None] = [status.value, sent_at, error_message]
        if attempt_count is not None:
            assignments.append("attempt_count = ?")
            params.append(attempt_count)
        if last_attempt_at is not None:
            assignments.append("last_attempt_at = ?")
            params.append(last_attempt_at)
        assignments.append("next_attempt_at = ?")
        params.append(next_attempt_at)
        params.append(notification_id)
        self.conn.execute(
            f"""
            UPDATE notifications
            SET {', '.join(assignments)}
            WHERE id = ?
            """,
            params,
        )

    def get_notification_provider_event(
        self,
        *,
        provider: str,
        provider_event_id: str,
    ) -> sqlite3.Row | None:
        return self.conn.execute(
            """
            SELECT *
            FROM notification_provider_events
            WHERE provider = ?
              AND provider_event_id = ?
            """,
            (provider, provider_event_id),
        ).fetchone()

    def create_notification_provider_event(
        self,
        event: NotificationProviderEvent,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO notification_provider_events (
                id,
                provider,
                provider_event_id,
                notification_id,
                organization_id,
                status,
                detail,
                occurred_at,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.id,
                event.provider,
                event.provider_event_id,
                event.notification_id,
                event.organization_id,
                event.status.value,
                event.detail,
                event.occurred_at.isoformat(),
                event.created_at.isoformat(),
            ),
        )

    def create_csp_violation_report(self, report: CspViolationReport) -> None:
        self.conn.execute(
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
                report.id,
                report.disposition.value,
                report.effective_directive.value,
                report.violated_directive.value,
                report.blocked_uri_category.value,
                report.document_origin,
                report.status_code,
                report.created_at.isoformat(),
            ),
        )

    def list_notification_provider_events(
        self,
        organization_id: str,
        notification_id: str | None = None,
    ) -> list[sqlite3.Row]:
        clauses = ["organization_id = ?"]
        params = [organization_id]
        if notification_id is not None:
            clauses.append("notification_id = ?")
            params.append(notification_id)
        return self.conn.execute(
            f"""
            SELECT *
            FROM notification_provider_events
            WHERE {' AND '.join(clauses)}
            ORDER BY occurred_at DESC, id
            """,
            params,
        ).fetchall()

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
            SELECT id, organization_id, name, timezone, location_label, latitude, longitude
            FROM sites
            WHERE organization_id = ?
            ORDER BY name
            """,
            (organization_id,),
        ).fetchall()
        return [self._site_from_row(row) for row in rows]

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

    def list_notifications(
        self,
        organization_id: str,
        recipient_user_id: str | None = None,
    ) -> list[sqlite3.Row]:
        clauses = ["organization_id = ?"]
        params = [organization_id]
        if recipient_user_id is not None:
            clauses.append("recipient_user_id = ?")
            params.append(recipient_user_id)
        return self.conn.execute(
            f"""
            SELECT *
            FROM notifications
            WHERE {' AND '.join(clauses)}
            ORDER BY created_at DESC, id
            """,
            params,
        ).fetchall()

    def list_pending_notifications(
        self,
        organization_id: str | None = None,
        limit: int | None = None,
        due_at: str | None = None,
    ) -> list[sqlite3.Row]:
        due_at_value = due_at or datetime.now(timezone.utc).isoformat()
        clauses = [
            "notifications.status = ?",
            "(notifications.next_attempt_at IS NULL OR notifications.next_attempt_at <= ?)",
        ]
        params: list[str | int] = [NotificationStatus.PENDING.value, due_at_value]
        if organization_id is not None:
            clauses.append("notifications.organization_id = ?")
            params.append(organization_id)

        limit_clause = ""
        if limit is not None:
            if limit < 1:
                raise ValueError("notification limit must be greater than zero")
            limit_clause = "LIMIT ?"
            params.append(limit)

        return self.conn.execute(
            f"""
            SELECT
                notifications.*,
                users.email AS recipient_email,
                users.display_name AS recipient_display_name
            FROM notifications
            JOIN users ON users.id = notifications.recipient_user_id
            WHERE {' AND '.join(clauses)}
            ORDER BY notifications.created_at ASC, notifications.id
            {limit_clause}
            """,
            params,
        ).fetchall()
