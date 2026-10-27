from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import uuid4
import sqlite3

from .access_control import require_permission
from .artifacts import TenantArtifactStore
from .domain import AuditAction, AuditEvent
from .rbac import Action
from .store import SaaSStore
from .subscriptions import ensure_default_subscription


@dataclass(frozen=True)
class ReportRetentionSummary:
    organization_id: str
    evaluated_count: int
    expired_count: int
    deleted_artifact_count: int
    missing_artifact_count: int
    deleted_evidence_package_artifact_count: int
    missing_evidence_package_artifact_count: int
    deleted_evidence_package_metadata_count: int
    retained_report_ids: list[str]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def parse_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def report_expires_at_for_organization(
    store: SaaSStore,
    organization_id: str,
    *,
    created_at: datetime | None = None,
) -> datetime | None:
    with store.conn:
        ensure_default_subscription(store, organization_id)
    subscription = store.get_organization_subscription(organization_id)
    if subscription is None:
        raise ValueError(f"Subscription for organization {organization_id} was not found")

    retention_days = subscription["report_retention_days"]
    if retention_days is None:
        return None
    return (created_at or _now_utc()) + timedelta(days=int(retention_days))


def report_is_expired(report: sqlite3.Row, *, now: datetime | None = None) -> bool:
    expires_at = parse_datetime(report["expires_at"])
    if expires_at is None:
        return False
    return expires_at <= (now or _now_utc())


def report_is_visible(report: sqlite3.Row, *, now: datetime | None = None) -> bool:
    return (
        bool(report["is_published"])
        and report["retention_applied_at"] is None
        and not report_is_expired(report, now=now)
    )


def require_report_visible(report: sqlite3.Row, *, now: datetime | None = None) -> None:
    if report["retention_applied_at"] is not None:
        raise ValueError(f"Report {report['id']} has been removed by retention policy")
    if report_is_expired(report, now=now):
        raise ValueError(f"Report {report['id']} has expired under the tenant retention policy")
    if not bool(report["is_published"]):
        raise ValueError(f"Report {report['id']} is not published")


def apply_report_retention_policy(
    store: SaaSStore,
    user_id: str,
    organization_id: str,
    *,
    artifact_store: TenantArtifactStore | None = None,
    now: datetime | None = None,
) -> ReportRetentionSummary:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.MANAGE_REPORTS,
    )

    effective_now = now or _now_utc()
    applied_at = effective_now.isoformat()
    reports = store.list_reports(organization_id)
    expired_reports = [
        report
        for report in reports
        if bool(report["is_published"])
        and report["retention_applied_at"] is None
        and report_is_expired(report, now=effective_now)
    ]

    retained_report_ids: list[str] = []
    deleted_artifact_count = 0
    missing_artifact_count = 0
    deleted_evidence_package_artifact_count = 0
    missing_evidence_package_artifact_count = 0
    deleted_evidence_package_metadata_count = 0
    with store.conn:
        for report in expired_reports:
            artifact_deleted: bool | None = None
            evidence_packages = store.list_report_evidence_packages(organization_id, report["id"])
            evidence_package_artifact_deleted_count = 0
            evidence_package_artifact_missing_count = 0
            evidence_package_metadata_deleted_count = 0
            if artifact_store is not None:
                artifact_deleted = artifact_store.delete(organization_id, report["storage_key"])
                if artifact_deleted:
                    deleted_artifact_count += 1
                else:
                    missing_artifact_count += 1

                for package in evidence_packages:
                    package_deleted = artifact_store.delete(organization_id, package["storage_key"])
                    if package_deleted:
                        deleted_artifact_count += 1
                        deleted_evidence_package_artifact_count += 1
                        evidence_package_artifact_deleted_count += 1
                    else:
                        missing_artifact_count += 1
                        missing_evidence_package_artifact_count += 1
                        evidence_package_artifact_missing_count += 1

                evidence_package_metadata_deleted_count = store.delete_report_evidence_packages(
                    organization_id,
                    report["id"],
                )
                deleted_evidence_package_metadata_count += (
                    evidence_package_metadata_deleted_count
                )

            store.mark_report_retention_applied(report["id"], applied_at)
            retained_report_ids.append(report["id"])
            store.create_audit_event(
                AuditEvent(
                    id=f"audit_report_retention_{uuid4().hex}",
                    organization_id=organization_id,
                    actor_user_id=user_id,
                    action=AuditAction.REPORT_RETENTION_APPLIED,
                    resource_type="report",
                    resource_id=report["id"],
                    metadata={
                        "artifact_location_configured": True,
                        "expires_at": report["expires_at"],
                        "artifact_deleted": artifact_deleted,
                        "evidence_package_count": len(evidence_packages),
                        "evidence_package_artifact_deleted_count": (
                            evidence_package_artifact_deleted_count
                        ),
                        "evidence_package_artifact_missing_count": (
                            evidence_package_artifact_missing_count
                        ),
                        "evidence_package_metadata_deleted_count": (
                            evidence_package_metadata_deleted_count
                        ),
                    },
                )
            )

    return ReportRetentionSummary(
        organization_id=organization_id,
        evaluated_count=len(reports),
        expired_count=len(retained_report_ids),
        deleted_artifact_count=deleted_artifact_count,
        missing_artifact_count=missing_artifact_count,
        deleted_evidence_package_artifact_count=deleted_evidence_package_artifact_count,
        missing_evidence_package_artifact_count=missing_evidence_package_artifact_count,
        deleted_evidence_package_metadata_count=deleted_evidence_package_metadata_count,
        retained_report_ids=retained_report_ids,
    )
