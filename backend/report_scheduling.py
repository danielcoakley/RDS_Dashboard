from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from typing import Any
from uuid import uuid4

from .artifacts import TenantArtifactStore
from .domain import (
    AuditAction,
    AuditEvent,
    ReportScheduleDelivery,
    ReportScheduleDeliveryStatus,
    ReportScheduleFrequency,
    User,
)
from .notifications import EmailMessage, EmailSender
from .report_artifact_digest import report_artifact_digest_lines
from .retention import report_is_visible
from .run_orchestration import sanitize_report_artifact_content
from .store import SaaSStore


SYSTEM_REPORT_SCHEDULER_USER_ID = "system_report_scheduler"
SYSTEM_REPORT_SCHEDULER_EMAIL = "report-scheduler@rds.local"


@dataclass(frozen=True)
class ScheduledReportDeliverySummary:
    attempted: int
    sent: int
    failed: int
    skipped: int = 0


@dataclass(frozen=True)
class ValidatedReportArtifact:
    content_length: int
    digest_lines: tuple[str, ...]


class ReportScheduleDeliveryError(RuntimeError):
    pass


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _add_months(value: datetime, months: int) -> datetime:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def _next_run_at(current: datetime, frequency: ReportScheduleFrequency, now: datetime) -> datetime:
    next_run_at = current
    while next_run_at <= now:
        if frequency == ReportScheduleFrequency.WEEKLY:
            next_run_at += timedelta(days=7)
        elif frequency == ReportScheduleFrequency.MONTHLY:
            next_run_at = _add_months(next_run_at, 1)
        elif frequency == ReportScheduleFrequency.QUARTERLY:
            next_run_at = _add_months(next_run_at, 3)
        else:
            raise ValueError(f"Unsupported report schedule frequency: {frequency}")
    return next_run_at


def _ensure_system_report_scheduler_user(store: SaaSStore) -> None:
    if store.get_user(SYSTEM_REPORT_SCHEDULER_USER_ID) is not None:
        return
    store.create_user(
        User(
            id=SYSTEM_REPORT_SCHEDULER_USER_ID,
            email=SYSTEM_REPORT_SCHEDULER_EMAIL,
            display_name="Report Scheduler",
        )
    )


def _validate_report_artifact(
    artifact_store: TenantArtifactStore,
    organization_id: str,
    storage_key: str,
) -> ValidatedReportArtifact:
    try:
        payload = artifact_store.get_text(organization_id, storage_key)
    except FileNotFoundError as exc:
        raise ReportScheduleDeliveryError("Report artifact is unavailable") from exc

    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ReportScheduleDeliveryError("Report artifact is not valid JSON") from exc
    if not isinstance(decoded, dict):
        raise ReportScheduleDeliveryError("Report artifact must be a JSON object")
    public_content = sanitize_report_artifact_content(decoded)
    return ValidatedReportArtifact(
        content_length=len(payload),
        digest_lines=tuple(report_artifact_digest_lines(public_content)),
    )


def _scheduled_report_email(
    schedule: Any,
    report: Any,
    artifact_digest_lines: tuple[str, ...] = (),
) -> tuple[str, str]:
    subject = f"Scheduled ISO 50001 report ready: {report['report_type']}"
    body_lines = [
        "Your scheduled ISO 50001 report is ready.",
        "",
        f"Report: {report['id']}",
        f"Report type: {report['report_type']}",
        f"Schedule: {schedule['id']}",
    ]
    if artifact_digest_lines:
        body_lines.extend(["", "Report digest:", *artifact_digest_lines])
    body_lines.extend(
        [
            "",
            "Sign in to the RDS dashboard to review or download the authenticated report artifact.",
        ]
    )
    body = "\n".join(body_lines)
    return subject, body


def _delivery_audit_metadata(
    *,
    delivery_id: str,
    status: ReportScheduleDeliveryStatus,
    frequency: ReportScheduleFrequency,
    next_run_at: str,
    report: Any | None,
    artifact_content_length: int | None,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "delivery_id": delivery_id,
        "delivery_status": status.value,
        "frequency": frequency.value,
        "next_run_at": next_run_at,
        "recipient_email_configured": True,
    }
    if report is not None:
        metadata["report_id"] = report["id"]
        metadata["report_type"] = report["report_type"]
    if artifact_content_length is not None:
        metadata["artifact_content_length"] = artifact_content_length
    return metadata


def process_due_report_schedules(
    store: SaaSStore,
    artifact_store: TenantArtifactStore,
    email_sender: EmailSender,
    organization_id: str | None = None,
    limit: int | None = None,
    now: datetime | None = None,
) -> ScheduledReportDeliverySummary:
    effective_now = (now or _now_utc()).astimezone(timezone.utc)
    due_schedules = store.list_due_report_schedules(
        effective_now.isoformat(),
        organization_id=organization_id,
        limit=limit,
    )
    sent = 0
    failed = 0

    for schedule in due_schedules:
        attempted_at = effective_now
        frequency = ReportScheduleFrequency(schedule["frequency"])
        next_run_at = _next_run_at(
            _parse_datetime(schedule["next_run_at"]),
            frequency,
            effective_now,
        ).isoformat()
        delivery_id = f"report_schedule_delivery_{uuid4().hex}"
        report = store.get_report(schedule["report_id"])
        status = ReportScheduleDeliveryStatus.SENT
        error_message: str | None = None
        artifact_content_length: int | None = None
        artifact_digest_lines: tuple[str, ...] = ()

        try:
            if report is None or report["organization_id"] != schedule["organization_id"]:
                raise ReportScheduleDeliveryError("Report is unavailable")
            if not report_is_visible(report, now=effective_now):
                raise ReportScheduleDeliveryError("Report is no longer published")
            validated_artifact = _validate_report_artifact(
                artifact_store,
                schedule["organization_id"],
                report["storage_key"],
            )
            artifact_content_length = validated_artifact.content_length
            artifact_digest_lines = validated_artifact.digest_lines
            subject, body = _scheduled_report_email(schedule, report, artifact_digest_lines)
            email_sender.send(
                EmailMessage(
                    notification_id=delivery_id,
                    organization_id=schedule["organization_id"],
                    recipient_user_id=schedule["created_by_user_id"],
                    to_email=schedule["recipient_email"],
                    subject=subject,
                    body=body,
                )
            )
        except Exception as exc:
            status = ReportScheduleDeliveryStatus.FAILED
            error_message = str(exc)

        action = (
            AuditAction.REPORT_SCHEDULE_DELIVERED
            if status == ReportScheduleDeliveryStatus.SENT
            else AuditAction.REPORT_SCHEDULE_FAILED
        )
        with store.conn:
            _ensure_system_report_scheduler_user(store)
            store.create_report_schedule_delivery(
                ReportScheduleDelivery(
                    id=delivery_id,
                    organization_id=schedule["organization_id"],
                    schedule_id=schedule["id"],
                    report_id=schedule["report_id"],
                    recipient_email=schedule["recipient_email"],
                    status=status,
                    error_message=error_message,
                    attempted_at=attempted_at,
                    sent_at=attempted_at if status == ReportScheduleDeliveryStatus.SENT else None,
                )
            )
            store.update_report_schedule_delivery_state(
                schedule["id"],
                next_run_at=next_run_at,
                last_delivery_at=attempted_at.isoformat(),
                last_delivery_status=status,
                last_delivery_error=error_message,
            )
            store.create_audit_event(
                AuditEvent(
                    id=f"audit_report_schedule_delivery_{uuid4().hex}",
                    organization_id=schedule["organization_id"],
                    actor_user_id=SYSTEM_REPORT_SCHEDULER_USER_ID,
                    action=action,
                    resource_type="report_schedule",
                    resource_id=schedule["id"],
                    metadata=_delivery_audit_metadata(
                        delivery_id=delivery_id,
                        status=status,
                        frequency=frequency,
                        next_run_at=next_run_at,
                        report=report,
                        artifact_content_length=artifact_content_length,
                    ),
                )
            )

        if status == ReportScheduleDeliveryStatus.SENT:
            sent += 1
        else:
            failed += 1

    return ScheduledReportDeliverySummary(
        attempted=len(due_schedules),
        sent=sent,
        failed=failed,
    )
