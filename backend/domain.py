from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable, Sequence


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required")


def _require_values(values: Iterable[str], field_name: str) -> None:
    for value in values:
        _require_text(value, field_name)


class Role(str, Enum):
    OWNER = "owner"
    MANAGER = "manager"
    VIEWER = "viewer"


class InviteStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REVOKED = "revoked"


class UploadStatus(str, Enum):
    PENDING = "pending"
    STORED = "stored"
    REJECTED = "rejected"


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class AuditAction(str, Enum):
    UPLOAD_STORED = "upload_stored"
    RUN_STARTED = "run_started"
    RUN_SUCCEEDED = "run_succeeded"
    RUN_FAILED = "run_failed"
    REPORT_CREATED = "report_created"
    INVITE_CREATED = "invite_created"
    INVITE_ACCEPTED = "invite_accepted"
    INVITE_REVOKED = "invite_revoked"


@dataclass(frozen=True)
class User:
    id: str
    email: str
    display_name: str
    is_active: bool = True
    created_at: datetime = field(default_factory=_now_utc)

    def __post_init__(self) -> None:
        _require_text(self.id, "user.id")
        _require_text(self.email, "user.email")
        _require_text(self.display_name, "user.display_name")


@dataclass(frozen=True)
class Organization:
    id: str
    name: str
    slug: str
    created_at: datetime = field(default_factory=_now_utc)

    def __post_init__(self) -> None:
        _require_text(self.id, "organization.id")
        _require_text(self.name, "organization.name")
        _require_text(self.slug, "organization.slug")


@dataclass(frozen=True)
class Membership:
    user_id: str
    organization_id: str
    role: Role
    invited_by_user_id: str | None = None
    created_at: datetime = field(default_factory=_now_utc)

    def __post_init__(self) -> None:
        _require_text(self.user_id, "membership.user_id")
        _require_text(self.organization_id, "membership.organization_id")
        if not isinstance(self.role, Role):
            raise ValueError("membership.role must be a Role")


@dataclass(frozen=True)
class OrganizationInvite:
    id: str
    organization_id: str
    email: str
    role: Role
    invited_by_user_id: str
    status: InviteStatus = InviteStatus.PENDING
    accepted_by_user_id: str | None = None
    created_at: datetime = field(default_factory=_now_utc)
    accepted_at: datetime | None = None

    def __post_init__(self) -> None:
        _require_values(
            [self.id, self.organization_id, self.email, self.invited_by_user_id],
            "organization invite field",
        )
        if not isinstance(self.role, Role):
            raise ValueError("organization_invite.role must be a Role")
        if not isinstance(self.status, InviteStatus):
            raise ValueError("organization_invite.status must be an InviteStatus")


@dataclass(frozen=True)
class Site:
    id: str
    organization_id: str
    name: str
    timezone: str
    created_at: datetime = field(default_factory=_now_utc)

    def __post_init__(self) -> None:
        _require_values([self.id, self.organization_id, self.name, self.timezone], "site field")


@dataclass(frozen=True)
class Meter:
    id: str
    organization_id: str
    site_id: str
    display_name: str
    commodity: str
    unit: str
    source_column: str
    is_seu: bool = False

    def __post_init__(self) -> None:
        _require_values(
            [
                self.id,
                self.organization_id,
                self.site_id,
                self.display_name,
                self.commodity,
                self.unit,
                self.source_column,
            ],
            "meter field",
        )


@dataclass(frozen=True)
class Upload:
    id: str
    organization_id: str
    site_id: str
    uploaded_by_user_id: str
    category: str
    storage_key: str
    checksum: str
    status: UploadStatus = UploadStatus.PENDING
    created_at: datetime = field(default_factory=_now_utc)

    def __post_init__(self) -> None:
        _require_values(
            [
                self.id,
                self.organization_id,
                self.site_id,
                self.uploaded_by_user_id,
                self.category,
                self.storage_key,
                self.checksum,
            ],
            "upload field",
        )
        if not isinstance(self.status, UploadStatus):
            raise ValueError("upload.status must be an UploadStatus")


@dataclass(frozen=True)
class Run:
    id: str
    organization_id: str
    site_id: str
    requested_by_user_id: str
    upload_ids: Sequence[str]
    status: RunStatus = RunStatus.QUEUED
    error_message: str | None = None
    created_at: datetime = field(default_factory=_now_utc)
    completed_at: datetime | None = None

    def __post_init__(self) -> None:
        _require_values(
            [self.id, self.organization_id, self.site_id, self.requested_by_user_id],
            "run field",
        )
        if not self.upload_ids:
            raise ValueError("run.upload_ids must contain at least one upload id")
        _require_values(self.upload_ids, "run.upload_ids")
        if not isinstance(self.status, RunStatus):
            raise ValueError("run.status must be a RunStatus")


@dataclass(frozen=True)
class Report:
    id: str
    organization_id: str
    run_id: str
    report_type: str
    storage_key: str
    is_published: bool = False
    created_at: datetime = field(default_factory=_now_utc)

    def __post_init__(self) -> None:
        _require_values(
            [self.id, self.organization_id, self.run_id, self.report_type, self.storage_key],
            "report field",
        )


@dataclass(frozen=True)
class AuditEvent:
    id: str
    organization_id: str
    actor_user_id: str
    action: AuditAction
    resource_type: str
    resource_id: str
    metadata: dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_now_utc)

    def __post_init__(self) -> None:
        _require_values(
            [
                self.id,
                self.organization_id,
                self.actor_user_id,
                self.resource_type,
                self.resource_id,
            ],
            "audit event field",
        )
        if not isinstance(self.action, AuditAction):
            raise ValueError("audit_event.action must be an AuditAction")
