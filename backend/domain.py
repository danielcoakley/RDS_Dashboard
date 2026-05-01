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


class UploadStatus(str, Enum):
    PENDING = "pending"
    STORED = "stored"
    REJECTED = "rejected"


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


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
