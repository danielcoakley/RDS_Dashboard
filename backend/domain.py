from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Sequence


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


class UploadCategory(str, Enum):
    ENERGY = "energy"
    # hdd/cdd stay as internal categories tagging auto-generated weather uploads
    # (backend/weather.py); no upload endpoint accepts them as a user-submitted category.
    HDD = "hdd"
    CDD = "cdd"
    SEU_MAPPING = "seu_mapping"


USER_UPLOADABLE_CATEGORIES = (UploadCategory.ENERGY.value, UploadCategory.SEU_MAPPING.value)


class DataConnectorType(str, Enum):
    SFTP = "sftp"
    API = "api"
    UTILITY_PORTAL = "utility_portal"


class DataConnectorStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"


class DataConnectorSchedule(str, Enum):
    MANUAL = "manual"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class DataConnectorSyncStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


UPLOAD_CATEGORY_VALUES = tuple(category.value for category in UploadCategory)


def normalize_upload_category(value: str) -> str:
    _require_text(value, "upload.category")
    normalized = value.strip().lower().replace("-", "_")
    try:
        return UploadCategory(normalized).value
    except ValueError as exc:
        raise ValueError(
            f"upload.category must be one of: {', '.join(UPLOAD_CATEGORY_VALUES)}"
        ) from exc


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class NotificationChannel(str, Enum):
    EMAIL = "email"


class NotificationStatus(str, Enum):
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"


class NotificationProviderEventStatus(str, Enum):
    DELIVERED = "delivered"
    BOUNCED = "bounced"
    COMPLAINED = "complained"
    DEFERRED = "deferred"
    DROPPED = "dropped"


class CspViolationReportDisposition(str, Enum):
    ENFORCE = "enforce"
    REPORT = "report"


class CspViolationReportDirective(str, Enum):
    DEFAULT_SRC = "default-src"
    SCRIPT_SRC = "script-src"
    STYLE_SRC = "style-src"
    IMG_SRC = "img-src"
    CONNECT_SRC = "connect-src"
    FONT_SRC = "font-src"
    FRAME_SRC = "frame-src"
    OBJECT_SRC = "object-src"
    BASE_URI = "base-uri"
    FORM_ACTION = "form-action"
    FRAME_ANCESTORS = "frame-ancestors"
    OTHER = "other"


class CspBlockedUriCategory(str, Enum):
    SELF = "self"
    INLINE = "inline"
    EVAL = "eval"
    DATA = "data"
    BLOB = "blob"
    HTTP = "http"
    HTTPS = "https"
    NONE = "none"
    OTHER = "other"


class SubscriptionStatus(str, Enum):
    TRIALING = "trialing"
    ACTIVE = "active"
    PAST_DUE = "past_due"
    CANCELED = "canceled"


class PrivacyRequestType(str, Enum):
    DATA_EXPORT = "data_export"
    DELETION = "deletion"
    CORRECTION = "correction"


class PrivacyRequestStatus(str, Enum):
    OPEN = "open"
    IN_REVIEW = "in_review"
    COMPLETED = "completed"
    REJECTED = "rejected"


class PilotReadinessStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    VERIFIED = "verified"
    BLOCKED = "blocked"


class MarketingInquiryStatus(str, Enum):
    NEW = "new"
    CONTACTED = "contacted"
    QUALIFIED = "qualified"
    CLOSED = "closed"


class ReportScheduleFrequency(str, Enum):
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


class ReportScheduleDeliveryStatus(str, Enum):
    SENT = "sent"
    FAILED = "failed"


class ReportReviewStatus(str, Enum):
    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"


class ReportApprovalAssignmentStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    CHANGES_REQUESTED = "changes_requested"
    CANCELED = "canceled"


class AuditAction(str, Enum):
    UPLOAD_STORED = "upload_stored"
    RUN_STARTED = "run_started"
    RUN_SUCCEEDED = "run_succeeded"
    RUN_FAILED = "run_failed"
    REPORT_CREATED = "report_created"
    REPORT_ACCESSED = "report_accessed"
    REPORT_RETENTION_APPLIED = "report_retention_applied"
    REPORT_COMMENT_CREATED = "report_comment_created"
    REPORT_REVIEW_RECORDED = "report_review_recorded"
    REPORT_EXPORTED = "report_exported"
    REPORT_EVIDENCE_EXPORTED = "report_evidence_exported"
    REPORT_EVIDENCE_PACKAGE_CREATED = "report_evidence_package_created"
    REPORT_EVIDENCE_PACKAGE_DOWNLOADED = "report_evidence_package_downloaded"
    REPORT_APPROVAL_ASSIGNED = "report_approval_assigned"
    REPORT_APPROVAL_UPDATED = "report_approval_updated"
    REPORT_APPROVAL_POLICY_UPDATED = "report_approval_policy_updated"
    REPORT_BRANDING_UPDATED = "report_branding_updated"
    REPORT_EVIDENCE_TEMPLATE_UPDATED = "report_evidence_template_updated"
    REPORT_SCHEDULE_CREATED = "report_schedule_created"
    REPORT_SCHEDULE_UPDATED = "report_schedule_updated"
    REPORT_SCHEDULE_DELIVERED = "report_schedule_delivered"
    REPORT_SCHEDULE_FAILED = "report_schedule_failed"
    PRIVACY_SETTINGS_UPDATED = "privacy_settings_updated"
    PRIVACY_REQUEST_CREATED = "privacy_request_created"
    PRIVACY_REQUEST_UPDATED = "privacy_request_updated"
    PILOT_READINESS_UPDATED = "pilot_readiness_updated"
    ORGANIZATION_UPDATED = "organization_updated"
    ANALYSIS_CONFIG_UPDATED = "analysis_config_updated"
    DATA_CONNECTOR_CREATED = "data_connector_created"
    DATA_CONNECTOR_UPDATED = "data_connector_updated"
    DATA_CONNECTOR_SYNC_RECORDED = "data_connector_sync_recorded"
    MEMBERSHIP_ROLE_UPDATED = "membership_role_updated"
    MEMBERSHIP_REMOVED = "membership_removed"
    SUBSCRIPTION_UPDATED = "subscription_updated"
    BILLING_CHECKOUT_CREATED = "billing_checkout_created"
    BILLING_PORTAL_CREATED = "billing_portal_created"
    SITE_CREATED = "site_created"
    METER_CREATED = "meter_created"
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
class MarketingInquiry:
    id: str
    email: str
    name: str
    organization_name: str | None
    role: str | None
    intended_use: str
    message: str
    source_page: str
    status: MarketingInquiryStatus = MarketingInquiryStatus.NEW
    created_at: datetime = field(default_factory=_now_utc)

    def __post_init__(self) -> None:
        _require_values(
            [
                self.id,
                self.email,
                self.name,
                self.intended_use,
                self.message,
                self.source_page,
            ],
            "marketing inquiry field",
        )
        if not isinstance(self.status, MarketingInquiryStatus):
            raise ValueError("marketing_inquiry.status must be a MarketingInquiryStatus")
        if self.organization_name is not None:
            _require_text(self.organization_name, "marketing_inquiry.organization_name")
        if self.role is not None:
            _require_text(self.role, "marketing_inquiry.role")


@dataclass(frozen=True)
class OrganizationReportBranding:
    organization_id: str
    evidence_pack_title: str
    primary_color: str
    logo_url: str | None = None
    support_email: str | None = None
    created_at: datetime = field(default_factory=_now_utc)
    updated_at: datetime = field(default_factory=_now_utc)

    def __post_init__(self) -> None:
        _require_values(
            [self.organization_id, self.evidence_pack_title, self.primary_color],
            "organization report branding field",
        )


@dataclass(frozen=True)
class OrganizationReportApprovalPolicy:
    organization_id: str
    auto_lock_evidence_on_final_approval: bool = True
    minimum_approval_assignments_for_lock: int = 0
    updated_by_user_id: str | None = None
    created_at: datetime = field(default_factory=_now_utc)
    updated_at: datetime = field(default_factory=_now_utc)

    def __post_init__(self) -> None:
        _require_values([self.organization_id], "organization report approval policy field")
        if self.minimum_approval_assignments_for_lock < 0:
            raise ValueError(
                "organization_report_approval_policy.minimum_approval_assignments_for_lock "
                "cannot be negative"
            )
        if self.updated_by_user_id is not None:
            _require_text(
                self.updated_by_user_id,
                "organization_report_approval_policy.updated_by_user_id",
            )


@dataclass(frozen=True)
class OrganizationEvidencePackageTemplate:
    organization_id: str
    template_name: str
    include_report_artifact: bool = True
    include_uploads: bool = True
    include_approval_assignments: bool = True
    include_reviews: bool = True
    include_comments: bool = True
    include_schedules: bool = True
    section_labels_json: str = "{}"
    section_order_json: str = "[]"
    updated_by_user_id: str | None = None
    created_at: datetime = field(default_factory=_now_utc)
    updated_at: datetime = field(default_factory=_now_utc)

    def __post_init__(self) -> None:
        _require_values(
            [
                self.organization_id,
                self.template_name,
                self.section_labels_json,
                self.section_order_json,
            ],
            "organization evidence package template field",
        )
        if self.updated_by_user_id is not None:
            _require_text(
                self.updated_by_user_id,
                "organization_evidence_package_template.updated_by_user_id",
            )


@dataclass(frozen=True)
class SubscriptionPlan:
    id: str
    name: str
    description: str
    monthly_price_cents: int | None
    included_users: int | None
    included_sites: int | None
    monthly_run_limit: int | None
    report_retention_days: int | None
    is_active: bool = True

    def __post_init__(self) -> None:
        _require_values([self.id, self.name, self.description], "subscription plan field")
        for value, field_name in [
            (self.monthly_price_cents, "subscription_plan.monthly_price_cents"),
            (self.included_users, "subscription_plan.included_users"),
            (self.included_sites, "subscription_plan.included_sites"),
            (self.monthly_run_limit, "subscription_plan.monthly_run_limit"),
            (self.report_retention_days, "subscription_plan.report_retention_days"),
        ]:
            if value is not None and value < 0:
                raise ValueError(f"{field_name} cannot be negative")


@dataclass(frozen=True)
class OrganizationSubscription:
    organization_id: str
    plan_id: str
    status: SubscriptionStatus
    current_period_start: datetime | None = None
    current_period_end: datetime | None = None
    trial_ends_at: datetime | None = None
    external_customer_id: str | None = None
    external_subscription_id: str | None = None
    created_at: datetime = field(default_factory=_now_utc)
    updated_at: datetime = field(default_factory=_now_utc)

    def __post_init__(self) -> None:
        _require_values(
            [self.organization_id, self.plan_id],
            "organization subscription field",
        )
        if not isinstance(self.status, SubscriptionStatus):
            raise ValueError("organization_subscription.status must be a SubscriptionStatus")


@dataclass(frozen=True)
class AnalysisConfig:
    organization_id: str
    config: dict[str, Any]
    updated_by_user_id: str
    created_at: datetime = field(default_factory=_now_utc)
    updated_at: datetime = field(default_factory=_now_utc)

    def __post_init__(self) -> None:
        _require_values(
            [self.organization_id, self.updated_by_user_id],
            "analysis config field",
        )
        if not isinstance(self.config, dict):
            raise ValueError("analysis_config.config must be a dict")


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
    location_label: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    created_at: datetime = field(default_factory=_now_utc)

    def __post_init__(self) -> None:
        _require_values([self.id, self.organization_id, self.name, self.timezone], "site field")
        if self.location_label is not None:
            _require_text(self.location_label, "site.location_label")
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("site.latitude and site.longitude must be set together")
        if self.latitude is not None and not -90 <= self.latitude <= 90:
            raise ValueError("site.latitude must be between -90 and 90")
        if self.longitude is not None and not -180 <= self.longitude <= 180:
            raise ValueError("site.longitude must be between -180 and 180")

    @property
    def has_location(self) -> bool:
        return self.latitude is not None and self.longitude is not None


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
class DataConnector:
    id: str
    organization_id: str
    site_id: str
    name: str
    connector_type: DataConnectorType
    source_category: str
    status: DataConnectorStatus
    schedule: DataConnectorSchedule
    created_by_user_id: str
    external_reference: str | None = None
    credential_reference: str | None = None
    updated_by_user_id: str | None = None
    last_sync_at: datetime | None = None
    next_sync_at: datetime | None = None
    created_at: datetime = field(default_factory=_now_utc)
    updated_at: datetime = field(default_factory=_now_utc)

    def __post_init__(self) -> None:
        _require_values(
            [
                self.id,
                self.organization_id,
                self.site_id,
                self.name,
                self.created_by_user_id,
            ],
            "data connector field",
        )
        if not isinstance(self.connector_type, DataConnectorType):
            raise ValueError("data_connector.connector_type must be a DataConnectorType")
        if not isinstance(self.status, DataConnectorStatus):
            raise ValueError("data_connector.status must be a DataConnectorStatus")
        if not isinstance(self.schedule, DataConnectorSchedule):
            raise ValueError("data_connector.schedule must be a DataConnectorSchedule")
        object.__setattr__(
            self,
            "source_category",
            normalize_upload_category(self.source_category),
        )
        if self.external_reference is not None:
            _require_text(self.external_reference, "data_connector.external_reference")
        if self.credential_reference is not None:
            _require_text(self.credential_reference, "data_connector.credential_reference")
        if self.updated_by_user_id is not None:
            _require_text(self.updated_by_user_id, "data_connector.updated_by_user_id")
        if self.last_sync_at is not None and not isinstance(self.last_sync_at, datetime):
            raise ValueError("data_connector.last_sync_at must be a datetime")
        if self.next_sync_at is not None and not isinstance(self.next_sync_at, datetime):
            raise ValueError("data_connector.next_sync_at must be a datetime")


@dataclass(frozen=True)
class DataConnectorSync:
    id: str
    organization_id: str
    connector_id: str
    site_id: str
    source_category: str
    status: DataConnectorSyncStatus
    triggered_by_user_id: str
    upload_id: str | None = None
    records_processed: int = 0
    status_reason: str | None = None
    started_at: datetime = field(default_factory=_now_utc)
    completed_at: datetime | None = None
    created_at: datetime = field(default_factory=_now_utc)

    def __post_init__(self) -> None:
        _require_values(
            [
                self.id,
                self.organization_id,
                self.connector_id,
                self.site_id,
                self.triggered_by_user_id,
            ],
            "data connector sync field",
        )
        if not isinstance(self.status, DataConnectorSyncStatus):
            raise ValueError("data_connector_sync.status must be a DataConnectorSyncStatus")
        object.__setattr__(
            self,
            "source_category",
            normalize_upload_category(self.source_category),
        )
        if self.upload_id is not None:
            _require_text(self.upload_id, "data_connector_sync.upload_id")
        if not isinstance(self.records_processed, int) or self.records_processed < 0:
            raise ValueError("data_connector_sync.records_processed cannot be negative")
        if self.status_reason is not None:
            _require_text(self.status_reason, "data_connector_sync.status_reason")
        if not isinstance(self.started_at, datetime):
            raise ValueError("data_connector_sync.started_at must be a datetime")
        if self.completed_at is not None and not isinstance(self.completed_at, datetime):
            raise ValueError("data_connector_sync.completed_at must be a datetime")
        if self.completed_at is not None and self.completed_at < self.started_at:
            raise ValueError("data_connector_sync.completed_at cannot be before started_at")


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
        object.__setattr__(self, "category", normalize_upload_category(self.category))
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
    expires_at: datetime | None = None
    retention_applied_at: datetime | None = None
    created_at: datetime = field(default_factory=_now_utc)

    def __post_init__(self) -> None:
        _require_values(
            [self.id, self.organization_id, self.run_id, self.report_type, self.storage_key],
            "report field",
        )


@dataclass(frozen=True)
class ReportComment:
    id: str
    organization_id: str
    report_id: str
    author_user_id: str
    body: str
    created_at: datetime = field(default_factory=_now_utc)

    def __post_init__(self) -> None:
        _require_values(
            [self.id, self.organization_id, self.report_id, self.author_user_id, self.body],
            "report comment field",
        )
        if not self.body.strip():
            raise ValueError("report_comment.body must not be blank")


@dataclass(frozen=True)
class ReportReview:
    id: str
    organization_id: str
    report_id: str
    reviewer_user_id: str
    status: ReportReviewStatus
    notes: str | None = None
    created_at: datetime = field(default_factory=_now_utc)

    def __post_init__(self) -> None:
        _require_values(
            [self.id, self.organization_id, self.report_id, self.reviewer_user_id],
            "report review field",
        )
        if not isinstance(self.status, ReportReviewStatus):
            raise ValueError("report_review.status must be a ReportReviewStatus")
        if self.notes is not None and not self.notes.strip():
            raise ValueError("report_review.notes must not be blank")


@dataclass(frozen=True)
class ReportApprovalAssignment:
    id: str
    organization_id: str
    report_id: str
    assigned_to_user_id: str
    assigned_by_user_id: str
    status: ReportApprovalAssignmentStatus = ReportApprovalAssignmentStatus.PENDING
    due_at: datetime | None = None
    notes: str | None = None
    completed_at: datetime | None = None
    created_at: datetime = field(default_factory=_now_utc)
    updated_at: datetime = field(default_factory=_now_utc)

    def __post_init__(self) -> None:
        _require_values(
            [
                self.id,
                self.organization_id,
                self.report_id,
                self.assigned_to_user_id,
                self.assigned_by_user_id,
            ],
            "report approval assignment field",
        )
        if not isinstance(self.status, ReportApprovalAssignmentStatus):
            raise ValueError(
                "report_approval_assignment.status must be a ReportApprovalAssignmentStatus"
            )
        if self.due_at is not None and not isinstance(self.due_at, datetime):
            raise ValueError("report_approval_assignment.due_at must be a datetime")
        if self.completed_at is not None and not isinstance(self.completed_at, datetime):
            raise ValueError("report_approval_assignment.completed_at must be a datetime")
        if self.notes is not None and not self.notes.strip():
            raise ValueError("report_approval_assignment.notes must not be blank")
        if (
            self.status == ReportApprovalAssignmentStatus.PENDING
            and self.completed_at is not None
        ):
            raise ValueError("pending report approval assignments cannot be completed")


@dataclass(frozen=True)
class ReportEvidencePackage:
    id: str
    organization_id: str
    report_id: str
    created_by_user_id: str
    storage_key: str
    filename: str
    content_type: str
    content_length: int
    package_type: str
    format: str
    comment_count: int = 0
    review_count: int = 0
    schedule_count: int = 0
    delivery_count: int = 0
    upload_count: int = 0
    created_at: datetime = field(default_factory=_now_utc)

    def __post_init__(self) -> None:
        _require_values(
            [
                self.id,
                self.organization_id,
                self.report_id,
                self.created_by_user_id,
                self.storage_key,
                self.filename,
                self.content_type,
                self.package_type,
                self.format,
            ],
            "report evidence package field",
        )
        for value, field_name in [
            (self.content_length, "report_evidence_package.content_length"),
            (self.comment_count, "report_evidence_package.comment_count"),
            (self.review_count, "report_evidence_package.review_count"),
            (self.schedule_count, "report_evidence_package.schedule_count"),
            (self.delivery_count, "report_evidence_package.delivery_count"),
            (self.upload_count, "report_evidence_package.upload_count"),
        ]:
            if value < 0:
                raise ValueError(f"{field_name} cannot be negative")


@dataclass(frozen=True)
class ReportSchedule:
    id: str
    organization_id: str
    report_id: str
    created_by_user_id: str
    frequency: ReportScheduleFrequency
    recipient_email: str
    next_run_at: datetime
    is_active: bool = True
    created_at: datetime = field(default_factory=_now_utc)
    updated_at: datetime = field(default_factory=_now_utc)

    def __post_init__(self) -> None:
        _require_values(
            [
                self.id,
                self.organization_id,
                self.report_id,
                self.created_by_user_id,
                self.recipient_email,
            ],
            "report schedule field",
        )
        if not isinstance(self.frequency, ReportScheduleFrequency):
            raise ValueError("report_schedule.frequency must be a ReportScheduleFrequency")
        if "@" not in self.recipient_email.strip():
            raise ValueError("report_schedule.recipient_email must be an email address")
        if not isinstance(self.next_run_at, datetime):
            raise ValueError("report_schedule.next_run_at must be a datetime")


@dataclass(frozen=True)
class ReportScheduleDelivery:
    id: str
    organization_id: str
    schedule_id: str
    report_id: str
    recipient_email: str
    status: ReportScheduleDeliveryStatus
    attempted_at: datetime
    error_message: str | None = None
    sent_at: datetime | None = None
    created_at: datetime = field(default_factory=_now_utc)

    def __post_init__(self) -> None:
        _require_values(
            [
                self.id,
                self.organization_id,
                self.schedule_id,
                self.report_id,
                self.recipient_email,
            ],
            "report schedule delivery field",
        )
        if "@" not in self.recipient_email.strip():
            raise ValueError("report_schedule_delivery.recipient_email must be an email address")
        if not isinstance(self.status, ReportScheduleDeliveryStatus):
            raise ValueError(
                "report_schedule_delivery.status must be a ReportScheduleDeliveryStatus"
            )
        if not isinstance(self.attempted_at, datetime):
            raise ValueError("report_schedule_delivery.attempted_at must be a datetime")
        if self.sent_at is not None and not isinstance(self.sent_at, datetime):
            raise ValueError("report_schedule_delivery.sent_at must be a datetime")


@dataclass(frozen=True)
class AnalysisJob:
    id: str
    organization_id: str
    run_id: str
    upload_id: str
    report_id: str
    requested_by_user_id: str
    client_config: dict[str, Any]
    status: RunStatus = RunStatus.QUEUED
    error_message: str | None = None
    created_at: datetime = field(default_factory=_now_utc)
    completed_at: datetime | None = None

    def __post_init__(self) -> None:
        _require_values(
            [
                self.id,
                self.organization_id,
                self.run_id,
                self.upload_id,
                self.report_id,
                self.requested_by_user_id,
            ],
            "analysis job field",
        )
        if not isinstance(self.status, RunStatus):
            raise ValueError("analysis_job.status must be a RunStatus")
        if not isinstance(self.client_config, dict):
            raise ValueError("analysis_job.client_config must be a dict")


@dataclass(frozen=True)
class Notification:
    id: str
    organization_id: str
    recipient_user_id: str
    channel: NotificationChannel
    event_type: str
    subject: str
    body: str
    resource_type: str
    resource_id: str
    status: NotificationStatus = NotificationStatus.PENDING
    error_message: str | None = None
    created_at: datetime = field(default_factory=_now_utc)
    sent_at: datetime | None = None
    attempt_count: int = 0
    last_attempt_at: datetime | None = None
    next_attempt_at: datetime | None = None

    def __post_init__(self) -> None:
        _require_values(
            [
                self.id,
                self.organization_id,
                self.recipient_user_id,
                self.event_type,
                self.subject,
                self.body,
                self.resource_type,
                self.resource_id,
            ],
            "notification field",
        )
        if not isinstance(self.channel, NotificationChannel):
            raise ValueError("notification.channel must be a NotificationChannel")
        if not isinstance(self.status, NotificationStatus):
            raise ValueError("notification.status must be a NotificationStatus")
        if self.attempt_count < 0:
            raise ValueError("notification.attempt_count cannot be negative")


@dataclass(frozen=True)
class NotificationProviderEvent:
    id: str
    provider: str
    provider_event_id: str
    notification_id: str
    organization_id: str
    status: NotificationProviderEventStatus
    detail: str = ""
    occurred_at: datetime = field(default_factory=_now_utc)
    created_at: datetime = field(default_factory=_now_utc)

    def __post_init__(self) -> None:
        _require_values(
            [
                self.id,
                self.provider,
                self.provider_event_id,
                self.notification_id,
                self.organization_id,
            ],
            "notification provider event field",
        )
        if not isinstance(self.status, NotificationProviderEventStatus):
            raise ValueError(
                "notification_provider_event.status must be a NotificationProviderEventStatus"
            )
        if not isinstance(self.detail, str):
            raise ValueError("notification_provider_event.detail must be a string")
        if not isinstance(self.occurred_at, datetime):
            raise ValueError("notification_provider_event.occurred_at must be a datetime")
        if not isinstance(self.created_at, datetime):
            raise ValueError("notification_provider_event.created_at must be a datetime")


@dataclass(frozen=True)
class CspViolationReport:
    id: str
    disposition: CspViolationReportDisposition
    effective_directive: CspViolationReportDirective
    violated_directive: CspViolationReportDirective
    blocked_uri_category: CspBlockedUriCategory
    document_origin: str = ""
    status_code: int | None = None
    created_at: datetime = field(default_factory=_now_utc)

    def __post_init__(self) -> None:
        _require_text(self.id, "csp_violation_report.id")
        if not isinstance(self.disposition, CspViolationReportDisposition):
            raise ValueError(
                "csp_violation_report.disposition must be a CspViolationReportDisposition"
            )
        if not isinstance(self.effective_directive, CspViolationReportDirective):
            raise ValueError(
                "csp_violation_report.effective_directive must be a CspViolationReportDirective"
            )
        if not isinstance(self.violated_directive, CspViolationReportDirective):
            raise ValueError(
                "csp_violation_report.violated_directive must be a CspViolationReportDirective"
            )
        if not isinstance(self.blocked_uri_category, CspBlockedUriCategory):
            raise ValueError(
                "csp_violation_report.blocked_uri_category must be a CspBlockedUriCategory"
            )
        if not isinstance(self.document_origin, str):
            raise ValueError("csp_violation_report.document_origin must be a string")
        if self.status_code is not None and not 0 <= self.status_code <= 999:
            raise ValueError("csp_violation_report.status_code must be between 0 and 999")
        if not isinstance(self.created_at, datetime):
            raise ValueError("csp_violation_report.created_at must be a datetime")


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
