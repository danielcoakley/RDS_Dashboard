from __future__ import annotations

import base64
import binascii
import calendar
import csv
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO, StringIO
import json
import re
from textwrap import wrap
from typing import Any
from uuid import uuid4

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, PlainTextResponse, Response
import pandas as pd
from pydantic import BaseModel, Field

from .access_control import AccessDenied, TenantContext, require_permission, require_resource_scope
from .analysis_configs import (
    TenantAnalysisConfig,
    get_tenant_analysis_config,
    update_tenant_analysis_config,
)
from .artifacts import (
    ArtifactStoreSmokeTestError,
    TenantArtifactStore,
    artifact_store_from_env,
    artifact_store_smoke_test,
)
from .auth_context import (
    CLERK_AUTH_MODE,
    AuthenticatedUser,
    dev_token_from_claims,
    request_authenticated_user,
)
from .billing import (
    BillingConfigurationError,
    BillingWebhookResult,
    BillingWebhookVerificationError,
    create_billing_checkout_session,
    create_billing_portal_session,
    process_billing_webhook_event,
)
from .clerk_sync import slug_part, sync_clerk_authenticated_user
from .clerk_webhooks import (
    ClerkWebhookConfigurationError,
    ClerkWebhookVerificationError,
    process_clerk_webhook_event,
    verify_clerk_webhook_event,
)
from .csp_reports import (
    CspViolationReportRetentionSummary,
    CspViolationReportPayloadError,
    CspViolationReportTooLargeError,
    apply_csp_violation_report_retention,
    process_csp_violation_report,
)
from .database import initialize_database_from_env
from .domain import (
    AuditAction,
    AuditEvent,
    CspViolationReport,
    CspViolationReportDirective,
    CspViolationReportDisposition,
    DataConnector,
    DataConnectorSchedule,
    DataConnectorStatus,
    DataConnectorSync,
    DataConnectorSyncStatus,
    DataConnectorType,
    MarketingInquiry,
    MarketingInquiryStatus,
    Meter,
    Notification,
    NotificationChannel,
    NotificationStatus,
    Organization,
    OrganizationEvidencePackageTemplate,
    OrganizationReportApprovalPolicy,
    OrganizationReportBranding,
    PilotReadinessStatus,
    PrivacyRequestStatus,
    PrivacyRequestType,
    Role,
    ReportApprovalAssignment,
    ReportApprovalAssignmentStatus,
    ReportComment,
    ReportReview,
    ReportReviewStatus,
    ReportSchedule,
    ReportScheduleDeliveryStatus,
    ReportScheduleFrequency,
    Run,
    RunStatus,
    ReportEvidencePackage,
    Site,
    SubscriptionPlan,
    Upload,
    UploadCategory,
    UploadStatus,
    normalize_upload_category,
    User,
)
from .invitations import (
    accept_organization_invite,
    create_organization_invite,
    revoke_organization_invite,
)
from .monitoring import (
    PROMETHEUS_CONTENT_TYPE,
    MetricsAuthenticationError,
    MetricsConfigurationError,
    collect_operational_metrics,
    format_prometheus_metrics,
    verify_metrics_request,
)
from .membership_admin import (
    MembershipAdminRecord,
    remove_membership,
    update_membership_role,
)
from .marketing_handoff import (
    MarketingInquiryCrmSender,
    marketing_inquiry_crm_sender_from_env,
)
from .marketing_rate_limit import (
    MarketingInquiryRateLimitConfigurationError,
    MarketingInquiryRateLimitExceeded,
    marketing_inquiry_rate_limit_key,
    marketing_inquiry_rate_limit_policy_from_env,
)
from .notifications import (
    EmailMessage,
    EmailSender,
    NotificationProviderWebhookConfigurationError,
    NotificationProviderWebhookPayloadError,
    NotificationProviderWebhookResult,
    NotificationProviderWebhookVerificationError,
    email_sender_from_env,
    marketing_inquiry_recipient_email_from_env,
    process_notification_provider_webhook_event,
    process_pending_notifications,
)
from .observability import REQUEST_ID_HEADER, configure_request_logging
from .onboarding import create_owner_organization
from .operations import (
    ALERT_DELIVERY_TEST_EVENT_TYPE,
    ARTIFACT_STORE_TEST_EVENT_TYPE,
    BILLING_WEBHOOK_REPLAY_EVENT_TYPE,
    CLERK_WEBHOOK_REPLAY_EVENT_TYPE,
    DATABASE_TEST_EVENT_TYPE,
    NOTIFICATION_PROVIDER_WEBHOOK_REPLAY_EVENT_TYPE,
    RUN_LIFECYCLE_TEST_EVENT_TYPE,
    SUPPORTED_PLATFORM_OPERATION_EVENT_STATUSES,
    SUPPORTED_PLATFORM_OPERATION_EVENT_TYPES,
    TENANT_ISOLATION_TEST_EVENT_TYPE,
    WORKER_RUN_LIFECYCLE_TEST_EVENT_TYPE,
    AlertDeliveryError,
    OperationsConfigurationError,
    RunLifecycleSmokeTestError,
    TenantIsolationSmokeTestError,
    WorkerRunLifecycleSmokeTestError,
    run_lifecycle_smoke_test,
    run_tenant_isolation_smoke_test,
    send_alert_delivery_test,
    worker_run_lifecycle_smoke_test,
)
from .readiness import readiness_report
from .report_artifact_digest import report_artifact_digest_lines
from .rbac import Action
from .retention import (
    ReportRetentionSummary as TenantReportRetentionSummary,
    apply_report_retention_policy,
    report_is_visible,
    require_report_visible,
)
from .run_orchestration import (
    AnalysisRunRequest,
    StoredUploadAnalysisRunRequest,
    execute_analysis_run,
    execute_analysis_run_from_upload,
    process_analysis_job,
    queue_analysis_run_from_upload,
    sanitize_report_artifact_content,
)
from .runtime_config import runtime_env_from_env
from .security import SecurityConfigurationError, configure_security_middleware
from .storage_keys import evidence_package_storage_key, safe_key_part, upload_storage_key
from .store import SaaSStore
from .support import (
    SupportAuthenticationError,
    SupportConfigurationError,
    verify_support_request,
)
from .subscriptions import (
    TenantSubscription,
    get_tenant_subscription,
    list_available_subscription_plans,
    require_run_capacity,
    require_site_capacity,
    SubscriptionLimitExceeded,
    update_tenant_subscription_plan,
)


API_TITLE = "RDS Energy Analytics SaaS API"
API_VERSION = "0.1.0"
REPORT_EVIDENCE_PACKAGE_CONTENT_TYPE = "application/json"
REPORT_EVIDENCE_PACKAGE_PDF_CONTENT_TYPE = "application/pdf"
REPORT_EVIDENCE_PACKAGE_TYPE = "iso50001_report_evidence"
UPLOAD_SOURCE_FILE_CONTRACT: tuple[tuple[UploadCategory, str, bool], ...] = (
    (UploadCategory.ENERGY, "Energy data", True),
    (UploadCategory.HDD, "Heating degree days", True),
    (UploadCategory.CDD, "Cooling degree days", True),
    (UploadCategory.SEU_MAPPING, "SEU mapping", True),
)
SOURCE_FILE_DATE_HEADERS = {"date", "timestamp", "datetime", "time"}
DEFAULT_EVIDENCE_PACK_TITLE = "ISO 50001 Evidence Package"
DEFAULT_EVIDENCE_PACK_PRIMARY_COLOR = "#0f766e"
DEFAULT_EVIDENCE_PACKAGE_TEMPLATE_NAME = "Full ISO 50001 Evidence Package"
EVIDENCE_PACKAGE_SECTION_DEFINITIONS: tuple[tuple[str, str, str], ...] = (
    ("report_artifact", "include_report_artifact", "Report Artifact Summary"),
    ("uploads", "include_uploads", "Upload Lineage"),
    (
        "approval_assignments",
        "include_approval_assignments",
        "Approval Assignments",
    ),
    ("reviews", "include_reviews", "Management Review Decisions"),
    ("comments", "include_comments", "Collaboration Evidence"),
    ("schedules", "include_schedules", "Scheduled Delivery Evidence"),
)
EVIDENCE_PACKAGE_SECTION_KEYS = tuple(
    section_key for section_key, _include_field, _default_label in EVIDENCE_PACKAGE_SECTION_DEFINITIONS
)
EVIDENCE_PACKAGE_SECTION_INCLUDE_FIELDS = {
    section_key: include_field
    for section_key, include_field, _default_label in EVIDENCE_PACKAGE_SECTION_DEFINITIONS
}
DEFAULT_EVIDENCE_PACKAGE_SECTION_LABELS = {
    section_key: default_label
    for section_key, _include_field, default_label in EVIDENCE_PACKAGE_SECTION_DEFINITIONS
}
DEFAULT_EVIDENCE_PACKAGE_SECTION_ORDER = tuple(EVIDENCE_PACKAGE_SECTION_KEYS)
DEFAULT_AUTO_LOCK_EVIDENCE_ON_FINAL_APPROVAL = True
DEFAULT_MINIMUM_APPROVAL_ASSIGNMENTS_FOR_LOCK = 0
MARKETING_INQUIRY_USE_CASES = {
    "audit_readiness",
    "consultant_delivery",
    "estate_portfolio",
    "measurement_verification",
    "other",
}
MARKETING_INQUIRY_DEFAULT_SOURCE = "landing"
MARKETING_INQUIRY_NOTIFICATION_ORGANIZATION_ID = "platform"
MARKETING_INQUIRY_NOTIFICATION_RECIPIENT_USER_ID = "marketing_inquiries"
MARKETING_INQUIRY_DUPLICATE_WINDOW_MINUTES = 30
PILOT_READINESS_ITEMS: dict[str, str] = {
    "dpa_signed": "Data processing agreement signed",
    "privacy_notice_reviewed": "Privacy notice reviewed",
    "production_auth_verified": "Production auth verified",
    "production_billing_verified": "Production billing verified",
    "email_delivery_verified": "Email delivery verified",
    "storage_backup_verified": "Storage and backup verified",
    "tenant_isolation_reviewed": "Tenant isolation reviewed",
    "support_owner_assigned": "Support owner assigned",
}


class OwnerOrganizationCreate(BaseModel):
    user_id: str
    email: str
    display_name: str
    organization_id: str
    organization_name: str
    organization_slug: str


class OwnerOrganizationResponse(BaseModel):
    user_id: str
    organization_id: str
    role: str


class DevSessionCreate(BaseModel):
    user_id: str
    organization_id: str | None = None


class DevSessionResponse(BaseModel):
    user_id: str
    organization_id: str | None
    role: str | None
    auth_token: str


class MarketingInquiryCreate(BaseModel):
    name: str
    email: str
    organization_name: str | None = None
    role: str | None = None
    intended_use: str
    message: str
    source_page: str | None = None
    website: str | None = None


class MarketingInquirySummary(BaseModel):
    id: str
    status: MarketingInquiryStatus
    created_at: str


class MarketingInquiryDetail(BaseModel):
    id: str
    email: str
    name: str
    organization_name: str | None
    role: str | None
    intended_use: str
    message: str
    source_page: str
    status: MarketingInquiryStatus
    created_at: str
    assigned_to: str | None
    follow_up_notes: str | None
    updated_at: str | None


class MarketingInquiryStatusUpdate(BaseModel):
    status: MarketingInquiryStatus | None = None
    assigned_to: str | None = None
    follow_up_notes: str | None = None


class MarketingInquiryStatusCount(BaseModel):
    status: MarketingInquiryStatus
    count: int


class MarketingInquiryAssigneeQueueSummary(BaseModel):
    assigned_to: str | None
    total_count: int
    by_status: list[MarketingInquiryStatusCount]


class MarketingInquiryQueueSummary(BaseModel):
    total_count: int
    unassigned_count: int
    by_status: list[MarketingInquiryStatusCount]
    by_assignee: list[MarketingInquiryAssigneeQueueSummary]


class MarketingInquiryActivitySummary(BaseModel):
    id: str
    inquiry_id: str
    actor_label: str
    action: str
    status: MarketingInquiryStatus | None
    assigned_to: str | None
    changed_fields: list[str]
    notes_changed: bool
    created_at: str


class ClerkSyncResponse(BaseModel):
    user_id: str
    organization_id: str
    organization_slug: str
    role: str


class ClerkWebhookResponse(BaseModel):
    event_type: str
    action: str
    user_id: str | None = None
    organization_id: str | None = None
    role: str | None = None


class OrganizationSummary(BaseModel):
    id: str
    name: str
    slug: str


class OrganizationProfileUpdate(BaseModel):
    name: str
    slug: str


class OrganizationReportBrandingSummary(BaseModel):
    organization_id: str
    evidence_pack_title: str
    primary_color: str
    logo_url: str | None
    support_email: str | None
    updated_at: str | None


class OrganizationReportBrandingUpdate(BaseModel):
    evidence_pack_title: str
    primary_color: str
    logo_url: str | None = None
    support_email: str | None = None


class OrganizationReportApprovalPolicySummary(BaseModel):
    organization_id: str
    auto_lock_evidence_on_final_approval: bool
    minimum_approval_assignments_for_lock: int
    updated_by_user_id: str | None
    updated_at: str | None


class OrganizationReportApprovalPolicyUpdate(BaseModel):
    auto_lock_evidence_on_final_approval: bool = DEFAULT_AUTO_LOCK_EVIDENCE_ON_FINAL_APPROVAL
    minimum_approval_assignments_for_lock: int = Field(
        DEFAULT_MINIMUM_APPROVAL_ASSIGNMENTS_FOR_LOCK,
        ge=0,
        le=25,
    )


class OrganizationEvidencePackageTemplateSummary(BaseModel):
    organization_id: str
    template_name: str
    include_report_artifact: bool
    include_uploads: bool
    include_approval_assignments: bool
    include_reviews: bool
    include_comments: bool
    include_schedules: bool
    section_labels: dict[str, str]
    section_order: list[str]
    updated_by_user_id: str | None
    updated_at: str | None


class OrganizationEvidencePackageTemplateUpdate(BaseModel):
    template_name: str = DEFAULT_EVIDENCE_PACKAGE_TEMPLATE_NAME
    include_report_artifact: bool = True
    include_uploads: bool = True
    include_approval_assignments: bool = True
    include_reviews: bool = True
    include_comments: bool = True
    include_schedules: bool = True
    section_labels: dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_EVIDENCE_PACKAGE_SECTION_LABELS),
    )
    section_order: list[str] = Field(
        default_factory=lambda: list(DEFAULT_EVIDENCE_PACKAGE_SECTION_ORDER),
    )


class OrganizationPrivacySettingsSummary(BaseModel):
    organization_id: str
    data_processing_acknowledged_at: str | None
    data_processing_acknowledged_by_user_id: str | None
    data_processing_contact_email: str | None
    subprocessor_notifications_enabled: bool
    analytics_data_sharing_enabled: bool
    updated_at: str | None


class OrganizationPrivacySettingsUpdate(BaseModel):
    data_processing_contact_email: str | None = None
    subprocessor_notifications_enabled: bool
    analytics_data_sharing_enabled: bool
    acknowledge_data_processing: bool = False


class OrganizationPrivacyRequestSummary(BaseModel):
    id: str
    organization_id: str
    requested_by_user_id: str
    subject_user_id: str | None
    request_type: PrivacyRequestType
    status: PrivacyRequestStatus
    description: str | None
    resolution_notes: str | None
    completed_at: str | None
    created_at: str
    updated_at: str


class OrganizationPrivacyRequestCreate(BaseModel):
    request_type: PrivacyRequestType
    subject_user_id: str | None = None
    description: str | None = None


class OrganizationPrivacyRequestStatusUpdate(BaseModel):
    status: PrivacyRequestStatus
    resolution_notes: str | None = None


class OrganizationPilotReadinessItemSummary(BaseModel):
    item_key: str
    label: str
    status: PilotReadinessStatus
    evidence_url: str | None
    notes: str | None
    updated_by_user_id: str | None
    updated_at: str | None


class OrganizationPilotReadinessSummary(BaseModel):
    organization_id: str
    verified_count: int
    total_count: int
    items: list[OrganizationPilotReadinessItemSummary]


class OrganizationPilotReadinessUpdate(BaseModel):
    status: PilotReadinessStatus
    evidence_url: str | None = None
    notes: str | None = None


class AnalysisConfigSummary(BaseModel):
    organization_id: str
    config: dict[str, Any]
    source: str
    updated_by_user_id: str | None
    updated_at: str | None


class AnalysisConfigUpdate(BaseModel):
    config: dict[str, Any]


class DataConnectorSummary(BaseModel):
    id: str
    organization_id: str
    site_id: str
    name: str
    connector_type: DataConnectorType
    source_category: str
    status: DataConnectorStatus
    schedule: DataConnectorSchedule
    external_reference: str | None
    credential_reference_configured: bool
    created_by_user_id: str
    updated_by_user_id: str | None
    last_sync_at: str | None
    last_sync_status: DataConnectorSyncStatus | None = None
    last_sync_error: str | None = None
    next_sync_at: str | None
    created_at: str
    updated_at: str


class DataConnectorCreate(BaseModel):
    connector_id: str
    site_id: str
    name: str
    connector_type: DataConnectorType
    source_category: UploadCategory
    status: DataConnectorStatus = DataConnectorStatus.DRAFT
    schedule: DataConnectorSchedule = DataConnectorSchedule.MANUAL
    external_reference: str | None = None
    credential_reference: str | None = None
    next_sync_at: str | None = None


class DataConnectorUpdate(BaseModel):
    name: str
    status: DataConnectorStatus
    schedule: DataConnectorSchedule
    external_reference: str | None = None
    credential_reference: str | None = None
    next_sync_at: str | None = None


class DataConnectorSyncSummary(BaseModel):
    id: str
    organization_id: str
    connector_id: str
    site_id: str
    source_category: str
    status: DataConnectorSyncStatus
    upload_id: str | None
    records_processed: int
    status_reason: str | None
    triggered_by_user_id: str
    started_at: str
    completed_at: str | None
    created_at: str


class DataConnectorSyncRecord(BaseModel):
    status: DataConnectorSyncStatus
    upload_id: str | None = None
    records_processed: int = Field(0, ge=0, le=10_000_000)
    status_reason: str | None = None
    started_at: str | None = None
    completed_at: str | None = None


class DataConnectorImportCreate(BaseModel):
    upload_id: str | None = None
    filename: str
    content_base64: str
    records_processed: int = Field(0, ge=0, le=10_000_000)
    status_reason: str | None = None
    started_at: str | None = None
    completed_at: str | None = None


class SiteSummary(BaseModel):
    id: str
    organization_id: str
    name: str
    timezone: str


class SiteCreate(BaseModel):
    site_id: str
    name: str
    timezone: str


class MembershipSummary(BaseModel):
    user_id: str
    organization_id: str
    role: str
    invited_by_user_id: str | None
    email: str
    display_name: str
    is_active: bool


class MembershipRoleUpdate(BaseModel):
    role: Role


class OrganizationInviteCreate(BaseModel):
    email: str
    role: Role


class OrganizationInviteAccept(BaseModel):
    user_id: str
    email: str
    display_name: str


class OrganizationInviteSummary(BaseModel):
    id: str
    organization_id: str
    email: str
    role: str
    invited_by_user_id: str
    status: str
    accepted_by_user_id: str | None
    accepted_at: str | None


class MeterSummary(BaseModel):
    id: str
    organization_id: str
    site_id: str
    display_name: str
    commodity: str
    unit: str
    source_column: str
    is_seu: bool


class MeterCreate(BaseModel):
    meter_id: str
    site_id: str
    display_name: str
    commodity: str
    unit: str
    source_column: str
    is_seu: bool = False


class UploadSummary(BaseModel):
    id: str
    organization_id: str
    site_id: str
    uploaded_by_user_id: str
    category: str
    checksum: str
    status: str


class DataConnectorImportSummary(BaseModel):
    connector: DataConnectorSummary
    upload: UploadSummary
    sync: DataConnectorSyncSummary


class UploadCreate(BaseModel):
    upload_id: str
    site_id: str
    category: str
    filename: str
    checksum: str


class UploadContentCreate(BaseModel):
    upload_id: str
    site_id: str
    category: str
    filename: str
    content_base64: str


class UploadReadinessFileSummary(BaseModel):
    category: str
    label: str
    is_required: bool
    is_ready: bool
    latest_upload: UploadSummary | None


class SiteUploadReadinessSummary(BaseModel):
    site_id: str
    site_name: str
    ready_count: int
    required_count: int
    is_ready: bool
    files: list[UploadReadinessFileSummary]


class UploadReadinessSummary(BaseModel):
    organization_id: str
    ready_site_count: int
    site_count: int
    required_categories: list[str]
    sites: list[SiteUploadReadinessSummary]


class RunCreate(BaseModel):
    run_id: str
    site_id: str
    upload_ids: list[str]


class RunSummary(BaseModel):
    id: str
    organization_id: str
    site_id: str
    requested_by_user_id: str
    status: str
    error_message: str | None
    completed_at: str | None


class AnalysisJobSummary(BaseModel):
    id: str
    organization_id: str
    run_id: str
    upload_id: str
    report_id: str
    requested_by_user_id: str
    status: str
    error_message: str | None
    completed_at: str | None


class NotificationSummary(BaseModel):
    id: str
    organization_id: str
    recipient_user_id: str
    channel: str
    event_type: str
    subject: str
    body: str
    resource_type: str
    resource_id: str
    status: str
    error_message: str | None
    sent_at: str | None
    attempt_count: int
    last_attempt_at: str | None
    next_attempt_at: str | None


class ReportSummary(BaseModel):
    id: str
    organization_id: str
    run_id: str
    report_type: str
    is_published: bool
    expires_at: str | None
    retention_applied_at: str | None


class ReportCommentSummary(BaseModel):
    id: str
    organization_id: str
    report_id: str
    author_user_id: str
    body: str
    created_at: str


class ReportCommentCreate(BaseModel):
    body: str


class ReportReviewSummary(BaseModel):
    id: str
    organization_id: str
    report_id: str
    reviewer_user_id: str
    status: ReportReviewStatus
    notes: str | None
    created_at: str


class ReportReviewCreate(BaseModel):
    status: ReportReviewStatus
    notes: str | None = None


class ReportApproverSummary(BaseModel):
    user_id: str
    organization_id: str
    role: Role
    email: str
    display_name: str


class ReportApprovalAssignmentSummary(BaseModel):
    id: str
    organization_id: str
    report_id: str
    assigned_to_user_id: str
    assigned_by_user_id: str
    status: ReportApprovalAssignmentStatus
    due_at: str | None = None
    notes: str | None = None
    completed_at: str | None = None
    created_at: str
    updated_at: str


class ReportApprovalAssignmentCreate(BaseModel):
    assigned_to_user_id: str
    due_at: str | None = None
    notes: str | None = None


class ReportApprovalAssignmentUpdate(BaseModel):
    status: ReportApprovalAssignmentStatus
    notes: str | None = None


class ReportScheduleSummary(BaseModel):
    id: str
    organization_id: str
    report_id: str
    created_by_user_id: str
    frequency: ReportScheduleFrequency
    recipient_email: str
    next_run_at: str
    is_active: bool
    last_delivery_at: str | None = None
    last_delivery_status: ReportScheduleDeliveryStatus | None = None
    last_delivery_error: str | None = None
    created_at: str
    updated_at: str


class ReportScheduleDeliverySummary(BaseModel):
    id: str
    organization_id: str
    schedule_id: str
    report_id: str
    recipient_email: str
    status: ReportScheduleDeliveryStatus
    error_message: str | None = None
    attempted_at: str
    sent_at: str | None = None
    created_at: str


class ReportScheduleCreate(BaseModel):
    frequency: ReportScheduleFrequency
    recipient_email: str
    next_run_at: str


class ReportScheduleUpdate(BaseModel):
    frequency: ReportScheduleFrequency | None = None
    recipient_email: str | None = None
    next_run_at: str | None = None
    is_active: bool | None = None


class ReportRetentionApplyResponse(BaseModel):
    organization_id: str
    evaluated_count: int
    expired_count: int
    deleted_artifact_count: int
    missing_artifact_count: int
    deleted_evidence_package_artifact_count: int
    missing_evidence_package_artifact_count: int
    deleted_evidence_package_metadata_count: int
    retained_report_ids: list[str]


class ReportArtifactResponse(BaseModel):
    report_id: str
    organization_id: str
    content: dict[str, Any]


class ReportEvidencePackageSummary(BaseModel):
    id: str
    organization_id: str
    report_id: str
    created_by_user_id: str
    filename: str
    content_type: str
    content_length: int
    package_type: str
    format: str
    comment_count: int
    review_count: int
    schedule_count: int
    delivery_count: int
    upload_count: int
    created_at: str


class AuditEventSummary(BaseModel):
    id: str
    organization_id: str
    created_at: str
    actor_user_id: str
    action: str
    resource_type: str
    resource_id: str
    metadata_json: str


class AuditExportResponse(BaseModel):
    filename: str
    content_type: str
    event_count: int
    content: str


class AlertDeliveryTestResponse(BaseModel):
    event_id: str
    event_type: str
    delivered_at: str
    status_code: int


class ArtifactStoreTestResponse(BaseModel):
    event_id: str
    event_type: str
    tested_at: str
    bytes_written: int


class DatabaseTestResponse(BaseModel):
    event_id: str
    event_type: str
    tested_at: str


class TenantIsolationTestResponse(BaseModel):
    event_id: str
    event_type: str
    tested_at: str
    checks_passed: int
    synthetic_records: int


class RunLifecycleTestResponse(BaseModel):
    event_id: str
    event_type: str
    tested_at: str
    checks_passed: int
    artifacts_verified: int
    synthetic_records: int


class WorkerRunLifecycleTestResponse(BaseModel):
    event_id: str
    event_type: str
    tested_at: str
    checks_passed: int
    jobs_attempted: int
    jobs_succeeded: int
    artifacts_verified: int
    synthetic_records: int


class PlatformOperationEventSummary(BaseModel):
    id: str
    event_type: str
    status: str
    request_id: str | None
    detail: str
    created_at: str


class SupportWorkerHeartbeatSummary(BaseModel):
    worker_fingerprint: str
    scope: str
    poll_seconds: float
    batch_size: int
    last_attempted: int
    last_succeeded: int
    last_failed: int
    last_skipped: int
    last_scheduled_attempted: int
    last_scheduled_sent: int
    last_scheduled_failed: int
    last_scheduled_skipped: int
    last_connector_attempted: int
    last_connector_succeeded: int
    last_connector_failed: int
    last_connector_skipped: int
    last_seen_at: str
    age_seconds: float | None


class SupportWorkerHeartbeatScopeCount(BaseModel):
    scope: str
    count: int


class SupportWorkerHeartbeatQueueSummary(BaseModel):
    total_count: int
    latest_all_tenant_last_seen_at: str | None
    latest_all_tenant_age_seconds: float | None
    by_scope: list[SupportWorkerHeartbeatScopeCount]


class CspViolationReportSummary(BaseModel):
    accepted: bool
    report_id: str
    disposition: str
    effective_directive: str
    blocked_uri_category: str


class SupportCspViolationReportSummary(BaseModel):
    id: str
    disposition: str
    effective_directive: str
    violated_directive: str
    blocked_uri_category: str
    document_origin: str
    status_code: int | None
    created_at: str


class SupportCspReportDispositionCount(BaseModel):
    disposition: str
    count: int


class SupportCspReportDirectiveCount(BaseModel):
    directive: str
    count: int


class SupportCspReportQueueSummary(BaseModel):
    total_count: int
    latest_report_created_at: str | None
    by_disposition: list[SupportCspReportDispositionCount]
    by_effective_directive: list[SupportCspReportDirectiveCount]


class SupportCspReportRetentionSummary(BaseModel):
    retention_days: int
    cutoff_created_before: str
    deleted_count: int
    remaining_count: int


_INVITE_AUDIT_STATUS_BY_ACTION = {
    AuditAction.INVITE_CREATED.value: "pending",
    AuditAction.INVITE_ACCEPTED.value: "accepted",
    AuditAction.INVITE_REVOKED.value: "revoked",
}
_ARTIFACT_LOCATION_AUDIT_ACTIONS = {
    AuditAction.UPLOAD_STORED.value,
    AuditAction.REPORT_ACCESSED.value,
    AuditAction.REPORT_RETENTION_APPLIED.value,
}
_REPORT_RETENTION_AUDIT_FIELDS = (
    "expires_at",
    "artifact_deleted",
    "evidence_package_count",
    "evidence_package_artifact_deleted_count",
    "evidence_package_artifact_missing_count",
    "evidence_package_metadata_deleted_count",
)


def _sanitized_audit_metadata_json(action: str, metadata_json: str) -> str:
    if action not in _INVITE_AUDIT_STATUS_BY_ACTION and action not in _ARTIFACT_LOCATION_AUDIT_ACTIONS:
        return metadata_json

    try:
        raw_metadata = json.loads(metadata_json or "{}")
    except json.JSONDecodeError:
        raw_metadata = {}
    if not isinstance(raw_metadata, dict):
        raw_metadata = {}

    if action == AuditAction.UPLOAD_STORED.value:
        sanitized_metadata = {
            field: raw_metadata[field]
            for field in ("category", "site_id")
            if isinstance(raw_metadata.get(field), str)
        }
        if "storage_key" in raw_metadata or raw_metadata.get("artifact_location_configured"):
            sanitized_metadata["artifact_location_configured"] = "true"
        return json.dumps(sanitized_metadata, sort_keys=True)

    if action == AuditAction.REPORT_ACCESSED.value:
        sanitized_metadata = {
            "artifact_location_configured": "true",
        }
        report_type = raw_metadata.get("report_type")
        if isinstance(report_type, str):
            sanitized_metadata["report_type"] = report_type
        return json.dumps(sanitized_metadata, sort_keys=True)

    if action == AuditAction.REPORT_RETENTION_APPLIED.value:
        sanitized_metadata = {
            field: raw_metadata[field]
            for field in _REPORT_RETENTION_AUDIT_FIELDS
            if field in raw_metadata
        }
        if "storage_key" in raw_metadata or raw_metadata.get("artifact_location_configured"):
            sanitized_metadata["artifact_location_configured"] = True
        return json.dumps(sanitized_metadata, sort_keys=True)

    sanitized_metadata: dict[str, str] = {
        "invite_email_configured": "true",
        "status": str(raw_metadata.get("status") or _INVITE_AUDIT_STATUS_BY_ACTION[action]),
    }
    role = raw_metadata.get("role")
    if isinstance(role, str):
        sanitized_metadata["role"] = role
    return json.dumps(sanitized_metadata, sort_keys=True)


class SubscriptionPlanSummary(BaseModel):
    id: str
    name: str
    description: str
    monthly_price_cents: int | None
    included_users: int | None
    included_sites: int | None
    monthly_run_limit: int | None
    report_retention_days: int | None
    is_active: bool


class SubscriptionUsageSummary(BaseModel):
    member_count: int
    active_user_count: int
    pending_invite_count: int
    site_count: int
    runs_this_period: int


class OrganizationSubscriptionSummary(BaseModel):
    organization_id: str
    plan: SubscriptionPlanSummary
    status: str
    current_period_start: str | None
    current_period_end: str | None
    trial_ends_at: str | None
    external_customer_id: str | None
    external_subscription_id: str | None
    usage: SubscriptionUsageSummary


class OrganizationSubscriptionUpdate(BaseModel):
    plan_id: str


class BillingCheckoutCreate(BaseModel):
    plan_id: str
    success_url: str | None = None
    cancel_url: str | None = None


class BillingCheckoutSessionSummary(BaseModel):
    organization_id: str
    plan_id: str
    provider: str
    checkout_session_id: str
    checkout_url: str


class BillingPortalCreate(BaseModel):
    return_url: str | None = None


class BillingPortalSessionSummary(BaseModel):
    organization_id: str
    provider: str
    portal_session_id: str
    portal_url: str


class BillingWebhookSummary(BaseModel):
    provider: str
    event_id: str
    event_type: str
    organization_id: str | None
    action: str
    duplicate: bool
    plan_id: str | None
    status: str | None
    external_customer_id: str | None
    external_subscription_id: str | None


class NotificationProviderWebhookSummary(BaseModel):
    provider: str
    event_id: str
    notification_id: str
    status: str
    duplicate: bool
    notification_status: str


class LocalAnalysisRunCreate(BaseModel):
    site_id: str
    upload_id: str
    run_id: str
    report_id: str
    filename: str
    rows: list[dict[str, Any]]
    client_config: dict[str, Any] | None = None


class StoredUploadAnalysisRunCreate(BaseModel):
    upload_id: str
    run_id: str
    report_id: str
    source_upload_ids: list[str] = Field(default_factory=list)
    client_config: dict[str, Any] | None = None


class LocalAnalysisRunResponse(BaseModel):
    upload_id: str
    run_upload_ids: list[str]
    run_id: str
    report_id: str
    run_status: str
    iso_summary: dict[str, Any]


class QueuedAnalysisRunResponse(BaseModel):
    job_id: str
    upload_id: str
    run_upload_ids: list[str]
    run_id: str
    report_id: str
    run_status: str


def health_check() -> dict[str, str]:
    return {"status": "ok"}


def readiness_check(
    store: SaaSStore | None = None,
    artifact_store: TenantArtifactStore | None = None,
) -> dict[str, Any]:
    return readiness_report(store=store, artifact_store=artifact_store)


def user_id_for_organization_route(user: AuthenticatedUser, organization_id: str) -> str:
    if user.auth_mode == CLERK_AUTH_MODE and user.organization_id is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Verified Clerk token must include an active organization context",
        )
    if user.organization_id is not None and user.organization_id != organization_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Authenticated organization context does not match requested organization",
        )
    return user.id


async def process_analysis_job_background(
    store: SaaSStore,
    artifact_store: TenantArtifactStore,
    job_id: str,
    email_sender: EmailSender | None = None,
) -> None:
    job_row = store.get_analysis_job(job_id)
    organization_id = job_row["organization_id"] if job_row is not None else None
    try:
        process_analysis_job(store, job_id, artifact_store)
    finally:
        if email_sender is not None and organization_id is not None:
            process_pending_notifications(
                store,
                email_sender,
                organization_id=organization_id,
            )


def _require_site_in_scope(
    context: TenantContext,
    site_id: str,
    store: SaaSStore,
) -> Site:
    site = store.get_site(site_id)
    if site is None:
        raise ValueError(f"Site {site_id} was not found")
    require_resource_scope(context, site.organization_id, "site")
    return site


def _require_upload_in_scope(
    context: TenantContext,
    upload_id: str,
    site_id: str,
    store: SaaSStore,
) -> None:
    upload = store.get_upload(upload_id)
    if upload is None:
        raise ValueError(f"Upload {upload_id} was not found")
    require_resource_scope(context, upload["organization_id"], "upload")
    if upload["site_id"] != site_id:
        raise ValueError(f"Upload {upload_id} does not belong to site {site_id}")


def create_dev_session(
    payload: DevSessionCreate,
    store: SaaSStore,
) -> DevSessionResponse:
    user_row = store.get_user(payload.user_id)
    if user_row is None:
        raise ValueError(f"User {payload.user_id} was not found")
    if not bool(user_row["is_active"]):
        raise ValueError(f"User {payload.user_id} is inactive")

    role: str | None = None
    claims: dict[str, str] = {"sub": payload.user_id}
    if payload.organization_id is not None:
        memberships = store.list_memberships(
            user_id=payload.user_id,
            organization_id=payload.organization_id,
        )
        if not memberships:
            raise ValueError(
                f"User {payload.user_id} has no membership in organization {payload.organization_id}"
            )
        role = memberships[0].role.value
        claims["org_id"] = payload.organization_id
        claims["org_role"] = role

    return DevSessionResponse(
        user_id=payload.user_id,
        organization_id=payload.organization_id,
        role=role,
        auth_token=dev_token_from_claims(claims),
    )


def sync_clerk_session(
    user: AuthenticatedUser,
    store: SaaSStore,
) -> ClerkSyncResponse:
    if user.auth_mode != CLERK_AUTH_MODE:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Clerk tenant sync requires a verified Clerk session",
        )
    if user.organization_id is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Verified Clerk token must include an active organization context",
        )

    try:
        result = sync_clerk_authenticated_user(store, user)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return ClerkSyncResponse(
        user_id=result.user.id,
        organization_id=result.organization.id,
        organization_slug=result.organization.slug,
        role=result.membership.role.value,
    )


def process_clerk_webhook(
    event: dict[str, Any],
    store: SaaSStore,
) -> ClerkWebhookResponse:
    result = process_clerk_webhook_event(event, store)
    return ClerkWebhookResponse(
        event_type=result.event_type,
        action=result.action,
        user_id=result.user_id,
        organization_id=result.organization_id,
        role=result.role,
    )


def onboard_owner_organization(
    payload: OwnerOrganizationCreate,
    store: SaaSStore,
) -> OwnerOrganizationResponse:
    result = create_owner_organization(
        store,
        user=User(
            id=payload.user_id,
            email=payload.email,
            display_name=payload.display_name,
        ),
        organization=Organization(
            id=payload.organization_id,
            name=payload.organization_name,
            slug=payload.organization_slug,
        ),
    )
    return OwnerOrganizationResponse(
        user_id=result.user.id,
        organization_id=result.organization.id,
        role=result.membership.role.value,
    )


def list_user_organization_summaries(
    user_id: str,
    store: SaaSStore,
) -> list[OrganizationSummary]:
    return [
        OrganizationSummary(id=organization.id, name=organization.name, slug=organization.slug)
        for organization in store.list_user_organizations(user_id)
    ]


def get_organization_summary(
    user_id: str,
    organization_id: str,
    store: SaaSStore,
) -> OrganizationSummary:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.READ,
    )
    organization = store.get_organization(organization_id)
    if organization is None:
        raise ValueError(f"Organization {organization_id} was not found")
    return OrganizationSummary(
        id=organization["id"],
        name=organization["name"],
        slug=organization["slug"],
    )


def _normalize_optional_text(value: str | None, max_length: int, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.strip().split())
    if not normalized:
        return None
    if len(normalized) > max_length:
        raise ValueError(f"{field_name} cannot be longer than {max_length} characters")
    return normalized


def _normalize_required_text(value: str, max_length: int, field_name: str) -> str:
    normalized = _normalize_optional_text(value, max_length, field_name)
    if normalized is None:
        raise ValueError(f"{field_name} is required")
    return normalized


def _normalize_contact_email(value: str, field_name: str) -> str:
    email = _normalize_required_text(value, 254, field_name).lower()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise ValueError(f"{field_name} must be a valid email address")
    return email


def _marketing_inquiry_rate_limit_source(request: Request) -> str:
    client_host = request.client.host if request.client is not None else "unknown"
    user_agent = (request.headers.get("user-agent") or "unknown")[:300]
    return f"{client_host}|{user_agent}"


def _marketing_inquiry_rate_limit_context(
    request: Request,
) -> tuple[str | None, str | None, int | None, int | None]:
    policy = marketing_inquiry_rate_limit_policy_from_env(request.app.state.runtime_env)
    if not policy.enabled:
        return None, None, None, None
    source_key_hash = marketing_inquiry_rate_limit_key(
        _marketing_inquiry_rate_limit_source(request),
        policy.secret,
    )
    since_iso = (
        datetime.now(timezone.utc) - timedelta(minutes=policy.window_minutes)
    ).isoformat()
    return source_key_hash, since_iso, policy.max_submissions, policy.window_minutes * 60


def _marketing_inquiry_handoff_message(
    inquiry: MarketingInquiry,
    recipient_email: str,
) -> EmailMessage:
    use_case = inquiry.intended_use.replace("_", " ")
    body_lines = [
        "A new RDS Energy Analytics pilot inquiry was submitted.",
        "",
        f"Inquiry: {inquiry.id}",
        f"Name: {inquiry.name}",
        f"Email: {inquiry.email}",
    ]
    if inquiry.organization_name is not None:
        body_lines.append(f"Organization: {inquiry.organization_name}")
    if inquiry.role is not None:
        body_lines.append(f"Role: {inquiry.role}")
    body_lines.extend(
        [
            f"Use case: {use_case}",
            f"Source page: {inquiry.source_page}",
            "",
            "Message:",
            inquiry.message,
        ]
    )
    return EmailMessage(
        notification_id=f"{inquiry.id}:marketing_handoff",
        organization_id=MARKETING_INQUIRY_NOTIFICATION_ORGANIZATION_ID,
        recipient_user_id=MARKETING_INQUIRY_NOTIFICATION_RECIPIENT_USER_ID,
        to_email=recipient_email,
        subject=f"RDS pilot inquiry: {use_case}",
        body="\n".join(body_lines),
    )


def _send_marketing_inquiry_handoff(
    inquiry: MarketingInquiry,
    email_sender: EmailSender | None,
    recipient_email: str | None,
    crm_sender: MarketingInquiryCrmSender | None = None,
) -> None:
    if email_sender is None or recipient_email is None:
        pass
    else:
        try:
            email_sender.send(_marketing_inquiry_handoff_message(inquiry, recipient_email))
        except Exception:
            pass

    if crm_sender is not None:
        try:
            crm_sender.send(inquiry)
        except Exception:
            pass


def create_marketing_inquiry_summary(
    payload: MarketingInquiryCreate,
    store: SaaSStore,
    email_sender: EmailSender | None = None,
    handoff_recipient_email: str | None = None,
    crm_handoff_sender: MarketingInquiryCrmSender | None = None,
    rate_limit_source_key_hash: str | None = None,
    rate_limit_since_iso: str | None = None,
    rate_limit_max_submissions: int | None = None,
    rate_limit_retry_after_seconds: int | None = None,
) -> MarketingInquirySummary:
    if payload.website is not None and payload.website.strip():
        raise ValueError("Marketing inquiry could not be accepted")

    intended_use = payload.intended_use.strip().lower().replace("-", "_")
    if intended_use not in MARKETING_INQUIRY_USE_CASES:
        raise ValueError(
            "Marketing inquiry intended use must be one of: "
            f"{', '.join(sorted(MARKETING_INQUIRY_USE_CASES))}"
        )
    source_page = (
        _normalize_optional_text(payload.source_page, 120, "Marketing inquiry source page")
        or MARKETING_INQUIRY_DEFAULT_SOURCE
    )
    if not re.fullmatch(r"[a-zA-Z0-9_./?#=&:-]+", source_page):
        raise ValueError("Marketing inquiry source page contains unsupported characters")

    email = _normalize_contact_email(payload.email, "Marketing inquiry email")
    duplicate_since = (
        datetime.now(timezone.utc) - timedelta(minutes=MARKETING_INQUIRY_DUPLICATE_WINDOW_MINUTES)
    ).isoformat()
    if store.count_recent_marketing_inquiries_for_email(email, duplicate_since) > 0:
        raise ValueError("A pilot request for this email was submitted recently")

    inquiry = MarketingInquiry(
        id=f"marketing_inquiry_{uuid4().hex}",
        email=email,
        name=_normalize_required_text(payload.name, 120, "Marketing inquiry name"),
        organization_name=_normalize_optional_text(
            payload.organization_name,
            160,
            "Marketing inquiry organization name",
        ),
        role=_normalize_optional_text(payload.role, 120, "Marketing inquiry role"),
        intended_use=intended_use,
        message=_normalize_required_text(payload.message, 1200, "Marketing inquiry message"),
        source_page=source_page,
    )
    rate_limit_exceeded = False
    with store.conn:
        if rate_limit_source_key_hash is not None:
            if rate_limit_since_iso is None or rate_limit_max_submissions is None:
                raise MarketingInquiryRateLimitConfigurationError(
                    "Marketing inquiry rate limit context is incomplete"
                )
            recent_source_count = store.count_recent_marketing_inquiry_submissions_for_source(
                rate_limit_source_key_hash,
                rate_limit_since_iso,
            )
            if recent_source_count >= rate_limit_max_submissions:
                store.create_marketing_inquiry_submission_event(
                    event_id=f"marketing_inquiry_submission_{uuid4().hex}",
                    source_key_hash=rate_limit_source_key_hash,
                    inquiry_id=None,
                    outcome="rate_limited",
                    created_at_iso=datetime.now(timezone.utc).isoformat(),
                )
                rate_limit_exceeded = True
        if rate_limit_exceeded:
            pass
        else:
            store.create_marketing_inquiry(inquiry)
            if rate_limit_source_key_hash is not None:
                store.create_marketing_inquiry_submission_event(
                    event_id=f"marketing_inquiry_submission_{uuid4().hex}",
                    source_key_hash=rate_limit_source_key_hash,
                    inquiry_id=inquiry.id,
                    outcome="accepted",
                    created_at_iso=inquiry.created_at.isoformat(),
                )
            store.create_marketing_inquiry_followup_event(
                event_id=f"marketing_inquiry_event_{uuid4().hex}",
                inquiry_id=inquiry.id,
                actor_label="public_form",
                action="created",
                status=inquiry.status,
                assigned_to=None,
                changed_fields=[
                    "email",
                    "name",
                    "organization_name",
                    "role",
                    "intended_use",
                    "message",
                    "source_page",
                    "status",
                ],
                notes_changed=False,
                created_at_iso=inquiry.created_at.isoformat(),
            )
    if rate_limit_exceeded:
        raise MarketingInquiryRateLimitExceeded(
            "Marketing inquiry could not be accepted right now",
            retry_after_seconds=rate_limit_retry_after_seconds or 60,
        )
    _send_marketing_inquiry_handoff(
        inquiry,
        email_sender,
        handoff_recipient_email,
        crm_handoff_sender,
    )
    return MarketingInquirySummary(
        id=inquiry.id,
        status=inquiry.status,
        created_at=inquiry.created_at.isoformat(),
    )


def _marketing_inquiry_detail_from_row(row: Any) -> MarketingInquiryDetail:
    return MarketingInquiryDetail(
        id=row["id"],
        email=row["email"],
        name=row["name"],
        organization_name=row["organization_name"],
        role=row["role"],
        intended_use=row["intended_use"],
        message=row["message"],
        source_page=row["source_page"],
        status=MarketingInquiryStatus(row["status"]),
        created_at=row["created_at"],
        assigned_to=row["assigned_to"],
        follow_up_notes=row["follow_up_notes"],
        updated_at=row["updated_at"],
    )


def _marketing_inquiry_activity_from_row(row: Any) -> MarketingInquiryActivitySummary:
    try:
        changed_fields = json.loads(row["changed_fields_json"])
    except (TypeError, json.JSONDecodeError):
        changed_fields = []
    if not isinstance(changed_fields, list):
        changed_fields = []
    safe_changed_fields = [field for field in changed_fields if isinstance(field, str)]
    return MarketingInquiryActivitySummary(
        id=row["id"],
        inquiry_id=row["inquiry_id"],
        actor_label=row["actor_label"],
        action=row["action"],
        status=MarketingInquiryStatus(row["status"]) if row["status"] is not None else None,
        assigned_to=row["assigned_to"],
        changed_fields=safe_changed_fields,
        notes_changed=bool(row["notes_changed"]),
        created_at=row["created_at"],
    )


def _marketing_inquiry_status_counts_from_dict(
    counts: dict[str, int],
) -> list[MarketingInquiryStatusCount]:
    return [
        MarketingInquiryStatusCount(status=status, count=counts.get(status.value, 0))
        for status in MarketingInquiryStatus
    ]


def create_marketing_inquiry_queue_summary(store: SaaSStore) -> MarketingInquiryQueueSummary:
    status_counts = store.count_marketing_inquiries_by_status()
    assignee_rows = store.count_marketing_inquiries_by_assignee_status()
    grouped: dict[str | None, dict[str, int]] = {}
    for row in assignee_rows:
        assignee = row["assigned_to"]
        grouped.setdefault(assignee, {})[row["status"]] = int(row["total"])

    assignee_summaries = [
        MarketingInquiryAssigneeQueueSummary(
            assigned_to=assignee,
            total_count=sum(counts.values()),
            by_status=_marketing_inquiry_status_counts_from_dict(counts),
        )
        for assignee, counts in sorted(
            grouped.items(),
            key=lambda item: (item[0] is not None, item[0] or ""),
        )
    ]
    return MarketingInquiryQueueSummary(
        total_count=sum(status_counts.values()),
        unassigned_count=sum(grouped.get(None, {}).values()),
        by_status=_marketing_inquiry_status_counts_from_dict(status_counts),
        by_assignee=assignee_summaries,
    )


def _normalize_marketing_inquiry_assignee(value: str | None) -> str | None:
    if value is not None and ("\n" in value or "\r" in value):
        raise ValueError("Marketing inquiry assignee cannot contain line breaks")
    assigned_to = _normalize_optional_text(value, 160, "Marketing inquiry assignee")
    if assigned_to is not None and "@" in assigned_to:
        return assigned_to.lower()
    return assigned_to


def _normalize_support_actor(value: str | None) -> str:
    if value is not None and ("\n" in value or "\r" in value):
        raise ValueError("Support actor cannot contain line breaks")
    return _normalize_optional_text(value, 160, "Support actor") or "support"


def _normalize_report_brand_color(value: str) -> str:
    normalized = value.strip().lower()
    if not re.fullmatch(r"#[0-9a-f]{6}", normalized):
        raise ValueError("Report branding primary color must be a hex color like #0f766e")
    return normalized


def _normalize_report_branding_update(
    organization_id: str,
    payload: OrganizationReportBrandingUpdate,
) -> OrganizationReportBranding:
    title = _normalize_optional_text(
        payload.evidence_pack_title,
        120,
        "Report branding evidence pack title",
    )
    if title is None:
        raise ValueError("Report branding evidence pack title is required")

    logo_url = _normalize_optional_text(payload.logo_url, 500, "Report branding logo URL")
    if logo_url is not None and not (
        logo_url.startswith("https://") or logo_url.startswith("http://")
    ):
        raise ValueError("Report branding logo URL must start with http:// or https://")

    support_email = _normalize_optional_text(
        payload.support_email,
        254,
        "Report branding support email",
    )
    if support_email is not None and "@" not in support_email:
        raise ValueError("Report branding support email must be an email address")

    return OrganizationReportBranding(
        organization_id=organization_id,
        evidence_pack_title=title,
        primary_color=_normalize_report_brand_color(payload.primary_color),
        logo_url=logo_url,
        support_email=support_email.lower() if support_email is not None else None,
    )


def create_report_branding_summary(
    organization_id: str,
    row: Any | None,
) -> OrganizationReportBrandingSummary:
    if row is None:
        return OrganizationReportBrandingSummary(
            organization_id=organization_id,
            evidence_pack_title=DEFAULT_EVIDENCE_PACK_TITLE,
            primary_color=DEFAULT_EVIDENCE_PACK_PRIMARY_COLOR,
            logo_url=None,
            support_email=None,
            updated_at=None,
        )
    return OrganizationReportBrandingSummary(
        organization_id=row["organization_id"],
        evidence_pack_title=row["evidence_pack_title"],
        primary_color=row["primary_color"],
        logo_url=row["logo_url"],
        support_email=row["support_email"],
        updated_at=row["updated_at"],
    )


def get_report_branding_summary(
    user_id: str,
    organization_id: str,
    store: SaaSStore,
) -> OrganizationReportBrandingSummary:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.READ,
    )
    if store.get_organization(organization_id) is None:
        raise ValueError(f"Organization {organization_id} was not found")
    return create_report_branding_summary(
        organization_id,
        store.get_organization_report_branding(organization_id),
    )


def update_report_branding_summary(
    user_id: str,
    organization_id: str,
    payload: OrganizationReportBrandingUpdate,
    store: SaaSStore,
) -> OrganizationReportBrandingSummary:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.MANAGE_ORGANIZATION,
    )
    if store.get_organization(organization_id) is None:
        raise ValueError(f"Organization {organization_id} was not found")

    branding = _normalize_report_branding_update(organization_id, payload)
    audit_event = AuditEvent(
        id=f"audit_report_branding_{uuid4().hex}",
        organization_id=organization_id,
        actor_user_id=user_id,
        action=AuditAction.REPORT_BRANDING_UPDATED,
        resource_type="report_branding",
        resource_id=organization_id,
        metadata={
            "evidence_pack_title_configured": (
                branding.evidence_pack_title != DEFAULT_EVIDENCE_PACK_TITLE
            ),
            "primary_color": branding.primary_color,
            "logo_url_configured": branding.logo_url is not None,
            "support_email_configured": branding.support_email is not None,
        },
    )
    with store.conn:
        store.upsert_organization_report_branding(branding)
        store.create_audit_event(audit_event)

    return create_report_branding_summary(
        organization_id,
        store.get_organization_report_branding(organization_id),
    )


def create_report_approval_policy_summary(
    organization_id: str,
    row: Any | None,
) -> OrganizationReportApprovalPolicySummary:
    if row is None:
        return OrganizationReportApprovalPolicySummary(
            organization_id=organization_id,
            auto_lock_evidence_on_final_approval=DEFAULT_AUTO_LOCK_EVIDENCE_ON_FINAL_APPROVAL,
            minimum_approval_assignments_for_lock=(
                DEFAULT_MINIMUM_APPROVAL_ASSIGNMENTS_FOR_LOCK
            ),
            updated_by_user_id=None,
            updated_at=None,
        )
    return OrganizationReportApprovalPolicySummary(
        organization_id=row["organization_id"],
        auto_lock_evidence_on_final_approval=bool(
            row["auto_lock_evidence_on_final_approval"]
        ),
        minimum_approval_assignments_for_lock=int(
            row["minimum_approval_assignments_for_lock"]
        ),
        updated_by_user_id=row["updated_by_user_id"],
        updated_at=row["updated_at"],
    )


def _report_approval_policy_from_store(
    organization_id: str,
    store: SaaSStore,
) -> OrganizationReportApprovalPolicySummary:
    return create_report_approval_policy_summary(
        organization_id,
        store.get_organization_report_approval_policy(organization_id),
    )


def get_report_approval_policy_summary(
    user_id: str,
    organization_id: str,
    store: SaaSStore,
) -> OrganizationReportApprovalPolicySummary:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.READ,
    )
    if store.get_organization(organization_id) is None:
        raise ValueError(f"Organization {organization_id} was not found")
    return _report_approval_policy_from_store(organization_id, store)


def update_report_approval_policy_summary(
    user_id: str,
    organization_id: str,
    payload: OrganizationReportApprovalPolicyUpdate,
    store: SaaSStore,
) -> OrganizationReportApprovalPolicySummary:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.MANAGE_ORGANIZATION,
    )
    if store.get_organization(organization_id) is None:
        raise ValueError(f"Organization {organization_id} was not found")

    policy = OrganizationReportApprovalPolicy(
        organization_id=organization_id,
        auto_lock_evidence_on_final_approval=payload.auto_lock_evidence_on_final_approval,
        minimum_approval_assignments_for_lock=payload.minimum_approval_assignments_for_lock,
        updated_by_user_id=user_id,
    )
    audit_event = AuditEvent(
        id=f"audit_report_approval_policy_{uuid4().hex}",
        organization_id=organization_id,
        actor_user_id=user_id,
        action=AuditAction.REPORT_APPROVAL_POLICY_UPDATED,
        resource_type="report_approval_policy",
        resource_id=organization_id,
        metadata={
            "auto_lock_evidence_on_final_approval": (
                policy.auto_lock_evidence_on_final_approval
            ),
            "minimum_approval_assignments_for_lock": (
                policy.minimum_approval_assignments_for_lock
            ),
        },
    )
    with store.conn:
        store.upsert_organization_report_approval_policy(policy)
        store.create_audit_event(audit_event)

    return _report_approval_policy_from_store(organization_id, store)


def _normalize_evidence_package_section_labels(
    labels: dict[str, str] | None,
) -> dict[str, str]:
    raw_labels = labels or {}
    unknown_keys = sorted(set(raw_labels) - set(EVIDENCE_PACKAGE_SECTION_KEYS))
    if unknown_keys:
        raise ValueError("Evidence package section labels include unknown sections")

    normalized_labels: dict[str, str] = {}
    for section_key, default_label in DEFAULT_EVIDENCE_PACKAGE_SECTION_LABELS.items():
        raw_label = raw_labels.get(section_key, default_label)
        normalized = _normalize_optional_text(
            str(raw_label),
            80,
            "Evidence package section label",
        )
        normalized_labels[section_key] = normalized or default_label
    return normalized_labels


def _normalize_evidence_package_section_order(order: list[str] | None) -> list[str]:
    raw_order = order or list(DEFAULT_EVIDENCE_PACKAGE_SECTION_ORDER)
    normalized_order: list[str] = []
    for raw_section_key in raw_order:
        section_key = str(raw_section_key).strip()
        if section_key not in EVIDENCE_PACKAGE_SECTION_KEYS:
            raise ValueError("Evidence package section order includes an unknown section")
        if section_key not in normalized_order:
            normalized_order.append(section_key)
    for section_key in DEFAULT_EVIDENCE_PACKAGE_SECTION_ORDER:
        if section_key not in normalized_order:
            normalized_order.append(section_key)
    return normalized_order


def _evidence_package_section_labels_from_json(value: Any) -> dict[str, str]:
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return _normalize_evidence_package_section_labels(
                {str(key): str(label) for key, label in parsed.items()}
            )
    return dict(DEFAULT_EVIDENCE_PACKAGE_SECTION_LABELS)


def _evidence_package_section_order_from_json(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return _normalize_evidence_package_section_order(
                [str(section_key) for section_key in parsed]
            )
    return list(DEFAULT_EVIDENCE_PACKAGE_SECTION_ORDER)


def create_evidence_package_template_summary(
    organization_id: str,
    row: Any | None,
) -> OrganizationEvidencePackageTemplateSummary:
    if row is None:
        return OrganizationEvidencePackageTemplateSummary(
            organization_id=organization_id,
            template_name=DEFAULT_EVIDENCE_PACKAGE_TEMPLATE_NAME,
            include_report_artifact=True,
            include_uploads=True,
            include_approval_assignments=True,
            include_reviews=True,
            include_comments=True,
            include_schedules=True,
            section_labels=dict(DEFAULT_EVIDENCE_PACKAGE_SECTION_LABELS),
            section_order=list(DEFAULT_EVIDENCE_PACKAGE_SECTION_ORDER),
            updated_by_user_id=None,
            updated_at=None,
        )
    return OrganizationEvidencePackageTemplateSummary(
        organization_id=row["organization_id"],
        template_name=row["template_name"],
        include_report_artifact=bool(row["include_report_artifact"]),
        include_uploads=bool(row["include_uploads"]),
        include_approval_assignments=bool(row["include_approval_assignments"]),
        include_reviews=bool(row["include_reviews"]),
        include_comments=bool(row["include_comments"]),
        include_schedules=bool(row["include_schedules"]),
        section_labels=_evidence_package_section_labels_from_json(row["section_labels_json"]),
        section_order=_evidence_package_section_order_from_json(row["section_order_json"]),
        updated_by_user_id=row["updated_by_user_id"],
        updated_at=row["updated_at"],
    )


def _evidence_package_template_from_store(
    organization_id: str,
    store: SaaSStore,
) -> OrganizationEvidencePackageTemplateSummary:
    return create_evidence_package_template_summary(
        organization_id,
        store.get_organization_evidence_package_template(organization_id),
    )


def _evidence_package_template_enabled_sections(
    template: OrganizationEvidencePackageTemplateSummary,
) -> int:
    return sum(
        int(enabled)
        for enabled in (
            template.include_report_artifact,
            template.include_uploads,
            template.include_approval_assignments,
            template.include_reviews,
            template.include_comments,
            template.include_schedules,
        )
    )


def _evidence_package_template_evidence(
    template: OrganizationEvidencePackageTemplateSummary,
) -> dict[str, Any]:
    return {
        "template_name": template.template_name,
        "include_report_artifact": template.include_report_artifact,
        "include_uploads": template.include_uploads,
        "include_approval_assignments": template.include_approval_assignments,
        "include_reviews": template.include_reviews,
        "include_comments": template.include_comments,
        "include_schedules": template.include_schedules,
        "section_labels": dict(template.section_labels),
        "section_order": list(template.section_order),
    }


def get_evidence_package_template_summary(
    user_id: str,
    organization_id: str,
    store: SaaSStore,
) -> OrganizationEvidencePackageTemplateSummary:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.READ,
    )
    if store.get_organization(organization_id) is None:
        raise ValueError(f"Organization {organization_id} was not found")
    return _evidence_package_template_from_store(organization_id, store)


def update_evidence_package_template_summary(
    user_id: str,
    organization_id: str,
    payload: OrganizationEvidencePackageTemplateUpdate,
    store: SaaSStore,
) -> OrganizationEvidencePackageTemplateSummary:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.MANAGE_ORGANIZATION,
    )
    if store.get_organization(organization_id) is None:
        raise ValueError(f"Organization {organization_id} was not found")

    template_name = _normalize_optional_text(
        payload.template_name,
        120,
        "Evidence package template name",
    )
    if template_name is None:
        raise ValueError("Evidence package template name is required")
    section_labels = _normalize_evidence_package_section_labels(payload.section_labels)
    section_order = _normalize_evidence_package_section_order(payload.section_order)

    template = OrganizationEvidencePackageTemplate(
        organization_id=organization_id,
        template_name=template_name,
        include_report_artifact=payload.include_report_artifact,
        include_uploads=payload.include_uploads,
        include_approval_assignments=payload.include_approval_assignments,
        include_reviews=payload.include_reviews,
        include_comments=payload.include_comments,
        include_schedules=payload.include_schedules,
        section_labels_json=json.dumps(section_labels, sort_keys=True),
        section_order_json=json.dumps(section_order),
        updated_by_user_id=user_id,
    )
    enabled_sections = sum(
        int(enabled)
        for enabled in (
            template.include_report_artifact,
            template.include_uploads,
            template.include_approval_assignments,
            template.include_reviews,
            template.include_comments,
            template.include_schedules,
        )
    )
    audit_event = AuditEvent(
        id=f"audit_report_evidence_template_{uuid4().hex}",
        organization_id=organization_id,
        actor_user_id=user_id,
        action=AuditAction.REPORT_EVIDENCE_TEMPLATE_UPDATED,
        resource_type="report_evidence_template",
        resource_id=organization_id,
        metadata={
            "enabled_section_count": enabled_sections,
            "include_approval_assignments": template.include_approval_assignments,
            "include_comments": template.include_comments,
            "include_report_artifact": template.include_report_artifact,
            "include_reviews": template.include_reviews,
            "include_schedules": template.include_schedules,
            "include_uploads": template.include_uploads,
            "section_labels_customized": (
                section_labels != DEFAULT_EVIDENCE_PACKAGE_SECTION_LABELS
            ),
            "section_order_customized": (
                section_order != list(DEFAULT_EVIDENCE_PACKAGE_SECTION_ORDER)
            ),
            "template_name_configured": (
                template.template_name != DEFAULT_EVIDENCE_PACKAGE_TEMPLATE_NAME
            ),
        },
    )
    with store.conn:
        store.upsert_organization_evidence_package_template(template)
        store.create_audit_event(audit_event)

    return _evidence_package_template_from_store(organization_id, store)


def create_subscription_plan_summary(plan: SubscriptionPlan) -> SubscriptionPlanSummary:
    return SubscriptionPlanSummary(
        id=plan.id,
        name=plan.name,
        description=plan.description,
        monthly_price_cents=plan.monthly_price_cents,
        included_users=plan.included_users,
        included_sites=plan.included_sites,
        monthly_run_limit=plan.monthly_run_limit,
        report_retention_days=plan.report_retention_days,
        is_active=plan.is_active,
    )


def list_subscription_plan_summaries(store: SaaSStore) -> list[SubscriptionPlanSummary]:
    return [
        create_subscription_plan_summary(plan)
        for plan in list_available_subscription_plans(store)
    ]


def create_subscription_summary(
    subscription: TenantSubscription,
) -> OrganizationSubscriptionSummary:
    return OrganizationSubscriptionSummary(
        organization_id=subscription.organization_id,
        plan=create_subscription_plan_summary(subscription.plan),
        status=subscription.status.value,
        current_period_start=subscription.current_period_start,
        current_period_end=subscription.current_period_end,
        trial_ends_at=subscription.trial_ends_at,
        external_customer_id=subscription.external_customer_id,
        external_subscription_id=subscription.external_subscription_id,
        usage=SubscriptionUsageSummary(
            member_count=subscription.usage.member_count,
            active_user_count=subscription.usage.active_user_count,
            pending_invite_count=subscription.usage.pending_invite_count,
            site_count=subscription.usage.site_count,
            runs_this_period=subscription.usage.runs_this_period,
        ),
    )


def get_subscription_summary(
    user_id: str,
    organization_id: str,
    store: SaaSStore,
) -> OrganizationSubscriptionSummary:
    return create_subscription_summary(
        get_tenant_subscription(
            store,
            user_id=user_id,
            organization_id=organization_id,
        )
    )


def update_subscription_summary(
    user_id: str,
    organization_id: str,
    payload: OrganizationSubscriptionUpdate,
    store: SaaSStore,
) -> OrganizationSubscriptionSummary:
    return create_subscription_summary(
        update_tenant_subscription_plan(
            store,
            user_id=user_id,
            organization_id=organization_id,
            plan_id=payload.plan_id,
        )
    )


def create_billing_checkout_summary(
    user_id: str,
    organization_id: str,
    payload: BillingCheckoutCreate,
    store: SaaSStore,
) -> BillingCheckoutSessionSummary:
    session = create_billing_checkout_session(
        store,
        user_id,
        organization_id,
        payload.plan_id,
        success_url=payload.success_url,
        cancel_url=payload.cancel_url,
    )
    return BillingCheckoutSessionSummary(
        organization_id=session.organization_id,
        plan_id=session.plan_id,
        provider=session.provider,
        checkout_session_id=session.checkout_session_id,
        checkout_url=session.checkout_url,
    )


def create_billing_portal_summary(
    user_id: str,
    organization_id: str,
    payload: BillingPortalCreate,
    store: SaaSStore,
) -> BillingPortalSessionSummary:
    session = create_billing_portal_session(
        store,
        user_id,
        organization_id,
        return_url=payload.return_url,
    )
    return BillingPortalSessionSummary(
        organization_id=session.organization_id,
        provider=session.provider,
        portal_session_id=session.portal_session_id,
        portal_url=session.portal_url,
    )


def create_billing_webhook_summary(result: BillingWebhookResult) -> BillingWebhookSummary:
    return BillingWebhookSummary(
        provider=result.provider,
        event_id=result.event_id,
        event_type=result.event_type,
        organization_id=result.organization_id,
        action=result.action,
        duplicate=result.duplicate,
        plan_id=result.plan_id,
        status=result.status,
        external_customer_id=result.external_customer_id,
        external_subscription_id=result.external_subscription_id,
    )


def create_notification_provider_webhook_summary(
    result: NotificationProviderWebhookResult,
) -> NotificationProviderWebhookSummary:
    return NotificationProviderWebhookSummary(
        provider=result.provider,
        event_id=result.provider_event_id,
        notification_id=result.notification_id,
        status=result.status.value,
        duplicate=result.duplicate,
        notification_status=result.notification_status.value,
    )


def create_csp_violation_report_summary(
    report: CspViolationReport,
) -> CspViolationReportSummary:
    return CspViolationReportSummary(
        accepted=True,
        report_id=report.id,
        disposition=report.disposition.value,
        effective_directive=report.effective_directive.value,
        blocked_uri_category=report.blocked_uri_category.value,
    )


def create_analysis_config_summary(config: TenantAnalysisConfig) -> AnalysisConfigSummary:
    return AnalysisConfigSummary(
        organization_id=config.organization_id,
        config=config.config,
        source=config.source,
        updated_by_user_id=config.updated_by_user_id,
        updated_at=config.updated_at,
    )


def get_analysis_config_summary(
    user_id: str,
    organization_id: str,
    store: SaaSStore,
    site_id: str | None = None,
) -> AnalysisConfigSummary:
    return create_analysis_config_summary(
        get_tenant_analysis_config(
            store,
            user_id=user_id,
            organization_id=organization_id,
            site_id=site_id,
        )
    )


def update_analysis_config_summary(
    user_id: str,
    organization_id: str,
    payload: AnalysisConfigUpdate,
    store: SaaSStore,
) -> AnalysisConfigSummary:
    return create_analysis_config_summary(
        update_tenant_analysis_config(
            store,
            user_id=user_id,
            organization_id=organization_id,
            config=payload.config,
        )
    )


def _parse_optional_connector_datetime(
    value: str | None,
    field_name: str = "Data connector next sync time",
) -> datetime | None:
    if value is None:
        return None
    raw_value = value.strip()
    if not raw_value:
        return None
    try:
        parsed = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be ISO-8601") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _add_connector_months(value: datetime, months: int) -> datetime:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def _next_data_connector_sync_at(
    current: datetime,
    schedule: DataConnectorSchedule,
) -> datetime | None:
    if schedule == DataConnectorSchedule.MANUAL:
        return None
    if schedule == DataConnectorSchedule.DAILY:
        return current + timedelta(days=1)
    if schedule == DataConnectorSchedule.WEEKLY:
        return current + timedelta(days=7)
    if schedule == DataConnectorSchedule.MONTHLY:
        return _add_connector_months(current, 1)
    raise ValueError(f"Unsupported data connector schedule: {schedule}")


def create_data_connector_summary_from_row(row: Any) -> DataConnectorSummary:
    return DataConnectorSummary(
        id=row["id"],
        organization_id=row["organization_id"],
        site_id=row["site_id"],
        name=row["name"],
        connector_type=DataConnectorType(row["connector_type"]),
        source_category=row["source_category"],
        status=DataConnectorStatus(row["status"]),
        schedule=DataConnectorSchedule(row["schedule"]),
        external_reference=row["external_reference"],
        credential_reference_configured=bool(row["credential_reference"]),
        created_by_user_id=row["created_by_user_id"],
        updated_by_user_id=row["updated_by_user_id"],
        last_sync_at=row["last_sync_at"],
        last_sync_status=(
            DataConnectorSyncStatus(row["last_sync_status"])
            if row["last_sync_status"]
            else None
        ),
        last_sync_error=row["last_sync_error"],
        next_sync_at=row["next_sync_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _data_connector_audit_metadata(connector: DataConnector) -> dict[str, Any]:
    return {
        "connector_type": connector.connector_type.value,
        "credential_reference_configured": connector.credential_reference is not None,
        "external_reference_configured": connector.external_reference is not None,
        "next_sync_configured": connector.next_sync_at is not None,
        "schedule": connector.schedule.value,
        "source_category": connector.source_category,
        "status": connector.status.value,
    }


def create_data_connector_sync_summary_from_row(row: Any) -> DataConnectorSyncSummary:
    return DataConnectorSyncSummary(
        id=row["id"],
        organization_id=row["organization_id"],
        connector_id=row["connector_id"],
        site_id=row["site_id"],
        source_category=row["source_category"],
        status=DataConnectorSyncStatus(row["status"]),
        upload_id=row["upload_id"],
        records_processed=int(row["records_processed"]),
        status_reason=row["status_reason"],
        triggered_by_user_id=row["triggered_by_user_id"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        created_at=row["created_at"],
    )


def _data_connector_sync_audit_metadata(
    connector: Any,
    sync: DataConnectorSync,
) -> dict[str, Any]:
    return {
        "connector_type": connector["connector_type"],
        "records_processed": sync.records_processed,
        "schedule": connector["schedule"],
        "source_category": sync.source_category,
        "status": sync.status.value,
        "status_reason_configured": sync.status_reason is not None,
        "upload_configured": sync.upload_id is not None,
    }


def list_data_connector_summaries(
    user_id: str,
    organization_id: str,
    store: SaaSStore,
    site_id: str | None = None,
) -> list[DataConnectorSummary]:
    context = require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.READ,
    )
    if site_id is not None:
        site = store.get_site(site_id)
        if site is None:
            raise ValueError(f"Site {site_id} was not found")
        require_resource_scope(context, site.organization_id, "site")
    return [
        create_data_connector_summary_from_row(connector)
        for connector in store.list_data_connectors(organization_id, site_id=site_id)
    ]


def create_data_connector_summary(
    user_id: str,
    organization_id: str,
    payload: DataConnectorCreate,
    store: SaaSStore,
) -> DataConnectorSummary:
    context = require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.MANAGE_ASSETS,
    )
    site = store.get_site(payload.site_id)
    if site is None:
        raise ValueError(f"Site {payload.site_id} was not found")
    require_resource_scope(context, site.organization_id, "site")
    connector = DataConnector(
        id=_normalize_required_text(payload.connector_id, 80, "Data connector id"),
        organization_id=organization_id,
        site_id=payload.site_id,
        name=_normalize_required_text(payload.name, 120, "Data connector name"),
        connector_type=payload.connector_type,
        source_category=normalize_upload_category(payload.source_category.value),
        status=payload.status,
        schedule=payload.schedule,
        external_reference=_normalize_optional_text(
            payload.external_reference,
            240,
            "Data connector external reference",
        ),
        credential_reference=_normalize_optional_text(
            payload.credential_reference,
            240,
            "Data connector credential reference",
        ),
        created_by_user_id=user_id,
        updated_by_user_id=user_id,
        next_sync_at=_parse_optional_connector_datetime(payload.next_sync_at),
    )
    audit_event = AuditEvent(
        id=f"audit_data_connector_created_{uuid4().hex}",
        organization_id=organization_id,
        actor_user_id=user_id,
        action=AuditAction.DATA_CONNECTOR_CREATED,
        resource_type="data_connector",
        resource_id=connector.id,
        metadata=_data_connector_audit_metadata(connector),
    )
    with store.conn:
        store.create_data_connector(connector)
        store.create_audit_event(audit_event)
    created = store.get_data_connector(connector.id)
    if created is None:
        raise ValueError("Data connector was not created")
    return create_data_connector_summary_from_row(created)


def update_data_connector_summary(
    user_id: str,
    organization_id: str,
    connector_id: str,
    payload: DataConnectorUpdate,
    store: SaaSStore,
) -> DataConnectorSummary:
    context = require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.MANAGE_ASSETS,
    )
    existing = store.get_data_connector(connector_id)
    if existing is None:
        raise ValueError(f"Data connector {connector_id} was not found")
    require_resource_scope(context, existing["organization_id"], "data_connector")
    if existing["organization_id"] != organization_id:
        raise ValueError(f"Data connector {connector_id} was not found")
    credential_reference = existing["credential_reference"]
    if "credential_reference" in payload.model_fields_set:
        credential_reference = _normalize_optional_text(
            payload.credential_reference,
            240,
            "Data connector credential reference",
        )
    connector = DataConnector(
        id=existing["id"],
        organization_id=organization_id,
        site_id=existing["site_id"],
        name=_normalize_required_text(payload.name, 120, "Data connector name"),
        connector_type=DataConnectorType(existing["connector_type"]),
        source_category=existing["source_category"],
        status=payload.status,
        schedule=payload.schedule,
        external_reference=_normalize_optional_text(
            payload.external_reference,
            240,
            "Data connector external reference",
        ),
        credential_reference=credential_reference,
        created_by_user_id=existing["created_by_user_id"],
        updated_by_user_id=user_id,
        last_sync_at=_parse_optional_connector_datetime(existing["last_sync_at"]),
        next_sync_at=_parse_optional_connector_datetime(payload.next_sync_at),
    )
    audit_event = AuditEvent(
        id=f"audit_data_connector_updated_{uuid4().hex}",
        organization_id=organization_id,
        actor_user_id=user_id,
        action=AuditAction.DATA_CONNECTOR_UPDATED,
        resource_type="data_connector",
        resource_id=connector.id,
        metadata=_data_connector_audit_metadata(connector),
    )
    with store.conn:
        store.update_data_connector(connector)
        store.create_audit_event(audit_event)
    updated = store.get_data_connector(connector.id)
    if updated is None:
        raise ValueError("Data connector was not updated")
    return create_data_connector_summary_from_row(updated)


def list_due_data_connector_summaries(
    user_id: str,
    organization_id: str,
    store: SaaSStore,
    due_at: str | None = None,
    limit: int = 50,
) -> list[DataConnectorSummary]:
    if limit < 1 or limit > 100:
        raise ValueError("Data connector due limit must be between 1 and 100")
    context = require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.READ,
    )
    effective_due_at = (
        _parse_optional_connector_datetime(due_at, "Data connector due time")
        or datetime.now(timezone.utc)
    )
    return [
        create_data_connector_summary_from_row(connector)
        for connector in store.list_due_data_connectors(
            effective_due_at.isoformat(),
            organization_id=context.organization_id,
            limit=limit,
        )
    ]


def list_data_connector_sync_summaries(
    user_id: str,
    organization_id: str,
    connector_id: str,
    store: SaaSStore,
    limit: int = 20,
) -> list[DataConnectorSyncSummary]:
    if limit < 1 or limit > 100:
        raise ValueError("Data connector sync limit must be between 1 and 100")
    context = require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.READ,
    )
    connector = store.get_data_connector(connector_id)
    if connector is None:
        raise ValueError(f"Data connector {connector_id} was not found")
    require_resource_scope(context, connector["organization_id"], "data_connector")
    if connector["organization_id"] != organization_id:
        raise ValueError(f"Data connector {connector_id} was not found")
    return [
        create_data_connector_sync_summary_from_row(sync)
        for sync in store.list_data_connector_syncs(organization_id, connector_id, limit=limit)
    ]


def record_data_connector_sync_summary(
    user_id: str,
    organization_id: str,
    connector_id: str,
    payload: DataConnectorSyncRecord,
    store: SaaSStore,
) -> DataConnectorSyncSummary:
    context = require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.MANAGE_ASSETS,
    )
    connector = store.get_data_connector(connector_id)
    if connector is None:
        raise ValueError(f"Data connector {connector_id} was not found")
    require_resource_scope(context, connector["organization_id"], "data_connector")
    if connector["organization_id"] != organization_id:
        raise ValueError(f"Data connector {connector_id} was not found")
    if connector["status"] != DataConnectorStatus.ACTIVE.value:
        raise ValueError("Only active data connectors can record sync attempts")

    upload_id = _normalize_optional_text(payload.upload_id, 80, "Data connector sync upload id")
    if payload.status != DataConnectorSyncStatus.SUCCEEDED and upload_id is not None:
        raise ValueError("Only successful data connector syncs can link an upload")
    if upload_id is not None:
        upload = store.get_upload(upload_id)
        if upload is None:
            raise ValueError(f"Upload {upload_id} was not found")
        require_resource_scope(context, upload["organization_id"], "upload")
        if upload["site_id"] != connector["site_id"]:
            raise ValueError("Data connector sync upload must belong to the connector site")
        if upload["category"] != connector["source_category"]:
            raise ValueError("Data connector sync upload category must match the connector source")

    completed_at = (
        _parse_optional_connector_datetime(
            payload.completed_at,
            "Data connector sync completed time",
        )
        or datetime.now(timezone.utc)
    )
    started_at = (
        _parse_optional_connector_datetime(
            payload.started_at,
            "Data connector sync started time",
        )
        or completed_at
    )
    status_reason = _normalize_optional_text(
        payload.status_reason,
        160,
        "Data connector sync status reason",
    )
    sync = DataConnectorSync(
        id=f"data_connector_sync_{uuid4().hex}",
        organization_id=organization_id,
        connector_id=connector["id"],
        site_id=connector["site_id"],
        source_category=connector["source_category"],
        status=payload.status,
        upload_id=upload_id,
        records_processed=payload.records_processed,
        status_reason=status_reason,
        triggered_by_user_id=user_id,
        started_at=started_at,
        completed_at=completed_at,
    )
    schedule = DataConnectorSchedule(connector["schedule"])
    next_sync_at = _next_data_connector_sync_at(completed_at, schedule)
    audit_event = AuditEvent(
        id=f"audit_data_connector_sync_{uuid4().hex}",
        organization_id=organization_id,
        actor_user_id=user_id,
        action=AuditAction.DATA_CONNECTOR_SYNC_RECORDED,
        resource_type="data_connector",
        resource_id=connector["id"],
        metadata=_data_connector_sync_audit_metadata(connector, sync),
    )
    with store.conn:
        store.create_data_connector_sync(sync)
        store.update_data_connector_sync_state(
            connector["id"],
            organization_id,
            last_sync_at=completed_at.isoformat(),
            last_sync_status=sync.status.value,
            last_sync_error=status_reason if sync.status == DataConnectorSyncStatus.FAILED else None,
            next_sync_at=next_sync_at.isoformat() if next_sync_at else None,
        )
        store.create_audit_event(audit_event)
    rows = store.list_data_connector_syncs(organization_id, connector_id, limit=1)
    if not rows:
        raise ValueError("Data connector sync was not recorded")
    return create_data_connector_sync_summary_from_row(rows[0])


def import_data_connector_content_summary(
    user_id: str,
    organization_id: str,
    connector_id: str,
    payload: DataConnectorImportCreate,
    store: SaaSStore,
    artifact_store: TenantArtifactStore,
) -> DataConnectorImportSummary:
    memberships = store.list_memberships(user_id=user_id, organization_id=organization_id)
    upload_context = require_permission(
        memberships,
        user_id=user_id,
        organization_id=organization_id,
        action=Action.MANAGE_UPLOADS,
    )
    asset_context = require_permission(
        memberships,
        user_id=user_id,
        organization_id=organization_id,
        action=Action.MANAGE_ASSETS,
    )
    connector = store.get_data_connector(connector_id)
    if connector is None:
        raise ValueError(f"Data connector {connector_id} was not found")
    require_resource_scope(asset_context, connector["organization_id"], "data_connector")
    if connector["organization_id"] != organization_id:
        raise ValueError(f"Data connector {connector_id} was not found")
    if connector["status"] != DataConnectorStatus.ACTIVE.value:
        raise ValueError("Only active data connectors can import source content")
    _require_site_in_scope(upload_context, connector["site_id"], store)

    content = _decode_base64_upload_content(payload.content_base64)
    category = normalize_upload_category(connector["source_category"])
    try:
        _validate_upload_source_content(category, content)
    except ValueError as exc:
        raise ValueError(f"{category} connector import failed source contract validation: {exc}") from exc

    upload_id = _normalize_optional_text(
        payload.upload_id,
        80,
        "Data connector import upload id",
    ) or f"connector_upload_{uuid4().hex}"
    filename = _normalize_required_text(
        payload.filename,
        180,
        "Data connector import filename",
    )
    upload = Upload(
        id=upload_id,
        organization_id=organization_id,
        site_id=connector["site_id"],
        uploaded_by_user_id=user_id,
        category=category,
        storage_key=upload_storage_key(
            organization_id=organization_id,
            site_id=connector["site_id"],
            upload_id=upload_id,
            filename=filename,
        ),
        checksum=sha256(content).hexdigest(),
        status=UploadStatus.STORED,
    )
    upload_event = AuditEvent(
        id=f"audit_upload_{upload.id}",
        organization_id=organization_id,
        actor_user_id=user_id,
        action=AuditAction.UPLOAD_STORED,
        resource_type="upload",
        resource_id=upload.id,
        metadata={
            "artifact_location_configured": "true",
            "category": upload.category,
            "site_id": upload.site_id,
            "source": "data_connector",
        },
    )
    completed_at = (
        _parse_optional_connector_datetime(
            payload.completed_at,
            "Data connector import completed time",
        )
        or datetime.now(timezone.utc)
    )
    started_at = (
        _parse_optional_connector_datetime(
            payload.started_at,
            "Data connector import started time",
        )
        or completed_at
    )
    status_reason = _normalize_optional_text(
        payload.status_reason,
        160,
        "Data connector import status reason",
    )
    sync = DataConnectorSync(
        id=f"data_connector_sync_{uuid4().hex}",
        organization_id=organization_id,
        connector_id=connector["id"],
        site_id=connector["site_id"],
        source_category=category,
        status=DataConnectorSyncStatus.SUCCEEDED,
        upload_id=upload.id,
        records_processed=payload.records_processed,
        status_reason=status_reason,
        triggered_by_user_id=user_id,
        started_at=started_at,
        completed_at=completed_at,
    )
    next_sync_at = _next_data_connector_sync_at(
        completed_at,
        DataConnectorSchedule(connector["schedule"]),
    )
    sync_event = AuditEvent(
        id=f"audit_data_connector_import_{uuid4().hex}",
        organization_id=organization_id,
        actor_user_id=user_id,
        action=AuditAction.DATA_CONNECTOR_SYNC_RECORDED,
        resource_type="data_connector",
        resource_id=connector["id"],
        metadata=_data_connector_sync_audit_metadata(connector, sync),
    )

    artifact_store.put_bytes(organization_id, upload.storage_key, content)
    with store.conn:
        store.create_upload(upload)
        store.create_audit_event(upload_event)
        store.create_data_connector_sync(sync)
        store.update_data_connector_sync_state(
            connector["id"],
            organization_id,
            last_sync_at=completed_at.isoformat(),
            last_sync_status=sync.status.value,
            last_sync_error=None,
            next_sync_at=next_sync_at.isoformat() if next_sync_at else None,
        )
        store.create_audit_event(sync_event)

    updated_connector = store.get_data_connector(connector["id"])
    sync_rows = store.list_data_connector_syncs(organization_id, connector["id"], limit=1)
    if updated_connector is None or not sync_rows:
        raise ValueError("Data connector import was not recorded")
    return DataConnectorImportSummary(
        connector=create_data_connector_summary_from_row(updated_connector),
        upload=_upload_summary_from_upload(upload),
        sync=create_data_connector_sync_summary_from_row(sync_rows[0]),
    )


def update_organization_summary(
    user_id: str,
    organization_id: str,
    payload: OrganizationProfileUpdate,
    store: SaaSStore,
) -> OrganizationSummary:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.MANAGE_ORGANIZATION,
    )
    current = store.get_organization(organization_id)
    if current is None:
        raise ValueError(f"Organization {organization_id} was not found")

    organization = Organization(
        id=organization_id,
        name=payload.name.strip(),
        slug=slug_part(payload.slug),
    )
    existing_slug = store.get_organization_by_slug(organization.slug)
    if existing_slug is not None and existing_slug["id"] != organization_id:
        raise ValueError(f"Organization slug {organization.slug} is already in use")

    audit_event = AuditEvent(
        id=f"audit_org_update_{uuid4().hex}",
        organization_id=organization_id,
        actor_user_id=user_id,
        action=AuditAction.ORGANIZATION_UPDATED,
        resource_type="organization",
        resource_id=organization_id,
        metadata={
            "previous_name": current["name"],
            "previous_slug": current["slug"],
            "name": organization.name,
            "slug": organization.slug,
        },
    )
    with store.conn:
        store.update_organization(organization)
        store.create_audit_event(audit_event)

    return OrganizationSummary(
        id=organization.id,
        name=organization.name,
        slug=organization.slug,
    )


def create_privacy_settings_summary(
    organization_id: str,
    row: Any | None,
) -> OrganizationPrivacySettingsSummary:
    if row is None:
        return OrganizationPrivacySettingsSummary(
            organization_id=organization_id,
            data_processing_acknowledged_at=None,
            data_processing_acknowledged_by_user_id=None,
            data_processing_contact_email=None,
            subprocessor_notifications_enabled=True,
            analytics_data_sharing_enabled=False,
            updated_at=None,
        )
    return OrganizationPrivacySettingsSummary(
        organization_id=row["organization_id"],
        data_processing_acknowledged_at=row["data_processing_acknowledged_at"],
        data_processing_acknowledged_by_user_id=row["data_processing_acknowledged_by_user_id"],
        data_processing_contact_email=row["data_processing_contact_email"],
        subprocessor_notifications_enabled=bool(row["subprocessor_notifications_enabled"]),
        analytics_data_sharing_enabled=bool(row["analytics_data_sharing_enabled"]),
        updated_at=row["updated_at"],
    )


def get_privacy_settings_summary(
    user_id: str,
    organization_id: str,
    store: SaaSStore,
) -> OrganizationPrivacySettingsSummary:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.READ,
    )
    if store.get_organization(organization_id) is None:
        raise ValueError(f"Organization {organization_id} was not found")
    return create_privacy_settings_summary(
        organization_id,
        store.get_organization_privacy_settings(organization_id),
    )


def update_privacy_settings_summary(
    user_id: str,
    organization_id: str,
    payload: OrganizationPrivacySettingsUpdate,
    store: SaaSStore,
) -> OrganizationPrivacySettingsSummary:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.MANAGE_ORGANIZATION,
    )
    if store.get_organization(organization_id) is None:
        raise ValueError(f"Organization {organization_id} was not found")

    current = store.get_organization_privacy_settings(organization_id)
    contact_email = (
        payload.data_processing_contact_email.strip()
        if payload.data_processing_contact_email is not None
        and payload.data_processing_contact_email.strip()
        else None
    )
    acknowledged_at = (
        datetime.now(timezone.utc).isoformat()
        if payload.acknowledge_data_processing
        else (current["data_processing_acknowledged_at"] if current is not None else None)
    )
    acknowledged_by = (
        user_id
        if payload.acknowledge_data_processing
        else (
            current["data_processing_acknowledged_by_user_id"]
            if current is not None
            else None
        )
    )
    audit_event = AuditEvent(
        id=f"audit_privacy_settings_{uuid4().hex}",
        organization_id=organization_id,
        actor_user_id=user_id,
        action=AuditAction.PRIVACY_SETTINGS_UPDATED,
        resource_type="privacy_settings",
        resource_id=organization_id,
        metadata={
            "data_processing_acknowledged": payload.acknowledge_data_processing,
            "data_processing_contact_email_configured": contact_email is not None,
            "subprocessor_notifications_enabled": payload.subprocessor_notifications_enabled,
            "analytics_data_sharing_enabled": payload.analytics_data_sharing_enabled,
        },
    )
    with store.conn:
        store.upsert_organization_privacy_settings(
            organization_id=organization_id,
            data_processing_acknowledged_at=acknowledged_at,
            data_processing_acknowledged_by_user_id=acknowledged_by,
            data_processing_contact_email=contact_email,
            subprocessor_notifications_enabled=payload.subprocessor_notifications_enabled,
            analytics_data_sharing_enabled=payload.analytics_data_sharing_enabled,
        )
        store.create_audit_event(audit_event)

    return create_privacy_settings_summary(
        organization_id,
        store.get_organization_privacy_settings(organization_id),
    )


def create_privacy_request_summary_from_row(row: Any) -> OrganizationPrivacyRequestSummary:
    return OrganizationPrivacyRequestSummary(
        id=row["id"],
        organization_id=row["organization_id"],
        requested_by_user_id=row["requested_by_user_id"],
        subject_user_id=row["subject_user_id"],
        request_type=PrivacyRequestType(row["request_type"]),
        status=PrivacyRequestStatus(row["status"]),
        description=row["description"],
        resolution_notes=row["resolution_notes"],
        completed_at=row["completed_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def list_privacy_request_summaries(
    user_id: str,
    organization_id: str,
    store: SaaSStore,
) -> list[OrganizationPrivacyRequestSummary]:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.MANAGE_ORGANIZATION,
    )
    if store.get_organization(organization_id) is None:
        raise ValueError(f"Organization {organization_id} was not found")
    return [
        create_privacy_request_summary_from_row(row)
        for row in store.list_organization_privacy_requests(organization_id)
    ]


def create_privacy_request_summary(
    user_id: str,
    organization_id: str,
    payload: OrganizationPrivacyRequestCreate,
    store: SaaSStore,
) -> OrganizationPrivacyRequestSummary:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.MANAGE_ORGANIZATION,
    )
    if store.get_organization(organization_id) is None:
        raise ValueError(f"Organization {organization_id} was not found")

    subject_user_id = payload.subject_user_id.strip() if payload.subject_user_id else None
    if subject_user_id and not store.list_memberships(
        user_id=subject_user_id,
        organization_id=organization_id,
    ):
        raise AccessDenied(
            f"Subject user {subject_user_id} is not a member of organization {organization_id}"
        )
    description = payload.description.strip() if payload.description and payload.description.strip() else None
    request_id = f"privacy_request_{uuid4().hex}"
    request_status = PrivacyRequestStatus.OPEN
    audit_event = AuditEvent(
        id=f"audit_privacy_request_created_{uuid4().hex}",
        organization_id=organization_id,
        actor_user_id=user_id,
        action=AuditAction.PRIVACY_REQUEST_CREATED,
        resource_type="privacy_request",
        resource_id=request_id,
        metadata={
            "request_type": payload.request_type.value,
            "status": request_status.value,
            "subject_user_id_configured": subject_user_id is not None,
            "description_configured": description is not None,
        },
    )
    with store.conn:
        store.create_organization_privacy_request(
            request_id=request_id,
            organization_id=organization_id,
            requested_by_user_id=user_id,
            subject_user_id=subject_user_id,
            request_type=payload.request_type,
            status=request_status,
            description=description,
        )
        store.create_audit_event(audit_event)

    row = store.get_organization_privacy_request(request_id)
    if row is None:
        raise ValueError(f"Privacy request {request_id} was not found")
    return create_privacy_request_summary_from_row(row)


def update_privacy_request_summary(
    user_id: str,
    organization_id: str,
    privacy_request_id: str,
    payload: OrganizationPrivacyRequestStatusUpdate,
    store: SaaSStore,
) -> OrganizationPrivacyRequestSummary:
    context = require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.MANAGE_ORGANIZATION,
    )
    row = store.get_organization_privacy_request(privacy_request_id)
    if row is None:
        raise ValueError(f"Privacy request {privacy_request_id} was not found")
    require_resource_scope(context, row["organization_id"], "privacy request")

    resolution_notes = (
        payload.resolution_notes.strip()
        if payload.resolution_notes and payload.resolution_notes.strip()
        else None
    )
    completed_at = (
        datetime.now(timezone.utc).isoformat()
        if payload.status == PrivacyRequestStatus.COMPLETED
        else None
    )
    audit_event = AuditEvent(
        id=f"audit_privacy_request_updated_{uuid4().hex}",
        organization_id=organization_id,
        actor_user_id=user_id,
        action=AuditAction.PRIVACY_REQUEST_UPDATED,
        resource_type="privacy_request",
        resource_id=privacy_request_id,
        metadata={
            "previous_status": row["status"],
            "status": payload.status.value,
            "resolution_notes_configured": resolution_notes is not None,
        },
    )
    with store.conn:
        store.update_organization_privacy_request_status(
            request_id=privacy_request_id,
            status=payload.status,
            resolution_notes=resolution_notes,
            completed_at=completed_at,
        )
        store.create_audit_event(audit_event)

    updated = store.get_organization_privacy_request(privacy_request_id)
    if updated is None:
        raise ValueError(f"Privacy request {privacy_request_id} was not found")
    return create_privacy_request_summary_from_row(updated)


def create_pilot_readiness_summary(
    organization_id: str,
    rows: list[Any],
) -> OrganizationPilotReadinessSummary:
    row_by_key = {row["item_key"]: row for row in rows}
    items: list[OrganizationPilotReadinessItemSummary] = []
    for item_key, label in PILOT_READINESS_ITEMS.items():
        row = row_by_key.get(item_key)
        status_value = row["status"] if row is not None else PilotReadinessStatus.PENDING.value
        items.append(
            OrganizationPilotReadinessItemSummary(
                item_key=item_key,
                label=label,
                status=PilotReadinessStatus(status_value),
                evidence_url=row["evidence_url"] if row is not None else None,
                notes=row["notes"] if row is not None else None,
                updated_by_user_id=row["updated_by_user_id"] if row is not None else None,
                updated_at=row["updated_at"] if row is not None else None,
            )
        )

    return OrganizationPilotReadinessSummary(
        organization_id=organization_id,
        verified_count=sum(
            1 for item in items if item.status == PilotReadinessStatus.VERIFIED
        ),
        total_count=len(items),
        items=items,
    )


def get_pilot_readiness_summary(
    user_id: str,
    organization_id: str,
    store: SaaSStore,
) -> OrganizationPilotReadinessSummary:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.MANAGE_ORGANIZATION,
    )
    if store.get_organization(organization_id) is None:
        raise ValueError(f"Organization {organization_id} was not found")
    return create_pilot_readiness_summary(
        organization_id,
        store.list_organization_pilot_readiness_items(organization_id),
    )


def update_pilot_readiness_item_summary(
    user_id: str,
    organization_id: str,
    item_key: str,
    payload: OrganizationPilotReadinessUpdate,
    store: SaaSStore,
) -> OrganizationPilotReadinessSummary:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.MANAGE_ORGANIZATION,
    )
    if store.get_organization(organization_id) is None:
        raise ValueError(f"Organization {organization_id} was not found")
    if item_key not in PILOT_READINESS_ITEMS:
        raise ValueError(f"Unsupported pilot readiness item: {item_key}")

    current = store.get_organization_pilot_readiness_item(organization_id, item_key)
    evidence_url = (
        payload.evidence_url.strip()
        if payload.evidence_url is not None and payload.evidence_url.strip()
        else None
    )
    notes = (
        payload.notes.strip()
        if payload.notes is not None and payload.notes.strip()
        else None
    )
    audit_event = AuditEvent(
        id=f"audit_pilot_readiness_{uuid4().hex}",
        organization_id=organization_id,
        actor_user_id=user_id,
        action=AuditAction.PILOT_READINESS_UPDATED,
        resource_type="pilot_readiness",
        resource_id=item_key,
        metadata={
            "item_key": item_key,
            "previous_status": (
                current["status"] if current is not None else PilotReadinessStatus.PENDING.value
            ),
            "status": payload.status.value,
            "evidence_url_configured": evidence_url is not None,
            "notes_configured": notes is not None,
        },
    )
    with store.conn:
        store.upsert_organization_pilot_readiness_item(
            organization_id=organization_id,
            item_key=item_key,
            status=payload.status,
            evidence_url=evidence_url,
            notes=notes,
            updated_by_user_id=user_id,
        )
        store.create_audit_event(audit_event)

    return get_pilot_readiness_summary(user_id, organization_id, store)


def list_site_summaries(
    user_id: str,
    organization_id: str,
    store: SaaSStore,
) -> list[SiteSummary]:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.READ,
    )
    return [
        SiteSummary(
            id=site.id,
            organization_id=site.organization_id,
            name=site.name,
            timezone=site.timezone,
        )
        for site in store.list_sites(organization_id)
    ]


def create_site_summary(
    user_id: str,
    organization_id: str,
    payload: SiteCreate,
    store: SaaSStore,
) -> SiteSummary:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.MANAGE_ASSETS,
    )
    require_site_capacity(store, organization_id)
    site = Site(
        id=payload.site_id,
        organization_id=organization_id,
        name=payload.name,
        timezone=payload.timezone,
    )
    audit_event = AuditEvent(
        id=f"audit_site_{site.id}",
        organization_id=organization_id,
        actor_user_id=user_id,
        action=AuditAction.SITE_CREATED,
        resource_type="site",
        resource_id=site.id,
        metadata={
            "name": site.name,
            "timezone": site.timezone,
        },
    )
    with store.conn:
        store.create_site(site)
        store.create_audit_event(audit_event)
    return SiteSummary(
        id=site.id,
        organization_id=site.organization_id,
        name=site.name,
        timezone=site.timezone,
    )


def list_membership_summaries(
    user_id: str,
    organization_id: str,
    store: SaaSStore,
) -> list[MembershipSummary]:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.MANAGE_MEMBERS,
    )
    return [
        MembershipSummary(
            user_id=row["user_id"],
            organization_id=row["organization_id"],
            role=row["role"],
            invited_by_user_id=row["invited_by_user_id"],
            email=row["email"],
            display_name=row["display_name"],
            is_active=bool(row["is_active"]),
        )
        for row in store.list_organization_members(organization_id)
    ]


def create_membership_summary(record: MembershipAdminRecord) -> MembershipSummary:
    return MembershipSummary(
        user_id=record.user_id,
        organization_id=record.organization_id,
        role=record.role.value,
        invited_by_user_id=record.invited_by_user_id,
        email=record.email,
        display_name=record.display_name,
        is_active=record.is_active,
    )


def update_membership_summary(
    user_id: str,
    organization_id: str,
    member_user_id: str,
    payload: MembershipRoleUpdate,
    store: SaaSStore,
) -> MembershipSummary:
    return create_membership_summary(
        update_membership_role(
            store,
            actor_user_id=user_id,
            organization_id=organization_id,
            member_user_id=member_user_id,
            role=payload.role,
        )
    )


def remove_membership_summary(
    user_id: str,
    organization_id: str,
    member_user_id: str,
    store: SaaSStore,
) -> MembershipSummary:
    return create_membership_summary(
        remove_membership(
            store,
            actor_user_id=user_id,
            organization_id=organization_id,
            member_user_id=member_user_id,
        )
    )


def list_organization_invite_summaries(
    user_id: str,
    organization_id: str,
    store: SaaSStore,
) -> list[OrganizationInviteSummary]:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.MANAGE_MEMBERS,
    )
    return [
        OrganizationInviteSummary(
            id=row["id"],
            organization_id=row["organization_id"],
            email=row["email"],
            role=row["role"],
            invited_by_user_id=row["invited_by_user_id"],
            status=row["status"],
            accepted_by_user_id=row["accepted_by_user_id"],
            accepted_at=row["accepted_at"],
        )
        for row in store.list_organization_invites(organization_id)
    ]


def list_meter_summaries(
    user_id: str,
    organization_id: str,
    store: SaaSStore,
    site_id: str | None = None,
) -> list[MeterSummary]:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.READ,
    )
    return [
        MeterSummary(
            id=meter.id,
            organization_id=meter.organization_id,
            site_id=meter.site_id,
            display_name=meter.display_name,
            commodity=meter.commodity,
            unit=meter.unit,
            source_column=meter.source_column,
            is_seu=meter.is_seu,
        )
        for meter in store.list_meters(organization_id=organization_id, site_id=site_id)
    ]


def create_meter_summary(
    user_id: str,
    organization_id: str,
    payload: MeterCreate,
    store: SaaSStore,
) -> MeterSummary:
    context = require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.MANAGE_ASSETS,
    )
    _require_site_in_scope(context, payload.site_id, store)
    meter = Meter(
        id=payload.meter_id,
        organization_id=organization_id,
        site_id=payload.site_id,
        display_name=payload.display_name,
        commodity=payload.commodity,
        unit=payload.unit,
        source_column=payload.source_column,
        is_seu=payload.is_seu,
    )
    audit_event = AuditEvent(
        id=f"audit_meter_{meter.id}",
        organization_id=organization_id,
        actor_user_id=user_id,
        action=AuditAction.METER_CREATED,
        resource_type="meter",
        resource_id=meter.id,
        metadata={
            "site_id": meter.site_id,
            "display_name": meter.display_name,
            "commodity": meter.commodity,
            "unit": meter.unit,
            "source_column": meter.source_column,
            "is_seu": str(meter.is_seu).lower(),
        },
    )
    with store.conn:
        store.create_meter(meter)
        store.create_audit_event(audit_event)
    return MeterSummary(
        id=meter.id,
        organization_id=meter.organization_id,
        site_id=meter.site_id,
        display_name=meter.display_name,
        commodity=meter.commodity,
        unit=meter.unit,
        source_column=meter.source_column,
        is_seu=meter.is_seu,
    )


def _upload_summary_from_row(upload: Any) -> UploadSummary:
    try:
        category = normalize_upload_category(upload["category"])
    except ValueError:
        category = upload["category"]
    return UploadSummary(
        id=upload["id"],
        organization_id=upload["organization_id"],
        site_id=upload["site_id"],
        uploaded_by_user_id=upload["uploaded_by_user_id"],
        category=category,
        checksum=upload["checksum"],
        status=upload["status"],
    )


def _upload_summary_from_upload(upload: Upload) -> UploadSummary:
    return UploadSummary(
        id=upload.id,
        organization_id=upload.organization_id,
        site_id=upload.site_id,
        uploaded_by_user_id=upload.uploaded_by_user_id,
        category=upload.category,
        checksum=upload.checksum,
        status=upload.status.value,
    )


def _decode_base64_upload_content(content_base64: str) -> bytes:
    try:
        content = base64.b64decode(content_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Upload content must be valid base64") from exc
    if not content:
        raise ValueError("Upload content cannot be empty")
    return content


def _decode_upload_csv(content: bytes) -> str:
    for encoding in ("utf-8-sig", "latin1"):
        try:
            text = content.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - latin1 can decode every byte sequence
        raise ValueError("Upload content must be a readable CSV text file")
    if "\x00" in text:
        raise ValueError("Upload content must be a CSV text file")
    return text


def _csv_rows_from_upload(content: bytes) -> list[list[str]]:
    text = _decode_upload_csv(content)
    try:
        rows = [
            [cell.strip() for cell in row]
            for row in csv.reader(StringIO(text))
            if any(cell.strip() for cell in row)
        ]
    except csv.Error as exc:
        raise ValueError(f"Upload content must be valid CSV: {exc}") from exc
    if not rows:
        raise ValueError("Upload CSV must contain a header and at least one data row")
    return rows


def _csv_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _header_positions(header: list[str]) -> dict[str, int]:
    positions: dict[str, int] = {}
    for index, value in enumerate(header):
        key = _csv_key(value)
        if key and key not in positions:
            positions[key] = index
    return positions


def _row_value(row: list[str], index: int) -> str:
    return row[index].strip() if index < len(row) else ""


def _is_parseable_date(value: str) -> bool:
    if not value.strip():
        return False
    return not pd.isna(pd.to_datetime(value, errors="coerce", dayfirst=True))


def _is_numeric_value(value: str) -> bool:
    if not value.strip():
        return False
    return not pd.isna(pd.to_numeric(value, errors="coerce"))


def _find_tabular_header(rows: list[list[str]]) -> tuple[int, list[str], dict[str, int]] | None:
    for index, row in enumerate(rows):
        positions = _header_positions(row)
        if SOURCE_FILE_DATE_HEADERS.intersection(positions):
            return index, row, positions
    return None


def _validate_date_value_csv(
    rows: list[list[str]],
    category: str,
    value_header_predicate: Any,
    description: str,
) -> None:
    header_match = _find_tabular_header(rows)
    if header_match is None:
        raise ValueError(f"{description} CSV must include a Date column")
    header_index, header, positions = header_match
    date_index = positions[next(header for header in SOURCE_FILE_DATE_HEADERS if header in positions)]
    value_indexes = [
        index
        for index, column in enumerate(header)
        if index != date_index and value_header_predicate(_csv_key(column))
    ]
    if not value_indexes:
        raise ValueError(f"{description} CSV must include a {category.upper()} value column")
    for row in rows[header_index + 1 :]:
        if _is_parseable_date(_row_value(row, date_index)) and any(
            _is_numeric_value(_row_value(row, value_index)) for value_index in value_indexes
        ):
            return
    raise ValueError(f"{description} CSV must include dated numeric {category.upper()} values")


def _validate_energy_upload_csv(rows: list[list[str]]) -> None:
    for index, row in enumerate(rows):
        positions = _header_positions(row)
        if {"metered_sector", "utility", "units", "period"}.issubset(positions):
            date_columns = [column for column in row if _is_parseable_date(column)]
            if date_columns and len(rows) > index + 1:
                return

    header_match = _find_tabular_header(rows)
    if header_match is None:
        raise ValueError(
            "Energy CSV must include Date plus meter columns, or legacy Metered Sector/Utility/Units/Period columns"
        )
    header_index, header, positions = header_match
    date_index = positions[next(header for header in SOURCE_FILE_DATE_HEADERS if header in positions)]
    value_indexes = [
        index
        for index, column in enumerate(header)
        if index != date_index and _csv_key(column) not in {"hdd", "cdd", "hdd_15_5", "cdd_15_5"}
    ]
    if not value_indexes:
        raise ValueError("Energy CSV must include at least one meter value column")
    for row in rows[header_index + 1 :]:
        if _is_parseable_date(_row_value(row, date_index)) and any(
            _is_numeric_value(_row_value(row, value_index)) for value_index in value_indexes
        ):
            return
    raise ValueError("Energy CSV must include dated numeric meter values")


def _validate_seu_mapping_csv(rows: list[list[str]]) -> None:
    for header_index, header in enumerate(rows):
        positions = _header_positions(header)
        if {"meter", "seu_category"}.issubset(positions):
            meter_index = positions["meter"]
            category_index = positions["seu_category"]
            for row in rows[header_index + 1 :]:
                if _row_value(row, meter_index) and _row_value(row, category_index):
                    return
            raise ValueError("SEU mapping CSV must include at least one meter/category row")
    raise ValueError("SEU mapping CSV must include Meter and SEU_Category columns")


def _validate_upload_source_content(category: str, content: bytes) -> None:
    normalized_category = normalize_upload_category(category)
    rows = _csv_rows_from_upload(content)
    if normalized_category == UploadCategory.ENERGY.value:
        _validate_energy_upload_csv(rows)
    elif normalized_category == UploadCategory.HDD.value:
        _validate_date_value_csv(
            rows,
            "hdd",
            lambda header: "hdd" in header or "heating_degree" in header,
            "HDD",
        )
    elif normalized_category == UploadCategory.CDD.value:
        _validate_date_value_csv(
            rows,
            "cdd",
            lambda header: "cdd" in header or "cooling_degree" in header,
            "CDD",
        )
    elif normalized_category == UploadCategory.SEU_MAPPING.value:
        _validate_seu_mapping_csv(rows)


def list_upload_summaries(
    user_id: str,
    organization_id: str,
    store: SaaSStore,
    site_id: str | None = None,
) -> list[UploadSummary]:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.READ,
    )
    return [
        _upload_summary_from_row(upload)
        for upload in store.list_uploads(organization_id=organization_id, site_id=site_id)
    ]


def get_upload_readiness_summary(
    user_id: str,
    organization_id: str,
    store: SaaSStore,
    site_id: str | None = None,
) -> UploadReadinessSummary:
    context = require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.READ,
    )
    if site_id is not None:
        sites = [_require_site_in_scope(context, site_id, store)]
    else:
        sites = store.list_sites(organization_id)

    latest_stored_uploads: dict[tuple[str, str], UploadSummary] = {}
    for upload in store.list_uploads(organization_id=organization_id, site_id=site_id):
        if upload["status"] != UploadStatus.STORED.value:
            continue
        try:
            category = normalize_upload_category(upload["category"])
        except ValueError:
            continue
        key = (upload["site_id"], category)
        if key not in latest_stored_uploads:
            latest_stored_uploads[key] = _upload_summary_from_row(upload)

    site_summaries: list[SiteUploadReadinessSummary] = []
    for site in sites:
        files: list[UploadReadinessFileSummary] = []
        for category, label, is_required in UPLOAD_SOURCE_FILE_CONTRACT:
            latest_upload = latest_stored_uploads.get((site.id, category.value))
            files.append(
                UploadReadinessFileSummary(
                    category=category.value,
                    label=label,
                    is_required=is_required,
                    is_ready=latest_upload is not None,
                    latest_upload=latest_upload,
                )
            )

        required_files = [file for file in files if file.is_required]
        ready_count = sum(1 for file in required_files if file.is_ready)
        required_count = len(required_files)
        site_summaries.append(
            SiteUploadReadinessSummary(
                site_id=site.id,
                site_name=site.name,
                ready_count=ready_count,
                required_count=required_count,
                is_ready=ready_count == required_count,
                files=files,
            )
        )

    return UploadReadinessSummary(
        organization_id=organization_id,
        ready_site_count=sum(1 for site_summary in site_summaries if site_summary.is_ready),
        site_count=len(site_summaries),
        required_categories=[
            category.value for category, _label, is_required in UPLOAD_SOURCE_FILE_CONTRACT if is_required
        ],
        sites=site_summaries,
    )


def create_upload_summary(
    user_id: str,
    organization_id: str,
    payload: UploadCreate,
    store: SaaSStore,
) -> UploadSummary:
    context = require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.MANAGE_UPLOADS,
    )
    _require_site_in_scope(context, payload.site_id, store)
    upload = Upload(
        id=payload.upload_id,
        organization_id=organization_id,
        site_id=payload.site_id,
        uploaded_by_user_id=user_id,
        category=payload.category,
        storage_key=upload_storage_key(
            organization_id=organization_id,
            site_id=payload.site_id,
            upload_id=payload.upload_id,
            filename=payload.filename,
        ),
        checksum=payload.checksum,
        status=UploadStatus.STORED,
    )
    event = AuditEvent(
        id=f"audit_upload_{upload.id}",
        organization_id=organization_id,
        actor_user_id=user_id,
        action=AuditAction.UPLOAD_STORED,
        resource_type="upload",
        resource_id=upload.id,
        metadata={"category": upload.category, "site_id": upload.site_id},
    )
    with store.conn:
        store.create_upload(upload)
        store.create_audit_event(event)

    return UploadSummary(
        id=upload.id,
        organization_id=upload.organization_id,
        site_id=upload.site_id,
        uploaded_by_user_id=upload.uploaded_by_user_id,
        category=upload.category,
        checksum=upload.checksum,
        status=upload.status.value,
    )


def create_upload_content_summary(
    user_id: str,
    organization_id: str,
    payload: UploadContentCreate,
    store: SaaSStore,
    artifact_store: TenantArtifactStore,
) -> UploadSummary:
    context = require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.MANAGE_UPLOADS,
    )
    _require_site_in_scope(context, payload.site_id, store)
    content = _decode_base64_upload_content(payload.content_base64)
    category = normalize_upload_category(payload.category)
    try:
        _validate_upload_source_content(category, content)
    except ValueError as exc:
        raise ValueError(f"{category} upload failed source contract validation: {exc}") from exc

    upload = Upload(
        id=payload.upload_id,
        organization_id=organization_id,
        site_id=payload.site_id,
        uploaded_by_user_id=user_id,
        category=category,
        storage_key=upload_storage_key(
            organization_id=organization_id,
            site_id=payload.site_id,
            upload_id=payload.upload_id,
            filename=payload.filename,
        ),
        checksum=sha256(content).hexdigest(),
        status=UploadStatus.STORED,
    )
    event = AuditEvent(
        id=f"audit_upload_{upload.id}",
        organization_id=organization_id,
        actor_user_id=user_id,
        action=AuditAction.UPLOAD_STORED,
        resource_type="upload",
        resource_id=upload.id,
        metadata={
            "artifact_location_configured": "true",
            "category": upload.category,
            "site_id": upload.site_id,
        },
    )

    artifact_store.put_bytes(organization_id, upload.storage_key, content)
    with store.conn:
        store.create_upload(upload)
        store.create_audit_event(event)

    return _upload_summary_from_upload(upload)


def list_run_summaries(
    user_id: str,
    organization_id: str,
    store: SaaSStore,
    site_id: str | None = None,
) -> list[RunSummary]:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.READ,
    )
    return [
        RunSummary(
            id=run["id"],
            organization_id=run["organization_id"],
            site_id=run["site_id"],
            requested_by_user_id=run["requested_by_user_id"],
            status=run["status"],
            error_message=run["error_message"],
            completed_at=run["completed_at"],
        )
        for run in store.list_runs(organization_id=organization_id, site_id=site_id)
    ]


def create_run_summary(
    user_id: str,
    organization_id: str,
    payload: RunCreate,
    store: SaaSStore,
) -> RunSummary:
    context = require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.MANAGE_RUNS,
    )
    _require_site_in_scope(context, payload.site_id, store)
    for upload_id in payload.upload_ids:
        _require_upload_in_scope(context, upload_id, payload.site_id, store)
    require_run_capacity(store, organization_id)
    run = Run(
        id=payload.run_id,
        organization_id=organization_id,
        site_id=payload.site_id,
        requested_by_user_id=user_id,
        upload_ids=payload.upload_ids,
        status=RunStatus.QUEUED,
    )
    with store.conn:
        store.create_run(run)
    return RunSummary(
        id=run.id,
        organization_id=run.organization_id,
        site_id=run.site_id,
        requested_by_user_id=run.requested_by_user_id,
        status=run.status.value,
        error_message=run.error_message,
        completed_at=run.completed_at.isoformat() if run.completed_at else None,
    )


def list_analysis_job_summaries(
    user_id: str,
    organization_id: str,
    store: SaaSStore,
) -> list[AnalysisJobSummary]:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.READ,
    )
    return [
        AnalysisJobSummary(
            id=job["id"],
            organization_id=job["organization_id"],
            run_id=job["run_id"],
            upload_id=job["upload_id"],
            report_id=job["report_id"],
            requested_by_user_id=job["requested_by_user_id"],
            status=job["status"],
            error_message=job["error_message"],
            completed_at=job["completed_at"],
        )
        for job in store.list_analysis_jobs(organization_id)
    ]


def list_notification_summaries(
    user_id: str,
    organization_id: str,
    store: SaaSStore,
) -> list[NotificationSummary]:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.READ,
    )
    return [
        NotificationSummary(
            id=notification["id"],
            organization_id=notification["organization_id"],
            recipient_user_id=notification["recipient_user_id"],
            channel=notification["channel"],
            event_type=notification["event_type"],
            subject=notification["subject"],
            body=notification["body"],
            resource_type=notification["resource_type"],
            resource_id=notification["resource_id"],
            status=notification["status"],
            error_message=notification["error_message"],
            sent_at=notification["sent_at"],
            attempt_count=notification["attempt_count"],
            last_attempt_at=notification["last_attempt_at"],
            next_attempt_at=notification["next_attempt_at"],
        )
        for notification in store.list_notifications(
            organization_id=organization_id,
            recipient_user_id=user_id,
        )
    ]


def list_report_summaries(
    user_id: str,
    organization_id: str,
    store: SaaSStore,
    run_id: str | None = None,
) -> list[ReportSummary]:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.READ,
    )
    return [
        ReportSummary(
            id=report["id"],
            organization_id=report["organization_id"],
            run_id=report["run_id"],
            report_type=report["report_type"],
            is_published=bool(report["is_published"]),
            expires_at=report["expires_at"],
            retention_applied_at=report["retention_applied_at"],
        )
        for report in store.list_reports(organization_id=organization_id, run_id=run_id)
        if report_is_visible(report)
    ]


def create_report_comment_summary_from_row(comment: Any) -> ReportCommentSummary:
    return ReportCommentSummary(
        id=comment["id"],
        organization_id=comment["organization_id"],
        report_id=comment["report_id"],
        author_user_id=comment["author_user_id"],
        body=comment["body"],
        created_at=comment["created_at"],
    )


def _visible_report_for_action(
    user_id: str,
    organization_id: str,
    report_id: str,
    store: SaaSStore,
    action: Action,
) -> Any:
    context = require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=action,
    )
    report = store.get_report(report_id)
    if report is None:
        raise ValueError(f"Report {report_id} was not found")
    require_resource_scope(context, report["organization_id"], "report")
    require_report_visible(report)
    return report


def _load_report_artifact_content(
    organization_id: str,
    report: Any,
    artifact_store: TenantArtifactStore,
) -> dict[str, Any]:
    try:
        content = json.loads(artifact_store.get_text(organization_id, report["storage_key"]))
    except FileNotFoundError as exc:
        raise ValueError(f"Report artifact {report['id']} was not found") from exc
    if not isinstance(content, dict):
        raise ValueError(f"Report artifact {report['id']} is not a JSON object")
    return sanitize_report_artifact_content(content)


def _report_artifact_digest_for_notification(
    organization_id: str,
    report: Any,
    artifact_store: TenantArtifactStore | None,
) -> tuple[str, ...]:
    if artifact_store is None:
        return ()
    try:
        return tuple(
            report_artifact_digest_lines(
                _load_report_artifact_content(organization_id, report, artifact_store)
            )
        )
    except ValueError:
        return ()


def list_report_comment_summaries(
    user_id: str,
    organization_id: str,
    report_id: str,
    store: SaaSStore,
) -> list[ReportCommentSummary]:
    _visible_report_for_action(
        user_id,
        organization_id,
        report_id,
        store,
        Action.READ,
    )
    return [
        create_report_comment_summary_from_row(comment)
        for comment in store.list_report_comments(organization_id, report_id)
    ]


def create_report_comment_summary(
    user_id: str,
    organization_id: str,
    report_id: str,
    payload: ReportCommentCreate,
    store: SaaSStore,
) -> ReportCommentSummary:
    report = _visible_report_for_action(
        user_id,
        organization_id,
        report_id,
        store,
        Action.COMMENT_REPORTS,
    )
    body = payload.body.strip()
    if not body:
        raise ValueError("Report comment body must not be blank")

    comment_id = f"report_comment_{uuid4().hex}"
    with store.conn:
        store.create_report_comment(
            ReportComment(
                id=comment_id,
                organization_id=organization_id,
                report_id=report["id"],
                author_user_id=user_id,
                body=body,
            )
        )
        store.create_audit_event(
            AuditEvent(
                id=f"audit_report_comment_{uuid4().hex}",
                organization_id=organization_id,
                actor_user_id=user_id,
                action=AuditAction.REPORT_COMMENT_CREATED,
                resource_type="report_comment",
                resource_id=comment_id,
                metadata={
                    "body_length": len(body),
                    "report_id": report["id"],
                    "report_type": report["report_type"],
                },
            )
        )

    comment = store.get_report_comment(comment_id)
    if comment is None:
        raise ValueError(f"Report comment {comment_id} was not found")
    return create_report_comment_summary_from_row(comment)


def create_report_review_summary_from_row(review: Any) -> ReportReviewSummary:
    return ReportReviewSummary(
        id=review["id"],
        organization_id=review["organization_id"],
        report_id=review["report_id"],
        reviewer_user_id=review["reviewer_user_id"],
        status=ReportReviewStatus(review["status"]),
        notes=review["notes"],
        created_at=review["created_at"],
    )


def create_report_approval_assignment_summary_from_row(
    assignment: Any,
) -> ReportApprovalAssignmentSummary:
    return ReportApprovalAssignmentSummary(
        id=assignment["id"],
        organization_id=assignment["organization_id"],
        report_id=assignment["report_id"],
        assigned_to_user_id=assignment["assigned_to_user_id"],
        assigned_by_user_id=assignment["assigned_by_user_id"],
        status=ReportApprovalAssignmentStatus(assignment["status"]),
        due_at=assignment["due_at"],
        notes=assignment["notes"],
        completed_at=assignment["completed_at"],
        created_at=assignment["created_at"],
        updated_at=assignment["updated_at"],
    )


def _normalize_report_review_notes(value: str | None) -> str | None:
    if value is None:
        return None
    notes = value.strip()
    if not notes:
        return None
    if len(notes) > 2000:
        raise ValueError("Report review notes must be 2000 characters or fewer")
    return notes


def _normalize_report_approval_assignment_notes(value: str | None) -> str | None:
    if value is None:
        return None
    notes = value.strip()
    if not notes:
        return None
    if len(notes) > 2000:
        raise ValueError("Report approval assignment notes must be 2000 characters or fewer")
    return notes


def _parse_report_approval_due_at(value: str | None) -> datetime | None:
    if value is None:
        return None
    raw_value = value.strip()
    if not raw_value:
        return None
    try:
        parsed = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Report approval assignment due date must be ISO-8601") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc)
    if parsed <= datetime.now(timezone.utc):
        raise ValueError("Report approval assignment due date must be in the future")
    return parsed


def _report_approval_assignment_notification(
    *,
    assignment_id: str,
    organization_id: str,
    report: Any,
    recipient_user_id: str,
    event_type: str,
    subject: str,
    body_lines: list[str],
) -> Notification:
    return Notification(
        id=f"{assignment_id}:email:{event_type}",
        organization_id=organization_id,
        recipient_user_id=recipient_user_id,
        channel=NotificationChannel.EMAIL,
        event_type=event_type,
        subject=subject,
        body="\n".join(body_lines),
        resource_type="report_approval_assignment",
        resource_id=assignment_id,
        status=NotificationStatus.PENDING,
    )


def _report_approval_assigned_notification(
    *,
    assignment_id: str,
    organization_id: str,
    report: Any,
    recipient_user_id: str,
    due_at: datetime | None,
    artifact_digest_lines: tuple[str, ...] = (),
) -> Notification:
    body_lines = [
        "A report approval has been assigned to you.",
        "",
        f"Report: {report['id']}",
        f"Report type: {report['report_type']}",
        f"Assignment: {assignment_id}",
    ]
    if due_at is not None:
        body_lines.append(f"Due: {due_at.isoformat()}")
    if artifact_digest_lines:
        body_lines.extend(["", "Report digest:", *artifact_digest_lines])
    body_lines.extend(
        [
            "",
            "Sign in to the RDS dashboard to review the report and record a decision.",
        ]
    )
    return _report_approval_assignment_notification(
        assignment_id=assignment_id,
        organization_id=organization_id,
        report=report,
        recipient_user_id=recipient_user_id,
        event_type="report_approval_assigned",
        subject=f"ISO 50001 report approval assigned: {report['report_type']}",
        body_lines=body_lines,
    )


def _report_approval_updated_notification(
    *,
    assignment_id: str,
    organization_id: str,
    report: Any,
    recipient_user_id: str,
    status: ReportApprovalAssignmentStatus,
    artifact_digest_lines: tuple[str, ...] = (),
) -> Notification:
    body_lines = [
        "A report approval assignment has been updated.",
        "",
        f"Report: {report['id']}",
        f"Report type: {report['report_type']}",
        f"Assignment: {assignment_id}",
        f"Status: {status.value}",
    ]
    if artifact_digest_lines:
        body_lines.extend(["", "Report digest:", *artifact_digest_lines])
    body_lines.extend(
        [
            "",
            "Sign in to the RDS dashboard to review the current approval trail.",
        ]
    )
    return _report_approval_assignment_notification(
        assignment_id=assignment_id,
        organization_id=organization_id,
        report=report,
        recipient_user_id=recipient_user_id,
        event_type="report_approval_updated",
        subject=f"ISO 50001 report approval {status.value}: {report['report_type']}",
        body_lines=body_lines,
    )


def _require_report_approver(
    assigned_to_user_id: str,
    organization_id: str,
    store: SaaSStore,
) -> None:
    assignee = store.get_user(assigned_to_user_id)
    if assignee is None:
        raise ValueError(f"Assigned approver {assigned_to_user_id} was not found")
    if not bool(assignee["is_active"]):
        raise ValueError(f"Assigned approver {assigned_to_user_id} is inactive")
    require_permission(
        store.list_memberships(user_id=assigned_to_user_id, organization_id=organization_id),
        user_id=assigned_to_user_id,
        organization_id=organization_id,
        action=Action.MANAGE_REPORTS,
    )


def list_report_approver_summaries(
    user_id: str,
    organization_id: str,
    store: SaaSStore,
) -> list[ReportApproverSummary]:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.MANAGE_REPORTS,
    )
    return [
        ReportApproverSummary(
            user_id=row["user_id"],
            organization_id=row["organization_id"],
            role=Role(row["role"]),
            email=row["email"],
            display_name=row["display_name"],
        )
        for row in store.list_organization_members(organization_id)
        if bool(row["is_active"]) and Role(row["role"]) in {Role.OWNER, Role.MANAGER}
    ]


def list_report_review_summaries(
    user_id: str,
    organization_id: str,
    report_id: str,
    store: SaaSStore,
) -> list[ReportReviewSummary]:
    _visible_report_for_action(
        user_id,
        organization_id,
        report_id,
        store,
        Action.READ,
    )
    return [
        create_report_review_summary_from_row(review)
        for review in store.list_report_reviews(organization_id, report_id)
    ]


def create_report_review_summary(
    user_id: str,
    organization_id: str,
    report_id: str,
    payload: ReportReviewCreate,
    store: SaaSStore,
) -> ReportReviewSummary:
    report = _visible_report_for_action(
        user_id,
        organization_id,
        report_id,
        store,
        Action.MANAGE_REPORTS,
    )
    notes = _normalize_report_review_notes(payload.notes)
    review_id = f"report_review_{uuid4().hex}"

    with store.conn:
        store.create_report_review(
            ReportReview(
                id=review_id,
                organization_id=organization_id,
                report_id=report["id"],
                reviewer_user_id=user_id,
                status=payload.status,
                notes=notes,
            )
        )
        store.create_audit_event(
            AuditEvent(
                id=f"audit_report_review_{uuid4().hex}",
                organization_id=organization_id,
                actor_user_id=user_id,
                action=AuditAction.REPORT_REVIEW_RECORDED,
                resource_type="report_review",
                resource_id=review_id,
                metadata={
                    "notes_length": len(notes or ""),
                    "report_id": report["id"],
                    "report_type": report["report_type"],
                    "review_status": payload.status.value,
                },
            )
        )

    review = store.get_report_review(review_id)
    if review is None:
        raise ValueError(f"Report review {review_id} was not found")
    return create_report_review_summary_from_row(review)


def list_report_approval_assignment_summaries(
    user_id: str,
    organization_id: str,
    report_id: str,
    store: SaaSStore,
) -> list[ReportApprovalAssignmentSummary]:
    _visible_report_for_action(
        user_id,
        organization_id,
        report_id,
        store,
        Action.READ,
    )
    return [
        create_report_approval_assignment_summary_from_row(assignment)
        for assignment in store.list_report_approval_assignments(organization_id, report_id)
    ]


def create_report_approval_assignment_summary(
    user_id: str,
    organization_id: str,
    report_id: str,
    payload: ReportApprovalAssignmentCreate,
    store: SaaSStore,
    artifact_store: TenantArtifactStore | None = None,
) -> ReportApprovalAssignmentSummary:
    report = _visible_report_for_action(
        user_id,
        organization_id,
        report_id,
        store,
        Action.MANAGE_REPORTS,
    )
    assigned_to_user_id = payload.assigned_to_user_id.strip()
    if not assigned_to_user_id:
        raise ValueError("Assigned approver is required")
    _require_report_approver(assigned_to_user_id, organization_id, store)
    due_at = _parse_report_approval_due_at(payload.due_at)
    notes = _normalize_report_approval_assignment_notes(payload.notes)
    now = datetime.now(timezone.utc)
    assignment_id = f"report_approval_{uuid4().hex}"
    artifact_digest_lines = _report_artifact_digest_for_notification(
        organization_id,
        report,
        artifact_store,
    )

    with store.conn:
        store.create_report_approval_assignment(
            ReportApprovalAssignment(
                id=assignment_id,
                organization_id=organization_id,
                report_id=report["id"],
                assigned_to_user_id=assigned_to_user_id,
                assigned_by_user_id=user_id,
                due_at=due_at,
                notes=notes,
                created_at=now,
                updated_at=now,
            )
        )
        store.create_notification(
            _report_approval_assigned_notification(
                assignment_id=assignment_id,
                organization_id=organization_id,
                report=report,
                recipient_user_id=assigned_to_user_id,
                due_at=due_at,
                artifact_digest_lines=artifact_digest_lines,
            )
        )
        store.create_audit_event(
            AuditEvent(
                id=f"audit_report_approval_{uuid4().hex}",
                organization_id=organization_id,
                actor_user_id=user_id,
                action=AuditAction.REPORT_APPROVAL_ASSIGNED,
                resource_type="report_approval_assignment",
                resource_id=assignment_id,
                metadata={
                    "assigned_to_user_id": assigned_to_user_id,
                    "due_at": due_at.isoformat() if due_at else None,
                    "notes_length": len(notes or ""),
                    "report_id": report["id"],
                    "report_type": report["report_type"],
                    "status": ReportApprovalAssignmentStatus.PENDING.value,
                },
            )
        )

    assignment = store.get_report_approval_assignment(assignment_id)
    if assignment is None:
        raise ValueError(f"Report approval assignment {assignment_id} was not found")
    return create_report_approval_assignment_summary_from_row(assignment)


def update_report_approval_assignment_summary(
    user_id: str,
    organization_id: str,
    report_id: str,
    assignment_id: str,
    payload: ReportApprovalAssignmentUpdate,
    store: SaaSStore,
    artifact_store: TenantArtifactStore | None = None,
) -> ReportApprovalAssignmentSummary:
    report = _visible_report_for_action(
        user_id,
        organization_id,
        report_id,
        store,
        Action.MANAGE_REPORTS,
    )
    assignment = store.get_report_approval_assignment(assignment_id)
    if assignment is None:
        raise ValueError(f"Report approval assignment {assignment_id} was not found")
    if assignment["organization_id"] != organization_id or assignment["report_id"] != report["id"]:
        raise AccessDenied("Report approval assignment belongs to a different tenant or report")
    if assignment["status"] != ReportApprovalAssignmentStatus.PENDING.value:
        raise ValueError("Report approval assignment is already complete")
    if payload.status == ReportApprovalAssignmentStatus.PENDING:
        raise ValueError("Report approval assignment updates must complete or cancel the assignment")

    notes = (
        assignment["notes"]
        if payload.notes is None
        else _normalize_report_approval_assignment_notes(payload.notes)
    )
    now = datetime.now(timezone.utc)
    completed_at = now.isoformat()
    review_status_by_assignment_status = {
        ReportApprovalAssignmentStatus.APPROVED: ReportReviewStatus.APPROVED,
        ReportApprovalAssignmentStatus.CHANGES_REQUESTED: ReportReviewStatus.CHANGES_REQUESTED,
    }
    review_status = review_status_by_assignment_status.get(payload.status)
    review_id = f"report_review_{uuid4().hex}" if review_status else None
    artifact_digest_lines = _report_artifact_digest_for_notification(
        organization_id,
        report,
        artifact_store,
    )

    with store.conn:
        store.update_report_approval_assignment(
            assignment_id,
            status=payload.status,
            notes=notes,
            completed_at=completed_at,
            updated_at=now.isoformat(),
        )
        if review_status is not None and review_id is not None:
            store.create_report_review(
                ReportReview(
                    id=review_id,
                    organization_id=organization_id,
                    report_id=report["id"],
                    reviewer_user_id=user_id,
                    status=review_status,
                    notes=notes,
                    created_at=now,
                )
            )
            store.create_audit_event(
                AuditEvent(
                    id=f"audit_report_review_{uuid4().hex}",
                    organization_id=organization_id,
                    actor_user_id=user_id,
                    action=AuditAction.REPORT_REVIEW_RECORDED,
                    resource_type="report_review",
                    resource_id=review_id,
                    metadata={
                        "assignment_id": assignment_id,
                        "notes_length": len(notes or ""),
                        "report_id": report["id"],
                        "report_type": report["report_type"],
                        "review_status": review_status.value,
                    },
                )
            )
        notification_recipient_user_id = (
            assignment["assigned_by_user_id"]
            if user_id != assignment["assigned_by_user_id"]
            else assignment["assigned_to_user_id"]
        )
        store.create_notification(
            _report_approval_updated_notification(
                assignment_id=assignment_id,
                organization_id=organization_id,
                report=report,
                recipient_user_id=notification_recipient_user_id,
                status=payload.status,
                artifact_digest_lines=artifact_digest_lines,
            )
        )
        store.create_audit_event(
            AuditEvent(
                id=f"audit_report_approval_{uuid4().hex}",
                organization_id=organization_id,
                actor_user_id=user_id,
                action=AuditAction.REPORT_APPROVAL_UPDATED,
                resource_type="report_approval_assignment",
                resource_id=assignment_id,
                metadata={
                    "assigned_to_user_id": assignment["assigned_to_user_id"],
                    "completed_at": completed_at,
                    "notes_length": len(notes or ""),
                    "previous_status": assignment["status"],
                    "report_id": report["id"],
                    "report_type": report["report_type"],
                    "review_id": review_id,
                    "status": payload.status.value,
                },
            )
        )

    if payload.status == ReportApprovalAssignmentStatus.APPROVED:
        _maybe_create_approval_linked_evidence_package_snapshot(
            user_id=user_id,
            organization_id=organization_id,
            report=report,
            assignment_id=assignment_id,
            store=store,
            artifact_store=artifact_store,
        )

    updated = store.get_report_approval_assignment(assignment_id)
    if updated is None:
        raise ValueError(f"Report approval assignment {assignment_id} was not found")
    return create_report_approval_assignment_summary_from_row(updated)


def create_report_schedule_summary_from_row(schedule: Any) -> ReportScheduleSummary:
    return ReportScheduleSummary(
        id=schedule["id"],
        organization_id=schedule["organization_id"],
        report_id=schedule["report_id"],
        created_by_user_id=schedule["created_by_user_id"],
        frequency=ReportScheduleFrequency(schedule["frequency"]),
        recipient_email=schedule["recipient_email"],
        next_run_at=schedule["next_run_at"],
        is_active=bool(schedule["is_active"]),
        last_delivery_at=schedule["last_delivery_at"],
        last_delivery_status=(
            ReportScheduleDeliveryStatus(schedule["last_delivery_status"])
            if schedule["last_delivery_status"]
            else None
        ),
        last_delivery_error=schedule["last_delivery_error"],
        created_at=schedule["created_at"],
        updated_at=schedule["updated_at"],
    )


def create_report_schedule_delivery_summary_from_row(
    delivery: Any,
) -> ReportScheduleDeliverySummary:
    return ReportScheduleDeliverySummary(
        id=delivery["id"],
        organization_id=delivery["organization_id"],
        schedule_id=delivery["schedule_id"],
        report_id=delivery["report_id"],
        recipient_email=delivery["recipient_email"],
        status=ReportScheduleDeliveryStatus(delivery["status"]),
        error_message=delivery["error_message"],
        attempted_at=delivery["attempted_at"],
        sent_at=delivery["sent_at"],
        created_at=delivery["created_at"],
    )


def _normalize_schedule_recipient_email(value: str) -> str:
    email = value.strip().lower()
    if not email or "@" not in email or email.startswith("@") or email.endswith("@"):
        raise ValueError("Report schedule recipient email must be valid")
    return email


def _parse_schedule_datetime(value: str, *, require_future: bool = True) -> datetime:
    raw_value = value.strip()
    if not raw_value:
        raise ValueError("Report schedule next run time is required")
    try:
        parsed = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Report schedule next run time must be ISO-8601") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc)
    if require_future and parsed <= datetime.now(timezone.utc):
        raise ValueError("Report schedule next run time must be in the future")
    return parsed


def list_report_schedule_summaries(
    user_id: str,
    organization_id: str,
    report_id: str,
    store: SaaSStore,
) -> list[ReportScheduleSummary]:
    _visible_report_for_action(
        user_id,
        organization_id,
        report_id,
        store,
        Action.READ,
    )
    return [
        create_report_schedule_summary_from_row(schedule)
        for schedule in store.list_report_schedules(organization_id, report_id)
    ]


def list_report_schedule_delivery_summaries(
    user_id: str,
    organization_id: str,
    report_id: str,
    schedule_id: str,
    store: SaaSStore,
) -> list[ReportScheduleDeliverySummary]:
    report = _visible_report_for_action(
        user_id,
        organization_id,
        report_id,
        store,
        Action.READ,
    )
    schedule = store.get_report_schedule(schedule_id)
    if schedule is None:
        raise ValueError(f"Report schedule {schedule_id} was not found")
    if schedule["organization_id"] != organization_id or schedule["report_id"] != report["id"]:
        raise AccessDenied("Report schedule belongs to a different tenant or report")
    return [
        create_report_schedule_delivery_summary_from_row(delivery)
        for delivery in store.list_report_schedule_deliveries(organization_id, schedule_id)
    ]


def create_report_schedule_summary(
    user_id: str,
    organization_id: str,
    report_id: str,
    payload: ReportScheduleCreate,
    store: SaaSStore,
) -> ReportScheduleSummary:
    report = _visible_report_for_action(
        user_id,
        organization_id,
        report_id,
        store,
        Action.MANAGE_REPORTS,
    )
    recipient_email = _normalize_schedule_recipient_email(payload.recipient_email)
    next_run_at = _parse_schedule_datetime(payload.next_run_at)
    schedule_id = f"report_schedule_{uuid4().hex}"

    with store.conn:
        store.create_report_schedule(
            ReportSchedule(
                id=schedule_id,
                organization_id=organization_id,
                report_id=report["id"],
                created_by_user_id=user_id,
                frequency=payload.frequency,
                recipient_email=recipient_email,
                next_run_at=next_run_at,
            )
        )
        store.create_audit_event(
            AuditEvent(
                id=f"audit_report_schedule_{uuid4().hex}",
                organization_id=organization_id,
                actor_user_id=user_id,
                action=AuditAction.REPORT_SCHEDULE_CREATED,
                resource_type="report_schedule",
                resource_id=schedule_id,
                metadata={
                    "frequency": payload.frequency.value,
                    "next_run_at": next_run_at.isoformat(),
                    "recipient_email_configured": True,
                    "report_id": report["id"],
                    "report_type": report["report_type"],
                },
            )
        )

    schedule = store.get_report_schedule(schedule_id)
    if schedule is None:
        raise ValueError(f"Report schedule {schedule_id} was not found")
    return create_report_schedule_summary_from_row(schedule)


def update_report_schedule_summary(
    user_id: str,
    organization_id: str,
    report_id: str,
    schedule_id: str,
    payload: ReportScheduleUpdate,
    store: SaaSStore,
) -> ReportScheduleSummary:
    report = _visible_report_for_action(
        user_id,
        organization_id,
        report_id,
        store,
        Action.MANAGE_REPORTS,
    )
    schedule = store.get_report_schedule(schedule_id)
    if schedule is None:
        raise ValueError(f"Report schedule {schedule_id} was not found")
    if schedule["organization_id"] != organization_id or schedule["report_id"] != report["id"]:
        raise AccessDenied("Report schedule belongs to a different tenant or report")

    next_is_active = bool(schedule["is_active"]) if payload.is_active is None else payload.is_active
    next_frequency = payload.frequency or ReportScheduleFrequency(schedule["frequency"])
    next_email = (
        schedule["recipient_email"]
        if payload.recipient_email is None
        else _normalize_schedule_recipient_email(payload.recipient_email)
    )
    next_run_at = (
        schedule["next_run_at"]
        if payload.next_run_at is None
        else _parse_schedule_datetime(payload.next_run_at, require_future=next_is_active).isoformat()
    )
    updated_at = datetime.now(timezone.utc).isoformat()

    with store.conn:
        store.update_report_schedule(
            schedule_id,
            frequency=next_frequency.value,
            recipient_email=next_email,
            next_run_at=next_run_at,
            is_active=next_is_active,
            updated_at=updated_at,
        )
        store.create_audit_event(
            AuditEvent(
                id=f"audit_report_schedule_{uuid4().hex}",
                organization_id=organization_id,
                actor_user_id=user_id,
                action=AuditAction.REPORT_SCHEDULE_UPDATED,
                resource_type="report_schedule",
                resource_id=schedule_id,
                metadata={
                    "frequency": next_frequency.value,
                    "is_active": next_is_active,
                    "next_run_at": next_run_at,
                    "previous_frequency": schedule["frequency"],
                    "previous_is_active": bool(schedule["is_active"]),
                    "previous_next_run_at": schedule["next_run_at"],
                    "recipient_email_configured": bool(next_email),
                    "report_id": report["id"],
                    "report_type": report["report_type"],
                },
            )
        )

    updated = store.get_report_schedule(schedule_id)
    if updated is None:
        raise ValueError(f"Report schedule {schedule_id} was not found")
    return create_report_schedule_summary_from_row(updated)


def get_report_artifact(
    user_id: str,
    organization_id: str,
    report_id: str,
    store: SaaSStore,
    artifact_store: TenantArtifactStore,
) -> ReportArtifactResponse:
    report = _visible_report_for_action(
        user_id,
        organization_id,
        report_id,
        store,
        Action.READ,
    )
    content = _load_report_artifact_content(organization_id, report, artifact_store)

    with store.conn:
        store.create_audit_event(
            AuditEvent(
                id=f"audit_report_access_{uuid4().hex}",
                organization_id=organization_id,
                actor_user_id=user_id,
                action=AuditAction.REPORT_ACCESSED,
                resource_type="report",
                resource_id=report["id"],
                metadata={
                    "artifact_location_configured": "true",
                    "report_type": report["report_type"],
                },
            )
        )

    return ReportArtifactResponse(
        report_id=report["id"],
        organization_id=report["organization_id"],
        content=content,
    )


def export_report_artifact(
    user_id: str,
    organization_id: str,
    report_id: str,
    store: SaaSStore,
    artifact_store: TenantArtifactStore,
) -> PlainTextResponse:
    report = _visible_report_for_action(
        user_id,
        organization_id,
        report_id,
        store,
        Action.READ,
    )
    content = _load_report_artifact_content(organization_id, report, artifact_store)
    export_content = json.dumps(content, indent=2, sort_keys=True) + "\n"
    filename = f"{safe_key_part(report['id'])}-{safe_key_part(report['report_type'])}.json"

    with store.conn:
        store.create_audit_event(
            AuditEvent(
                id=f"audit_report_export_{uuid4().hex}",
                organization_id=organization_id,
                actor_user_id=user_id,
                action=AuditAction.REPORT_EXPORTED,
                resource_type="report",
                resource_id=report["id"],
                metadata={
                    "content_length": len(export_content),
                    "format": "json",
                    "report_id": report["id"],
                    "report_type": report["report_type"],
                },
            )
        )

    return PlainTextResponse(
        export_content,
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


def _site_evidence(site: Site | None) -> dict[str, str] | None:
    if site is None:
        return None
    return {
        "id": site.id,
        "name": site.name,
        "timezone": site.timezone,
    }


def _comment_evidence(comment: Any) -> dict[str, Any]:
    return {
        "id": comment["id"],
        "author_user_id": comment["author_user_id"],
        "body": comment["body"],
        "created_at": comment["created_at"],
    }


def _review_evidence(review: Any) -> dict[str, Any]:
    return {
        "id": review["id"],
        "reviewer_user_id": review["reviewer_user_id"],
        "status": review["status"],
        "notes": review["notes"],
        "created_at": review["created_at"],
    }


def _approval_assignment_evidence(assignment: Any) -> dict[str, Any]:
    return {
        "id": assignment["id"],
        "assigned_to_user_id": assignment["assigned_to_user_id"],
        "assigned_by_user_id": assignment["assigned_by_user_id"],
        "status": assignment["status"],
        "due_at": assignment["due_at"],
        "notes": assignment["notes"],
        "completed_at": assignment["completed_at"],
        "created_at": assignment["created_at"],
        "updated_at": assignment["updated_at"],
    }


def _schedule_evidence(store: SaaSStore, organization_id: str, schedule: Any) -> dict[str, Any]:
    deliveries = store.list_report_schedule_deliveries(organization_id, schedule["id"])
    return {
        "id": schedule["id"],
        "created_by_user_id": schedule["created_by_user_id"],
        "frequency": schedule["frequency"],
        "recipient_email": schedule["recipient_email"],
        "next_run_at": schedule["next_run_at"],
        "is_active": bool(schedule["is_active"]),
        "last_delivery_at": schedule["last_delivery_at"],
        "last_delivery_status": schedule["last_delivery_status"],
        "last_delivery_error": schedule["last_delivery_error"],
        "created_at": schedule["created_at"],
        "updated_at": schedule["updated_at"],
        "deliveries": [
            {
                "id": delivery["id"],
                "recipient_email": delivery["recipient_email"],
                "status": delivery["status"],
                "error_message": delivery["error_message"],
                "attempted_at": delivery["attempted_at"],
                "sent_at": delivery["sent_at"],
                "created_at": delivery["created_at"],
            }
            for delivery in deliveries
        ],
    }


def _report_branding_evidence(organization_id: str, row: Any | None) -> dict[str, Any]:
    branding = create_report_branding_summary(organization_id, row)
    return {
        "evidence_pack_title": branding.evidence_pack_title,
        "primary_color": branding.primary_color,
        "logo_url": branding.logo_url,
        "support_email": branding.support_email,
    }


def create_report_evidence_package_summary_from_row(row: Any) -> ReportEvidencePackageSummary:
    return ReportEvidencePackageSummary(
        id=row["id"],
        organization_id=row["organization_id"],
        report_id=row["report_id"],
        created_by_user_id=row["created_by_user_id"],
        filename=row["filename"],
        content_type=row["content_type"],
        content_length=row["content_length"],
        package_type=row["package_type"],
        format=row["format"],
        comment_count=row["comment_count"],
        review_count=row["review_count"],
        schedule_count=row["schedule_count"],
        delivery_count=row["delivery_count"],
        upload_count=row["upload_count"],
        created_at=row["created_at"],
    )


def _report_evidence_package_filename(report: Any) -> str:
    return (
        f"{safe_key_part(report['id'])}-{safe_key_part(report['report_type'])}"
        "-evidence-package.json"
    )


def _report_evidence_package_pdf_filename(report: Any) -> str:
    return (
        f"{safe_key_part(report['id'])}-{safe_key_part(report['report_type'])}"
        "-evidence-package.pdf"
    )


def _evidence_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _evidence_list(package: dict[str, Any], key: str) -> list[Any]:
    value = package.get(key)
    return value if isinstance(value, list) else []


def _template_section_enabled(package: dict[str, Any], key: str) -> bool:
    template = _evidence_dict(package.get("template"))
    value = template.get(key)
    return value if isinstance(value, bool) else True


def _template_section_labels(package: dict[str, Any]) -> dict[str, str]:
    template = _evidence_dict(package.get("template"))
    labels = template.get("section_labels")
    if isinstance(labels, dict):
        return _normalize_evidence_package_section_labels(
            {str(key): str(label) for key, label in labels.items()}
        )
    return dict(DEFAULT_EVIDENCE_PACKAGE_SECTION_LABELS)


def _template_section_order(package: dict[str, Any]) -> list[str]:
    template = _evidence_dict(package.get("template"))
    order = template.get("section_order")
    if isinstance(order, list):
        return _normalize_evidence_package_section_order([str(key) for key in order])
    return list(DEFAULT_EVIDENCE_PACKAGE_SECTION_ORDER)


def _template_section_label(package: dict[str, Any], section_key: str) -> str:
    return _template_section_labels(package).get(
        section_key,
        DEFAULT_EVIDENCE_PACKAGE_SECTION_LABELS[section_key],
    )


def _template_section_enabled_by_key(package: dict[str, Any], section_key: str) -> bool:
    include_field = EVIDENCE_PACKAGE_SECTION_INCLUDE_FIELDS[section_key]
    return _template_section_enabled(package, include_field)


def _pdf_text(value: Any, max_length: int = 240) -> str:
    if value is None:
        return "Not recorded"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (dict, list)):
        text = json.dumps(value, sort_keys=True)
    else:
        text = str(value)
    text = " ".join(text.split())
    if not text:
        return "Not recorded"
    if len(text) > max_length:
        return f"{text[: max_length - 3]}..."
    return text


def _status_summary(items: list[Any], key: str = "status") -> str:
    counts: dict[str, int] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        status_value = _pdf_text(item.get(key), max_length=60)
        counts[status_value] = counts.get(status_value, 0) + 1
    if not counts:
        return "None"
    return ", ".join(f"{status_value}: {count}" for status_value, count in sorted(counts.items()))


def _artifact_summary_lines(artifact: dict[str, Any]) -> list[str]:
    if not artifact:
        return ["No artifact fields were available."]

    lines: list[str] = []
    for index, (key, value) in enumerate(sorted(artifact.items()), start=1):
        if index > 30:
            lines.append(f"Additional fields: {len(artifact) - 30}")
            break
        if isinstance(value, dict):
            lines.append(f"{key}: {len(value)} nested fields")
        elif isinstance(value, list):
            lines.append(f"{key}: {len(value)} items")
        else:
            lines.append(f"{key}: {_pdf_text(value)}")
    return lines


def _weather_normalization_evidence_lines(artifact: dict[str, Any]) -> list[str]:
    legacy_baseline = _evidence_dict(artifact.get("legacy_baseline"))
    weather_normalization = _evidence_dict(legacy_baseline.get("weather_normalization"))
    if not weather_normalization:
        return ["Weather-normalized model summary: Not recorded"]

    models = weather_normalization.get("models")
    model_rows = models if isinstance(models, list) else []
    skipped_meters = weather_normalization.get("skipped_meters")
    skipped_rows = skipped_meters if isinstance(skipped_meters, list) else []
    lines = [
        f"Status: {_pdf_text(weather_normalization.get('status'))}",
        f"Baseline year: {_pdf_text(weather_normalization.get('baseline_year'))}",
        f"Comparison year: {_pdf_text(weather_normalization.get('comparison_year'))}",
        f"Modeled meters: {_pdf_text(weather_normalization.get('model_count'))}",
        f"Skipped meters: {_pdf_text(weather_normalization.get('skipped_meter_count'))}",
        f"Estimated savings: {_pdf_text(weather_normalization.get('estimated_savings'))}",
    ]

    if model_rows:
        for index, model in enumerate(model_rows[:12], start=1):
            model_dict = _evidence_dict(model)
            lines.append(
                "Model "
                f"{index}: {_pdf_text(model_dict.get('meter'))} | "
                f"SEU {_pdf_text(model_dict.get('seu_category'))} | "
                f"{_pdf_text(model_dict.get('normalization'))} | "
                f"actual {_pdf_text(model_dict.get('actual_consumption'))} | "
                f"predicted {_pdf_text(model_dict.get('predicted_consumption'))} | "
                f"savings {_pdf_text(model_dict.get('estimated_savings'))} | "
                f"R2 {_pdf_text(model_dict.get('r2'))}"
            )
        if len(model_rows) > 12:
            lines.append(f"Additional modeled meters: {len(model_rows) - 12}")
    else:
        lines.append("Models: None fitted")

    if skipped_rows:
        for index, skipped in enumerate(skipped_rows[:12], start=1):
            skipped_dict = _evidence_dict(skipped)
            lines.append(
                "Skipped "
                f"{index}: {_pdf_text(skipped_dict.get('meter'))} | "
                f"SEU {_pdf_text(skipped_dict.get('seu_category'))} | "
                f"utility {_pdf_text(skipped_dict.get('utility'))} | "
                f"reason {_pdf_text(skipped_dict.get('reason'))}"
            )
        if len(skipped_rows) > 12:
            lines.append(f"Additional skipped meters: {len(skipped_rows) - 12}")
    else:
        lines.append("Skipped meters: None")

    return lines


def _report_evidence_package_pdf_sections(
    package: dict[str, Any],
    counts: dict[str, int],
) -> list[tuple[str, list[str]]]:
    organization = _evidence_dict(package.get("organization"))
    branding = _evidence_dict(package.get("branding"))
    template = _evidence_dict(package.get("template"))
    report = _evidence_dict(package.get("report"))
    run = _evidence_dict(package.get("run"))
    site = _evidence_dict(package.get("site"))
    artifact = _evidence_dict(package.get("artifact"))
    uploads = _evidence_list(package, "uploads")
    assignments = _evidence_list(package, "approval_assignments")
    reviews = _evidence_list(package, "reviews")
    comments = _evidence_list(package, "comments")
    schedules = _evidence_list(package, "schedules")
    section_labels = _template_section_labels(package)
    section_order = _template_section_order(package)
    include_artifact = _template_section_enabled_by_key(package, "report_artifact")
    evidence_pack_title = _pdf_text(
        branding.get("evidence_pack_title") or DEFAULT_EVIDENCE_PACK_TITLE,
        max_length=120,
    )
    included_sections = [
        section_labels[section_key]
        for section_key in section_order
        if _template_section_enabled_by_key(package, section_key)
    ]

    summary = [
        evidence_pack_title,
        f"Template: {_pdf_text(template.get('template_name') or DEFAULT_EVIDENCE_PACKAGE_TEMPLATE_NAME)}",
        f"Included sections: {_pdf_text(', '.join(included_sections) or 'None')}",
        (
            "Organization: "
            f"{_pdf_text(organization.get('name') or package.get('organization_id'))} "
            f"({_pdf_text(organization.get('slug') or package.get('organization_id'))})"
        ),
        f"Brand color: {_pdf_text(branding.get('primary_color') or DEFAULT_EVIDENCE_PACK_PRIMARY_COLOR)}",
        f"Brand logo URL: {_pdf_text(branding.get('logo_url'))}",
        f"Support contact: {_pdf_text(branding.get('support_email'))}",
        f"Generated at: {_pdf_text(package.get('generated_at'))}",
        f"Package type: {_pdf_text(package.get('package_type'))}",
        f"Report: {_pdf_text(report.get('id'))} ({_pdf_text(report.get('report_type'))})",
        f"Site: {_pdf_text(site.get('name'))} ({_pdf_text(site.get('timezone'))})",
        f"Run: {_pdf_text(run.get('id'))} ({_pdf_text(run.get('status'))})",
        (
            "Evidence counts: "
            f"{counts['upload_count']} uploads, "
            f"{counts['review_count']} reviews, "
            f"{counts['comment_count']} comments, "
            f"{counts['schedule_count']} schedules, "
            f"{counts['delivery_count']} delivery attempts"
        ),
        f"Approval assignments: {_status_summary(assignments)}",
        f"Management reviews: {_status_summary(reviews)}",
    ]

    lineage = [
        f"Report id: {_pdf_text(report.get('id'))}",
        f"Report type: {_pdf_text(report.get('report_type'))}",
        f"Report created at: {_pdf_text(report.get('created_at'))}",
        f"Report published: {_pdf_text(report.get('is_published'))}",
        f"Report expires at: {_pdf_text(report.get('expires_at'))}",
        f"Retention applied at: {_pdf_text(report.get('retention_applied_at'))}",
        f"Run id: {_pdf_text(run.get('id'))}",
        f"Run requested by: {_pdf_text(run.get('requested_by_user_id'))}",
        f"Run status: {_pdf_text(run.get('status'))}",
        f"Run completed at: {_pdf_text(run.get('completed_at'))}",
        f"Site id: {_pdf_text(site.get('id'))}",
        f"Site name: {_pdf_text(site.get('name'))}",
    ]
    upload_lines: list[str] = []
    if uploads:
        for index, upload in enumerate(uploads[:12], start=1):
            upload_dict = _evidence_dict(upload)
            upload_lines.append(
                "Upload "
                f"{index}: {_pdf_text(upload_dict.get('id'))} | "
                f"{_pdf_text(upload_dict.get('category'))} | "
                f"{_pdf_text(upload_dict.get('status'))} | "
                f"checksum {_pdf_text(upload_dict.get('checksum'), max_length=80)}"
            )
        if len(uploads) > 12:
            upload_lines.append(f"Additional uploads: {len(uploads) - 12}")
    else:
        upload_lines.append("Uploads: None recorded")

    approval = [f"Approval assignment status: {_status_summary(assignments)}"]
    if assignments:
        for index, assignment in enumerate(assignments[:12], start=1):
            assignment_dict = _evidence_dict(assignment)
            approval.append(
                "Assignment "
                f"{index}: assigned to {_pdf_text(assignment_dict.get('assigned_to_user_id'))}; "
                f"status {_pdf_text(assignment_dict.get('status'))}; "
                f"due {_pdf_text(assignment_dict.get('due_at'))}; "
                f"completed {_pdf_text(assignment_dict.get('completed_at'))}"
            )
            if assignment_dict.get("notes"):
                approval.append(f"Assignment {index} notes: {_pdf_text(assignment_dict.get('notes'))}")
        if len(assignments) > 12:
            approval.append(f"Additional assignments: {len(assignments) - 12}")
    else:
        approval.append("Assignments: None recorded")

    approval.append(f"Review status: {_status_summary(reviews)}")
    if reviews:
        for index, review in enumerate(reviews[:12], start=1):
            review_dict = _evidence_dict(review)
            approval.append(
                "Review "
                f"{index}: {_pdf_text(review_dict.get('status'))} by "
                f"{_pdf_text(review_dict.get('reviewer_user_id'))} at "
                f"{_pdf_text(review_dict.get('created_at'))}"
            )
            if review_dict.get("notes"):
                approval.append(f"Review {index} notes: {_pdf_text(review_dict.get('notes'))}")
        if len(reviews) > 12:
            approval.append(f"Additional reviews: {len(reviews) - 12}")
    else:
        approval.append("Reviews: None recorded")

    collaboration = []
    if comments:
        for index, comment in enumerate(comments[:20], start=1):
            comment_dict = _evidence_dict(comment)
            collaboration.append(
                "Comment "
                f"{index}: {_pdf_text(comment_dict.get('author_user_id'))} at "
                f"{_pdf_text(comment_dict.get('created_at'))}"
            )
            collaboration.append(f"Comment {index} body: {_pdf_text(comment_dict.get('body'))}")
        if len(comments) > 20:
            collaboration.append(f"Additional comments: {len(comments) - 20}")
    else:
        collaboration.append("Comments: None recorded")

    schedule_lines = []
    if schedules:
        for index, schedule in enumerate(schedules[:12], start=1):
            schedule_dict = _evidence_dict(schedule)
            deliveries = schedule_dict.get("deliveries")
            delivery_count = len(deliveries) if isinstance(deliveries, list) else 0
            schedule_lines.append(
                "Schedule "
                f"{index}: {_pdf_text(schedule_dict.get('frequency'))} to "
                f"{_pdf_text(schedule_dict.get('recipient_email'))}; "
                f"active {_pdf_text(schedule_dict.get('is_active'))}; "
                f"next {_pdf_text(schedule_dict.get('next_run_at'))}; "
                f"last {_pdf_text(schedule_dict.get('last_delivery_status'))}"
            )
            schedule_lines.append(f"Schedule {index} delivery attempts: {delivery_count}")
        if len(schedules) > 12:
            schedule_lines.append(f"Additional schedules: {len(schedules) - 12}")
    else:
        schedule_lines.append("Schedules: None recorded")

    sections = [
        ("Evidence Package Summary", summary),
        ("Report Lineage", lineage),
    ]
    approval_section_added = False
    for section_key in section_order:
        if not _template_section_enabled_by_key(package, section_key):
            continue
        if section_key == "uploads":
            sections.append((section_labels[section_key], upload_lines))
        elif section_key in {"approval_assignments", "reviews"}:
            include_assignments = _template_section_enabled_by_key(
                package,
                "approval_assignments",
            )
            include_reviews = _template_section_enabled_by_key(package, "reviews")
            if approval_section_added or not (include_assignments or include_reviews):
                continue
            default_approval_labels = (
                section_labels["approval_assignments"]
                == DEFAULT_EVIDENCE_PACKAGE_SECTION_LABELS["approval_assignments"]
                and section_labels["reviews"]
                == DEFAULT_EVIDENCE_PACKAGE_SECTION_LABELS["reviews"]
            )
            if default_approval_labels:
                approval_title = "Approval And Review Evidence"
            elif include_assignments and include_reviews:
                approval_title = (
                    f"{section_labels['approval_assignments']} / "
                    f"{section_labels['reviews']}"
                )
            elif include_assignments:
                approval_title = section_labels["approval_assignments"]
            else:
                approval_title = section_labels["reviews"]
            sections.append((approval_title, approval))
            approval_section_added = True
        elif section_key == "comments":
            sections.append((section_labels[section_key], collaboration))
        elif section_key == "schedules":
            sections.append((section_labels[section_key], schedule_lines))
        elif section_key == "report_artifact" and include_artifact:
            if (
                section_labels[section_key]
                == DEFAULT_EVIDENCE_PACKAGE_SECTION_LABELS[section_key]
            ):
                sections.extend(
                    [
                        (
                            "Weather Normalized Evidence",
                            _weather_normalization_evidence_lines(artifact),
                        ),
                        ("Report Artifact Summary", _artifact_summary_lines(artifact)),
                    ]
                )
            else:
                sections.extend(
                    [
                        (
                            f"{section_labels[section_key]} - Weather Normalized Evidence",
                            _weather_normalization_evidence_lines(artifact),
                        ),
                        (
                            f"{section_labels[section_key]} - Summary",
                            _artifact_summary_lines(artifact),
                        ),
                    ]
                )
    return sections


def _render_report_evidence_package_pdf(
    package: dict[str, Any],
    counts: dict[str, int],
) -> bytes:
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure

    buffer = BytesIO()
    sections = _report_evidence_package_pdf_sections(package, counts)
    organization = _evidence_dict(package.get("organization"))
    branding = _evidence_dict(package.get("branding"))
    organization_name = _pdf_text(organization.get("name") or package.get("organization_id"), max_length=90)
    evidence_pack_title = _pdf_text(
        branding.get("evidence_pack_title") or DEFAULT_EVIDENCE_PACK_TITLE,
        max_length=90,
    )
    primary_color = _pdf_text(
        branding.get("primary_color") or DEFAULT_EVIDENCE_PACK_PRIMARY_COLOR,
        max_length=7,
    )
    subtitle = f"{organization_name} | Powered by RDS Energy Analytics | {evidence_pack_title}"
    with PdfPages(buffer) as pdf:
        metadata = pdf.infodict()
        metadata["Title"] = f"{evidence_pack_title} - {organization_name}"
        metadata["Author"] = f"RDS Energy Analytics for {organization_name}"
        metadata["Subject"] = "ISO 50001 report evidence package"
        metadata["CreationDate"] = datetime.now(timezone.utc).replace(tzinfo=None)

        page_number = 1
        for title, lines in sections:
            wrapped_lines: list[str] = []
            for line in lines:
                line_text = _pdf_text(line, max_length=500)
                wrapped = wrap(
                    line_text,
                    width=96,
                    break_long_words=True,
                    break_on_hyphens=False,
                )
                wrapped_lines.extend(wrapped or [""])

            for page_index, start in enumerate(range(0, max(len(wrapped_lines), 1), 42), start=1):
                fig = Figure(figsize=(8.27, 11.69), facecolor="white")
                axis = fig.add_axes([0, 0, 1, 1])
                axis.axis("off")
                axis.set_xlim(0, 1)
                axis.set_ylim(0, 1)
                page_title = title if page_index == 1 else f"{title} (continued)"
                axis.text(
                    0.08,
                    0.95,
                    page_title,
                    fontsize=16,
                    fontweight="bold",
                    color="#0f172a",
                    transform=axis.transAxes,
                )
                axis.text(
                    0.08,
                    0.92,
                    subtitle,
                    fontsize=9,
                    color="#475569",
                    transform=axis.transAxes,
                )
                axis.plot(
                    [0.08, 0.92],
                    [0.895, 0.895],
                    color=primary_color,
                    linewidth=1,
                    transform=axis.transAxes,
                )

                y = 0.86
                for line in wrapped_lines[start : start + 42] or ["No evidence recorded."]:
                    axis.text(
                        0.08,
                        y,
                        line,
                        fontsize=9,
                        color="#111827",
                        va="top",
                        transform=axis.transAxes,
                    )
                    y -= 0.0185

                axis.text(
                    0.08,
                    0.06,
                    f"Generated at {_pdf_text(package.get('generated_at'))}",
                    fontsize=8,
                    color="#64748b",
                    transform=axis.transAxes,
                )
                axis.text(
                    0.88,
                    0.06,
                    f"Page {page_number}",
                    fontsize=8,
                    color="#64748b",
                    transform=axis.transAxes,
                )
                pdf.savefig(fig)
                page_number += 1
                fig.clear()

    return buffer.getvalue()


def _build_report_evidence_package(
    organization_id: str,
    report: Any,
    store: SaaSStore,
    artifact_store: TenantArtifactStore,
) -> tuple[str, str, dict[str, int]]:
    artifact_content = _load_report_artifact_content(organization_id, report, artifact_store)
    organization = store.get_organization(organization_id)
    run = store.get_run(report["run_id"])
    site = store.get_site(run["site_id"]) if run is not None else None
    uploads = store.list_run_uploads(organization_id, report["run_id"])
    comments = store.list_report_comments(organization_id, report["id"])
    reviews = store.list_report_reviews(organization_id, report["id"])
    approval_assignments = store.list_report_approval_assignments(organization_id, report["id"])
    schedules = store.list_report_schedules(organization_id, report["id"])
    template = _evidence_package_template_from_store(organization_id, store)
    upload_evidence = [
        {
            "id": upload["id"],
            "site_id": upload["site_id"],
            "uploaded_by_user_id": upload["uploaded_by_user_id"],
            "category": upload["category"],
            "checksum": upload["checksum"],
            "status": upload["status"],
            "created_at": upload["created_at"],
        }
        for upload in uploads
    ] if template.include_uploads else []
    artifact_evidence = artifact_content if template.include_report_artifact else {}
    assignment_evidence = [
        _approval_assignment_evidence(assignment)
        for assignment in approval_assignments
    ] if template.include_approval_assignments else []
    review_evidence = [
        _review_evidence(review)
        for review in reviews
    ] if template.include_reviews else []
    comment_evidence = [
        _comment_evidence(comment)
        for comment in comments
    ] if template.include_comments else []
    schedule_evidence = [
        _schedule_evidence(store, organization_id, schedule)
        for schedule in schedules
    ] if template.include_schedules else []
    delivery_count = sum(len(schedule["deliveries"]) for schedule in schedule_evidence)
    generated_at = datetime.now(timezone.utc).isoformat()

    package = {
        "package_type": REPORT_EVIDENCE_PACKAGE_TYPE,
        "generated_at": generated_at,
        "organization_id": organization_id,
        "organization": (
            {
                "id": organization["id"],
                "name": organization["name"],
                "slug": organization["slug"],
            }
            if organization is not None
            else {
                "id": organization_id,
                "name": organization_id,
                "slug": organization_id,
            }
        ),
        "branding": _report_branding_evidence(
            organization_id,
            store.get_organization_report_branding(organization_id),
        ),
        "template": _evidence_package_template_evidence(template),
        "report": {
            "id": report["id"],
            "run_id": report["run_id"],
            "report_type": report["report_type"],
            "is_published": bool(report["is_published"]),
            "expires_at": report["expires_at"],
            "retention_applied_at": report["retention_applied_at"],
            "created_at": report["created_at"],
        },
        "run": (
            {
                "id": run["id"],
                "site_id": run["site_id"],
                "requested_by_user_id": run["requested_by_user_id"],
                "status": run["status"],
                "error_message": run["error_message"],
                "created_at": run["created_at"],
                "completed_at": run["completed_at"],
            }
            if run is not None
            else None
        ),
        "site": _site_evidence(site),
        "uploads": upload_evidence,
        "artifact": artifact_evidence,
        "approval_assignments": assignment_evidence,
        "reviews": review_evidence,
        "comments": comment_evidence,
        "schedules": schedule_evidence,
    }
    export_content = json.dumps(package, indent=2, sort_keys=True) + "\n"
    counts = {
        "comment_count": len(comment_evidence),
        "delivery_count": delivery_count,
        "review_count": len(review_evidence),
        "schedule_count": len(schedule_evidence),
        "template_enabled_section_count": _evidence_package_template_enabled_sections(template),
        "upload_count": len(upload_evidence),
    }
    return export_content, _report_evidence_package_filename(report), counts


def _build_report_evidence_package_pdf(
    organization_id: str,
    report: Any,
    store: SaaSStore,
    artifact_store: TenantArtifactStore,
) -> tuple[bytes, str, dict[str, int]]:
    export_content, _filename, counts = _build_report_evidence_package(
        organization_id,
        report,
        store,
        artifact_store,
    )
    package = json.loads(export_content)
    return (
        _render_report_evidence_package_pdf(package, counts),
        _report_evidence_package_pdf_filename(report),
        counts,
    )


def _report_evidence_package_audit_metadata(
    report: Any,
    export_content: str | bytes,
    counts: dict[str, int],
    package_format: str = "json",
) -> dict[str, Any]:
    return {
        **counts,
        "content_length": len(export_content),
        "format": package_format,
        "package_type": REPORT_EVIDENCE_PACKAGE_TYPE,
        "report_id": report["id"],
        "report_type": report["report_type"],
    }


def _approved_report_approval_assignment_count(assignments: list[Any]) -> int:
    return sum(
        1
        for assignment in assignments
        if assignment["status"] == ReportApprovalAssignmentStatus.APPROVED.value
    )


def _require_report_approved_for_evidence_lock(
    organization_id: str,
    report: Any,
    store: SaaSStore,
) -> dict[str, Any]:
    assignments = store.list_report_approval_assignments(organization_id, report["id"])
    policy = _report_approval_policy_from_store(organization_id, store)
    pending_assignment_ids = [
        assignment["id"]
        for assignment in assignments
        if assignment["status"] == ReportApprovalAssignmentStatus.PENDING.value
    ]
    if pending_assignment_ids:
        raise ValueError(
            "Pending report approval assignments must be completed before locking evidence package"
        )

    approved_assignment_count = _approved_report_approval_assignment_count(assignments)
    if approved_assignment_count < policy.minimum_approval_assignments_for_lock:
        raise ValueError(
            "Report requires at least "
            f"{policy.minimum_approval_assignments_for_lock} approved approval assignments "
            "before locking evidence package"
        )

    reviews = store.list_report_reviews(organization_id, report["id"])
    if not reviews:
        raise ValueError("Report must have an approved management review before locking evidence package")

    latest_review = reviews[0]
    if latest_review["status"] != ReportReviewStatus.APPROVED.value:
        raise ValueError("Latest management review must approve the report before locking evidence package")

    return {
        "approval_assignment_count": len(assignments),
        "approval_assignment_approved_count": approved_assignment_count,
        "approval_policy_auto_lock_evidence_on_final_approval": (
            policy.auto_lock_evidence_on_final_approval
        ),
        "approval_policy_minimum_assignments_for_lock": (
            policy.minimum_approval_assignments_for_lock
        ),
        "approval_review_id": latest_review["id"],
        "approval_status": latest_review["status"],
    }


def _create_report_evidence_package_snapshot_for_report(
    user_id: str,
    organization_id: str,
    report: Any,
    store: SaaSStore,
    artifact_store: TenantArtifactStore,
    *,
    audit_metadata: dict[str, Any] | None = None,
) -> ReportEvidencePackageSummary:
    approval_lock_metadata = _require_report_approved_for_evidence_lock(
        organization_id,
        report,
        store,
    )
    export_content, filename, counts = _build_report_evidence_package(
        organization_id,
        report,
        store,
        artifact_store,
    )
    package_id = f"report_evidence_package_{uuid4().hex}"
    storage_key = evidence_package_storage_key(
        organization_id=organization_id,
        report_id=report["id"],
        package_id=package_id,
        filename=filename,
    )
    artifact_store.put_text(organization_id, storage_key, export_content)
    package = ReportEvidencePackage(
        id=package_id,
        organization_id=organization_id,
        report_id=report["id"],
        created_by_user_id=user_id,
        storage_key=storage_key,
        filename=filename,
        content_type=REPORT_EVIDENCE_PACKAGE_CONTENT_TYPE,
        content_length=len(export_content),
        package_type=REPORT_EVIDENCE_PACKAGE_TYPE,
        format="json",
        comment_count=counts["comment_count"],
        review_count=counts["review_count"],
        schedule_count=counts["schedule_count"],
        delivery_count=counts["delivery_count"],
        upload_count=counts["upload_count"],
    )

    with store.conn:
        store.create_report_evidence_package(package)
        store.create_audit_event(
            AuditEvent(
                id=f"audit_report_evidence_package_create_{uuid4().hex}",
                organization_id=organization_id,
                actor_user_id=user_id,
                action=AuditAction.REPORT_EVIDENCE_PACKAGE_CREATED,
                resource_type="report_evidence_package",
                resource_id=package.id,
                metadata={
                    **_report_evidence_package_audit_metadata(report, export_content, counts),
                    **approval_lock_metadata,
                    **(audit_metadata or {}),
                    "package_id": package.id,
                    "storage_key_configured": True,
                },
            )
        )

    created = store.get_report_evidence_package(package.id)
    if created is None:
        raise ValueError(f"Report evidence package {package.id} was not found")
    return create_report_evidence_package_summary_from_row(created)


def _maybe_create_approval_linked_evidence_package_snapshot(
    *,
    user_id: str,
    organization_id: str,
    report: Any,
    assignment_id: str,
    store: SaaSStore,
    artifact_store: TenantArtifactStore | None,
) -> ReportEvidencePackageSummary | None:
    if artifact_store is None:
        return None
    policy = _report_approval_policy_from_store(organization_id, store)
    if not policy.auto_lock_evidence_on_final_approval:
        return None
    assignments = store.list_report_approval_assignments(organization_id, report["id"])
    if any(
        assignment["status"] == ReportApprovalAssignmentStatus.PENDING.value
        for assignment in assignments
    ):
        return None
    approved_assignment_count = _approved_report_approval_assignment_count(assignments)
    if approved_assignment_count < policy.minimum_approval_assignments_for_lock:
        return None
    reviews = store.list_report_reviews(organization_id, report["id"])
    if not reviews or reviews[0]["status"] != ReportReviewStatus.APPROVED.value:
        return None
    return _create_report_evidence_package_snapshot_for_report(
        user_id,
        organization_id,
        report,
        store,
        artifact_store,
        audit_metadata={
            "approval_assignment_id": assignment_id,
            "created_from": "approval_assignment",
            "auto_created": True,
        },
    )


def export_report_evidence_package(
    user_id: str,
    organization_id: str,
    report_id: str,
    store: SaaSStore,
    artifact_store: TenantArtifactStore,
) -> PlainTextResponse:
    report = _visible_report_for_action(
        user_id,
        organization_id,
        report_id,
        store,
        Action.READ,
    )
    export_content, filename, counts = _build_report_evidence_package(
        organization_id,
        report,
        store,
        artifact_store,
    )
    with store.conn:
        store.create_audit_event(
            AuditEvent(
                id=f"audit_report_evidence_export_{uuid4().hex}",
                organization_id=organization_id,
                actor_user_id=user_id,
                action=AuditAction.REPORT_EVIDENCE_EXPORTED,
                resource_type="report",
                resource_id=report["id"],
                metadata=_report_evidence_package_audit_metadata(report, export_content, counts),
            )
        )

    return PlainTextResponse(
        export_content,
        media_type=REPORT_EVIDENCE_PACKAGE_CONTENT_TYPE,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


def export_report_evidence_package_pdf(
    user_id: str,
    organization_id: str,
    report_id: str,
    store: SaaSStore,
    artifact_store: TenantArtifactStore,
) -> Response:
    report = _visible_report_for_action(
        user_id,
        organization_id,
        report_id,
        store,
        Action.READ,
    )
    pdf_content, filename, counts = _build_report_evidence_package_pdf(
        organization_id,
        report,
        store,
        artifact_store,
    )
    with store.conn:
        store.create_audit_event(
            AuditEvent(
                id=f"audit_report_evidence_export_{uuid4().hex}",
                organization_id=organization_id,
                actor_user_id=user_id,
                action=AuditAction.REPORT_EVIDENCE_EXPORTED,
                resource_type="report",
                resource_id=report["id"],
                metadata=_report_evidence_package_audit_metadata(
                    report,
                    pdf_content,
                    counts,
                    package_format="pdf",
                ),
            )
        )

    return Response(
        content=pdf_content,
        media_type=REPORT_EVIDENCE_PACKAGE_PDF_CONTENT_TYPE,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


def list_report_evidence_package_summaries(
    user_id: str,
    organization_id: str,
    report_id: str,
    store: SaaSStore,
) -> list[ReportEvidencePackageSummary]:
    report = _visible_report_for_action(
        user_id,
        organization_id,
        report_id,
        store,
        Action.READ,
    )
    return [
        create_report_evidence_package_summary_from_row(package)
        for package in store.list_report_evidence_packages(organization_id, report["id"])
    ]


def create_report_evidence_package_snapshot(
    user_id: str,
    organization_id: str,
    report_id: str,
    store: SaaSStore,
    artifact_store: TenantArtifactStore,
) -> ReportEvidencePackageSummary:
    report = _visible_report_for_action(
        user_id,
        organization_id,
        report_id,
        store,
        Action.MANAGE_REPORTS,
    )
    return _create_report_evidence_package_snapshot_for_report(
        user_id,
        organization_id,
        report,
        store,
        artifact_store,
    )


def download_report_evidence_package_snapshot(
    user_id: str,
    organization_id: str,
    report_id: str,
    package_id: str,
    store: SaaSStore,
    artifact_store: TenantArtifactStore,
) -> PlainTextResponse:
    report = _visible_report_for_action(
        user_id,
        organization_id,
        report_id,
        store,
        Action.READ,
    )
    package = store.get_report_evidence_package(package_id)
    if package is None:
        raise ValueError(f"Report evidence package {package_id} was not found")
    if package["organization_id"] != organization_id or package["report_id"] != report["id"]:
        raise AccessDenied("Report evidence package belongs to a different tenant or report")
    try:
        export_content = artifact_store.get_text(organization_id, package["storage_key"])
    except FileNotFoundError as exc:
        raise ValueError(f"Report evidence package {package_id} artifact was not found") from exc

    counts = {
        "comment_count": package["comment_count"],
        "delivery_count": package["delivery_count"],
        "review_count": package["review_count"],
        "schedule_count": package["schedule_count"],
        "upload_count": package["upload_count"],
    }
    with store.conn:
        store.create_audit_event(
            AuditEvent(
                id=f"audit_report_evidence_package_download_{uuid4().hex}",
                organization_id=organization_id,
                actor_user_id=user_id,
                action=AuditAction.REPORT_EVIDENCE_PACKAGE_DOWNLOADED,
                resource_type="report_evidence_package",
                resource_id=package["id"],
                metadata={
                    **_report_evidence_package_audit_metadata(report, export_content, counts),
                    "package_id": package["id"],
                },
            )
        )

    return PlainTextResponse(
        export_content,
        media_type=package["content_type"],
        headers={
            "Content-Disposition": f'attachment; filename="{package["filename"]}"',
        },
    )


def create_report_retention_summary(
    summary: TenantReportRetentionSummary,
) -> ReportRetentionApplyResponse:
    return ReportRetentionApplyResponse(
        organization_id=summary.organization_id,
        evaluated_count=summary.evaluated_count,
        expired_count=summary.expired_count,
        deleted_artifact_count=summary.deleted_artifact_count,
        missing_artifact_count=summary.missing_artifact_count,
        deleted_evidence_package_artifact_count=(
            summary.deleted_evidence_package_artifact_count
        ),
        missing_evidence_package_artifact_count=(
            summary.missing_evidence_package_artifact_count
        ),
        deleted_evidence_package_metadata_count=(
            summary.deleted_evidence_package_metadata_count
        ),
        retained_report_ids=summary.retained_report_ids,
    )


def apply_report_retention_summary(
    user_id: str,
    organization_id: str,
    store: SaaSStore,
    artifact_store: TenantArtifactStore | None = None,
) -> ReportRetentionApplyResponse:
    return create_report_retention_summary(
        apply_report_retention_policy(
            store,
            user_id=user_id,
            organization_id=organization_id,
            artifact_store=artifact_store,
        )
    )


def list_audit_event_summaries(
    user_id: str,
    organization_id: str,
    store: SaaSStore,
) -> list[AuditEventSummary]:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.VIEW_AUDIT,
    )
    return [
        AuditEventSummary(
            id=event["id"],
            organization_id=event["organization_id"],
            created_at=event["created_at"],
            actor_user_id=event["actor_user_id"],
            action=event["action"],
            resource_type=event["resource_type"],
            resource_id=event["resource_id"],
            metadata_json=_sanitized_audit_metadata_json(event["action"], event["metadata_json"]),
        )
        for event in store.list_audit_events(organization_id)
    ]


def export_audit_events(
    user_id: str,
    organization_id: str,
    store: SaaSStore,
) -> AuditExportResponse:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.VIEW_AUDIT,
    )
    events = store.list_audit_events(organization_id)
    output = StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "id",
            "organization_id",
            "created_at",
            "actor_user_id",
            "action",
            "resource_type",
            "resource_id",
            "metadata_json",
        ],
        lineterminator="\n",
    )
    writer.writeheader()
    for event in events:
        writer.writerow(
            {
                "id": event["id"],
                "organization_id": event["organization_id"],
                "created_at": event["created_at"],
                "actor_user_id": event["actor_user_id"],
                "action": event["action"],
                "resource_type": event["resource_type"],
                "resource_id": event["resource_id"],
                "metadata_json": _sanitized_audit_metadata_json(
                    event["action"],
                    event["metadata_json"],
                ),
            }
        )

    return AuditExportResponse(
        filename=f"{organization_id}-audit-events.csv",
        content_type="text/csv",
        event_count=len(events),
        content=output.getvalue(),
    )


def execute_local_analysis_run(
    user_id: str,
    organization_id: str,
    payload: LocalAnalysisRunCreate,
    store: SaaSStore,
    artifact_store: TenantArtifactStore | None = None,
) -> LocalAnalysisRunResponse:
    outcome = execute_analysis_run(
        store,
        AnalysisRunRequest(
            user_id=user_id,
            organization_id=organization_id,
            site_id=payload.site_id,
            upload_id=payload.upload_id,
            run_id=payload.run_id,
            report_id=payload.report_id,
            filename=payload.filename,
            raw_meter_data=pd.DataFrame(payload.rows),
            client_config=payload.client_config,
        ),
        artifact_store=artifact_store,
    )
    run_row = store.list_runs(organization_id, site_id=payload.site_id)[0]
    return LocalAnalysisRunResponse(
        upload_id=outcome.upload.id,
        run_upload_ids=list(outcome.run.upload_ids),
        run_id=outcome.run.id,
        report_id=outcome.report.id,
        run_status=run_row["status"],
        iso_summary=outcome.analytics.iso_summary,
    )


def execute_stored_upload_analysis_run(
    user_id: str,
    organization_id: str,
    payload: StoredUploadAnalysisRunCreate,
    store: SaaSStore,
    artifact_store: TenantArtifactStore,
) -> LocalAnalysisRunResponse:
    outcome = execute_analysis_run_from_upload(
        store,
        StoredUploadAnalysisRunRequest(
            user_id=user_id,
            organization_id=organization_id,
            upload_id=payload.upload_id,
            run_id=payload.run_id,
            report_id=payload.report_id,
            source_upload_ids=tuple(payload.source_upload_ids),
            client_config=payload.client_config,
        ),
        artifact_store=artifact_store,
    )
    run_row = store.list_runs(organization_id, site_id=outcome.upload.site_id)[0]
    return LocalAnalysisRunResponse(
        upload_id=outcome.upload.id,
        run_upload_ids=list(outcome.run.upload_ids),
        run_id=outcome.run.id,
        report_id=outcome.report.id,
        run_status=run_row["status"],
        iso_summary=outcome.analytics.iso_summary,
    )


def queue_stored_upload_analysis_run(
    user_id: str,
    organization_id: str,
    payload: StoredUploadAnalysisRunCreate,
    store: SaaSStore,
) -> QueuedAnalysisRunResponse:
    queued = queue_analysis_run_from_upload(
        store,
        StoredUploadAnalysisRunRequest(
            user_id=user_id,
            organization_id=organization_id,
            upload_id=payload.upload_id,
            run_id=payload.run_id,
            report_id=payload.report_id,
            source_upload_ids=tuple(payload.source_upload_ids),
            client_config=payload.client_config,
        ),
    )
    return QueuedAnalysisRunResponse(
        job_id=queued.job.id,
        upload_id=queued.upload.id,
        run_upload_ids=list(queued.run.upload_ids),
        run_id=queued.run.id,
        report_id=queued.job.report_id,
        run_status=queued.run.status.value,
    )


def create_invite(
    user_id: str,
    organization_id: str,
    payload: OrganizationInviteCreate,
    store: SaaSStore,
) -> OrganizationInviteSummary:
    invite = create_organization_invite(
        store,
        actor_user_id=user_id,
        organization_id=organization_id,
        invite_id=None,
        email=payload.email,
        role=payload.role,
    )
    return OrganizationInviteSummary(
        id=invite.id,
        organization_id=invite.organization_id,
        email=invite.email,
        role=invite.role.value,
        invited_by_user_id=invite.invited_by_user_id,
        status=invite.status.value,
        accepted_by_user_id=invite.accepted_by_user_id,
        accepted_at=invite.accepted_at.isoformat() if invite.accepted_at else None,
    )


def accept_invite(
    invite_id: str,
    payload: OrganizationInviteAccept,
    store: SaaSStore,
) -> OrganizationInviteSummary:
    result = accept_organization_invite(
        store,
        invite_id=invite_id,
        user=User(
            id=payload.user_id,
            email=payload.email,
            display_name=payload.display_name,
        ),
    )
    return OrganizationInviteSummary(
        id=result.invite.id,
        organization_id=result.invite.organization_id,
        email=result.invite.email,
        role=result.invite.role.value,
        invited_by_user_id=result.invite.invited_by_user_id,
        status=result.invite.status.value,
        accepted_by_user_id=result.invite.accepted_by_user_id,
        accepted_at=result.invite.accepted_at.isoformat() if result.invite.accepted_at else None,
    )


def revoke_invite(
    user_id: str,
    organization_id: str,
    invite_id: str,
    store: SaaSStore,
) -> OrganizationInviteSummary:
    invite = revoke_organization_invite(
        store,
        actor_user_id=user_id,
        organization_id=organization_id,
        invite_id=invite_id,
    )
    return OrganizationInviteSummary(
        id=invite.id,
        organization_id=invite.organization_id,
        email=invite.email,
        role=invite.role.value,
        invited_by_user_id=invite.invited_by_user_id,
        status=invite.status.value,
        accepted_by_user_id=invite.accepted_by_user_id,
        accepted_at=invite.accepted_at.isoformat() if invite.accepted_at else None,
    )


def _platform_operation_event_from_row(row: Any) -> PlatformOperationEventSummary:
    return PlatformOperationEventSummary(
        id=row["id"],
        event_type=row["event_type"],
        status=row["status"],
        request_id=row["request_id"],
        detail=row["detail"],
        created_at=row["created_at"],
    )


def _parse_observed_at(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _worker_heartbeat_age_seconds(value: str, now: datetime) -> float | None:
    try:
        observed_at = _parse_observed_at(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, (now - observed_at).total_seconds())


def _worker_fingerprint(worker_id: str) -> str:
    return sha256(worker_id.encode("utf-8")).hexdigest()[:12]


def _worker_heartbeat_from_row(
    row: Any,
    *,
    now: datetime,
) -> SupportWorkerHeartbeatSummary:
    return SupportWorkerHeartbeatSummary(
        worker_fingerprint=_worker_fingerprint(row["worker_id"]),
        scope=row["scope"],
        poll_seconds=float(row["poll_seconds"]),
        batch_size=int(row["batch_size"]),
        last_attempted=int(row["last_attempted"]),
        last_succeeded=int(row["last_succeeded"]),
        last_failed=int(row["last_failed"]),
        last_skipped=int(row["last_skipped"]),
        last_scheduled_attempted=int(row["last_scheduled_attempted"]),
        last_scheduled_sent=int(row["last_scheduled_sent"]),
        last_scheduled_failed=int(row["last_scheduled_failed"]),
        last_scheduled_skipped=int(row["last_scheduled_skipped"]),
        last_connector_attempted=int(row["last_connector_attempted"]),
        last_connector_succeeded=int(row["last_connector_succeeded"]),
        last_connector_failed=int(row["last_connector_failed"]),
        last_connector_skipped=int(row["last_connector_skipped"]),
        last_seen_at=row["last_seen_at"],
        age_seconds=_worker_heartbeat_age_seconds(row["last_seen_at"], now),
    )


def create_worker_heartbeat_queue_summary(
    store: SaaSStore,
    *,
    now: datetime | None = None,
) -> SupportWorkerHeartbeatQueueSummary:
    observed_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    counts = store.count_worker_heartbeats_by_scope()
    latest_all_tenant = store.get_latest_worker_heartbeat()
    latest_seen_at = (
        latest_all_tenant["last_seen_at"] if latest_all_tenant is not None else None
    )
    return SupportWorkerHeartbeatQueueSummary(
        total_count=sum(counts.values()),
        latest_all_tenant_last_seen_at=latest_seen_at,
        latest_all_tenant_age_seconds=(
            _worker_heartbeat_age_seconds(latest_seen_at, observed_at)
            if latest_seen_at is not None
            else None
        ),
        by_scope=[
            SupportWorkerHeartbeatScopeCount(scope=scope, count=counts.get(scope, 0))
            for scope in ("all", "organization")
        ],
    )


def _csp_violation_report_from_row(row: Any) -> SupportCspViolationReportSummary:
    return SupportCspViolationReportSummary(
        id=row["id"],
        disposition=row["disposition"],
        effective_directive=row["effective_directive"],
        violated_directive=row["violated_directive"],
        blocked_uri_category=row["blocked_uri_category"],
        document_origin=row["document_origin"],
        status_code=row["status_code"],
        created_at=row["created_at"],
    )


def create_csp_violation_report_queue_summary(
    store: SaaSStore,
) -> SupportCspReportQueueSummary:
    disposition_counts = store.count_csp_violation_reports_by_disposition()
    directive_counts = store.count_csp_violation_reports_by_effective_directive()
    return SupportCspReportQueueSummary(
        total_count=store.count_csp_violation_reports(),
        latest_report_created_at=store.get_latest_csp_violation_report_created_at(),
        by_disposition=[
            SupportCspReportDispositionCount(
                disposition=disposition.value,
                count=disposition_counts.get(disposition.value, 0),
            )
            for disposition in CspViolationReportDisposition
        ],
        by_effective_directive=[
            SupportCspReportDirectiveCount(
                directive=directive.value,
                count=directive_counts.get(directive.value, 0),
            )
            for directive in CspViolationReportDirective
        ],
    )


def create_csp_violation_report_retention_summary(
    summary: CspViolationReportRetentionSummary,
) -> SupportCspReportRetentionSummary:
    return SupportCspReportRetentionSummary(
        retention_days=summary.retention_days,
        cutoff_created_before=summary.cutoff_created_before,
        deleted_count=summary.deleted_count,
        remaining_count=summary.remaining_count,
    )


def _record_platform_operation_event(
    store: SaaSStore,
    *,
    event_type: str,
    event_status: str,
    request_id: str | None,
    detail: str,
    created_at: str | None = None,
) -> str:
    event_id = f"platform_operation_{uuid4().hex}"
    trimmed_detail = detail.strip()[:240]
    store.create_platform_operation_event(
        event_id=event_id,
        event_type=event_type,
        event_status=event_status,
        request_id=request_id.strip()[:160] if request_id else None,
        detail=trimmed_detail,
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
    )
    return event_id


def _record_clerk_webhook_replay_event(
    store: SaaSStore,
    request: Request,
    result: ClerkWebhookResponse,
) -> None:
    _record_platform_operation_event(
        store,
        event_type=CLERK_WEBHOOK_REPLAY_EVENT_TYPE,
        event_status="succeeded" if result.action != "ignored" else "failed",
        request_id=request.headers.get(REQUEST_ID_HEADER),
        detail=f"action={result.action}",
    )


def _record_billing_webhook_replay_event(
    store: SaaSStore,
    request: Request,
    result: BillingWebhookSummary,
) -> None:
    _record_platform_operation_event(
        store,
        event_type=BILLING_WEBHOOK_REPLAY_EVENT_TYPE,
        event_status="succeeded",
        request_id=request.headers.get(REQUEST_ID_HEADER),
        detail=f"provider={result.provider} action={result.action} duplicate={str(result.duplicate).lower()}",
    )


def _record_notification_provider_webhook_replay_event(
    store: SaaSStore,
    request: Request,
    result: NotificationProviderWebhookSummary,
) -> None:
    _record_platform_operation_event(
        store,
        event_type=NOTIFICATION_PROVIDER_WEBHOOK_REPLAY_EVENT_TYPE,
        event_status="succeeded",
        request_id=request.headers.get(REQUEST_ID_HEADER),
        detail=f"status={result.status} duplicate={str(result.duplicate).lower()}",
    )


def create_app(
    store: SaaSStore | None = None,
    artifact_store: TenantArtifactStore | None = None,
    email_sender: EmailSender | None = None,
    marketing_inquiry_recipient_email: str | None = None,
    marketing_inquiry_crm_sender: MarketingInquiryCrmSender | None = None,
) -> FastAPI:
    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description="Backend API boundary for the SaaS energy analytics platform.",
    )
    app.state.store = store or SaaSStore(initialize_database_from_env())
    app.state.artifact_store = artifact_store or artifact_store_from_env()
    app.state.runtime_env = runtime_env_from_env()
    app.state.email_sender = email_sender or email_sender_from_env()
    app.state.marketing_inquiry_recipient_email = (
        marketing_inquiry_recipient_email
        if marketing_inquiry_recipient_email is not None
        else marketing_inquiry_recipient_email_from_env()
    )
    app.state.marketing_inquiry_crm_sender = (
        marketing_inquiry_crm_sender
        if marketing_inquiry_crm_sender is not None
        else marketing_inquiry_crm_sender_from_env(app.state.runtime_env)
    )
    app.state.alert_webhook_sender = None
    configure_security_middleware(app, app.state.runtime_env)
    configure_request_logging(app)

    @app.exception_handler(AccessDenied)
    async def access_denied_handler(request: Request, exc: AccessDenied) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"detail": str(exc)},
        )

    @app.exception_handler(SubscriptionLimitExceeded)
    async def subscription_limit_handler(
        request: Request,
        exc: SubscriptionLimitExceeded,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            content={"detail": str(exc)},
        )

    @app.exception_handler(BillingConfigurationError)
    async def billing_configuration_handler(
        request: Request,
        exc: BillingConfigurationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": str(exc)},
        )

    @app.exception_handler(BillingWebhookVerificationError)
    async def billing_webhook_verification_handler(
        request: Request,
        exc: BillingWebhookVerificationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc)},
        )

    @app.exception_handler(SecurityConfigurationError)
    async def security_configuration_handler(
        request: Request,
        exc: SecurityConfigurationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": str(exc)},
        )

    @app.exception_handler(OperationsConfigurationError)
    async def operations_configuration_handler(
        request: Request,
        exc: OperationsConfigurationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": str(exc)},
        )

    @app.exception_handler(AlertDeliveryError)
    async def alert_delivery_handler(
        request: Request,
        exc: AlertDeliveryError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content={"detail": str(exc)},
        )

    @app.exception_handler(SupportConfigurationError)
    async def support_configuration_handler(
        request: Request,
        exc: SupportConfigurationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": str(exc)},
        )

    @app.exception_handler(SupportAuthenticationError)
    async def support_authentication_handler(
        request: Request,
        exc: SupportAuthenticationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": str(exc)},
            headers={"WWW-Authenticate": "Bearer"},
        )

    @app.exception_handler(MarketingInquiryRateLimitConfigurationError)
    async def marketing_rate_limit_configuration_handler(
        request: Request,
        exc: MarketingInquiryRateLimitConfigurationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": str(exc)},
        )

    @app.exception_handler(MarketingInquiryRateLimitExceeded)
    async def marketing_rate_limit_handler(
        request: Request,
        exc: MarketingInquiryRateLimitExceeded,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": str(exc)},
            headers={"Retry-After": str(exc.retry_after_seconds)},
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc)},
        )

    app.add_api_route("/health", health_check, methods=["GET"], tags=["system"])

    @app.get("/ready", tags=["system"])
    def api_readiness(request: Request) -> JSONResponse:
        report = readiness_check(
            request.app.state.store,
            request.app.state.artifact_store,
        )
        return JSONResponse(
            status_code=status.HTTP_200_OK
            if report["status"] == "ready"
            else status.HTTP_503_SERVICE_UNAVAILABLE,
            content=report,
        )

    @app.get("/metrics", tags=["system"])
    def api_metrics(request: Request) -> PlainTextResponse:
        try:
            verify_metrics_request(request.headers.get("authorization"))
        except MetricsConfigurationError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        except MetricsAuthenticationError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(exc),
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc

        metrics = collect_operational_metrics(request.app.state.store)
        return PlainTextResponse(
            format_prometheus_metrics(metrics),
            media_type=PROMETHEUS_CONTENT_TYPE,
        )

    @app.post(
        "/security/csp-reports",
        response_model=CspViolationReportSummary,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["security"],
    )
    async def security_csp_reports(request: Request) -> CspViolationReportSummary:
        try:
            report = process_csp_violation_report(
                request.app.state.store,
                await request.body(),
            )
        except CspViolationReportTooLargeError as exc:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=str(exc),
            ) from exc
        except CspViolationReportPayloadError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        return create_csp_violation_report_summary(report)

    @app.post(
        "/marketing/inquiries",
        response_model=MarketingInquirySummary,
        tags=["marketing"],
    )
    def marketing_inquiry(
        payload: MarketingInquiryCreate,
        request: Request,
    ) -> MarketingInquirySummary:
        (
            rate_limit_source_key_hash,
            rate_limit_since_iso,
            rate_limit_max_submissions,
            rate_limit_retry_after_seconds,
        ) = _marketing_inquiry_rate_limit_context(request)
        return create_marketing_inquiry_summary(
            payload,
            request.app.state.store,
            request.app.state.email_sender,
            request.app.state.marketing_inquiry_recipient_email,
            request.app.state.marketing_inquiry_crm_sender,
            rate_limit_source_key_hash,
            rate_limit_since_iso,
            rate_limit_max_submissions,
            rate_limit_retry_after_seconds,
        )

    @app.get(
        "/support/marketing-inquiries",
        response_model=list[MarketingInquiryDetail],
        tags=["support"],
    )
    def support_marketing_inquiries(
        request: Request,
        inquiry_status: MarketingInquiryStatus | None = Query(None, alias="status"),
        assigned_to: str | None = Query(None),
        unassigned: bool = Query(False),
        limit: int = Query(100, ge=1, le=200),
    ) -> list[MarketingInquiryDetail]:
        verify_support_request(request.headers.get("authorization"))
        if assigned_to is not None and unassigned:
            raise ValueError("Support inquiry list cannot filter by assignee and unassigned")
        normalized_assignee = (
            _normalize_marketing_inquiry_assignee(assigned_to)
            if assigned_to is not None
            else None
        )
        rows = request.app.state.store.list_marketing_inquiries(
            inquiry_status,
            limit=limit,
            assigned_to=normalized_assignee,
            unassigned=unassigned,
        )
        return [_marketing_inquiry_detail_from_row(row) for row in rows]

    @app.get(
        "/support/marketing-inquiries/summary",
        response_model=MarketingInquiryQueueSummary,
        tags=["support"],
    )
    def support_marketing_inquiry_summary(request: Request) -> MarketingInquiryQueueSummary:
        verify_support_request(request.headers.get("authorization"))
        return create_marketing_inquiry_queue_summary(request.app.state.store)

    @app.patch(
        "/support/marketing-inquiries/{inquiry_id}",
        response_model=MarketingInquiryDetail,
        tags=["support"],
    )
    def support_marketing_inquiry_status(
        inquiry_id: str,
        payload: MarketingInquiryStatusUpdate,
        request: Request,
    ) -> MarketingInquiryDetail:
        verify_support_request(request.headers.get("authorization"))
        update_fields = payload.model_fields_set
        if not update_fields:
            raise ValueError("Marketing inquiry follow-up update must include a change")
        actor_label = _normalize_support_actor(request.headers.get("x-support-actor"))
        assigned_to = (
            _normalize_marketing_inquiry_assignee(payload.assigned_to)
            if "assigned_to" in update_fields
            else None
        )
        follow_up_notes = (
            _normalize_optional_text(
                payload.follow_up_notes,
                1200,
                "Marketing inquiry follow-up notes",
            )
            if "follow_up_notes" in update_fields
            else None
        )
        changed_fields = []
        if payload.status is not None:
            changed_fields.append("status")
        if "assigned_to" in update_fields:
            changed_fields.append("assigned_to")
        if "follow_up_notes" in update_fields:
            changed_fields.append("follow_up_notes")
        updated_at_iso = datetime.now(timezone.utc).isoformat()
        with request.app.state.store.conn:
            row = request.app.state.store.update_marketing_inquiry_followup(
                inquiry_id,
                status=payload.status,
                assigned_to_set="assigned_to" in update_fields,
                assigned_to=assigned_to,
                follow_up_notes_set="follow_up_notes" in update_fields,
                follow_up_notes=follow_up_notes,
                updated_at_iso=updated_at_iso,
            )
            if row is not None:
                request.app.state.store.create_marketing_inquiry_followup_event(
                    event_id=f"marketing_inquiry_event_{uuid4().hex}",
                    inquiry_id=inquiry_id,
                    actor_label=actor_label,
                    action="updated",
                    status=MarketingInquiryStatus(row["status"]),
                    assigned_to=row["assigned_to"],
                    changed_fields=changed_fields,
                    notes_changed="follow_up_notes" in update_fields,
                    created_at_iso=updated_at_iso,
                )
        if row is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Marketing inquiry {inquiry_id} was not found",
            )
        return _marketing_inquiry_detail_from_row(row)

    @app.get(
        "/support/marketing-inquiries/{inquiry_id}/activity",
        response_model=list[MarketingInquiryActivitySummary],
        tags=["support"],
    )
    def support_marketing_inquiry_activity(
        inquiry_id: str,
        request: Request,
    ) -> list[MarketingInquiryActivitySummary]:
        verify_support_request(request.headers.get("authorization"))
        if request.app.state.store.get_marketing_inquiry(inquiry_id) is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Marketing inquiry {inquiry_id} was not found",
            )
        rows = request.app.state.store.list_marketing_inquiry_followup_events(inquiry_id)
        return [_marketing_inquiry_activity_from_row(row) for row in rows]

    @app.post(
        "/support/alert-test",
        response_model=AlertDeliveryTestResponse,
        tags=["support"],
    )
    def support_alert_delivery_test(request: Request) -> AlertDeliveryTestResponse:
        verify_support_request(request.headers.get("authorization"))
        request_id = request.headers.get(REQUEST_ID_HEADER)
        try:
            result = send_alert_delivery_test(
                request_id=request_id,
                runtime_env=request.app.state.runtime_env,
                webhook_sender=request.app.state.alert_webhook_sender,
            )
        except AlertDeliveryError as exc:
            with request.app.state.store.conn:
                _record_platform_operation_event(
                    request.app.state.store,
                    event_type=ALERT_DELIVERY_TEST_EVENT_TYPE,
                    event_status="failed",
                    request_id=request_id,
                    detail=str(exc),
                )
            raise

        with request.app.state.store.conn:
            event_id = _record_platform_operation_event(
                request.app.state.store,
                event_type=result.event_type,
                event_status="succeeded",
                request_id=request_id,
                detail=f"http_status={result.status_code}",
                created_at=result.delivered_at,
            )
        return AlertDeliveryTestResponse(
            event_id=event_id,
            event_type=result.event_type,
            delivered_at=result.delivered_at,
            status_code=result.status_code,
        )

    @app.post(
        "/support/database-test",
        response_model=DatabaseTestResponse,
        tags=["support"],
    )
    def support_database_test(request: Request) -> DatabaseTestResponse:
        verify_support_request(request.headers.get("authorization"))
        request_id = request.headers.get(REQUEST_ID_HEADER)
        tested_at = datetime.now(timezone.utc).isoformat()
        event_id = f"platform_operation_{uuid4().hex}"

        try:
            with request.app.state.store.conn:
                request.app.state.store.create_platform_operation_event(
                    event_id=event_id,
                    event_type=DATABASE_TEST_EVENT_TYPE,
                    event_status="succeeded",
                    request_id=request_id.strip()[:160] if request_id else None,
                    detail="write_read_verified=true",
                    created_at=tested_at,
                )
                event = request.app.state.store.get_platform_operation_event(event_id)
                if event is None:
                    raise RuntimeError("database smoke event was not readable")
                if (
                    event["event_type"] != DATABASE_TEST_EVENT_TYPE
                    or event["status"] != "succeeded"
                    or event["created_at"] != tested_at
                ):
                    raise RuntimeError("database smoke event did not round trip")
        except Exception as exc:
            try:
                with request.app.state.store.conn:
                    _record_platform_operation_event(
                        request.app.state.store,
                        event_type=DATABASE_TEST_EVENT_TYPE,
                        event_status="failed",
                        request_id=request_id,
                        detail="Database smoke test failed during write/read",
                    )
            except Exception:
                pass
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Database smoke test failed during write/read",
            ) from exc

        return DatabaseTestResponse(
            event_id=event_id,
            event_type=DATABASE_TEST_EVENT_TYPE,
            tested_at=tested_at,
        )

    @app.post(
        "/support/tenant-isolation-test",
        response_model=TenantIsolationTestResponse,
        tags=["support"],
    )
    def support_tenant_isolation_test(request: Request) -> TenantIsolationTestResponse:
        verify_support_request(request.headers.get("authorization"))
        request_id = request.headers.get(REQUEST_ID_HEADER)
        try:
            result = run_tenant_isolation_smoke_test(request.app.state.store)
        except TenantIsolationSmokeTestError as exc:
            with request.app.state.store.conn:
                _record_platform_operation_event(
                    request.app.state.store,
                    event_type=TENANT_ISOLATION_TEST_EVENT_TYPE,
                    event_status="failed",
                    request_id=request_id,
                    detail=str(exc),
                )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(exc),
            ) from exc

        with request.app.state.store.conn:
            event_id = _record_platform_operation_event(
                request.app.state.store,
                event_type=result.event_type,
                event_status="succeeded",
                request_id=request_id,
                detail=(
                    f"checks_passed={result.checks_passed} "
                    f"synthetic_records={result.synthetic_records}"
                ),
                created_at=result.tested_at,
            )
        return TenantIsolationTestResponse(
            event_id=event_id,
            event_type=result.event_type,
            tested_at=result.tested_at,
            checks_passed=result.checks_passed,
            synthetic_records=result.synthetic_records,
        )

    @app.post(
        "/support/run-lifecycle-test",
        response_model=RunLifecycleTestResponse,
        tags=["support"],
    )
    def support_run_lifecycle_test(request: Request) -> RunLifecycleTestResponse:
        verify_support_request(request.headers.get("authorization"))
        request_id = request.headers.get(REQUEST_ID_HEADER)
        try:
            result = run_lifecycle_smoke_test(
                request.app.state.store,
                request.app.state.artifact_store,
            )
        except RunLifecycleSmokeTestError as exc:
            with request.app.state.store.conn:
                _record_platform_operation_event(
                    request.app.state.store,
                    event_type=RUN_LIFECYCLE_TEST_EVENT_TYPE,
                    event_status="failed",
                    request_id=request_id,
                    detail=str(exc),
                )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(exc),
            ) from exc

        with request.app.state.store.conn:
            event_id = _record_platform_operation_event(
                request.app.state.store,
                event_type=result.event_type,
                event_status="succeeded",
                request_id=request_id,
                detail=(
                    f"checks_passed={result.checks_passed} "
                    f"artifacts_verified={result.artifacts_verified} "
                    f"synthetic_records={result.synthetic_records}"
                ),
                created_at=result.tested_at,
            )
        return RunLifecycleTestResponse(
            event_id=event_id,
            event_type=result.event_type,
            tested_at=result.tested_at,
            checks_passed=result.checks_passed,
            artifacts_verified=result.artifacts_verified,
            synthetic_records=result.synthetic_records,
        )

    @app.post(
        "/support/worker-run-lifecycle-test",
        response_model=WorkerRunLifecycleTestResponse,
        tags=["support"],
    )
    def support_worker_run_lifecycle_test(request: Request) -> WorkerRunLifecycleTestResponse:
        verify_support_request(request.headers.get("authorization"))
        request_id = request.headers.get(REQUEST_ID_HEADER)
        try:
            result = worker_run_lifecycle_smoke_test(
                request.app.state.store,
                request.app.state.artifact_store,
            )
        except WorkerRunLifecycleSmokeTestError as exc:
            with request.app.state.store.conn:
                _record_platform_operation_event(
                    request.app.state.store,
                    event_type=WORKER_RUN_LIFECYCLE_TEST_EVENT_TYPE,
                    event_status="failed",
                    request_id=request_id,
                    detail=str(exc),
                )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(exc),
            ) from exc

        with request.app.state.store.conn:
            event_id = _record_platform_operation_event(
                request.app.state.store,
                event_type=result.event_type,
                event_status="succeeded",
                request_id=request_id,
                detail=(
                    f"checks_passed={result.checks_passed} "
                    f"jobs_attempted={result.jobs_attempted} "
                    f"jobs_succeeded={result.jobs_succeeded} "
                    f"artifacts_verified={result.artifacts_verified} "
                    f"synthetic_records={result.synthetic_records}"
                ),
                created_at=result.tested_at,
            )
        return WorkerRunLifecycleTestResponse(
            event_id=event_id,
            event_type=result.event_type,
            tested_at=result.tested_at,
            checks_passed=result.checks_passed,
            jobs_attempted=result.jobs_attempted,
            jobs_succeeded=result.jobs_succeeded,
            artifacts_verified=result.artifacts_verified,
            synthetic_records=result.synthetic_records,
        )

    @app.post(
        "/support/artifact-store-test",
        response_model=ArtifactStoreTestResponse,
        tags=["support"],
    )
    def support_artifact_store_test(request: Request) -> ArtifactStoreTestResponse:
        verify_support_request(request.headers.get("authorization"))
        request_id = request.headers.get(REQUEST_ID_HEADER)
        try:
            result = artifact_store_smoke_test(request.app.state.artifact_store)
        except ArtifactStoreSmokeTestError as exc:
            with request.app.state.store.conn:
                _record_platform_operation_event(
                    request.app.state.store,
                    event_type=ARTIFACT_STORE_TEST_EVENT_TYPE,
                    event_status="failed",
                    request_id=request_id,
                    detail=str(exc),
                )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=str(exc),
            ) from exc

        with request.app.state.store.conn:
            event_id = _record_platform_operation_event(
                request.app.state.store,
                event_type=ARTIFACT_STORE_TEST_EVENT_TYPE,
                event_status="succeeded",
                request_id=request_id,
                detail=f"bytes_written={result.bytes_written}",
                created_at=result.tested_at,
            )
        return ArtifactStoreTestResponse(
            event_id=event_id,
            event_type=ARTIFACT_STORE_TEST_EVENT_TYPE,
            tested_at=result.tested_at,
            bytes_written=result.bytes_written,
        )

    @app.get(
        "/support/operation-events",
        response_model=list[PlatformOperationEventSummary],
        tags=["support"],
    )
    def support_operation_events(
        request: Request,
        event_type: str | None = Query(None),
        event_status: str | None = Query(None, alias="status"),
        limit: int = Query(100, ge=1, le=200),
    ) -> list[PlatformOperationEventSummary]:
        verify_support_request(request.headers.get("authorization"))
        if event_type is not None and event_type not in SUPPORTED_PLATFORM_OPERATION_EVENT_TYPES:
            raise ValueError(f"Unsupported platform operation event type: {event_type}")
        if (
            event_status is not None
            and event_status not in SUPPORTED_PLATFORM_OPERATION_EVENT_STATUSES
        ):
            raise ValueError(f"Unsupported platform operation event status: {event_status}")
        rows = request.app.state.store.list_platform_operation_events(
            event_type=event_type,
            event_status=event_status,
            limit=limit,
        )
        return [_platform_operation_event_from_row(row) for row in rows]

    @app.get(
        "/support/worker-heartbeats/summary",
        response_model=SupportWorkerHeartbeatQueueSummary,
        tags=["support"],
    )
    def support_worker_heartbeat_summary(
        request: Request,
    ) -> SupportWorkerHeartbeatQueueSummary:
        verify_support_request(request.headers.get("authorization"))
        return create_worker_heartbeat_queue_summary(request.app.state.store)

    @app.get(
        "/support/worker-heartbeats",
        response_model=list[SupportWorkerHeartbeatSummary],
        tags=["support"],
    )
    def support_worker_heartbeats(
        request: Request,
        scope: str | None = Query(None),
        limit: int = Query(100, ge=1, le=200),
    ) -> list[SupportWorkerHeartbeatSummary]:
        verify_support_request(request.headers.get("authorization"))
        if scope is not None and scope not in {"all", "organization"}:
            raise ValueError(f"Unsupported worker heartbeat scope: {scope}")
        observed_at = datetime.now(timezone.utc)
        rows = request.app.state.store.list_worker_heartbeats(
            scope=scope,
            limit=limit,
        )
        return [_worker_heartbeat_from_row(row, now=observed_at) for row in rows]

    @app.get(
        "/support/csp-reports/summary",
        response_model=SupportCspReportQueueSummary,
        tags=["support"],
    )
    def support_csp_report_summary(request: Request) -> SupportCspReportQueueSummary:
        verify_support_request(request.headers.get("authorization"))
        return create_csp_violation_report_queue_summary(request.app.state.store)

    @app.post(
        "/support/csp-reports/retention/apply",
        response_model=SupportCspReportRetentionSummary,
        tags=["support"],
    )
    def support_apply_csp_report_retention(
        request: Request,
        retention_days: int | None = Query(None, ge=1, le=3650),
    ) -> SupportCspReportRetentionSummary:
        verify_support_request(request.headers.get("authorization"))
        return create_csp_violation_report_retention_summary(
            apply_csp_violation_report_retention(
                request.app.state.store,
                retention_days=retention_days,
            )
        )

    @app.get(
        "/support/csp-reports",
        response_model=list[SupportCspViolationReportSummary],
        tags=["support"],
    )
    def support_csp_reports(
        request: Request,
        disposition: str | None = Query(None),
        directive: str | None = Query(None),
        limit: int = Query(100, ge=1, le=200),
    ) -> list[SupportCspViolationReportSummary]:
        verify_support_request(request.headers.get("authorization"))
        normalized_disposition = None
        normalized_directive = None
        if disposition is not None:
            try:
                normalized_disposition = CspViolationReportDisposition(
                    disposition.strip().lower()
                ).value
            except ValueError as exc:
                raise ValueError(f"Unsupported CSP report disposition: {disposition}") from exc
        if directive is not None:
            try:
                normalized_directive = CspViolationReportDirective(
                    directive.strip().lower()
                ).value
            except ValueError as exc:
                raise ValueError(f"Unsupported CSP report directive: {directive}") from exc
        rows = request.app.state.store.list_csp_violation_reports(
            disposition=normalized_disposition,
            effective_directive=normalized_directive,
            limit=limit,
        )
        return [_csp_violation_report_from_row(row) for row in rows]

    @app.post(
        "/auth/dev/session",
        response_model=DevSessionResponse,
        tags=["auth"],
    )
    def auth_dev_session(payload: DevSessionCreate, request: Request) -> DevSessionResponse:
        try:
            return create_dev_session(payload, request.app.state.store)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

    @app.post(
        "/auth/clerk/sync",
        response_model=ClerkSyncResponse,
        tags=["auth"],
    )
    def auth_clerk_sync(
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> ClerkSyncResponse:
        return sync_clerk_session(user, request.app.state.store)

    @app.post(
        "/webhooks/clerk",
        response_model=ClerkWebhookResponse,
        tags=["webhooks"],
    )
    async def clerk_webhook(request: Request) -> ClerkWebhookResponse:
        payload = await request.body()
        try:
            event = verify_clerk_webhook_event(payload, request.headers)
        except ClerkWebhookConfigurationError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc
        except ClerkWebhookVerificationError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

        result = process_clerk_webhook(event, request.app.state.store)
        _record_clerk_webhook_replay_event(request.app.state.store, request, result)
        return result

    @app.post(
        "/webhooks/billing",
        response_model=BillingWebhookSummary,
        tags=["webhooks"],
    )
    async def billing_webhook(request: Request) -> BillingWebhookSummary:
        result = create_billing_webhook_summary(
            process_billing_webhook_event(
                request.app.state.store,
                await request.body(),
                request.headers,
            )
        )
        _record_billing_webhook_replay_event(request.app.state.store, request, result)
        return result

    @app.post(
        "/webhooks/notifications/provider",
        response_model=NotificationProviderWebhookSummary,
        tags=["webhooks"],
    )
    async def notification_provider_webhook(
        request: Request,
    ) -> NotificationProviderWebhookSummary:
        payload = await request.body()
        try:
            result = process_notification_provider_webhook_event(
                request.app.state.store,
                payload,
                request.headers,
            )
        except NotificationProviderWebhookConfigurationError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc
        except (
            NotificationProviderWebhookPayloadError,
            NotificationProviderWebhookVerificationError,
        ) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        summary = create_notification_provider_webhook_summary(result)
        _record_notification_provider_webhook_replay_event(
            request.app.state.store,
            request,
            summary,
        )
        return summary

    @app.post(
        "/organizations/onboard-owner",
        response_model=OwnerOrganizationResponse,
        tags=["organizations"],
    )
    def onboard_owner(payload: OwnerOrganizationCreate, request: Request) -> OwnerOrganizationResponse:
        return onboard_owner_organization(payload, request.app.state.store)

    @app.get(
        "/me/organizations",
        response_model=list[OrganizationSummary],
        tags=["organizations"],
    )
    def user_organizations(
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> list[OrganizationSummary]:
        return list_user_organization_summaries(user.id, request.app.state.store)

    @app.get(
        "/billing/plans",
        response_model=list[SubscriptionPlanSummary],
        tags=["billing"],
    )
    def billing_plans(request: Request) -> list[SubscriptionPlanSummary]:
        return list_subscription_plan_summaries(request.app.state.store)

    @app.get(
        "/organizations/{organization_id}",
        response_model=OrganizationSummary,
        tags=["organizations"],
    )
    def organization_profile(
        organization_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> OrganizationSummary:
        return get_organization_summary(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            request.app.state.store,
        )

    @app.patch(
        "/organizations/{organization_id}",
        response_model=OrganizationSummary,
        tags=["organizations"],
    )
    def organization_update_profile(
        organization_id: str,
        payload: OrganizationProfileUpdate,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> OrganizationSummary:
        return update_organization_summary(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            payload,
            request.app.state.store,
        )

    @app.get(
        "/organizations/{organization_id}/report-branding",
        response_model=OrganizationReportBrandingSummary,
        tags=["organizations"],
    )
    def organization_report_branding(
        organization_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> OrganizationReportBrandingSummary:
        return get_report_branding_summary(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            request.app.state.store,
        )

    @app.patch(
        "/organizations/{organization_id}/report-branding",
        response_model=OrganizationReportBrandingSummary,
        tags=["organizations"],
    )
    def organization_update_report_branding(
        organization_id: str,
        payload: OrganizationReportBrandingUpdate,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> OrganizationReportBrandingSummary:
        return update_report_branding_summary(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            payload,
            request.app.state.store,
        )

    @app.get(
        "/organizations/{organization_id}/report-approval-policy",
        response_model=OrganizationReportApprovalPolicySummary,
        tags=["organizations"],
    )
    def organization_report_approval_policy(
        organization_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> OrganizationReportApprovalPolicySummary:
        return get_report_approval_policy_summary(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            request.app.state.store,
        )

    @app.patch(
        "/organizations/{organization_id}/report-approval-policy",
        response_model=OrganizationReportApprovalPolicySummary,
        tags=["organizations"],
    )
    def organization_update_report_approval_policy(
        organization_id: str,
        payload: OrganizationReportApprovalPolicyUpdate,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> OrganizationReportApprovalPolicySummary:
        return update_report_approval_policy_summary(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            payload,
            request.app.state.store,
        )

    @app.get(
        "/organizations/{organization_id}/evidence-package-template",
        response_model=OrganizationEvidencePackageTemplateSummary,
        tags=["organizations"],
    )
    def organization_evidence_package_template(
        organization_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> OrganizationEvidencePackageTemplateSummary:
        return get_evidence_package_template_summary(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            request.app.state.store,
        )

    @app.patch(
        "/organizations/{organization_id}/evidence-package-template",
        response_model=OrganizationEvidencePackageTemplateSummary,
        tags=["organizations"],
    )
    def organization_update_evidence_package_template(
        organization_id: str,
        payload: OrganizationEvidencePackageTemplateUpdate,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> OrganizationEvidencePackageTemplateSummary:
        return update_evidence_package_template_summary(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            payload,
            request.app.state.store,
        )

    @app.get(
        "/organizations/{organization_id}/privacy-settings",
        response_model=OrganizationPrivacySettingsSummary,
        tags=["organizations"],
    )
    def organization_privacy_settings(
        organization_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> OrganizationPrivacySettingsSummary:
        return get_privacy_settings_summary(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            request.app.state.store,
        )

    @app.patch(
        "/organizations/{organization_id}/privacy-settings",
        response_model=OrganizationPrivacySettingsSummary,
        tags=["organizations"],
    )
    def organization_update_privacy_settings(
        organization_id: str,
        payload: OrganizationPrivacySettingsUpdate,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> OrganizationPrivacySettingsSummary:
        return update_privacy_settings_summary(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            payload,
            request.app.state.store,
        )

    @app.get(
        "/organizations/{organization_id}/privacy-requests",
        response_model=list[OrganizationPrivacyRequestSummary],
        tags=["organizations"],
    )
    def organization_privacy_requests(
        organization_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> list[OrganizationPrivacyRequestSummary]:
        return list_privacy_request_summaries(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            request.app.state.store,
        )

    @app.post(
        "/organizations/{organization_id}/privacy-requests",
        response_model=OrganizationPrivacyRequestSummary,
        tags=["organizations"],
    )
    def organization_create_privacy_request(
        organization_id: str,
        payload: OrganizationPrivacyRequestCreate,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> OrganizationPrivacyRequestSummary:
        return create_privacy_request_summary(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            payload,
            request.app.state.store,
        )

    @app.patch(
        "/organizations/{organization_id}/privacy-requests/{privacy_request_id}",
        response_model=OrganizationPrivacyRequestSummary,
        tags=["organizations"],
    )
    def organization_update_privacy_request(
        organization_id: str,
        privacy_request_id: str,
        payload: OrganizationPrivacyRequestStatusUpdate,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> OrganizationPrivacyRequestSummary:
        return update_privacy_request_summary(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            privacy_request_id,
            payload,
            request.app.state.store,
        )

    @app.get(
        "/organizations/{organization_id}/pilot-readiness",
        response_model=OrganizationPilotReadinessSummary,
        tags=["organizations"],
    )
    def organization_pilot_readiness(
        organization_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> OrganizationPilotReadinessSummary:
        return get_pilot_readiness_summary(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            request.app.state.store,
        )

    @app.patch(
        "/organizations/{organization_id}/pilot-readiness/{item_key}",
        response_model=OrganizationPilotReadinessSummary,
        tags=["organizations"],
    )
    def organization_update_pilot_readiness(
        organization_id: str,
        item_key: str,
        payload: OrganizationPilotReadinessUpdate,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> OrganizationPilotReadinessSummary:
        return update_pilot_readiness_item_summary(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            item_key,
            payload,
            request.app.state.store,
        )

    @app.get(
        "/organizations/{organization_id}/subscription",
        response_model=OrganizationSubscriptionSummary,
        tags=["billing"],
    )
    def organization_subscription(
        organization_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> OrganizationSubscriptionSummary:
        return get_subscription_summary(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            request.app.state.store,
        )

    @app.patch(
        "/organizations/{organization_id}/subscription",
        response_model=OrganizationSubscriptionSummary,
        tags=["billing"],
    )
    def organization_update_subscription(
        organization_id: str,
        payload: OrganizationSubscriptionUpdate,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> OrganizationSubscriptionSummary:
        return update_subscription_summary(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            payload,
            request.app.state.store,
        )

    @app.post(
        "/organizations/{organization_id}/billing/checkout",
        response_model=BillingCheckoutSessionSummary,
        tags=["billing"],
    )
    def organization_billing_checkout(
        organization_id: str,
        payload: BillingCheckoutCreate,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> BillingCheckoutSessionSummary:
        return create_billing_checkout_summary(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            payload,
            request.app.state.store,
        )

    @app.post(
        "/organizations/{organization_id}/billing/portal",
        response_model=BillingPortalSessionSummary,
        tags=["billing"],
    )
    def organization_billing_portal(
        organization_id: str,
        payload: BillingPortalCreate,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> BillingPortalSessionSummary:
        return create_billing_portal_summary(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            payload,
            request.app.state.store,
        )

    @app.get(
        "/organizations/{organization_id}/analysis-config",
        response_model=AnalysisConfigSummary,
        tags=["organizations"],
    )
    def organization_analysis_config(
        organization_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
        site_id: str | None = None,
    ) -> AnalysisConfigSummary:
        return get_analysis_config_summary(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            request.app.state.store,
            site_id=site_id,
        )

    @app.patch(
        "/organizations/{organization_id}/analysis-config",
        response_model=AnalysisConfigSummary,
        tags=["organizations"],
    )
    def organization_update_analysis_config(
        organization_id: str,
        payload: AnalysisConfigUpdate,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> AnalysisConfigSummary:
        return update_analysis_config_summary(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            payload,
            request.app.state.store,
        )

    @app.get(
        "/organizations/{organization_id}/data-connectors",
        response_model=list[DataConnectorSummary],
        tags=["organizations"],
    )
    def organization_data_connectors(
        organization_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
        site_id: str | None = None,
    ) -> list[DataConnectorSummary]:
        return list_data_connector_summaries(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            request.app.state.store,
            site_id=site_id,
        )

    @app.post(
        "/organizations/{organization_id}/data-connectors",
        response_model=DataConnectorSummary,
        tags=["organizations"],
    )
    def organization_create_data_connector(
        organization_id: str,
        payload: DataConnectorCreate,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> DataConnectorSummary:
        return create_data_connector_summary(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            payload,
            request.app.state.store,
        )

    @app.get(
        "/organizations/{organization_id}/data-connectors/due",
        response_model=list[DataConnectorSummary],
        tags=["organizations"],
    )
    def organization_due_data_connectors(
        organization_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
        due_at: str | None = None,
        limit: int = Query(50, ge=1, le=100),
    ) -> list[DataConnectorSummary]:
        return list_due_data_connector_summaries(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            request.app.state.store,
            due_at=due_at,
            limit=limit,
        )

    @app.get(
        "/organizations/{organization_id}/data-connectors/{connector_id}/syncs",
        response_model=list[DataConnectorSyncSummary],
        tags=["organizations"],
    )
    def organization_data_connector_syncs(
        organization_id: str,
        connector_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
        limit: int = Query(20, ge=1, le=100),
    ) -> list[DataConnectorSyncSummary]:
        return list_data_connector_sync_summaries(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            connector_id,
            request.app.state.store,
            limit=limit,
        )

    @app.post(
        "/organizations/{organization_id}/data-connectors/{connector_id}/syncs",
        response_model=DataConnectorSyncSummary,
        tags=["organizations"],
    )
    def organization_record_data_connector_sync(
        organization_id: str,
        connector_id: str,
        payload: DataConnectorSyncRecord,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> DataConnectorSyncSummary:
        return record_data_connector_sync_summary(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            connector_id,
            payload,
            request.app.state.store,
        )

    @app.post(
        "/organizations/{organization_id}/data-connectors/{connector_id}/imports",
        response_model=DataConnectorImportSummary,
        tags=["organizations"],
    )
    def organization_import_data_connector_content(
        organization_id: str,
        connector_id: str,
        payload: DataConnectorImportCreate,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> DataConnectorImportSummary:
        return import_data_connector_content_summary(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            connector_id,
            payload,
            request.app.state.store,
            request.app.state.artifact_store,
        )

    @app.patch(
        "/organizations/{organization_id}/data-connectors/{connector_id}",
        response_model=DataConnectorSummary,
        tags=["organizations"],
    )
    def organization_update_data_connector(
        organization_id: str,
        connector_id: str,
        payload: DataConnectorUpdate,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> DataConnectorSummary:
        return update_data_connector_summary(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            connector_id,
            payload,
            request.app.state.store,
        )

    @app.get(
        "/organizations/{organization_id}/sites",
        response_model=list[SiteSummary],
        tags=["organizations"],
    )
    def organization_sites(
        organization_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> list[SiteSummary]:
        return list_site_summaries(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            request.app.state.store,
        )

    @app.post(
        "/organizations/{organization_id}/sites",
        response_model=SiteSummary,
        tags=["organizations"],
    )
    def organization_create_site(
        organization_id: str,
        payload: SiteCreate,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> SiteSummary:
        return create_site_summary(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            payload,
            request.app.state.store,
        )

    @app.get(
        "/organizations/{organization_id}/memberships",
        response_model=list[MembershipSummary],
        tags=["organizations"],
    )
    def organization_memberships(
        organization_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> list[MembershipSummary]:
        return list_membership_summaries(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            request.app.state.store,
        )

    @app.get(
        "/organizations/{organization_id}/report-approvers",
        response_model=list[ReportApproverSummary],
        tags=["organizations"],
    )
    def organization_report_approvers(
        organization_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> list[ReportApproverSummary]:
        return list_report_approver_summaries(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            request.app.state.store,
        )

    @app.patch(
        "/organizations/{organization_id}/memberships/{member_user_id}",
        response_model=MembershipSummary,
        tags=["organizations"],
    )
    def organization_update_membership(
        organization_id: str,
        member_user_id: str,
        payload: MembershipRoleUpdate,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> MembershipSummary:
        return update_membership_summary(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            member_user_id,
            payload,
            request.app.state.store,
        )

    @app.delete(
        "/organizations/{organization_id}/memberships/{member_user_id}",
        response_model=MembershipSummary,
        tags=["organizations"],
    )
    def organization_remove_membership(
        organization_id: str,
        member_user_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> MembershipSummary:
        return remove_membership_summary(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            member_user_id,
            request.app.state.store,
        )

    @app.get(
        "/organizations/{organization_id}/invites",
        response_model=list[OrganizationInviteSummary],
        tags=["organizations"],
    )
    def organization_invites(
        organization_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> list[OrganizationInviteSummary]:
        return list_organization_invite_summaries(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            request.app.state.store,
        )

    @app.post(
        "/organizations/{organization_id}/invites",
        response_model=OrganizationInviteSummary,
        tags=["organizations"],
    )
    def organization_create_invite(
        organization_id: str,
        payload: OrganizationInviteCreate,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> OrganizationInviteSummary:
        return create_invite(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            payload,
            request.app.state.store,
        )

    @app.post(
        "/organizations/{organization_id}/invites/{invite_id}/revoke",
        response_model=OrganizationInviteSummary,
        tags=["organizations"],
    )
    def organization_revoke_invite(
        organization_id: str,
        invite_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> OrganizationInviteSummary:
        return revoke_invite(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            invite_id,
            request.app.state.store,
        )

    @app.get(
        "/organizations/{organization_id}/meters",
        response_model=list[MeterSummary],
        tags=["organizations"],
    )
    def organization_meters(
        organization_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
        site_id: str | None = None,
    ) -> list[MeterSummary]:
        return list_meter_summaries(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            request.app.state.store,
            site_id=site_id,
        )

    @app.post(
        "/organizations/{organization_id}/meters",
        response_model=MeterSummary,
        tags=["organizations"],
    )
    def organization_create_meter(
        organization_id: str,
        payload: MeterCreate,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> MeterSummary:
        return create_meter_summary(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            payload,
            request.app.state.store,
        )

    @app.get(
        "/organizations/{organization_id}/uploads",
        response_model=list[UploadSummary],
        tags=["uploads"],
    )
    def organization_uploads(
        organization_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
        site_id: str | None = None,
    ) -> list[UploadSummary]:
        return list_upload_summaries(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            request.app.state.store,
            site_id=site_id,
        )

    @app.get(
        "/organizations/{organization_id}/uploads/readiness",
        response_model=UploadReadinessSummary,
        tags=["uploads"],
    )
    def organization_upload_readiness(
        organization_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
        site_id: str | None = None,
    ) -> UploadReadinessSummary:
        return get_upload_readiness_summary(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            request.app.state.store,
            site_id=site_id,
        )

    @app.post(
        "/organizations/{organization_id}/uploads",
        response_model=UploadSummary,
        tags=["uploads"],
    )
    def organization_create_upload(
        organization_id: str,
        payload: UploadCreate,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> UploadSummary:
        return create_upload_summary(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            payload,
            request.app.state.store,
        )

    @app.post(
        "/organizations/{organization_id}/uploads/content",
        response_model=UploadSummary,
        tags=["uploads"],
    )
    def organization_create_upload_content(
        organization_id: str,
        payload: UploadContentCreate,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> UploadSummary:
        return create_upload_content_summary(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            payload,
            request.app.state.store,
            request.app.state.artifact_store,
        )

    @app.get(
        "/organizations/{organization_id}/runs",
        response_model=list[RunSummary],
        tags=["runs"],
    )
    def organization_runs(
        organization_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
        site_id: str | None = None,
    ) -> list[RunSummary]:
        return list_run_summaries(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            request.app.state.store,
            site_id=site_id,
        )

    @app.post(
        "/organizations/{organization_id}/runs",
        response_model=RunSummary,
        tags=["runs"],
    )
    def organization_create_run(
        organization_id: str,
        payload: RunCreate,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> RunSummary:
        return create_run_summary(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            payload,
            request.app.state.store,
        )

    @app.get(
        "/organizations/{organization_id}/analysis-jobs",
        response_model=list[AnalysisJobSummary],
        tags=["runs"],
    )
    def organization_analysis_jobs(
        organization_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> list[AnalysisJobSummary]:
        return list_analysis_job_summaries(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            request.app.state.store,
        )

    @app.get(
        "/organizations/{organization_id}/notifications",
        response_model=list[NotificationSummary],
        tags=["notifications"],
    )
    def organization_notifications(
        organization_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> list[NotificationSummary]:
        return list_notification_summaries(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            request.app.state.store,
        )

    @app.get(
        "/organizations/{organization_id}/reports",
        response_model=list[ReportSummary],
        tags=["reports"],
    )
    def organization_reports(
        organization_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
        run_id: str | None = None,
    ) -> list[ReportSummary]:
        return list_report_summaries(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            request.app.state.store,
            run_id=run_id,
        )

    @app.post(
        "/organizations/{organization_id}/reports/retention/apply",
        response_model=ReportRetentionApplyResponse,
        tags=["reports"],
    )
    def organization_apply_report_retention(
        organization_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> ReportRetentionApplyResponse:
        return apply_report_retention_summary(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            request.app.state.store,
            request.app.state.artifact_store,
        )

    @app.get(
        "/organizations/{organization_id}/reports/{report_id}/artifact",
        response_model=ReportArtifactResponse,
        tags=["reports"],
    )
    def organization_report_artifact(
        organization_id: str,
        report_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> ReportArtifactResponse:
        return get_report_artifact(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            report_id,
            request.app.state.store,
            request.app.state.artifact_store,
        )

    @app.get(
        "/organizations/{organization_id}/reports/{report_id}/export",
        tags=["reports"],
    )
    def organization_report_export(
        organization_id: str,
        report_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> PlainTextResponse:
        return export_report_artifact(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            report_id,
            request.app.state.store,
            request.app.state.artifact_store,
        )

    @app.get(
        "/organizations/{organization_id}/reports/{report_id}/evidence-package",
        tags=["reports"],
    )
    def organization_report_evidence_package(
        organization_id: str,
        report_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> PlainTextResponse:
        return export_report_evidence_package(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            report_id,
            request.app.state.store,
            request.app.state.artifact_store,
        )

    @app.get(
        "/organizations/{organization_id}/reports/{report_id}/evidence-package/pdf",
        tags=["reports"],
    )
    def organization_report_evidence_package_pdf(
        organization_id: str,
        report_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> Response:
        return export_report_evidence_package_pdf(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            report_id,
            request.app.state.store,
            request.app.state.artifact_store,
        )

    @app.get(
        "/organizations/{organization_id}/reports/{report_id}/evidence-packages",
        response_model=list[ReportEvidencePackageSummary],
        tags=["reports"],
    )
    def organization_report_evidence_packages(
        organization_id: str,
        report_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> list[ReportEvidencePackageSummary]:
        return list_report_evidence_package_summaries(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            report_id,
            request.app.state.store,
        )

    @app.post(
        "/organizations/{organization_id}/reports/{report_id}/evidence-packages",
        response_model=ReportEvidencePackageSummary,
        status_code=status.HTTP_201_CREATED,
        tags=["reports"],
    )
    def organization_create_report_evidence_package(
        organization_id: str,
        report_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> ReportEvidencePackageSummary:
        return create_report_evidence_package_snapshot(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            report_id,
            request.app.state.store,
            request.app.state.artifact_store,
        )

    @app.get(
        "/organizations/{organization_id}/reports/{report_id}/evidence-packages/{package_id}/download",
        tags=["reports"],
    )
    def organization_report_evidence_package_download(
        organization_id: str,
        report_id: str,
        package_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> PlainTextResponse:
        return download_report_evidence_package_snapshot(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            report_id,
            package_id,
            request.app.state.store,
            request.app.state.artifact_store,
        )

    @app.get(
        "/organizations/{organization_id}/reports/{report_id}/reviews",
        response_model=list[ReportReviewSummary],
        tags=["reports"],
    )
    def organization_report_reviews(
        organization_id: str,
        report_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> list[ReportReviewSummary]:
        return list_report_review_summaries(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            report_id,
            request.app.state.store,
        )

    @app.post(
        "/organizations/{organization_id}/reports/{report_id}/reviews",
        response_model=ReportReviewSummary,
        tags=["reports"],
    )
    def organization_create_report_review(
        organization_id: str,
        report_id: str,
        payload: ReportReviewCreate,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> ReportReviewSummary:
        return create_report_review_summary(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            report_id,
            payload,
            request.app.state.store,
        )

    @app.get(
        "/organizations/{organization_id}/reports/{report_id}/approval-assignments",
        response_model=list[ReportApprovalAssignmentSummary],
        tags=["reports"],
    )
    def organization_report_approval_assignments(
        organization_id: str,
        report_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> list[ReportApprovalAssignmentSummary]:
        return list_report_approval_assignment_summaries(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            report_id,
            request.app.state.store,
        )

    @app.post(
        "/organizations/{organization_id}/reports/{report_id}/approval-assignments",
        response_model=ReportApprovalAssignmentSummary,
        tags=["reports"],
    )
    def organization_create_report_approval_assignment(
        organization_id: str,
        report_id: str,
        payload: ReportApprovalAssignmentCreate,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> ReportApprovalAssignmentSummary:
        return create_report_approval_assignment_summary(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            report_id,
            payload,
            request.app.state.store,
            request.app.state.artifact_store,
        )

    @app.patch(
        "/organizations/{organization_id}/reports/{report_id}/approval-assignments/{assignment_id}",
        response_model=ReportApprovalAssignmentSummary,
        tags=["reports"],
    )
    def organization_update_report_approval_assignment(
        organization_id: str,
        report_id: str,
        assignment_id: str,
        payload: ReportApprovalAssignmentUpdate,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> ReportApprovalAssignmentSummary:
        return update_report_approval_assignment_summary(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            report_id,
            assignment_id,
            payload,
            request.app.state.store,
            request.app.state.artifact_store,
        )

    @app.get(
        "/organizations/{organization_id}/reports/{report_id}/schedules",
        response_model=list[ReportScheduleSummary],
        tags=["reports"],
    )
    def organization_report_schedules(
        organization_id: str,
        report_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> list[ReportScheduleSummary]:
        return list_report_schedule_summaries(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            report_id,
            request.app.state.store,
        )

    @app.post(
        "/organizations/{organization_id}/reports/{report_id}/schedules",
        response_model=ReportScheduleSummary,
        tags=["reports"],
    )
    def organization_create_report_schedule(
        organization_id: str,
        report_id: str,
        payload: ReportScheduleCreate,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> ReportScheduleSummary:
        return create_report_schedule_summary(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            report_id,
            payload,
            request.app.state.store,
        )

    @app.patch(
        "/organizations/{organization_id}/reports/{report_id}/schedules/{schedule_id}",
        response_model=ReportScheduleSummary,
        tags=["reports"],
    )
    def organization_update_report_schedule(
        organization_id: str,
        report_id: str,
        schedule_id: str,
        payload: ReportScheduleUpdate,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> ReportScheduleSummary:
        return update_report_schedule_summary(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            report_id,
            schedule_id,
            payload,
            request.app.state.store,
        )

    @app.get(
        "/organizations/{organization_id}/reports/{report_id}/schedules/{schedule_id}/deliveries",
        response_model=list[ReportScheduleDeliverySummary],
        tags=["reports"],
    )
    def organization_report_schedule_deliveries(
        organization_id: str,
        report_id: str,
        schedule_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> list[ReportScheduleDeliverySummary]:
        return list_report_schedule_delivery_summaries(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            report_id,
            schedule_id,
            request.app.state.store,
        )

    @app.get(
        "/organizations/{organization_id}/reports/{report_id}/comments",
        response_model=list[ReportCommentSummary],
        tags=["reports"],
    )
    def organization_report_comments(
        organization_id: str,
        report_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> list[ReportCommentSummary]:
        return list_report_comment_summaries(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            report_id,
            request.app.state.store,
        )

    @app.post(
        "/organizations/{organization_id}/reports/{report_id}/comments",
        response_model=ReportCommentSummary,
        tags=["reports"],
    )
    def organization_create_report_comment(
        organization_id: str,
        report_id: str,
        payload: ReportCommentCreate,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> ReportCommentSummary:
        return create_report_comment_summary(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            report_id,
            payload,
            request.app.state.store,
        )

    @app.get(
        "/organizations/{organization_id}/audit-events",
        response_model=list[AuditEventSummary],
        tags=["audit"],
    )
    def organization_audit_events(
        organization_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> list[AuditEventSummary]:
        return list_audit_event_summaries(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            request.app.state.store,
        )

    @app.get(
        "/organizations/{organization_id}/audit-events/export",
        response_model=AuditExportResponse,
        tags=["audit"],
    )
    def organization_audit_events_export(
        organization_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> AuditExportResponse:
        return export_audit_events(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            request.app.state.store,
        )

    @app.post(
        "/organizations/{organization_id}/runs/execute-local",
        response_model=LocalAnalysisRunResponse,
        tags=["runs"],
    )
    def organization_execute_local_run(
        organization_id: str,
        payload: LocalAnalysisRunCreate,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> LocalAnalysisRunResponse:
        return execute_local_analysis_run(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            payload,
            request.app.state.store,
            request.app.state.artifact_store,
        )

    @app.post(
        "/organizations/{organization_id}/runs/execute-upload",
        response_model=LocalAnalysisRunResponse,
        tags=["runs"],
    )
    def organization_execute_stored_upload_run(
        organization_id: str,
        payload: StoredUploadAnalysisRunCreate,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> LocalAnalysisRunResponse:
        return execute_stored_upload_analysis_run(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            payload,
            request.app.state.store,
            request.app.state.artifact_store,
        )

    @app.post(
        "/organizations/{organization_id}/runs/queue-upload",
        response_model=QueuedAnalysisRunResponse,
        tags=["runs"],
    )
    def organization_queue_stored_upload_run(
        organization_id: str,
        payload: StoredUploadAnalysisRunCreate,
        request: Request,
        background_tasks: BackgroundTasks,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> QueuedAnalysisRunResponse:
        response = queue_stored_upload_analysis_run(
            user_id_for_organization_route(user, organization_id),
            organization_id,
            payload,
            request.app.state.store,
        )
        background_tasks.add_task(
            process_analysis_job_background,
            request.app.state.store,
            request.app.state.artifact_store,
            response.job_id,
            request.app.state.email_sender,
        )
        return response

    @app.post(
        "/invites/{invite_id}/accept",
        response_model=OrganizationInviteSummary,
        tags=["organizations"],
    )
    def organization_accept_invite(
        invite_id: str,
        payload: OrganizationInviteAccept,
        request: Request,
    ) -> OrganizationInviteSummary:
        return accept_invite(invite_id, payload, request.app.state.store)

    return app


app = create_app()
