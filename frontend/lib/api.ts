export type OrganizationSummary = {
  id: string;
  name: string;
  slug: string;
};

export type OwnerOrganizationCreatePayload = {
  user_id: string;
  email: string;
  display_name: string;
  organization_id: string;
  organization_name: string;
  organization_slug: string;
};

export type OwnerOrganizationResponse = {
  user_id: string;
  organization_id: string;
  role: string;
};

export type OrganizationProfileUpdatePayload = {
  name: string;
  slug: string;
};

export type OrganizationReportBrandingSummary = {
  organization_id: string;
  evidence_pack_title: string;
  primary_color: string;
  logo_url: string | null;
  support_email: string | null;
  updated_at: string | null;
};

export type OrganizationReportBrandingUpdatePayload = {
  evidence_pack_title: string;
  primary_color: string;
  logo_url?: string | null;
  support_email?: string | null;
};

export type OrganizationReportApprovalPolicySummary = {
  organization_id: string;
  auto_lock_evidence_on_final_approval: boolean;
  minimum_approval_assignments_for_lock: number;
  updated_by_user_id: string | null;
  updated_at: string | null;
};

export type OrganizationReportApprovalPolicyUpdatePayload = {
  auto_lock_evidence_on_final_approval: boolean;
  minimum_approval_assignments_for_lock: number;
};

export type OrganizationEvidencePackageTemplateSummary = {
  organization_id: string;
  template_name: string;
  include_report_artifact: boolean;
  include_uploads: boolean;
  include_approval_assignments: boolean;
  include_reviews: boolean;
  include_comments: boolean;
  include_schedules: boolean;
  section_labels: Record<string, string>;
  section_order: string[];
  updated_by_user_id: string | null;
  updated_at: string | null;
};

export type OrganizationEvidencePackageTemplateUpdatePayload = {
  template_name: string;
  include_report_artifact: boolean;
  include_uploads: boolean;
  include_approval_assignments: boolean;
  include_reviews: boolean;
  include_comments: boolean;
  include_schedules: boolean;
  section_labels: Record<string, string>;
  section_order: string[];
};

export type OrganizationPrivacySettingsSummary = {
  organization_id: string;
  data_processing_acknowledged_at: string | null;
  data_processing_acknowledged_by_user_id: string | null;
  data_processing_contact_email: string | null;
  subprocessor_notifications_enabled: boolean;
  analytics_data_sharing_enabled: boolean;
  updated_at: string | null;
};

export type OrganizationPrivacySettingsUpdatePayload = {
  data_processing_contact_email: string | null;
  subprocessor_notifications_enabled: boolean;
  analytics_data_sharing_enabled: boolean;
  acknowledge_data_processing?: boolean;
};

export type OrganizationPrivacyRequestType = "data_export" | "deletion" | "correction";

export type OrganizationPrivacyRequestStatus = "open" | "in_review" | "completed" | "rejected";

export type OrganizationPrivacyRequestSummary = {
  id: string;
  organization_id: string;
  requested_by_user_id: string;
  subject_user_id: string | null;
  request_type: OrganizationPrivacyRequestType;
  status: OrganizationPrivacyRequestStatus;
  description: string | null;
  resolution_notes: string | null;
  completed_at: string | null;
  created_at: string;
  updated_at: string;
};

export type OrganizationPrivacyRequestCreatePayload = {
  request_type: OrganizationPrivacyRequestType;
  subject_user_id?: string | null;
  description?: string | null;
};

export type OrganizationPrivacyRequestStatusUpdatePayload = {
  status: OrganizationPrivacyRequestStatus;
  resolution_notes?: string | null;
};

export type OrganizationPilotReadinessStatus =
  | "pending"
  | "in_progress"
  | "verified"
  | "blocked";

export type OrganizationPilotReadinessItemSummary = {
  item_key: string;
  label: string;
  status: OrganizationPilotReadinessStatus;
  evidence_url: string | null;
  notes: string | null;
  updated_by_user_id: string | null;
  updated_at: string | null;
};

export type OrganizationPilotReadinessSummary = {
  organization_id: string;
  verified_count: number;
  total_count: number;
  items: OrganizationPilotReadinessItemSummary[];
};

export type OrganizationPilotReadinessUpdatePayload = {
  status: OrganizationPilotReadinessStatus;
  evidence_url?: string | null;
  notes?: string | null;
};

export type AnalysisConfigSummary = {
  organization_id: string;
  config: Record<string, unknown>;
  source: string;
  updated_by_user_id: string | null;
  updated_at: string | null;
};

export type AnalysisConfigUpdatePayload = {
  config: Record<string, unknown>;
};

export type DataConnectorType = "sftp" | "api" | "utility_portal";

export type DataConnectorStatus = "draft" | "active" | "paused";

export type DataConnectorSchedule = "manual" | "daily" | "weekly" | "monthly";

export type DataConnectorSyncStatus = "succeeded" | "failed" | "skipped";

export type DataConnectorSummary = {
  id: string;
  organization_id: string;
  site_id: string;
  name: string;
  connector_type: DataConnectorType;
  source_category: string;
  status: DataConnectorStatus;
  schedule: DataConnectorSchedule;
  external_reference: string | null;
  credential_reference_configured: boolean;
  created_by_user_id: string;
  updated_by_user_id: string | null;
  last_sync_at: string | null;
  last_sync_status: DataConnectorSyncStatus | null;
  last_sync_error: string | null;
  next_sync_at: string | null;
  created_at: string;
  updated_at: string;
};

export type DataConnectorCreatePayload = {
  connector_id: string;
  site_id: string;
  name: string;
  connector_type: DataConnectorType;
  source_category: string;
  status: DataConnectorStatus;
  schedule: DataConnectorSchedule;
  external_reference?: string | null;
  credential_reference?: string | null;
  next_sync_at?: string | null;
};

export type DataConnectorUpdatePayload = {
  name: string;
  status: DataConnectorStatus;
  schedule: DataConnectorSchedule;
  external_reference?: string | null;
  credential_reference?: string | null;
  next_sync_at?: string | null;
};

export type DataConnectorSyncSummary = {
  id: string;
  organization_id: string;
  connector_id: string;
  site_id: string;
  source_category: string;
  status: DataConnectorSyncStatus;
  upload_id: string | null;
  records_processed: number;
  status_reason: string | null;
  triggered_by_user_id: string;
  started_at: string;
  completed_at: string | null;
  created_at: string;
};

export type DataConnectorSyncRecordPayload = {
  status: DataConnectorSyncStatus;
  upload_id?: string | null;
  records_processed?: number;
  status_reason?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
};

export type DataConnectorImportPayload = {
  upload_id?: string | null;
  filename: string;
  content_base64: string;
  records_processed?: number;
  status_reason?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
};

export type DataConnectorImportSummary = {
  connector: DataConnectorSummary;
  upload: UploadSummary;
  sync: DataConnectorSyncSummary;
};

export type SubscriptionPlanSummary = {
  id: string;
  name: string;
  description: string;
  monthly_price_cents: number | null;
  included_users: number | null;
  included_sites: number | null;
  monthly_run_limit: number | null;
  report_retention_days: number | null;
  is_active: boolean;
};

export type SubscriptionUsageSummary = {
  member_count: number;
  active_user_count: number;
  pending_invite_count: number;
  site_count: number;
  runs_this_period: number;
};

export type OrganizationSubscriptionSummary = {
  organization_id: string;
  plan: SubscriptionPlanSummary;
  status: string;
  current_period_start: string | null;
  current_period_end: string | null;
  trial_ends_at: string | null;
  external_customer_id: string | null;
  external_subscription_id: string | null;
  usage: SubscriptionUsageSummary;
};

export type OrganizationSubscriptionUpdatePayload = {
  plan_id: string;
};

export type BillingCheckoutCreatePayload = {
  plan_id: string;
  success_url?: string;
  cancel_url?: string;
};

export type BillingCheckoutSessionSummary = {
  organization_id: string;
  plan_id: string;
  provider: string;
  checkout_session_id: string;
  checkout_url: string;
};

export type BillingPortalCreatePayload = {
  return_url?: string;
};

export type BillingPortalSessionSummary = {
  organization_id: string;
  provider: string;
  portal_session_id: string;
  portal_url: string;
};

export type DevSessionCreatePayload = {
  user_id: string;
  organization_id?: string;
};

export type DevSessionResponse = {
  user_id: string;
  organization_id: string | null;
  role: string | null;
  auth_token: string;
};

export type MarketingInquiryPayload = {
  name: string;
  email: string;
  organization_name?: string | null;
  role?: string | null;
  intended_use: string;
  message: string;
  source_page?: string | null;
  website?: string | null;
};

export type MarketingInquirySummary = {
  id: string;
  status: "new" | "contacted" | "qualified" | "closed";
  created_at: string;
};

export type ClerkSyncResponse = {
  user_id: string;
  organization_id: string;
  organization_slug: string;
  role: string;
};

export type SiteSummary = {
  id: string;
  organization_id: string;
  name: string;
  timezone: string;
};

export type SiteCreatePayload = {
  site_id: string;
  name: string;
  timezone: string;
};

export type MembershipSummary = {
  user_id: string;
  organization_id: string;
  role: string;
  invited_by_user_id: string | null;
  email: string;
  display_name: string;
  is_active: boolean;
};

export type ReportApproverSummary = {
  user_id: string;
  organization_id: string;
  role: "owner" | "manager";
  email: string;
  display_name: string;
};

export type OrganizationInviteSummary = {
  id: string;
  organization_id: string;
  email: string;
  role: string;
  invited_by_user_id: string;
  status: string;
  accepted_by_user_id: string | null;
  accepted_at: string | null;
};

export type OrganizationInviteCreatePayload = {
  email: string;
  role: "owner" | "manager" | "viewer";
};

export type MembershipRoleUpdatePayload = {
  role: "owner" | "manager" | "viewer";
};

export type AcceptInvitePayload = {
  user_id: string;
  email: string;
  display_name: string;
};

export type MeterSummary = {
  id: string;
  organization_id: string;
  site_id: string;
  display_name: string;
  commodity: string;
  unit: string;
  source_column: string;
  is_seu: boolean;
};

export type MeterCreatePayload = {
  meter_id: string;
  site_id: string;
  display_name: string;
  commodity: string;
  unit: string;
  source_column: string;
  is_seu?: boolean;
};

export type LocalAnalysisRunPayload = {
  site_id: string;
  upload_id: string;
  run_id: string;
  report_id: string;
  filename: string;
  rows: Array<Record<string, unknown>>;
  client_config?: Record<string, unknown> | null;
};

export type StoredUploadAnalysisRunPayload = {
  upload_id: string;
  run_id: string;
  report_id: string;
  source_upload_ids?: string[];
  client_config?: Record<string, unknown> | null;
};

export type LocalAnalysisRunResponse = {
  upload_id: string;
  run_upload_ids: string[];
  run_id: string;
  report_id: string;
  run_status: string;
  iso_summary: Record<string, unknown>;
};

export type QueuedAnalysisRunResponse = {
  job_id: string;
  upload_id: string;
  run_upload_ids: string[];
  run_id: string;
  report_id: string;
  run_status: string;
};

export type ReportSummary = {
  id: string;
  organization_id: string;
  run_id: string;
  report_type: string;
  is_published: boolean;
  expires_at: string | null;
  retention_applied_at: string | null;
};

export type ReportArtifactResponse = {
  report_id: string;
  organization_id: string;
  content: Record<string, unknown>;
};

export type ReportRetentionApplyResponse = {
  organization_id: string;
  evaluated_count: number;
  expired_count: number;
  deleted_artifact_count: number;
  missing_artifact_count: number;
  deleted_evidence_package_artifact_count: number;
  missing_evidence_package_artifact_count: number;
  deleted_evidence_package_metadata_count: number;
  retained_report_ids: string[];
};

export type ReportExportFile = {
  filename: string;
  content_type: string;
  content: string;
};

export type ReportBinaryExportFile = {
  filename: string;
  content_type: string;
  content: ArrayBuffer;
};

export type ReportEvidencePackageSummary = {
  id: string;
  organization_id: string;
  report_id: string;
  created_by_user_id: string;
  filename: string;
  content_type: string;
  content_length: number;
  package_type: string;
  format: string;
  comment_count: number;
  review_count: number;
  schedule_count: number;
  delivery_count: number;
  upload_count: number;
  created_at: string;
};

export type ReportCommentSummary = {
  id: string;
  organization_id: string;
  report_id: string;
  author_user_id: string;
  body: string;
  created_at: string;
};

export type ReportCommentCreatePayload = {
  body: string;
};

export type ReportReviewStatus = "approved" | "changes_requested";

export type ReportReviewSummary = {
  id: string;
  organization_id: string;
  report_id: string;
  reviewer_user_id: string;
  status: ReportReviewStatus;
  notes: string | null;
  created_at: string;
};

export type ReportReviewCreatePayload = {
  status: ReportReviewStatus;
  notes?: string | null;
};

export type ReportApprovalAssignmentStatus =
  | "pending"
  | "approved"
  | "changes_requested"
  | "canceled";

export type ReportApprovalAssignmentSummary = {
  id: string;
  organization_id: string;
  report_id: string;
  assigned_to_user_id: string;
  assigned_by_user_id: string;
  status: ReportApprovalAssignmentStatus;
  due_at?: string | null;
  notes?: string | null;
  completed_at?: string | null;
  created_at: string;
  updated_at: string;
};

export type ReportApprovalAssignmentCreatePayload = {
  assigned_to_user_id: string;
  due_at?: string | null;
  notes?: string | null;
};

export type ReportApprovalAssignmentUpdatePayload = {
  status: ReportApprovalAssignmentStatus;
  notes?: string | null;
};

export type ReportScheduleFrequency = "weekly" | "monthly" | "quarterly";
export type ReportScheduleDeliveryStatus = "sent" | "failed";

export type ReportScheduleSummary = {
  id: string;
  organization_id: string;
  report_id: string;
  created_by_user_id: string;
  frequency: ReportScheduleFrequency;
  recipient_email: string;
  next_run_at: string;
  is_active: boolean;
  last_delivery_at?: string | null;
  last_delivery_status?: ReportScheduleDeliveryStatus | null;
  last_delivery_error?: string | null;
  created_at: string;
  updated_at: string;
};

export type ReportScheduleDeliverySummary = {
  id: string;
  organization_id: string;
  schedule_id: string;
  report_id: string;
  recipient_email: string;
  status: ReportScheduleDeliveryStatus;
  error_message?: string | null;
  attempted_at: string;
  sent_at?: string | null;
  created_at: string;
};

export type ReportScheduleCreatePayload = {
  frequency: ReportScheduleFrequency;
  recipient_email: string;
  next_run_at: string;
};

export type ReportScheduleUpdatePayload = {
  frequency?: ReportScheduleFrequency;
  recipient_email?: string;
  next_run_at?: string;
  is_active?: boolean;
};

export type UploadSummary = {
  id: string;
  organization_id: string;
  site_id: string;
  uploaded_by_user_id: string;
  category: string;
  checksum: string;
  status: string;
};

export type UploadCreatePayload = {
  upload_id: string;
  site_id: string;
  category: string;
  filename: string;
  checksum: string;
};

export type UploadContentCreatePayload = {
  upload_id: string;
  site_id: string;
  category: string;
  filename: string;
  content_base64: string;
};

export type UploadReadinessFileSummary = {
  category: string;
  label: string;
  is_required: boolean;
  is_ready: boolean;
  latest_upload: UploadSummary | null;
};

export type SiteUploadReadinessSummary = {
  site_id: string;
  site_name: string;
  ready_count: number;
  required_count: number;
  is_ready: boolean;
  files: UploadReadinessFileSummary[];
};

export type UploadReadinessSummary = {
  organization_id: string;
  ready_site_count: number;
  site_count: number;
  required_categories: string[];
  sites: SiteUploadReadinessSummary[];
};

export type RunSummary = {
  id: string;
  organization_id: string;
  site_id: string;
  requested_by_user_id: string;
  status: string;
  error_message: string | null;
  completed_at: string | null;
};

export type AnalysisJobSummary = {
  id: string;
  organization_id: string;
  run_id: string;
  upload_id: string;
  report_id: string;
  requested_by_user_id: string;
  status: string;
  error_message: string | null;
  completed_at: string | null;
};

export type NotificationSummary = {
  id: string;
  organization_id: string;
  recipient_user_id: string;
  channel: string;
  event_type: string;
  subject: string;
  body: string;
  resource_type: string;
  resource_id: string;
  status: string;
  error_message: string | null;
  sent_at: string | null;
  attempt_count: number;
  last_attempt_at: string | null;
  next_attempt_at: string | null;
};

export type RunCreatePayload = {
  run_id: string;
  site_id: string;
  upload_ids: string[];
};

export type AuditEventSummary = {
  id: string;
  organization_id: string;
  created_at: string;
  actor_user_id: string;
  action: string;
  resource_type: string;
  resource_id: string;
  metadata_json: string;
};

export type AuditExportResponse = {
  filename: string;
  content_type: string;
  event_count: number;
  content: string;
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const AUTH_TOKEN =
  process.env.NEXT_PUBLIC_DEMO_AUTH_TOKEN ?? process.env.DEMO_AUTH_TOKEN ?? null;

export class ApiRequestError extends Error {
  status: number;
  statusText: string;
  retryAfterSeconds: number | null;

  constructor(response: Response) {
    super(`API request failed: ${response.status} ${response.statusText}`);
    this.name = "ApiRequestError";
    this.status = response.status;
    this.statusText = response.statusText;
    this.retryAfterSeconds = retryAfterSecondsFromHeader(response.headers.get("retry-after"));
  }
}

function retryAfterSecondsFromHeader(value: string | null): number | null {
  if (!value) {
    return null;
  }
  const parsedSeconds = Number(value);
  if (Number.isFinite(parsedSeconds) && parsedSeconds >= 0) {
    return Math.ceil(parsedSeconds);
  }
  const parsedDateMs = Date.parse(value);
  if (Number.isNaN(parsedDateMs)) {
    return null;
  }
  return Math.max(0, Math.ceil((parsedDateMs - Date.now()) / 1000));
}

function buildAuthHeaders(userId: string): HeadersInit {
  if (AUTH_TOKEN && AUTH_TOKEN.trim()) {
    return {
      "Content-Type": "application/json",
      Authorization: `Bearer ${AUTH_TOKEN.trim()}`
    };
  }

  return {
    "Content-Type": "application/json",
    "X-User-Id": userId
  };
}

function buildRequestHeaders(userId: string, authToken?: string | null): HeadersInit {
  if (authToken && authToken.trim()) {
    return {
      "Content-Type": "application/json",
      Authorization: `Bearer ${authToken.trim()}`
    };
  }
  return buildAuthHeaders(userId);
}

async function apiFetch<T>(
  path: string,
  userId: string,
  init?: RequestInit,
  authToken?: string | null
): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      ...buildRequestHeaders(userId, authToken),
      ...(init?.headers ?? {})
    }
  });

  if (!response.ok) {
    throw new ApiRequestError(response);
  }

  return response.json() as Promise<T>;
}

function filenameFromContentDisposition(value: string | null, fallback: string): string {
  if (!value) {
    return fallback;
  }
  const match = value.match(/filename="([^"]+)"/i) ?? value.match(/filename=([^;]+)/i);
  return match?.[1]?.trim() || fallback;
}

async function apiFetchText(
  path: string,
  userId: string,
  init?: RequestInit,
  authToken?: string | null
): Promise<ReportExportFile> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      ...buildRequestHeaders(userId, authToken),
      ...(init?.headers ?? {})
    }
  });

  if (!response.ok) {
    throw new ApiRequestError(response);
  }

  return {
    filename: filenameFromContentDisposition(
      response.headers.get("content-disposition"),
      "report-export.json"
    ),
    content_type: response.headers.get("content-type") ?? "application/json",
    content: await response.text()
  };
}

async function apiFetchBinary(
  path: string,
  userId: string,
  init?: RequestInit,
  authToken?: string | null
): Promise<ReportBinaryExportFile> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      ...buildRequestHeaders(userId, authToken),
      ...(init?.headers ?? {})
    }
  });

  if (!response.ok) {
    throw new ApiRequestError(response);
  }

  return {
    filename: filenameFromContentDisposition(
      response.headers.get("content-disposition"),
      "report-evidence-package.pdf"
    ),
    content_type: response.headers.get("content-type") ?? "application/pdf",
    content: await response.arrayBuffer()
  };
}

export function getMyOrganizations(
  userId: string,
  authToken?: string | null
): Promise<OrganizationSummary[]> {
  return apiFetch<OrganizationSummary[]>("/me/organizations", userId, undefined, authToken);
}

export function getOrganizationProfile(
  userId: string,
  organizationId: string,
  authToken?: string | null
): Promise<OrganizationSummary> {
  return apiFetch<OrganizationSummary>(`/organizations/${organizationId}`, userId, undefined, authToken);
}

export function updateOrganizationProfile(
  userId: string,
  organizationId: string,
  payload: OrganizationProfileUpdatePayload,
  authToken?: string | null
): Promise<OrganizationSummary> {
  return apiFetch<OrganizationSummary>(
    `/organizations/${organizationId}`,
    userId,
    {
      method: "PATCH",
      body: JSON.stringify(payload)
    },
    authToken
  );
}

export function getOrganizationReportBranding(
  userId: string,
  organizationId: string,
  authToken?: string | null
): Promise<OrganizationReportBrandingSummary> {
  return apiFetch<OrganizationReportBrandingSummary>(
    `/organizations/${organizationId}/report-branding`,
    userId,
    undefined,
    authToken
  );
}

export function updateOrganizationReportBranding(
  userId: string,
  organizationId: string,
  payload: OrganizationReportBrandingUpdatePayload,
  authToken?: string | null
): Promise<OrganizationReportBrandingSummary> {
  return apiFetch<OrganizationReportBrandingSummary>(
    `/organizations/${organizationId}/report-branding`,
    userId,
    {
      method: "PATCH",
      body: JSON.stringify(payload)
    },
    authToken
  );
}

export function getOrganizationReportApprovalPolicy(
  userId: string,
  organizationId: string,
  authToken?: string | null
): Promise<OrganizationReportApprovalPolicySummary> {
  return apiFetch<OrganizationReportApprovalPolicySummary>(
    `/organizations/${organizationId}/report-approval-policy`,
    userId,
    undefined,
    authToken
  );
}

export function updateOrganizationReportApprovalPolicy(
  userId: string,
  organizationId: string,
  payload: OrganizationReportApprovalPolicyUpdatePayload,
  authToken?: string | null
): Promise<OrganizationReportApprovalPolicySummary> {
  return apiFetch<OrganizationReportApprovalPolicySummary>(
    `/organizations/${organizationId}/report-approval-policy`,
    userId,
    {
      method: "PATCH",
      body: JSON.stringify(payload)
    },
    authToken
  );
}

export function getOrganizationEvidencePackageTemplate(
  userId: string,
  organizationId: string,
  authToken?: string | null
): Promise<OrganizationEvidencePackageTemplateSummary> {
  return apiFetch<OrganizationEvidencePackageTemplateSummary>(
    `/organizations/${organizationId}/evidence-package-template`,
    userId,
    undefined,
    authToken
  );
}

export function updateOrganizationEvidencePackageTemplate(
  userId: string,
  organizationId: string,
  payload: OrganizationEvidencePackageTemplateUpdatePayload,
  authToken?: string | null
): Promise<OrganizationEvidencePackageTemplateSummary> {
  return apiFetch<OrganizationEvidencePackageTemplateSummary>(
    `/organizations/${organizationId}/evidence-package-template`,
    userId,
    {
      method: "PATCH",
      body: JSON.stringify(payload)
    },
    authToken
  );
}

export function getOrganizationPrivacySettings(
  userId: string,
  organizationId: string,
  authToken?: string | null
): Promise<OrganizationPrivacySettingsSummary> {
  return apiFetch<OrganizationPrivacySettingsSummary>(
    `/organizations/${organizationId}/privacy-settings`,
    userId,
    undefined,
    authToken
  );
}

export function updateOrganizationPrivacySettings(
  userId: string,
  organizationId: string,
  payload: OrganizationPrivacySettingsUpdatePayload,
  authToken?: string | null
): Promise<OrganizationPrivacySettingsSummary> {
  return apiFetch<OrganizationPrivacySettingsSummary>(
    `/organizations/${organizationId}/privacy-settings`,
    userId,
    {
      method: "PATCH",
      body: JSON.stringify(payload)
    },
    authToken
  );
}

export function getOrganizationPrivacyRequests(
  userId: string,
  organizationId: string,
  authToken?: string | null
): Promise<OrganizationPrivacyRequestSummary[]> {
  return apiFetch<OrganizationPrivacyRequestSummary[]>(
    `/organizations/${organizationId}/privacy-requests`,
    userId,
    undefined,
    authToken
  );
}

export function createOrganizationPrivacyRequest(
  userId: string,
  organizationId: string,
  payload: OrganizationPrivacyRequestCreatePayload,
  authToken?: string | null
): Promise<OrganizationPrivacyRequestSummary> {
  return apiFetch<OrganizationPrivacyRequestSummary>(
    `/organizations/${organizationId}/privacy-requests`,
    userId,
    {
      method: "POST",
      body: JSON.stringify(payload)
    },
    authToken
  );
}

export function updateOrganizationPrivacyRequest(
  userId: string,
  organizationId: string,
  privacyRequestId: string,
  payload: OrganizationPrivacyRequestStatusUpdatePayload,
  authToken?: string | null
): Promise<OrganizationPrivacyRequestSummary> {
  return apiFetch<OrganizationPrivacyRequestSummary>(
    `/organizations/${organizationId}/privacy-requests/${privacyRequestId}`,
    userId,
    {
      method: "PATCH",
      body: JSON.stringify(payload)
    },
    authToken
  );
}

export function getOrganizationPilotReadiness(
  userId: string,
  organizationId: string,
  authToken?: string | null
): Promise<OrganizationPilotReadinessSummary> {
  return apiFetch<OrganizationPilotReadinessSummary>(
    `/organizations/${organizationId}/pilot-readiness`,
    userId,
    undefined,
    authToken
  );
}

export function updateOrganizationPilotReadinessItem(
  userId: string,
  organizationId: string,
  itemKey: string,
  payload: OrganizationPilotReadinessUpdatePayload,
  authToken?: string | null
): Promise<OrganizationPilotReadinessSummary> {
  return apiFetch<OrganizationPilotReadinessSummary>(
    `/organizations/${organizationId}/pilot-readiness/${itemKey}`,
    userId,
    {
      method: "PATCH",
      body: JSON.stringify(payload)
    },
    authToken
  );
}

export function getOrganizationAnalysisConfig(
  userId: string,
  organizationId: string,
  siteId?: string,
  authToken?: string | null
): Promise<AnalysisConfigSummary> {
  const query = siteId ? `?site_id=${encodeURIComponent(siteId)}` : "";
  return apiFetch<AnalysisConfigSummary>(
    `/organizations/${organizationId}/analysis-config${query}`,
    userId,
    undefined,
    authToken
  );
}

export function updateOrganizationAnalysisConfig(
  userId: string,
  organizationId: string,
  payload: AnalysisConfigUpdatePayload,
  authToken?: string | null
): Promise<AnalysisConfigSummary> {
  return apiFetch<AnalysisConfigSummary>(
    `/organizations/${organizationId}/analysis-config`,
    userId,
    {
      method: "PATCH",
      body: JSON.stringify(payload)
    },
    authToken
  );
}

export function getOrganizationDataConnectors(
  userId: string,
  organizationId: string,
  siteId?: string,
  authToken?: string | null
): Promise<DataConnectorSummary[]> {
  const query = siteId ? `?site_id=${encodeURIComponent(siteId)}` : "";
  return apiFetch<DataConnectorSummary[]>(
    `/organizations/${organizationId}/data-connectors${query}`,
    userId,
    undefined,
    authToken
  );
}

export function getOrganizationDueDataConnectors(
  userId: string,
  organizationId: string,
  dueAt?: string,
  authToken?: string | null
): Promise<DataConnectorSummary[]> {
  const query = dueAt ? `?due_at=${encodeURIComponent(dueAt)}` : "";
  return apiFetch<DataConnectorSummary[]>(
    `/organizations/${organizationId}/data-connectors/due${query}`,
    userId,
    undefined,
    authToken
  );
}

export function createOrganizationDataConnector(
  userId: string,
  organizationId: string,
  payload: DataConnectorCreatePayload,
  authToken?: string | null
): Promise<DataConnectorSummary> {
  return apiFetch<DataConnectorSummary>(
    `/organizations/${organizationId}/data-connectors`,
    userId,
    {
      method: "POST",
      body: JSON.stringify(payload)
    },
    authToken
  );
}

export function updateOrganizationDataConnector(
  userId: string,
  organizationId: string,
  connectorId: string,
  payload: DataConnectorUpdatePayload,
  authToken?: string | null
): Promise<DataConnectorSummary> {
  return apiFetch<DataConnectorSummary>(
    `/organizations/${organizationId}/data-connectors/${connectorId}`,
    userId,
    {
      method: "PATCH",
      body: JSON.stringify(payload)
    },
    authToken
  );
}

export function getOrganizationDataConnectorSyncs(
  userId: string,
  organizationId: string,
  connectorId: string,
  authToken?: string | null
): Promise<DataConnectorSyncSummary[]> {
  return apiFetch<DataConnectorSyncSummary[]>(
    `/organizations/${organizationId}/data-connectors/${connectorId}/syncs`,
    userId,
    undefined,
    authToken
  );
}

export function recordOrganizationDataConnectorSync(
  userId: string,
  organizationId: string,
  connectorId: string,
  payload: DataConnectorSyncRecordPayload,
  authToken?: string | null
): Promise<DataConnectorSyncSummary> {
  return apiFetch<DataConnectorSyncSummary>(
    `/organizations/${organizationId}/data-connectors/${connectorId}/syncs`,
    userId,
    {
      method: "POST",
      body: JSON.stringify(payload)
    },
    authToken
  );
}

export function importOrganizationDataConnectorContent(
  userId: string,
  organizationId: string,
  connectorId: string,
  payload: DataConnectorImportPayload,
  authToken?: string | null
): Promise<DataConnectorImportSummary> {
  return apiFetch<DataConnectorImportSummary>(
    `/organizations/${organizationId}/data-connectors/${connectorId}/imports`,
    userId,
    {
      method: "POST",
      body: JSON.stringify(payload)
    },
    authToken
  );
}

export function getBillingPlans(
  userId: string,
  authToken?: string | null
): Promise<SubscriptionPlanSummary[]> {
  return apiFetch<SubscriptionPlanSummary[]>("/billing/plans", userId, undefined, authToken);
}

export function getOrganizationSubscription(
  userId: string,
  organizationId: string,
  authToken?: string | null
): Promise<OrganizationSubscriptionSummary> {
  return apiFetch<OrganizationSubscriptionSummary>(
    `/organizations/${organizationId}/subscription`,
    userId,
    undefined,
    authToken
  );
}

export function updateOrganizationSubscription(
  userId: string,
  organizationId: string,
  payload: OrganizationSubscriptionUpdatePayload,
  authToken?: string | null
): Promise<OrganizationSubscriptionSummary> {
  return apiFetch<OrganizationSubscriptionSummary>(
    `/organizations/${organizationId}/subscription`,
    userId,
    {
      method: "PATCH",
      body: JSON.stringify(payload)
    },
    authToken
  );
}

export function createBillingCheckoutSession(
  userId: string,
  organizationId: string,
  payload: BillingCheckoutCreatePayload,
  authToken?: string | null
): Promise<BillingCheckoutSessionSummary> {
  return apiFetch<BillingCheckoutSessionSummary>(
    `/organizations/${organizationId}/billing/checkout`,
    userId,
    {
      method: "POST",
      body: JSON.stringify(payload)
    },
    authToken
  );
}

export function createBillingPortalSession(
  userId: string,
  organizationId: string,
  payload: BillingPortalCreatePayload,
  authToken?: string | null
): Promise<BillingPortalSessionSummary> {
  return apiFetch<BillingPortalSessionSummary>(
    `/organizations/${organizationId}/billing/portal`,
    userId,
    {
      method: "POST",
      body: JSON.stringify(payload)
    },
    authToken
  );
}

export async function createOwnerOrganization(
  payload: OwnerOrganizationCreatePayload
): Promise<OwnerOrganizationResponse> {
  const response = await fetch(`${API_BASE_URL}/organizations/onboard-owner`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    throw new ApiRequestError(response);
  }

  return response.json() as Promise<OwnerOrganizationResponse>;
}

export async function createDevSession(
  payload: DevSessionCreatePayload
): Promise<DevSessionResponse> {
  const response = await fetch(`${API_BASE_URL}/auth/dev/session`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    throw new ApiRequestError(response);
  }

  return response.json() as Promise<DevSessionResponse>;
}

export async function createMarketingInquiry(
  payload: MarketingInquiryPayload
): Promise<MarketingInquirySummary> {
  const response = await fetch(`${API_BASE_URL}/marketing/inquiries`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    throw new ApiRequestError(response);
  }

  return response.json() as Promise<MarketingInquirySummary>;
}

export function syncClerkSession(
  userId: string,
  authToken: string
): Promise<ClerkSyncResponse> {
  return apiFetch<ClerkSyncResponse>(
    "/auth/clerk/sync",
    userId,
    {
      method: "POST"
    },
    authToken
  );
}

export function getOrganizationSites(
  userId: string,
  organizationId: string,
  authToken?: string | null
): Promise<SiteSummary[]> {
  return apiFetch<SiteSummary[]>(`/organizations/${organizationId}/sites`, userId, undefined, authToken);
}

export function createOrganizationSite(
  userId: string,
  organizationId: string,
  payload: SiteCreatePayload,
  authToken?: string | null
): Promise<SiteSummary> {
  return apiFetch<SiteSummary>(
    `/organizations/${organizationId}/sites`,
    userId,
    {
      method: "POST",
      body: JSON.stringify(payload)
    },
    authToken
  );
}

export function getOrganizationMemberships(
  userId: string,
  organizationId: string,
  authToken?: string | null
): Promise<MembershipSummary[]> {
  return apiFetch<MembershipSummary[]>(
    `/organizations/${organizationId}/memberships`,
    userId,
    undefined,
    authToken
  );
}

export function getOrganizationReportApprovers(
  userId: string,
  organizationId: string,
  authToken?: string | null
): Promise<ReportApproverSummary[]> {
  return apiFetch<ReportApproverSummary[]>(
    `/organizations/${organizationId}/report-approvers`,
    userId,
    undefined,
    authToken
  );
}

export function updateOrganizationMembership(
  userId: string,
  organizationId: string,
  memberUserId: string,
  payload: MembershipRoleUpdatePayload,
  authToken?: string | null
): Promise<MembershipSummary> {
  return apiFetch<MembershipSummary>(
    `/organizations/${organizationId}/memberships/${memberUserId}`,
    userId,
    {
      method: "PATCH",
      body: JSON.stringify(payload)
    },
    authToken
  );
}

export function removeOrganizationMembership(
  userId: string,
  organizationId: string,
  memberUserId: string,
  authToken?: string | null
): Promise<MembershipSummary> {
  return apiFetch<MembershipSummary>(
    `/organizations/${organizationId}/memberships/${memberUserId}`,
    userId,
    {
      method: "DELETE"
    },
    authToken
  );
}

export function getOrganizationInvites(
  userId: string,
  organizationId: string,
  authToken?: string | null
): Promise<OrganizationInviteSummary[]> {
  return apiFetch<OrganizationInviteSummary[]>(
    `/organizations/${organizationId}/invites`,
    userId,
    undefined,
    authToken
  );
}

export function createOrganizationInvite(
  userId: string,
  organizationId: string,
  payload: OrganizationInviteCreatePayload,
  authToken?: string | null
): Promise<OrganizationInviteSummary> {
  return apiFetch<OrganizationInviteSummary>(
    `/organizations/${organizationId}/invites`,
    userId,
    {
      method: "POST",
      body: JSON.stringify(payload)
    },
    authToken
  );
}

export function revokeOrganizationInvite(
  userId: string,
  organizationId: string,
  inviteId: string,
  authToken?: string | null
): Promise<OrganizationInviteSummary> {
  return apiFetch<OrganizationInviteSummary>(
    `/organizations/${organizationId}/invites/${inviteId}/revoke`,
    userId,
    {
      method: "POST"
    },
    authToken
  );
}

export async function acceptOrganizationInvite(
  inviteId: string,
  payload: AcceptInvitePayload
): Promise<OrganizationInviteSummary> {
  const response = await fetch(`${API_BASE_URL}/invites/${inviteId}/accept`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    throw new ApiRequestError(response);
  }

  return response.json() as Promise<OrganizationInviteSummary>;
}

export function getOrganizationMeters(
  userId: string,
  organizationId: string,
  siteId?: string,
  authToken?: string | null
): Promise<MeterSummary[]> {
  const query = siteId ? `?site_id=${encodeURIComponent(siteId)}` : "";
  return apiFetch<MeterSummary[]>(
    `/organizations/${organizationId}/meters${query}`,
    userId,
    undefined,
    authToken
  );
}

export function createOrganizationMeter(
  userId: string,
  organizationId: string,
  payload: MeterCreatePayload,
  authToken?: string | null
): Promise<MeterSummary> {
  return apiFetch<MeterSummary>(
    `/organizations/${organizationId}/meters`,
    userId,
    {
      method: "POST",
      body: JSON.stringify(payload)
    },
    authToken
  );
}

export function executeLocalAnalysisRun(
  userId: string,
  organizationId: string,
  payload: LocalAnalysisRunPayload,
  authToken?: string | null
): Promise<LocalAnalysisRunResponse> {
  return apiFetch<LocalAnalysisRunResponse>(
    `/organizations/${organizationId}/runs/execute-local`,
    userId,
    {
      method: "POST",
      body: JSON.stringify(payload)
    },
    authToken
  );
}

export function executeStoredUploadAnalysisRun(
  userId: string,
  organizationId: string,
  payload: StoredUploadAnalysisRunPayload,
  authToken?: string | null
): Promise<LocalAnalysisRunResponse> {
  return apiFetch<LocalAnalysisRunResponse>(
    `/organizations/${organizationId}/runs/execute-upload`,
    userId,
    {
      method: "POST",
      body: JSON.stringify(payload)
    },
    authToken
  );
}

export function queueStoredUploadAnalysisRun(
  userId: string,
  organizationId: string,
  payload: StoredUploadAnalysisRunPayload,
  authToken?: string | null
): Promise<QueuedAnalysisRunResponse> {
  return apiFetch<QueuedAnalysisRunResponse>(
    `/organizations/${organizationId}/runs/queue-upload`,
    userId,
    {
      method: "POST",
      body: JSON.stringify(payload)
    },
    authToken
  );
}

export function getOrganizationReports(
  userId: string,
  organizationId: string,
  runId?: string,
  authToken?: string | null
): Promise<ReportSummary[]> {
  const query = runId ? `?run_id=${encodeURIComponent(runId)}` : "";
  return apiFetch<ReportSummary[]>(
    `/organizations/${organizationId}/reports${query}`,
    userId,
    undefined,
    authToken
  );
}

export function getOrganizationReportArtifact(
  userId: string,
  organizationId: string,
  reportId: string,
  authToken?: string | null
): Promise<ReportArtifactResponse> {
  return apiFetch<ReportArtifactResponse>(
    `/organizations/${organizationId}/reports/${reportId}/artifact`,
    userId,
    undefined,
    authToken
  );
}

export function applyOrganizationReportRetention(
  userId: string,
  organizationId: string,
  authToken?: string | null
): Promise<ReportRetentionApplyResponse> {
  return apiFetch<ReportRetentionApplyResponse>(
    `/organizations/${organizationId}/reports/retention/apply`,
    userId,
    {
      method: "POST"
    },
    authToken
  );
}

export function getOrganizationReportExport(
  userId: string,
  organizationId: string,
  reportId: string,
  authToken?: string | null
): Promise<ReportExportFile> {
  return apiFetchText(
    `/organizations/${organizationId}/reports/${reportId}/export`,
    userId,
    undefined,
    authToken
  );
}

export function getOrganizationReportEvidencePackage(
  userId: string,
  organizationId: string,
  reportId: string,
  authToken?: string | null
): Promise<ReportExportFile> {
  return apiFetchText(
    `/organizations/${organizationId}/reports/${reportId}/evidence-package`,
    userId,
    undefined,
    authToken
  );
}

export function getOrganizationReportEvidencePackagePdf(
  userId: string,
  organizationId: string,
  reportId: string,
  authToken?: string | null
): Promise<ReportBinaryExportFile> {
  return apiFetchBinary(
    `/organizations/${organizationId}/reports/${reportId}/evidence-package/pdf`,
    userId,
    undefined,
    authToken
  );
}

export function getOrganizationReportEvidencePackages(
  userId: string,
  organizationId: string,
  reportId: string,
  authToken?: string | null
): Promise<ReportEvidencePackageSummary[]> {
  return apiFetch<ReportEvidencePackageSummary[]>(
    `/organizations/${organizationId}/reports/${reportId}/evidence-packages`,
    userId,
    undefined,
    authToken
  );
}

export function createOrganizationReportEvidencePackage(
  userId: string,
  organizationId: string,
  reportId: string,
  authToken?: string | null
): Promise<ReportEvidencePackageSummary> {
  return apiFetch<ReportEvidencePackageSummary>(
    `/organizations/${organizationId}/reports/${reportId}/evidence-packages`,
    userId,
    {
      method: "POST"
    },
    authToken
  );
}

export function getOrganizationReportEvidencePackageSnapshot(
  userId: string,
  organizationId: string,
  reportId: string,
  packageId: string,
  authToken?: string | null
): Promise<ReportExportFile> {
  return apiFetchText(
    `/organizations/${organizationId}/reports/${reportId}/evidence-packages/${packageId}/download`,
    userId,
    undefined,
    authToken
  );
}

export function getOrganizationReportComments(
  userId: string,
  organizationId: string,
  reportId: string,
  authToken?: string | null
): Promise<ReportCommentSummary[]> {
  return apiFetch<ReportCommentSummary[]>(
    `/organizations/${organizationId}/reports/${reportId}/comments`,
    userId,
    undefined,
    authToken
  );
}

export function getOrganizationReportReviews(
  userId: string,
  organizationId: string,
  reportId: string,
  authToken?: string | null
): Promise<ReportReviewSummary[]> {
  return apiFetch<ReportReviewSummary[]>(
    `/organizations/${organizationId}/reports/${reportId}/reviews`,
    userId,
    undefined,
    authToken
  );
}

export function createOrganizationReportReview(
  userId: string,
  organizationId: string,
  reportId: string,
  payload: ReportReviewCreatePayload,
  authToken?: string | null
): Promise<ReportReviewSummary> {
  return apiFetch<ReportReviewSummary>(
    `/organizations/${organizationId}/reports/${reportId}/reviews`,
    userId,
    {
      method: "POST",
      body: JSON.stringify(payload)
    },
    authToken
  );
}

export function getOrganizationReportApprovalAssignments(
  userId: string,
  organizationId: string,
  reportId: string,
  authToken?: string | null
): Promise<ReportApprovalAssignmentSummary[]> {
  return apiFetch<ReportApprovalAssignmentSummary[]>(
    `/organizations/${organizationId}/reports/${reportId}/approval-assignments`,
    userId,
    undefined,
    authToken
  );
}

export function createOrganizationReportApprovalAssignment(
  userId: string,
  organizationId: string,
  reportId: string,
  payload: ReportApprovalAssignmentCreatePayload,
  authToken?: string | null
): Promise<ReportApprovalAssignmentSummary> {
  return apiFetch<ReportApprovalAssignmentSummary>(
    `/organizations/${organizationId}/reports/${reportId}/approval-assignments`,
    userId,
    {
      method: "POST",
      body: JSON.stringify(payload)
    },
    authToken
  );
}

export function updateOrganizationReportApprovalAssignment(
  userId: string,
  organizationId: string,
  reportId: string,
  assignmentId: string,
  payload: ReportApprovalAssignmentUpdatePayload,
  authToken?: string | null
): Promise<ReportApprovalAssignmentSummary> {
  return apiFetch<ReportApprovalAssignmentSummary>(
    `/organizations/${organizationId}/reports/${reportId}/approval-assignments/${assignmentId}`,
    userId,
    {
      method: "PATCH",
      body: JSON.stringify(payload)
    },
    authToken
  );
}

export function getOrganizationReportSchedules(
  userId: string,
  organizationId: string,
  reportId: string,
  authToken?: string | null
): Promise<ReportScheduleSummary[]> {
  return apiFetch<ReportScheduleSummary[]>(
    `/organizations/${organizationId}/reports/${reportId}/schedules`,
    userId,
    undefined,
    authToken
  );
}

export function createOrganizationReportSchedule(
  userId: string,
  organizationId: string,
  reportId: string,
  payload: ReportScheduleCreatePayload,
  authToken?: string | null
): Promise<ReportScheduleSummary> {
  return apiFetch<ReportScheduleSummary>(
    `/organizations/${organizationId}/reports/${reportId}/schedules`,
    userId,
    {
      method: "POST",
      body: JSON.stringify(payload)
    },
    authToken
  );
}

export function updateOrganizationReportSchedule(
  userId: string,
  organizationId: string,
  reportId: string,
  scheduleId: string,
  payload: ReportScheduleUpdatePayload,
  authToken?: string | null
): Promise<ReportScheduleSummary> {
  return apiFetch<ReportScheduleSummary>(
    `/organizations/${organizationId}/reports/${reportId}/schedules/${scheduleId}`,
    userId,
    {
      method: "PATCH",
      body: JSON.stringify(payload)
    },
    authToken
  );
}

export function getOrganizationReportScheduleDeliveries(
  userId: string,
  organizationId: string,
  reportId: string,
  scheduleId: string,
  authToken?: string | null
): Promise<ReportScheduleDeliverySummary[]> {
  return apiFetch<ReportScheduleDeliverySummary[]>(
    `/organizations/${organizationId}/reports/${reportId}/schedules/${scheduleId}/deliveries`,
    userId,
    undefined,
    authToken
  );
}

export function createOrganizationReportComment(
  userId: string,
  organizationId: string,
  reportId: string,
  payload: ReportCommentCreatePayload,
  authToken?: string | null
): Promise<ReportCommentSummary> {
  return apiFetch<ReportCommentSummary>(
    `/organizations/${organizationId}/reports/${reportId}/comments`,
    userId,
    {
      method: "POST",
      body: JSON.stringify(payload)
    },
    authToken
  );
}

export function getOrganizationUploads(
  userId: string,
  organizationId: string,
  siteId?: string,
  authToken?: string | null
): Promise<UploadSummary[]> {
  const query = siteId ? `?site_id=${encodeURIComponent(siteId)}` : "";
  return apiFetch<UploadSummary[]>(
    `/organizations/${organizationId}/uploads${query}`,
    userId,
    undefined,
    authToken
  );
}

export function getOrganizationUploadReadiness(
  userId: string,
  organizationId: string,
  siteId?: string,
  authToken?: string | null
): Promise<UploadReadinessSummary> {
  const query = siteId ? `?site_id=${encodeURIComponent(siteId)}` : "";
  return apiFetch<UploadReadinessSummary>(
    `/organizations/${organizationId}/uploads/readiness${query}`,
    userId,
    undefined,
    authToken
  );
}

export function createOrganizationUpload(
  userId: string,
  organizationId: string,
  payload: UploadCreatePayload,
  authToken?: string | null
): Promise<UploadSummary> {
  return apiFetch<UploadSummary>(
    `/organizations/${organizationId}/uploads`,
    userId,
    {
      method: "POST",
      body: JSON.stringify(payload)
    },
    authToken
  );
}

export function createOrganizationUploadContent(
  userId: string,
  organizationId: string,
  payload: UploadContentCreatePayload,
  authToken?: string | null
): Promise<UploadSummary> {
  return apiFetch<UploadSummary>(
    `/organizations/${organizationId}/uploads/content`,
    userId,
    {
      method: "POST",
      body: JSON.stringify(payload)
    },
    authToken
  );
}

export function getOrganizationRuns(
  userId: string,
  organizationId: string,
  siteId?: string,
  authToken?: string | null
): Promise<RunSummary[]> {
  const query = siteId ? `?site_id=${encodeURIComponent(siteId)}` : "";
  return apiFetch<RunSummary[]>(`/organizations/${organizationId}/runs${query}`, userId, undefined, authToken);
}

export function getOrganizationAnalysisJobs(
  userId: string,
  organizationId: string,
  authToken?: string | null
): Promise<AnalysisJobSummary[]> {
  return apiFetch<AnalysisJobSummary[]>(
    `/organizations/${organizationId}/analysis-jobs`,
    userId,
    undefined,
    authToken
  );
}

export function getOrganizationNotifications(
  userId: string,
  organizationId: string,
  authToken?: string | null
): Promise<NotificationSummary[]> {
  return apiFetch<NotificationSummary[]>(
    `/organizations/${organizationId}/notifications`,
    userId,
    undefined,
    authToken
  );
}

export function createOrganizationRun(
  userId: string,
  organizationId: string,
  payload: RunCreatePayload,
  authToken?: string | null
): Promise<RunSummary> {
  return apiFetch<RunSummary>(
    `/organizations/${organizationId}/runs`,
    userId,
    {
      method: "POST",
      body: JSON.stringify(payload)
    },
    authToken
  );
}

export function getOrganizationAuditEvents(
  userId: string,
  organizationId: string,
  authToken?: string | null
): Promise<AuditEventSummary[]> {
  return apiFetch<AuditEventSummary[]>(
    `/organizations/${organizationId}/audit-events`,
    userId,
    undefined,
    authToken
  );
}

export function getOrganizationAuditExport(
  userId: string,
  organizationId: string,
  authToken?: string | null
): Promise<AuditExportResponse> {
  return apiFetch<AuditExportResponse>(
    `/organizations/${organizationId}/audit-events/export`,
    userId,
    undefined,
    authToken
  );
}
