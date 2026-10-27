import { revalidatePath } from "next/cache";
import Link from "next/link";
import { redirect } from "next/navigation";
import { WorkspaceShell } from "../../components/WorkspaceShell";
import {
  applyOrganizationReportRetention,
  createBillingCheckoutSession,
  createBillingPortalSession,
  createOrganizationDataConnector,
  updateOrganizationEvidencePackageTemplate,
  createOrganizationMeter,
  createOrganizationInvite,
  createOrganizationPrivacyRequest,
  createOrganizationSite,
  removeOrganizationMembership,
  revokeOrganizationInvite,
  updateOrganizationMembership,
  updateOrganizationAnalysisConfig,
  updateOrganizationPilotReadinessItem,
  updateOrganizationPrivacyRequest,
  updateOrganizationPrivacySettings,
  updateOrganizationReportApprovalPolicy,
  updateOrganizationReportBranding,
  updateOrganizationDataConnector,
  updateOrganizationProfile
} from "../../lib/api";
import type {
  DataConnectorSchedule,
  DataConnectorStatus,
  DataConnectorType,
  OrganizationEvidencePackageTemplateSummary,
  OrganizationPilotReadinessStatus,
  OrganizationPrivacyRequestStatus,
  OrganizationPrivacyRequestType
} from "../../lib/api";
import { readAppSession } from "../../lib/session";
import { loadSettingsData } from "../../lib/settings-data";

type SettingsPageProps = {
  searchParams: Promise<{
    status?: string;
    error?: string;
    expired?: string;
    deleted?: string;
    missing?: string;
    packageDeleted?: string;
    packageMissing?: string;
    packageMetadata?: string;
  }>;
};

const privacyRequestTypes: OrganizationPrivacyRequestType[] = [
  "data_export",
  "deletion",
  "correction"
];

const privacyRequestStatuses: OrganizationPrivacyRequestStatus[] = [
  "open",
  "in_review",
  "completed",
  "rejected"
];

const pilotReadinessStatuses: OrganizationPilotReadinessStatus[] = [
  "pending",
  "in_progress",
  "verified",
  "blocked"
];

const dataConnectorTypes: { value: DataConnectorType; label: string }[] = [
  { value: "sftp", label: "SFTP" },
  { value: "api", label: "API" },
  { value: "utility_portal", label: "Utility portal" }
];

const dataConnectorStatuses: { value: DataConnectorStatus; label: string }[] = [
  { value: "draft", label: "Draft" },
  { value: "active", label: "Active" },
  { value: "paused", label: "Paused" }
];

const dataConnectorSchedules: { value: DataConnectorSchedule; label: string }[] = [
  { value: "manual", label: "Manual" },
  { value: "daily", label: "Daily" },
  { value: "weekly", label: "Weekly" },
  { value: "monthly", label: "Monthly" }
];

const sourceCategoryOptions = [
  { value: "energy", label: "Energy" },
  { value: "hdd", label: "HDD" },
  { value: "cdd", label: "CDD" },
  { value: "seu_mapping", label: "SEU mapping" }
];

const evidenceTemplateSections = [
  {
    key: "report_artifact",
    includeField: "include_report_artifact",
    name: "Report artifact",
    defaultLabel: "Report Artifact Summary"
  },
  {
    key: "uploads",
    includeField: "include_uploads",
    name: "Upload lineage",
    defaultLabel: "Upload Lineage"
  },
  {
    key: "approval_assignments",
    includeField: "include_approval_assignments",
    name: "Approval assignments",
    defaultLabel: "Approval Assignments"
  },
  {
    key: "reviews",
    includeField: "include_reviews",
    name: "Review decisions",
    defaultLabel: "Management Review Decisions"
  },
  {
    key: "comments",
    includeField: "include_comments",
    name: "Review comments",
    defaultLabel: "Collaboration Evidence"
  },
  {
    key: "schedules",
    includeField: "include_schedules",
    name: "Delivery schedules",
    defaultLabel: "Scheduled Delivery Evidence"
  }
] as const;

type EvidenceTemplateSection = (typeof evidenceTemplateSections)[number];

function evidenceTemplateOrderValue(
  template: OrganizationEvidencePackageTemplateSummary,
  section: EvidenceTemplateSection
): number {
  const configuredIndex = template.section_order.indexOf(section.key);
  if (configuredIndex >= 0) {
    return configuredIndex + 1;
  }
  return evidenceTemplateSections.findIndex((candidate) => candidate.key === section.key) + 1;
}

function evidenceTemplateSectionLabel(
  template: OrganizationEvidencePackageTemplateSummary,
  section: EvidenceTemplateSection
): string {
  return template.section_labels[section.key] ?? section.defaultLabel;
}

function orderedEvidenceTemplateSections(
  template: OrganizationEvidencePackageTemplateSummary
): EvidenceTemplateSection[] {
  return [...evidenceTemplateSections].sort(
    (left, right) =>
      evidenceTemplateOrderValue(template, left) - evidenceTemplateOrderValue(template, right)
  );
}

function enabledEvidenceTemplateSectionCount(
  template: OrganizationEvidencePackageTemplateSummary
): number {
  return evidenceTemplateSections.filter((section) => template[section.includeField]).length;
}

function analysisSettings(config: Record<string, unknown>): Record<string, unknown> {
  const rawAnalysis = config.analysis;
  if (rawAnalysis && typeof rawAnalysis === "object" && !Array.isArray(rawAnalysis)) {
    return rawAnalysis as Record<string, unknown>;
  }
  return {};
}

function analysisBooleanSetting(
  config: Record<string, unknown>,
  key: string,
  fallback: boolean
): boolean {
  const value = analysisSettings(config)[key];
  return typeof value === "boolean" ? value : fallback;
}

function analysisNumberSetting(
  config: Record<string, unknown>,
  key: string,
  fallback: number
): number {
  const value = analysisSettings(config)[key];
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function settingsMessage(
  status: string | undefined,
  error: string | undefined
): { tone: "success" | "error"; title: string; body: string } | null {
  if (status === "invite-created") {
    return {
      tone: "success",
      title: "Invite created",
      body: "The new invite is available in the tenant invite list."
    };
  }
  if (status === "invite-revoked") {
    return {
      tone: "success",
      title: "Invite revoked",
      body: "The invite is no longer available for acceptance."
    };
  }
  if (status === "member-updated") {
    return {
      tone: "success",
      title: "Member updated",
      body: "The member role was updated for this tenant workspace."
    };
  }
  if (status === "member-removed") {
    return {
      tone: "success",
      title: "Member removed",
      body: "The member no longer has access to this tenant workspace."
    };
  }
  if (status === "site-created") {
    return {
      tone: "success",
      title: "Site added",
      body: "The new site is now part of your organization."
    };
  }
  if (status === "meter-created") {
    return {
      tone: "success",
      title: "Meter added",
      body: "The new meter is now available for uploads and analysis."
    };
  }
  if (status === "profile-updated") {
    return {
      tone: "success",
      title: "Profile updated",
      body: "The tenant profile now reflects the latest organization details."
    };
  }
  if (status === "privacy-settings-updated") {
    return {
      tone: "success",
      title: "Privacy settings updated",
      body: "Tenant privacy preferences and data-processing acknowledgement are saved."
    };
  }
  if (status === "report-branding-updated") {
    return {
      tone: "success",
      title: "Report branding updated",
      body: "Future evidence packages will use the tenant report branding settings."
    };
  }
  if (status === "report-approval-policy-updated") {
    return {
      tone: "success",
      title: "Approval policy updated",
      body: "Report evidence locking will follow the tenant approval policy."
    };
  }
  if (status === "evidence-template-updated") {
    return {
      tone: "success",
      title: "Evidence template updated",
      body: "Future evidence packages will use the selected tenant evidence sections."
    };
  }
  if (status === "data-connector-created") {
    return {
      tone: "success",
      title: "Connector created",
      body: "The tenant data connector is ready for future ingestion automation."
    };
  }
  if (status === "data-connector-updated") {
    return {
      tone: "success",
      title: "Connector updated",
      body: "The tenant connector status and sync cadence were saved."
    };
  }
  if (status === "privacy-request-created") {
    return {
      tone: "success",
      title: "Privacy request created",
      body: "The request is available in the tenant privacy operations list."
    };
  }
  if (status === "privacy-request-updated") {
    return {
      tone: "success",
      title: "Privacy request updated",
      body: "The request status was saved and audited for this tenant."
    };
  }
  if (status === "pilot-readiness-updated") {
    return {
      tone: "success",
      title: "Pilot readiness updated",
      body: "The tenant launch checklist item was saved and audited."
    };
  }
  if (status === "analysis-config-updated") {
    return {
      tone: "success",
      title: "Analysis settings updated",
      body: "Future tenant runs will use the saved analysis settings."
    };
  }
  if (status === "billing-checkout-created") {
    return {
      tone: "success",
      title: "Checkout session created",
      body: "The payment-provider checkout session is ready for this tenant."
    };
  }
  if (status === "billing-portal-created") {
    return {
      tone: "success",
      title: "Billing portal opened",
      body: "The payment-provider portal session is ready for this tenant."
    };
  }
  if (status === "retention-applied") {
    return {
      tone: "success",
      title: "Retention applied",
      body: "Expired report artifacts and locked evidence packages were processed."
    };
  }
  if (error === "missing-fields") {
    return {
      tone: "error",
      title: "Action not completed",
      body: "Fill in all required fields before submitting."
    };
  }
  if (error === "session-missing") {
    return {
      tone: "error",
      title: "Action not completed",
      body: "Sign in to a tenant workspace before managing administration."
    };
  }
  if (error === "invite-failed") {
    return {
      tone: "error",
      title: "Invite not created",
      body: "We could not create that invite. Check for duplicate email addresses or permissions."
    };
  }
  if (error === "member-failed") {
    return {
      tone: "error",
      title: "Member not updated",
      body: "We could not change that membership. Keep at least one owner in the tenant."
    };
  }
  if (error === "site-failed") {
    return {
      tone: "error",
      title: "Site not added",
      body: "We could not resolve that location. Check the address or postcode and try again."
    };
  }
  if (error === "meter-failed") {
    return {
      tone: "error",
      title: "Meter not added",
      body: "We could not add that meter. Check details and try again."
    };
  }
  if (error === "profile-failed") {
    return {
      tone: "error",
      title: "Profile not updated",
      body: "We could not update the tenant profile. Check the slug and your permissions."
    };
  }
  if (error === "privacy-settings-failed") {
    return {
      tone: "error",
      title: "Privacy settings not updated",
      body: "We could not save the tenant privacy controls. Check details and permissions."
    };
  }
  if (error === "report-approval-policy-failed") {
    return {
      tone: "error",
      title: "Approval policy not updated",
      body: "We could not save the tenant report approval policy."
    };
  }
  if (error === "evidence-template-failed") {
    return {
      tone: "error",
      title: "Evidence template not updated",
      body: "We could not save the tenant evidence package template."
    };
  }
  if (error === "report-branding-failed") {
    return {
      tone: "error",
      title: "Report branding not updated",
      body: "We could not save evidence package branding. Check the color, URLs, and permissions."
    };
  }
  if (error === "privacy-request-failed") {
    return {
      tone: "error",
      title: "Privacy request not updated",
      body: "We could not save the tenant privacy request. Check details and permissions."
    };
  }
  if (error === "pilot-readiness-failed") {
    return {
      tone: "error",
      title: "Pilot readiness not updated",
      body: "We could not save that launch checklist item. Check details and permissions."
    };
  }
  if (error === "analysis-config-failed") {
    return {
      tone: "error",
      title: "Analysis settings not updated",
      body: "We could not save that analysis configuration. Check the JSON and required meter fields."
    };
  }
  if (error === "data-connector-failed") {
    return {
      tone: "error",
      title: "Connector not saved",
      body: "We could not save that data connector. Check site, schedule, and permissions."
    };
  }
  if (error === "billing-failed") {
    return {
      tone: "error",
      title: "Billing action not started",
      body: "We could not create a billing session. Check the selected plan and your permissions."
    };
  }
  if (error === "retention-failed") {
    return {
      tone: "error",
      title: "Retention not applied",
      body: "We could not apply report retention. Check report-management permissions."
    };
  }
  return null;
}

function formatLimit(value: number | null): string {
  return value === null ? "Custom" : value.toLocaleString("en-GB");
}

function formatMonthlyPrice(cents: number | null): string {
  if (cents === null) {
    return "Custom";
  }
  if (cents === 0) {
    return "Included";
  }
  return new Intl.NumberFormat("en-GB", {
    style: "currency",
    currency: "GBP",
    maximumFractionDigits: 0
  }).format(cents / 100);
}

function formatDateTime(value: string | null): string {
  if (!value) {
    return "Not recorded";
  }
  return new Intl.DateTimeFormat("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit"
  }).format(new Date(value));
}

function formatDateTimeInput(value: string | null): string {
  if (!value) {
    return "";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "";
  }
  return date.toISOString().slice(0, 16);
}

function formatQueryCount(value: string | undefined): string {
  const parsed = Number(value ?? "0");
  return Number.isFinite(parsed) ? parsed.toLocaleString("en-GB") : "0";
}

function formatPrivacyRequestType(value: OrganizationPrivacyRequestType): string {
  const labels: Record<OrganizationPrivacyRequestType, string> = {
    data_export: "Data export",
    deletion: "Deletion",
    correction: "Correction"
  };
  return labels[value];
}

function formatPrivacyRequestStatus(value: OrganizationPrivacyRequestStatus): string {
  const labels: Record<OrganizationPrivacyRequestStatus, string> = {
    open: "Open",
    in_review: "In review",
    completed: "Completed",
    rejected: "Rejected"
  };
  return labels[value];
}

function formatPilotReadinessStatus(value: OrganizationPilotReadinessStatus): string {
  const labels: Record<OrganizationPilotReadinessStatus, string> = {
    pending: "Pending",
    in_progress: "In progress",
    verified: "Verified",
    blocked: "Blocked"
  };
  return labels[value];
}

export default async function SettingsPage({ searchParams }: SettingsPageProps) {
  const query = await searchParams;
  const { mode, organization, evidencePackageTemplate, reportBranding, reportApprovalPolicy, pilotReadiness, privacySettings, privacyRequests, analysisConfig, dataConnectors, subscription, billingPlans, sites, meters, memberships, invites, auditEvents, permissions, loadError } =
    await loadSettingsData();
  const activeMembers = memberships.filter((membership) => membership.is_active);
  const seuMeters = meters.filter((meter) => meter.is_seu);
  const pageMessage = settingsMessage(query.status, query.error);
  const canCreateMeter = permissions.canManageAssets && sites.length > 0;
  const checkoutPlans = billingPlans.filter(
    (plan) => plan.monthly_price_cents !== null && plan.monthly_price_cents > 0
  );
  const activePrivacyRequestCount = privacyRequests.filter(
    (request) => request.status === "open" || request.status === "in_review"
  ).length;
  const pilotReadinessLabel = pilotReadiness
    ? `${pilotReadiness.verified_count}/${pilotReadiness.total_count}`
    : "0/0";

  async function createInviteAction(formData: FormData) {
    "use server";

    const email = String(formData.get("email") ?? "").trim();
    const roleValue = String(formData.get("role") ?? "viewer").trim();
    const session = await readAppSession();

    if (!email || !roleValue) {
      redirect("/settings?error=missing-fields");
    }
    if (!session.userId || !session.organizationId) {
      redirect("/settings?error=session-missing");
    }

    try {
      await createOrganizationInvite(
        session.userId,
        session.organizationId,
        {
          email,
          role: roleValue as "owner" | "manager" | "viewer"
        },
        session.authToken
      );
      revalidatePath("/settings");
    } catch {
      redirect("/settings?error=invite-failed");
    }
    redirect("/settings?status=invite-created");
  }

  async function revokeInviteAction(formData: FormData) {
    "use server";

    const inviteId = String(formData.get("invite_id") ?? "").trim();
    const session = await readAppSession();

    if (!inviteId) {
      redirect("/settings?error=invite-failed");
    }
    if (!session.userId || !session.organizationId) {
      redirect("/settings?error=session-missing");
    }

    try {
      await revokeOrganizationInvite(
        session.userId,
        session.organizationId,
        inviteId,
        session.authToken
      );
      revalidatePath("/settings");
    } catch {
      redirect("/settings?error=invite-failed");
    }
    redirect("/settings?status=invite-revoked");
  }

  async function updateMemberRoleAction(formData: FormData) {
    "use server";

    const memberUserId = String(formData.get("member_user_id") ?? "").trim();
    const roleValue = String(formData.get("role") ?? "").trim();
    const session = await readAppSession();

    if (!memberUserId || !roleValue) {
      redirect("/settings?error=missing-fields");
    }
    if (!session.userId || !session.organizationId) {
      redirect("/settings?error=session-missing");
    }

    try {
      await updateOrganizationMembership(
        session.userId,
        session.organizationId,
        memberUserId,
        { role: roleValue as "owner" | "manager" | "viewer" },
        session.authToken
      );
      revalidatePath("/settings");
    } catch {
      redirect("/settings?error=member-failed");
    }
    redirect("/settings?status=member-updated");
  }

  async function removeMemberAction(formData: FormData) {
    "use server";

    const memberUserId = String(formData.get("member_user_id") ?? "").trim();
    const session = await readAppSession();

    if (!memberUserId) {
      redirect("/settings?error=member-failed");
    }
    if (!session.userId || !session.organizationId) {
      redirect("/settings?error=session-missing");
    }

    try {
      await removeOrganizationMembership(
        session.userId,
        session.organizationId,
        memberUserId,
        session.authToken
      );
      revalidatePath("/settings");
    } catch {
      redirect("/settings?error=member-failed");
    }
    redirect("/settings?status=member-removed");
  }

  async function createSiteAction(formData: FormData) {
    "use server";

    const name = String(formData.get("name") ?? "").trim();
    const timezone = String(formData.get("timezone") ?? "").trim();
    const location = String(formData.get("location") ?? "").trim();
    const session = await readAppSession();

    if (!name || !timezone || !location) {
      redirect("/settings?error=missing-fields");
    }
    if (!session.userId || !session.organizationId) {
      redirect("/settings?error=session-missing");
    }

    const siteId = `site_${Date.now()}`;
    try {
      await createOrganizationSite(
        session.userId,
        session.organizationId,
        {
          site_id: siteId,
          name,
          timezone,
          location
        },
        session.authToken
      );
      revalidatePath("/settings");
      revalidatePath("/dashboard");
    } catch {
      redirect("/settings?error=site-failed");
    }
    redirect("/settings?status=site-created");
  }

  async function createMeterAction(formData: FormData) {
    "use server";

    const siteId = String(formData.get("site_id") ?? "").trim();
    const displayName = String(formData.get("display_name") ?? "").trim();
    const commodity = String(formData.get("commodity") ?? "").trim();
    const unit = String(formData.get("unit") ?? "").trim();
    const sourceColumn = String(formData.get("source_column") ?? "").trim();
    const isSeu = String(formData.get("is_seu") ?? "").trim() === "on";
    const session = await readAppSession();

    if (!siteId || !displayName || !commodity || !unit || !sourceColumn) {
      redirect("/settings?error=missing-fields");
    }
    if (!session.userId || !session.organizationId) {
      redirect("/settings?error=session-missing");
    }

    const meterId = `meter_${Date.now()}`;
    try {
      await createOrganizationMeter(
        session.userId,
        session.organizationId,
        {
          meter_id: meterId,
          site_id: siteId,
          display_name: displayName,
          commodity,
          unit,
          source_column: sourceColumn,
          is_seu: isSeu
        },
        session.authToken
      );
      revalidatePath("/settings");
      revalidatePath("/dashboard");
    } catch {
      redirect("/settings?error=meter-failed");
    }
    redirect("/settings?status=meter-created");
  }

  async function updateProfileAction(formData: FormData) {
    "use server";

    const name = String(formData.get("name") ?? "").trim();
    const slug = String(formData.get("slug") ?? "").trim();
    const session = await readAppSession();

    if (!name || !slug) {
      redirect("/settings?error=missing-fields");
    }
    if (!session.userId || !session.organizationId) {
      redirect("/settings?error=session-missing");
    }

    try {
      await updateOrganizationProfile(
        session.userId,
        session.organizationId,
        {
          name,
          slug
        },
        session.authToken
      );
    } catch {
      redirect("/settings?error=profile-failed");
    }

    revalidatePath("/settings");
    revalidatePath("/dashboard");
    revalidatePath("/organizations");
    redirect("/settings?status=profile-updated");
  }

  async function updatePrivacySettingsAction(formData: FormData) {
    "use server";

    const contactEmail = String(formData.get("data_processing_contact_email") ?? "").trim();
    const subprocessorNotifications =
      String(formData.get("subprocessor_notifications_enabled") ?? "").trim() === "on";
    const analyticsDataSharing =
      String(formData.get("analytics_data_sharing_enabled") ?? "").trim() === "on";
    const acknowledgeDataProcessing =
      String(formData.get("acknowledge_data_processing") ?? "").trim() === "on";
    const session = await readAppSession();

    if (!session.userId || !session.organizationId) {
      redirect("/settings?error=session-missing");
    }

    try {
      await updateOrganizationPrivacySettings(
        session.userId,
        session.organizationId,
        {
          data_processing_contact_email: contactEmail || null,
          subprocessor_notifications_enabled: subprocessorNotifications,
          analytics_data_sharing_enabled: analyticsDataSharing,
          acknowledge_data_processing: acknowledgeDataProcessing
        },
        session.authToken
      );
    } catch {
      redirect("/settings?error=privacy-settings-failed");
    }

    revalidatePath("/settings");
    redirect("/settings?status=privacy-settings-updated");
  }

  async function updateReportBrandingAction(formData: FormData) {
    "use server";

    const evidencePackTitle = String(formData.get("evidence_pack_title") ?? "").trim();
    const primaryColor = String(formData.get("primary_color") ?? "").trim();
    const logoUrl = String(formData.get("logo_url") ?? "").trim();
    const supportEmail = String(formData.get("support_email") ?? "").trim();
    const session = await readAppSession();

    if (!evidencePackTitle || !primaryColor) {
      redirect("/settings?error=missing-fields");
    }
    if (!session.userId || !session.organizationId) {
      redirect("/settings?error=session-missing");
    }

    try {
      await updateOrganizationReportBranding(
        session.userId,
        session.organizationId,
        {
          evidence_pack_title: evidencePackTitle,
          primary_color: primaryColor,
          logo_url: logoUrl || null,
          support_email: supportEmail || null
        },
        session.authToken
      );
    } catch {
      redirect("/settings?error=report-branding-failed");
    }

    revalidatePath("/settings");
    revalidatePath("/reports");
    redirect("/settings?status=report-branding-updated");
  }

  async function updateReportApprovalPolicyAction(formData: FormData) {
    "use server";

    const autoLockEvidence = formData.get("auto_lock_evidence_on_final_approval") === "on";
    const minimumApprovalsRaw = String(
      formData.get("minimum_approval_assignments_for_lock") ?? "0"
    ).trim();
    const minimumApprovals = Number.parseInt(minimumApprovalsRaw, 10);
    const session = await readAppSession();

    if (!Number.isInteger(minimumApprovals) || minimumApprovals < 0) {
      redirect("/settings?error=report-approval-policy-failed");
    }
    if (!session.userId || !session.organizationId) {
      redirect("/settings?error=session-missing");
    }

    try {
      await updateOrganizationReportApprovalPolicy(
        session.userId,
        session.organizationId,
        {
          auto_lock_evidence_on_final_approval: autoLockEvidence,
          minimum_approval_assignments_for_lock: minimumApprovals
        },
        session.authToken
      );
    } catch {
      redirect("/settings?error=report-approval-policy-failed");
    }

    revalidatePath("/settings");
    revalidatePath("/reports");
    redirect("/settings?status=report-approval-policy-updated");
  }

  async function updateEvidencePackageTemplateAction(formData: FormData) {
    "use server";

    const templateName = String(formData.get("template_name") ?? "").trim();
    const session = await readAppSession();

    if (!templateName) {
      redirect("/settings?error=missing-fields");
    }
    if (!session.userId || !session.organizationId) {
      redirect("/settings?error=session-missing");
    }

    const sectionOrder = evidenceTemplateSections
      .map((section, defaultIndex) => {
        const rawOrder = Number(formData.get(`section_order_${section.key}`));
        const order = Number.isFinite(rawOrder) && rawOrder > 0 ? rawOrder : defaultIndex + 1;
        return { key: section.key, order, defaultIndex };
      })
      .sort((left, right) => left.order - right.order || left.defaultIndex - right.defaultIndex)
      .map((section) => section.key);
    const sectionLabels = Object.fromEntries(
      evidenceTemplateSections.map((section) => [
        section.key,
        String(formData.get(`section_label_${section.key}`) ?? section.defaultLabel).trim()
      ])
    );

    try {
      await updateOrganizationEvidencePackageTemplate(
        session.userId,
        session.organizationId,
        {
          template_name: templateName,
          include_report_artifact: formData.get("include_report_artifact") === "on",
          include_uploads: formData.get("include_uploads") === "on",
          include_approval_assignments: formData.get("include_approval_assignments") === "on",
          include_reviews: formData.get("include_reviews") === "on",
          include_comments: formData.get("include_comments") === "on",
          include_schedules: formData.get("include_schedules") === "on",
          section_labels: sectionLabels,
          section_order: sectionOrder
        },
        session.authToken
      );
    } catch {
      redirect("/settings?error=evidence-template-failed");
    }

    revalidatePath("/settings");
    revalidatePath("/reports");
    redirect("/settings?status=evidence-template-updated");
  }

  async function createPrivacyRequestAction(formData: FormData) {
    "use server";

    const requestType = String(formData.get("request_type") ?? "").trim() as OrganizationPrivacyRequestType;
    const subjectUserId = String(formData.get("subject_user_id") ?? "").trim();
    const description = String(formData.get("description") ?? "").trim();
    const session = await readAppSession();

    if (!privacyRequestTypes.includes(requestType)) {
      redirect("/settings?error=missing-fields");
    }
    if (!session.userId || !session.organizationId) {
      redirect("/settings?error=session-missing");
    }

    try {
      await createOrganizationPrivacyRequest(
        session.userId,
        session.organizationId,
        {
          request_type: requestType,
          subject_user_id: subjectUserId || null,
          description: description || null
        },
        session.authToken
      );
    } catch {
      redirect("/settings?error=privacy-request-failed");
    }

    revalidatePath("/settings");
    redirect("/settings?status=privacy-request-created");
  }

  async function updatePrivacyRequestAction(formData: FormData) {
    "use server";

    const privacyRequestId = String(formData.get("privacy_request_id") ?? "").trim();
    const requestStatus = String(formData.get("status") ?? "").trim() as OrganizationPrivacyRequestStatus;
    const resolutionNotes = String(formData.get("resolution_notes") ?? "").trim();
    const session = await readAppSession();

    if (!privacyRequestId || !privacyRequestStatuses.includes(requestStatus)) {
      redirect("/settings?error=missing-fields");
    }
    if (!session.userId || !session.organizationId) {
      redirect("/settings?error=session-missing");
    }

    try {
      await updateOrganizationPrivacyRequest(
        session.userId,
        session.organizationId,
        privacyRequestId,
        {
          status: requestStatus,
          resolution_notes: resolutionNotes || null
        },
        session.authToken
      );
    } catch {
      redirect("/settings?error=privacy-request-failed");
    }

    revalidatePath("/settings");
    redirect("/settings?status=privacy-request-updated");
  }

  async function updatePilotReadinessAction(formData: FormData) {
    "use server";

    const itemKey = String(formData.get("item_key") ?? "").trim();
    const itemStatus = String(formData.get("status") ?? "").trim() as OrganizationPilotReadinessStatus;
    const evidenceUrl = String(formData.get("evidence_url") ?? "").trim();
    const notes = String(formData.get("notes") ?? "").trim();
    const session = await readAppSession();

    if (!itemKey || !pilotReadinessStatuses.includes(itemStatus)) {
      redirect("/settings?error=missing-fields");
    }
    if (!session.userId || !session.organizationId) {
      redirect("/settings?error=session-missing");
    }

    try {
      await updateOrganizationPilotReadinessItem(
        session.userId,
        session.organizationId,
        itemKey,
        {
          status: itemStatus,
          evidence_url: evidenceUrl || null,
          notes: notes || null
        },
        session.authToken
      );
    } catch {
      redirect("/settings?error=pilot-readiness-failed");
    }

    revalidatePath("/settings");
    redirect("/settings?status=pilot-readiness-updated");
  }

  async function updateAnalysisConfigAction(formData: FormData) {
    "use server";

    const rawConfig = String(formData.get("analysis_config") ?? "").trim();
    const session = await readAppSession();

    if (!rawConfig) {
      redirect("/settings?error=missing-fields");
    }
    if (!session.userId || !session.organizationId) {
      redirect("/settings?error=session-missing");
    }

    try {
      const config = JSON.parse(rawConfig) as Record<string, unknown>;
      const currentAnalysis =
        config.analysis && typeof config.analysis === "object" && !Array.isArray(config.analysis)
          ? (config.analysis as Record<string, unknown>)
          : {};
      config.analysis = {
        ...currentAnalysis,
        weather_normalisation: formData.get("weather_normalisation") === "on",
        weather_model_min_baseline_records: Number(
          formData.get("weather_model_min_baseline_records") ?? 10
        ),
        weather_model_min_comparison_records: Number(
          formData.get("weather_model_min_comparison_records") ?? 1
        )
      };
      await updateOrganizationAnalysisConfig(
        session.userId,
        session.organizationId,
        { config },
        session.authToken
      );
    } catch {
      redirect("/settings?error=analysis-config-failed");
    }

    revalidatePath("/settings");
    revalidatePath("/dashboard");
    redirect("/settings?status=analysis-config-updated");
  }

  async function createDataConnectorAction(formData: FormData) {
    "use server";

    const siteId = String(formData.get("site_id") ?? "").trim();
    const name = String(formData.get("name") ?? "").trim();
    const connectorType = String(formData.get("connector_type") ?? "").trim() as DataConnectorType;
    const sourceCategory = String(formData.get("source_category") ?? "").trim();
    const schedule = String(formData.get("schedule") ?? "manual").trim() as DataConnectorSchedule;
    const externalReference = String(formData.get("external_reference") ?? "").trim();
    const credentialReference = String(formData.get("credential_reference") ?? "").trim();
    const nextSyncAt = String(formData.get("next_sync_at") ?? "").trim();
    const session = await readAppSession();

    if (!siteId || !name || !connectorType || !sourceCategory || !schedule) {
      redirect("/settings?error=missing-fields");
    }
    if (!session.userId || !session.organizationId) {
      redirect("/settings?error=session-missing");
    }

    try {
      await createOrganizationDataConnector(
        session.userId,
        session.organizationId,
        {
          connector_id: `connector_${crypto.randomUUID()}`,
          site_id: siteId,
          name,
          connector_type: connectorType,
          source_category: sourceCategory,
          status: "draft",
          schedule,
          external_reference: externalReference || null,
          credential_reference: credentialReference || null,
          next_sync_at: nextSyncAt || null
        },
        session.authToken
      );
    } catch {
      redirect("/settings?error=data-connector-failed");
    }

    revalidatePath("/settings");
    revalidatePath("/dashboard");
    redirect("/settings?status=data-connector-created");
  }

  async function updateDataConnectorAction(formData: FormData) {
    "use server";

    const connectorId = String(formData.get("connector_id") ?? "").trim();
    const name = String(formData.get("name") ?? "").trim();
    const status = String(formData.get("status") ?? "").trim() as DataConnectorStatus;
    const schedule = String(formData.get("schedule") ?? "").trim() as DataConnectorSchedule;
    const externalReference = String(formData.get("external_reference") ?? "").trim();
    const credentialReference = String(formData.get("credential_reference") ?? "").trim();
    const nextSyncAt = String(formData.get("next_sync_at") ?? "").trim();
    const session = await readAppSession();

    if (!connectorId || !name || !status || !schedule) {
      redirect("/settings?error=missing-fields");
    }
    if (!session.userId || !session.organizationId) {
      redirect("/settings?error=session-missing");
    }

    try {
      const payload = {
        name,
        status,
        schedule,
        external_reference: externalReference || null,
        next_sync_at: nextSyncAt || null
      } as const;
      await updateOrganizationDataConnector(
        session.userId,
        session.organizationId,
        connectorId,
        credentialReference ? { ...payload, credential_reference: credentialReference } : payload,
        session.authToken
      );
    } catch {
      redirect("/settings?error=data-connector-failed");
    }

    revalidatePath("/settings");
    revalidatePath("/dashboard");
    redirect("/settings?status=data-connector-updated");
  }

  async function startBillingCheckoutAction(formData: FormData) {
    "use server";

    const planId = String(formData.get("plan_id") ?? "").trim();
    const session = await readAppSession();

    if (!planId) {
      redirect("/settings?error=missing-fields");
    }
    if (!session.userId || !session.organizationId) {
      redirect("/settings?error=session-missing");
    }

    let checkoutUrl = "";
    try {
      const checkout = await createBillingCheckoutSession(
        session.userId,
        session.organizationId,
        {
          plan_id: planId,
          success_url: "/settings?status=billing-updated",
          cancel_url: "/settings?error=billing-failed"
        },
        session.authToken
      );
      checkoutUrl = checkout.checkout_url;
    } catch {
      redirect("/settings?error=billing-failed");
    }

    redirect(checkoutUrl);
  }

  async function openBillingPortalAction() {
    "use server";

    const session = await readAppSession();

    if (!session.userId || !session.organizationId) {
      redirect("/settings?error=session-missing");
    }

    let portalUrl = "";
    try {
      const portal = await createBillingPortalSession(
        session.userId,
        session.organizationId,
        {
          return_url: "/settings"
        },
        session.authToken
      );
      portalUrl = portal.portal_url;
    } catch {
      redirect("/settings?error=billing-failed");
    }

    redirect(portalUrl);
  }

  async function applyReportRetentionAction() {
    "use server";

    const session = await readAppSession();

    if (!session.userId || !session.organizationId) {
      redirect("/settings?error=session-missing");
    }

    try {
      const retention = await applyOrganizationReportRetention(
        session.userId,
        session.organizationId,
        session.authToken
      );
      revalidatePath("/settings");
      revalidatePath("/reports");
      redirect(
        `/settings?status=retention-applied&expired=${retention.expired_count}` +
          `&deleted=${retention.deleted_artifact_count}` +
          `&missing=${retention.missing_artifact_count}` +
          `&packageDeleted=${retention.deleted_evidence_package_artifact_count}` +
          `&packageMissing=${retention.missing_evidence_package_artifact_count}` +
          `&packageMetadata=${retention.deleted_evidence_package_metadata_count}`
      );
    } catch {
      redirect("/settings?error=retention-failed");
    }
  }

  return (
    <WorkspaceShell
      currentPath="/settings"
      title="Administration"
      modeLabel={
        loadError ? "Workspace data unavailable" : mode === "live" ? "Live workspace" : "Sample workspace"
      }
      modeDescription={
        loadError
          ? "We could not load your organization's live data from the backend. Reload the page to try again."
          : mode === "live"
            ? "Members, invites, and organization records are loading from the backend."
            : "This page is showing sample administration data until a live organization is selected."
      }
      modeTone={loadError ? "error" : "info"}
    >
      {pageMessage ? (
        <div
          className={`authNotice ${pageMessage.tone === "success" ? "authSuccess" : "authError"}`}
          role={pageMessage.tone === "success" ? "status" : "alert"}
        >
          <strong>{pageMessage.title}</strong>
          <span>{pageMessage.body}</span>
        </div>
      ) : null}
      {loadError ? (
        <div className="authNotice authError" role="alert">
          <strong>Workspace data unavailable</strong>
          <span>Tenant setup data could not be loaded from the API.</span>
        </div>
      ) : null}

      <section className="summaryGrid" aria-label="Settings metrics">
        {permissions.canManageMembers ? (
          <>
            <div className="summaryTile">
              <span>Team members</span>
              <strong>{memberships.length}</strong>
            </div>
            <div className="summaryTile">
              <span>Active users</span>
              <strong>{activeMembers.length}</strong>
            </div>
          </>
        ) : null}
        <div className="summaryTile">
          <span>Sites</span>
          <strong>{sites.length}</strong>
        </div>
        <div className="summaryTile">
          <span>SEU meters</span>
          <strong>{seuMeters.length}</strong>
        </div>
        {privacySettings ? (
          <div className="summaryTile">
            <span>Privacy status</span>
            <strong>
              {privacySettings.data_processing_acknowledged_at ? "Acknowledged" : "Pending"}
            </strong>
          </div>
        ) : null}
        {permissions.canManageOrganization ? (
          <div className="summaryTile">
            <span>Open privacy requests</span>
            <strong>{activePrivacyRequestCount}</strong>
          </div>
        ) : null}
        {permissions.canViewAudit ? (
          <div className="summaryTile">
            <span>Audit records</span>
            <strong>{auditEvents.length}</strong>
          </div>
        ) : null}
        {subscription ? (
          <div className="summaryTile">
            <span>Plan</span>
            <strong>{subscription.plan.name}</strong>
          </div>
        ) : null}
        {permissions.canManageOrganization && pilotReadiness ? (
          <div className="summaryTile">
            <span>Pilot readiness</span>
            <strong>{pilotReadinessLabel}</strong>
          </div>
        ) : null}
        {permissions.canManageMembers ? (
          <div className="summaryTile">
            <span>Open invites</span>
            <strong>{invites.filter((invite) => invite.status === "pending").length}</strong>
          </div>
        ) : null}
      </section>

      <section className="contentGrid">
        <div className="listPanel wide">
          <div className="sectionHeader">
            <h2>Tenant profile</h2>
            <span>Organization identity</span>
          </div>
          {permissions.canManageOrganization && organization ? (
            <form action={updateProfileAction} className="inlineFormWide">
              <label className="authField">
                <span>Name</span>
                <input name="name" type="text" defaultValue={organization.name} />
              </label>
              <label className="authField">
                <span>Slug</span>
                <input name="slug" type="text" defaultValue={organization.slug} />
              </label>
              <button type="submit" className="btn btnPrimary btnSm">
                Update profile
              </button>
            </form>
          ) : organization ? (
            <div className="rowList">
              <div className="dataRow">
                <div>
                  <strong>{organization.name}</strong>
                  <span>{organization.slug}</span>
                </div>
                <div>
                  <strong>{organization.id}</strong>
                  <span>Tenant ID</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="emptyState">
              <strong>No tenant profile loaded</strong>
              <span>Select a live workspace to manage tenant details.</span>
            </div>
          )}
        </div>

        <div className="listPanel wide">
          <div className="sectionHeader">
            <h2>Report branding</h2>
            <span>{reportBranding?.updated_at ? "Customized" : "Default pack"}</span>
          </div>
          {reportBranding ? (
            permissions.canManageOrganization ? (
              <form action={updateReportBrandingAction} className="inlineFormWide">
                <label className="authField">
                  <span>Evidence pack title</span>
                  <input
                    name="evidence_pack_title"
                    type="text"
                    defaultValue={reportBranding.evidence_pack_title}
                  />
                </label>
                <label className="authField">
                  <span>Accent color</span>
                  <input name="primary_color" type="color" defaultValue={reportBranding.primary_color} />
                </label>
                <label className="authField">
                  <span>Logo URL</span>
                  <input
                    name="logo_url"
                    type="url"
                    defaultValue={reportBranding.logo_url ?? ""}
                    placeholder="https://example.com/logo.png"
                  />
                </label>
                <label className="authField">
                  <span>Support email</span>
                  <input
                    name="support_email"
                    type="email"
                    defaultValue={reportBranding.support_email ?? ""}
                    placeholder="energy.team@example.com"
                  />
                </label>
                <button type="submit" className="btn btnPrimary btnSm">
                  Save report branding
                </button>
              </form>
            ) : (
              <div className="rowList">
                <div className="dataRow">
                  <div>
                    <strong>{reportBranding.evidence_pack_title}</strong>
                    <span>Evidence pack title</span>
                  </div>
                  <div>
                    <strong>{reportBranding.primary_color}</strong>
                    <span>Accent color</span>
                  </div>
                </div>
                <div className="dataRow">
                  <div>
                    <strong>{reportBranding.logo_url ?? "Not configured"}</strong>
                    <span>Logo URL</span>
                  </div>
                  <div>
                    <strong>{reportBranding.support_email ?? "Not configured"}</strong>
                    <span>Support email</span>
                  </div>
                </div>
              </div>
            )
          ) : (
            <div className="emptyState">
              <strong>Report branding unavailable</strong>
              <span>Select a live workspace to manage tenant evidence package branding.</span>
            </div>
          )}
        </div>

        <div className="listPanel wide">
          <div className="sectionHeader">
            <h2>Evidence template</h2>
            <span>{evidencePackageTemplate?.updated_at ? "Customized" : "Full package"}</span>
          </div>
          {evidencePackageTemplate ? (
            permissions.canManageOrganization ? (
              <form action={updateEvidencePackageTemplateAction} className="evidenceTemplateForm">
                <label className="authField">
                  <span>Template name</span>
                  <input
                    name="template_name"
                    type="text"
                    defaultValue={evidencePackageTemplate.template_name}
                  />
                </label>
                <div className="templateSectionGrid">
                  {evidenceTemplateSections.map((section) => (
                    <div className="templateSectionRow" key={section.key}>
                      <label className="authCheckbox">
                        <input
                          name={section.includeField}
                          type="checkbox"
                          defaultChecked={evidencePackageTemplate[section.includeField]}
                        />
                        <span>{section.name}</span>
                      </label>
                      <label className="authField">
                        <span>Label</span>
                        <input
                          name={`section_label_${section.key}`}
                          type="text"
                          defaultValue={evidenceTemplateSectionLabel(
                            evidencePackageTemplate,
                            section
                          )}
                        />
                      </label>
                      <label className="authField templateOrderField">
                        <span>Order</span>
                        <input
                          name={`section_order_${section.key}`}
                          type="number"
                          min="1"
                          max={evidenceTemplateSections.length}
                          defaultValue={evidenceTemplateOrderValue(
                            evidencePackageTemplate,
                            section
                          )}
                        />
                      </label>
                    </div>
                  ))}
                </div>
                <button type="submit" className="btn btnPrimary btnSm">
                  Save evidence template
                </button>
              </form>
            ) : (
              <div className="rowList">
                <div className="dataRow">
                  <div>
                    <strong>{evidencePackageTemplate.template_name}</strong>
                    <span>Template name</span>
                  </div>
                  <div>
                    <strong>
                      {enabledEvidenceTemplateSectionCount(evidencePackageTemplate)}
                    </strong>
                    <span>Enabled sections</span>
                  </div>
                  <div>
                    <strong>
                      {orderedEvidenceTemplateSections(evidencePackageTemplate)
                        .filter((section) => evidencePackageTemplate[section.includeField])
                        .map((section) =>
                          evidenceTemplateSectionLabel(evidencePackageTemplate, section)
                        )
                        .join(" / ") || "None"}
                    </strong>
                    <span>Section order</span>
                  </div>
                </div>
              </div>
            )
          ) : (
            <div className="emptyState">
              <strong>Evidence template unavailable</strong>
              <span>Select a live workspace to manage evidence package sections.</span>
            </div>
          )}
        </div>

        <div className="listPanel wide">
          <div className="sectionHeader">
            <h2>Approval policy</h2>
            <span>{reportApprovalPolicy?.updated_at ? "Customized" : "Default rules"}</span>
          </div>
          {reportApprovalPolicy ? (
            permissions.canManageOrganization ? (
              <form action={updateReportApprovalPolicyAction} className="inlineFormWide">
                <label className="authCheckbox">
                  <input
                    name="auto_lock_evidence_on_final_approval"
                    type="checkbox"
                    defaultChecked={reportApprovalPolicy.auto_lock_evidence_on_final_approval}
                  />
                  <span>Lock evidence after final approval</span>
                </label>
                <label className="authField">
                  <span>Minimum approved assignments</span>
                  <input
                    name="minimum_approval_assignments_for_lock"
                    type="number"
                    min="0"
                    max="25"
                    defaultValue={reportApprovalPolicy.minimum_approval_assignments_for_lock}
                  />
                </label>
                <button type="submit" className="btn btnPrimary btnSm">
                  Save approval policy
                </button>
              </form>
            ) : (
              <div className="rowList">
                <div className="dataRow">
                  <div>
                    <strong>
                      {reportApprovalPolicy.auto_lock_evidence_on_final_approval
                        ? "Enabled"
                        : "Disabled"}
                    </strong>
                    <span>Automatic evidence locking</span>
                  </div>
                  <div>
                    <strong>{reportApprovalPolicy.minimum_approval_assignments_for_lock}</strong>
                    <span>Minimum approved assignments</span>
                  </div>
                </div>
              </div>
            )
          ) : (
            <div className="emptyState">
              <strong>Approval policy unavailable</strong>
              <span>Select a live workspace to manage report approval rules.</span>
            </div>
          )}
        </div>

        <div className="listPanel wide">
          <div className="sectionHeader">
            <h2>Privacy and consent</h2>
            <span>
              {privacySettings?.data_processing_acknowledged_at
                ? "Data processing acknowledged"
                : "Data processing acknowledgement pending"}
            </span>
          </div>
          {privacySettings ? (
            permissions.canManageOrganization ? (
              <form action={updatePrivacySettingsAction} className="inlineFormWide">
                <label className="authField">
                  <span>Privacy contact email</span>
                  <input
                    name="data_processing_contact_email"
                    type="email"
                    defaultValue={privacySettings.data_processing_contact_email ?? ""}
                    placeholder="privacy@example.com"
                  />
                </label>
                <label className="authCheckbox">
                  <input
                    name="subprocessor_notifications_enabled"
                    type="checkbox"
                    defaultChecked={privacySettings.subprocessor_notifications_enabled}
                  />
                  <span>Receive subprocessor change notices</span>
                </label>
                <label className="authCheckbox">
                  <input
                    name="analytics_data_sharing_enabled"
                    type="checkbox"
                    defaultChecked={privacySettings.analytics_data_sharing_enabled}
                  />
                  <span>Allow anonymized product analytics</span>
                </label>
                {!privacySettings.data_processing_acknowledged_at ? (
                  <label className="authCheckbox">
                    <input name="acknowledge_data_processing" type="checkbox" />
                    <span>Record data-processing acknowledgement</span>
                  </label>
                ) : null}
                <button type="submit" className="btn btnPrimary btnSm">
                  Save privacy settings
                </button>
              </form>
            ) : (
              <div className="rowList">
                <div className="dataRow">
                  <div>
                    <strong>{formatDateTime(privacySettings.data_processing_acknowledged_at)}</strong>
                    <span>Data processing acknowledgement</span>
                  </div>
                  <div>
                    <strong>
                      {privacySettings.data_processing_acknowledged_by_user_id ?? "Not recorded"}
                    </strong>
                    <span>Acknowledged by</span>
                  </div>
                </div>
                <div className="dataRow">
                  <div>
                    <strong>{privacySettings.data_processing_contact_email ?? "Not configured"}</strong>
                    <span>Privacy contact</span>
                  </div>
                  <div>
                    <strong>
                      {privacySettings.subprocessor_notifications_enabled ? "Enabled" : "Disabled"}
                    </strong>
                    <span>Subprocessor notices</span>
                  </div>
                  <div>
                    <strong>
                      {privacySettings.analytics_data_sharing_enabled ? "Enabled" : "Disabled"}
                    </strong>
                    <span>Anonymized product analytics</span>
                  </div>
                </div>
              </div>
            )
          ) : (
            <div className="emptyState">
              <strong>Privacy controls unavailable</strong>
              <span>Select a live workspace to review tenant privacy preferences.</span>
            </div>
          )}
        </div>

        {permissions.canManageOrganization ? (
          <div className="listPanel wide">
            <div className="sectionHeader">
              <h2>Privacy requests</h2>
              <span>{activePrivacyRequestCount} open</span>
            </div>
            <form action={createPrivacyRequestAction} className="inlineFormWide">
              <label className="authField">
                <span>Type</span>
                <select name="request_type" defaultValue="deletion">
                  {privacyRequestTypes.map((requestType) => (
                    <option key={requestType} value={requestType}>
                      {formatPrivacyRequestType(requestType)}
                    </option>
                  ))}
                </select>
              </label>
              <label className="authField">
                <span>Subject user</span>
                <select name="subject_user_id" defaultValue="">
                  <option value="">Tenant-level request</option>
                  {memberships.map((membership) => (
                    <option key={membership.user_id} value={membership.user_id}>
                      {membership.display_name}
                    </option>
                  ))}
                </select>
              </label>
              <label className="authField">
                <span>Description</span>
                <input name="description" type="text" placeholder="Request context" />
              </label>
              <button type="submit" className="btn btnPrimary btnSm">
                Create request
              </button>
            </form>
            {privacyRequests.length > 0 ? (
              <div className="rowList">
                {privacyRequests.map((privacyRequest) => (
                  <div className="dataRow" key={privacyRequest.id}>
                    <div>
                      <strong>{formatPrivacyRequestType(privacyRequest.request_type)}</strong>
                      <span>{privacyRequest.description ?? privacyRequest.id}</span>
                    </div>
                    <div>
                      <strong>{formatPrivacyRequestStatus(privacyRequest.status)}</strong>
                      <span>{formatDateTime(privacyRequest.created_at)}</span>
                    </div>
                    <div>
                      <strong>{privacyRequest.subject_user_id ?? "Tenant-level"}</strong>
                      <span>Subject user</span>
                    </div>
                    {privacyRequest.completed_at ? (
                      <div>
                        <strong>{formatDateTime(privacyRequest.completed_at)}</strong>
                        <span>Completed</span>
                      </div>
                    ) : null}
                    <form action={updatePrivacyRequestAction} className="memberActionForm">
                      <input
                        type="hidden"
                        name="privacy_request_id"
                        value={privacyRequest.id}
                      />
                      <label className="srOnly" htmlFor={`privacy-status-${privacyRequest.id}`}>
                        Status
                      </label>
                      <select
                        id={`privacy-status-${privacyRequest.id}`}
                        name="status"
                        defaultValue={privacyRequest.status}
                      >
                        {privacyRequestStatuses.map((requestStatus) => (
                          <option key={requestStatus} value={requestStatus}>
                            {formatPrivacyRequestStatus(requestStatus)}
                          </option>
                        ))}
                      </select>
                      <label className="srOnly" htmlFor={`privacy-notes-${privacyRequest.id}`}>
                        Resolution notes
                      </label>
                      <input
                        id={`privacy-notes-${privacyRequest.id}`}
                        name="resolution_notes"
                        type="text"
                        defaultValue={privacyRequest.resolution_notes ?? ""}
                        placeholder="Resolution notes"
                      />
                      <button type="submit" className="btn btnGhost btnSm">
                        Update
                      </button>
                    </form>
                  </div>
                ))}
              </div>
            ) : (
              <div className="emptyState">
                <strong>No privacy requests</strong>
                <span>Tenant privacy operations are clear.</span>
              </div>
            )}
          </div>
        ) : null}

        <div className="listPanel wide">
          <div className="sectionHeader">
            <h2>Subscription</h2>
            <span>{subscription ? subscription.status : "Not configured"}</span>
          </div>
          {subscription ? (
            <div className="rowList">
              <div className="dataRow">
                <div>
                  <strong>{subscription.plan.name}</strong>
                  <span>{subscription.plan.description}</span>
                </div>
                <div>
                  <strong>{formatMonthlyPrice(subscription.plan.monthly_price_cents)}</strong>
                  <span>Monthly price</span>
                </div>
              </div>
              <div className="dataRow">
                <div>
                  <strong>
                    {subscription.usage.active_user_count}/{formatLimit(subscription.plan.included_users)}
                  </strong>
                  <span>Active users</span>
                </div>
                <div>
                  <strong>{subscription.usage.pending_invite_count}</strong>
                  <span>Pending invite seats</span>
                </div>
                <div>
                  <strong>
                    {subscription.usage.site_count}/{formatLimit(subscription.plan.included_sites)}
                  </strong>
                  <span>Sites</span>
                </div>
                <div>
                  <strong>
                    {subscription.usage.runs_this_period}/{formatLimit(subscription.plan.monthly_run_limit)}
                  </strong>
                  <span>Runs this period</span>
                </div>
                <div>
                  <strong>{formatLimit(subscription.plan.report_retention_days)}</strong>
                  <span>Report retention days</span>
                </div>
              </div>
              <div className="dataRow">
                <div>
                  <strong>{subscription.current_period_end ?? "Open ended"}</strong>
                  <span>Current period end</span>
                </div>
                <div>
                  <strong>{subscription.trial_ends_at ?? "No trial"}</strong>
                  <span>Trial end</span>
                </div>
              </div>
              {permissions.canManageBilling ? (
                <div className="dataRow">
                  <form action={startBillingCheckoutAction} className="memberActionForm">
                    <label className="srOnly" htmlFor="billing-plan">
                      Plan
                    </label>
                    <select
                      id="billing-plan"
                      name="plan_id"
                      defaultValue={checkoutPlans[0]?.id ?? ""}
                      disabled={checkoutPlans.length === 0}
                    >
                      {checkoutPlans.map((plan) => (
                        <option key={plan.id} value={plan.id}>
                          {plan.name} - {formatMonthlyPrice(plan.monthly_price_cents)}
                        </option>
                      ))}
                    </select>
                    <button
                      type="submit"
                      className="btn btnPrimary btnSm"
                      disabled={checkoutPlans.length === 0}
                    >
                      Start checkout
                    </button>
                  </form>
                  {subscription.external_customer_id ? (
                    <form action={openBillingPortalAction}>
                      <button type="submit" className="btn btnGhost btnSm">
                        Billing portal
                      </button>
                    </form>
                  ) : null}
                </div>
              ) : null}
            </div>
          ) : (
            <div className="emptyState">
              <strong>No subscription loaded</strong>
              <span>Select a live workspace to review plan status.</span>
            </div>
          )}
        </div>

        <div className="listPanel wide">
          <div className="sectionHeader">
            <h2>Report retention</h2>
            <span>{subscription ? `${formatLimit(subscription.plan.report_retention_days)} days` : "Plan controlled"}</span>
          </div>
          <div className="rowList">
            <div className="dataRow">
              <div>
                <strong>Apply retention policy</strong>
                <span>Expired reports and locked evidence packages are cleaned through tenant storage.</span>
              </div>
              <div>
                {mode === "live" && permissions.canManageReports ? (
                  <form action={applyReportRetentionAction} className="memberActionForm">
                    <button type="submit" className="btn btnPrimary btnSm">
                      Apply retention
                    </button>
                  </form>
                ) : (
                  <span>{mode === "live" ? "Report managers" : "Live workspace"}</span>
                )}
              </div>
            </div>
            {query.status === "retention-applied" ? (
              <div className="dataRow">
                <div>
                  <strong>{formatQueryCount(query.expired)}</strong>
                  <span>Expired reports</span>
                </div>
                <div>
                  <strong>{formatQueryCount(query.deleted)}</strong>
                  <span>Artifacts deleted</span>
                </div>
                <div>
                  <strong>{formatQueryCount(query.missing)}</strong>
                  <span>Artifacts missing</span>
                </div>
                <div>
                  <strong>{formatQueryCount(query.packageDeleted)}</strong>
                  <span>Evidence packages deleted</span>
                </div>
                <div>
                  <strong>{formatQueryCount(query.packageMissing)}</strong>
                  <span>Evidence packages missing</span>
                </div>
                <div>
                  <strong>{formatQueryCount(query.packageMetadata)}</strong>
                  <span>Package records removed</span>
                </div>
              </div>
            ) : null}
          </div>
        </div>

        {permissions.canManageOrganization ? (
          <div className="listPanel wide">
            <div className="sectionHeader">
              <h2>Pilot readiness</h2>
              <span>{pilotReadiness ? `${pilotReadinessLabel} verified` : "Not loaded"}</span>
            </div>
            {pilotReadiness ? (
              <div className="rowList">
                {pilotReadiness.items.map((item) => (
                  <div className="dataRow" key={item.item_key}>
                    <div>
                      <strong>{item.label}</strong>
                      <span>{item.notes ?? item.item_key}</span>
                    </div>
                    <div>
                      <strong>{formatPilotReadinessStatus(item.status)}</strong>
                      <span>{item.evidence_url ?? "No evidence link"}</span>
                    </div>
                    <div>
                      <strong>{item.updated_by_user_id ?? "Not updated"}</strong>
                      <span>{formatDateTime(item.updated_at)}</span>
                    </div>
                    <form action={updatePilotReadinessAction} className="memberActionForm">
                      <input type="hidden" name="item_key" value={item.item_key} />
                      <label className="srOnly" htmlFor={`pilot-status-${item.item_key}`}>
                        Status
                      </label>
                      <select
                        id={`pilot-status-${item.item_key}`}
                        name="status"
                        defaultValue={item.status}
                      >
                        {pilotReadinessStatuses.map((statusOption) => (
                          <option key={statusOption} value={statusOption}>
                            {formatPilotReadinessStatus(statusOption)}
                          </option>
                        ))}
                      </select>
                      <label className="srOnly" htmlFor={`pilot-evidence-${item.item_key}`}>
                        Evidence URL
                      </label>
                      <input
                        id={`pilot-evidence-${item.item_key}`}
                        name="evidence_url"
                        type="url"
                        defaultValue={item.evidence_url ?? ""}
                        placeholder="Evidence URL"
                      />
                      <label className="srOnly" htmlFor={`pilot-notes-${item.item_key}`}>
                        Notes
                      </label>
                      <input
                        id={`pilot-notes-${item.item_key}`}
                        name="notes"
                        type="text"
                        defaultValue={item.notes ?? ""}
                        placeholder="Notes"
                      />
                      <button type="submit" className="btn btnGhost btnSm">
                        Update
                      </button>
                    </form>
                  </div>
                ))}
              </div>
            ) : (
              <div className="emptyState">
                <strong>Pilot checklist unavailable</strong>
                <span>Select a live workspace to manage launch evidence.</span>
              </div>
            )}
          </div>
        ) : null}

        {permissions.canManageMembers ? (
          <div className="listPanel wide">
            <div className="sectionHeader">
              <h2>Members</h2>
              <span>Access and roles</span>
            </div>
            <div className="rowList">
              {memberships.map((membership) => (
                <div className="dataRow" key={membership.user_id}>
                  <div>
                    <strong>{membership.display_name}</strong>
                    <span>{membership.email}</span>
                  </div>
                  <div>
                    <strong>{membership.role}</strong>
                    <span>{membership.is_active ? "Active" : "Disabled"}</span>
                  </div>
                  <form action={updateMemberRoleAction} className="memberActionForm">
                    <input type="hidden" name="member_user_id" value={membership.user_id} />
                    <label className="srOnly" htmlFor={`role-${membership.user_id}`}>
                      Role
                    </label>
                    <select
                      id={`role-${membership.user_id}`}
                      name="role"
                      defaultValue={membership.role}
                    >
                      <option value="viewer">Viewer</option>
                      <option value="manager">Manager</option>
                      <option value="owner">Owner</option>
                    </select>
                    <button type="submit" className="btn btnGhost btnSm">
                      Update
                    </button>
                  </form>
                  <form action={removeMemberAction}>
                    <input type="hidden" name="member_user_id" value={membership.user_id} />
                    <button type="submit" className="btn btnGhost btnSm">
                      Remove
                    </button>
                  </form>
                </div>
              ))}
            </div>
          </div>
        ) : null}

        <div className="listPanel">
          <div className="sectionHeader">
            <h2>Sites</h2>
            <span>Organization footprint</span>
          </div>
          {permissions.canManageAssets ? (
            <form action={createSiteAction} className="inlineForm">
              <label className="authField">
                <span>Name</span>
                <input name="name" type="text" placeholder="Main Site" />
              </label>
              <label className="authField">
                <span>Timezone</span>
                <input name="timezone" type="text" placeholder="Europe/London" />
              </label>
              <label className="authField">
                <span>Location (address or postcode)</span>
                <input name="location" type="text" placeholder="London, UK" />
              </label>
              <button type="submit" className="btn btnPrimary btnSm">
                Add site
              </button>
            </form>
          ) : null}
          <div className="rowList">
            {sites.map((site) => (
              <div className="dataRow" key={site.id}>
                <div>
                  <strong>{site.name}</strong>
                  <span>{site.timezone}</span>
                </div>
                <div>
                  <strong>{meters.filter((meter) => meter.site_id === site.id).length} meters</strong>
                  <span>
                    {site.location_label ?? "No location set"}
                    {site.latitude !== null && site.longitude !== null
                      ? " (auto weather enabled)"
                      : " (add a location to enable automatic HDD/CDD)"}
                  </span>
                </div>
              </div>
            ))}
          </div>
          {canCreateMeter ? (
            <form action={createMeterAction} className="inlineFormWide">
              <label className="authField">
                <span>Site</span>
                <select name="site_id" defaultValue={sites[0]?.id ?? ""}>
                  {sites.map((site) => (
                    <option key={site.id} value={site.id}>
                      {site.name}
                    </option>
                  ))}
                </select>
              </label>
              <label className="authField">
                <span>Meter name</span>
                <input name="display_name" type="text" placeholder="Main Electricity" />
              </label>
              <label className="authField">
                <span>Commodity</span>
                <input name="commodity" type="text" placeholder="electricity" />
              </label>
              <label className="authField">
                <span>Unit</span>
                <input name="unit" type="text" placeholder="kWh" />
              </label>
              <label className="authField">
                <span>Source column</span>
                <input name="source_column" type="text" placeholder="Main Electricity" />
              </label>
              <label className="authCheckbox">
                <input name="is_seu" type="checkbox" />
                <span>SEU meter</span>
              </label>
              <button type="submit" className="btn btnPrimary btnSm">
                Add meter
              </button>
            </form>
          ) : null}
        </div>

        <div className="listPanel wide">
          <div className="sectionHeader">
            <h2>Data connectors</h2>
            <span>{dataConnectors.length === 1 ? "1 connector" : `${dataConnectors.length} connectors`}</span>
          </div>
          {permissions.canManageAssets && sites.length > 0 ? (
            <form action={createDataConnectorAction} className="connectorForm">
              <label className="authField">
                <span>Name</span>
                <input name="name" type="text" placeholder="Main utility portal" />
              </label>
              <label className="authField">
                <span>Site</span>
                <select name="site_id" defaultValue={sites[0]?.id ?? ""}>
                  {sites.map((site) => (
                    <option key={site.id} value={site.id}>
                      {site.name}
                    </option>
                  ))}
                </select>
              </label>
              <label className="authField">
                <span>Type</span>
                <select name="connector_type" defaultValue="utility_portal">
                  {dataConnectorTypes.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="authField">
                <span>Source</span>
                <select name="source_category" defaultValue="energy">
                  {sourceCategoryOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="authField">
                <span>Cadence</span>
                <select name="schedule" defaultValue="monthly">
                  {dataConnectorSchedules.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="authField">
                <span>External reference</span>
                <input name="external_reference" type="text" placeholder="Provider, folder, or account ref" />
              </label>
              <label className="authField">
                <span>Credential reference</span>
                <input name="credential_reference" type="text" placeholder="Secret manager key" />
              </label>
              <label className="authField">
                <span>Next sync</span>
                <input name="next_sync_at" type="datetime-local" />
              </label>
              <button type="submit" className="btn btnPrimary btnSm">
                Add connector
              </button>
            </form>
          ) : null}
          {dataConnectors.length === 0 ? (
            <div className="emptyState">
              <strong>No connectors registered</strong>
              <span>Manual CSV upload remains available while connector automation is configured.</span>
            </div>
          ) : (
            <div className="rowList">
              {dataConnectors.map((connector) => {
                const siteName =
                  sites.find((site) => site.id === connector.site_id)?.name ?? connector.site_id;
                const typeLabel =
                  dataConnectorTypes.find((option) => option.value === connector.connector_type)
                    ?.label ?? connector.connector_type;
                const scheduleLabel =
                  dataConnectorSchedules.find((option) => option.value === connector.schedule)
                    ?.label ?? connector.schedule;
                const lastSyncLabel = connector.last_sync_status
                  ? `${connector.last_sync_status} at ${formatDateTime(connector.last_sync_at)}`
                  : "No sync evidence";
                return (
                  <div className="dataRow connectorRow" key={connector.id}>
                    <div>
                      <strong>{connector.name}</strong>
                      <span>
                        {typeLabel} - {siteName} - {connector.source_category}
                      </span>
                    </div>
                    <div>
                      <strong>{connector.status}</strong>
                      <span>
                        {scheduleLabel}; next {formatDateTime(connector.next_sync_at)}
                      </span>
                    </div>
                    <div>
                      <strong>Last sync</strong>
                      <span>{connector.last_sync_error ?? lastSyncLabel}</span>
                    </div>
                    <div>
                      <strong>{connector.credential_reference_configured ? "Configured" : "Missing"}</strong>
                      <span>{connector.external_reference ?? "No external reference"}</span>
                    </div>
                    {permissions.canManageAssets ? (
                      <form action={updateDataConnectorAction} className="connectorUpdateForm">
                        <input type="hidden" name="connector_id" value={connector.id} />
                        <label className="srOnly" htmlFor={`connector-name-${connector.id}`}>
                          Name
                        </label>
                        <input
                          id={`connector-name-${connector.id}`}
                          name="name"
                          type="text"
                          defaultValue={connector.name}
                        />
                        <label className="srOnly" htmlFor={`connector-status-${connector.id}`}>
                          Status
                        </label>
                        <select
                          id={`connector-status-${connector.id}`}
                          name="status"
                          defaultValue={connector.status}
                        >
                          {dataConnectorStatuses.map((option) => (
                            <option key={option.value} value={option.value}>
                              {option.label}
                            </option>
                          ))}
                        </select>
                        <label className="srOnly" htmlFor={`connector-schedule-${connector.id}`}>
                          Cadence
                        </label>
                        <select
                          id={`connector-schedule-${connector.id}`}
                          name="schedule"
                          defaultValue={connector.schedule}
                        >
                          {dataConnectorSchedules.map((option) => (
                            <option key={option.value} value={option.value}>
                              {option.label}
                            </option>
                          ))}
                        </select>
                        <label className="srOnly" htmlFor={`connector-reference-${connector.id}`}>
                          External reference
                        </label>
                        <input
                          id={`connector-reference-${connector.id}`}
                          name="external_reference"
                          type="text"
                          defaultValue={connector.external_reference ?? ""}
                          placeholder="External ref"
                        />
                        <label className="srOnly" htmlFor={`connector-credential-${connector.id}`}>
                          Credential reference
                        </label>
                        <input
                          id={`connector-credential-${connector.id}`}
                          name="credential_reference"
                          type="text"
                          placeholder={
                            connector.credential_reference_configured
                              ? "Leave blank to keep"
                              : "Secret manager key"
                          }
                        />
                        <label className="srOnly" htmlFor={`connector-next-${connector.id}`}>
                          Next sync
                        </label>
                        <input
                          id={`connector-next-${connector.id}`}
                          name="next_sync_at"
                          type="datetime-local"
                          defaultValue={formatDateTimeInput(connector.next_sync_at)}
                        />
                        <button type="submit" className="btn btnGhost btnSm">
                          Update
                        </button>
                      </form>
                    ) : null}
                  </div>
                );
              })}
            </div>
          )}
        </div>

        <div className="listPanel wide">
          <div className="sectionHeader">
            <h2>Analysis settings</h2>
            <span>{analysisConfig?.source === "saved" ? "Saved tenant config" : "Tenant asset config"}</span>
          </div>
          {analysisConfig ? (
            permissions.canManageAssets ? (
              <form action={updateAnalysisConfigAction} className="analysisConfigForm">
                <div className="analysisSettingsGrid">
                  <label className="authCheckbox">
                    <input
                      name="weather_normalisation"
                      type="checkbox"
                      defaultChecked={analysisBooleanSetting(
                        analysisConfig.config,
                        "weather_normalisation",
                        true
                      )}
                    />
                    <span>Weather-normalized M&amp;V models</span>
                  </label>
                  <label className="authField">
                    <span>Minimum baseline records</span>
                    <input
                      name="weather_model_min_baseline_records"
                      type="number"
                      min="2"
                      max="3650"
                      defaultValue={analysisNumberSetting(
                        analysisConfig.config,
                        "weather_model_min_baseline_records",
                        10
                      )}
                    />
                  </label>
                  <label className="authField">
                    <span>Minimum comparison records</span>
                    <input
                      name="weather_model_min_comparison_records"
                      type="number"
                      min="1"
                      max="3650"
                      defaultValue={analysisNumberSetting(
                        analysisConfig.config,
                        "weather_model_min_comparison_records",
                        1
                      )}
                    />
                  </label>
                </div>
                <label className="authField">
                  <span>Configuration JSON</span>
                  <textarea
                    name="analysis_config"
                    rows={18}
                    defaultValue={JSON.stringify(analysisConfig.config, null, 2)}
                  />
                </label>
                <div className="toolbarRow">
                  <button type="submit" className="btn btnPrimary btnSm">
                    Save analysis settings
                  </button>
                </div>
              </form>
            ) : (
              <pre className="codeBlock">{JSON.stringify(analysisConfig.config, null, 2)}</pre>
            )
          ) : (
            <div className="emptyState">
              <strong>Analysis settings unavailable</strong>
              <span>Add at least one site and meter before saving tenant analysis settings.</span>
            </div>
          )}
        </div>

        {permissions.canManageMembers ? (
          <div className="listPanel">
            <div className="sectionHeader">
              <h2>Invites</h2>
              <span>Pending access</span>
            </div>
            <form action={createInviteAction} className="inlineForm">
              <label className="authField">
                <span>Email</span>
                <input name="email" type="email" placeholder="invitee@example.com" />
              </label>
              <label className="authField">
                <span>Role</span>
                <select name="role" defaultValue="viewer">
                  <option value="viewer">Viewer</option>
                  <option value="manager">Manager</option>
                  <option value="owner">Owner</option>
                </select>
              </label>
              <button type="submit" className="btn btnPrimary btnSm">
                Send invite
              </button>
            </form>
            <div className="rowList">
              {invites.map((invite) => (
                <div className="dataRow" key={invite.id}>
                  <div>
                    <strong>{invite.email}</strong>
                    <span>{invite.id}</span>
                  </div>
                  <div>
                    <strong>{invite.role}</strong>
                    <span>{invite.status}</span>
                  </div>
                  {invite.status === "pending" ? (
                    <form action={revokeInviteAction}>
                      <input type="hidden" name="invite_id" value={invite.id} />
                      <button type="submit" className="btn btnGhost btnSm">
                        Revoke
                      </button>
                    </form>
                  ) : null}
                </div>
              ))}
            </div>
          </div>
        ) : null}

        {permissions.canViewAudit ? (
          <div className="listPanel">
            <div className="sectionHeader">
              <h2>Activity log</h2>
              <span>Recent admin events</span>
            </div>
            <div className="toolbarRow">
              <Link href="/settings/audit-export" className="btn btnGhost btnSm">
                Open CSV export
              </Link>
            </div>
            <div className="rowList">
              {auditEvents.map((event) => (
                <div className="dataRow" key={event.id}>
                  <div>
                    <strong>{event.action}</strong>
                    <span>
                      {event.resource_type} - {event.resource_id}
                    </span>
                  </div>
                  <div>
                    <strong>{event.actor_user_id}</strong>
                    <span>{formatDateTime(event.created_at)}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ) : null}
      </section>
    </WorkspaceShell>
  );
}
