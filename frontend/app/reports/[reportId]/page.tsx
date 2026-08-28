import { revalidatePath } from "next/cache";
import Link from "next/link";
import { notFound, redirect } from "next/navigation";
import { ReportArtifactSummary } from "../../../components/ReportArtifactSummary";
import { WorkspaceShell } from "../../../components/WorkspaceShell";
import {
  createOrganizationReportComment,
  createOrganizationReportEvidencePackage,
  createOrganizationReportApprovalAssignment,
  createOrganizationReportReview,
  createOrganizationReportSchedule,
  getOrganizationReportApprovers,
  getOrganizationReportApprovalAssignments,
  getOrganizationReportArtifact,
  getOrganizationReportComments,
  getOrganizationReportEvidencePackages,
  getOrganizationReportReviews,
  getOrganizationReportScheduleDeliveries,
  getOrganizationReportSchedules,
  updateOrganizationReportApprovalAssignment,
  updateOrganizationReportSchedule,
  type ReportApproverSummary,
  type ReportArtifactResponse,
  type ReportCommentSummary,
  type ReportEvidencePackageSummary,
  type ReportApprovalAssignmentStatus,
  type ReportApprovalAssignmentSummary,
  type ReportReviewStatus,
  type ReportReviewSummary,
  type ReportScheduleDeliverySummary,
  type ReportScheduleFrequency,
  type ReportScheduleSummary
} from "../../../lib/api";
import { loadDashboardData } from "../../../lib/dashboard-data";
import { readAppSession } from "../../../lib/session";

const scheduleFrequencyOptions: { value: ReportScheduleFrequency; label: string }[] = [
  { value: "weekly", label: "Weekly" },
  { value: "monthly", label: "Monthly" },
  { value: "quarterly", label: "Quarterly" }
];

const reviewStatusOptions: { value: ReportReviewStatus; label: string }[] = [
  { value: "approved", label: "Approved" },
  { value: "changes_requested", label: "Changes requested" }
];

const approvalAssignmentStatusOptions: {
  value: Exclude<ReportApprovalAssignmentStatus, "pending">;
  label: string;
}[] = [
  { value: "approved", label: "Approved" },
  { value: "changes_requested", label: "Changes requested" },
  { value: "canceled", label: "Canceled" }
];

type ReportDetailPageProps = {
  params: Promise<{ reportId: string }>;
  searchParams: Promise<{ status?: string; error?: string }>;
};

function formatArtifactValue(value: unknown): string {
  if (value === null || value === undefined) {
    return "Not recorded";
  }
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  return JSON.stringify(value);
}

function formatCommentTime(value: string): string {
  const parsed = new Date(value);
  if (Number.isNaN(parsed.valueOf())) {
    return value;
  }
  return new Intl.DateTimeFormat("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit"
  }).format(parsed);
}

function formatFileSize(value: number): string {
  if (value < 1024) {
    return `${value} B`;
  }
  if (value < 1024 * 1024) {
    return `${(value / 1024).toFixed(1)} KB`;
  }
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

function formatEvidencePackageCounts(packageSummary: ReportEvidencePackageSummary): string {
  return [
    `${packageSummary.upload_count} uploads`,
    `${packageSummary.review_count} reviews`,
    `${packageSummary.comment_count} notes`,
    `${packageSummary.schedule_count} schedules`
  ].join(" - ");
}

function formatScheduleFrequency(value: ReportScheduleFrequency): string {
  return scheduleFrequencyOptions.find((option) => option.value === value)?.label ?? value;
}

function formatScheduleDeliveryStatus(schedule: ReportScheduleSummary): string {
  if (!schedule.last_delivery_status || !schedule.last_delivery_at) {
    return "No delivery attempt yet";
  }
  const label = schedule.last_delivery_status === "sent" ? "Last sent" : "Last failed";
  return `${label} ${formatCommentTime(schedule.last_delivery_at)}`;
}

function formatScheduleDeliveryAttempt(delivery: ReportScheduleDeliverySummary): string {
  const label = delivery.status === "sent" ? "Sent" : "Failed";
  return `${label} ${formatCommentTime(delivery.sent_at ?? delivery.attempted_at)}`;
}

function formatReviewStatus(value: ReportReviewStatus): string {
  return reviewStatusOptions.find((option) => option.value === value)?.label ?? value;
}

function formatApprovalAssignmentStatus(value: ReportApprovalAssignmentStatus): string {
  if (value === "pending") {
    return "Pending";
  }
  return approvalAssignmentStatusOptions.find((option) => option.value === value)?.label ?? value;
}

function formatApprovalAssignee(
  assignment: ReportApprovalAssignmentSummary,
  approvers: ReportApproverSummary[]
): string {
  const approver = approvers.find((item) => item.user_id === assignment.assigned_to_user_id);
  return approver ? `${approver.display_name} (${approver.role})` : assignment.assigned_to_user_id;
}

function reportMessage(
  status: string | undefined,
  error: string | undefined
): { tone: "success" | "error"; title: string; body: string } | null {
  if (status === "comment-created") {
    return {
      tone: "success",
      title: "Comment added",
      body: "The report review note is ready for the workspace."
    };
  }
  if (status === "schedule-created") {
    return {
      tone: "success",
      title: "Schedule saved",
      body: "The report delivery schedule is ready for the workspace."
    };
  }
  if (status === "schedule-updated") {
    return {
      tone: "success",
      title: "Schedule updated",
      body: "The report delivery schedule has been updated."
    };
  }
  if (status === "review-recorded") {
    return {
      tone: "success",
      title: "Review recorded",
      body: "The management review decision is now part of the tenant evidence trail."
    };
  }
  if (status === "approval-assigned") {
    return {
      tone: "success",
      title: "Approval assigned",
      body: "The report approval request is now in the review workflow."
    };
  }
  if (status === "approval-updated") {
    return {
      tone: "success",
      title: "Approval updated",
      body: "The approval assignment and review trail have been updated."
    };
  }
  if (status === "evidence-package-created") {
    return {
      tone: "success",
      title: "Evidence package locked",
      body: "The current report evidence has been archived for this tenant."
    };
  }
  if (error === "missing-comment") {
    return {
      tone: "error",
      title: "Comment not added",
      body: "Enter a review note before adding it to the report."
    };
  }
  if (error === "missing-schedule-fields") {
    return {
      tone: "error",
      title: "Schedule not saved",
      body: "Enter a recipient, frequency, and next delivery time."
    };
  }
  if (error === "missing-review-status") {
    return {
      tone: "error",
      title: "Review not recorded",
      body: "Choose an approval decision before saving the management review."
    };
  }
  if (error === "missing-approval-assignment") {
    return {
      tone: "error",
      title: "Assignment not saved",
      body: "Choose an approver before assigning report approval."
    };
  }
  if (error === "session-missing") {
    return {
      tone: "error",
      title: "Workspace session required",
      body: "Sign in to an organization workspace before updating report review details."
    };
  }
  if (error === "comment-failed") {
    return {
      tone: "error",
      title: "Comment not added",
      body: "The report may no longer be published or available under the tenant retention policy."
    };
  }
  if (error === "schedule-failed") {
    return {
      tone: "error",
      title: "Schedule not saved",
      body: "Only report managers can schedule a report that is still published and available."
    };
  }
  if (error === "review-failed") {
    return {
      tone: "error",
      title: "Review not recorded",
      body: "Only report managers can approve or request changes on an available report."
    };
  }
  if (error === "approval-assignment-failed") {
    return {
      tone: "error",
      title: "Assignment not saved",
      body: "Only report managers can route approval for an available report."
    };
  }
  if (error === "evidence-package-failed") {
    return {
      tone: "error",
      title: "Evidence package not locked",
      body: "Evidence locking requires report-manager access, no pending approval assignments, and an approved latest review."
    };
  }
  return null;
}

export default async function ReportDetailPage({ params, searchParams }: ReportDetailPageProps) {
  const { reportId } = await params;
  const query = await searchParams;
  const { mode, reports, runs, sites } = await loadDashboardData();

  if (mode === "error") {
    return (
      <WorkspaceShell
        currentPath="/reports"
        title={`Report ${reportId}`}
        modeLabel="Workspace data unavailable"
        modeDescription="We could not load your organization's live data from the backend. Reload the page to try again."
        modeTone="error"
      >
        <div className="authNotice authError" role="alert">
          <strong>Workspace data unavailable</strong>
          <span>
            This report&apos;s data could not be loaded, so no sample or fabricated content is
            shown. Reload the page or try again shortly.
          </span>
        </div>
      </WorkspaceShell>
    );
  }

  const report = reports.find((item) => item.id === reportId);

  if (!report) {
    notFound();
  }

  const run = runs.find((item) => item.id === report.run_id) ?? null;
  const site = run ? sites.find((item) => item.id === run.site_id) ?? null : null;
  const session = await readAppSession();
  const sessionCanManageReports = ["owner", "manager"].includes((session.role ?? "").toLowerCase());
  let artifact: ReportArtifactResponse | null = null;
  let comments: ReportCommentSummary[] = [];
  let evidencePackages: ReportEvidencePackageSummary[] = [];
  let reportApprovers: ReportApproverSummary[] = [];
  let approvalAssignments: ReportApprovalAssignmentSummary[] = [];
  let reviews: ReportReviewSummary[] = [];
  let schedules: ReportScheduleSummary[] = [];
  let scheduleDeliveriesByScheduleId: Record<string, ReportScheduleDeliverySummary[]> = {};

  if (mode === "live" && session.userId && session.organizationId) {
    const liveUserId = session.userId;
    const liveOrganizationId = session.organizationId;
    const liveAuthToken = session.authToken;

    try {
      artifact = await getOrganizationReportArtifact(
        liveUserId,
        liveOrganizationId,
        report.id,
        liveAuthToken
      );
    } catch {
      artifact = null;
    }
    try {
      comments = await getOrganizationReportComments(
        liveUserId,
        liveOrganizationId,
        report.id,
        liveAuthToken
      );
    } catch {
      comments = [];
    }
    try {
      evidencePackages = await getOrganizationReportEvidencePackages(
        liveUserId,
        liveOrganizationId,
        report.id,
        liveAuthToken
      );
    } catch {
      evidencePackages = [];
    }
    try {
      reviews = await getOrganizationReportReviews(
        liveUserId,
        liveOrganizationId,
        report.id,
        liveAuthToken
      );
    } catch {
      reviews = [];
    }
    try {
      approvalAssignments = await getOrganizationReportApprovalAssignments(
        liveUserId,
        liveOrganizationId,
        report.id,
        liveAuthToken
      );
    } catch {
      approvalAssignments = [];
    }
    if (sessionCanManageReports) {
      try {
        reportApprovers = await getOrganizationReportApprovers(
          liveUserId,
          liveOrganizationId,
          liveAuthToken
        );
      } catch {
        reportApprovers = [];
      }
    }
    try {
      schedules = await getOrganizationReportSchedules(
        liveUserId,
        liveOrganizationId,
        report.id,
        liveAuthToken
      );
      const deliveryEntries = await Promise.all(
        schedules.map(
          async (schedule): Promise<[string, ReportScheduleDeliverySummary[]]> => {
            try {
              const deliveries = await getOrganizationReportScheduleDeliveries(
                liveUserId,
                liveOrganizationId,
                report.id,
                schedule.id,
                liveAuthToken
              );
              return [schedule.id, deliveries];
            } catch {
              return [schedule.id, []];
            }
          }
        )
      );
      scheduleDeliveriesByScheduleId = Object.fromEntries(deliveryEntries);
    } catch {
      schedules = [];
      scheduleDeliveriesByScheduleId = {};
    }
  } else if (mode === "demo") {
    comments = [
      {
        id: "comment_1",
        organization_id: report.organization_id,
        report_id: report.id,
        author_user_id: "user_1",
        body: "Ready for management review after SEU totals are confirmed.",
        created_at: "2026-07-20T09:00:00+00:00"
      }
    ];
    evidencePackages = [
      {
        id: "report_evidence_package_1",
        organization_id: report.organization_id,
        report_id: report.id,
        created_by_user_id: "user_1",
        filename: "report_1-iso_summary-evidence-package.json",
        content_type: "application/json",
        content_length: 2400,
        package_type: "iso50001_report_evidence",
        format: "json",
        comment_count: 1,
        review_count: 1,
        schedule_count: 1,
        delivery_count: 0,
        upload_count: 1,
        created_at: "2026-07-20T10:15:00+00:00"
      }
    ];
    reviews = [
      {
        id: "review_1",
        organization_id: report.organization_id,
        report_id: report.id,
        reviewer_user_id: "user_1",
        status: "approved",
        notes: "Approved for management evidence pack.",
        created_at: "2026-07-20T10:00:00+00:00"
      }
    ];
    reportApprovers = [
      {
        user_id: "user_1",
        organization_id: report.organization_id,
        role: "manager",
        email: "energy.manager@example.com",
        display_name: "Energy Manager"
      }
    ];
    approvalAssignments = [
      {
        id: "approval_1",
        organization_id: report.organization_id,
        report_id: report.id,
        assigned_to_user_id: "user_1",
        assigned_by_user_id: "owner_1",
        status: "approved",
        due_at: "2026-07-22T09:00:00+00:00",
        notes: "Approved for management evidence pack.",
        completed_at: "2026-07-20T10:00:00+00:00",
        created_at: "2026-07-19T09:00:00+00:00",
        updated_at: "2026-07-20T10:00:00+00:00"
      }
    ];
    schedules = [
      {
        id: "schedule_1",
        organization_id: report.organization_id,
        report_id: report.id,
        created_by_user_id: "user_1",
        frequency: "monthly",
        recipient_email: "energy.manager@example.com",
        next_run_at: "2026-08-01T09:00:00+00:00",
        is_active: true,
        last_delivery_at: "2026-07-20T09:05:00+00:00",
        last_delivery_status: "sent",
        last_delivery_error: null,
        created_at: "2026-07-20T09:00:00+00:00",
        updated_at: "2026-07-20T09:00:00+00:00"
      }
    ];
    scheduleDeliveriesByScheduleId = {
      schedule_1: [
        {
          id: "delivery_1",
          organization_id: report.organization_id,
          schedule_id: "schedule_1",
          report_id: report.id,
          recipient_email: "energy.manager@example.com",
          status: "sent",
          error_message: null,
          attempted_at: "2026-07-20T09:05:00+00:00",
          sent_at: "2026-07-20T09:05:00+00:00",
          created_at: "2026-07-20T09:05:00+00:00"
        }
      ]
    };
  }

  const pageMessage = reportMessage(query.status, query.error);
  const commentCountLabel =
    comments.length === 1 ? "1 review note" : `${comments.length} review notes`;
  const reviewCountLabel =
    reviews.length === 1 ? "1 review decision" : `${reviews.length} review decisions`;
  const approvalAssignmentCountLabel =
    approvalAssignments.length === 1
      ? "1 approval assignment"
      : `${approvalAssignments.length} approval assignments`;
  const scheduleCountLabel =
    schedules.length === 1 ? "1 delivery schedule" : `${schedules.length} delivery schedules`;
  const evidencePackageCountLabel =
    evidencePackages.length === 1
      ? "1 archived package"
      : `${evidencePackages.length} archived packages`;
  const canManageReportSchedules = mode === "live" && sessionCanManageReports;
  const canManageReportReviews = canManageReportSchedules;
  const canManageEvidencePackages = canManageReportSchedules;
  const currentReviewStatus = reviews[0]?.status ? formatReviewStatus(reviews[0].status) : "Pending review";
  const pendingApprovalAssignmentCount = approvalAssignments.filter(
    (assignment) => assignment.status === "pending"
  ).length;
  const canLockApprovedEvidencePackage =
    canManageEvidencePackages &&
    pendingApprovalAssignmentCount === 0 &&
    reviews[0]?.status === "approved";
  const artifactContent =
    artifact?.content ??
    (mode === "demo"
      ? {
          client: report.organization_id,
          site: site?.name ?? run?.site_id ?? "Main Site",
          total_records: 4,
          seu_total_consumption: 370,
          source_package: {
            upload_count: 4,
            categories: ["energy", "hdd", "cdd", "seu_mapping"],
            analysis_context: {
              execution_input: "legacy_energy_export"
            }
          },
          legacy_baseline: {
            energy_record_count: 60,
            meter_count: 3,
            mapped_meter_count: 2,
            seu_categories: ["Boiler", "Lighting"],
            weather_coverage: {
              hdd: { record_count: 24 },
              cdd: { record_count: 24 }
            },
            weather_normalization: {
              status: "modeled",
              baseline_year: 2024,
              comparison_year: 2025,
              model_count: 2,
              skipped_meter_count: 1,
              estimated_savings: 180,
              models: [
                {
                  meter: "Boiler gas",
                  utility: "gas",
                  seu_category: "Boiler",
                  normalization: "hdd",
                  baseline_record_count: 12,
                  comparison_record_count: 12,
                  actual_consumption: 1656,
                  predicted_consumption: 1776,
                  estimated_savings: 120,
                  r2: 1
                },
                {
                  meter: "Office power",
                  utility: "electricity",
                  seu_category: "Lighting",
                  normalization: "cdd",
                  baseline_record_count: 12,
                  comparison_record_count: 12,
                  actual_consumption: 1740,
                  predicted_consumption: 1800,
                  estimated_savings: 60,
                  r2: 1
                }
              ],
              skipped_meters: [
                {
                  meter: "Backup generator",
                  utility: "diesel",
                  seu_category: "Unknown",
                  reason: "unsupported_utility"
                }
              ]
            }
          }
        }
      : null);
  const artifactEntries = Object.entries(artifactContent ?? {});

  async function createReportCommentAction(formData: FormData) {
    "use server";

    const commentReportId = String(formData.get("report_id") ?? "").trim();
    const body = String(formData.get("body") ?? "").trim();
    const session = await readAppSession();

    if (!commentReportId || !body) {
      redirect(`/reports/${reportId}?error=missing-comment`);
    }
    if (!session.userId || !session.organizationId) {
      redirect(`/reports/${commentReportId}?error=session-missing`);
    }

    try {
      await createOrganizationReportComment(
        session.userId,
        session.organizationId,
        commentReportId,
        { body },
        session.authToken
      );
    } catch {
      redirect(`/reports/${commentReportId}?error=comment-failed`);
    }

    revalidatePath(`/reports/${commentReportId}`);
    redirect(`/reports/${commentReportId}?status=comment-created`);
  }

  async function createReportReviewAction(formData: FormData) {
    "use server";

    const reviewReportId = String(formData.get("report_id") ?? "").trim();
    const reviewStatus = String(formData.get("status") ?? "").trim();
    const notes = String(formData.get("notes") ?? "").trim();
    const session = await readAppSession();
    const allowedStatuses = reviewStatusOptions.map((option) => option.value);

    if (!reviewReportId || !allowedStatuses.includes(reviewStatus as ReportReviewStatus)) {
      redirect(`/reports/${reportId}?error=missing-review-status`);
    }
    if (!session.userId || !session.organizationId) {
      redirect(`/reports/${reviewReportId}?error=session-missing`);
    }

    try {
      await createOrganizationReportReview(
        session.userId,
        session.organizationId,
        reviewReportId,
        {
          status: reviewStatus as ReportReviewStatus,
          notes: notes || null
        },
        session.authToken
      );
    } catch {
      redirect(`/reports/${reviewReportId}?error=review-failed`);
    }

    revalidatePath(`/reports/${reviewReportId}`);
    redirect(`/reports/${reviewReportId}?status=review-recorded`);
  }

  async function createApprovalAssignmentAction(formData: FormData) {
    "use server";

    const assignmentReportId = String(formData.get("report_id") ?? "").trim();
    const assignedToUserId = String(formData.get("assigned_to_user_id") ?? "").trim();
    const dueAt = String(formData.get("due_at") ?? "").trim();
    const notes = String(formData.get("notes") ?? "").trim();
    const session = await readAppSession();

    if (!assignmentReportId || !assignedToUserId) {
      redirect(`/reports/${reportId}?error=missing-approval-assignment`);
    }
    if (!session.userId || !session.organizationId) {
      redirect(`/reports/${assignmentReportId}?error=session-missing`);
    }

    try {
      await createOrganizationReportApprovalAssignment(
        session.userId,
        session.organizationId,
        assignmentReportId,
        {
          assigned_to_user_id: assignedToUserId,
          due_at: dueAt || null,
          notes: notes || null
        },
        session.authToken
      );
    } catch {
      redirect(`/reports/${assignmentReportId}?error=approval-assignment-failed`);
    }

    revalidatePath(`/reports/${assignmentReportId}`);
    redirect(`/reports/${assignmentReportId}?status=approval-assigned`);
  }

  async function updateApprovalAssignmentAction(formData: FormData) {
    "use server";

    const assignmentReportId = String(formData.get("report_id") ?? "").trim();
    const assignmentId = String(formData.get("assignment_id") ?? "").trim();
    const assignmentStatus = String(formData.get("status") ?? "").trim();
    const notes = String(formData.get("notes") ?? "").trim();
    const session = await readAppSession();
    const allowedStatuses = approvalAssignmentStatusOptions.map((option) => option.value);

    if (
      !assignmentReportId ||
      !assignmentId ||
      !allowedStatuses.includes(assignmentStatus as Exclude<ReportApprovalAssignmentStatus, "pending">)
    ) {
      redirect(`/reports/${reportId}?error=approval-assignment-failed`);
    }
    if (!session.userId || !session.organizationId) {
      redirect(`/reports/${assignmentReportId}?error=session-missing`);
    }

    try {
      await updateOrganizationReportApprovalAssignment(
        session.userId,
        session.organizationId,
        assignmentReportId,
        assignmentId,
        {
          status: assignmentStatus as Exclude<ReportApprovalAssignmentStatus, "pending">,
          notes: notes || null
        },
        session.authToken
      );
    } catch {
      redirect(`/reports/${assignmentReportId}?error=approval-assignment-failed`);
    }

    revalidatePath(`/reports/${assignmentReportId}`);
    redirect(`/reports/${assignmentReportId}?status=approval-updated`);
  }

  async function createReportScheduleAction(formData: FormData) {
    "use server";

    const scheduleReportId = String(formData.get("report_id") ?? "").trim();
    const frequency = String(formData.get("frequency") ?? "").trim();
    const recipientEmail = String(formData.get("recipient_email") ?? "").trim();
    const nextRunAt = String(formData.get("next_run_at") ?? "").trim();
    const session = await readAppSession();
    const allowedFrequencies = scheduleFrequencyOptions.map((option) => option.value);

    if (
      !scheduleReportId ||
      !recipientEmail ||
      !nextRunAt ||
      !allowedFrequencies.includes(frequency as ReportScheduleFrequency)
    ) {
      redirect(`/reports/${reportId}?error=missing-schedule-fields`);
    }
    if (!session.userId || !session.organizationId) {
      redirect(`/reports/${scheduleReportId}?error=session-missing`);
    }

    try {
      await createOrganizationReportSchedule(
        session.userId,
        session.organizationId,
        scheduleReportId,
        {
          frequency: frequency as ReportScheduleFrequency,
          recipient_email: recipientEmail,
          next_run_at: nextRunAt
        },
        session.authToken
      );
    } catch {
      redirect(`/reports/${scheduleReportId}?error=schedule-failed`);
    }

    revalidatePath(`/reports/${scheduleReportId}`);
    redirect(`/reports/${scheduleReportId}?status=schedule-created`);
  }

  async function updateReportScheduleAction(formData: FormData) {
    "use server";

    const scheduleReportId = String(formData.get("report_id") ?? "").trim();
    const scheduleId = String(formData.get("schedule_id") ?? "").trim();
    const isActive = String(formData.get("is_active") ?? "true") === "true";
    const session = await readAppSession();

    if (!scheduleReportId || !scheduleId) {
      redirect(`/reports/${reportId}?error=schedule-failed`);
    }
    if (!session.userId || !session.organizationId) {
      redirect(`/reports/${scheduleReportId}?error=session-missing`);
    }

    try {
      await updateOrganizationReportSchedule(
        session.userId,
        session.organizationId,
        scheduleReportId,
        scheduleId,
        { is_active: isActive },
        session.authToken
      );
    } catch {
      redirect(`/reports/${scheduleReportId}?error=schedule-failed`);
    }

    revalidatePath(`/reports/${scheduleReportId}`);
    redirect(`/reports/${scheduleReportId}?status=schedule-updated`);
  }

  async function createEvidencePackageAction(formData: FormData) {
    "use server";

    const evidenceReportId = String(formData.get("report_id") ?? "").trim();
    const session = await readAppSession();

    if (!evidenceReportId) {
      redirect(`/reports/${reportId}?error=evidence-package-failed`);
    }
    if (!session.userId || !session.organizationId) {
      redirect(`/reports/${evidenceReportId}?error=session-missing`);
    }

    try {
      await createOrganizationReportEvidencePackage(
        session.userId,
        session.organizationId,
        evidenceReportId,
        session.authToken
      );
    } catch {
      redirect(`/reports/${evidenceReportId}?error=evidence-package-failed`);
    }

    revalidatePath(`/reports/${evidenceReportId}`);
    redirect(`/reports/${evidenceReportId}?status=evidence-package-created`);
  }

  return (
    <WorkspaceShell
      currentPath="/reports"
      title={`Report ${report.id}`}
      modeLabel={mode === "live" ? "Live workspace" : "Sample workspace"}
      modeDescription={
        mode === "live"
          ? "Report metadata and lineage are loading from the backend."
          : "This report is shown from sample data until a live organization is selected."
      }
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

      <section className="summaryGrid" aria-label="Report summary">
        <div className="summaryTile">
          <span>Status</span>
          <strong>{report.is_published ? "Published" : "Draft"}</strong>
        </div>
        <div className="summaryTile">
          <span>Review</span>
          <strong>{currentReviewStatus}</strong>
        </div>
        <div className="summaryTile">
          <span>Report type</span>
          <strong>{report.report_type}</strong>
        </div>
        <div className="summaryTile">
          <span>Run</span>
          <strong>{report.run_id}</strong>
        </div>
        <div className="summaryTile">
          <span>Site</span>
          <strong>{site?.name ?? run?.site_id ?? "Unknown"}</strong>
        </div>
      </section>

      <section className="contentGrid">
        <div className="chartPanel wide">
          <div className="sectionHeader">
            <h2>Report evidence summary</h2>
            <span>ISO 50001 indicators from the generated artifact</span>
          </div>
          <ReportArtifactSummary content={artifactContent} />
        </div>

        <div className="listPanel wide">
          <div className="sectionHeader">
            <h2>Storage and metadata</h2>
            <span>Canonical identifiers for this report</span>
          </div>
          <div className="rowList">
            <div className="dataRow">
              <div>
                <strong>Report ID</strong>
                <span>{report.id}</span>
              </div>
              <div>
                <strong>Organization</strong>
                <span>{report.organization_id}</span>
              </div>
            </div>
            <div className="dataRow">
              <div>
                <strong>Artifact</strong>
                <span>Available through authenticated report exports</span>
              </div>
              <div>
                <strong>Run link</strong>
                <span>{report.run_id}</span>
              </div>
            </div>
          </div>
        </div>

        <div className="listPanel wide">
          <div className="sectionHeader">
            <h2>ISO summary artifact</h2>
            <span>Generated report content</span>
          </div>
          <div className="rowList">
            {artifactEntries.length === 0 ? (
              <div className="dataRow">
                <div>
                  <strong>Artifact not available</strong>
                  <span>Run analysis in a live workspace to generate stored ISO summary content.</span>
                </div>
              </div>
            ) : (
              artifactEntries.map(([key, value]) => (
                <div className="dataRow" key={key}>
                  <div>
                    <strong>{key.replace(/_/g, " ")}</strong>
                    <span>{formatArtifactValue(value)}</span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="listPanel wide">
          <div className="sectionHeader">
            <h2>Management review</h2>
            <span>
              {approvalAssignmentCountLabel} - {reviewCountLabel}
            </span>
          </div>
          {canManageReportReviews && reportApprovers.length > 0 ? (
            <form action={createApprovalAssignmentAction} className="commentForm">
              <input type="hidden" name="report_id" value={report.id} />
              <label className="authField">
                <span>Approver</span>
                <select name="assigned_to_user_id">
                  {reportApprovers.map((approver) => (
                    <option value={approver.user_id} key={approver.user_id}>
                      {approver.display_name} ({approver.role})
                    </option>
                  ))}
                </select>
              </label>
              <label className="authField">
                <span>Due</span>
                <input name="due_at" type="datetime-local" />
              </label>
              <label className="authField">
                <span>Assignment note</span>
                <textarea name="notes" rows={2} placeholder="Optional approval routing note" />
              </label>
              <div className="toolbarRow">
                <button type="submit" className="btn btnGhost btnSm">
                  Assign approval
                </button>
              </div>
            </form>
          ) : canManageReportReviews ? (
            <div className="emptyState">
              <strong>No approvers available</strong>
              <span>Owner and manager approvers will appear here.</span>
            </div>
          ) : null}
          {canManageReportReviews ? (
            <form action={createReportReviewAction} className="commentForm">
              <input type="hidden" name="report_id" value={report.id} />
              <label className="authField">
                <span>Decision</span>
                <select name="status" defaultValue="approved">
                  {reviewStatusOptions.map((option) => (
                    <option value={option.value} key={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="authField">
                <span>Review note</span>
                <textarea name="notes" rows={3} placeholder="Optional management review note" />
              </label>
              <div className="toolbarRow">
                <button type="submit" className="btn btnPrimary btnSm">
                  Save review
                </button>
              </div>
            </form>
          ) : mode === "live" ? (
            <div className="emptyState">
              <strong>Review decisions unavailable</strong>
              <span>Owners and managers can approve reports or request changes.</span>
            </div>
          ) : null}
          <div className="rowList">
            {approvalAssignments.length === 0 ? (
              <div className="emptyState">
                <strong>No approval assignments yet</strong>
                <span>Routed report approvals will appear here.</span>
              </div>
            ) : (
              approvalAssignments.map((assignment) => (
                <div className="dataRow" key={assignment.id}>
                  <div>
                    <strong>{formatApprovalAssignmentStatus(assignment.status)}</strong>
                    <span>{formatApprovalAssignee(assignment, reportApprovers)}</span>
                    <span>{assignment.notes ?? "No assignment note recorded"}</span>
                  </div>
                  <div>
                    <strong>
                      {assignment.due_at ? formatCommentTime(assignment.due_at) : "No due date"}
                    </strong>
                    {canManageReportReviews && assignment.status === "pending" ? (
                      <form action={updateApprovalAssignmentAction} className="memberActionForm">
                        <input type="hidden" name="report_id" value={report.id} />
                        <input type="hidden" name="assignment_id" value={assignment.id} />
                        <select name="status" defaultValue="approved">
                          {approvalAssignmentStatusOptions.map((option) => (
                            <option value={option.value} key={option.value}>
                              {option.label}
                            </option>
                          ))}
                        </select>
                        <input name="notes" placeholder="Decision note" />
                        <button type="submit" className="btn btnPrimary btnSm">
                          Update
                        </button>
                      </form>
                    ) : (
                      <span>{assignment.completed_at ?? assignment.assigned_by_user_id}</span>
                    )}
                  </div>
                </div>
              ))
            )}
            {reviews.length === 0 ? (
              <div className="emptyState">
                <strong>No management review yet</strong>
                <span>Report approval decisions will appear here.</span>
              </div>
            ) : (
              reviews.map((review) => (
                <div className="dataRow" key={review.id}>
                  <div>
                    <strong>{formatReviewStatus(review.status)}</strong>
                    <span>{review.notes ?? "No note recorded"}</span>
                  </div>
                  <div>
                    <strong>{formatCommentTime(review.created_at)}</strong>
                    <span>{review.reviewer_user_id}</span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="listPanel wide">
          <div className="sectionHeader">
            <h2>Collaboration</h2>
            <span>{commentCountLabel}</span>
          </div>
          {mode === "live" ? (
            <form action={createReportCommentAction} className="commentForm">
              <input type="hidden" name="report_id" value={report.id} />
              <label className="authField">
                <span>Review note</span>
                <textarea
                  name="body"
                  rows={4}
                  placeholder="Add a report review note for this organization"
                />
              </label>
              <div className="toolbarRow">
                <button type="submit" className="btn btnPrimary btnSm">
                  Add comment
                </button>
              </div>
            </form>
          ) : null}
          <div className="rowList">
            {comments.length === 0 ? (
              <div className="emptyState">
                <strong>No review notes yet</strong>
                <span>Management, evidence, and follow-up notes will appear here.</span>
              </div>
            ) : (
              comments.map((comment) => (
                <div className="dataRow" key={comment.id}>
                  <div>
                    <strong>{comment.author_user_id}</strong>
                    <span>{comment.body}</span>
                  </div>
                  <div>
                    <strong>{formatCommentTime(comment.created_at)}</strong>
                    <span>{comment.report_id}</span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="listPanel wide">
          <div className="sectionHeader">
            <h2>Evidence packages</h2>
            <span>{evidencePackageCountLabel}</span>
          </div>
          <div className="toolbarRow">
            {mode === "live" ? (
              <>
                <a href={`/reports/${report.id}/evidence-package`} className="btn btnGhost btnSm">
                  Download current
                </a>
                <a href={`/reports/${report.id}/evidence-package/pdf`} className="btn btnGhost btnSm">
                  Download PDF
                </a>
                {canManageEvidencePackages ? (
                  <form action={createEvidencePackageAction} className="memberActionForm">
                    <input type="hidden" name="report_id" value={report.id} />
                    <button
                      type="submit"
                      className="btn btnPrimary btnSm"
                      disabled={!canLockApprovedEvidencePackage}
                    >
                      {canLockApprovedEvidencePackage ? "Lock package" : "Approval required"}
                    </button>
                  </form>
                ) : null}
              </>
            ) : (
              <span>Live workspace</span>
            )}
          </div>
          {mode === "live" && !canManageEvidencePackages ? (
            <div className="emptyState">
              <strong>Package locking unavailable</strong>
              <span>Owners and managers can archive a point-in-time evidence package.</span>
            </div>
          ) : null}
          {mode === "live" && canManageEvidencePackages && !canLockApprovedEvidencePackage ? (
            <div className="emptyState">
              <strong>Approved review required</strong>
              <span>Complete pending approval assignments and approve the latest review before locking.</span>
            </div>
          ) : null}
          <div className="rowList">
            {evidencePackages.length === 0 ? (
              <div className="emptyState">
                <strong>No archived packages yet</strong>
                <span>Locked evidence packages will appear here.</span>
              </div>
            ) : (
              evidencePackages.map((packageSummary) => (
                <div className="dataRow" key={packageSummary.id}>
                  <div>
                    <strong>{packageSummary.filename}</strong>
                    <span>{formatEvidencePackageCounts(packageSummary)}</span>
                    <span>{formatFileSize(packageSummary.content_length)}</span>
                  </div>
                  <div>
                    <strong>{formatCommentTime(packageSummary.created_at)}</strong>
                    {mode === "live" ? (
                      <a
                        href={`/reports/${report.id}/evidence-packages/${packageSummary.id}/download`}
                        className="btn btnGhost btnSm"
                      >
                        Download
                      </a>
                    ) : (
                      <span>{packageSummary.created_by_user_id}</span>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="listPanel wide">
          <div className="sectionHeader">
            <h2>Scheduled delivery</h2>
            <span>{scheduleCountLabel}</span>
          </div>
          {canManageReportSchedules ? (
            <form action={createReportScheduleAction} className="scheduleForm">
              <input type="hidden" name="report_id" value={report.id} />
              <label className="authField">
                <span>Recipient</span>
                <input name="recipient_email" type="email" placeholder="energy.manager@example.com" />
              </label>
              <label className="authField">
                <span>Frequency</span>
                <select name="frequency" defaultValue="monthly">
                  {scheduleFrequencyOptions.map((option) => (
                    <option value={option.value} key={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
              <label className="authField">
                <span>Next delivery</span>
                <input name="next_run_at" type="datetime-local" />
              </label>
              <button type="submit" className="btn btnPrimary btnSm">
                Save schedule
              </button>
            </form>
          ) : mode === "live" ? (
            <div className="emptyState">
              <strong>Schedule management unavailable</strong>
              <span>Owners and managers can create or pause report delivery schedules.</span>
            </div>
          ) : null}
          <div className="rowList">
            {schedules.length === 0 ? (
              <div className="emptyState">
                <strong>No delivery schedules yet</strong>
                <span>Scheduled report deliveries will appear here.</span>
              </div>
            ) : (
              schedules.map((schedule) => {
                const recentDeliveries = (scheduleDeliveriesByScheduleId[schedule.id] ?? [])
                  .slice(-3)
                  .reverse();
                return (
                  <div className="dataRow" key={schedule.id}>
                    <div>
                      <strong>{schedule.recipient_email}</strong>
                      <span>
                        {formatScheduleFrequency(schedule.frequency)} -{" "}
                        {schedule.is_active ? "Active" : "Paused"}
                      </span>
                      <span>{formatScheduleDeliveryStatus(schedule)}</span>
                      {recentDeliveries.map((delivery) => (
                        <span key={delivery.id}>
                          {formatScheduleDeliveryAttempt(delivery)}
                          {delivery.status === "failed" && delivery.error_message
                            ? ` - ${delivery.error_message}`
                            : ""}
                        </span>
                      ))}
                    </div>
                    <div>
                      <strong>{formatCommentTime(schedule.next_run_at)}</strong>
                      {canManageReportSchedules && schedule.is_active ? (
                        <form action={updateReportScheduleAction} className="memberActionForm">
                          <input type="hidden" name="report_id" value={report.id} />
                          <input type="hidden" name="schedule_id" value={schedule.id} />
                          <input type="hidden" name="is_active" value="false" />
                          <button type="submit" className="btn btnGhost btnSm">
                            Pause
                          </button>
                        </form>
                      ) : (
                        <span>{schedule.created_by_user_id}</span>
                      )}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>

        <div className="listPanel">
          <div className="sectionHeader">
            <h2>Run lineage</h2>
            <span>Execution source for this report</span>
          </div>
          <div className="rowList">
            <div className="dataRow">
              <div>
                <strong>Run status</strong>
                <span>{run?.status ?? "Unknown"}</span>
              </div>
              <div>
                <strong>Completion</strong>
                <span>{run?.completed_at ?? run?.error_message ?? "Awaiting completion"}</span>
              </div>
            </div>
            <div className="dataRow">
              <div>
                <strong>Requested by</strong>
                <span>{run?.requested_by_user_id ?? "Unknown"}</span>
              </div>
              <div>
                <strong>Site ID</strong>
                <span>{run?.site_id ?? "Unknown"}</span>
              </div>
            </div>
          </div>
        </div>

        <div className="listPanel">
          <div className="sectionHeader">
            <h2>Navigation</h2>
            <span>Continue workflow</span>
          </div>
          <div className="rowList">
            <div className="dataRow">
              <div>
                <strong>Download report</strong>
                <span>Export the JSON report artifact</span>
              </div>
              <div>
                {mode === "live" ? (
                  <a href={`/reports/${report.id}/download`} className="btn btnGhost btnSm">
                    Download JSON
                  </a>
                ) : (
                  <span>Live workspace</span>
                )}
              </div>
            </div>
            <div className="dataRow">
              <div>
                <strong>Current evidence package</strong>
                <span>Export report artifact, lineage, reviews, comments, and schedules</span>
              </div>
              <div>
                {mode === "live" ? (
                  <div className="actionGroup">
                    <a href={`/reports/${report.id}/evidence-package`} className="btn btnGhost btnSm">
                      Download JSON
                    </a>
                    <a href={`/reports/${report.id}/evidence-package/pdf`} className="btn btnGhost btnSm">
                      Download PDF
                    </a>
                  </div>
                ) : (
                  <span>Live workspace</span>
                )}
              </div>
            </div>
            <div className="dataRow">
              <div>
                <strong>Back to reports</strong>
                <span>View the full report library</span>
              </div>
              <div>
                <Link href="/reports" className="btn btnGhost btnSm">
                  Open reports
                </Link>
              </div>
            </div>
            <div className="dataRow">
              <div>
                <strong>Open runs</strong>
                <span>Review run queue and statuses</span>
              </div>
              <div>
                <Link href="/runs" className="btn btnGhost btnSm">
                  Open runs
                </Link>
              </div>
            </div>
          </div>
        </div>
      </section>
    </WorkspaceShell>
  );
}
