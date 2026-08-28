import {
  getOrganizationAnalysisJobs,
  getOrganizationAuditEvents,
  getOrganizationMeters,
  getOrganizationNotifications,
  getOrganizationReports,
  getOrganizationRuns,
  getOrganizationSites,
  getOrganizationUploadReadiness,
  getOrganizationUploads,
  type AnalysisJobSummary,
  type AuditEventSummary,
  type MeterSummary,
  type NotificationSummary,
  type ReportSummary,
  type RunSummary,
  type SiteSummary,
  type UploadReadinessSummary,
  type UploadSummary
} from "./api";
import { redirect } from "next/navigation";
import { isClerkFrontendEnabled } from "./auth-config";
import { readAppSession } from "./session";

export type DashboardMode = "demo" | "live" | "error";

export type DashboardData = {
  mode: DashboardMode;
  sites: SiteSummary[];
  meters: MeterSummary[];
  uploads: UploadSummary[];
  uploadReadiness: UploadReadinessSummary;
  runs: RunSummary[];
  analysisJobs: AnalysisJobSummary[];
  notifications: NotificationSummary[];
  reports: ReportSummary[];
  auditEvents: AuditEventSummary[];
};

const demoData: DashboardData = {
  mode: "demo",
  sites: [
    {
      id: "site_1",
      organization_id: "org_1",
      name: "Main Site",
      timezone: "Europe/London"
    }
  ],
  meters: [
    {
      id: "meter_1",
      organization_id: "org_1",
      site_id: "site_1",
      display_name: "Main Electricity",
      commodity: "electricity",
      unit: "kWh",
      source_column: "Main Electricity",
      is_seu: true
    },
    {
      id: "meter_2",
      organization_id: "org_1",
      site_id: "site_1",
      display_name: "Main Gas",
      commodity: "gas",
      unit: "kWh",
      source_column: "Main Gas",
      is_seu: true
    }
  ],
  uploads: [
    {
      id: "upload_1",
      organization_id: "org_1",
      site_id: "site_1",
      uploaded_by_user_id: "user_1",
      category: "energy",
      checksum: "demo",
      status: "stored"
    }
  ],
  uploadReadiness: {
    organization_id: "org_1",
    ready_site_count: 0,
    site_count: 1,
    required_categories: ["energy", "hdd", "cdd", "seu_mapping"],
    sites: [
      {
        site_id: "site_1",
        site_name: "Main Site",
        ready_count: 1,
        required_count: 4,
        is_ready: false,
        files: [
          {
            category: "energy",
            label: "Energy data",
            is_required: true,
            is_ready: true,
            latest_upload: {
              id: "upload_1",
              organization_id: "org_1",
              site_id: "site_1",
              uploaded_by_user_id: "user_1",
              category: "energy",
              checksum: "demo",
              status: "stored"
            }
          },
          {
            category: "hdd",
            label: "Heating degree days",
            is_required: true,
            is_ready: false,
            latest_upload: null
          },
          {
            category: "cdd",
            label: "Cooling degree days",
            is_required: true,
            is_ready: false,
            latest_upload: null
          },
          {
            category: "seu_mapping",
            label: "SEU mapping",
            is_required: true,
            is_ready: false,
            latest_upload: null
          }
        ]
      }
    ]
  },
  runs: [
    {
      id: "run_1",
      organization_id: "org_1",
      site_id: "site_1",
      requested_by_user_id: "user_1",
      status: "succeeded",
      error_message: null,
      completed_at: null
    }
  ],
  analysisJobs: [
    {
      id: "job_run_1",
      organization_id: "org_1",
      run_id: "run_1",
      upload_id: "upload_1",
      report_id: "report_1",
      requested_by_user_id: "user_1",
      status: "succeeded",
      error_message: null,
      completed_at: null
    }
  ],
  notifications: [
    {
      id: "notification_run_1",
      organization_id: "org_1",
      recipient_user_id: "user_1",
      channel: "email",
      event_type: "run_succeeded",
      subject: "Energy analysis run_1 completed",
      body: "Run run_1 completed successfully. Report report_1 is ready.",
      resource_type: "run",
      resource_id: "run_1",
      status: "pending",
      error_message: null,
      sent_at: null,
      attempt_count: 0,
      last_attempt_at: null,
      next_attempt_at: null
    },
    {
      id: "notification_approval_1",
      organization_id: "org_1",
      recipient_user_id: "user_1",
      channel: "email",
      event_type: "report_approval_assigned",
      subject: "ISO 50001 report approval assigned: iso_summary",
      body: "A report approval has been assigned to you. Report report_1 is ready for review.",
      resource_type: "report_approval_assignment",
      resource_id: "approval_1",
      status: "pending",
      error_message: "SMTP timeout",
      sent_at: null,
      attempt_count: 1,
      last_attempt_at: "2026-07-20T12:00:00+00:00",
      next_attempt_at: "2026-07-20T12:05:00+00:00"
    }
  ],
  reports: [
    {
      id: "report_1",
      organization_id: "org_1",
      run_id: "run_1",
      report_type: "iso_summary",
      is_published: true,
      expires_at: "2026-10-18T00:00:00+00:00",
      retention_applied_at: null
    }
  ],
  auditEvents: [
    {
      id: "audit_1",
      organization_id: "org_1",
      created_at: "2026-07-20T09:30:00+00:00",
      actor_user_id: "user_1",
      action: "run_succeeded",
      resource_type: "run",
      resource_id: "run_1",
      metadata_json: "{\"status\":\"succeeded\"}"
    },
    {
      id: "audit_2",
      organization_id: "org_1",
      created_at: "2026-07-20T09:35:00+00:00",
      actor_user_id: "user_1",
      action: "report_created",
      resource_type: "report",
      resource_id: "report_1",
      metadata_json: "{\"report_type\":\"iso_summary\"}"
    }
  ]
};

function errorData(organizationId: string): DashboardData {
  return {
    mode: "error",
    sites: [],
    meters: [],
    uploads: [],
    uploadReadiness: {
      organization_id: organizationId,
      ready_site_count: 0,
      site_count: 0,
      required_categories: [],
      sites: []
    },
    runs: [],
    analysisJobs: [],
    notifications: [],
    reports: [],
    auditEvents: []
  };
}

function canViewAudit(role: string | null): boolean {
  const normalizedRole = role?.toLowerCase() ?? null;
  if (!normalizedRole) {
    return true;
  }
  return normalizedRole === "owner" || normalizedRole === "manager";
}

async function emptyOnDenied<T>(fetchData: () => Promise<T[]>): Promise<T[]> {
  try {
    return await fetchData();
  } catch {
    return [];
  }
}

export async function loadDashboardData(): Promise<DashboardData> {
  const session = await readAppSession();
  const userId = session.userId ?? process.env.DEMO_USER_ID;
  const organizationId = session.organizationId ?? process.env.DEMO_ORGANIZATION_ID;
  const authToken = session.authToken;
  const auditAllowed = canViewAudit(session.role ?? process.env.DEMO_ROLE ?? null);

  if (isClerkFrontendEnabled() && !session.userId) {
    redirect("/login");
  }

  if (session.userId && !session.organizationId) {
    redirect("/organizations");
  }

  if (!userId || !organizationId) {
    return demoData;
  }

  try {
    const [sites, meters, uploads, uploadReadiness, runs, analysisJobs, notifications, reports, auditEvents] =
      await Promise.all([
      getOrganizationSites(userId, organizationId, authToken),
      getOrganizationMeters(userId, organizationId, undefined, authToken),
      getOrganizationUploads(userId, organizationId, undefined, authToken),
      getOrganizationUploadReadiness(userId, organizationId, undefined, authToken),
      getOrganizationRuns(userId, organizationId, undefined, authToken),
      getOrganizationAnalysisJobs(userId, organizationId, authToken),
      getOrganizationNotifications(userId, organizationId, authToken),
      getOrganizationReports(userId, organizationId, undefined, authToken),
      auditAllowed
        ? emptyOnDenied(() => getOrganizationAuditEvents(userId, organizationId, authToken))
        : Promise.resolve([])
    ]);
    return {
      mode: "live",
      sites,
      meters,
      uploads,
      uploadReadiness,
      runs,
      analysisJobs,
      notifications,
      reports,
      auditEvents
    };
  } catch {
    if (session.userId && session.organizationId) {
      return errorData(session.organizationId);
    }
    return demoData;
  }
}
