import {
  getOrganizationMeters,
  getOrganizationReports,
  getOrganizationRuns,
  getOrganizationSites,
  getOrganizationUploads,
  type MeterSummary,
  type ReportSummary,
  type RunSummary,
  type SiteSummary,
  type UploadSummary
} from "./api";

type DashboardData = {
  sites: SiteSummary[];
  meters: MeterSummary[];
  uploads: UploadSummary[];
  runs: RunSummary[];
  reports: ReportSummary[];
};

const demoData: DashboardData = {
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
      storage_key: "tenants/org_1/sites/site_1/uploads/upload_1/energy.csv",
      checksum: "demo",
      status: "stored"
    }
  ],
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
  reports: [
    {
      id: "report_1",
      organization_id: "org_1",
      run_id: "run_1",
      report_type: "iso_summary",
      storage_key: "tenants/org_1/sites/site_1/runs/run_1/reports/iso-summary.json",
      is_published: true
    }
  ]
};

export async function loadDashboardData(): Promise<DashboardData> {
  const userId = process.env.DEMO_USER_ID;
  const organizationId = process.env.DEMO_ORGANIZATION_ID;

  if (!userId || !organizationId) {
    return demoData;
  }

  try {
    const [sites, meters, uploads, runs, reports] = await Promise.all([
      getOrganizationSites(userId, organizationId),
      getOrganizationMeters(userId, organizationId),
      getOrganizationUploads(userId, organizationId),
      getOrganizationRuns(userId, organizationId),
      getOrganizationReports(userId, organizationId)
    ]);
    return { sites, meters, uploads, runs, reports };
  } catch {
    return demoData;
  }
}
