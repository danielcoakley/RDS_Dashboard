export type OrganizationSummary = {
  id: string;
  name: string;
  slug: string;
};

export type SiteSummary = {
  id: string;
  organization_id: string;
  name: string;
  timezone: string;
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

export type LocalAnalysisRunPayload = {
  site_id: string;
  upload_id: string;
  run_id: string;
  report_id: string;
  filename: string;
  rows: Array<Record<string, unknown>>;
  client_config: Record<string, unknown>;
};

export type LocalAnalysisRunResponse = {
  upload_id: string;
  run_id: string;
  report_id: string;
  run_status: string;
  report_storage_key: string;
  iso_summary: Record<string, unknown>;
};

export type ReportSummary = {
  id: string;
  organization_id: string;
  run_id: string;
  report_type: string;
  storage_key: string;
  is_published: boolean;
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

async function apiFetch<T>(path: string, userId: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      "X-User-Id": userId,
      ...(init?.headers ?? {})
    }
  });

  if (!response.ok) {
    throw new Error(`API request failed: ${response.status} ${response.statusText}`);
  }

  return response.json() as Promise<T>;
}

export function getMyOrganizations(userId: string): Promise<OrganizationSummary[]> {
  return apiFetch<OrganizationSummary[]>("/me/organizations", userId);
}

export function getOrganizationSites(
  userId: string,
  organizationId: string
): Promise<SiteSummary[]> {
  return apiFetch<SiteSummary[]>(`/organizations/${organizationId}/sites`, userId);
}

export function getOrganizationMeters(
  userId: string,
  organizationId: string,
  siteId?: string
): Promise<MeterSummary[]> {
  const query = siteId ? `?site_id=${encodeURIComponent(siteId)}` : "";
  return apiFetch<MeterSummary[]>(`/organizations/${organizationId}/meters${query}`, userId);
}

export function executeLocalAnalysisRun(
  userId: string,
  organizationId: string,
  payload: LocalAnalysisRunPayload
): Promise<LocalAnalysisRunResponse> {
  return apiFetch<LocalAnalysisRunResponse>(
    `/organizations/${organizationId}/runs/execute-local`,
    userId,
    {
      method: "POST",
      body: JSON.stringify(payload)
    }
  );
}

export function getOrganizationReports(
  userId: string,
  organizationId: string,
  runId?: string
): Promise<ReportSummary[]> {
  const query = runId ? `?run_id=${encodeURIComponent(runId)}` : "";
  return apiFetch<ReportSummary[]>(`/organizations/${organizationId}/reports${query}`, userId);
}
