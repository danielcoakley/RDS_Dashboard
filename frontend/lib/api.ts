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

export type UploadSummary = {
  id: string;
  organization_id: string;
  site_id: string;
  uploaded_by_user_id: string;
  category: string;
  storage_key: string;
  checksum: string;
  status: string;
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

export type AuditEventSummary = {
  id: string;
  organization_id: string;
  actor_user_id: string;
  action: string;
  resource_type: string;
  resource_id: string;
  metadata_json: string;
};

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const AUTH_TOKEN =
  process.env.NEXT_PUBLIC_DEMO_AUTH_TOKEN ?? process.env.DEMO_AUTH_TOKEN ?? null;

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

async function apiFetch<T>(path: string, userId: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      ...buildAuthHeaders(userId),
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

export function getOrganizationUploads(
  userId: string,
  organizationId: string,
  siteId?: string
): Promise<UploadSummary[]> {
  const query = siteId ? `?site_id=${encodeURIComponent(siteId)}` : "";
  return apiFetch<UploadSummary[]>(`/organizations/${organizationId}/uploads${query}`, userId);
}

export function getOrganizationRuns(
  userId: string,
  organizationId: string,
  siteId?: string
): Promise<RunSummary[]> {
  const query = siteId ? `?site_id=${encodeURIComponent(siteId)}` : "";
  return apiFetch<RunSummary[]>(`/organizations/${organizationId}/runs${query}`, userId);
}

export function getOrganizationAuditEvents(
  userId: string,
  organizationId: string
): Promise<AuditEventSummary[]> {
  return apiFetch<AuditEventSummary[]>(`/organizations/${organizationId}/audit-events`, userId);
}
