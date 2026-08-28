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
  invite_id: string;
  email: string;
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

export type UploadCreatePayload = {
  upload_id: string;
  site_id: string;
  category: string;
  filename: string;
  checksum: string;
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

export type RunCreatePayload = {
  run_id: string;
  site_id: string;
  upload_ids: string[];
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
    throw new Error(`API request failed: ${response.status} ${response.statusText}`);
  }

  return response.json() as Promise<T>;
}

export function getMyOrganizations(
  userId: string,
  authToken?: string | null
): Promise<OrganizationSummary[]> {
  return apiFetch<OrganizationSummary[]>("/me/organizations", userId, undefined, authToken);
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
    throw new Error(`API request failed: ${response.status} ${response.statusText}`);
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
    throw new Error(`API request failed: ${response.status} ${response.statusText}`);
  }

  return response.json() as Promise<DevSessionResponse>;
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
    throw new Error(`API request failed: ${response.status} ${response.statusText}`);
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

export function getOrganizationRuns(
  userId: string,
  organizationId: string,
  siteId?: string,
  authToken?: string | null
): Promise<RunSummary[]> {
  const query = siteId ? `?site_id=${encodeURIComponent(siteId)}` : "";
  return apiFetch<RunSummary[]>(`/organizations/${organizationId}/runs${query}`, userId, undefined, authToken);
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
