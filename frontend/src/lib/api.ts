const API_URL = process.env.NEXT_PUBLIC_API_URL || '';

function getToken(): string | null {
  if (typeof window === 'undefined') return null;
  return localStorage.getItem('enms_token');
}

export function setToken(token: string) {
  if (typeof window !== 'undefined') {
    localStorage.setItem('enms_token', token);
  }
}

export function clearToken() {
  if (typeof window !== 'undefined') {
    localStorage.removeItem('enms_token');
    localStorage.removeItem('enms_user');
  }
}

export function getUser() {
  if (typeof window === 'undefined') return null;
  const raw = localStorage.getItem('enms_user');
  return raw ? JSON.parse(raw) : null;
}

export function setUser(user: any) {
  if (typeof window !== 'undefined') {
    localStorage.setItem('enms_user', JSON.stringify(user));
  }
}

async function rawRequest(path: string, options: RequestInit = {}): Promise<Response> {
  const token = getToken();
  const isFormData = typeof FormData !== 'undefined' && options.body instanceof FormData;
  const headers: Record<string, string> = {
    ...(!isFormData && { 'Content-Type': 'application/json' }),
    ...(options.headers as Record<string, string>),
  };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const res = await fetch(`${API_URL}${path}`, { ...options, headers });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail.detail || 'Request failed');
  }
  return res;
}

async function request<T = any>(path: string, options: RequestInit = {}): Promise<T> {
  return (await rawRequest(path, options)).json();
}

export const api = {
  // Auth
  signup: (data: any) => request('/api/auth/signup', { method: 'POST', body: JSON.stringify(data) }),
  login: (data: any) => request('/api/auth/login', { method: 'POST', body: JSON.stringify(data) }),
  demo: () => request('/api/auth/demo', { method: 'POST' }),
  me: () => request('/api/auth/me'),

  // Sites
  listSites: () => request('/api/sites'),
  createSite: (data: any) => request('/api/sites', { method: 'POST', body: JSON.stringify(data) }),
  getSite: (id: number) => request(`/api/sites/${id}`),
  deleteSite: (id: number) => request(`/api/sites/${id}`, { method: 'DELETE' }),

  // Meters
  listMeters: (siteId: number) => request(`/api/sites/${siteId}/meters`),
  createMeter: (siteId: number, data: any) => request(`/api/sites/${siteId}/meters`, { method: 'POST', body: JSON.stringify(data) }),
  deleteMeter: (siteId: number, meterId: number) => request(`/api/sites/${siteId}/meters/${meterId}`, { method: 'DELETE' }),

  // Data
  uploadData: (siteId: number, file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return request(`/api/data/upload/${siteId}`, { method: 'POST', body: formData });
  },
  refreshWeather: (siteId: number) => request(`/api/data/weather/${siteId}/fetch`, { method: 'POST' }),
  weatherStatus: (siteId: number) => request(`/api/data/weather/${siteId}/status`),
  uploadSeuMapping: (siteId: number, file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    return request(`/api/data/seu-mapping/${siteId}`, { method: 'POST', body: formData });
  },
  listUploads: (siteId: number) => request(`/api/data/uploads/${siteId}`),
  dataSummary: (siteId: number) => request(`/api/data/summary/${siteId}`),

  // Onboarding / smart upload
  getOrgProfile: () => request('/api/sites/org-profile'),
  updateOrgProfile: (data: any) => request('/api/sites/org-profile', { method: 'PUT', body: JSON.stringify(data) }),
  geocode: (address: string) => request('/api/sites/geocode', { method: 'POST', body: JSON.stringify({ address }) }),
  previewUpload: (siteId: number, file: File, mapping: Record<string, string> = {}) => {
    const formData = new FormData();
    formData.append('file', file);
    Object.entries(mapping).forEach(([k, v]) => v != null && formData.append(k, v));
    return request(`/api/data/preview/${siteId}`, { method: 'POST', body: formData });
  },
  loadSampleData: (siteId: number) => request(`/api/data/sample/${siteId}`, { method: 'POST' }),
  templateUrl: `${API_URL}/api/data/template`,

  // Analytics
  runAnalysis: (data: any) => request('/api/analytics/analysis', { method: 'POST', body: JSON.stringify(data) }),
  analysisBundle: (siteId: number, baselineYear: number, comparisonYear: number) =>
    request(`/api/analytics/bundle/${siteId}?baseline_year=${baselineYear}&comparison_year=${comparisonYear}`),
  availableYears: (siteId: number) => request(`/api/analytics/years/${siteId}`),
  monthlyComparison: (siteId: number, baselineYear: number, comparisonYear: number) =>
    request(`/api/analytics/monthly/${siteId}?baseline_year=${baselineYear}&comparison_year=${comparisonYear}`),
  sankey: (siteId: number, baselineYear: number, comparisonYear: number) =>
    request(`/api/analytics/sankey/${siteId}?baseline_year=${baselineYear}&comparison_year=${comparisonYear}`),
  seuSummary: (siteId: number, baselineYear: number, comparisonYear: number) =>
    request(`/api/analytics/seu-summary/${siteId}?baseline_year=${baselineYear}&comparison_year=${comparisonYear}`),
  seuMonthly: (siteId: number, baselineYear: number, comparisonYear: number) =>
    request(`/api/analytics/seu-monthly/${siteId}?baseline_year=${baselineYear}&comparison_year=${comparisonYear}`),

  // Objectives
  listObjectives: () => request('/api/objectives'),
  createObjective: (data: any) => request('/api/objectives', { method: 'POST', body: JSON.stringify(data) }),
  updateObjective: (id: number, data: any) => request(`/api/objectives/${id}`, { method: 'PUT', body: JSON.stringify(data) }),
  deleteObjective: (id: number) => request(`/api/objectives/${id}`, { method: 'DELETE' }),

  // Compliance
  listCompliance: () => request('/api/compliance'),
  updateCompliance: (clauseRef: string, data: any) => request(`/api/compliance/${clauseRef}`, { method: 'PUT', body: JSON.stringify(data) }),
  complianceScore: () => request('/api/compliance/score'),
  complianceReadiness: () => request<ComplianceReadiness>('/api/compliance/readiness'),

  // Energy Review
  listReviews: (siteId: number) => request(`/api/energy-review/${siteId}`),
  createReview: (data: any) => request('/api/energy-review', { method: 'POST', body: JSON.stringify(data) }),
  updateReview: (id: number, data: any) => request(`/api/energy-review/${id}`, { method: 'PUT', body: JSON.stringify(data) }),
  deleteReview: (id: number) => request(`/api/energy-review/${id}`, { method: 'DELETE' }),

  // Reports & internal audit
  managementReview: (siteId: number, baselineYear: number, comparisonYear: number, elecPrice: number, gasPrice: number) =>
    request(`/api/reports/management-review/${siteId}?baseline_year=${baselineYear}&comparison_year=${comparisonYear}&elec_price=${elecPrice}&gas_price=${gasPrice}`),
  listFindings: () => request('/api/reports/audit-findings'),
  createFinding: (data: any) => request('/api/reports/audit-findings', { method: 'POST', body: JSON.stringify(data) }),
  updateFinding: (id: number, data: any) => request(`/api/reports/audit-findings/${id}`, { method: 'PUT', body: JSON.stringify(data) }),
  deleteFinding: (id: number) => request(`/api/reports/audit-findings/${id}`, { method: 'DELETE' }),

  // Team & organization
  getTeam: () => request('/api/team'),
  createInvite: (data: any) => request('/api/team/invites', { method: 'POST', body: JSON.stringify(data) }),
  revokeInvite: (id: number) => request(`/api/team/invites/${id}`, { method: 'DELETE' }),
  updateMember: (id: number, data: any) => request(`/api/team/members/${id}`, { method: 'PATCH', body: JSON.stringify(data) }),
  removeMember: (id: number) => request(`/api/team/members/${id}`, { method: 'DELETE' }),
  getOrganization: () => request('/api/team/organization'),
  updateOrganization: (data: any) => request('/api/team/organization', { method: 'PUT', body: JSON.stringify(data) }),
  getInvite: (token: string) => request(`/api/team/invites/accept/${token}`),
  acceptInvite: (token: string, data: any) => request(`/api/team/invites/accept/${token}`, { method: 'POST', body: JSON.stringify(data) }),

  // Audit log
  listAudit: (params: Record<string, string>) => request(`/api/audit?${new URLSearchParams(params)}`),
  auditActions: () => request('/api/audit/actions'),
  exportAudit: async (params: Record<string, string>) => (await rawRequest(`/api/audit/export?${new URLSearchParams(params)}`)).blob(),
};

export interface ComplianceReadiness {
  score: number;
  complete: number;
  in_progress: number;
  not_started: number;
  total: number;
  by_section: { section: string; title: string; score: number; complete: number; total: number }[];
  next_actions: { clause_ref: string; clause_title: string; status: string }[];
}

export const DEMO_EMAIL_DOMAIN = '@demo.enms.app';

export function isDemoUser(user: { email?: string } | null | undefined): boolean {
  return !!user?.email && user.email.toLowerCase().endsWith(DEMO_EMAIL_DOMAIN);
}

/** Multipart POST with upload progress (0-100). Resolves with parsed JSON. */
export function uploadWithProgress(path: string, formData: FormData, onProgress: (pct: number) => void): Promise<any> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${API_URL}${path}`);
    const token = getToken();
    if (token) xhr.setRequestHeader('Authorization', `Bearer ${token}`);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(Math.round((e.loaded / e.total) * 100));
    xhr.onload = () => {
      let body: any = {};
      try { body = JSON.parse(xhr.responseText); } catch {}
      xhr.status < 400 ? resolve(body) : reject(new Error(body.detail || 'Request failed'));
    };
    xhr.onerror = () => reject(new Error('Network error'));
    xhr.send(formData);
  });
}
