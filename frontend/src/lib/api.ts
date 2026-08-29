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

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options.headers as Record<string, string>),
  };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const res = await fetch(`${API_URL}${path}`, { ...options, headers });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(detail.detail || 'Request failed');
  }
  return res.json();
}

export const api = {
  // Auth
  signup: (data: any) => request('/api/auth/signup', { method: 'POST', body: JSON.stringify(data) }),
  login: (data: any) => request('/api/auth/login', { method: 'POST', body: JSON.stringify(data) }),
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

  // Analytics
  runAnalysis: (data: any) => request('/api/analytics/analysis', { method: 'POST', body: JSON.stringify(data) }),
  availableYears: (siteId: number) => request(`/api/analytics/years/${siteId}`),
  monthlyComparison: (siteId: number, baselineYear: number, comparisonYear: number) =>
    request(`/api/analytics/monthly/${siteId}?baseline_year=${baselineYear}&comparison_year=${comparisonYear}`),
  sankey: (siteId: number, baselineYear: number, comparisonYear: number) =>
    request(`/api/analytics/sankey/${siteId}?baseline_year=${baselineYear}&comparison_year=${comparisonYear}`),
  seuSummary: (siteId: number, baselineYear: number, comparisonYear: number) =>
    request(`/api/analytics/seu-summary/${siteId}?baseline_year=${baselineYear}&comparison_year=${comparisonYear}`),

  // Objectives
  listObjectives: () => request('/api/objectives'),
  createObjective: (data: any) => request('/api/objectives', { method: 'POST', body: JSON.stringify(data) }),
  updateObjective: (id: number, data: any) => request(`/api/objectives/${id}`, { method: 'PUT', body: JSON.stringify(data) }),
  deleteObjective: (id: number) => request(`/api/objectives/${id}`, { method: 'DELETE' }),

  // Compliance
  listCompliance: () => request('/api/compliance'),
  updateCompliance: (clauseRef: string, data: any) => request(`/api/compliance/${clauseRef}`, { method: 'PUT', body: JSON.stringify(data) }),
  complianceScore: () => request('/api/compliance/score'),

  // Energy Review
  listReviews: (siteId: number) => request(`/api/energy-review/${siteId}`),
  createReview: (data: any) => request('/api/energy-review', { method: 'POST', body: JSON.stringify(data) }),
  updateReview: (id: number, data: any) => request(`/api/energy-review/${id}`, { method: 'PUT', body: JSON.stringify(data) }),
  deleteReview: (id: number) => request(`/api/energy-review/${id}`, { method: 'DELETE' }),
};
