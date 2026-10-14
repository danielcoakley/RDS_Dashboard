export const ROLES = [
  { value: 'owner', label: 'Owner', help: 'Manage team, organization and all data' },
  { value: 'manager', label: 'Manager', help: 'Edit sites, data, objectives and compliance' },
  { value: 'viewer', label: 'Viewer', help: 'Read-only access' },
];

// ISO 50001 clause 5.3 — roles, responsibilities and authorities
export const TEAM_ROLES = [
  { value: 'top_management', label: 'Top management' },
  { value: 'energy_manager', label: 'Energy manager' },
  { value: 'energy_team_member', label: 'Energy team member' },
];

export const teamRoleLabel = (v?: string | null) => TEAM_ROLES.find((t) => t.value === v)?.label ?? '—';
