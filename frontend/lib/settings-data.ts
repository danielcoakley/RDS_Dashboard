import {
  getOrganizationAuditEvents,
  getOrganizationInvites,
  getOrganizationMemberships,
  getOrganizationMeters,
  getOrganizationSites,
  type AuditEventSummary,
  type MembershipSummary,
  type MeterSummary,
  type OrganizationInviteSummary,
  type SiteSummary
} from "./api";
import type { DashboardMode } from "./dashboard-data";
import { readAppSession } from "./session";

export type SettingsData = {
  mode: DashboardMode;
  sites: SiteSummary[];
  meters: MeterSummary[];
  memberships: MembershipSummary[];
  invites: OrganizationInviteSummary[];
  auditEvents: AuditEventSummary[];
};

const demoSettingsData: SettingsData = {
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
      is_seu: false
    }
  ],
  memberships: [
    {
      user_id: "user_1",
      organization_id: "org_1",
      role: "owner",
      invited_by_user_id: null,
      email: "owner@example.com",
      display_name: "Owner",
      is_active: true
    },
    {
      user_id: "user_2",
      organization_id: "org_1",
      role: "viewer",
      invited_by_user_id: "user_1",
      email: "viewer@example.com",
      display_name: "Viewer",
      is_active: true
    }
  ],
  invites: [
    {
      id: "invite_1",
      organization_id: "org_1",
      email: "invitee@example.com",
      role: "viewer",
      invited_by_user_id: "user_1",
      status: "pending",
      accepted_by_user_id: null,
      accepted_at: null
    }
  ],
  auditEvents: [
    {
      id: "audit_1",
      organization_id: "org_1",
      actor_user_id: "user_1",
      action: "report_created",
      resource_type: "report",
      resource_id: "report_1",
      metadata_json: "{\"report_type\":\"iso_summary\"}"
    }
  ]
};

export async function loadSettingsData(): Promise<SettingsData> {
  const session = await readAppSession();
  const userId = session.userId ?? process.env.DEMO_USER_ID;
  const organizationId = session.organizationId ?? process.env.DEMO_ORGANIZATION_ID;
  const authToken = session.authToken;

  if (!userId || !organizationId) {
    return demoSettingsData;
  }

  try {
    const [sites, meters, memberships, invites, auditEvents] = await Promise.all([
      getOrganizationSites(userId, organizationId, authToken),
      getOrganizationMeters(userId, organizationId, undefined, authToken),
      getOrganizationMemberships(userId, organizationId, authToken),
      getOrganizationInvites(userId, organizationId, authToken),
      getOrganizationAuditEvents(userId, organizationId, authToken)
    ]);

    return {
      mode: "live",
      sites,
      meters,
      memberships,
      invites,
      auditEvents
    };
  } catch {
    return demoSettingsData;
  }
}
