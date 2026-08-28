import {
  getOrganizationAnalysisConfig,
  getOrganizationAuditEvents,
  getOrganizationDataConnectors,
  getOrganizationEvidencePackageTemplate,
  getOrganizationProfile,
  getOrganizationInvites,
  getOrganizationMemberships,
  getOrganizationMeters,
  getOrganizationPilotReadiness,
  getOrganizationPrivacyRequests,
  getOrganizationPrivacySettings,
  getOrganizationReportApprovalPolicy,
  getOrganizationReportBranding,
  getOrganizationSites,
  getOrganizationSubscription,
  getBillingPlans,
  type AnalysisConfigSummary,
  type AuditEventSummary,
  type DataConnectorSummary,
  type MembershipSummary,
  type MeterSummary,
  type OrganizationEvidencePackageTemplateSummary,
  type OrganizationPilotReadinessSummary,
  type OrganizationSummary,
  type OrganizationPrivacyRequestSummary,
  type OrganizationPrivacySettingsSummary,
  type OrganizationReportApprovalPolicySummary,
  type OrganizationReportBrandingSummary,
  type OrganizationInviteSummary,
  type OrganizationSubscriptionSummary,
  type SubscriptionPlanSummary,
  type SiteSummary
} from "./api";
import type { DashboardMode } from "./dashboard-data";
import { redirect } from "next/navigation";
import { isClerkFrontendEnabled } from "./auth-config";
import { readAppSession } from "./session";

export type SettingsData = {
  mode: DashboardMode;
  sites: SiteSummary[];
  meters: MeterSummary[];
  memberships: MembershipSummary[];
  invites: OrganizationInviteSummary[];
  auditEvents: AuditEventSummary[];
  organization: OrganizationSummary | null;
  evidencePackageTemplate: OrganizationEvidencePackageTemplateSummary | null;
  reportBranding: OrganizationReportBrandingSummary | null;
  reportApprovalPolicy: OrganizationReportApprovalPolicySummary | null;
  pilotReadiness: OrganizationPilotReadinessSummary | null;
  privacySettings: OrganizationPrivacySettingsSummary | null;
  privacyRequests: OrganizationPrivacyRequestSummary[];
  analysisConfig: AnalysisConfigSummary | null;
  dataConnectors: DataConnectorSummary[];
  subscription: OrganizationSubscriptionSummary | null;
  billingPlans: SubscriptionPlanSummary[];
  permissions: SettingsPermissions;
  loadError?: boolean;
};

export type SettingsPermissions = {
  canManageAssets: boolean;
  canManageReports: boolean;
  canManageOrganization: boolean;
  canManageBilling: boolean;
  canManageMembers: boolean;
  canViewAudit: boolean;
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
      created_at: "2026-07-20T09:35:00+00:00",
      actor_user_id: "user_1",
      action: "report_created",
      resource_type: "report",
      resource_id: "report_1",
      metadata_json: "{\"report_type\":\"iso_summary\"}"
    }
  ],
  organization: {
    id: "org_1",
    name: "Example Energy",
    slug: "example-energy"
  },
  reportBranding: {
    organization_id: "org_1",
    evidence_pack_title: "Example Energy Audit Pack",
    primary_color: "#0f766e",
    logo_url: "https://example.com/brand/example-energy-logo.png",
    support_email: "energy.team@example.com",
    updated_at: "2026-07-20T00:00:00+00:00"
  },
  evidencePackageTemplate: {
    organization_id: "org_1",
    template_name: "Full ISO 50001 Evidence Package",
    include_report_artifact: true,
    include_uploads: true,
    include_approval_assignments: true,
    include_reviews: true,
    include_comments: true,
    include_schedules: true,
    section_labels: {
      report_artifact: "Report Artifact Summary",
      uploads: "Upload Lineage",
      approval_assignments: "Approval Assignments",
      reviews: "Management Review Decisions",
      comments: "Collaboration Evidence",
      schedules: "Scheduled Delivery Evidence"
    },
    section_order: [
      "report_artifact",
      "uploads",
      "approval_assignments",
      "reviews",
      "comments",
      "schedules"
    ],
    updated_by_user_id: "user_1",
    updated_at: "2026-07-20T00:00:00+00:00"
  },
  reportApprovalPolicy: {
    organization_id: "org_1",
    auto_lock_evidence_on_final_approval: true,
    minimum_approval_assignments_for_lock: 1,
    updated_by_user_id: "user_1",
    updated_at: "2026-07-20T00:00:00+00:00"
  },
  pilotReadiness: {
    organization_id: "org_1",
    verified_count: 3,
    total_count: 8,
    items: [
      {
        item_key: "dpa_signed",
        label: "Data processing agreement signed",
        status: "verified",
        evidence_url: "https://ops.example.com/dpa",
        notes: "Pilot DPA signed by tenant owner.",
        updated_by_user_id: "user_1",
        updated_at: "2026-07-20T00:00:00+00:00"
      },
      {
        item_key: "storage_backup_verified",
        label: "Storage and backup verified",
        status: "verified",
        evidence_url: "https://ops.example.com/restore-test",
        notes: "Restore smoke tested for the pilot metadata store.",
        updated_by_user_id: "user_1",
        updated_at: "2026-07-20T00:00:00+00:00"
      },
      {
        item_key: "tenant_isolation_reviewed",
        label: "Tenant isolation reviewed",
        status: "verified",
        evidence_url: null,
        notes: "Service/API tenant isolation tests passed.",
        updated_by_user_id: "user_1",
        updated_at: "2026-07-20T00:00:00+00:00"
      },
      {
        item_key: "production_auth_verified",
        label: "Production auth verified",
        status: "in_progress",
        evidence_url: null,
        notes: "Waiting on live Clerk dashboard webhook replay.",
        updated_by_user_id: "user_1",
        updated_at: "2026-07-20T00:00:00+00:00"
      },
      {
        item_key: "privacy_notice_reviewed",
        label: "Privacy notice reviewed",
        status: "in_progress",
        evidence_url: null,
        notes: "Awaiting counsel review before pilot launch.",
        updated_by_user_id: "user_1",
        updated_at: "2026-07-20T00:00:00+00:00"
      },
      {
        item_key: "production_billing_verified",
        label: "Production billing verified",
        status: "blocked",
        evidence_url: null,
        notes: "Stripe dashboard setup has not been replay-tested.",
        updated_by_user_id: "user_1",
        updated_at: "2026-07-20T00:00:00+00:00"
      },
      {
        item_key: "email_delivery_verified",
        label: "Email delivery verified",
        status: "pending",
        evidence_url: null,
        notes: null,
        updated_by_user_id: null,
        updated_at: null
      },
      {
        item_key: "support_owner_assigned",
        label: "Support owner assigned",
        status: "pending",
        evidence_url: null,
        notes: null,
        updated_by_user_id: null,
        updated_at: null
      }
    ]
  },
  privacySettings: {
    organization_id: "org_1",
    data_processing_acknowledged_at: "2026-07-20T00:00:00+00:00",
    data_processing_acknowledged_by_user_id: "user_1",
    data_processing_contact_email: "privacy@example.com",
    subprocessor_notifications_enabled: true,
    analytics_data_sharing_enabled: false,
    updated_at: "2026-07-20T00:00:00+00:00"
  },
  privacyRequests: [
    {
      id: "privacy_request_1",
      organization_id: "org_1",
      requested_by_user_id: "user_1",
      subject_user_id: "user_2",
      request_type: "deletion",
      status: "in_review",
      description: "Review deletion request for a departed user.",
      resolution_notes: null,
      completed_at: null,
      created_at: "2026-07-20T00:00:00+00:00",
      updated_at: "2026-07-20T00:00:00+00:00"
    }
  ],
  analysisConfig: {
    organization_id: "org_1",
    source: "tenant_assets",
    updated_by_user_id: null,
    updated_at: null,
    config: {
      client: {
        id: "org_1",
        name: "Example Energy",
        timezone: "Europe/London",
        reporting_currency: "GBP"
      },
      site: {
        name: "Main Site",
        floor_area_m2: null,
        activity_metric: null
      },
      meters: [
        {
          meter_id: "meter_1",
          display_name: "Main Electricity",
          commodity: "electricity",
          unit: "kWh",
          source_column: "Main Electricity",
          is_seu: true
        },
        {
          meter_id: "meter_2",
          display_name: "Main Gas",
          commodity: "gas",
          unit: "kWh",
          source_column: "Main Gas",
          is_seu: false
        }
      ],
      analysis: {
        baseline_start: null,
        baseline_end: null,
        reporting_start: null,
        reporting_end: null,
        weather_normalisation: true,
        hdd_base_temperature: 15.5,
        cdd_base_temperature: 22.0,
        weather_model_min_baseline_records: 10,
        weather_model_min_comparison_records: 1,
        outlier_removal: true
      },
      iso50001: {
        organisation_context: "",
        energy_policy_reference: "",
        audit_notes: ""
      }
    }
  },
  dataConnectors: [
    {
      id: "connector_utility_portal",
      organization_id: "org_1",
      site_id: "site_1",
      name: "Utility portal interval export",
      connector_type: "utility_portal",
      source_category: "energy",
      status: "draft",
      schedule: "monthly",
      external_reference: "utility-portal:main-site",
      credential_reference_configured: true,
      created_by_user_id: "user_1",
      updated_by_user_id: "user_1",
      last_sync_at: null,
      last_sync_status: null,
      last_sync_error: null,
      next_sync_at: "2026-08-01T09:00:00+00:00",
      created_at: "2026-07-20T00:00:00+00:00",
      updated_at: "2026-07-20T00:00:00+00:00"
    }
  ],
  subscription: {
    organization_id: "org_1",
    status: "trialing",
    current_period_start: "2026-07-20T00:00:00+00:00",
    current_period_end: "2026-08-19T00:00:00+00:00",
    trial_ends_at: "2026-08-19T00:00:00+00:00",
    external_customer_id: null,
    external_subscription_id: null,
    plan: {
      id: "trial",
      name: "Pilot trial",
      description: "Self-serve ISO 50001 pilot workspace",
      monthly_price_cents: 0,
      included_users: 5,
      included_sites: 3,
      monthly_run_limit: 25,
      report_retention_days: 90,
      is_active: true
    },
    usage: {
      member_count: 2,
      active_user_count: 2,
      pending_invite_count: 1,
      site_count: 1,
      runs_this_period: 1
    }
  },
  billingPlans: [
    {
      id: "trial",
      name: "Pilot trial",
      description: "Self-serve ISO 50001 pilot workspace",
      monthly_price_cents: 0,
      included_users: 5,
      included_sites: 3,
      monthly_run_limit: 25,
      report_retention_days: 90,
      is_active: true
    },
    {
      id: "team",
      name: "Team",
      description: "Multi-client energy management workspace for a single organization",
      monthly_price_cents: 49000,
      included_users: 15,
      included_sites: 10,
      monthly_run_limit: 250,
      report_retention_days: 365,
      is_active: true
    },
    {
      id: "enterprise",
      name: "Enterprise",
      description: "Portfolio workspace with custom limits and support",
      monthly_price_cents: null,
      included_users: null,
      included_sites: null,
      monthly_run_limit: null,
      report_retention_days: null,
      is_active: true
    }
  ],
  permissions: {
    canManageAssets: true,
    canManageReports: true,
    canManageOrganization: true,
    canManageBilling: true,
    canManageMembers: true,
    canViewAudit: true
  }
};

function permissionsForRole(role: string | null): SettingsPermissions {
  const normalizedRole = role?.toLowerCase() ?? null;

  if (!normalizedRole) {
    return {
      canManageAssets: true,
      canManageReports: true,
      canManageOrganization: true,
      canManageBilling: true,
      canManageMembers: true,
      canViewAudit: true
    };
  }

  return {
    canManageAssets: normalizedRole === "owner" || normalizedRole === "manager",
    canManageReports: normalizedRole === "owner" || normalizedRole === "manager",
    canManageOrganization: normalizedRole === "owner",
    canManageBilling: normalizedRole === "owner",
    canManageMembers: normalizedRole === "owner",
    canViewAudit: normalizedRole === "owner" || normalizedRole === "manager"
  };
}

async function emptyOnDenied<T>(fetchData: () => Promise<T[]>): Promise<T[]> {
  try {
    return await fetchData();
  } catch {
    return [];
  }
}

export async function loadSettingsData(): Promise<SettingsData> {
  const session = await readAppSession();
  const userId = session.userId ?? process.env.DEMO_USER_ID;
  const organizationId = session.organizationId ?? process.env.DEMO_ORGANIZATION_ID;
  const authToken = session.authToken;
  const permissions = permissionsForRole(session.role ?? process.env.DEMO_ROLE ?? null);

  if (isClerkFrontendEnabled() && !session.userId) {
    redirect("/login");
  }

  if (session.userId && !session.organizationId) {
    redirect("/organizations");
  }

  if (!userId || !organizationId) {
    return demoSettingsData;
  }

  try {
    const [organization, sites, meters] = await Promise.all([
      getOrganizationProfile(userId, organizationId, authToken),
      getOrganizationSites(userId, organizationId, authToken),
      getOrganizationMeters(userId, organizationId, undefined, authToken)
    ]);
    const [
      memberships,
      invites,
      auditEvents,
      pilotReadiness,
      privacySettings,
      reportBranding,
      evidencePackageTemplate,
      reportApprovalPolicy,
      privacyRequests,
      analysisConfig,
      dataConnectors,
      subscription,
      billingPlans
    ] = await Promise.all([
      permissions.canManageMembers
        ? emptyOnDenied(() => getOrganizationMemberships(userId, organizationId, authToken))
        : Promise.resolve([]),
      permissions.canManageMembers
        ? emptyOnDenied(() => getOrganizationInvites(userId, organizationId, authToken))
        : Promise.resolve([]),
      permissions.canViewAudit
        ? emptyOnDenied(() => getOrganizationAuditEvents(userId, organizationId, authToken))
        : Promise.resolve([]),
      permissions.canManageOrganization
        ? getOrganizationPilotReadiness(userId, organizationId, authToken).catch(() => null)
        : Promise.resolve(null),
      getOrganizationPrivacySettings(userId, organizationId, authToken).catch(() => null),
      getOrganizationReportBranding(userId, organizationId, authToken).catch(() => null),
      getOrganizationEvidencePackageTemplate(userId, organizationId, authToken).catch(() => null),
      getOrganizationReportApprovalPolicy(userId, organizationId, authToken).catch(() => null),
      permissions.canManageOrganization
        ? emptyOnDenied(() => getOrganizationPrivacyRequests(userId, organizationId, authToken))
        : Promise.resolve([]),
      getOrganizationAnalysisConfig(userId, organizationId, sites[0]?.id, authToken).catch(() => null),
      getOrganizationDataConnectors(userId, organizationId, undefined, authToken).catch(() => []),
      getOrganizationSubscription(userId, organizationId, authToken).catch(() => null),
      getBillingPlans(userId, authToken).catch(() => [])
    ]);

    return {
      mode: "live",
      sites,
      meters,
      memberships,
      invites,
      auditEvents,
      organization,
      evidencePackageTemplate,
      pilotReadiness,
      privacySettings,
      reportBranding,
      reportApprovalPolicy,
      privacyRequests,
      analysisConfig,
      dataConnectors,
      subscription,
      billingPlans,
      permissions
    };
  } catch {
    if (session.userId) {
      return {
        mode: "live",
        sites: [],
        meters: [],
        memberships: [],
        invites: [],
        auditEvents: [],
        organization: null,
        evidencePackageTemplate: null,
        reportBranding: null,
        reportApprovalPolicy: null,
        pilotReadiness: null,
        privacySettings: null,
        privacyRequests: [],
        analysisConfig: null,
        dataConnectors: [],
        subscription: null,
        billingPlans: [],
        permissions,
        loadError: true
      };
    }

    return demoSettingsData;
  }
}
