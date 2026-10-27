import { getMyOrganizations } from "./api";
import { readAppSession } from "./session";

export type WorkspaceContext = {
  orgName: string;
  orgSlug: string;
  roleLabel: string;
  hasLiveSession: boolean;
  provider: "clerk" | "dev" | null;
};

const demoWorkspaceContext: WorkspaceContext = {
  orgName: "Sample organization",
  orgSlug: "sample-organization",
  roleLabel: "Owner",
  hasLiveSession: false,
  provider: null
};

function humanizeRole(role: string | null): string {
  if (!role) {
    return "Member";
  }
  return role.charAt(0).toUpperCase() + role.slice(1);
}

export async function loadWorkspaceContext(): Promise<WorkspaceContext> {
  const session = await readAppSession();

  if (!session.userId) {
    return demoWorkspaceContext;
  }

  if (session.provider === "clerk" && session.organizationId) {
    try {
      const organizations = await getMyOrganizations(session.userId, session.authToken);
      const currentOrganization = organizations.find(
        (organization) => organization.id === session.organizationId
      );

      if (currentOrganization) {
        return {
          orgName: currentOrganization.name,
          orgSlug: currentOrganization.slug,
          roleLabel: humanizeRole(session.role),
          hasLiveSession: true,
          provider: "clerk"
        };
      }
    } catch {
      // Fall back to the verified Clerk session context below.
    }

    return {
      orgName: session.organizationId,
      orgSlug: session.organizationSlug ?? "active-clerk-organization",
      roleLabel: humanizeRole(session.role),
      hasLiveSession: true,
      provider: "clerk"
    };
  }

  try {
    const organizations = await getMyOrganizations(session.userId, session.authToken);
    const currentOrganization = organizations.find(
      (organization) => organization.id === session.organizationId
    );

    if (!currentOrganization) {
      return {
        orgName: "Organization required",
        orgSlug: "select-organization",
        roleLabel: humanizeRole(session.role),
        hasLiveSession: true,
        provider: session.provider ?? "dev"
      };
    }

    return {
      orgName: currentOrganization.name,
      orgSlug: currentOrganization.slug,
      roleLabel: humanizeRole(session.role),
      hasLiveSession: true,
      provider: session.provider ?? "dev"
    };
  } catch {
    return demoWorkspaceContext;
  }
}
