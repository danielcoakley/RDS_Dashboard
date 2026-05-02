import { getMyOrganizations } from "./api";
import { readAppSession } from "./session";

export type WorkspaceContext = {
  orgName: string;
  orgSlug: string;
  roleLabel: string;
  hasLiveSession: boolean;
};

const demoWorkspaceContext: WorkspaceContext = {
  orgName: "Sample organization",
  orgSlug: "sample-organization",
  roleLabel: "Owner",
  hasLiveSession: false
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
        hasLiveSession: true
      };
    }

    return {
      orgName: currentOrganization.name,
      orgSlug: currentOrganization.slug,
      roleLabel: humanizeRole(session.role),
      hasLiveSession: true
    };
  } catch {
    return demoWorkspaceContext;
  }
}
