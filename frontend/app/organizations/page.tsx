import Link from "next/link";
import { redirect } from "next/navigation";
import { ClerkOrganizationSelector } from "../../components/ClerkControls";
import { createDevSession, getMyOrganizations, syncClerkSession } from "../../lib/api";
import { isClerkFrontendEnabled } from "../../lib/auth-config";
import { readAppSession, writeAppSession } from "../../lib/session";

type OrganizationsPageProps = {
  searchParams: Promise<{ error?: string }>;
};

function organizationsError(error: string | undefined): string | null {
  if (!error) {
    return null;
  }
  if (error === "session-missing") {
    return "Sign in before selecting an organization.";
  }
  if (error === "organization-missing") {
    return "Choose an active Clerk organization before opening the workspace.";
  }
  if (error === "sync-failed") {
    return "We could not activate that Clerk organization in the workspace.";
  }
  return "We could not switch to that tenant. Try selecting the organization again.";
}

export default async function OrganizationsPage({ searchParams }: OrganizationsPageProps) {
  const query = await searchParams;
  const session = await readAppSession();
  const useClerk = isClerkFrontendEnabled();
  const errorMessage = organizationsError(query.error);

  if (!session.userId) {
    redirect("/login");
  }

  if (useClerk && session.provider === "clerk") {
    async function activateClerkOrganizationAction() {
      "use server";

      const currentSession = await readAppSession();
      if (!currentSession.userId || currentSession.provider !== "clerk" || !currentSession.authToken) {
        redirect("/organizations?error=session-missing");
      }
      if (!currentSession.organizationId) {
        redirect("/organizations?error=organization-missing");
      }

      try {
        await syncClerkSession(currentSession.userId, currentSession.authToken);
      } catch {
        redirect("/organizations?error=sync-failed");
      }

      redirect("/dashboard");
    }

    return (
      <main className="authShell">
        <section className="authCard authCardCentered">
          <p className="eyebrow">Tenant switcher</p>
          <h1>Select your organization</h1>
          <p className="authLead">
            Choose the active Clerk organization before opening tenant-scoped analytics.
          </p>

          <ClerkOrganizationSelector enabled={useClerk} />

          {errorMessage ? (
            <div className="authNotice authError" role="alert">
              <strong>Organization not active</strong>
              <span>{errorMessage}</span>
            </div>
          ) : null}

          <div className="authActions">
            <form action={activateClerkOrganizationAction}>
              <button type="submit" className="btn btnPrimary btnLg">
                Open workspace
              </button>
            </form>
          </div>
        </section>
      </main>
    );
  }

  const organizations = await getMyOrganizations(session.userId, session.authToken);

  async function selectOrganizationAction(formData: FormData) {
    "use server";

    const organizationId = String(formData.get("organization_id") ?? "").trim();
    const currentSession = await readAppSession();

    if (!currentSession.userId) {
      redirect("/organizations?error=session-missing");
    }

    let nextSession;
    try {
      nextSession = await createDevSession({
        user_id: currentSession.userId,
        organization_id: organizationId
      });
      await writeAppSession({
        userId: nextSession.user_id,
        organizationId: nextSession.organization_id,
        role: nextSession.role,
        authToken: nextSession.auth_token
      });
    } catch {
      redirect("/organizations?error=switch-failed");
    }

    redirect("/dashboard");
  }

  return (
    <main className="authShell">
      <section className="authCard">
        <p className="eyebrow">Tenant switcher</p>
        <h1>Select your organization</h1>
        <p className="authLead">
          Your session is active, but the app still needs an organization context before loading
          tenant-scoped resources.
        </p>

        {errorMessage ? (
          <div className="authNotice authError" role="alert">
            <strong>Organization not selected</strong>
            <span>{errorMessage}</span>
          </div>
        ) : null}

        <div className="rowList">
          {organizations.map((organization) => (
            <form action={selectOrganizationAction} className="dataRow formRow" key={organization.id}>
              <input type="hidden" name="organization_id" value={organization.id} />
              <div>
                <strong>{organization.name}</strong>
                <span>{organization.slug}</span>
              </div>
              <button type="submit" className="btn btnPrimary btnSm">
                Open
              </button>
            </form>
          ))}
        </div>

        <div className="authActions">
          <Link href="/logout" className="btn btnGhost btnLg">
            Sign out
          </Link>
        </div>
      </section>
    </main>
  );
}
