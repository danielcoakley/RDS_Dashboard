import Link from "next/link";
import { redirect } from "next/navigation";
import { createDevSession, getMyOrganizations } from "../../lib/api";
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
  return "We could not switch to that tenant. Try selecting the organization again.";
}

export default async function OrganizationsPage({ searchParams }: OrganizationsPageProps) {
  const query = await searchParams;
  const session = await readAppSession();

  if (!session.userId) {
    redirect("/login");
  }

  const organizations = await getMyOrganizations(session.userId, session.authToken);
  const errorMessage = organizationsError(query.error);

  async function selectOrganizationAction(formData: FormData) {
    "use server";

    const organizationId = String(formData.get("organization_id") ?? "").trim();
    const currentSession = await readAppSession();

    if (!currentSession.userId) {
      redirect("/organizations?error=session-missing");
    }

    try {
      const nextSession = await createDevSession({
        user_id: currentSession.userId,
        organization_id: organizationId
      });
      await writeAppSession({
        userId: nextSession.user_id,
        organizationId: nextSession.organization_id,
        role: nextSession.role,
        authToken: nextSession.auth_token
      });
      redirect("/dashboard");
    } catch {
      redirect("/organizations?error=switch-failed");
    }
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
