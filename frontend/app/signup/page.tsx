import Link from "next/link";
import { redirect } from "next/navigation";
import { createDevSession, createOwnerOrganization } from "../../lib/api";
import { writeAppSession } from "../../lib/session";

type SignupPageProps = {
  searchParams: Promise<{ error?: string }>;
};

function signupErrorMessage(error: string | undefined): string | null {
  if (!error) {
    return null;
  }
  if (error === "missing-fields") {
    return "Fill in the user and organization details to create the workspace.";
  }
  return "We could not create that workspace. Check for duplicate IDs or slugs and try again.";
}

export default async function SignupPage({ searchParams }: SignupPageProps) {
  const query = await searchParams;
  const errorMessage = signupErrorMessage(query.error);

  async function signupAction(formData: FormData) {
    "use server";

    const userId = String(formData.get("user_id") ?? "").trim();
    const email = String(formData.get("email") ?? "").trim();
    const displayName = String(formData.get("display_name") ?? "").trim();
    const organizationId = String(formData.get("organization_id") ?? "").trim();
    const organizationName = String(formData.get("organization_name") ?? "").trim();
    const organizationSlug = String(formData.get("organization_slug") ?? "").trim();

    if (!userId || !email || !displayName || !organizationId || !organizationName || !organizationSlug) {
      redirect("/signup?error=missing-fields");
    }

    try {
      await createOwnerOrganization({
        user_id: userId,
        email,
        display_name: displayName,
        organization_id: organizationId,
        organization_name: organizationName,
        organization_slug: organizationSlug
      });

      const session = await createDevSession({
        user_id: userId,
        organization_id: organizationId
      });
      await writeAppSession({
        userId: session.user_id,
        organizationId: session.organization_id,
        role: session.role,
        authToken: session.auth_token
      });
      redirect("/dashboard");
    } catch {
      redirect("/signup?error=signup-failed");
    }
  }

  return (
    <main className="authShell">
      <section className="authCard">
        <p className="eyebrow">Create workspace</p>
        <h1>Set up your tenant</h1>
        <p className="authLead">
          This local onboarding page creates the owner user, organization, and first development
          session so we can exercise the SaaS flows without live identity infrastructure yet.
        </p>

        {errorMessage ? (
          <div className="authNotice authError" role="alert">
            <strong>Workspace not created</strong>
            <span>{errorMessage}</span>
          </div>
        ) : null}

        <form action={signupAction} className="authForm">
          <label className="authField">
            <span>User ID</span>
            <input name="user_id" type="text" placeholder="owner_1" autoComplete="username" />
          </label>
          <label className="authField">
            <span>Email</span>
            <input name="email" type="email" placeholder="owner@example.com" autoComplete="email" />
          </label>
          <label className="authField">
            <span>Display name</span>
            <input name="display_name" type="text" placeholder="Energy Lead" autoComplete="name" />
          </label>
          <label className="authField">
            <span>Organization ID</span>
            <input name="organization_id" type="text" placeholder="org_1" />
          </label>
          <label className="authField">
            <span>Organization name</span>
            <input name="organization_name" type="text" placeholder="Example Energy" />
          </label>
          <label className="authField">
            <span>Organization slug</span>
            <input name="organization_slug" type="text" placeholder="example-energy" />
          </label>

          <div className="authActions">
            <button type="submit" className="btn btnPrimary btnLg">
              Create tenant
            </button>
            <Link href="/login" className="btn btnGhost btnLg">
              I already have access
            </Link>
          </div>
        </form>
      </section>
    </main>
  );
}
