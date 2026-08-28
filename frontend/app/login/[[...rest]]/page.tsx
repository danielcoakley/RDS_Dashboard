import Link from "next/link";
import { SignIn } from "@clerk/nextjs";
import { redirect } from "next/navigation";
import { createDevSession } from "../../../lib/api";
import { isClerkFrontendEnabled } from "../../../lib/auth-config";
import { writeAppSession } from "../../../lib/session";

type LoginPageProps = {
  searchParams: Promise<{ error?: string }>;
};

function loginErrorMessage(error: string | undefined): string | null {
  if (!error) {
    return null;
  }
  if (error === "missing-fields") {
    return "Enter your user ID to continue.";
  }
  return "We could not sign you in. Check your user ID and organization and try again.";
}

export default async function LoginPage({ searchParams }: LoginPageProps) {
  const query = await searchParams;
  const errorMessage = loginErrorMessage(query.error);
  const useClerk = isClerkFrontendEnabled();

  async function loginAction(formData: FormData) {
    "use server";

    const userId = String(formData.get("user_id") ?? "").trim();
    const organizationId = String(formData.get("organization_id") ?? "").trim();

    if (!userId) {
      redirect("/login?error=missing-fields");
    }

    let hasOrganization = false;
    try {
      const session = await createDevSession({
        user_id: userId,
        organization_id: organizationId || undefined
      });
      await writeAppSession({
        userId: session.user_id,
        organizationId: session.organization_id,
        role: session.role,
        authToken: session.auth_token
      });
      hasOrganization = Boolean(session.organization_id);
    } catch {
      redirect("/login?error=session-failed");
    }
    redirect(hasOrganization ? "/dashboard" : "/organizations");
  }

  if (useClerk) {
    return (
      <main className="authShell">
        <section className="authCard authCardCentered">
          <p className="eyebrow">Sign In</p>
          <h1>Open your workspace</h1>
          <p className="authLead">
            Sign in with your managed account, then choose the active client organization for this
            session.
          </p>
          <div className="clerkAuthFrame">
            <SignIn />
          </div>
          <div className="authActions">
            <Link href="/reset" className="btn btnGhost btnLg">
              Recover account
            </Link>
          </div>
        </section>
      </main>
    );
  }

  return (
    <main className="authShell">
      <section className="authCard">
        <p className="eyebrow">Sign In</p>
        <h1>Open your workspace</h1>
        <p className="authLead">
          Sign in with your user ID and optional organization ID to continue to your tenant workspace.
        </p>

        {errorMessage ? (
          <div className="authNotice authError" role="alert">
            <strong>Session not created</strong>
            <span>{errorMessage}</span>
          </div>
        ) : null}

        <form action={loginAction} className="authForm">
          <label className="authField">
            <span>User ID</span>
            <input name="user_id" type="text" placeholder="user_1" autoComplete="username" required />
          </label>
          <label className="authField">
            <span>Organization ID</span>
            <input name="organization_id" type="text" placeholder="org_1" />
          </label>

          <div className="authActions">
            <button type="submit" className="btn btnPrimary btnLg">
              Sign in
            </button>
            <Link href="/reset" className="btn btnGhost btnLg">
              Recover account
            </Link>
            <Link href="/signup" className="btn btnGhost btnLg">
              Create workspace
            </Link>
          </div>
        </form>
      </section>
    </main>
  );
}
