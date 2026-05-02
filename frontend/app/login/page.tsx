import Link from "next/link";
import { redirect } from "next/navigation";
import { createDevSession } from "../../lib/api";
import { writeAppSession } from "../../lib/session";

type LoginPageProps = {
  searchParams: Promise<{ error?: string }>;
};

function loginErrorMessage(error: string | undefined): string | null {
  if (!error) {
    return null;
  }
  if (error === "missing-fields") {
    return "Enter a user ID to open a local development session.";
  }
  return "We could not create a session for that user. Check the user ID and organization.";
}

export default async function LoginPage({ searchParams }: LoginPageProps) {
  const query = await searchParams;
  const errorMessage = loginErrorMessage(query.error);

  async function loginAction(formData: FormData) {
    "use server";

    const userId = String(formData.get("user_id") ?? "").trim();
    const organizationId = String(formData.get("organization_id") ?? "").trim();

    if (!userId) {
      redirect("/login?error=missing-fields");
    }

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
      redirect(session.organization_id ? "/dashboard" : "/organizations");
    } catch {
      redirect("/login?error=session-failed");
    }
  }

  return (
    <main className="authShell">
      <section className="authCard">
        <p className="eyebrow">Development sign in</p>
        <h1>Open your tenant workspace</h1>
        <p className="authLead">
          This local sign-in page creates a development session token using existing user and
          membership records. It is a bridge to the future Clerk-based auth flow, not the final auth UI.
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
            <input name="user_id" type="text" placeholder="user_1" autoComplete="username" />
          </label>
          <label className="authField">
            <span>Organization ID</span>
            <input name="organization_id" type="text" placeholder="org_1" />
          </label>

          <div className="authActions">
            <button type="submit" className="btn btnPrimary btnLg">
              Sign in
            </button>
            <Link href="/signup" className="btn btnGhost btnLg">
              Create workspace
            </Link>
          </div>
        </form>
      </section>
    </main>
  );
}
