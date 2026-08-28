import Link from "next/link";
import { redirect } from "next/navigation";
import { acceptOrganizationInvite, createDevSession } from "../../../lib/api";
import { writeAppSession } from "../../../lib/session";

type InvitePageProps = {
  params: Promise<{ inviteId: string }>;
  searchParams: Promise<{ error?: string }>;
};

function messageForError(error: string | undefined): string | null {
  if (!error) {
    return null;
  }
  if (error === "missing-fields") {
    return "Enter a user ID, email address, and display name to accept the invite.";
  }
  return "We could not accept that invite right now. Check the invite details and try again.";
}

export default async function InviteAcceptancePage({
  params,
  searchParams
}: InvitePageProps) {
  const { inviteId } = await params;
  const query = await searchParams;
  const errorMessage = messageForError(query.error);

  async function acceptInviteAction(formData: FormData) {
    "use server";

    const userId = String(formData.get("user_id") ?? "").trim();
    const email = String(formData.get("email") ?? "").trim();
    const displayName = String(formData.get("display_name") ?? "").trim();

    if (!userId || !email || !displayName) {
      redirect(`/join/${inviteId}?error=missing-fields`);
    }

    try {
      await acceptOrganizationInvite(inviteId, {
        user_id: userId,
        email,
        display_name: displayName
      });
      const session = await createDevSession({
        user_id: userId
      });
      await writeAppSession({
        userId: session.user_id,
        organizationId: session.organization_id,
        role: session.role,
        authToken: session.auth_token
      });
      redirect(`/dashboard?joined=1`);
    } catch {
      redirect(`/join/${inviteId}?error=accept-failed`);
    }
  }

  return (
    <main className="authShell">
      <section className="authCard">
        <p className="eyebrow">Organization invite</p>
        <h1>Join your RDS workspace</h1>
        <p className="authLead">
          Accepting this invite creates your tenant membership and links your profile to the
          organization workspace.
        </p>

        <div className="authInfoGrid">
          <div className="authInfoTile">
            <span>Invite ID</span>
            <strong>{inviteId}</strong>
          </div>
          <div className="authInfoTile">
            <span>Flow</span>
            <strong>Invite acceptance</strong>
          </div>
        </div>

        {errorMessage ? (
          <div className="authNotice authError" role="alert">
            <strong>Invite not accepted</strong>
            <span>{errorMessage}</span>
          </div>
        ) : null}

        <form action={acceptInviteAction} className="authForm">
          <label className="authField">
            <span>User ID</span>
            <input name="user_id" type="text" placeholder="user_123" autoComplete="username" />
          </label>
          <label className="authField">
            <span>Email</span>
            <input name="email" type="email" placeholder="you@example.com" autoComplete="email" />
          </label>
          <label className="authField">
            <span>Display name</span>
            <input
              name="display_name"
              type="text"
              placeholder="Energy Manager"
              autoComplete="name"
            />
          </label>

          <div className="authActions">
            <button type="submit" className="btn btnPrimary btnLg">
              Accept invite
            </button>
            <Link href="/dashboard" className="btn btnGhost btnLg">
              Return to dashboard
            </Link>
          </div>
        </form>
      </section>
    </main>
  );
}
