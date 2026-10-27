import Link from "next/link";
import { SignIn } from "@clerk/nextjs";
import { isClerkFrontendEnabled } from "../../lib/auth-config";

export default function ResetPage() {
  const useClerk = isClerkFrontendEnabled();

  if (useClerk) {
    return (
      <main className="authShell">
        <section className="authCard authCardCentered">
          <p className="eyebrow">Account Recovery</p>
          <h1>Recover access</h1>
          <p className="authLead">
            Use the managed sign-in flow to reset your password or verify your account before
            opening a client workspace.
          </p>
          <div className="clerkAuthFrame">
            <SignIn routing="hash" />
          </div>
          <div className="authActions">
            <Link href="/login" className="btn btnGhost btnLg">
              Back to sign in
            </Link>
          </div>
        </section>
      </main>
    );
  }

  return (
    <main className="authShell">
      <section className="authCard">
        <p className="eyebrow">Account Recovery</p>
        <h1>Recover access</h1>
        <p className="authLead">
          Local development sessions use seeded workspace IDs instead of passwords. Production
          recovery is handled by the configured identity provider.
        </p>
        <div className="authNotice">
          <strong>Development mode</strong>
          <span>
            Ask a workspace owner to confirm your user ID or invite you again, then sign in with
            the matching tenant context.
          </span>
        </div>
        <div className="authActions">
          <Link href="/login" className="btn btnPrimary btnLg">
            Back to sign in
          </Link>
          <Link href="/signup" className="btn btnGhost btnLg">
            Create workspace
          </Link>
        </div>
      </section>
    </main>
  );
}
