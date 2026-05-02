import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import { WorkspaceShell } from "../../components/WorkspaceShell";
import { createOrganizationInvite } from "../../lib/api";
import { readAppSession } from "../../lib/session";
import { loadSettingsData } from "../../lib/settings-data";

type SettingsPageProps = {
  searchParams: Promise<{ status?: string; error?: string }>;
};

function settingsMessage(
  status: string | undefined,
  error: string | undefined
): { tone: "success" | "error"; title: string; body: string } | null {
  if (status === "invite-created") {
    return {
      tone: "success",
      title: "Invite created",
      body: "The new invite is available in the tenant invite list."
    };
  }
  if (error === "missing-fields") {
    return {
      tone: "error",
      title: "Invite not created",
      body: "Enter an email address and role before sending an invite."
    };
  }
  if (error === "session-missing") {
    return {
      tone: "error",
      title: "Invite not created",
      body: "Sign in to a tenant workspace before managing invitations."
    };
  }
  if (error === "invite-failed") {
    return {
      tone: "error",
      title: "Invite not created",
      body: "We could not create that invite. Check for duplicate email addresses or permissions."
    };
  }
  return null;
}

export default async function SettingsPage({ searchParams }: SettingsPageProps) {
  const query = await searchParams;
  const { mode, sites, meters, memberships, invites, auditEvents } = await loadSettingsData();
  const activeMembers = memberships.filter((membership) => membership.is_active);
  const seuMeters = meters.filter((meter) => meter.is_seu);
  const pageMessage = settingsMessage(query.status, query.error);

  async function createInviteAction(formData: FormData) {
    "use server";

    const email = String(formData.get("email") ?? "").trim();
    const roleValue = String(formData.get("role") ?? "viewer").trim();
    const session = await readAppSession();

    if (!email || !roleValue) {
      redirect("/settings?error=missing-fields");
    }
    if (!session.userId || !session.organizationId) {
      redirect("/settings?error=session-missing");
    }

    const inviteLocalId = `invite_${Date.now()}`;
    try {
      await createOrganizationInvite(
        session.userId,
        session.organizationId,
        {
          invite_id: inviteLocalId,
          email,
          role: roleValue as "owner" | "manager" | "viewer"
        },
        session.authToken
      );
      revalidatePath("/settings");
      redirect("/settings?status=invite-created");
    } catch {
      redirect("/settings?error=invite-failed");
    }
  }

  return (
    <WorkspaceShell
      currentPath="/settings"
      title="Tenant settings"
      modeLabel={mode === "live" ? "Live tenant data" : "Demo workspace"}
      modeDescription={
        mode === "live"
          ? "Memberships, site inventory, and audit visibility are coming from tenant-aware API routes."
          : "Live settings data will appear here when demo workspace variables point at the backend."
      }
    >
      {pageMessage ? (
        <div
          className={`authNotice ${pageMessage.tone === "success" ? "authSuccess" : "authError"}`}
          role={pageMessage.tone === "success" ? "status" : "alert"}
        >
          <strong>{pageMessage.title}</strong>
          <span>{pageMessage.body}</span>
        </div>
      ) : null}

      <section className="summaryGrid" aria-label="Settings metrics">
        <div className="summaryTile">
          <span>Team members</span>
          <strong>{memberships.length}</strong>
        </div>
        <div className="summaryTile">
          <span>Active users</span>
          <strong>{activeMembers.length}</strong>
        </div>
        <div className="summaryTile">
          <span>Sites</span>
          <strong>{sites.length}</strong>
        </div>
        <div className="summaryTile">
          <span>SEU meters</span>
          <strong>{seuMeters.length}</strong>
        </div>
        <div className="summaryTile">
          <span>Audit records</span>
          <strong>{auditEvents.length}</strong>
        </div>
        <div className="summaryTile">
          <span>Open invites</span>
          <strong>{invites.filter((invite) => invite.status === "pending").length}</strong>
        </div>
      </section>

      <section className="contentGrid">
        <div className="listPanel wide">
          <div className="sectionHeader">
            <h2>Organization members</h2>
            <span>Owner-managed access</span>
          </div>
          <div className="rowList">
            {memberships.map((membership) => (
              <div className="dataRow" key={membership.user_id}>
                <div>
                  <strong>{membership.display_name}</strong>
                  <span>{membership.email}</span>
                </div>
                <div>
                  <strong>{membership.role}</strong>
                  <span>{membership.is_active ? "Active" : "Disabled"}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="listPanel">
          <div className="sectionHeader">
            <h2>Sites</h2>
            <span>Tenant footprint</span>
          </div>
          <div className="rowList">
            {sites.map((site) => (
              <div className="dataRow" key={site.id}>
                <div>
                  <strong>{site.name}</strong>
                  <span>{site.timezone}</span>
                </div>
                <div>
                  <strong>{meters.filter((meter) => meter.site_id === site.id).length} meters</strong>
                  <span>{site.id}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="listPanel">
          <div className="sectionHeader">
            <h2>Pending invites</h2>
            <span>Invite workflow</span>
          </div>
          <form action={createInviteAction} className="inlineForm">
            <label className="authField">
              <span>Email</span>
              <input name="email" type="email" placeholder="invitee@example.com" />
            </label>
            <label className="authField">
              <span>Role</span>
              <select name="role" defaultValue="viewer">
                <option value="viewer">Viewer</option>
                <option value="manager">Manager</option>
                <option value="owner">Owner</option>
              </select>
            </label>
            <button type="submit" className="btn btnPrimary btnSm">
              Send invite
            </button>
          </form>
          <div className="rowList">
            {invites.map((invite) => (
              <div className="dataRow" key={invite.id}>
                <div>
                  <strong>{invite.email}</strong>
                  <span>{invite.id}</span>
                </div>
                <div>
                  <strong>{invite.role}</strong>
                  <span>{invite.status}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="listPanel">
          <div className="sectionHeader">
            <h2>Audit visibility</h2>
            <span>Recent activity</span>
          </div>
          <div className="rowList">
            {auditEvents.map((event) => (
              <div className="dataRow" key={event.id}>
                <div>
                  <strong>{event.action}</strong>
                  <span>{event.resource_type}</span>
                </div>
                <div>
                  <strong>{event.actor_user_id}</strong>
                  <span>{event.resource_id}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </WorkspaceShell>
  );
}
