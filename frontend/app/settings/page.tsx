import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import { WorkspaceShell } from "../../components/WorkspaceShell";
import {
  createOrganizationMeter,
  createOrganizationInvite,
  createOrganizationSite,
  revokeOrganizationInvite
} from "../../lib/api";
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
  if (status === "invite-revoked") {
    return {
      tone: "success",
      title: "Invite revoked",
      body: "The invite is no longer available for acceptance."
    };
  }
  if (status === "site-created") {
    return {
      tone: "success",
      title: "Site added",
      body: "The new site is now part of your organization."
    };
  }
  if (status === "meter-created") {
    return {
      tone: "success",
      title: "Meter added",
      body: "The new meter is now available for uploads and analysis."
    };
  }
  if (error === "missing-fields") {
    return {
      tone: "error",
      title: "Action not completed",
      body: "Fill in all required fields before submitting."
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
  if (error === "site-failed") {
    return {
      tone: "error",
      title: "Site not added",
      body: "We could not add that site. Check details and try again."
    };
  }
  if (error === "meter-failed") {
    return {
      tone: "error",
      title: "Meter not added",
      body: "We could not add that meter. Check details and try again."
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

  async function revokeInviteAction(formData: FormData) {
    "use server";

    const inviteId = String(formData.get("invite_id") ?? "").trim();
    const session = await readAppSession();

    if (!inviteId) {
      redirect("/settings?error=invite-failed");
    }
    if (!session.userId || !session.organizationId) {
      redirect("/settings?error=session-missing");
    }

    try {
      await revokeOrganizationInvite(
        session.userId,
        session.organizationId,
        inviteId,
        session.authToken
      );
      revalidatePath("/settings");
      redirect("/settings?status=invite-revoked");
    } catch {
      redirect("/settings?error=invite-failed");
    }
  }

  async function createSiteAction(formData: FormData) {
    "use server";

    const name = String(formData.get("name") ?? "").trim();
    const timezone = String(formData.get("timezone") ?? "").trim();
    const session = await readAppSession();

    if (!name || !timezone) {
      redirect("/settings?error=missing-fields");
    }
    if (!session.userId || !session.organizationId) {
      redirect("/settings?error=session-missing");
    }

    const siteId = `site_${Date.now()}`;
    try {
      await createOrganizationSite(
        session.userId,
        session.organizationId,
        {
          site_id: siteId,
          name,
          timezone
        },
        session.authToken
      );
      revalidatePath("/settings");
      revalidatePath("/dashboard");
      revalidatePath("/uploads");
      redirect("/settings?status=site-created");
    } catch {
      redirect("/settings?error=site-failed");
    }
  }

  async function createMeterAction(formData: FormData) {
    "use server";

    const siteId = String(formData.get("site_id") ?? "").trim();
    const displayName = String(formData.get("display_name") ?? "").trim();
    const commodity = String(formData.get("commodity") ?? "").trim();
    const unit = String(formData.get("unit") ?? "").trim();
    const sourceColumn = String(formData.get("source_column") ?? "").trim();
    const isSeu = String(formData.get("is_seu") ?? "").trim() === "on";
    const session = await readAppSession();

    if (!siteId || !displayName || !commodity || !unit || !sourceColumn) {
      redirect("/settings?error=missing-fields");
    }
    if (!session.userId || !session.organizationId) {
      redirect("/settings?error=session-missing");
    }

    const meterId = `meter_${Date.now()}`;
    try {
      await createOrganizationMeter(
        session.userId,
        session.organizationId,
        {
          meter_id: meterId,
          site_id: siteId,
          display_name: displayName,
          commodity,
          unit,
          source_column: sourceColumn,
          is_seu: isSeu
        },
        session.authToken
      );
      revalidatePath("/settings");
      revalidatePath("/dashboard");
      revalidatePath("/uploads");
      redirect("/settings?status=meter-created");
    } catch {
      redirect("/settings?error=meter-failed");
    }
  }

  return (
    <WorkspaceShell
      currentPath="/settings"
      title="Administration"
      modeLabel={mode === "live" ? "Live workspace" : "Sample workspace"}
      modeDescription={
        mode === "live"
          ? "Members, invites, and organization records are loading from the backend."
          : "This page is showing sample administration data until a live organization is selected."
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
            <h2>Members</h2>
            <span>Access and roles</span>
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
            <span>Organization footprint</span>
          </div>
          <form action={createSiteAction} className="inlineForm">
            <label className="authField">
              <span>Name</span>
              <input name="name" type="text" placeholder="Main Site" />
            </label>
            <label className="authField">
              <span>Timezone</span>
              <input name="timezone" type="text" placeholder="Europe/London" />
            </label>
            <button type="submit" className="btn btnPrimary btnSm">
              Add site
            </button>
          </form>
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
          <form action={createMeterAction} className="inlineFormWide">
            <label className="authField">
              <span>Site</span>
              <select name="site_id" defaultValue={sites[0]?.id ?? ""}>
                {sites.map((site) => (
                  <option key={site.id} value={site.id}>
                    {site.name}
                  </option>
                ))}
              </select>
            </label>
            <label className="authField">
              <span>Meter name</span>
              <input name="display_name" type="text" placeholder="Main Electricity" />
            </label>
            <label className="authField">
              <span>Commodity</span>
              <input name="commodity" type="text" placeholder="electricity" />
            </label>
            <label className="authField">
              <span>Unit</span>
              <input name="unit" type="text" placeholder="kWh" />
            </label>
            <label className="authField">
              <span>Source column</span>
              <input name="source_column" type="text" placeholder="Main Electricity" />
            </label>
            <label className="authCheckbox">
              <input name="is_seu" type="checkbox" />
              <span>SEU meter</span>
            </label>
            <button type="submit" className="btn btnPrimary btnSm">
              Add meter
            </button>
          </form>
        </div>

        <div className="listPanel">
          <div className="sectionHeader">
            <h2>Invites</h2>
            <span>Pending access</span>
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
                {invite.status === "pending" ? (
                  <form action={revokeInviteAction}>
                    <input type="hidden" name="invite_id" value={invite.id} />
                    <button type="submit" className="btn btnGhost btnSm">
                      Revoke
                    </button>
                  </form>
                ) : null}
              </div>
            ))}
          </div>
        </div>

        <div className="listPanel">
          <div className="sectionHeader">
            <h2>Activity log</h2>
            <span>Recent admin events</span>
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
