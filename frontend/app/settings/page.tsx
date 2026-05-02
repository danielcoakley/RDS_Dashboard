import { WorkspaceShell } from "../../components/WorkspaceShell";
import { loadSettingsData } from "../../lib/settings-data";

export default async function SettingsPage() {
  const { mode, sites, meters, memberships, invites, auditEvents } = await loadSettingsData();
  const activeMembers = memberships.filter((membership) => membership.is_active);
  const seuMeters = meters.filter((meter) => meter.is_seu);

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
