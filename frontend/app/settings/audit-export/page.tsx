import Link from "next/link";
import { redirect } from "next/navigation";
import { WorkspaceShell } from "../../../components/WorkspaceShell";
import { getOrganizationAuditExport } from "../../../lib/api";
import { readAppSession } from "../../../lib/session";

export default async function AuditExportPage() {
  const session = await readAppSession();

  if (!session.userId || !session.organizationId) {
    redirect("/login");
  }

  const exportFile = await getOrganizationAuditExport(
    session.userId,
    session.organizationId,
    session.authToken
  );

  return (
    <WorkspaceShell
      currentPath="/settings"
      title="Audit export"
      modeLabel="Live workspace"
      modeDescription="Tenant-scoped CSV evidence generated from the audit event trail."
    >
      <section className="summaryGrid" aria-label="Audit export metrics">
        <div className="summaryTile">
          <span>Filename</span>
          <strong>{exportFile.filename}</strong>
        </div>
        <div className="summaryTile">
          <span>Format</span>
          <strong>{exportFile.content_type}</strong>
        </div>
        <div className="summaryTile">
          <span>Events</span>
          <strong>{exportFile.event_count}</strong>
        </div>
      </section>

      <section className="contentGrid">
        <div className="listPanel wide">
          <div className="sectionHeader">
            <h2>CSV evidence</h2>
            <span>Audit event export</span>
          </div>
          <pre className="codeBlock">{exportFile.content}</pre>
        </div>

        <div className="listPanel">
          <div className="sectionHeader">
            <h2>Navigation</h2>
            <span>Administration</span>
          </div>
          <div className="rowList">
            <div className="dataRow">
              <div>
                <strong>Back to settings</strong>
                <span>Return to activity log and tenant administration.</span>
              </div>
              <Link href="/settings" className="btn btnGhost btnSm">
                Open settings
              </Link>
            </div>
          </div>
        </div>
      </section>
    </WorkspaceShell>
  );
}
