import { WorkspaceShell } from "../../components/WorkspaceShell";
import { loadDashboardData } from "../../lib/dashboard-data";

export default async function RunsPage() {
  const { mode, runs, reports, auditEvents } = await loadDashboardData();
  const successfulRuns = runs.filter((run) => run.status === "succeeded");
  const failedRuns = runs.filter((run) => run.status === "failed");
  const runAuditEvents = auditEvents.filter((event) => event.resource_type === "run");

  return (
    <WorkspaceShell
      currentPath="/runs"
      title="Run operations"
      modeLabel={mode === "live" ? "Live tenant data" : "Demo workspace"}
      modeDescription={
        mode === "live"
          ? "Run status, outcomes, and audit events are coming from the backend orchestration seam."
          : "Set demo workspace variables to inspect live run metadata and lifecycle events."
      }
    >
      <section className="summaryGrid" aria-label="Run metrics">
        <div className="summaryTile">
          <span>Total runs</span>
          <strong>{runs.length}</strong>
        </div>
        <div className="summaryTile">
          <span>Succeeded</span>
          <strong>{successfulRuns.length}</strong>
        </div>
        <div className="summaryTile">
          <span>Failed</span>
          <strong>{failedRuns.length}</strong>
        </div>
        <div className="summaryTile">
          <span>Generated reports</span>
          <strong>{reports.length}</strong>
        </div>
        <div className="summaryTile">
          <span>Run audit events</span>
          <strong>{runAuditEvents.length}</strong>
        </div>
      </section>

      <section className="contentGrid">
        <div className="listPanel wide">
          <div className="sectionHeader">
            <h2>Run queue</h2>
            <span>Status lifecycle</span>
          </div>
          <div className="rowList">
            {runs.map((run) => (
              <div className="dataRow" key={run.id}>
                <div>
                  <strong>{run.id}</strong>
                  <span>{run.site_id}</span>
                </div>
                <div>
                  <strong>{run.status}</strong>
                  <span>{run.completed_at ?? run.error_message ?? "Awaiting completion"}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="listPanel">
          <div className="sectionHeader">
            <h2>Report outputs</h2>
            <span>Run to report</span>
          </div>
          <div className="rowList">
            {reports.map((report) => (
              <div className="dataRow" key={report.id}>
                <div>
                  <strong>{report.report_type}</strong>
                  <span>{report.run_id}</span>
                </div>
                <div>
                  <strong>{report.is_published ? "Published" : "Draft"}</strong>
                  <span>{report.id}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="listPanel">
          <div className="sectionHeader">
            <h2>Run activity</h2>
            <span>Audit visibility</span>
          </div>
          <div className="rowList">
            {runAuditEvents.map((event) => (
              <div className="dataRow" key={event.id}>
                <div>
                  <strong>{event.action}</strong>
                  <span>{event.resource_id}</span>
                </div>
                <div>
                  <strong>{event.actor_user_id}</strong>
                  <span>{event.metadata_json}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </WorkspaceShell>
  );
}
