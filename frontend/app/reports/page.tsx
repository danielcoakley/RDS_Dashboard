import { WorkspaceShell } from "../../components/WorkspaceShell";
import { loadDashboardData } from "../../lib/dashboard-data";

export default async function ReportsPage() {
  const { mode, reports, runs } = await loadDashboardData();
  const publishedReports = reports.filter((report) => report.is_published);

  return (
    <WorkspaceShell
      currentPath="/reports"
      title="Report library"
      modeLabel={mode === "live" ? "Live tenant data" : "Demo workspace"}
      modeDescription={
        mode === "live"
          ? "Report metadata is coming from tenant-scoped backend records."
          : "Set demo workspace variables to inspect live report artifacts from the API."
      }
    >
      <section className="summaryGrid" aria-label="Report metrics">
        <div className="summaryTile">
          <span>Total reports</span>
          <strong>{reports.length}</strong>
        </div>
        <div className="summaryTile">
          <span>Published</span>
          <strong>{publishedReports.length}</strong>
        </div>
        <div className="summaryTile">
          <span>Draft</span>
          <strong>{reports.length - publishedReports.length}</strong>
        </div>
        <div className="summaryTile">
          <span>Runs represented</span>
          <strong>{new Set(reports.map((report) => report.run_id)).size}</strong>
        </div>
        <div className="summaryTile">
          <span>Latest run</span>
          <strong>{runs[0]?.id ?? "None"}</strong>
        </div>
      </section>

      <section className="contentGrid">
        <div className="listPanel wide">
          <div className="sectionHeader">
            <h2>Artifacts</h2>
            <span>Storage references</span>
          </div>
          <div className="rowList">
            {reports.map((report) => (
              <div className="dataRow" key={report.id}>
                <div>
                  <strong>{report.report_type}</strong>
                  <span>{report.storage_key}</span>
                </div>
                <div>
                  <strong>{report.is_published ? "Published" : "Draft"}</strong>
                  <span>{report.run_id}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="listPanel">
          <div className="sectionHeader">
            <h2>Published outputs</h2>
            <span>Ready for review</span>
          </div>
          <div className="rowList">
            {publishedReports.map((report) => (
              <div className="dataRow" key={`${report.id}-published`}>
                <div>
                  <strong>{report.id}</strong>
                  <span>{report.organization_id}</span>
                </div>
                <div>
                  <strong>{report.report_type}</strong>
                  <span>{report.run_id}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="listPanel">
          <div className="sectionHeader">
            <h2>Run links</h2>
            <span>Artifact lineage</span>
          </div>
          <div className="rowList">
            {runs.map((run) => (
              <div className="dataRow" key={`${run.id}-lineage`}>
                <div>
                  <strong>{run.id}</strong>
                  <span>{run.status}</span>
                </div>
                <div>
                  <strong>{reports.filter((report) => report.run_id === run.id).length} reports</strong>
                  <span>{run.site_id}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </WorkspaceShell>
  );
}
