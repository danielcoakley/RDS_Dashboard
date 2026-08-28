import Link from "next/link";
import { WorkspaceShell } from "../../components/WorkspaceShell";
import { loadDashboardData } from "../../lib/dashboard-data";

export default async function ReportsPage() {
  const { mode, reports, runs } = await loadDashboardData();
  const publishedReports = reports.filter((report) => report.is_published);
  const hasReports = reports.length > 0;

  return (
    <WorkspaceShell
      currentPath="/reports"
      title="Reports"
      modeLabel={mode === "live" ? "Live workspace" : "Sample workspace"}
      modeDescription={
        mode === "live"
          ? "Report metadata is loading from the backend."
          : "This page is showing sample report records until a live organization is selected."
      }
    >
      {!hasReports ? (
        <div className="authNotice authError" role="alert">
          <strong>No reports yet</strong>
          <span>
            Request a run from the runs page after uploading tenant data. Generated reports will
            appear here.
          </span>
        </div>
      ) : null}
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
            <h2>Report library</h2>
            <span>Available files</span>
          </div>
          <div className="rowList">
            {reports.length === 0 ? (
              <div className="dataRow">
                <div>
                  <strong>Waiting for first report</strong>
                  <span>Complete upload and run steps to generate report records.</span>
                </div>
                <div>
                  <Link href="/runs" className="btn btnGhost btnSm">
                    Open runs
                  </Link>
                </div>
              </div>
            ) : (
              reports.map((report) => (
                <div className="dataRow" key={report.id}>
                  <div>
                    <strong>
                      <Link href={`/reports/${report.id}`}>{report.report_type}</Link>
                    </strong>
                    <span>{report.storage_key}</span>
                  </div>
                  <div>
                    <strong>{report.is_published ? "Published" : "Draft"}</strong>
                    <span>
                      {report.run_id} - <Link href={`/reports/${report.id}`}>View details</Link>
                    </span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="listPanel">
          <div className="sectionHeader">
            <h2>Published reports</h2>
            <span>Ready for review</span>
          </div>
          <div className="rowList">
            {publishedReports.length === 0 ? (
              <div className="dataRow">
                <div>
                  <strong>No published reports</strong>
                  <span>Published outputs will appear as runs complete successfully.</span>
                </div>
              </div>
            ) : (
              publishedReports.map((report) => (
                <div className="dataRow" key={`${report.id}-published`}>
                  <div>
                    <strong>
                      <Link href={`/reports/${report.id}`}>{report.id}</Link>
                    </strong>
                    <span>{report.organization_id}</span>
                  </div>
                  <div>
                    <strong>{report.report_type}</strong>
                    <span>
                      {report.run_id} - <Link href={`/reports/${report.id}`}>Open</Link>
                    </span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="listPanel">
          <div className="sectionHeader">
            <h2>Linked runs</h2>
            <span>Report source</span>
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
