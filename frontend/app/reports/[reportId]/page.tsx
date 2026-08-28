import Link from "next/link";
import { notFound } from "next/navigation";
import { WorkspaceShell } from "../../../components/WorkspaceShell";
import { loadDashboardData } from "../../../lib/dashboard-data";

type ReportDetailPageProps = {
  params: Promise<{ reportId: string }>;
};

export default async function ReportDetailPage({ params }: ReportDetailPageProps) {
  const { reportId } = await params;
  const { mode, reports, runs, sites } = await loadDashboardData();
  const report = reports.find((item) => item.id === reportId);

  if (!report) {
    notFound();
  }

  const run = runs.find((item) => item.id === report.run_id) ?? null;
  const site = run ? sites.find((item) => item.id === run.site_id) ?? null : null;

  return (
    <WorkspaceShell
      currentPath="/reports"
      title={`Report ${report.id}`}
      modeLabel={mode === "live" ? "Live workspace" : "Sample workspace"}
      modeDescription={
        mode === "live"
          ? "Report metadata and lineage are loading from the backend."
          : "This report is shown from sample data until a live organization is selected."
      }
    >
      <section className="summaryGrid" aria-label="Report summary">
        <div className="summaryTile">
          <span>Status</span>
          <strong>{report.is_published ? "Published" : "Draft"}</strong>
        </div>
        <div className="summaryTile">
          <span>Report type</span>
          <strong>{report.report_type}</strong>
        </div>
        <div className="summaryTile">
          <span>Run</span>
          <strong>{report.run_id}</strong>
        </div>
        <div className="summaryTile">
          <span>Site</span>
          <strong>{site?.name ?? run?.site_id ?? "Unknown"}</strong>
        </div>
      </section>

      <section className="contentGrid">
        <div className="listPanel wide">
          <div className="sectionHeader">
            <h2>Storage and metadata</h2>
            <span>Canonical identifiers for this report</span>
          </div>
          <div className="rowList">
            <div className="dataRow">
              <div>
                <strong>Report ID</strong>
                <span>{report.id}</span>
              </div>
              <div>
                <strong>Organization</strong>
                <span>{report.organization_id}</span>
              </div>
            </div>
            <div className="dataRow">
              <div>
                <strong>Storage key</strong>
                <span>{report.storage_key}</span>
              </div>
              <div>
                <strong>Run link</strong>
                <span>{report.run_id}</span>
              </div>
            </div>
          </div>
        </div>

        <div className="listPanel">
          <div className="sectionHeader">
            <h2>Run lineage</h2>
            <span>Execution source for this report</span>
          </div>
          <div className="rowList">
            <div className="dataRow">
              <div>
                <strong>Run status</strong>
                <span>{run?.status ?? "Unknown"}</span>
              </div>
              <div>
                <strong>Completion</strong>
                <span>{run?.completed_at ?? run?.error_message ?? "Awaiting completion"}</span>
              </div>
            </div>
            <div className="dataRow">
              <div>
                <strong>Requested by</strong>
                <span>{run?.requested_by_user_id ?? "Unknown"}</span>
              </div>
              <div>
                <strong>Site ID</strong>
                <span>{run?.site_id ?? "Unknown"}</span>
              </div>
            </div>
          </div>
        </div>

        <div className="listPanel">
          <div className="sectionHeader">
            <h2>Navigation</h2>
            <span>Continue workflow</span>
          </div>
          <div className="rowList">
            <div className="dataRow">
              <div>
                <strong>Back to reports</strong>
                <span>View the full report library</span>
              </div>
              <div>
                <Link href="/reports" className="btn btnGhost btnSm">
                  Open reports
                </Link>
              </div>
            </div>
            <div className="dataRow">
              <div>
                <strong>Open runs</strong>
                <span>Review run queue and statuses</span>
              </div>
              <div>
                <Link href="/runs" className="btn btnGhost btnSm">
                  Open runs
                </Link>
              </div>
            </div>
          </div>
        </div>
      </section>
    </WorkspaceShell>
  );
}
