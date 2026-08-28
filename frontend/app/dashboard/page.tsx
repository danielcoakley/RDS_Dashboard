import Link from "next/link";
import { EnergyPerformanceChart } from "../../components/EnergyPerformanceChart";
import { WorkspaceShell } from "../../components/WorkspaceShell";
import { loadDashboardData } from "../../lib/dashboard-data";

const requiredFiles = [
  { key: "energy", label: "Energy data", hint: "Meter consumption CSV" },
  { key: "hdd", label: "HDD data", hint: "Heating degree days CSV" },
  { key: "cdd", label: "CDD data", hint: "Cooling degree days CSV" },
  { key: "seu_mapping", label: "SEU mapping", hint: "Meter to SEU category CSV" }
];

function statusCounts<T extends { status: string }>(items: T[]) {
  const statuses = ["queued", "running", "succeeded", "failed"];
  return statuses.map((status) => ({
    status,
    count: items.filter((item) => item.status === status).length
  }));
}

function formatDateTime(value: string): string {
  return new Intl.DateTimeFormat("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit"
  }).format(new Date(value));
}

export default async function DashboardPage() {
  const { mode, sites, meters, uploadReadiness, runs, analysisJobs, reports, auditEvents } =
    await loadDashboardData();
  const bestSourceSite = uploadReadiness.sites.reduce(
    (bestSite, site) => (site.ready_count > (bestSite?.ready_count ?? -1) ? site : bestSite),
    uploadReadiness.sites[0]
  );
  const requiredFileCount = uploadReadiness.required_categories.length || requiredFiles.length;
  const filesReady = bestSourceSite?.ready_count ?? 0;
  const latestRun = runs[0];
  const latestReport = reports.find((report) => report.run_id === latestRun?.id) ?? reports[0];
  const seuMeters = meters.filter((meter) => meter.is_seu);
  const hasWorkflowInputs = uploadReadiness.ready_site_count > 0;
  const sourceReadiness = requiredFiles.map((file) => ({
    label: file.label,
    ready: uploadReadiness.sites.some((site) =>
      site.files.some((sourceFile) => sourceFile.category === file.key && sourceFile.is_ready)
    ),
    storedCount: uploadReadiness.sites.filter((site) =>
      site.files.some((sourceFile) => sourceFile.category === file.key && sourceFile.is_ready)
    ).length
  }));
  const meterMix = Array.from(
    meters.reduce((groups, meter) => {
      const commodity = meter.commodity || "unknown";
      const key = commodity.toLowerCase();
      const current = groups.get(key) ?? {
        commodity: commodity.charAt(0).toUpperCase() + commodity.slice(1),
        total: 0,
        seu: 0
      };
      current.total += 1;
      current.seu += meter.is_seu ? 1 : 0;
      groups.set(key, current);
      return groups;
    }, new Map<string, { commodity: string; total: number; seu: number }>())
  ).map(([, value]) => value);

  return (
    <WorkspaceShell
      currentPath="/dashboard"
      title="Energy baseline workflow"
      eyebrow="Analysis"
      modeLabel={
        mode === "live" ? "Live workspace" : mode === "error" ? "Workspace data unavailable" : "Sample workflow"
      }
      modeDescription={
        mode === "live"
          ? "Use the same baseline workflow as the original dashboard: upload source files, choose years, run analysis, then review outputs."
          : mode === "error"
            ? "We could not load your organization's live data from the backend. Reload the page to try again."
            : "Sample data is shown until a live organization is selected."
      }
      modeTone={mode === "error" ? "error" : "info"}
    >
      {mode === "error" ? (
        <div className="authNotice authError" role="alert">
          <strong>Workspace data unavailable</strong>
          <span>
            Your real workspace data could not be loaded, so no sample or fabricated figures are
            shown below. Reload the page or try again shortly.
          </span>
        </div>
      ) : null}

      <section className="workflowHero" aria-label="Analysis workflow">
        <div>
          <h2>Run ISO 50001 energy analysis</h2>
          <p>
            Start with the four source files, confirm baseline settings, run the model, then review
            electricity, gas, and SEU outputs.
          </p>
        </div>
        <div className="workflowHeroActions">
          <Link href="/uploads" className="btn btnPrimary btnSm">
            Add source files
          </Link>
          <Link href="/runs" className="btn btnGhost btnSm">
            Run analysis
          </Link>
        </div>
      </section>

      <section className="summaryGrid" id="overview" aria-label="Workflow metrics">
        <div className="summaryTile">
          <span>Source files ready</span>
          <strong>
            {filesReady}/{requiredFileCount}
          </strong>
        </div>
        <div className="summaryTile">
          <span>Configured meters</span>
          <strong>{meters.length}</strong>
        </div>
        <div className="summaryTile">
          <span>SEU meters</span>
          <strong>{seuMeters.length}</strong>
        </div>
        <div className="summaryTile">
          <span>Current run</span>
          <strong>{latestRun?.status ?? "Not started"}</strong>
        </div>
        <div className="summaryTile">
          <span>Reports</span>
          <strong>{reports.length}</strong>
        </div>
      </section>

      <section className="workflowGrid">
        <div className="listPanel">
          <div className="sectionHeader">
            <h2>1. Required files</h2>
            <span>{hasWorkflowInputs ? `${uploadReadiness.ready_site_count} site ready` : "Complete the input set"}</span>
          </div>
          <div className="stepList">
            {requiredFiles.map((file) => {
              const sourceFile = bestSourceSite?.files.find((item) => item.category === file.key);
              const isReady = sourceFile?.is_ready ?? false;
              return (
                <div className="workflowStep" key={file.key}>
                  <span className={isReady ? "stepStatusReady" : "stepStatusPending"}>
                    {isReady ? "Ready" : "Needed"}
                  </span>
                  <div>
                    <strong>{file.label}</strong>
                    <span>{bestSourceSite ? bestSourceSite.site_name : file.hint}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="listPanel">
          <div className="sectionHeader">
            <h2>2. Analysis settings</h2>
            <span>Baseline comparison</span>
          </div>
          <div className="settingsGrid">
            <label className="authField">
              <span>Client config</span>
              <select defaultValue="rds_client.yaml">
                <option>rds_client.yaml</option>
                <option>example_client.yaml</option>
              </select>
            </label>
            <label className="authField">
              <span>Baseline year</span>
              <select defaultValue="2025">
                <option>2024</option>
                <option>2025</option>
              </select>
            </label>
            <label className="authField">
              <span>Comparison year</span>
              <select defaultValue="2026">
                <option>2025</option>
                <option>2026</option>
              </select>
            </label>
          </div>
        </div>

        <div className="listPanel">
          <div className="sectionHeader">
            <h2>3. Run status</h2>
            <span>{latestRun?.id ?? "No run yet"}</span>
          </div>
          <div className="dataRow">
            <div>
              <strong>{latestRun?.status ?? "Ready when files are complete"}</strong>
              <span>{latestRun?.error_message ?? latestRun?.site_id ?? "Upload the required files first"}</span>
            </div>
            <Link href="/runs" className="btn btnPrimary btnSm">
              Open runs
            </Link>
          </div>
        </div>
      </section>

      <section className="workflowTabs" aria-label="Analysis outputs">
        <a href="#summary">General Summary</a>
        <a href="#electricity">Electricity Analysis</a>
        <a href="#gas">Gas Analysis</a>
        <a href="#seu">SEU Analysis</a>
      </section>

      <section className="contentGrid">
        <div className="chartPanel wide" id="summary">
          <div className="sectionHeader">
            <h2>General summary</h2>
            <span>Tenant workflow visualization</span>
          </div>
          <EnergyPerformanceChart
            sourceReadiness={sourceReadiness}
            runStatuses={statusCounts(runs)}
            jobStatuses={statusCounts(analysisJobs)}
            meterMix={meterMix}
          />
        </div>

        <div className="listPanel" id="electricity">
          <div className="sectionHeader">
            <h2>Electricity analysis</h2>
            <span>Tracked electricity meters</span>
          </div>
          <div className="rowList">
            {meters
              .filter((meter) => meter.commodity.toLowerCase().includes("electric"))
              .map((meter) => (
                <div className="dataRow" key={meter.id}>
                  <div>
                    <strong>{meter.display_name}</strong>
                    <span>{meter.source_column}</span>
                  </div>
                  <div>
                    <strong>{meter.unit}</strong>
                    <span>{meter.is_seu ? "SEU" : "Monitor"}</span>
                  </div>
                </div>
              ))}
          </div>
        </div>

        <div className="listPanel" id="gas">
          <div className="sectionHeader">
            <h2>Gas analysis</h2>
            <span>Tracked gas meters</span>
          </div>
          <div className="rowList">
            {meters
              .filter((meter) => meter.commodity.toLowerCase().includes("gas"))
              .map((meter) => (
                <div className="dataRow" key={meter.id}>
                  <div>
                    <strong>{meter.display_name}</strong>
                    <span>{meter.source_column}</span>
                  </div>
                  <div>
                    <strong>{meter.unit}</strong>
                    <span>{meter.is_seu ? "SEU" : "Monitor"}</span>
                  </div>
                </div>
              ))}
          </div>
        </div>

        <div className="listPanel wide" id="seu">
          <div className="sectionHeader">
            <h2>SEU analysis</h2>
            <span>ISO 50001 significant energy users</span>
          </div>
          <div className="rowList">
            {seuMeters.map((meter) => (
              <div className="dataRow" key={meter.id}>
                <div>
                  <strong>{meter.display_name}</strong>
                  <span>
                    {meter.commodity} / {meter.source_column}
                  </span>
                </div>
                <div>
                  <strong>{meter.unit}</strong>
                  <span>{meter.site_id}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="listPanel wide">
          <div className="sectionHeader">
            <h2>Latest report</h2>
            <span>{latestReport ? "Generated output" : "No report generated"}</span>
          </div>
          <div className="dataRow">
            <div>
              <strong>{latestReport?.report_type ?? "Awaiting report"}</strong>
              <span>
                {latestReport
                  ? `Report ${latestReport.id} from run ${latestReport.run_id}`
                  : "Run analysis to create ISO summary output"}
              </span>
            </div>
            {latestReport ? (
              <Link href={`/reports/${latestReport.id}`} className="btn btnGhost btnSm">
                View report
              </Link>
            ) : (
              <Link href="/runs" className="btn btnGhost btnSm">
                Request run
              </Link>
            )}
          </div>
        </div>

        <div className="listPanel wide">
          <div className="sectionHeader">
            <h2>Audit trail</h2>
            <span>{auditEvents.length} workflow events</span>
          </div>
          <div className="rowList">
            {auditEvents.slice(0, 6).map((event) => (
              <div className="dataRow" key={event.id}>
                <div>
                  <strong>{event.action}</strong>
                  <span>
                    {event.resource_type} - {event.resource_id}
                  </span>
                </div>
                <div>
                  <strong>{event.actor_user_id}</strong>
                  <span>{formatDateTime(event.created_at)}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </WorkspaceShell>
  );
}
