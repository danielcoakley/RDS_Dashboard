import Link from "next/link";
import { redirect } from "next/navigation";
import { revalidatePath } from "next/cache";
import { EnergyPerformanceChart } from "../../components/EnergyPerformanceChart";
import { WorkspaceShell } from "../../components/WorkspaceShell";
import { createOrganizationUploadContent, executeEnergyUploadAnalysisRun } from "../../lib/api";
import { loadDashboardData } from "../../lib/dashboard-data";
import { readAppSession } from "../../lib/session";

const sourceReadinessCategories = [
  { key: "energy", label: "Energy data" },
  { key: "hdd", label: "HDD data (auto)" },
  { key: "cdd", label: "CDD data (auto)" },
  { key: "seu_mapping", label: "SEU mapping" }
];

type WeatherCoverage = {
  record_count?: number;
  matched_energy_record_count?: number;
};

type WeatherNormalizationModel = {
  meter: string;
  utility: string;
  normalization: string;
  estimated_savings: number;
};

type WeatherNormalization = {
  status: string;
  estimated_savings?: number;
  model_count?: number;
  models?: WeatherNormalizationModel[];
};

type LegacyBaseline = {
  gas_record_count?: number;
  electricity_record_count?: number;
  weather_coverage?: { hdd?: WeatherCoverage; cdd?: WeatherCoverage };
  weather_normalization?: WeatherNormalization;
};

type RunResult = {
  client?: string;
  site?: string;
  total_records?: number;
  seu_total_consumption?: number;
  legacy_baseline?: LegacyBaseline;
};

type DashboardPageProps = {
  searchParams: Promise<{ status?: string; error?: string; result?: string }>;
};

function decodeRunResult(encoded: string | undefined): RunResult | null {
  if (!encoded) {
    return null;
  }
  try {
    return JSON.parse(Buffer.from(encoded, "base64url").toString("utf-8")) as RunResult;
  } catch {
    return null;
  }
}

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

function runMessage(
  status: string | undefined,
  error: string | undefined
): { tone: "success" | "error"; title: string; body: string } | null {
  if (status === "run-complete") {
    return {
      tone: "success",
      title: "Analysis complete",
      body: "Results are shown below. HDD/CDD weather data is fetched automatically from the site's location."
    };
  }
  if (error === "missing-fields") {
    return {
      tone: "error",
      title: "Analysis not run",
      body: "Choose a site and an energy CSV before running analysis."
    };
  }
  if (error === "session-missing") {
    return {
      tone: "error",
      title: "Analysis not run",
      body: "Sign in to an organization workspace before running analysis."
    };
  }
  if (error === "run-failed") {
    return {
      tone: "error",
      title: "Analysis not run",
      body: "We could not run that analysis. Check the CSV format and try again."
    };
  }
  return null;
}

export default async function DashboardPage({ searchParams }: DashboardPageProps) {
  const query = await searchParams;
  const { mode, sites, meters, uploadReadiness, runs, analysisJobs, reports, auditEvents } =
    await loadDashboardData();
  const canRun = mode === "live" && sites.length > 0;
  const pageMessage = runMessage(query.status, query.error);
  const result = decodeRunResult(query.result);
  const seuMeters = meters.filter((meter) => meter.is_seu);
  const sourceReadiness = sourceReadinessCategories.map((file) => ({
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
  const weatherCoverage = result?.legacy_baseline?.weather_coverage;
  const weatherNormalization = result?.legacy_baseline?.weather_normalization;

  async function requestRunAction(formData: FormData) {
    "use server";

    const siteId = String(formData.get("site_id") ?? "").trim();
    const energyFile = formData.get("energy_file");
    const seuFile = formData.get("seu_mapping_file");
    const session = await readAppSession();

    if (!siteId || !(energyFile instanceof File) || energyFile.size === 0) {
      redirect("/dashboard?error=missing-fields");
    }
    if (!session.userId || !session.organizationId) {
      redirect("/dashboard?error=session-missing");
    }

    const runId = `run_${Date.now()}`;
    const uploadId = `upload_${Date.now()}`;
    const reportId = `report_${Date.now()}`;
    let encodedResult: string;
    try {
      let seuMappingUploadId: string | undefined;
      if (seuFile instanceof File && seuFile.size > 0) {
        seuMappingUploadId = `upload_seu_${Date.now()}`;
        await createOrganizationUploadContent(
          session.userId,
          session.organizationId,
          {
            upload_id: seuMappingUploadId,
            site_id: siteId,
            category: "seu_mapping",
            filename: seuFile.name,
            content_base64: Buffer.from(await seuFile.arrayBuffer()).toString("base64")
          },
          session.authToken
        );
      }

      const response = await executeEnergyUploadAnalysisRun(
        session.userId,
        session.organizationId,
        {
          site_id: siteId,
          upload_id: uploadId,
          filename: energyFile.name,
          content_base64: Buffer.from(await energyFile.arrayBuffer()).toString("base64"),
          run_id: runId,
          report_id: reportId,
          seu_mapping_upload_id: seuMappingUploadId ?? null
        },
        session.authToken
      );
      encodedResult = Buffer.from(JSON.stringify(response.iso_summary)).toString("base64url");
    } catch {
      redirect("/dashboard?error=run-failed");
    }
    revalidatePath("/dashboard");
    revalidatePath("/reports");
    redirect(`/dashboard?status=run-complete&result=${encodedResult}`);
  }

  return (
    <WorkspaceShell
      currentPath="/dashboard"
      title="Analyze"
      eyebrow="Energy baseline"
      modeLabel={
        mode === "live" ? "Live workspace" : mode === "error" ? "Workspace data unavailable" : "Sample workflow"
      }
      modeDescription={
        mode === "live"
          ? "Upload energy data and run the baseline analysis - HDD/CDD weather data is fetched automatically from the site's location."
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

      {pageMessage ? (
        <div
          className={`authNotice ${pageMessage.tone === "success" ? "authSuccess" : "authError"}`}
          role={pageMessage.tone === "success" ? "status" : "alert"}
        >
          <strong>{pageMessage.title}</strong>
          <span>{pageMessage.body}</span>
        </div>
      ) : null}

      <section className="contentGrid">
        <div className="listPanel wide">
          <div className="sectionHeader">
            <h2>Run analysis</h2>
            <span>Upload energy data for a site and run the baseline model</span>
          </div>
          {!canRun ? (
            <div className="authNotice authError" role="alert">
              <strong>Add a site first</strong>
              <span>
                Go to <Link href="/settings">Setup</Link> to add a site with a location - HDD/CDD are
                then fetched automatically for every run.
              </span>
            </div>
          ) : null}
          <form action={requestRunAction} className="inlineForm">
            <label className="authField">
              <span>Site</span>
              <select name="site_id" defaultValue={sites[0]?.id ?? ""} disabled={!canRun}>
                {sites.map((site) => (
                  <option key={site.id} value={site.id}>
                    {site.name}
                    {site.location_label ? ` - ${site.location_label}` : " - no location set"}
                  </option>
                ))}
              </select>
            </label>
            <label className="authField">
              <span>Energy CSV</span>
              <input name="energy_file" type="file" accept=".csv,text/csv" disabled={!canRun} />
            </label>
            <label className="authField">
              <span>SEU mapping CSV (optional)</span>
              <input name="seu_mapping_file" type="file" accept=".csv,text/csv" disabled={!canRun} />
            </label>
            <button type="submit" className="btn btnPrimary btnSm" disabled={!canRun}>
              Run analysis
            </button>
          </form>
        </div>

        {result ? (
          <>
            <div className="listPanel wide" id="summary">
              <div className="sectionHeader">
                <h2>General summary</h2>
                <span>
                  {result.client ?? "Workspace"} - {result.site ?? "Site"}
                </span>
              </div>
              <div className="summaryGrid" aria-label="Run result">
                <div className="summaryTile">
                  <span>Records analysed</span>
                  <strong>{result.total_records ?? 0}</strong>
                </div>
                <div className="summaryTile">
                  <span>SEU consumption</span>
                  <strong>{result.seu_total_consumption ?? 0}</strong>
                </div>
                {result.legacy_baseline ? (
                  <>
                    <div className="summaryTile">
                      <span>Gas records</span>
                      <strong>{result.legacy_baseline.gas_record_count ?? 0}</strong>
                    </div>
                    <div className="summaryTile">
                      <span>Electricity records</span>
                      <strong>{result.legacy_baseline.electricity_record_count ?? 0}</strong>
                    </div>
                  </>
                ) : null}
              </div>
            </div>

            <div className="listPanel wide" id="weather">
              <div className="sectionHeader">
                <h2>Automated HDD/CDD</h2>
                <span>Fetched from the site's location for this run's date range</span>
              </div>
              {weatherCoverage ? (
                <div className="rowList">
                  <div className="dataRow">
                    <div>
                      <strong>Heating degree days</strong>
                      <span>{weatherCoverage.hdd?.record_count ?? 0} days fetched</span>
                    </div>
                    <div>
                      <strong>Cooling degree days</strong>
                      <span>{weatherCoverage.cdd?.record_count ?? 0} days fetched</span>
                    </div>
                  </div>
                  {weatherNormalization ? (
                    <div className="dataRow">
                      <div>
                        <strong>Weather-normalized model</strong>
                        <span>
                          {weatherNormalization.status} - {weatherNormalization.model_count ?? 0} meters modeled
                        </span>
                      </div>
                      <div>
                        <strong>Estimated savings</strong>
                        <span>{weatherNormalization.estimated_savings ?? 0}</span>
                      </div>
                    </div>
                  ) : null}
                </div>
              ) : (
                <div className="authNotice authError" role="status">
                  <strong>No weather data for this run</strong>
                  <span>
                    Set a location on this site in Setup, and upload an energy CSV in the original
                    dashboard export format, to get automatic HDD/CDD and weather-normalized savings.
                  </span>
                </div>
              )}
            </div>
          </>
        ) : null}

        <div className="chartPanel wide" id="charts">
          <div className="sectionHeader">
            <h2>Workspace overview</h2>
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
            <h2>Electricity meters</h2>
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
            <h2>Gas meters</h2>
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
            <h2>SEU meters</h2>
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
            <h2>Recent runs and reports</h2>
            <span>{runs.length} runs - {reports.length} reports</span>
          </div>
          <div className="rowList">
            {runs.length === 0 ? (
              <div className="dataRow">
                <div>
                  <strong>No runs yet</strong>
                  <span>Run analysis above to see history here.</span>
                </div>
              </div>
            ) : (
              runs.slice(0, 5).map((run) => (
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
              ))
            )}
          </div>
          <div className="toolbarRow">
            <Link href="/reports" className="btn btnGhost btnSm">
              Open reports
            </Link>
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
