import { revalidatePath } from "next/cache";
import Link from "next/link";
import { redirect } from "next/navigation";
import { WorkspaceShell } from "../../components/WorkspaceShell";
import { createOrganizationRun } from "../../lib/api";
import { loadDashboardData } from "../../lib/dashboard-data";
import { readAppSession } from "../../lib/session";

const requiredFiles = [
  { key: "energy", label: "Energy data" },
  { key: "hdd", label: "HDD data" },
  { key: "cdd", label: "CDD data" },
  { key: "seu_mapping", label: "SEU mapping" }
];

type RunsPageProps = {
  searchParams: Promise<{ status?: string; error?: string }>;
};

function runsMessage(
  status: string | undefined,
  error: string | undefined
): { tone: "success" | "error"; title: string; body: string } | null {
  if (status === "run-requested") {
    return {
      tone: "success",
      title: "Analysis queued",
      body: "The run is now in the history list."
    };
  }
  if (error === "missing-fields") {
    return {
      tone: "error",
      title: "Analysis not queued",
      body: "Choose a site and an energy upload before starting analysis."
    };
  }
  if (error === "session-missing") {
    return {
      tone: "error",
      title: "Analysis not queued",
      body: "Sign in to an organization workspace before running analysis."
    };
  }
  if (error === "run-failed") {
    return {
      tone: "error",
      title: "Analysis not queued",
      body: "We could not queue that run. Check the selected source file and try again."
    };
  }
  return null;
}

function hasUploadCategory(uploadCategories: Set<string>, key: string): boolean {
  return uploadCategories.has(key) || uploadCategories.has(key.replace("_", "-"));
}

export default async function RunsPage({ searchParams }: RunsPageProps) {
  const query = await searchParams;
  const { mode, runs, reports, uploads, sites } = await loadDashboardData();
  const uploadCategories = new Set(uploads.map((upload) => upload.category.toLowerCase()));
  const filesReady = requiredFiles.filter((file) => hasUploadCategory(uploadCategories, file.key)).length;
  const energyUploads = uploads.filter((upload) => upload.category.toLowerCase() === "energy");
  const canRun = mode === "live" && sites.length > 0 && energyUploads.length > 0;
  const latestRun = runs[0];
  const pageMessage = runsMessage(query.status, query.error);

  async function requestRunAction(formData: FormData) {
    "use server";

    const siteId = String(formData.get("site_id") ?? "").trim();
    const uploadId = String(formData.get("upload_id") ?? "").trim();
    const session = await readAppSession();

    if (!siteId || !uploadId) {
      redirect("/runs?error=missing-fields");
    }
    if (!session.userId || !session.organizationId) {
      redirect("/runs?error=session-missing");
    }

    const runId = `run_${Date.now()}`;
    try {
      await createOrganizationRun(
        session.userId,
        session.organizationId,
        {
          run_id: runId,
          site_id: siteId,
          upload_ids: [uploadId]
        },
        session.authToken
      );
      revalidatePath("/dashboard");
      revalidatePath("/runs");
      revalidatePath("/reports");
      redirect("/runs?status=run-requested");
    } catch {
      redirect("/runs?error=run-failed");
    }
  }

  return (
    <WorkspaceShell
      currentPath="/runs"
      title="Run analysis"
      eyebrow="Workflow"
      modeLabel={mode === "live" ? "Live workspace" : "Sample workflow"}
      modeDescription="Start the baseline analysis once the source files and site are ready."
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

      <section className="summaryGrid" aria-label="Run metrics">
        <div className="summaryTile">
          <span>Source files</span>
          <strong>
            {filesReady}/{requiredFiles.length}
          </strong>
        </div>
        <div className="summaryTile">
          <span>Sites</span>
          <strong>{sites.length}</strong>
        </div>
        <div className="summaryTile">
          <span>Runs</span>
          <strong>{runs.length}</strong>
        </div>
        <div className="summaryTile">
          <span>Latest status</span>
          <strong>{latestRun?.status ?? "Not started"}</strong>
        </div>
      </section>

      <section className="contentGrid">
        <div className="listPanel wide">
          <div className="sectionHeader">
            <h2>Start analysis</h2>
            <span>Baseline and comparison outputs</span>
          </div>
          {!canRun ? (
            <div className="authNotice authError" role="alert">
              <strong>Inputs needed</strong>
              <span>Add a site and at least one energy data source file before running analysis.</span>
            </div>
          ) : null}
          <form action={requestRunAction} className="inlineForm">
            <label className="authField">
              <span>Site</span>
              <select name="site_id" defaultValue={sites[0]?.id ?? ""} disabled={!canRun}>
                {sites.map((site) => (
                  <option key={site.id} value={site.id}>
                    {site.name}
                  </option>
                ))}
              </select>
            </label>
            <label className="authField">
              <span>Energy data</span>
              <select name="upload_id" defaultValue={energyUploads[0]?.id ?? ""} disabled={!canRun}>
                {energyUploads.map((upload) => (
                  <option key={upload.id} value={upload.id}>
                    {upload.id}
                  </option>
                ))}
              </select>
            </label>
            <button type="submit" className="btn btnPrimary btnSm" disabled={!canRun}>
              Run analysis
            </button>
          </form>
        </div>

        <div className="listPanel">
          <div className="sectionHeader">
            <h2>Source file readiness</h2>
            <span>Original dashboard inputs</span>
          </div>
          <div className="stepList">
            {requiredFiles.map((file) => {
              const isReady = hasUploadCategory(uploadCategories, file.key);
              return (
                <div className="workflowStep" key={file.key}>
                  <span className={isReady ? "stepStatusReady" : "stepStatusPending"}>
                    {isReady ? "Ready" : "Needed"}
                  </span>
                  <div>
                    <strong>{file.label}</strong>
                    <span>{isReady ? "Recorded" : "Add from Source files"}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="listPanel">
          <div className="sectionHeader">
            <h2>Generated reports</h2>
            <span>Analysis outputs</span>
          </div>
          <div className="rowList">
            {reports.length === 0 ? (
              <div className="dataRow">
                <div>
                  <strong>No reports yet</strong>
                  <span>Run analysis to generate report artifacts.</span>
                </div>
              </div>
            ) : (
              reports.map((report) => (
                <div className="dataRow" key={report.id}>
                  <div>
                    <strong>{report.report_type}</strong>
                    <span>{report.run_id}</span>
                  </div>
                  <Link href={`/reports/${report.id}`} className="btn btnGhost btnSm">
                    Open
                  </Link>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="listPanel wide">
          <div className="sectionHeader">
            <h2>Run history</h2>
            <span>Status and completion</span>
          </div>
          <div className="rowList">
            {runs.length === 0 ? (
              <div className="dataRow">
                <div>
                  <strong>No runs yet</strong>
                  <span>Complete source files, then start the first analysis run.</span>
                </div>
              </div>
            ) : (
              runs.map((run) => (
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
        </div>
      </section>
    </WorkspaceShell>
  );
}
