import { revalidatePath } from "next/cache";
import Link from "next/link";
import { redirect } from "next/navigation";
import { WorkspaceShell } from "../../components/WorkspaceShell";
import { queueStoredUploadAnalysisRun } from "../../lib/api";
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

type SourcePackage = {
  encodedValue: string;
  uploadId: string;
  sourceUploadIds: string[];
  siteId: string;
  siteName: string;
  readyCount: number;
  requiredCount: number;
  isReady: boolean;
};

function runsMessage(
  status: string | undefined,
  error: string | undefined
): { tone: "success" | "error"; title: string; body: string } | null {
  if (status === "run-requested") {
    return {
      tone: "success",
      title: "Analysis queued",
      body: "The selected tenant upload was queued for analysis. Run history will update as processing completes."
    };
  }
  if (error === "missing-fields") {
    return {
      tone: "error",
      title: "Analysis not run",
      body: "Choose a site source package with stored energy data before starting analysis."
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
      body: "We could not queue that run. Check the selected source file and try again."
    };
  }
  return null;
}

function sourcePackageValue(uploadId: string, sourceUploadIds: string[]): string {
  return JSON.stringify({ upload_id: uploadId, source_upload_ids: sourceUploadIds });
}

function notificationDeliveryDetail(
  notification: Awaited<ReturnType<typeof loadDashboardData>>["notifications"][number]
): string {
  if (notification.next_attempt_at) {
    return `Retry ${notification.next_attempt_at}`;
  }
  if (notification.sent_at) {
    return `Sent ${notification.sent_at}`;
  }
  if (notification.last_attempt_at) {
    return `Attempted ${notification.last_attempt_at}`;
  }
  return notification.channel;
}

function buildSourcePackages(
  uploadReadiness: Awaited<ReturnType<typeof loadDashboardData>>["uploadReadiness"]
): SourcePackage[] {
  return uploadReadiness.sites.flatMap((site) => {
    const energyFile = site.files.find(
      (file) => file.category === "energy" && file.latest_upload !== null
    );
    if (!energyFile?.latest_upload) {
      return [];
    }
    const sourceUploadIds = site.files
      .filter((file) => file.category !== "energy" && file.latest_upload !== null)
      .map((file) => file.latest_upload?.id)
      .filter((uploadId): uploadId is string => Boolean(uploadId));
    return [
      {
        encodedValue: sourcePackageValue(energyFile.latest_upload.id, sourceUploadIds),
        uploadId: energyFile.latest_upload.id,
        sourceUploadIds,
        siteId: site.site_id,
        siteName: site.site_name,
        readyCount: site.ready_count,
        requiredCount: site.required_count,
        isReady: site.is_ready
      }
    ];
  });
}

export default async function RunsPage({ searchParams }: RunsPageProps) {
  const query = await searchParams;
  const { mode, runs, analysisJobs, notifications, reports, uploadReadiness, sites } = await loadDashboardData();
  const sourcePackages = buildSourcePackages(uploadReadiness);
  const openJobs = analysisJobs.filter((job) => job.status === "queued" || job.status === "running");
  const pendingNotifications = notifications.filter((notification) => notification.status === "pending");
  const requiredFileCount = uploadReadiness.required_categories.length || requiredFiles.length;
  const filesReady = uploadReadiness.sites.reduce(
    (bestReadyCount, site) => Math.max(bestReadyCount, site.ready_count),
    0
  );
  const canRun = mode === "live" && sourcePackages.length > 0;
  const latestRun = runs[0];
  const pageMessage = runsMessage(query.status, query.error);

  async function requestRunAction(formData: FormData) {
    "use server";

    const sourcePackage = String(formData.get("source_package") ?? "").trim();
    const session = await readAppSession();
    let selectedPackage: { upload_id?: unknown; source_upload_ids?: unknown } = {};

    try {
      selectedPackage = JSON.parse(sourcePackage);
    } catch {
      redirect("/runs?error=missing-fields");
    }
    const uploadId =
      typeof selectedPackage.upload_id === "string" ? selectedPackage.upload_id.trim() : "";
    const sourceUploadIds = Array.isArray(selectedPackage.source_upload_ids)
      ? selectedPackage.source_upload_ids
          .filter((sourceUploadId): sourceUploadId is string => typeof sourceUploadId === "string")
          .map((sourceUploadId) => sourceUploadId.trim())
          .filter((sourceUploadId) => sourceUploadId && sourceUploadId !== uploadId)
      : [];

    if (!uploadId) {
      redirect("/runs?error=missing-fields");
    }
    if (!session.userId || !session.organizationId) {
      redirect("/runs?error=session-missing");
    }

    const runId = `run_${Date.now()}`;
    const reportId = `report_${Date.now()}`;
    try {
      await queueStoredUploadAnalysisRun(
        session.userId,
        session.organizationId,
        {
          upload_id: uploadId,
          run_id: runId,
          report_id: reportId,
          source_upload_ids: sourceUploadIds
        },
        session.authToken
      );
      revalidatePath("/dashboard");
      revalidatePath("/runs");
      revalidatePath("/reports");
    } catch {
      redirect("/runs?error=run-failed");
    }
    redirect("/runs?status=run-requested");
  }

  return (
    <WorkspaceShell
      currentPath="/runs"
      title="Run analysis"
      eyebrow="Workflow"
      modeLabel={
        mode === "live" ? "Live workspace" : mode === "error" ? "Workspace data unavailable" : "Sample workflow"
      }
      modeDescription={
        mode === "error"
          ? "We could not load your organization's live data from the backend. Reload the page to try again."
          : "Start the baseline analysis once the source files and site are ready."
      }
      modeTone={mode === "error" ? "error" : "info"}
    >
      {mode === "error" ? (
        <div className="authNotice authError" role="alert">
          <strong>Workspace data unavailable</strong>
          <span>
            Your real run history and job data could not be loaded, so no sample or fabricated
            figures are shown below. Reload the page or try again shortly.
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

      <section className="summaryGrid" aria-label="Run metrics">
        <div className="summaryTile">
          <span>Source files</span>
          <strong>
            {filesReady}/{requiredFileCount}
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
          <span>Open jobs</span>
          <strong>{openJobs.length}</strong>
        </div>
        <div className="summaryTile">
          <span>Email notices</span>
          <strong>{pendingNotifications.length}</strong>
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
              <span>Upload an energy CSV in Source files before starting analysis.</span>
            </div>
          ) : null}
          <form action={requestRunAction} className="inlineForm">
            <label className="authField">
              <span>Site source package</span>
              <select
                name="source_package"
                defaultValue={sourcePackages[0]?.encodedValue ?? ""}
                disabled={!canRun}
              >
                {sourcePackages.map((sourcePackage) => (
                  <option key={sourcePackage.uploadId} value={sourcePackage.encodedValue}>
                    {sourcePackage.siteName} - {sourcePackage.readyCount}/{sourcePackage.requiredCount} files
                  </option>
                ))}
              </select>
            </label>
            <label className="authField">
              <span>Primary energy upload</span>
              <select name="upload_preview" defaultValue={sourcePackages[0]?.uploadId ?? ""} disabled>
                {sourcePackages.map((sourcePackage) => (
                  <option key={`${sourcePackage.uploadId}-key`} value={sourcePackage.uploadId}>
                    {sourcePackage.uploadId} - {sourcePackage.sourceUploadIds.length + 1} source files
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
            <span>Original dashboard inputs by site</span>
          </div>
          <div className="stepList">
            {uploadReadiness.sites.length === 0 ? (
              <div className="workflowStep">
                <span className="stepStatusPending">Needed</span>
                <div>
                  <strong>No sites configured</strong>
                  <span>Add a site before running analysis.</span>
                </div>
              </div>
            ) : (
              uploadReadiness.sites.map((site) => (
                <div className="workflowStep" key={site.site_id}>
                  <span className={site.is_ready ? "stepStatusReady" : "stepStatusPending"}>
                    {site.ready_count === site.required_count ? "Ready" : "Needed"}
                  </span>
                  <div>
                    <strong>{site.site_name}</strong>
                    <span>
                      {site.ready_count}/{site.required_count} source files recorded
                    </span>
                    <div className="readinessFileList" aria-label={`${site.site_name} source files`}>
                      {site.files.map((file) => (
                        <span
                          className={file.is_ready ? "stepStatusReady" : "stepStatusPending"}
                          key={`${site.site_id}-${file.category}`}
                        >
                          {file.label}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="listPanel">
          <div className="sectionHeader">
            <h2>Job queue</h2>
            <span>Background analysis lifecycle</span>
          </div>
          <div className="rowList">
            {analysisJobs.length === 0 ? (
              <div className="dataRow">
                <div>
                  <strong>No queued jobs</strong>
                  <span>Queued analysis jobs appear here after a run starts.</span>
                </div>
              </div>
            ) : (
              analysisJobs.map((job) => (
                <div className="dataRow" key={job.id}>
                  <div>
                    <strong>{job.id}</strong>
                    <span>
                      {job.upload_id} - {job.report_id}
                    </span>
                  </div>
                  <div>
                    <strong>{job.status}</strong>
                    <span>{job.completed_at ?? job.error_message ?? job.run_id}</span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="listPanel">
          <div className="sectionHeader">
            <h2>Notifications</h2>
            <span>Email notification outbox</span>
          </div>
          <div className="rowList">
            {notifications.length === 0 ? (
              <div className="dataRow">
                <div>
                  <strong>No notifications yet</strong>
                  <span>Workflow emails are queued after runs and report approvals change.</span>
                </div>
              </div>
            ) : (
              notifications.map((notification) => (
                <div className="dataRow" key={notification.id}>
                  <div>
                    <strong>{notification.subject}</strong>
                    <span>{notification.body}</span>
                  </div>
                  <div>
                    <strong>
                      {notification.status} - {notification.attempt_count} attempts
                    </strong>
                    <span>{notification.error_message ?? notificationDeliveryDetail(notification)}</span>
                  </div>
                </div>
              ))
            )}
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
