import { revalidatePath } from "next/cache";
import Link from "next/link";
import { redirect } from "next/navigation";
import { WorkspaceShell } from "../../components/WorkspaceShell";
import { createOrganizationRun } from "../../lib/api";
import { loadDashboardData } from "../../lib/dashboard-data";
import { readAppSession } from "../../lib/session";

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
      title: "Run requested",
      body: "The run was queued and now appears in run history."
    };
  }
  if (error === "missing-fields") {
    return {
      tone: "error",
      title: "Run not requested",
      body: "Choose a site and upload before requesting a run."
    };
  }
  if (error === "run-failed") {
    return {
      tone: "error",
      title: "Run not requested",
      body: "We could not queue this run. Confirm tenant access and upload selection."
    };
  }
  if (error === "session-missing") {
    return {
      tone: "error",
      title: "Run not requested",
      body: "Sign in to a tenant workspace before requesting runs."
    };
  }
  return null;
}

export default async function RunsPage({ searchParams }: RunsPageProps) {
  const query = await searchParams;
  const { mode, runs, reports, uploads, sites, auditEvents } = await loadDashboardData();
  const canRequestRun = mode === "live" && sites.length > 0 && uploads.length > 0;
  const successfulRuns = runs.filter((run) => run.status === "succeeded");
  const failedRuns = runs.filter((run) => run.status === "failed");
  const runAuditEvents = auditEvents.filter((event) => event.resource_type === "run");
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
      revalidatePath("/runs");
      revalidatePath("/reports");
      revalidatePath("/dashboard");
      redirect("/runs?status=run-requested");
    } catch {
      redirect("/runs?error=run-failed");
    }
  }

  return (
    <WorkspaceShell
      currentPath="/runs"
      title="Run history"
      modeLabel={mode === "live" ? "Live workspace" : "Sample workspace"}
      modeDescription={
        mode === "live"
          ? "Run status and outputs are loading from the backend."
          : "This page is showing sample run history until a live organization is selected."
      }
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
      {!canRequestRun ? (
        <div className="authNotice authError" role="alert">
          <strong>Run requests need an active tenant workspace</strong>
          <span>
            Select an organization before requesting runs. If you are not signed in, complete{" "}
            <Link href="/login">sign in</Link> first.
          </span>
        </div>
      ) : null}

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
            <h2>Run history</h2>
            <span>Status and completion</span>
          </div>
          <form action={requestRunAction} className="inlineForm">
            <label className="authField">
              <span>Site</span>
              <select name="site_id" defaultValue={sites[0]?.id ?? ""} disabled={!canRequestRun}>
                {sites.map((site) => (
                  <option key={site.id} value={site.id}>
                    {site.name}
                  </option>
                ))}
              </select>
            </label>
            <label className="authField">
              <span>Upload</span>
              <select name="upload_id" defaultValue={uploads[0]?.id ?? ""} disabled={!canRequestRun}>
                {uploads.map((upload) => (
                  <option key={upload.id} value={upload.id}>
                    {upload.id}
                  </option>
                ))}
              </select>
            </label>
            <button type="submit" className="btn btnPrimary btnSm" disabled={!canRequestRun}>
              Request run
            </button>
          </form>
          <div className="rowList">
            {runs.map((run) => (
              <div className="dataRow" key={run.id}>
                <div>
                  <strong>{run.id}</strong>
                  <span>
                    {run.site_id} · {reports.filter((report) => report.run_id === run.id).length} reports
                  </span>
                </div>
                <div>
                  <strong>{run.status}</strong>
                  <span>
                    {run.completed_at ??
                      reports.find((report) => report.run_id === run.id)?.id ??
                      run.error_message ??
                      "Awaiting completion"}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="listPanel">
          <div className="sectionHeader">
            <h2>Report outputs</h2>
            <span>Generated files</span>
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
            <span>Audit events</span>
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
