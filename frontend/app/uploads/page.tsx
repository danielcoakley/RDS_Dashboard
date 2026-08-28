import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import { WorkspaceShell } from "../../components/WorkspaceShell";
import { createOrganizationUploadContent } from "../../lib/api";
import { loadDashboardData } from "../../lib/dashboard-data";
import { readAppSession } from "../../lib/session";

const sourceFileTypes = [
  { value: "energy", label: "Energy Data" },
  { value: "hdd", label: "HDD Data" },
  { value: "cdd", label: "CDD Data" },
  { value: "seu_mapping", label: "SEU Mapping" }
];

function sourceFileLabel(category: string): string {
  return sourceFileTypes.find((fileType) => fileType.value === category)?.label ?? category;
}

function shortChecksum(value: string): string {
  return value.length > 12 ? `${value.slice(0, 12)}...` : value;
}

type UploadsPageProps = {
  searchParams: Promise<{ status?: string; error?: string }>;
};

function uploadsMessage(
  status: string | undefined,
  error: string | undefined
): { tone: "success" | "error"; title: string; body: string } | null {
  if (status === "created") {
    return {
      tone: "success",
      title: "Source file uploaded",
      body: "The file is stored in the tenant workspace and included in the baseline workflow checklist."
    };
  }
  if (error === "missing-fields") {
    return {
      tone: "error",
      title: "Source file not uploaded",
      body: "Choose a site, source file type, and CSV file."
    };
  }
  if (error === "session-missing") {
    return {
      tone: "error",
      title: "Source file not uploaded",
      body: "Sign in to an organization workspace before adding source files."
    };
  }
  if (error === "create-failed") {
    return {
      tone: "error",
      title: "Source file not uploaded",
      body: "We could not store that file. Check that it is a CSV matching the selected source type."
    };
  }
  return null;
}

export default async function UploadsPage({ searchParams }: UploadsPageProps) {
  const query = await searchParams;
  const { mode, sites, uploads, uploadReadiness } = await loadDashboardData();
  const storedUploads = uploads.filter((upload) => upload.status === "stored");
  const pageMessage = uploadsMessage(query.status, query.error);

  async function createUploadAction(formData: FormData) {
    "use server";

    const siteId = String(formData.get("site_id") ?? "").trim();
    const category = String(formData.get("category") ?? "").trim();
    const sourceFile = formData.get("source_file");
    const session = await readAppSession();

    if (!siteId || !category || !(sourceFile instanceof File) || sourceFile.size === 0) {
      redirect("/uploads?error=missing-fields");
    }
    if (!session.userId || !session.organizationId) {
      redirect("/uploads?error=session-missing");
    }

    const uploadId = `upload_${Date.now()}`;
    try {
      const contentBase64 = Buffer.from(await sourceFile.arrayBuffer()).toString("base64");
      await createOrganizationUploadContent(
        session.userId,
        session.organizationId,
        {
          upload_id: uploadId,
          site_id: siteId,
          category,
          filename: sourceFile.name,
          content_base64: contentBase64
        },
        session.authToken
      );
      revalidatePath("/uploads");
      revalidatePath("/dashboard");
    } catch {
      redirect("/uploads?error=create-failed");
    }
    redirect("/uploads?status=created");
  }

  return (
    <WorkspaceShell
      currentPath="/uploads"
      title="Source files"
      eyebrow="Workflow"
      modeLabel={
        mode === "live" ? "Live workspace" : mode === "error" ? "Workspace data unavailable" : "Sample workspace"
      }
      modeDescription={
        mode === "live"
          ? "Upload the four CSV inputs used by the original baseline dashboard workflow."
          : mode === "error"
            ? "We could not load your organization's live data from the backend. Reload the page to try again."
            : "Sample source files are shown until a live organization is selected."
      }
      modeTone={mode === "error" ? "error" : "info"}
    >
      {mode === "error" ? (
        <div className="authNotice authError" role="alert">
          <strong>Workspace data unavailable</strong>
          <span>
            Your real source files and site readiness could not be loaded, so no sample or
            fabricated figures are shown below. Reload the page or try again shortly.
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

      <section className="summaryGrid" aria-label="Upload metrics">
        <div className="summaryTile">
          <span>Total uploads</span>
          <strong>{uploads.length}</strong>
        </div>
        <div className="summaryTile">
          <span>Stored successfully</span>
          <strong>{storedUploads.length}</strong>
        </div>
        <div className="summaryTile">
          <span>Ready sites</span>
          <strong>
            {uploadReadiness.ready_site_count}/{uploadReadiness.site_count}
          </strong>
        </div>
        <div className="summaryTile">
          <span>Sites covered</span>
          <strong>{sites.filter((site) => uploads.some((upload) => upload.site_id === site.id)).length}</strong>
        </div>
        <div className="summaryTile">
          <span>Latest uploader</span>
          <strong>{uploads[0]?.uploaded_by_user_id ?? "Pending"}</strong>
        </div>
      </section>

      <section className="contentGrid">
        <div className="listPanel wide">
          <div className="sectionHeader">
            <h2>Required source files</h2>
            <span>Energy, weather, and SEU mapping inputs</span>
          </div>
          <form action={createUploadAction} className="inlineFormWide">
            <label className="authField">
              <span>Site</span>
              <select name="site_id" defaultValue={sites[0]?.id ?? ""}>
                {sites.map((site) => (
                  <option key={site.id} value={site.id}>
                    {site.name}
                  </option>
                ))}
              </select>
            </label>
            <label className="authField">
              <span>File type</span>
              <select name="category" defaultValue="energy">
                {sourceFileTypes.map((fileType) => (
                  <option key={fileType.value} value={fileType.value}>
                    {fileType.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="authField">
              <span>CSV file</span>
              <input name="source_file" type="file" accept=".csv,text/csv" />
            </label>
            <button type="submit" className="btn btnPrimary btnSm">
              Upload source file
            </button>
          </form>
          <div className="rowList">
            {uploads.map((upload) => (
              <div className="dataRow" key={upload.id}>
                <div>
                  <strong>{sourceFileLabel(upload.category)}</strong>
                  <span>
                    {upload.id} - checksum {shortChecksum(upload.checksum)}
                  </span>
                </div>
                <div>
                  <strong>{upload.status}</strong>
                  <span>{upload.site_id}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="listPanel">
          <div className="sectionHeader">
            <h2>Site readiness</h2>
            <span>Source contract by site</span>
          </div>
          <div className="rowList">
            {uploadReadiness.sites.length === 0 ? (
              <div className="dataRow">
                <div>
                  <strong>No sites configured</strong>
                  <span>Add a site in Settings before uploading source files.</span>
                </div>
              </div>
            ) : (
              uploadReadiness.sites.map((site) => (
                <div className="dataRow readinessRow" key={site.site_id}>
                  <div>
                    <strong>{site.site_name}</strong>
                    <span>
                      {site.ready_count}/{site.required_count} required source files ready
                    </span>
                    <div className="readinessFileList" aria-label={`${site.site_name} source file readiness`}>
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
                  <div>
                    <strong>{site.is_ready ? "Ready" : "Incomplete"}</strong>
                    <span>{site.site_id}</span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        <div className="listPanel">
          <div className="sectionHeader">
            <h2>File checks</h2>
            <span>Integrity fields</span>
          </div>
          <div className="rowList">
            {uploads.map((upload) => (
              <div className="dataRow" key={`${upload.id}-checksum`}>
                <div>
                  <strong>{upload.id}</strong>
                  <span>{upload.checksum}</span>
                </div>
                <div>
                  <strong>{upload.uploaded_by_user_id}</strong>
                  <span>{upload.organization_id}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </WorkspaceShell>
  );
}
