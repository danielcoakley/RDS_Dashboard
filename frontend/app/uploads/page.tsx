import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import { WorkspaceShell } from "../../components/WorkspaceShell";
import { createOrganizationUpload } from "../../lib/api";
import { loadDashboardData } from "../../lib/dashboard-data";
import { readAppSession } from "../../lib/session";

const sourceFileTypes = [
  { value: "energy", label: "Energy Data" },
  { value: "hdd", label: "HDD Data" },
  { value: "cdd", label: "CDD Data" },
  { value: "seu_mapping", label: "SEU Mapping" }
];

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
      title: "Source file recorded",
      body: "The file is now included in the baseline workflow checklist."
    };
  }
  if (error === "missing-fields") {
    return {
      tone: "error",
      title: "Source file not recorded",
      body: "Choose a site, source file type, filename, and checksum."
    };
  }
  if (error === "session-missing") {
    return {
      tone: "error",
      title: "Source file not recorded",
      body: "Sign in to an organization workspace before adding source files."
    };
  }
  if (error === "create-failed") {
    return {
      tone: "error",
      title: "Source file not recorded",
      body: "We could not create that upload record. Check fields and try again."
    };
  }
  return null;
}

export default async function UploadsPage({ searchParams }: UploadsPageProps) {
  const query = await searchParams;
  const { mode, sites, uploads } = await loadDashboardData();
  const storedUploads = uploads.filter((upload) => upload.status === "stored");
  const uploadCategories = new Set(uploads.map((upload) => upload.category)).size;
  const pageMessage = uploadsMessage(query.status, query.error);

  async function createUploadAction(formData: FormData) {
    "use server";

    const siteId = String(formData.get("site_id") ?? "").trim();
    const category = String(formData.get("category") ?? "").trim();
    const filename = String(formData.get("filename") ?? "").trim();
    const checksum = String(formData.get("checksum") ?? "").trim();
    const session = await readAppSession();

    if (!siteId || !category || !filename || !checksum) {
      redirect("/uploads?error=missing-fields");
    }
    if (!session.userId || !session.organizationId) {
      redirect("/uploads?error=session-missing");
    }

    const uploadId = `upload_${Date.now()}`;
    try {
      await createOrganizationUpload(
        session.userId,
        session.organizationId,
        {
          upload_id: uploadId,
          site_id: siteId,
          category,
          filename,
          checksum
        },
        session.authToken
      );
      revalidatePath("/uploads");
      revalidatePath("/dashboard");
      redirect("/uploads?status=created");
    } catch {
      redirect("/uploads?error=create-failed");
    }
  }

  return (
    <WorkspaceShell
      currentPath="/uploads"
      title="Source files"
      eyebrow="Workflow"
      modeLabel={mode === "live" ? "Live workspace" : "Sample workspace"}
      modeDescription={
        mode === "live"
          ? "Record the four CSV inputs used by the original baseline dashboard workflow."
          : "Sample source files are shown until a live organization is selected."
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
          <span>Categories</span>
          <strong>{uploadCategories}</strong>
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
              <span>Filename</span>
              <input name="filename" type="text" placeholder="energy.csv" />
            </label>
            <label className="authField">
              <span>Checksum</span>
              <input name="checksum" type="text" placeholder="abc123" />
            </label>
            <button type="submit" className="btn btnPrimary btnSm">
              Add source file
            </button>
          </form>
          <div className="rowList">
            {uploads.map((upload) => (
              <div className="dataRow" key={upload.id}>
                <div>
                  <strong>
                    {sourceFileTypes.find((fileType) => fileType.value === upload.category)?.label ??
                      upload.category}
                  </strong>
                  <span>{upload.storage_key}</span>
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
            <h2>Site coverage</h2>
            <span>Files by site</span>
          </div>
          <div className="rowList">
            {sites.map((site) => {
              const siteUploads = uploads.filter((upload) => upload.site_id === site.id);
              return (
                <div className="dataRow" key={site.id}>
                  <div>
                    <strong>{site.name}</strong>
                    <span>{site.timezone}</span>
                  </div>
                  <div>
                    <strong>{siteUploads.length} files</strong>
                    <span>{site.id}</span>
                  </div>
                </div>
              );
            })}
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
