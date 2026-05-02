import { revalidatePath } from "next/cache";
import { redirect } from "next/navigation";
import { WorkspaceShell } from "../../components/WorkspaceShell";
import { createOrganizationUpload } from "../../lib/api";
import { loadDashboardData } from "../../lib/dashboard-data";
import { readAppSession } from "../../lib/session";

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
      title: "Upload recorded",
      body: "The upload metadata was added to your organization."
    };
  }
  if (error === "missing-fields") {
    return {
      tone: "error",
      title: "Upload not recorded",
      body: "Choose a site and enter category, filename, and checksum."
    };
  }
  if (error === "session-missing") {
    return {
      tone: "error",
      title: "Upload not recorded",
      body: "Sign in to an organization workspace before creating uploads."
    };
  }
  if (error === "create-failed") {
    return {
      tone: "error",
      title: "Upload not recorded",
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
      title="Data uploads"
      modeLabel={mode === "live" ? "Live workspace" : "Sample workspace"}
      modeDescription={
        mode === "live"
          ? "Upload records are loading from the backend."
          : "This page is showing sample uploads until a live organization is selected."
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
            <h2>Upload history</h2>
            <span>File records</span>
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
              <span>Category</span>
              <input name="category" type="text" placeholder="energy" />
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
              Add upload
            </button>
          </form>
          <div className="rowList">
            {uploads.map((upload) => (
              <div className="dataRow" key={upload.id}>
                <div>
                  <strong>{upload.category}</strong>
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
