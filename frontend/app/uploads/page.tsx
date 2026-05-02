import { WorkspaceShell } from "../../components/WorkspaceShell";
import { loadDashboardData } from "../../lib/dashboard-data";

export default async function UploadsPage() {
  const { mode, sites, uploads } = await loadDashboardData();
  const storedUploads = uploads.filter((upload) => upload.status === "stored");
  const uploadCategories = new Set(uploads.map((upload) => upload.category)).size;

  return (
    <WorkspaceShell
      currentPath="/uploads"
      title="Tenant uploads"
      modeLabel={mode === "live" ? "Live tenant data" : "Demo workspace"}
      modeDescription={
        mode === "live"
          ? "Uploads are being listed from tenant-scoped backend metadata."
          : "Connect the demo workspace variables to view live upload records."
      }
    >
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
            <h2>Upload inventory</h2>
            <span>Storage-bound metadata</span>
          </div>
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
            <span>Upload counts</span>
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
            <h2>Checksums</h2>
            <span>Integrity trace</span>
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
