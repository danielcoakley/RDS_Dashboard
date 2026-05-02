import Link from "next/link";
import { EnergyPerformanceChart } from "../../components/EnergyPerformanceChart";
import { loadDashboardData } from "../../lib/dashboard-data";

export default async function DashboardPage() {
  const { mode, sites, meters, uploads, runs, reports, auditEvents } = await loadDashboardData();
  const latestRun = runs[0];
  const modeLabel = mode === "live" ? "Live tenant data" : "Demo workspace";

  return (
    <main className="appShell">
      <aside className="sidebar">
        <Link href="/" className="sidebarBrandRow" aria-label="RDS Energy — return to home">
          <span className="sidebarBrandMark" aria-hidden />
          <div className="sidebarBrandMeta">
            <strong>RDS Energy</strong>
            <span className="sidebarTagline">Marketing &amp; overview</span>
          </div>
        </Link>
        <div className="tenantBlock">
          <span className="tenantLabel">Tenant</span>
          <strong>RDS Site</strong>
        </div>
        <nav className="sideNav" aria-label="Dashboard sections">
          <a href="#overview">Overview</a>
          <a href="#meters">Meters</a>
          <a href="#runs">Runs</a>
          <a href="#reports">Reports</a>
          <a href="#settings">Settings</a>
        </nav>
      </aside>

      <section className="workspace">
        <header className="workspaceHeader">
          <div>
            <p className="eyebrow">Tenant dashboard</p>
            <h1>Energy performance overview</h1>
          </div>
          <div className="userMenu">Owner</div>
        </header>

        <div className="modeBanner" role="status" aria-live="polite">
          <strong>{modeLabel}</strong>
          <span>
            {mode === "live"
              ? "This view is reading tenant-scoped resources from the FastAPI backend."
              : "Set DEMO_USER_ID and DEMO_ORGANIZATION_ID to hydrate this dashboard from the API."}
          </span>
        </div>

        <section className="summaryGrid" id="overview" aria-label="Summary metrics">
          <div className="summaryTile">
            <span>Current run status</span>
            <strong>{latestRun?.status ?? "Ready"}</strong>
          </div>
          <div className="summaryTile">
            <span>Active meters</span>
            <strong>{meters.length}</strong>
          </div>
          <div className="summaryTile">
            <span>Recent uploads</span>
            <strong>{uploads.length}</strong>
          </div>
          <div className="summaryTile">
            <span>Report artifacts</span>
            <strong>{reports.length}</strong>
          </div>
          <div className="summaryTile">
            <span>Audit events</span>
            <strong>{auditEvents.length}</strong>
          </div>
        </section>

        <section className="contentGrid">
          <div className="chartPanel">
            <div className="sectionHeader">
              <h2>Baseline vs actual</h2>
              <span>2026 YTD</span>
            </div>
            <EnergyPerformanceChart />
          </div>

          <div className="listPanel" id="meters">
            <div className="sectionHeader">
              <h2>Meters</h2>
              <span>Tenant scoped</span>
            </div>
            <div className="rowList">
              {meters.map((meter) => (
                <div className="dataRow" key={meter.id}>
                  <div>
                    <strong>{meter.display_name}</strong>
                    <span>{meter.id}</span>
                  </div>
                  <div>
                    <strong>{meter.unit}</strong>
                    <span>{meter.is_seu ? "SEU" : "Monitor"}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="listPanel wide" id="runs">
            <div className="sectionHeader">
              <h2>Recent runs</h2>
              <span>Run lifecycle</span>
            </div>
            <div className="rowList">
              {runs.map((run) => (
                <div className="dataRow" key={run.id}>
                  <div>
                    <strong>{run.id}</strong>
                    <span>{run.site_id}</span>
                  </div>
                  <div>
                    <strong>{run.status}</strong>
                    <span>{run.error_message ?? "No errors"}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="listPanel" id="uploads">
            <div className="sectionHeader">
              <h2>Uploads</h2>
              <span>Tenant resources</span>
            </div>
            <div className="rowList">
              {uploads.map((upload) => (
                <div className="dataRow" key={upload.id}>
                  <div>
                    <strong>{upload.category}</strong>
                    <span>{upload.site_id}</span>
                  </div>
                  <div>
                    <strong>{upload.status}</strong>
                    <span>{upload.uploaded_by_user_id}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="listPanel" id="reports">
            <div className="sectionHeader">
              <h2>Reports</h2>
              <span>Published artifacts</span>
            </div>
            <div className="rowList">
              {reports.map((report) => (
                <div className="dataRow" key={report.id}>
                  <div>
                    <strong>{report.report_type}</strong>
                    <span>{report.id}</span>
                  </div>
                  <div>
                    <strong>{report.is_published ? "Published" : "Draft"}</strong>
                    <span>{report.run_id}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="listPanel" id="settings">
            <div className="sectionHeader">
              <h2>Sites and uploads</h2>
              <span>Tenant resources</span>
            </div>
            <div className="rowList">
              {sites.map((site) => (
                <div className="dataRow" key={site.id}>
                  <div>
                    <strong>{site.name}</strong>
                    <span>{site.timezone}</span>
                  </div>
                  <div>
                    <strong>{uploads.filter((upload) => upload.site_id === site.id).length} uploads</strong>
                    <span>{site.id}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="listPanel wide" id="audit">
            <div className="sectionHeader">
              <h2>Audit trail</h2>
              <span>Owner visibility</span>
            </div>
            <div className="rowList">
              {auditEvents.map((event) => (
                <div className="dataRow" key={event.id}>
                  <div>
                    <strong>{event.action}</strong>
                    <span>
                      {event.resource_type} · {event.resource_id}
                    </span>
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
      </section>
    </main>
  );
}
