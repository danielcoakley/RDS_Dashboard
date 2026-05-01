import { EnergyPerformanceChart } from "../../components/EnergyPerformanceChart";

const meters = [
  { id: "meter-001", name: "Main Electricity", value: "412,800 kWh", status: "SEU" },
  { id: "meter-002", name: "Main Gas", value: "289,400 kWh", status: "SEU" },
  { id: "meter-003", name: "Office Lighting", value: "54,900 kWh", status: "Monitor" }
];

const runs = [
  { id: "run-1042", label: "April baseline comparison", status: "Succeeded" },
  { id: "run-1041", label: "Q1 ISO summary", status: "Succeeded" },
  { id: "run-1040", label: "SEU variance review", status: "Queued" }
];

export default function DashboardPage() {
  return (
    <main className="appShell">
      <aside className="sidebar">
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

        <section className="summaryGrid" id="overview" aria-label="Summary metrics">
          <div className="summaryTile">
            <span>Current run status</span>
            <strong>Ready</strong>
          </div>
          <div className="summaryTile">
            <span>Active meters</span>
            <strong>3</strong>
          </div>
          <div className="summaryTile">
            <span>Report artifacts</span>
            <strong>12</strong>
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
                    <strong>{meter.name}</strong>
                    <span>{meter.id}</span>
                  </div>
                  <div>
                    <strong>{meter.value}</strong>
                    <span>{meter.status}</span>
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
                    <strong>{run.label}</strong>
                    <span>{run.id}</span>
                  </div>
                  <div>
                    <strong>{run.status}</strong>
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
