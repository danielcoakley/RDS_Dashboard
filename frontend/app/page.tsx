import Link from "next/link";

const outcomes = [
  "Tenant-scoped energy performance analysis",
  "ISO 50001 evidence-ready reporting",
  "Run history for repeatable M&V workflows"
];

export default function HomePage() {
  return (
    <main className="landing">
      <nav className="topbar" aria-label="Main navigation">
        <span className="brand">RDS Energy Analytics</span>
        <Link className="navLink" href="/dashboard">
          Open dashboard
        </Link>
      </nav>

      <section className="landingHero">
        <div className="heroCopy">
          <p className="eyebrow">ISO 50001 analytics platform</p>
          <h1>Energy performance evidence for every tenant, site, and meter.</h1>
          <p className="heroText">
            Turn uploads, baselines, and reporting runs into a governed SaaS workflow
            for energy teams and consultants.
          </p>
          <Link className="primaryAction" href="/dashboard">
            View SaaS shell
          </Link>
        </div>
        <div className="heroPanel" aria-label="MVP outcomes">
          {outcomes.map((outcome) => (
            <div className="metricStrip" key={outcome}>
              <span className="metricDot" />
              <span>{outcome}</span>
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}
