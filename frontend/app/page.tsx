import Link from "next/link";

const pillars = [
  {
    title: "Evidence you can defend",
    body: "Structured baselines, SEU alignment, and run history built for audits—not ad‑hoc spreadsheets."
  },
  {
    title: "Built for portfolios",
    body: "Separate tenants, scopes, and teams so every site keeps its boundaries without slowing delivery."
  },
  {
    title: "Analytics that scale",
    body: "From upload to modeled EnPI trends, the workflow stays repeatable as your meter count grows."
  }
];

const heroStats = [
  { label: "Tenant isolation", value: "By design", hint: "RBAC-ready shell" },
  { label: "Reporting runs", value: "Traceable", hint: "Queue → success path" },
  { label: "ISO 50001 fit", value: "Native", hint: "EnPI-style views" }
];

export default function HomePage() {
  return (
    <main className="landing">
      <header className="topbar landingTopbar">
        <Link href="/" className="brandLink" aria-label="RDS Energy Analytics home">
          <span className="brandMark" aria-hidden />
          <span className="brandName">RDS Energy</span>
        </Link>
        <nav className="topNav" aria-label="Main navigation">
          <a className="topNavMuted" href="#platform">
            Platform
          </a>
          <Link className="topNavMuted" href="/join/demo-invite">
            Accept invite
          </Link>
          <Link className="btn btnGhost btnSm" href="/dashboard">
            Dashboard
          </Link>
          <Link className="btn btnPrimary btnSm" href="/dashboard">
            Open workspace
          </Link>
        </nav>
      </header>

      <section className="heroShell" aria-labelledby="landing-hero-title">
        <div className="heroAmbient" aria-hidden>
          <span className="heroBlob heroBlobOne" />
          <span className="heroBlob heroBlobTwo" />
          <span className="heroBlob heroBlobThree" />
        </div>
        <div className="heroGridLines" aria-hidden />

        <div className="landingHero">
          <div className="heroCopy">
            <p className="eyebrow">ISO 50001 energy intelligence</p>
            <h1 id="landing-hero-title">
              Clarify energy performance—with proof your stakeholders trust.
            </h1>
            <p className="heroLead">
              RDS brings uploads, modeled baselines, and reporting runs into one modern workspace.
              Spend less time wrangling files and more time improving performance.
            </p>
            <div className="heroActions">
              <Link className="btn btnPrimary btnLg" href="/dashboard">
                Explore the dashboard
              </Link>
              <a className="btn btnGhost btnLg" href="#platform">
                See what&apos;s included
              </a>
            </div>
            <dl className="heroTrust">
              <div>
                <dt>Privacy-first layout</dt>
                <dd>Tenant-scoped navigation and placeholders mirror production boundaries.</dd>
              </div>
              <div>
                <dt>Consultant-ready</dt>
                <dd>Flows align with repeatable M&amp;V-style reviews and documented runs.</dd>
              </div>
            </dl>
          </div>

          <div className="heroVisualCard" aria-label="Platform snapshot">
            <div className="heroVisualGlow" aria-hidden />
            <div className="heroOrb" aria-hidden>
              <span className="heroOrbInner" />
            </div>
            <p className="heroVisualEyebrow">Live-style snapshot</p>
            <h2 className="heroVisualTitle">Energy performance cockpit</h2>
            <p className="heroVisualSubtitle">
              Baselines, actuals, and meter narratives in one uninterrupted view.
            </p>
            <ul className="heroStatStrip">
              {heroStats.map((item) => (
                <li key={item.label} className="heroStatChip">
                  <span className="heroStatLabel">{item.label}</span>
                  <span className="heroStatValue">{item.value}</span>
                  <span className="heroStatHint">{item.hint}</span>
                </li>
              ))}
            </ul>
            <Link className="heroVisualLink" href="/dashboard">
              Launch preview tenant →
            </Link>
          </div>
        </div>
      </section>

      <section className="valueSection" id="platform">
        <div className="valueSectionHeader">
          <p className="eyebrow">Why teams adopt RDS</p>
          <h2 className="valueTitle">
            Governance, speed, and analytics—without the enterprise bloat.
          </h2>
          <p className="valueIntro">
            The landing experience mirrors the SaaS roadmap: onboarding clarity today, uploads and audited
            run history tomorrow—always scoped to your organization.
          </p>
        </div>
        <div className="valueGrid">
          {pillars.map((pillar) => (
            <article className="valueCard" key={pillar.title}>
              <span className="valueCardAccent" aria-hidden />
              <h3>{pillar.title}</h3>
              <p>{pillar.body}</p>
            </article>
          ))}
        </div>
      </section>

      <footer className="landingFooter">
        <div>
          <strong className="footerBrand">RDS Energy Analytics</strong>
          <p className="footerNote">ISO 50001 analytics platform • SaaS expansion</p>
        </div>
        <Link className="btn btnGhost btnSm" href="/dashboard">
          Continue to dashboard
        </Link>
      </footer>
    </main>
  );
}
