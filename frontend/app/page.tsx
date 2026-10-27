import Link from "next/link";
import { redirect } from "next/navigation";
import { ApiRequestError, createMarketingInquiry } from "../lib/api";

const navItems = [
  { href: "#use-cases", label: "Use cases" },
  { href: "#workflow", label: "Workflow" },
  { href: "#platform", label: "Platform" },
  { href: "#evidence", label: "Evidence" },
  { href: "#contact", label: "Pilot" }
];

const proofPoints = [
  { value: "EnPI", label: "Weather-normalized baseline and comparison models" },
  { value: "ISO 50001", label: "Evidence formatted for management reviews and audits" },
  { value: "Auto QA", label: "Every run flags data gaps and outliers before reporting" }
];

// Placeholder sector labels until real pilot customer logos can go here.
const trustSectors = [
  "Manufacturing",
  "Higher Education",
  "Retail Portfolios",
  "Healthcare Estates",
  "Energy Consultancies"
];

const painPoints = [
  "Meter, weather, and SEU data live in separate spreadsheets nobody fully trusts.",
  "Baseline models get rebuilt by hand before every audit or management review.",
  "Data gaps and outliers surface for the first time inside a client meeting."
];

const solutionPoints = [
  "One upload flow standardises energy, HDD/CDD, and SEU data automatically.",
  "Baseline and comparison-year EnPI models run on demand, the same way every time.",
  "Gaps and outliers are flagged before a report ever reaches a reviewer."
];

type BenefitIcon = "evidence" | "weather" | "speed" | "portfolio";

const benefits: { title: string; body: string; icon: BenefitIcon }[] = [
  {
    title: "Audit-ready evidence",
    body: "Every baseline, comparison, and SEU output traces back to its source files for external audits and management reviews.",
    icon: "evidence"
  },
  {
    title: "Weather-normalized accuracy",
    body: "HDD and CDD inputs correct for weather swings, so performance changes reflect real operational shifts, not a mild winter.",
    icon: "weather"
  },
  {
    title: "Faster audit prep",
    body: "Run baseline and comparison analyses on demand instead of rebuilding spreadsheets before every review cycle.",
    icon: "speed"
  },
  {
    title: "Built for portfolios",
    body: "Keep every site, building, or client on the same consistent evidence process as the portfolio grows.",
    icon: "portfolio"
  }
];

function BenefitGlyph({ icon }: { icon: BenefitIcon }) {
  switch (icon) {
    case "evidence":
      return (
        <svg viewBox="0 0 24 24" width="22" height="22" fill="none" aria-hidden="true">
          <path d="M6 3h9l3 3v15H6z" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
          <path d="M9 12.5l2 2 4-4.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      );
    case "weather":
      return (
        <svg viewBox="0 0 24 24" width="22" height="22" fill="none" aria-hidden="true">
          <circle cx="12" cy="12" r="4" stroke="currentColor" strokeWidth="1.6" />
          <path
            d="M12 2v2M12 20v2M4.2 4.2l1.4 1.4M18.4 18.4l1.4 1.4M2 12h2M20 12h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
          />
        </svg>
      );
    case "speed":
      return (
        <svg viewBox="0 0 24 24" width="22" height="22" fill="none" aria-hidden="true">
          <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.6" />
          <path d="M12 7v5l3.5 2" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      );
    case "portfolio":
      return (
        <svg viewBox="0 0 24 24" width="22" height="22" fill="none" aria-hidden="true">
          <rect x="3" y="10" width="6" height="11" stroke="currentColor" strokeWidth="1.6" />
          <rect x="10.5" y="5" width="6" height="16" stroke="currentColor" strokeWidth="1.6" />
          <rect x="18" y="13" width="3" height="8" stroke="currentColor" strokeWidth="1.6" />
        </svg>
      );
    default:
      return null;
  }
}

type TourMockup = "baseline" | "quality" | "report";

const tourPanels: { eyebrow: string; title: string; body: string; mockup: TourMockup }[] = [
  {
    eyebrow: "Baseline & EnPI",
    title: "See performance the way an auditor will",
    body: "Baseline and comparison-year models sit side by side with a clear EnPI trend, built from the same data every time a run executes.",
    mockup: "baseline"
  },
  {
    eyebrow: "Data quality",
    title: "Catch gaps and outliers before anyone else does",
    body: "Every run flags missing intervals and anomalous readings up front, so they get resolved before they show up in front of a reviewer.",
    mockup: "quality"
  },
  {
    eyebrow: "Evidence & reporting",
    title: "Package the proof, not just the numbers",
    body: "Report artifacts pull directly from the run history, so every figure in the evidence pack traces back to a source file.",
    mockup: "report"
  }
];

function BaselineMockup() {
  const bars = [
    { height: 38, variant: "comparison" },
    { height: 52, variant: "baseline" },
    { height: 46, variant: "baseline" },
    { height: 61, variant: "comparison" },
    { height: 58, variant: "baseline" },
    { height: 70, variant: "baseline" },
    { height: 64, variant: "comparison" },
    { height: 74, variant: "baseline" }
  ];
  return (
    <div className="tourMockup">
      <div className="tourMockupHeader">
        <span />
        <span />
        <span />
        <span className="tourMockupHeaderLabel">Baseline vs. comparison</span>
      </div>
      <div className="tourChartRows">
        {bars.map((bar, index) => (
          <div
            className="tourChartBar"
            data-variant={bar.variant}
            key={index}
            style={{ height: `${bar.height}%` }}
          />
        ))}
      </div>
      <div className="tourChartLegend">
        <span>
          <i className="tourLegendDot tourLegendDotBaseline" aria-hidden />
          Baseline
        </span>
        <span>
          <i className="tourLegendDot tourLegendDotComparison" aria-hidden />
          Comparison year
        </span>
      </div>
    </div>
  );
}

function DataQualityMockup() {
  const rows: { label: string; status: "ok" | "gap" | "outlier" }[] = [
    { label: "Jan interval readings", status: "ok" },
    { label: "Feb interval readings", status: "gap" },
    { label: "Mar interval readings", status: "outlier" },
    { label: "Apr interval readings", status: "ok" }
  ];
  const statusLabel: Record<string, string> = {
    ok: "Clean",
    gap: "Gap flagged",
    outlier: "Outlier flagged"
  };
  return (
    <div className="tourMockup">
      <div className="tourMockupHeader">
        <span />
        <span />
        <span />
        <span className="tourMockupHeaderLabel">Data quality checks</span>
      </div>
      <div className="tourQualityRows">
        {rows.map((row) => (
          <div className="tourQualityRow" key={row.label}>
            <span className={`tourQualityDot tourQualityDot--${row.status}`} aria-hidden />
            <span>{row.label}</span>
            <span className="tourQualityTag">{statusLabel[row.status]}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function ReportMockup() {
  const rows = ["Baseline model summary", "Comparison year EnPI", "SEU drill-down", "Source file lineage"];
  return (
    <div className="tourMockup">
      <div className="tourMockupHeader">
        <span />
        <span />
        <span />
        <span className="tourMockupHeaderLabel">Evidence pack</span>
      </div>
      <div className="tourReportRows">
        {rows.map((row) => (
          <div className="tourReportRow" key={row}>
            <svg viewBox="0 0 20 20" width="14" height="14" aria-hidden="true">
              <path
                d="M4 10.5l3.2 3.2L16 5"
                stroke="currentColor"
                strokeWidth="1.8"
                fill="none"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
            <span>{row}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

const useCases = [
  {
    title: "ISO 50001 evidence packs",
    body: "Build traceable baseline, comparison, and SEU outputs for management reviews and external audits."
  },
  {
    title: "Consultant portfolio delivery",
    body: "Keep each client on a consistent upload, run, and reporting process instead of rebuilding it per engagement."
  },
  {
    title: "Campus and estates teams",
    body: "Turn meter, weather, and operational context into a shared view of energy performance."
  },
  {
    title: "M&V style reviews",
    body: "Compare actual consumption against modeled performance and identify material changes early."
  }
];

const workflowSteps = [
  {
    step: "01",
    title: "Set up your organization",
    body: "Create your account, invite your team, and get everyone the right access before data starts moving."
  },
  {
    step: "02",
    title: "Upload source files",
    body: "Collect energy data, heating degree days, cooling degree days, and SEU mappings in one scoped workflow."
  },
  {
    step: "03",
    title: "Run the analysis",
    body: "Execute baseline and comparison runs that produce clear status, history, and retry paths."
  },
  {
    step: "04",
    title: "Review reports",
    body: "Use dashboard outputs and report artifacts to brief stakeholders with consistent evidence."
  }
];

const platformFeatures = [
  {
    title: "Data & modeling",
    items: [
      "Energy, weather, and SEU intake",
      "Automated standardisation and cleaning",
      "Weather-normalized EnPI baselines",
      "Gap and outlier detection"
    ]
  },
  {
    title: "Evidence & reporting",
    items: [
      "ISO 50001-aligned summaries",
      "Baseline vs. comparison reporting",
      "Report artifact history",
      "Traceable source-file lineage"
    ]
  },
  {
    title: "Built for growing portfolios",
    items: [
      "Multi-site and multi-client setup",
      "Consistent process across every site",
      "Activity history for every run",
      "Sample and live data modes"
    ]
  }
];

const evidenceItems = [
  "Baseline versus comparison year performance",
  "Electricity, gas, and SEU drill-downs",
  "Site and client-scoped run history and report access",
  "Traceable activity records for key workflow events"
];

// Placeholder attributions pending real pilot customer quotes.
const testimonials = [
  {
    quote:
      "The baseline model that used to take us a week now runs in an afternoon, and every number still traces back to a source file.",
    name: "Facilities Director",
    org: "Manufacturing Portfolio"
  },
  {
    quote:
      "Our external auditor asked for evidence, not just charts. This is the first tool that hands over both at once.",
    name: "Sustainability Lead",
    org: "Higher Education Estate"
  },
  {
    quote: "We run the same process across every client site now instead of rebuilding spreadsheets each time.",
    name: "Energy Consultant",
    org: "Portfolio Consultancy"
  }
];

const inquiryUseCases = [
  { value: "audit_readiness", label: "Audit readiness" },
  { value: "consultant_delivery", label: "Consultant delivery" },
  { value: "estate_portfolio", label: "Estate portfolio" },
  { value: "measurement_verification", label: "M&V review" },
  { value: "other", label: "Something else" }
];

type HomePageProps = {
  searchParams: Promise<{ status?: string; error?: string; retryAfter?: string }>;
};

function formatRetryAfter(value: string | undefined): string {
  const seconds = Number(value);
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return "Please wait a little while before trying again.";
  }
  const minutes = Math.max(1, Math.ceil(seconds / 60));
  if (minutes < 60) {
    return `Please try again in about ${minutes} minute${minutes === 1 ? "" : "s"}.`;
  }
  const hours = Math.ceil(minutes / 60);
  return `Please try again in about ${hours} hour${hours === 1 ? "" : "s"}.`;
}

function landingInquiryMessage(
  status: string | undefined,
  error: string | undefined,
  retryAfter: string | undefined
): { tone: "success" | "error"; title: string; body: string } | null {
  if (status === "inquiry-sent") {
    return {
      tone: "success",
      title: "Pilot request received",
      body: "Thanks. The request is stored for follow-up with the RDS team."
    };
  }
  if (error === "missing-inquiry-fields") {
    return {
      tone: "error",
      title: "Pilot request not sent",
      body: "Add your name, email, use case, and a short note."
    };
  }
  if (error === "inquiry-failed") {
    return {
      tone: "error",
      title: "Pilot request not sent",
      body: "We could not store the request. Check the email address and try again."
    };
  }
  if (error === "inquiry-rate-limited") {
    return {
      tone: "error",
      title: "Pilot request paused",
      body: `Too many pilot requests have been sent from this source. ${formatRetryAfter(retryAfter)}`
    };
  }
  return null;
}

export default async function HomePage({ searchParams }: HomePageProps) {
  const query = await searchParams;
  const inquiryMessage = landingInquiryMessage(query.status, query.error, query.retryAfter);

  async function createInquiryAction(formData: FormData) {
    "use server";

    const name = String(formData.get("name") ?? "").trim();
    const email = String(formData.get("email") ?? "").trim();
    const organizationName = String(formData.get("organization_name") ?? "").trim();
    const role = String(formData.get("role") ?? "").trim();
    const intendedUse = String(formData.get("intended_use") ?? "").trim();
    const message = String(formData.get("message") ?? "").trim();
    const website = String(formData.get("website") ?? "").trim();

    if (!name || !email || !intendedUse || !message) {
      redirect("/?error=missing-inquiry-fields#contact");
    }

    try {
      await createMarketingInquiry({
        name,
        email,
        organization_name: organizationName || null,
        role: role || null,
        intended_use: intendedUse,
        message,
        source_page: "/",
        website: website || null
      });
    } catch (error) {
      if (error instanceof ApiRequestError && error.status === 429) {
        const retryAfter = error.retryAfterSeconds ?? "";
        redirect(`/?error=inquiry-rate-limited&retryAfter=${retryAfter}#contact`);
      }
      redirect("/?error=inquiry-failed#contact");
    }

    redirect("/?status=inquiry-sent#contact");
  }

  return (
    <main className="landing">
      <header className="topbar landingTopbar">
        <Link href="/" className="brandLink" aria-label="RDS Energy Analytics home">
          <span className="brandMark" aria-hidden />
          <span className="brandName">RDS Energy Analytics</span>
        </Link>
        <nav className="topNav" aria-label="Landing navigation">
          {navItems.map((item) => (
            <a className="topNavMuted" href={item.href} key={item.href}>
              {item.label}
            </a>
          ))}
          <Link className="btn btnGhost btnSm" href="/login">
            Sign in
          </Link>
          <Link className="btn btnPrimary btnSm" href="/signup">
            Create workspace
          </Link>
        </nav>
      </header>

      <section className="landingHero" aria-labelledby="landing-hero-title">
        <div className="heroBackdrop" aria-hidden />
        <div className="heroContent">
          <p className="eyebrow heroEyebrow">Energy performance evidence platform</p>
          <h1 id="landing-hero-title">Prove your energy performance, not just chart it.</h1>
          <p className="heroLead">
            RDS Energy Analytics turns raw meter, weather, and SEU data into weather-normalized ISO 50001
            baselines, so your next certification audit or management review starts with evidence, not a
            spreadsheet scramble.
          </p>
          <div className="heroActions" aria-label="Primary actions">
            <Link className="btn btnPrimary btnLg" href="/signup">
              Create workspace
            </Link>
            <a className="btn btnGhost btnLg" href="#contact">
              Request pilot call
            </a>
          </div>
        </div>
        <dl className="heroProof" aria-label="Platform proof points">
          {proofPoints.map((item) => (
            <div key={item.value}>
              <dt>{item.value}</dt>
              <dd>{item.label}</dd>
            </div>
          ))}
        </dl>
      </section>

      <section className="trustBar" aria-label="Industries served">
        <p className="trustBarLabel">Built for energy and sustainability teams across</p>
        <ul className="trustBarList">
          {trustSectors.map((sector) => (
            <li key={sector}>{sector}</li>
          ))}
        </ul>
      </section>

      <section className="approachSection" id="approach" aria-labelledby="approach-title">
        <div className="sectionIntro">
          <p className="eyebrow">The problem</p>
          <h2 id="approach-title">Stop rebuilding your evidence trail from scratch.</h2>
        </div>
        <div className="approachGrid">
          <div className="approachCard">
            <span className="approachTag">Without RDS</span>
            <ul>
              {painPoints.map((point) => (
                <li key={point}>{point}</li>
              ))}
            </ul>
          </div>
          <div className="approachCard approachCardAccent">
            <span className="approachTag approachTagAccent">With RDS</span>
            <ul>
              {solutionPoints.map((point) => (
                <li key={point}>{point}</li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      <section className="benefitsSection" id="benefits" aria-labelledby="benefits-title">
        <div className="sectionIntro">
          <p className="eyebrow">Why RDS</p>
          <h2 id="benefits-title">Built around what an energy manager actually has to prove.</h2>
        </div>
        <div className="benefitsGrid">
          {benefits.map((benefit) => (
            <article className="benefitCard" key={benefit.title}>
              <span className="benefitIcon">
                <BenefitGlyph icon={benefit.icon} />
              </span>
              <h3>{benefit.title}</h3>
              <p>{benefit.body}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="tourSection" id="product-tour" aria-labelledby="tour-title">
        <div className="sectionIntro">
          <p className="eyebrow">Inside the platform</p>
          <h2 id="tour-title">From raw files to a defensible performance story.</h2>
        </div>
        <div className="tourPanels">
          {tourPanels.map((panel, index) => (
            <article className={`tourPanel${index % 2 === 1 ? " tourPanelReverse" : ""}`} key={panel.title}>
              <div className="tourPanelCopy">
                <p className="eyebrow">{panel.eyebrow}</p>
                <h3>{panel.title}</h3>
                <p>{panel.body}</p>
              </div>
              <div className="tourPanelMedia">
                {panel.mockup === "baseline" ? (
                  <BaselineMockup />
                ) : panel.mockup === "quality" ? (
                  <DataQualityMockup />
                ) : (
                  <ReportMockup />
                )}
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="useCaseSection" id="use-cases" aria-labelledby="use-cases-title">
        <div className="sectionIntro">
          <p className="eyebrow">Use cases</p>
          <h2 id="use-cases-title">Built for teams who need defensible energy performance evidence.</h2>
          <p>
            Whether you&rsquo;re preparing for certification, running a portfolio of client sites, or managing
            a single campus, RDS turns your source data into a defensible performance record.
          </p>
        </div>
        <div className="useCaseGrid">
          {useCases.map((useCase) => (
            <article className="landingCard" key={useCase.title}>
              <h3>{useCase.title}</h3>
              <p>{useCase.body}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="workflowSection" id="workflow" aria-labelledby="workflow-title">
        <div className="sectionIntro">
          <p className="eyebrow">Workflow</p>
          <h2 id="workflow-title">From raw files to repeatable reporting runs.</h2>
          <p>
            From first upload to final report, the same four steps apply whether you&rsquo;re managing one
            site or fifty client portfolios.
          </p>
        </div>
        <div className="workflowTrack">
          {workflowSteps.map((item) => (
            <article className="workflowItem" key={item.step}>
              <span>{item.step}</span>
              <h3>{item.title}</h3>
              <p>{item.body}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="platformSection" id="platform" aria-labelledby="platform-title">
        <div className="sectionIntro">
          <p className="eyebrow">Platform capabilities</p>
          <h2 id="platform-title">Everything a site or portfolio needs to prove energy performance.</h2>
          <p>
            From first upload to final report, RDS keeps every site or client on one consistent,
            evidence-ready process.
          </p>
        </div>
        <div className="featureColumns">
          {platformFeatures.map((group) => (
            <article className="featureColumn" key={group.title}>
              <h3>{group.title}</h3>
              <ul>
                {group.items.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </article>
          ))}
        </div>
      </section>

      <section className="evidenceSection" id="evidence" aria-labelledby="evidence-title">
        <div className="evidenceMedia">
          <img
            src="/marketing/rds-energy-dashboard-summary.png"
            alt="RDS Energy Analytics dashboard showing consumption summary and SEU energy flow"
          />
        </div>
        <div className="evidenceCopy">
          <p className="eyebrow">Marketing proof</p>
          <h2 id="evidence-title">Show the work behind every claim.</h2>
          <p>
            Energy leaders need more than a chart. They need a clean path from source files to a
            performance narrative, with the traceability an ISO 50001 auditor or management review
            actually expects.
          </p>
          <ul className="evidenceList">
            {evidenceItems.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      </section>

      <section className="resultsSection" id="results" aria-labelledby="results-title">
        <div className="sectionIntro">
          <p className="eyebrow">Early access feedback</p>
          <h2 id="results-title">What energy and sustainability teams are telling us early.</h2>
        </div>
        <div className="resultsGrid">
          {testimonials.map((testimonial) => (
            <figure className="resultCard" key={testimonial.name}>
              <blockquote>&ldquo;{testimonial.quote}&rdquo;</blockquote>
              <figcaption>
                <strong>{testimonial.name}</strong>
                <span>{testimonial.org}</span>
              </figcaption>
            </figure>
          ))}
        </div>
      </section>

      <section className="conversionBand" id="contact" aria-labelledby="conversion-title">
        <div className="conversionGrid">
          <div className="conversionCopy">
            <p className="eyebrow">Ready for pilots</p>
            <h2 id="conversion-title">Launch the workspace, invite the team, and start building the evidence trail.</h2>
            <p>
              Capture the client context up front so onboarding can start with the right sites,
              source files, reviewers, and reporting outcomes.
            </p>
            <div className="conversionActions">
              <Link className="btn btnPrimary btnLg" href="/signup">
                Create workspace
              </Link>
              <Link className="btn btnGhost btnLg" href="/join/demo-invite">
                Accept demo invite
              </Link>
            </div>
          </div>
          <form action={createInquiryAction} className="inquiryForm">
            <div>
              <p className="eyebrow">Pilot request</p>
              <h3>Talk through a client rollout</h3>
            </div>
            {inquiryMessage ? (
              <div
                className={`authNotice ${inquiryMessage.tone === "success" ? "authSuccess" : "authError"}`}
                role={inquiryMessage.tone === "success" ? "status" : "alert"}
              >
                <strong>{inquiryMessage.title}</strong>
                <span>{inquiryMessage.body}</span>
              </div>
            ) : null}
            <div className="formGridTwo">
              <label>
                <span>Name</span>
                <input name="name" type="text" autoComplete="name" required maxLength={120} />
              </label>
              <label>
                <span>Email</span>
                <input name="email" type="email" autoComplete="email" required maxLength={254} />
              </label>
            </div>
            <div className="formGridTwo">
              <label>
                <span>Organization</span>
                <input name="organization_name" type="text" autoComplete="organization" maxLength={160} />
              </label>
              <label>
                <span>Role</span>
                <input name="role" type="text" autoComplete="organization-title" maxLength={120} />
              </label>
            </div>
            <label>
              <span>Use case</span>
              <select name="intended_use" required defaultValue="audit_readiness">
                {inquiryUseCases.map((useCase) => (
                  <option key={useCase.value} value={useCase.value}>
                    {useCase.label}
                  </option>
                ))}
              </select>
            </label>
            <label>
              <span>What needs to happen first?</span>
              <textarea name="message" required maxLength={1200} rows={4} />
            </label>
            <label className="inquiryHoneypot" aria-hidden="true">
              <span>Website</span>
              <input name="website" type="text" autoComplete="off" tabIndex={-1} />
            </label>
            <button className="btn btnPrimary btnLg" type="submit">
              Request pilot call
            </button>
          </form>
        </div>
      </section>

      <footer className="landingFooter">
        <div>
          <strong className="footerBrand">RDS Energy Analytics</strong>
          <p className="footerNote">ISO 50001 energy analytics evidence for energy, weather, runs, and reports.</p>
        </div>
        <div className="footerLinks" aria-label="Footer navigation">
          <Link href="/login">Sign in</Link>
          <Link href="/signup">Create workspace</Link>
          <Link href="/dashboard">Sample dashboard</Link>
        </div>
      </footer>
    </main>
  );
}
