import type { Metadata } from 'next';
import Link from 'next/link';
import {
  ArrowRight, BarChart3, CheckCircle2, ClipboardCheck, CloudSun, FileSpreadsheet,
  Gauge, Layers, ListChecks, Quote, ShieldCheck, Target, Upload, Users, XCircle,
} from 'lucide-react';
import { MarketingNav } from '@/components/marketing/Nav';
import { MarketingFooter } from '@/components/marketing/Footer';
import { ProductMock } from '@/components/marketing/ProductMock';
import { CtaButtons, FinalCta } from '@/components/marketing/CtaSection';

export const metadata: Metadata = {
  title: 'EnMS — ISO 50001 software for energy managers | Audit-ready EnMS in weeks',
  description:
    'Replace spreadsheets and consultant hours with one ISO 50001:2018 workspace: weather-normalised baselines and EnPIs, SEU identification, action plans and clause-by-clause audit readiness.',
};

const pains = [
  { t: 'Spreadsheet sprawl', d: 'Meter exports, degree-day lookups and regression models scattered across workbooks only one person understands.' },
  { t: 'Consultant dependency', d: 'Paying day rates every time the baseline needs re-running or an EnPI needs explaining to the auditor.' },
  { t: 'Audit-week scramble', d: 'Chasing evidence for Clauses 4–10 across inboxes and shared drives days before the certification body arrives.' },
  { t: 'Baselines nobody trusts', d: 'Weather swings hide real savings — or invent them — when consumption is not normalised for HDD and CDD.' },
];

const steps = [
  { icon: Upload, t: 'Upload meter data', d: 'Drop in CSV exports from your BMS, utility portal or AMR. We validate units, gaps and intervals.' },
  { icon: CloudSun, t: 'Auto weather-normalised baseline', d: 'Local HDD/CDD fetched automatically and regressed against consumption, with R², CV(RMSE) and p-values.' },
  { icon: Target, t: 'SEUs & EnPIs', d: 'Pareto-ranked significant energy uses and EnPIs tracked against the expected baseline, month by month.' },
  { icon: ClipboardCheck, t: 'Audit-ready evidence', d: 'Objectives, action plans and reports linked to each clause, with a live readiness score.' },
];

const features = [
  { icon: Layers, clause: 'Clause 4', t: 'Context & scope', d: 'Define sites, boundaries and energy types once; every analysis is scoped to them.' },
  { icon: Users, clause: 'Clause 5', t: 'Leadership & roles', d: 'Owner, Manager and Viewer roles with an audit log of who changed what.' },
  { icon: Gauge, clause: 'Clause 6.3–6.5', t: 'Energy review & baselines', d: 'Weather-normalised EnB with statistical validation and documented adjustment rules.' },
  { icon: BarChart3, clause: 'Clause 6.4', t: 'EnPIs that hold up', d: 'Regression-based EnPIs comparing actual vs expected consumption, not raw kWh.' },
  { icon: ListChecks, clause: 'Clause 6.2', t: 'Objectives & action plans', d: 'Targets, owners, deadlines and verified savings tied back to the SEU they address.' },
  { icon: ShieldCheck, clause: 'Clauses 9–10', t: 'Audit & improvement', d: 'Internal audit checklists, nonconformities and evidence of continual improvement.' },
];

const faqs = [
  { q: 'Does EnMS certify us to ISO 50001?', a: 'No software can — certification is granted by an accredited certification body. EnMS gives you the energy review, baselines, EnPIs, action plans and documented evidence the auditor will ask for, organised by clause.' },
  { q: 'What data do I need to get started?', a: 'At minimum, 12 months of energy consumption (monthly, daily or interval) per meter or site, plus a postcode or coordinates. We fetch heating and cooling degree days for you.' },
  { q: 'How is the baseline weather-normalised?', a: 'We fit a regression of consumption against HDD and CDD (and optional production variables), report R², CV(RMSE) and significance, and use the model to calculate expected consumption for every reporting period.' },
  { q: 'Can our consultant or auditor get access?', a: 'Yes. Invite them as a Viewer to read reports and evidence without editing rights. Every action is recorded in the audit log.' },
  { q: 'Is our data kept separate from other customers?', a: 'Every workspace is a separate tenant. Uploads, runs and reports are scoped to your organisation and access is enforced at the API on every request.' },
  { q: 'We already have ISO 14001 or 9001 — does that help?', a: 'Yes. ISO 50001:2018 shares the Annex SL high-level structure, so your management review, document control and internal audit processes carry over. EnMS focuses on the energy-specific requirements.' },
];

const quotes = [
  { q: 'Our baseline used to live in a spreadsheet only our consultant could run. Now the regression updates every month and I can explain it to the auditor myself.', r: 'Energy Manager, food manufacturing' },
  { q: 'Seeing readiness by clause told us exactly where our gaps were months before the surveillance audit — not days.', r: 'Sustainability Lead, multi-site retail' },
  { q: 'Weather normalisation finally separated real savings from a mild winter. That changed how we report to the board.', r: 'Head of Facilities, higher education' },
];

const checklist: [string, string, boolean][] = [
  ['6.3', 'Energy review & SEUs', true],
  ['6.5', 'Energy baseline (EnB)', true],
  ['6.4', 'EnPIs defined & reviewed', true],
  ['6.6', 'Energy data collection plan', false],
  ['8.3', 'Design considerations for SEUs', false],
  ['9.2', 'Internal audit record', false],
  ['9.3', 'Management review', true],
];

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-white">
      <MarketingNav />

      {/* Hero */}
      <section className="relative overflow-hidden">
        <div className="pointer-events-none absolute inset-0 marketing-grid" />
        <div className="pointer-events-none absolute -top-40 right-0 h-96 w-96 rounded-full bg-brand-200/40 blur-3xl" />
        <div className="relative mx-auto grid max-w-7xl items-center gap-14 px-6 pb-20 pt-16 lg:grid-cols-2 lg:pt-24">
          <div>
            <Link href="/iso-50001" className="inline-flex items-center gap-2 rounded-full border border-brand-200 bg-brand-50 px-3 py-1 text-xs font-semibold text-brand-700 hover:bg-brand-100">
              Built for ISO 50001:2018 <ArrowRight className="h-3 w-3" />
            </Link>
            <h1 className="mt-6 text-4xl font-bold leading-[1.08] tracking-tight text-ink-950 sm:text-5xl lg:text-[3.5rem]">
              ISO 50001 certification,{' '}
              <span className="bg-gradient-to-r from-brand-600 to-emerald-500 bg-clip-text text-transparent">without the spreadsheet scramble</span>
            </h1>
            <p className="mt-6 max-w-xl text-lg leading-relaxed text-ink-600">
              EnMS turns raw meter data into weather-normalised baselines, significant energy uses and EnPIs —
              then organises the evidence for Clauses 4–10 so you are audit-ready all year, not just audit week.
            </p>
            <div className="mt-8"><CtaButtons /></div>
            <ul className="mt-8 flex flex-wrap gap-x-6 gap-y-2 text-sm text-ink-500">
              {['14-day free trial', 'No credit card', 'Results from your first upload'].map((t) => (
                <li key={t} className="flex items-center gap-1.5"><CheckCircle2 className="h-4 w-4 text-brand-600" /> {t}</li>
              ))}
            </ul>
          </div>
          <ProductMock />
        </div>
      </section>

      {/* Credibility strip */}
      <section className="border-y border-ink-100 bg-ink-50/60">
        <div className="mx-auto grid max-w-7xl grid-cols-2 gap-6 px-6 py-8 text-center md:grid-cols-4">
          {[
            ['Clauses 4–10', 'mapped to product features'],
            ['HDD / CDD', 'regression baselines'],
            ['Owner · Manager · Viewer', 'role-based access'],
            ['Tenant-isolated', 'data per organisation'],
          ].map(([a, b]) => (
            <div key={a}>
              <p className="text-base font-semibold text-ink-950">{a}</p>
              <p className="text-sm text-ink-500">{b}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Problem */}
      <section className="mx-auto max-w-7xl px-6 py-24">
        <div className="max-w-2xl">
          <p className="text-sm font-semibold uppercase tracking-wider text-brand-600">The problem</p>
          <h2 className="mt-3 text-3xl font-bold tracking-tight text-ink-950 sm:text-4xl">
            Running an EnMS shouldn&apos;t depend on one workbook and a consultant&apos;s calendar
          </h2>
        </div>
        <div className="mt-12 grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
          {pains.map((p) => (
            <div key={p.t} className="rounded-2xl border border-ink-100 bg-white p-6">
              <XCircle className="h-5 w-5 text-rose-400" />
              <h3 className="mt-4 font-semibold text-ink-950">{p.t}</h3>
              <p className="mt-2 text-sm leading-relaxed text-ink-500">{p.d}</p>
            </div>
          ))}
        </div>
      </section>

      {/* How it works */}
      <section id="how-it-works" className="scroll-mt-20 bg-ink-950 py-24 text-white">
        <div className="mx-auto max-w-7xl px-6">
          <div className="max-w-2xl">
            <p className="text-sm font-semibold uppercase tracking-wider text-brand-400">How it works</p>
            <h2 className="mt-3 text-3xl font-bold tracking-tight sm:text-4xl">From meter export to audit evidence in four steps</h2>
          </div>
          <ol className="mt-14 grid gap-6 md:grid-cols-2 lg:grid-cols-4">
            {steps.map((s, i) => (
              <li key={s.t} className="relative rounded-2xl border border-white/10 bg-white/[0.03] p-6">
                <div className="flex items-center justify-between">
                  <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-brand-500/15 text-brand-300">
                    <s.icon className="h-5 w-5" />
                  </span>
                  <span className="font-mono text-xs text-ink-400">0{i + 1}</span>
                </div>
                <h3 className="mt-5 font-semibold">{s.t}</h3>
                <p className="mt-2 text-sm leading-relaxed text-ink-300">{s.d}</p>
              </li>
            ))}
          </ol>
        </div>
      </section>

      {/* Features */}
      <section id="features" className="mx-auto max-w-7xl scroll-mt-20 px-6 py-24">
        <div className="flex flex-col justify-between gap-6 md:flex-row md:items-end">
          <div className="max-w-2xl">
            <p className="text-sm font-semibold uppercase tracking-wider text-brand-600">Features</p>
            <h2 className="mt-3 text-3xl font-bold tracking-tight text-ink-950 sm:text-4xl">Every feature maps to a clause your auditor will check</h2>
          </div>
          <Link href="/iso-50001#clauses" className="inline-flex items-center gap-1.5 text-sm font-semibold text-brand-700 hover:text-brand-800">
            See the full clause mapping <ArrowRight className="h-4 w-4" />
          </Link>
        </div>
        <div className="mt-12 grid gap-px overflow-hidden rounded-2xl border border-ink-100 bg-ink-100 sm:grid-cols-2 lg:grid-cols-3">
          {features.map((f) => (
            <div key={f.t} className="bg-white p-7 transition hover:bg-ink-50/60">
              <div className="flex items-center justify-between">
                <f.icon className="h-5 w-5 text-brand-600" />
                <span className="rounded-full bg-ink-50 px-2.5 py-0.5 text-[11px] font-semibold text-ink-500 ring-1 ring-ink-100">{f.clause}</span>
              </div>
              <h3 className="mt-5 font-semibold text-ink-950">{f.t}</h3>
              <p className="mt-2 text-sm leading-relaxed text-ink-500">{f.d}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Readiness */}
      <section id="readiness" className="scroll-mt-20 border-y border-ink-100 bg-gradient-to-b from-brand-50/60 to-white py-24">
        <div className="mx-auto grid max-w-7xl items-center gap-14 px-6 lg:grid-cols-2">
          <div>
            <p className="text-sm font-semibold uppercase tracking-wider text-brand-600">Compliance readiness</p>
            <h2 className="mt-3 text-3xl font-bold tracking-tight text-ink-950 sm:text-4xl">Know where you stand, clause by clause</h2>
            <p className="mt-4 text-lg leading-relaxed text-ink-600">
              The readiness score checks your workspace against the requirements of ISO 50001:2018 and tells you
              exactly which evidence is missing — before your Stage 1 audit, surveillance visit or recertification.
            </p>
            <ul className="mt-8 space-y-3">
              {[
                'Energy review complete with SEUs identified and justified',
                'Baseline statistically valid, with documented adjustment criteria',
                'EnPIs reported against expected consumption each period',
                'Objectives, targets and action plans with owners and dates',
                'Internal audit and management review records attached',
              ].map((t) => (
                <li key={t} className="flex items-start gap-3 text-ink-700">
                  <CheckCircle2 className="mt-0.5 h-5 w-5 flex-none text-brand-600" /> {t}
                </li>
              ))}
            </ul>
          </div>
          <div className="rounded-2xl border border-ink-100 bg-white p-6 shadow-xl shadow-ink-900/5">
            <div className="flex items-center justify-between">
              <p className="font-semibold text-ink-950">Readiness checklist</p>
              <span className="rounded-full bg-amber-50 px-2.5 py-0.5 text-xs font-semibold text-amber-700 ring-1 ring-amber-100">3 gaps</span>
            </div>
            <ul className="mt-5 divide-y divide-ink-100">
              {checklist.map(([c, t, ok]) => (
                <li key={c} className="flex items-center gap-3 py-3 text-sm">
                  <span className="w-10 font-mono text-xs text-ink-400">{c}</span>
                  <span className="flex-1 text-ink-800">{t}</span>
                  {ok ? (
                    <span className="flex items-center gap-1 text-xs font-medium text-brand-700"><CheckCircle2 className="h-4 w-4" /> Evidence</span>
                  ) : (
                    <span className="text-xs font-medium text-amber-600">Action needed</span>
                  )}
                </li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      {/* Testimonials */}
      <section className="mx-auto max-w-7xl px-6 py-24">
        <div className="max-w-2xl">
          <p className="text-sm font-semibold uppercase tracking-wider text-brand-600">Why teams switch</p>
          <h2 className="mt-3 text-3xl font-bold tracking-tight text-ink-950 sm:text-4xl">Built around how energy managers actually work</h2>
        </div>
        <div className="mt-12 grid gap-6 md:grid-cols-3">
          {quotes.map((q) => (
            <figure key={q.r} className="flex flex-col justify-between rounded-2xl border border-ink-100 bg-ink-50/40 p-7">
              <div>
                <Quote className="h-5 w-5 text-brand-500" />
                <blockquote className="mt-4 leading-relaxed text-ink-700">{q.q}</blockquote>
              </div>
              <figcaption className="mt-6 text-sm font-semibold text-ink-950">{q.r}</figcaption>
            </figure>
          ))}
        </div>
        <p className="mt-6 text-xs text-ink-400">Illustrative scenarios based on common energy-manager workflows.</p>
      </section>

      {/* Spreadsheet vs EnMS */}
      <section className="mx-auto max-w-5xl px-6 pb-8">
        <div className="grid overflow-hidden rounded-2xl border border-ink-100 md:grid-cols-2">
          <div className="bg-ink-50/60 p-8">
            <p className="flex items-center gap-2 font-semibold text-ink-700"><FileSpreadsheet className="h-5 w-5 text-ink-400" /> Spreadsheets + consultant</p>
            <ul className="mt-5 space-y-2.5 text-sm text-ink-500">
              {['Manual degree-day lookups', 'Baseline rebuilt each year', 'Evidence in email threads', 'Readiness unknown until the audit'].map((t) => (
                <li key={t} className="flex gap-2"><XCircle className="h-4 w-4 flex-none text-ink-300" /> {t}</li>
              ))}
            </ul>
          </div>
          <div className="bg-white p-8">
            <p className="flex items-center gap-2 font-semibold text-ink-950"><ShieldCheck className="h-5 w-5 text-brand-600" /> EnMS</p>
            <ul className="mt-5 space-y-2.5 text-sm text-ink-700">
              {['HDD/CDD fetched automatically', 'Baseline re-validated every run', 'Evidence linked to each clause', 'Live readiness score'].map((t) => (
                <li key={t} className="flex gap-2"><CheckCircle2 className="h-4 w-4 flex-none text-brand-600" /> {t}</li>
              ))}
            </ul>
          </div>
        </div>
      </section>

      {/* FAQ */}
      <section id="faq" className="mx-auto max-w-3xl scroll-mt-20 px-6 py-24">
        <h2 className="text-center text-3xl font-bold tracking-tight text-ink-950 sm:text-4xl">Frequently asked questions</h2>
        <div className="mt-12 divide-y divide-ink-100 rounded-2xl border border-ink-100">
          {faqs.map((f) => (
            <details key={f.q} className="group p-6 [&_summary::-webkit-details-marker]:hidden">
              <summary className="flex cursor-pointer list-none items-center justify-between gap-4 font-semibold text-ink-950">
                {f.q}
                <span className="flex h-6 w-6 flex-none items-center justify-center rounded-full bg-ink-50 text-ink-500 transition group-open:rotate-45">+</span>
              </summary>
              <p className="mt-3 text-sm leading-relaxed text-ink-600">{f.a}</p>
            </details>
          ))}
        </div>
        <p className="mt-6 text-center text-sm text-ink-500">
          Questions about pricing? <Link href="/pricing" className="font-semibold text-brand-700 hover:text-brand-800">Compare plans</Link>
        </p>
      </section>

      <FinalCta />
      <MarketingFooter />
    </div>
  );
}
