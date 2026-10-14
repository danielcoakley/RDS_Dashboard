import type { Metadata } from 'next';
import Link from 'next/link';
import { ArrowRight, RefreshCw } from 'lucide-react';
import { MarketingNav } from '@/components/marketing/Nav';
import { MarketingFooter } from '@/components/marketing/Footer';
import { CtaButtons, FinalCta } from '@/components/marketing/CtaSection';

export const metadata: Metadata = {
  title: 'ISO 50001:2018 explained — clause 4 to 10 requirements | EnMS',
  description:
    'What ISO 50001:2018 requires from an energy management system: energy review, SEUs, baselines, EnPIs, objectives, internal audit and management review — and how EnMS maps each clause to evidence.',
};

const clauses = [
  {
    n: '4',
    t: 'Context of the organisation',
    req: 'Understand internal and external issues, interested parties, and define the scope and boundaries of the EnMS.',
    feature: 'Organisation, site and boundary setup; energy types and meters scoped per site.',
  },
  {
    n: '5',
    t: 'Leadership',
    req: 'Top management commitment, an energy policy, and assigned roles including the energy management team.',
    feature: 'Owner / Manager / Viewer roles, energy policy record and team responsibilities with an audit log.',
  },
  {
    n: '6',
    t: 'Planning',
    req: 'Actions for risks and opportunities, objectives and action plans, energy review, SEUs, EnPIs, energy baseline and data collection plan.',
    feature: 'Automated energy review, Pareto SEU ranking, HDD/CDD regression baselines with R² and CV(RMSE), EnPI tracking and action plans.',
  },
  {
    n: '7',
    t: 'Support',
    req: 'Resources, competence, awareness, communication and control of documented information.',
    feature: 'Versioned uploads, run history and report artifacts stored per tenant as controlled evidence.',
  },
  {
    n: '8',
    t: 'Operation',
    req: 'Operational planning and control of SEUs, design considerations and energy-related procurement.',
    feature: 'SEU operating criteria and deviation flags when actual consumption exceeds the expected baseline.',
  },
  {
    n: '9',
    t: 'Performance evaluation',
    req: 'Monitoring, measurement and analysis of energy performance, evaluation of compliance, internal audit and management review.',
    feature: 'Actual vs expected EnPI reporting, internal audit checklists and management review inputs generated from your data.',
  },
  {
    n: '10',
    t: 'Improvement',
    req: 'Address nonconformities with corrective action and demonstrate continual improvement of energy performance.',
    feature: 'Nonconformity log, corrective actions and verified savings trend versus the baseline.',
  },
];

const terms = [
  ['EnMS', 'Energy management system — the policies, processes and records you use to improve energy performance.'],
  ['SEU', 'Significant energy use — equipment or processes that account for substantial consumption or improvement potential.'],
  ['EnB', 'Energy baseline — the quantitative reference used to compare energy performance over time.'],
  ['EnPI', 'Energy performance indicator — a measure of energy performance, ideally normalised for relevant variables.'],
  ['HDD / CDD', 'Heating and cooling degree days — weather variables used to normalise consumption.'],
];

export default function Iso50001Page() {
  return (
    <div className="min-h-screen bg-white">
      <MarketingNav />

      <section className="relative overflow-hidden">
        <div className="pointer-events-none absolute inset-0 marketing-grid" />
        <div className="relative mx-auto max-w-4xl px-6 pb-16 pt-20 text-center">
          <p className="text-sm font-semibold uppercase tracking-wider text-brand-600">ISO 50001:2018 explained</p>
          <h1 className="mt-3 text-4xl font-bold tracking-tight text-ink-950 sm:text-5xl">
            What the standard asks for — and how to evidence it
          </h1>
          <p className="mx-auto mt-5 max-w-2xl text-lg text-ink-600">
            ISO 50001 sets out the requirements for an energy management system that continually improves energy performance.
            Here is what each clause expects, and where EnMS produces the evidence.
          </p>
          <div className="mt-8"><CtaButtons center /></div>
        </div>
      </section>

      {/* PDCA */}
      <section className="mx-auto max-w-7xl px-6 pb-16">
        <div className="grid gap-4 rounded-2xl border border-ink-100 bg-ink-50/50 p-6 sm:grid-cols-4">
          {[
            ['Plan', 'Clauses 4–7', 'Context, leadership, energy review, baseline, EnPIs, objectives'],
            ['Do', 'Clause 8', 'Operate and control SEUs, design and procurement'],
            ['Check', 'Clause 9', 'Monitor EnPIs, internal audit, management review'],
            ['Act', 'Clause 10', 'Corrective action and continual improvement'],
          ].map(([a, b, c]) => (
            <div key={a} className="rounded-xl bg-white p-5 ring-1 ring-ink-100">
              <p className="flex items-center gap-2 text-sm font-semibold text-brand-700"><RefreshCw className="h-4 w-4" /> {a}</p>
              <p className="mt-1 text-xs font-medium text-ink-400">{b}</p>
              <p className="mt-2 text-sm text-ink-600">{c}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Clause table */}
      <section id="clauses" className="mx-auto max-w-7xl scroll-mt-20 px-6 pb-20">
        <h2 className="text-3xl font-bold tracking-tight text-ink-950">Clause 4–10 mapping</h2>
        <p className="mt-3 max-w-2xl text-ink-600">Clauses 1–3 cover scope, references and definitions. The auditable requirements start at Clause 4.</p>
        <div className="mt-8 overflow-x-auto rounded-2xl border border-ink-100">
          <table className="w-full min-w-[720px] text-left text-sm">
            <thead className="bg-ink-50 text-xs uppercase tracking-wide text-ink-500">
              <tr>
                <th className="px-5 py-3 font-semibold">Clause</th>
                <th className="px-5 py-3 font-semibold">What the standard requires</th>
                <th className="px-5 py-3 font-semibold">How EnMS supports it</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-ink-100">
              {clauses.map((c) => (
                <tr key={c.n} className="align-top">
                  <td className="w-56 px-5 py-5">
                    <span className="inline-flex h-7 w-7 items-center justify-center rounded-lg bg-brand-50 text-xs font-bold text-brand-700 ring-1 ring-brand-100">{c.n}</span>
                    <p className="mt-2 font-semibold text-ink-950">{c.t}</p>
                  </td>
                  <td className="px-5 py-5 text-ink-600">{c.req}</td>
                  <td className="px-5 py-5 text-ink-800">{c.feature}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Glossary */}
      <section className="border-y border-ink-100 bg-ink-50/50 py-20">
        <div className="mx-auto max-w-7xl px-6">
          <h2 className="text-3xl font-bold tracking-tight text-ink-950">Key terms</h2>
          <dl className="mt-8 grid gap-5 sm:grid-cols-2 lg:grid-cols-5">
            {terms.map(([t, d]) => (
              <div key={t} className="rounded-2xl bg-white p-5 ring-1 ring-ink-100">
                <dt className="font-mono text-sm font-bold text-brand-700">{t}</dt>
                <dd className="mt-2 text-sm leading-relaxed text-ink-600">{d}</dd>
              </div>
            ))}
          </dl>
        </div>
      </section>

      <section className="mx-auto max-w-3xl px-6 py-20 text-center">
        <h2 className="text-2xl font-bold tracking-tight text-ink-950">Why weather normalisation matters</h2>
        <p className="mt-4 leading-relaxed text-ink-600">
          ISO 50001 requires EnPIs and baselines to be normalised when relevant variables significantly affect energy use.
          For most buildings and many processes, weather is the biggest one. EnMS fetches local heating and cooling degree days
          and fits a regression model so you can show auditors genuine performance improvement — not a mild winter.
        </p>
        <Link href="/demo" className="mt-6 inline-flex items-center gap-1.5 text-sm font-semibold text-brand-700 hover:text-brand-800">
          See a weather-normalised baseline in the live demo <ArrowRight className="h-4 w-4" />
        </Link>
      </section>

      <FinalCta />
      <MarketingFooter />
    </div>
  );
}
