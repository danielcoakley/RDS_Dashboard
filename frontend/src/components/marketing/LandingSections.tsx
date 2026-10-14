import { FileSpreadsheet, Receipt, AlarmClock, CloudSun, Upload, LineChart, Target, ShieldCheck, Quote } from 'lucide-react';
import ProductMock from './ProductMock';
import { CTAButtons, Eyebrow, SectionHeading } from './ui';
import { clauses } from './clauses';
import FAQ from './FAQ';

const wrap = 'mx-auto max-w-7xl px-6';

export function Hero() {
  return (
    <section className="relative overflow-hidden border-b border-white/10">
      <div className="mk-grid pointer-events-none absolute inset-0" aria-hidden />
      <div className={`${wrap} relative grid items-center gap-16 py-20 lg:grid-cols-2 lg:py-32`}>
        <div>
          <span className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 font-mono text-xs text-ink-200">
            <span className="h-1.5 w-1.5 rounded-full bg-brand-400 shadow-[0_0_6px] shadow-brand-400" /> Built for ISO 50001:2018
          </span>
          <h1 className="mt-6 text-4xl font-semibold leading-[1.1] tracking-tight text-white sm:text-5xl lg:text-6xl">
            From meter data to audit-ready EnMS. <span className="text-brand-400">Without the spreadsheets.</span>
          </h1>
          <p className="mt-6 max-w-xl text-lg leading-relaxed text-ink-300">
            Upload your meter data and get weather-normalised baselines, automatic SEU identification and a clause-by-clause readiness score — all the evidence your certification auditor asks for, in one place.
          </p>
          <div className="mt-10"><CTAButtons /></div>
          <p className="mt-4 font-mono text-xs text-ink-400">14-day free trial · No credit card required</p>
        </div>
        <div className="pb-10 lg:pl-8"><ProductMock /></div>
      </div>
    </section>
  );
}

const problems = [
  { icon: FileSpreadsheet, title: 'Spreadsheet sprawl', desc: 'Energy reviews spread across dozens of workbooks, with no version history an auditor will trust.' },
  { icon: Receipt, title: 'Consultant costs', desc: 'Paying day rates for someone to rebuild the same regression model every year.' },
  { icon: AlarmClock, title: 'Audit scramble', desc: 'Weeks spent chasing evidence for each clause before the certification body arrives.' },
  { icon: CloudSun, title: 'Baselines are hard', desc: 'Weather-normalising with HDD/CDD regression needs statistics most teams don’t have time for.' },
];

export function Problem() {
  return (
    <section className={`${wrap} py-24 lg:py-32`}>
      <SectionHeading eyebrow="The problem" title="ISO 50001 shouldn’t live in spreadsheets." sub="Energy managers spend more time assembling evidence than improving performance." />
      <div className="mt-14 grid gap-px overflow-hidden rounded-lg border border-white/10 bg-white/10 sm:grid-cols-2 lg:grid-cols-4">
        {problems.map((p) => (
          <div key={p.title} className="bg-ink-950 p-6">
            <p.icon className="h-5 w-5 text-amber-400" />
            <h3 className="mt-4 font-semibold text-white">{p.title}</h3>
            <p className="mt-2 text-sm leading-relaxed text-ink-300">{p.desc}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

const steps = [
  { icon: Upload, title: 'Upload meter data', desc: 'Drop in CSV exports from your BMS, AMR or utility portal.' },
  { icon: CloudSun, title: 'Auto weather-normalised baseline', desc: 'We fetch local weather and fit HDD/CDD regression models per meter.' },
  { icon: LineChart, title: 'SEUs & EnPIs', desc: 'Significant energy uses are identified and EnPIs tracked against baseline.' },
  { icon: ShieldCheck, title: 'Audit-ready evidence', desc: 'Every clause mapped to evidence, exportable for your certification body.' },
];

export function HowItWorks() {
  return (
    <section id="how" className="border-y border-white/10 bg-ink-900/40">
      <div className={`${wrap} py-24 lg:py-32`}>
        <SectionHeading eyebrow="How it works" title="Four steps from raw data to certification evidence." />
        <ol className="relative mt-14 grid gap-8 md:grid-cols-4">
          <div className="mk-flow absolute left-0 right-0 top-6 hidden h-px md:block" aria-hidden />
          {steps.map((s, i) => (
            <li key={s.title} className="relative">
              <div className="flex h-12 w-12 items-center justify-center rounded-sm border border-brand-400/40 bg-ink-950 text-brand-400">
                <s.icon className="h-5 w-5" />
              </div>
              <p className="mt-5 font-mono text-xs text-ink-400">STEP 0{i + 1}</p>
              <h3 className="mt-1 font-semibold text-white">{s.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-ink-300">{s.desc}</p>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}

export function FeatureGrid() {
  return (
    <section id="features" className={`${wrap} py-24 lg:py-32`}>
      <SectionHeading eyebrow="Features" title="Mapped to every clause, 4 to 10." sub="Each capability produces evidence for a specific requirement of ISO 50001:2018." />
      <div className="mt-14 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {clauses.map((c) => (
          <div key={c.n} className="group rounded-lg border border-white/10 bg-ink-900/50 p-6 transition hover:border-brand-400/40">
            <span className="font-mono text-xs text-brand-400">CLAUSE {c.n} · {c.title}</span>
            <h3 className="mt-3 text-lg font-semibold text-white">{c.feature}</h3>
            <p className="mt-2 text-sm leading-relaxed text-ink-300">{c.desc}</p>
          </div>
        ))}
        <div className="flex flex-col justify-center rounded-lg border border-dashed border-white/15 p-6">
          <Target className="h-5 w-5 text-brand-400" />
          <p className="mt-3 text-sm text-ink-300">See exactly what evidence each clause produces.</p>
          <a href="/iso-50001" className="mt-3 text-sm font-medium text-brand-400 hover:text-brand-300">Explore the clause map →</a>
        </div>
      </div>
    </section>
  );
}

const scores = [['4', 100], ['5', 100], ['6', 94], ['7', 72], ['8', 88], ['9', 65], ['10', 80]] as const;

export function Readiness() {
  return (
    <section id="readiness" className="border-y border-white/10 bg-ink-900/40">
      <div className={`${wrap} grid items-center gap-16 py-24 lg:grid-cols-2 lg:py-32`}>
        <div>
          <SectionHeading eyebrow="Compliance readiness" title="Know exactly where you stand before the auditor does." sub="A live, clause-by-clause readiness score shows which requirements have evidence and which still need work — so stage 1 and stage 2 audits hold no surprises." />
        </div>
        <div className="rounded-lg border border-white/10 bg-ink-950 p-6">
          <div className="flex items-baseline justify-between">
            <span className="font-mono text-xs uppercase tracking-widest text-ink-400">Overall readiness</span>
            <span className="font-mono text-3xl text-white">86%</span>
          </div>
          <ul className="mt-6 space-y-4">
            {scores.map(([n, v]) => (
              <li key={n}>
                <div className="flex justify-between font-mono text-xs">
                  <span className="flex items-center gap-2 text-ink-200">
                    <span className={`h-1.5 w-1.5 rounded-full ${v >= 85 ? 'bg-brand-400 shadow-[0_0_6px] shadow-brand-400' : 'bg-amber-400'}`} />Clause {n}
                  </span>
                  <span className="text-ink-300">{v}%</span>
                </div>
                <div className="mt-1.5 h-1.5 rounded-full bg-white/5">
                  <div className={`h-full rounded-full ${v >= 85 ? 'bg-brand-400' : 'bg-amber-400'}`} style={{ width: `${v}%` }} />
                </div>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}

const quotes = [
  { q: 'Our weather-normalised baseline used to take a consultant two weeks. Now it’s ready the day we upload the data.', who: 'Energy Manager, multi-site manufacturer' },
  { q: 'The readiness score turned our audit prep from a scramble into a checklist.', who: 'Head of Sustainability, logistics group' },
  { q: 'Finally one place for SEUs, EnPIs and evidence — our auditor could follow every number.', who: 'Facilities Director, higher education estate' },
];

export function Quotes() {
  return (
    <section className={`${wrap} py-24 lg:py-32`}>
      <Eyebrow>From energy teams</Eyebrow>
      <div className="mt-10 grid gap-4 md:grid-cols-3">
        {quotes.map((t) => (
          <figure key={t.who} className="rounded-lg border border-white/10 bg-ink-900/50 p-6">
            <Quote className="h-5 w-5 text-brand-400" />
            <blockquote className="mt-4 leading-relaxed text-ink-100">{t.q}</blockquote>
            <figcaption className="mt-6 font-mono text-xs text-ink-400">— {t.who}</figcaption>
          </figure>
        ))}
      </div>
    </section>
  );
}

export function FAQSection() {
  return (
    <section id="faq" className="border-t border-white/10">
      <div className={`${wrap} grid gap-12 py-24 lg:grid-cols-3 lg:py-32`}>
        <SectionHeading eyebrow="FAQ" title="Questions energy managers ask." />
        <div className="lg:col-span-2"><FAQ /></div>
      </div>
    </section>
  );
}

export function FinalCTA() {
  return (
    <section className="relative overflow-hidden border-t border-white/10">
      <div className="mk-grid pointer-events-none absolute inset-0" aria-hidden />
      <div className={`${wrap} relative py-24 text-center lg:py-32`}>
        <h2 className="mx-auto max-w-2xl text-3xl font-semibold tracking-tight text-white sm:text-5xl">Get audit-ready this quarter.</h2>
        <p className="mx-auto mt-4 max-w-xl text-ink-300">Start with one site, see your weather-normalised baseline in minutes.</p>
        <div className="mt-10"><CTAButtons center /></div>
      </div>
    </section>
  );
}
