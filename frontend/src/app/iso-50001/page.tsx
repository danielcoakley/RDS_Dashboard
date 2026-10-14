import { FileCheck2 } from 'lucide-react';
import Nav from '@/components/marketing/Nav';
import Footer from '@/components/marketing/Footer';
import { CTAButtons, MarketingShell, SectionHeading } from '@/components/marketing/ui';
import { clauses } from '@/components/marketing/clauses';

export const metadata = { title: 'ISO 50001:2018 clause map — EnMS' };

export default function IsoPage() {
  return (
    <MarketingShell>
      <Nav />
      <main className="mx-auto max-w-5xl px-6 py-20 lg:py-28">
        <SectionHeading eyebrow="ISO 50001:2018" title="Every clause, mapped to evidence." sub="See how EnMS supports each requirement of clauses 4 to 10 — and what evidence it produces for your certification audit." />
        <ol className="relative mt-16 space-y-4 border-l border-white/10 pl-8">
          {clauses.map((c) => (
            <li key={c.n} className="group relative">
              <span className={`absolute -left-[37px] top-7 h-2.5 w-2.5 rounded-full ${c.status === 'done' ? 'bg-brand-400 shadow-[0_0_8px] shadow-brand-400' : 'bg-amber-400'}`} />
              <div className="rounded-lg border border-white/10 bg-white/[0.03] p-6 backdrop-blur transition group-hover:border-brand-400/40 group-hover:bg-white/[0.05]">
                <div className="flex flex-wrap items-baseline gap-x-4">
                  <span className="font-mono text-sm text-brand-400">Clause {c.n}</span>
                  <h3 className="text-lg font-semibold text-white">{c.title}</h3>
                </div>
                <p className="mt-2 text-ink-300">{c.desc}</p>
                <div className="grid grid-rows-[0fr] transition-all duration-300 group-hover:grid-rows-[1fr] group-focus-within:grid-rows-[1fr]">
                  <div className="overflow-hidden">
                    <p className="mt-4 flex items-start gap-2 rounded-sm border border-white/5 bg-ink-950/60 p-3 font-mono text-xs text-ink-200">
                      <FileCheck2 className="h-4 w-4 shrink-0 text-brand-400" /> Evidence generated: {c.evidence}
                    </p>
                  </div>
                </div>
              </div>
            </li>
          ))}
        </ol>
        <div className="mt-20 rounded-lg border border-white/10 bg-ink-900/50 p-8 text-center">
          <h2 className="text-2xl font-semibold text-white">See your own clause readiness score.</h2>
          <div className="mt-6"><CTAButtons center /></div>
        </div>
      </main>
      <Footer />
    </MarketingShell>
  );
}
