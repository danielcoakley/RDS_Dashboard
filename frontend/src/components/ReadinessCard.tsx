'use client';

import Link from 'next/link';
import { ArrowRight, ShieldCheck } from 'lucide-react';

function Radial({ score }: { score: number }) {
  const r = 52;
  const c = 2 * Math.PI * r;
  return (
    <div className="relative h-36 w-36 shrink-0">
      <svg viewBox="0 0 120 120" className="h-full w-full -rotate-90">
        <circle cx="60" cy="60" r={r} fill="none" strokeWidth="10" className="stroke-ink-100" />
        <circle
          cx="60" cy="60" r={r} fill="none" strokeWidth="10" strokeLinecap="round"
          className="stroke-brand-600 transition-all duration-700"
          strokeDasharray={c} strokeDashoffset={c * (1 - score / 100)}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-3xl font-bold text-ink-900">{score}%</span>
        <span className="text-[11px] font-medium uppercase tracking-wide text-ink-400">Ready</span>
      </div>
    </div>
  );
}

export default function ReadinessCard({ data }: { data: any }) {
  return (
    <div className="mb-8 rounded-2xl border border-ink-100 bg-white p-6 shadow-sm">
      <div className="mb-5 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <ShieldCheck className="h-5 w-5 text-brand-600" />
          <h2 className="font-semibold text-ink-900">ISO 50001 audit readiness</h2>
        </div>
        <Link href="/compliance" className="text-sm font-medium text-brand-600 hover:text-brand-700">
          Open checklist →
        </Link>
      </div>
      <div className="grid gap-8 lg:grid-cols-[auto_1fr_1fr]">
        <div className="flex flex-col items-center gap-3">
          <Radial score={data.score} />
          <div className="flex gap-4 text-xs text-ink-500">
            <span><b className="text-brand-600">{data.counts.complete}</b> done</span>
            <span><b className="text-amber-600">{data.counts.in_progress}</b> in progress</span>
            <span><b className="text-ink-700">{data.counts.not_started}</b> to do</span>
          </div>
        </div>
        <div className="space-y-2.5">
          <p className="text-xs font-semibold uppercase tracking-wide text-ink-400">By clause section</p>
          {data.by_section.map((s: any) => (
            <div key={s.section}>
              <div className="mb-1 flex justify-between text-xs">
                <span className="truncate text-ink-600">§{s.section} {s.title}</span>
                <span className="ml-2 font-medium text-ink-900">{s.score}%</span>
              </div>
              <div className="h-1.5 overflow-hidden rounded-full bg-ink-100">
                <div className="h-full rounded-full bg-brand-500" style={{ width: `${s.score}%` }} />
              </div>
            </div>
          ))}
        </div>
        <div>
          <p className="mb-2.5 text-xs font-semibold uppercase tracking-wide text-ink-400">Next actions</p>
          {data.next_actions.length === 0 ? (
            <p className="text-sm text-ink-500">All clauses complete — you're audit ready.</p>
          ) : (
            <ul className="space-y-1.5">
              {data.next_actions.map((a: any) => (
                <li key={a.clause_ref}>
                  <Link
                    href={`/compliance#clause-${a.clause_ref}`}
                    className="group flex items-center justify-between gap-2 rounded-lg border border-ink-100 px-3 py-2 text-sm hover:border-brand-200 hover:bg-brand-50"
                  >
                    <span className="truncate text-ink-700">
                      <b className="text-ink-900">§{a.clause_ref}</b> {a.clause_title}
                    </span>
                    <span className="flex shrink-0 items-center gap-1">
                      <span className={`rounded-full px-2 py-0.5 text-[10px] font-medium ${a.status === 'in_progress' ? 'bg-amber-50 text-amber-700' : 'bg-ink-100 text-ink-500'}`}>
                        {a.status === 'in_progress' ? 'In progress' : 'Not started'}
                      </span>
                      <ArrowRight className="h-3.5 w-3.5 text-ink-300 group-hover:text-brand-600" />
                    </span>
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}
