'use client';

import Link from 'next/link';
import { ArrowRight, ShieldCheck, Clock, Circle } from 'lucide-react';
import type { ComplianceReadiness } from '@/lib/api';

export function RadialScore({ score, size = 132 }: { score: number; size?: number }) {
  const stroke = 12;
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const clamped = Math.max(0, Math.min(100, score));
  const color = clamped >= 75 ? '#0f9359' : clamped >= 40 ? '#d97706' : '#dc2626';
  return (
    <div className="relative shrink-0" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="#ebeef3" strokeWidth={stroke} />
        <circle
          cx={size / 2} cy={size / 2} r={r} fill="none" stroke={color} strokeWidth={stroke}
          strokeLinecap="round" strokeDasharray={c} strokeDashoffset={c * (1 - clamped / 100)}
          style={{ transition: 'stroke-dashoffset 0.8s ease' }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-3xl font-bold text-ink-900">{Math.round(clamped)}%</span>
        <span className="text-[11px] font-medium uppercase tracking-wide text-ink-400">ready</span>
      </div>
    </div>
  );
}

export function SectionBar({ section, title, score, complete, total }: ComplianceReadiness['by_section'][number]) {
  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between gap-2 text-xs">
        <span className="truncate font-medium text-ink-700">
          <span className="text-ink-400">§{section}</span> {title}
        </span>
        <span className="shrink-0 text-ink-400">{complete}/{total}</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-ink-100">
        <div className="h-full rounded-full bg-brand-500 transition-all duration-700" style={{ width: `${score}%` }} />
      </div>
    </div>
  );
}

export default function ReadinessCard({ data, error }: { data: ComplianceReadiness | null; error?: string | null }) {
  if (!data) {
    return (
      <div className="rounded-2xl border border-ink-100 bg-white p-6 text-sm text-ink-500">
        {error ? `Audit readiness unavailable: ${error}` : 'Loading audit readiness…'}
      </div>
    );
  }

  return (
    <section className="rounded-2xl border border-ink-100 bg-white p-6 shadow-sm">
      <div className="mb-5 flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <ShieldCheck className="h-5 w-5 text-brand-600" />
          <h2 className="font-semibold text-ink-900">ISO 50001 audit readiness</h2>
        </div>
        <Link href="/compliance" className="inline-flex items-center gap-1 text-sm font-medium text-brand-700 hover:text-brand-800">
          Open checklist <ArrowRight className="h-3.5 w-3.5" />
        </Link>
      </div>

      <div className="grid gap-6 lg:grid-cols-[auto_1fr_1fr]">
        <div className="flex items-center gap-5">
          <RadialScore score={data.score} />
          <dl className="space-y-1.5 text-sm">
            <Stat color="bg-brand-500" label="Complete" value={data.complete} />
            <Stat color="bg-amber-500" label="In progress" value={data.in_progress} />
            <Stat color="bg-ink-300" label="Not started" value={data.not_started} />
          </dl>
        </div>

        <div className="space-y-2.5">
          <p className="text-xs font-semibold uppercase tracking-wide text-ink-400">By clause</p>
          {data.by_section.map((s) => <SectionBar key={s.section} {...s} />)}
        </div>

        <div>
          <p className="mb-2.5 text-xs font-semibold uppercase tracking-wide text-ink-400">Next actions</p>
          {data.next_actions.length === 0 ? (
            <p className="text-sm text-ink-500">All clauses complete. You&apos;re audit ready.</p>
          ) : (
            <ul className="space-y-1.5">
              {data.next_actions.slice(0, 5).map((a) => (
                <li key={a.clause_ref}>
                  <Link
                    href={`/compliance#clause-${a.clause_ref}`}
                    className="flex items-center gap-2 rounded-lg px-2 py-1.5 text-sm text-ink-700 hover:bg-ink-50"
                  >
                    {a.status === 'in_progress'
                      ? <Clock className="h-4 w-4 shrink-0 text-amber-500" />
                      : <Circle className="h-4 w-4 shrink-0 text-ink-300" />}
                    <span className="shrink-0 font-medium text-ink-400">§{a.clause_ref}</span>
                    <span className="truncate">{a.clause_title}</span>
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </section>
  );
}

function Stat({ color, label, value }: { color: string; label: string; value: number }) {
  return (
    <div className="flex items-center gap-2">
      <span className={`h-2.5 w-2.5 rounded-full ${color}`} />
      <dt className="text-ink-500">{label}</dt>
      <dd className="ml-auto pl-3 font-semibold text-ink-900">{value}</dd>
    </div>
  );
}
