'use client';

import dynamic from 'next/dynamic';
import { Flame, Zap, TrendingDown, TrendingUp, Lightbulb } from 'lucide-react';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false }) as any;

const kwh = (v: number | null | undefined) => (v == null ? '—' : `${Math.round(v).toLocaleString()} kWh`);
const eur = (v: number | null | undefined) => (v == null ? '—' : `€${Math.round(v).toLocaleString()}`);
const pct = (v: number | null | undefined) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`);
const r2 = (v: number | null | undefined) => (v == null ? 'n/a' : v.toFixed(2));
const num = (v: number | null | undefined) => (v == null ? '—' : Math.round(v).toLocaleString());

function Card({ title, clause, children }: { title: string; clause?: string; children: React.ReactNode }) {
  return (
    <section className="rounded-xl border border-ink-100 bg-white p-6">
      <div className="mb-4 flex items-baseline justify-between gap-4">
        <h2 className="text-lg font-semibold text-ink-900">{title}</h2>
        {clause && <span className="shrink-0 text-xs font-medium text-ink-400">ISO 50001 §{clause}</span>}
      </div>
      {children}
    </section>
  );
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-xl border border-ink-100 bg-white p-5">
      <p className="text-xs font-medium uppercase tracking-wide text-ink-400">{label}</p>
      <p className="mt-1 text-2xl font-bold text-ink-900">{value}</p>
      {sub && <p className="mt-1 text-xs text-ink-500">{sub}</p>}
    </div>
  );
}

export function SignificanceBadge({ significant }: { significant: boolean }) {
  return significant ? (
    <span className="inline-flex rounded-full bg-amber-100 px-2.5 py-0.5 text-xs font-semibold text-amber-800">Significant</span>
  ) : (
    <span className="inline-flex rounded-full bg-ink-100 px-2.5 py-0.5 text-xs font-medium text-ink-500">Non-significant</span>
  );
}

function R2Cell({ v, relevant }: { v: number | null; relevant: boolean }) {
  return <td className={`px-3 py-2 text-right tabular-nums ${relevant ? 'font-semibold text-brand-700' : 'text-ink-500'}`}>{r2(v)}</td>;
}

function Delta({ v }: { v: number | null }) {
  if (v == null) return <span className="text-ink-400">—</span>;
  const good = v >= 0;
  return (
    <span className={`inline-flex items-center gap-1 font-semibold ${good ? 'text-emerald-700' : 'text-red-600'}`}>
      {good ? <TrendingDown className="h-3.5 w-3.5" /> : <TrendingUp className="h-3.5 w-3.5" />}{Math.abs(v)}%
    </span>
  );
}

export default function ReviewReport({ review }: { review: any }) {
  const t = review.enpi_total || {};
  const seus: any[] = review.seus || [];
  const threshold = (review.pareto_threshold || 0.8) * 100;
  const improved = (t.improvement_pct ?? 0) >= 0;

  return (
    <div className="space-y-6">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label={`Total consumption ${review.reporting_year}`} value={kwh(review.total_kwh)} sub={`${eur(review.total_cost_eur)} est. cost`} />
        <Stat label="Significant energy uses" value={`${seus.filter(s => s.significant).length} of ${seus.length}`} sub={`Pareto ${threshold}% rule`} />
        <Stat label="EnPI vs baseline" value={t.enpi_ratio != null ? t.enpi_ratio.toFixed(3) : '—'}
          sub={t.improvement_pct != null ? `${Math.abs(t.improvement_pct)}% ${improved ? 'better' : 'worse'} than expected` : undefined} />
        <Stat label="Savings potential" value={eur(review.total_eur_savings)} sub={`${kwh(review.total_kwh_savings)} / year`} />
      </div>

      <Card title="Consumption by energy source" clause="6.3 a)">
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {(review.energy_sources || []).map((s: any) => (
            <div key={s.source} className="flex items-center gap-4 rounded-lg bg-ink-50 p-4">
              <div className="rounded-lg bg-white p-2.5">
                {s.source === 'gas' ? <Flame className="h-5 w-5 text-orange-500" /> : <Zap className="h-5 w-5 text-brand-600" />}
              </div>
              <div className="flex-1">
                <p className="font-semibold capitalize text-ink-900">{s.source}</p>
                <p className="text-sm text-ink-500">{kwh(s.kwh)} · {eur(s.cost_eur)}</p>
                <div className="mt-2 h-1.5 rounded-full bg-ink-200">
                  <div className="h-1.5 rounded-full bg-brand-600" style={{ width: `${s.share * 100}%` }} />
                </div>
              </div>
              <span className="text-sm font-semibold text-ink-700">{pct(s.share)}</span>
            </div>
          ))}
        </div>
        <p className="mt-3 text-xs text-ink-400">Tariffs: electricity €{review.tariffs?.electricity}/kWh, gas €{review.tariffs?.gas}/kWh</p>
      </Card>

      <Card title="Pareto analysis — SEU identification" clause="6.3 b)">
        <Plot
          data={[
            { type: 'bar', x: seus.map(s => s.category), y: seus.map(s => s.kwh), name: 'Consumption (kWh)',
              marker: { color: seus.map(s => (s.significant ? '#d97706' : '#cbd5e1')) } },
            { type: 'scatter', mode: 'lines+markers', x: seus.map(s => s.category), y: seus.map(s => s.cumulative_share * 100),
              name: 'Cumulative %', yaxis: 'y2', line: { color: '#1e293b' } },
            { type: 'scatter', mode: 'lines', x: seus.map(s => s.category), y: seus.map(() => threshold),
              name: `${threshold}% threshold`, yaxis: 'y2', line: { color: '#ef4444', dash: 'dash' } },
          ]}
          layout={{
            autosize: true, height: 380, margin: { l: 70, r: 60, t: 20, b: 90 },
            yaxis: { title: 'kWh' }, yaxis2: { title: 'Cumulative %', overlaying: 'y', side: 'right', range: [0, 105] },
            legend: { orientation: 'h', y: 1.12 }, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%' }}
          useResizeHandler
        />
      </Card>

      <Card title="Significant Energy Uses & relevant variables" clause="6.3 c)">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-ink-100 text-left text-xs uppercase text-ink-400">
                <th className="px-3 py-2">SEU category</th><th className="px-3 py-2">Status</th>
                <th className="px-3 py-2 text-right">kWh</th><th className="px-3 py-2 text-right">Share</th>
                <th className="px-3 py-2 text-right">Cum.</th><th className="px-3 py-2 text-right">Cost</th>
                <th className="px-3 py-2 text-right">R² HDD</th><th className="px-3 py-2 text-right">R² CDD</th>
                <th className="px-3 py-2 text-right">R² Prod. days</th><th className="px-3 py-2 text-right">R² Combined</th>
              </tr>
            </thead>
            <tbody>
              {seus.map(s => (
                <tr key={s.category} className={`border-b border-ink-50 ${s.significant ? 'bg-amber-50/40' : ''}`}>
                  <td className="px-3 py-2 font-medium text-ink-900">{s.category}<span className="ml-1 text-xs text-ink-400">({s.meters} meters)</span></td>
                  <td className="px-3 py-2"><SignificanceBadge significant={s.significant} /></td>
                  <td className="px-3 py-2 text-right tabular-nums">{num(s.kwh)}</td>
                  <td className="px-3 py-2 text-right tabular-nums">{pct(s.share)}</td>
                  <td className="px-3 py-2 text-right tabular-nums text-ink-500">{pct(s.cumulative_share)}</td>
                  <td className="px-3 py-2 text-right tabular-nums">{eur(s.cost_eur)}</td>
                  <R2Cell v={s.r2.hdd} relevant={s.relevant_variables.includes('hdd')} />
                  <R2Cell v={s.r2.cdd} relevant={s.relevant_variables.includes('cdd')} />
                  <R2Cell v={s.r2.production_days} relevant={s.relevant_variables.includes('production_days')} />
                  <R2Cell v={s.r2.combined} relevant={false} />
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs text-ink-400">Variables with R² ≥ 0.30 are flagged as relevant (highlighted). Production days = days with consumption &gt; 0.</p>
      </Card>

      <Card title="Energy performance improvement opportunities" clause="6.3 d)">
        <ol className="space-y-3">
          {(review.opportunities || []).map((o: any) => (
            <li key={`${o.category}-${o.type}`} className="flex items-center gap-4 rounded-lg border border-ink-100 p-4">
              <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-brand-50 text-sm font-bold text-brand-700">{o.rank}</span>
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <p className="font-semibold text-ink-900">{o.title}</p>
                  {o.type === 'drift' && <span className="rounded-full bg-red-100 px-2 py-0.5 text-xs font-medium text-red-700">Baseline drift</span>}
                  {o.significant && <SignificanceBadge significant />}
                </div>
                <p className="mt-0.5 flex items-center gap-1.5 text-sm text-ink-500">
                  <Lightbulb className="h-3.5 w-3.5" /> {o.category} · {o.saving_pct}% saving
                </p>
              </div>
              <div className="text-right">
                <p className="font-bold text-emerald-700">{eur(o.eur_savings)}</p>
                <p className="text-xs text-ink-500">{kwh(o.kwh_savings)}</p>
              </div>
            </li>
          ))}
        </ol>
      </Card>

      <Card title={`EnPI comparison — baseline ${review.baseline_year} vs reporting ${review.reporting_year}`} clause="6.4 / 6.5">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-ink-100 text-left text-xs uppercase text-ink-400">
                <th className="px-3 py-2">SEU</th><th className="px-3 py-2 text-right">Baseline kWh</th>
                <th className="px-3 py-2 text-right">Reporting kWh</th><th className="px-3 py-2 text-right">Expected kWh</th>
                <th className="px-3 py-2 text-right">kWh/day (base → rep.)</th><th className="px-3 py-2 text-right">EnPI ratio</th>
                <th className="px-3 py-2 text-right">Improvement</th>
              </tr>
            </thead>
            <tbody>
              {(review.enpi || []).map((e: any) => (
                <tr key={e.category} className="border-b border-ink-50">
                  <td className="px-3 py-2 font-medium text-ink-900">{e.category}</td>
                  <td className="px-3 py-2 text-right tabular-nums">{num(e.baseline_kwh)}</td>
                  <td className="px-3 py-2 text-right tabular-nums">{num(e.reporting_kwh)}</td>
                  <td className="px-3 py-2 text-right tabular-nums text-ink-500">{num(e.expected_kwh)}</td>
                  <td className="px-3 py-2 text-right tabular-nums text-ink-500">{num(e.baseline_kwh_per_day)} → {num(e.reporting_kwh_per_day)}</td>
                  <td className="px-3 py-2 text-right tabular-nums">{e.enpi_ratio ?? '—'}</td>
                  <td className="px-3 py-2 text-right"><Delta v={e.improvement_pct} /></td>
                </tr>
              ))}
              <tr className="font-semibold">
                <td className="px-3 py-2">Site total</td>
                <td className="px-3 py-2 text-right tabular-nums">{num(t.baseline_kwh)}</td>
                <td className="px-3 py-2 text-right tabular-nums">{num(t.reporting_kwh)}</td>
                <td className="px-3 py-2 text-right tabular-nums">{num(t.expected_kwh)}</td>
                <td className="px-3 py-2" />
                <td className="px-3 py-2 text-right tabular-nums">{t.enpi_ratio ?? '—'}</td>
                <td className="px-3 py-2 text-right"><Delta v={t.improvement_pct} /></td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs text-ink-400">Expected kWh = baseline regression model applied to reporting-period variables. EnPI ratio &lt; 1 means improved performance. Baseline is aligned to the reporting period&apos;s date range.</p>
      </Card>
    </div>
  );
}
