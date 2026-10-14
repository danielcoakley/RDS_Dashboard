'use client';

import { useState } from 'react';
import { Printer, Copy, Loader2, FileText, Check } from 'lucide-react';
import { api } from '@/lib/api';
import { useAnalysis } from '@/lib/analysis';
import TrendChart from './TrendChart';

const fmt = (n: number, d = 0) => (n ?? 0).toLocaleString('en-GB', { maximumFractionDigits: d, minimumFractionDigits: d });
const eur = (n: number) => '€' + fmt(n, 0);

const statusLabel: Record<string, string> = {
  complete: 'Complete', in_progress: 'In progress', not_started: 'Not started',
  active: 'Active', completed: 'Completed', at_risk: 'At risk', open: 'Open', closed: 'Closed',
};

// ISO 50001:2018 clause 9.3 — required management review inputs and outputs.
const INPUTS = [
  { k: 'actions', t: 'Status of actions from previous management reviews' },
  { k: 'issues', t: 'Changes in external and internal issues and associated risks and opportunities' },
  { k: 'performance', t: 'EnMS performance: nonconformities and corrective actions, monitoring and measurement results, audit results, compliance obligations', auto: true },
  { k: 'objectives', t: 'Extent to which objectives and energy targets have been met', auto: true },
  { k: 'enpi', t: 'Energy performance and improvement based on monitoring and measurement, including EnPIs', auto: true },
  { k: 'plans', t: 'Status of action plans' },
  { k: 'opportunities', t: 'Opportunities for continual improvement, including for competence' },
  { k: 'policy', t: 'The energy policy' },
];
const OUTPUTS = [
  'Opportunities to improve energy performance',
  'The energy policy',
  'The EnPIs or energy baseline(s)',
  'Objectives, energy targets, action plans or other EnMS elements, and action to be taken if not achieved',
  'Opportunities to improve integration with business processes',
  'Allocation of resources',
  'Improvement of competence, awareness and communication',
];

function Page({ children, n, meta }: { children: React.ReactNode; n: number; meta: string }) {
  return (
    <section className="a4-page">
      {children}
      <div className="a4-footer a4-sans"><span>{meta}</span><span>Page {n}</span></div>
    </section>
  );
}

export default function ManagementReport() {
  const { sites, selectedSiteId, baselineYear, comparisonYear } = useAnalysis();
  const [elecPrice, setElecPrice] = useState(0.25);
  const [gasPrice, setGasPrice] = useState(0.08);
  const [report, setReport] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [copied, setCopied] = useState(false);

  const ready = selectedSiteId && baselineYear && comparisonYear;

  const generate = async () => {
    if (!ready) return;
    setLoading(true);
    setError('');
    try {
      setReport(await api.managementReview(selectedSiteId!, baselineYear!, comparisonYear!, elecPrice, gasPrice));
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const summary = (r: any) => {
    const t = r.savings.total;
    const ncs = r.open_findings.filter((f: any) => f.severity !== 'observation').length;
    const done = r.compliance.filter((c: any) => c.status === 'complete').length;
    return [
      `${r.organization} — ISO 50001 Management Review, ${r.site.name} (${r.comparison_year} vs baseline ${r.baseline_year})`,
      `Energy consumed: ${fmt(t.actual_kwh)} kWh vs ${fmt(t.predicted_kwh)} kWh weather-normalised expected.`,
      `Savings: ${fmt(t.savings_kwh)} kWh (${fmt(t.savings_pct, 1)}%), ${eur(t.savings_eur)}, ${fmt(t.savings_tco2e, 1)} tCO2e.`,
      `Electricity: ${fmt(r.savings.electricity.savings_kwh)} kWh (${fmt(r.savings.electricity.savings_pct, 1)}%); Gas: ${fmt(r.savings.gas.savings_kwh)} kWh (${fmt(r.savings.gas.savings_pct, 1)}%).`,
      `Objectives: ${r.objectives.length} tracked, ${r.objectives.filter((o: any) => o.status === 'at_risk').length} at risk.`,
      `Compliance: ${done}/${r.compliance.length} clauses complete; ${ncs} open nonconformities, ${r.open_findings.length - ncs} open observations.`,
    ].join('\n');
  };

  const copy = async () => {
    await navigator.clipboard.writeText(summary(report));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const inputCls = 'w-24 rounded-lg border border-ink-200 px-2 py-1.5 text-sm';
  const siteName = sites.find((s) => s.id === selectedSiteId)?.name;

  return (
    <div>
      <div className="no-print mb-6 flex flex-wrap items-end gap-4 rounded-xl border border-ink-100 bg-white p-5">
        <div className="text-sm">
          <p className="text-xs font-medium text-ink-400">Site & period (from sidebar)</p>
          <p className="font-medium text-ink-900">
            {ready ? `${siteName} · ${comparisonYear} vs ${baselineYear}` : 'Select a site and years in the sidebar'}
          </p>
        </div>
        <label className="text-xs text-ink-500">Electricity €/kWh<br />
          <input type="number" step="0.01" min="0" className={inputCls} value={elecPrice} onChange={(e) => setElecPrice(Number(e.target.value))} />
        </label>
        <label className="text-xs text-ink-500">Gas €/kWh<br />
          <input type="number" step="0.01" min="0" className={inputCls} value={gasPrice} onChange={(e) => setGasPrice(Number(e.target.value))} />
        </label>
        <button onClick={generate} disabled={!ready || loading} className="flex items-center gap-2 rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-700 disabled:opacity-50">
          {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <FileText className="h-4 w-4" />} Generate report
        </button>
        {report && (
          <>
            <button onClick={() => window.print()} className="flex items-center gap-2 rounded-lg border border-ink-200 px-4 py-2 text-sm font-medium text-ink-700 hover:bg-ink-50">
              <Printer className="h-4 w-4" /> Download PDF
            </button>
            <button onClick={copy} className="flex items-center gap-2 rounded-lg border border-ink-200 px-4 py-2 text-sm font-medium text-ink-700 hover:bg-ink-50">
              {copied ? <Check className="h-4 w-4 text-brand-600" /> : <Copy className="h-4 w-4" />} {copied ? 'Copied' : 'Copy summary'}
            </button>
          </>
        )}
        {error && <p className="w-full text-sm text-red-600">{error}</p>}
      </div>

      {report && <ReportDocument r={report} summary={summary(report)} />}
    </div>
  );
}

function ReportDocument({ r, summary }: { r: any; summary: string }) {
  const s = r.savings;
  const date = new Date(r.generated_at).toLocaleDateString('en-GB', { day: 'numeric', month: 'long', year: 'numeric' });
  const meta = `${r.organization} · Management Review ${r.comparison_year} · ${r.site.name}`;
  const ncs = r.open_findings;
  const covered = (k: string) => INPUTS.find((i) => i.k === k)?.auto;

  return (
    <div className="bg-ink-100 py-6 print:bg-white print:py-0">
      {/* Cover */}
      <section className="a4-page flex flex-col">
        <div className="a4-sans text-xs font-semibold uppercase tracking-[0.3em] text-brand-700">ISO 50001:2018 · Clauses 9.1 · 9.2 · 9.3</div>
        <div className="mt-auto mb-auto">
          <div className="mb-6 h-1 w-24 bg-brand-600" />
          <h1 style={{ fontSize: '32pt', lineHeight: 1.15, fontWeight: 600 }}>Energy Management<br />Review Report</h1>
          <p className="mt-4 text-lg text-ink-600">{r.site.name}{r.site.address ? ` — ${r.site.address}` : ''}</p>
          <p className="text-ink-600">Reporting period {r.comparison_year} against baseline {r.baseline_year}</p>
        </div>
        <div className="border-t border-ink-200 pt-4">
          <p style={{ fontSize: '16pt', fontWeight: 600 }}>{r.organization}</p>
          <p className="text-ink-600">Prepared for top management · {date}</p>
        </div>
      </section>

      <Page n={2} meta={meta}>
        <h2>1. Executive summary</h2>
        <p className="mb-4">
          In {r.comparison_year}, {r.site.name} consumed {fmt(s.total.actual_kwh)} kWh against a weather-normalised expected
          consumption of {fmt(s.total.predicted_kwh)} kWh derived from the {r.baseline_year} energy baseline. This represents a
          {s.total.savings_kwh >= 0 ? ' saving' : 'n increase'} of <b>{fmt(Math.abs(s.total.savings_kwh))} kWh ({fmt(Math.abs(s.total.savings_pct), 1)}%)</b>,
          worth <b>{eur(Math.abs(s.total.savings_eur))}</b> and <b>{fmt(Math.abs(s.total.savings_tco2e), 1)} tCO₂e</b>.
          There are {ncs.filter((f: any) => f.severity !== 'observation').length} open nonconformities and{' '}
          {r.compliance.filter((c: any) => c.status === 'complete').length} of {r.compliance.length} clauses are complete.
        </p>
        <div className="a4-sans mb-6 grid grid-cols-4 gap-3">
          {[
            ['Energy saved', `${fmt(s.total.savings_kwh)} kWh`],
            ['Cost saved', eur(s.total.savings_eur)],
            ['Carbon avoided', `${fmt(s.total.savings_tco2e, 1)} tCO₂e`],
            ['Improvement', `${fmt(s.total.savings_pct, 1)}%`],
          ].map(([l, v]) => (
            <div key={l} className="rounded-lg border border-ink-200 p-3">
              <p className="text-[10px] uppercase tracking-wide text-ink-500">{l}</p>
              <p className="text-lg font-semibold text-brand-700">{v}</p>
            </div>
          ))}
        </div>
        <h3>Savings achieved</h3>
        <table>
          <thead><tr><th>Utility</th><th className="num">Expected kWh</th><th className="num">Actual kWh</th><th className="num">Saved kWh</th><th className="num">%</th><th className="num">EUR</th><th className="num">tCO₂e</th></tr></thead>
          <tbody>
            {(['electricity', 'gas'] as const).map((u) => (
              <tr key={u}>
                <td className="capitalize">{u}</td>
                <td className="num">{fmt(s[u].predicted_kwh)}</td><td className="num">{fmt(s[u].actual_kwh)}</td>
                <td className="num">{fmt(s[u].savings_kwh)}</td><td className="num">{fmt(s[u].savings_pct, 1)}</td>
                <td className="num">{eur(s[u].savings_eur)}</td><td className="num">{fmt(s[u].savings_tco2e, 2)}</td>
              </tr>
            ))}
            <tr style={{ fontWeight: 600 }}>
              <td>Total</td><td className="num">{fmt(s.total.predicted_kwh)}</td><td className="num">{fmt(s.total.actual_kwh)}</td>
              <td className="num">{fmt(s.total.savings_kwh)}</td><td className="num">{fmt(s.total.savings_pct, 1)}</td>
              <td className="num">{eur(s.total.savings_eur)}</td><td className="num">{fmt(s.total.savings_tco2e, 2)}</td>
            </tr>
          </tbody>
        </table>
        <p className="text-ink-500" style={{ fontSize: '8pt' }}>
          Emission factors: electricity {s.electricity.emission_factor_kg_per_kwh} kgCO₂e/kWh, gas {s.gas.emission_factor_kg_per_kwh} kgCO₂e/kWh.
          Tariffs: electricity €{s.electricity.price_eur_per_kwh}/kWh, gas €{s.gas.price_eur_per_kwh}/kWh. Savings = weather-normalised expected − actual.
        </p>
      </Page>

      <Page n={3} meta={meta}>
        <h2>2. Energy performance & EnPI trend (9.1)</h2>
        <p className="mb-3">Monthly consumption compared with the weather-normalised baseline model (HDD for gas, CDD for electricity) across {r.meter_count} meters.</p>
        <TrendChart title="Electricity" months={r.trend.months} series={r.trend.electricity} />
        <TrendChart title="Gas" months={r.trend.months} series={r.trend.gas} />
      </Page>

      <Page n={4} meta={meta}>
        <h2>3. Significant energy uses</h2>
        <table>
          <thead><tr><th>SEU</th><th>Utility</th><th className="num">Actual kWh</th><th className="num">Expected kWh</th><th className="num">Saved kWh</th><th className="num">Share</th></tr></thead>
          <tbody>
            {r.seu.length === 0 && <tr><td colSpan={6}>No SEU data for this period.</td></tr>}
            {r.seu.map((x: any) => (
              <tr key={x.utility + x.seu}>
                <td>{x.seu}</td><td className="capitalize">{x.utility}</td>
                <td className="num">{fmt(x.actual_kwh)}</td><td className="num">{fmt(x.predicted_kwh)}</td><td className="num">{fmt(x.savings_kwh)}</td>
                <td className="num">{s.total.actual_kwh ? fmt((100 * x.actual_kwh) / s.total.actual_kwh, 1) : 0}%</td>
              </tr>
            ))}
          </tbody>
        </table>
        <h2 style={{ marginTop: 24 }}>4. Objectives & energy targets</h2>
        <table>
          <thead><tr><th>Objective</th><th className="num">Target</th><th className="num">Baseline</th><th className="num">Current</th><th>Deadline</th><th>Status</th></tr></thead>
          <tbody>
            {r.objectives.length === 0 && <tr><td colSpan={6}>No objectives recorded.</td></tr>}
            {r.objectives.map((o: any, i: number) => (
              <tr key={i}>
                <td>{o.title}</td><td className="num">{o.target_pct != null ? `${o.target_pct}%` : '—'}</td>
                <td className="num">{o.baseline_value != null ? fmt(o.baseline_value) : '—'}</td>
                <td className="num">{o.current_value != null ? fmt(o.current_value) : '—'}</td>
                <td>{o.deadline || '—'}</td><td>{statusLabel[o.status] || o.status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Page>

      <Page n={5} meta={meta}>
        <h2>5. Clause compliance & nonconformities (9.2)</h2>
        <table>
          <thead><tr><th>Clause</th><th>Title</th><th>Status</th></tr></thead>
          <tbody>
            {r.compliance.map((c: any) => (
              <tr key={c.clause_ref}><td>§{c.clause_ref}</td><td>{c.clause_title}</td><td>{statusLabel[c.status] || c.status}</td></tr>
            ))}
          </tbody>
        </table>
        <h3>Open audit findings</h3>
        <table>
          <thead><tr><th>Clause</th><th>Severity</th><th>Finding</th><th>Corrective action</th><th>Owner</th><th>Due</th></tr></thead>
          <tbody>
            {ncs.length === 0 && <tr><td colSpan={6}>No open findings.</td></tr>}
            {ncs.map((f: any) => (
              <tr key={f.id}>
                <td>§{f.clause_ref}</td><td className="capitalize">{f.severity}</td><td>{f.finding}</td>
                <td>{f.corrective_action || '—'}</td><td>{f.owner || '—'}</td><td>{f.due_date || '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Page>

      <Page n={6} meta={meta}>
        <h2>6. Management review checklist (9.3)</h2>
        <h3>Required inputs (9.3.3)</h3>
        <ChecklistTable rows={INPUTS.map((i) => ({ t: i.t, done: !!covered(i.k) }))} />
        <h3>Required outputs (9.3.4)</h3>
        <ChecklistTable rows={OUTPUTS.map((t) => ({ t, done: false }))} />
        <h3>Summary</h3>
        <pre className="a4-sans whitespace-pre-wrap rounded border border-ink-200 bg-ink-50 p-3" style={{ fontSize: '9pt' }}>{summary}</pre>
        <div className="a4-sans mt-8 grid grid-cols-2 gap-8 text-xs text-ink-600">
          <div className="border-t border-ink-400 pt-1">Top management signature / date</div>
          <div className="border-t border-ink-400 pt-1">Energy manager signature / date</div>
        </div>
      </Page>
    </div>
  );
}

function ChecklistTable({ rows }: { rows: { t: string; done: boolean }[] }) {
  return (
    <table>
      <tbody>
        {rows.map((row, i) => (
          <tr key={i}>
            <td style={{ width: 28 }}><input type="checkbox" defaultChecked={row.done} /></td>
            <td>{row.t}</td>
            <td className="a4-sans text-ink-500" style={{ width: 150, fontSize: '8pt' }}>{row.done ? 'Covered in this report' : 'Record at review meeting'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
