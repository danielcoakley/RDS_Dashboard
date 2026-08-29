'use client';

import { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false }) as any;

export default function SEUAnalysisPage() {
  const [sites, setSites] = useState<any[]>([]);
  const [selectedSite, setSelectedSite] = useState<number | null>(null);
  const [years, setYears] = useState<number[]>([]);
  const [baselineYear, setBaselineYear] = useState<number | null>(null);
  const [comparisonYear, setComparisonYear] = useState<number | null>(null);
  const [sankey, setSankey] = useState<any>(null);
  const [seuSummary, setSeuSummary] = useState<any>(null);
  const [seuMonthly, setSeuMonthly] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => { api.listSites().then(setSites); }, []);

  useEffect(() => {
    if (selectedSite) {
      api.availableYears(selectedSite).then((res: any) => {
        setYears(res.years);
        if (res.years.length >= 2) {
          setBaselineYear(res.years[0]);
          setComparisonYear(res.years[res.years.length - 1]);
        }
      });
      setSankey(null);
      setSeuSummary(null);
      setSeuMonthly(null);
    }
  }, [selectedSite]);

  const runAnalysis = async () => {
    if (!selectedSite || !baselineYear || !comparisonYear) return;
    setLoading(true);
    try {
      const [sk, ss, sm] = await Promise.all([
        api.sankey(selectedSite, baselineYear, comparisonYear),
        api.seuSummary(selectedSite, baselineYear, comparisonYear),
        api.seuMonthly(selectedSite, baselineYear, comparisonYear),
      ]);
      setSankey(sk);
      setSeuSummary(ss);
      setSeuMonthly(sm);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h1 className="mb-2 text-2xl font-bold text-ink-900">SEU Analysis</h1>
      <p className="mb-8 text-sm text-ink-500">Significant Energy Use breakdown and energy flow analysis (ISO 50001 §6.3)</p>

      <div className="mb-6 flex flex-wrap items-end gap-4 rounded-xl border border-ink-100 bg-white p-5">
        <div>
          <label className="mb-1.5 block text-sm font-medium text-ink-700">Site</label>
          <select value={selectedSite || ''} onChange={(e) => setSelectedSite(Number(e.target.value))}
            className="rounded-lg border border-ink-200 px-4 py-2.5 focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-100">
            <option value="">Select site...</option>
            {sites.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}
          </select>
        </div>
        {years.length >= 2 && (
          <>
            <div>
              <label className="mb-1.5 block text-sm font-medium text-ink-700">Baseline</label>
              <select value={baselineYear || ''} onChange={(e) => setBaselineYear(Number(e.target.value))}
                className="rounded-lg border border-ink-200 px-4 py-2.5">
                {years.map(y => <option key={y} value={y}>{y}</option>)}
              </select>
            </div>
            <div>
              <label className="mb-1.5 block text-sm font-medium text-ink-700">Comparison</label>
              <select value={comparisonYear || ''} onChange={(e) => setComparisonYear(Number(e.target.value))}
                className="rounded-lg border border-ink-200 px-4 py-2.5">
                {years.map(y => <option key={y} value={y}>{y}</option>)}
              </select>
            </div>
            <button onClick={runAnalysis} disabled={loading}
              className="rounded-lg bg-brand-600 px-6 py-2.5 font-semibold text-white hover:bg-brand-700 disabled:opacity-50">
              {loading ? 'Analyzing...' : 'Run Analysis'}
            </button>
          </>
        )}
      </div>

      {sankey && sankey.labels.length > 3 && (
        <div className="mb-6 rounded-xl border border-ink-100 bg-white p-5">
          <h2 className="mb-4 text-lg font-semibold text-ink-900">Energy Flow by SEU</h2>
          <Plot
            data={[{
              type: 'sankey',
              node: {
                pad: 20, thickness: 24,
                label: sankey.labels,
                color: ['#888', '#ff9933', '#3399ff'] + Array(sankey.labels.length - 3).fill('#ccc'),
              },
              link: {
                source: sankey.sources, target: sankey.targets, value: sankey.values,
                color: sankey.targets.map((t: number) => t === 1 || (t >= 3 && sankey.targets.indexOf(t) < sankey.labels.indexOf('Electricity')) ? 'rgba(255,153,51,0.4)' : 'rgba(51,153,255,0.4)'),
              },
            }]}
            layout={{ height: 500, margin: { l: 10, r: 10, t: 10, b: 10 }, font: { size: 13 } }}
            config={{ responsive: true, displayModeBar: false }}
            style={{ width: '100%' }}
          />
        </div>
      )}

      {seuSummary && (
        <div className="grid gap-6 sm:grid-cols-2">
          {seuSummary.gas?.length > 0 && <SEUTable title="Gas SEU Summary" data={seuSummary.gas} />}
          {seuSummary.electricity?.length > 0 && <SEUTable title="Electricity SEU Summary" data={seuSummary.electricity} />}
        </div>
      )}

      {/* Monthly SEU charts */}
      {seuMonthly && (seuMonthly.gas?.length > 0 || seuMonthly.electricity?.length > 0) && (
        <div className="mt-6">
          {seuMonthly.gas?.length > 0 && (
            <div className="mb-8">
              <h2 className="mb-4 text-lg font-semibold text-ink-900">Monthly Consumption by SEU — Gas</h2>
              <div className="grid gap-4 lg:grid-cols-2">
                {seuMonthly.gas.map((chart: any, i: number) => (
                  <div key={`g${i}`} className="rounded-xl border border-ink-100 bg-white p-5">
                    <h3 className="mb-3 font-medium text-ink-900">{chart.seu_category}</h3>
                    <Plot
                      data={[
                        { x: chart.months, y: chart.baseline, type: 'bar', name: `${baselineYear} Actual`, marker: { color: 'rgba(200,200,200,0.6)' } },
                        { x: chart.months, y: chart.actual, type: 'bar', name: `${comparisonYear} Actual`, marker: { color: 'rgba(255,153,51,0.7)' } },
                        { x: chart.months, y: chart.predicted, type: 'scatter', mode: 'lines+markers', name: 'Predicted', line: { dash: 'dash', color: '#ff6b6b' } },
                      ]}
                      layout={{ height: 280, margin: { l: 50, r: 20, t: 10, b: 40 }, xaxis: { title: 'Month' }, yaxis: { title: 'kWh' }, legend: { x: 0, y: 1.15, orientation: 'h' } }}
                      config={{ responsive: true, displayModeBar: false }}
                      style={{ width: '100%' }}
                    />
                  </div>
                ))}
              </div>
            </div>
          )}
          {seuMonthly.electricity?.length > 0 && (
            <div>
              <h2 className="mb-4 text-lg font-semibold text-ink-900">Monthly Consumption by SEU — Electricity</h2>
              <div className="grid gap-4 lg:grid-cols-2">
                {seuMonthly.electricity.map((chart: any, i: number) => (
                  <div key={`e${i}`} className="rounded-xl border border-ink-100 bg-white p-5">
                    <h3 className="mb-3 font-medium text-ink-900">{chart.seu_category}</h3>
                    <Plot
                      data={[
                        { x: chart.months, y: chart.baseline, type: 'bar', name: `${baselineYear} Actual`, marker: { color: 'rgba(200,200,200,0.6)' } },
                        { x: chart.months, y: chart.actual, type: 'bar', name: `${comparisonYear} Actual`, marker: { color: 'rgba(51,153,255,0.7)' } },
                        { x: chart.months, y: chart.predicted, type: 'scatter', mode: 'lines+markers', name: 'Predicted', line: { dash: 'dash', color: '#ff6b6b' } },
                      ]}
                      layout={{ height: 280, margin: { l: 50, r: 20, t: 10, b: 40 }, xaxis: { title: 'Month' }, yaxis: { title: 'kWh' }, legend: { x: 0, y: 1.15, orientation: 'h' } }}
                      config={{ responsive: true, displayModeBar: false }}
                      style={{ width: '100%' }}
                    />
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {selectedSite && years.length < 2 && (
        <div className="rounded-xl border border-dashed border-ink-200 bg-white p-12 text-center">
          <p className="text-ink-500">Not enough years of data. Upload energy data spanning at least 2 years.</p>
        </div>
      )}
    </div>
  );
}

function SEUTable({ title, data }: any) {
  return (
    <div className="overflow-hidden rounded-xl border border-ink-100 bg-white">
      <div className="border-b border-ink-100 bg-ink-50 px-4 py-3">
        <h2 className="font-semibold text-ink-900">{title}</h2>
      </div>
      <table className="w-full text-sm">
        <thead className="border-b border-ink-100">
          <tr className="text-left text-ink-500">
            <th className="px-4 py-2 font-medium">SEU Category</th>
            <th className="px-4 py-2 font-medium text-right">Baseline</th>
            <th className="px-4 py-2 font-medium text-right">Predicted</th>
            <th className="px-4 py-2 font-medium text-right">Actual</th>
            <th className="px-4 py-2 font-medium text-right">% Savings</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-ink-50">
          {data.map((row: any, i: number) => (
            <tr key={i} className="hover:bg-ink-50">
              <td className="px-4 py-2 font-medium text-ink-900">{row.seu_category}</td>
              <td className="px-4 py-2 text-right text-ink-600">{row.baseline?.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
              <td className="px-4 py-2 text-right text-ink-600">{row.predicted?.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
              <td className="px-4 py-2 text-right text-ink-600">{row.actual?.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
              <td className="px-4 py-2 text-right">
                <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${row.pct_savings > 0 ? 'bg-brand-50 text-brand-700' : row.pct_savings < 0 ? 'bg-red-50 text-red-700' : 'bg-ink-50 text-ink-500'}`}>
                  {row.pct_savings > 0 ? '+' : ''}{row.pct_savings}%
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
