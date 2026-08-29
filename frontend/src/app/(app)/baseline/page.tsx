'use client';

import { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false }) as any;

export default function BaselinePage() {
  const [sites, setSites] = useState<any[]>([]);
  const [selectedSite, setSelectedSite] = useState<number | null>(null);
  const [years, setYears] = useState<number[]>([]);
  const [baselineYear, setBaselineYear] = useState<number | null>(null);
  const [comparisonYear, setComparisonYear] = useState<number | null>(null);
  const [analysis, setAnalysis] = useState<any>(null);
  const [monthly, setMonthly] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

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
      setAnalysis(null);
      setMonthly(null);
    }
  }, [selectedSite]);

  const runAnalysis = async () => {
    if (!selectedSite || !baselineYear || !comparisonYear) return;
    setLoading(true);
    setError('');
    try {
      const [a, m] = await Promise.all([
        api.runAnalysis({ site_id: selectedSite, baseline_year: baselineYear, comparison_year: comparisonYear }),
        api.monthlyComparison(selectedSite, baselineYear, comparisonYear),
      ]);
      setAnalysis(a);
      setMonthly(m);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h1 className="mb-2 text-2xl font-bold text-ink-900">Baseline & EnPI Analysis</h1>
      <p className="mb-8 text-sm text-ink-500">Climate-normalized energy baseline with regression models per meter (ISO 50001 §6.4–6.5)</p>

      {/* Controls */}
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
              <label className="mb-1.5 block text-sm font-medium text-ink-700">Baseline Year</label>
              <select value={baselineYear || ''} onChange={(e) => setBaselineYear(Number(e.target.value))}
                className="rounded-lg border border-ink-200 px-4 py-2.5 focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-100">
                {years.map(y => <option key={y} value={y}>{y}</option>)}
              </select>
            </div>
            <div>
              <label className="mb-1.5 block text-sm font-medium text-ink-700">Comparison Year</label>
              <select value={comparisonYear || ''} onChange={(e) => setComparisonYear(Number(e.target.value))}
                className="rounded-lg border border-ink-200 px-4 py-2.5 focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-100">
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

      {error && <p className="mb-4 text-sm text-red-600">{error}</p>}

      {analysis && (
        <>
          {/* Summary cards */}
          <div className="mb-6 grid gap-4 sm:grid-cols-2">
            <SummaryCard title="Gas Consumption" predicted={analysis.totals.gas_predicted} actual={analysis.totals.gas_actual} savingsPct={analysis.totals.gas_savings_pct} color="orange" />
            <SummaryCard title="Electricity Consumption" predicted={analysis.totals.elec_predicted} actual={analysis.totals.elec_actual} savingsPct={analysis.totals.elec_savings_pct} color="blue" />
          </div>

          {/* Tables */}
          {analysis.gas.length > 0 && (
            <MeterTable title="Gas Meters" data={analysis.gas} />
          )}
          {analysis.electricity.length > 0 && (
            <MeterTable title="Electricity Meters" data={analysis.electricity} />
          )}

          {/* Monthly charts */}
          {monthly && (monthly.gas?.length > 0 || monthly.electricity?.length > 0) && (
            <div className="mt-6">
              <h2 className="mb-4 text-lg font-semibold text-ink-900">Monthly Actual vs Predicted</h2>
              {[...(monthly.gas || []), ...(monthly.electricity || [])].map((chart: any, i: number) => (
                <div key={i} className="mb-6 rounded-xl border border-ink-100 bg-white p-5">
                  <h3 className="mb-3 font-medium text-ink-900">{chart.meter}</h3>
                  <Plot
                    data={[
                      { x: chart.months, y: chart.baseline, type: 'bar', name: `${baselineYear} Actual`, marker: { color: 'rgba(200,200,200,0.6)' } },
                      { x: chart.months, y: chart.actual, type: 'bar', name: `${comparisonYear} Actual`, marker: { color: 'rgba(51,153,255,0.7)' } },
                      { x: chart.months, y: chart.predicted, type: 'scatter', mode: 'lines+markers', name: 'Predicted', line: { dash: 'dash', color: '#ff6b6b' } },
                    ]}
                    layout={{ height: 300, margin: { l: 50, r: 20, t: 10, b: 40 }, xaxis: { title: 'Month' }, yaxis: { title: 'kWh' }, legend: { x: 0, y: 1.15, orientation: 'h' } }}
                    config={{ responsive: true, displayModeBar: false }}
                    style={{ width: '100%' }}
                  />
                </div>
              ))}
            </div>
          )}
        </>
      )}

      {selectedSite && years.length < 2 && (
        <div className="rounded-xl border border-dashed border-ink-200 bg-white p-12 text-center">
          <p className="text-ink-500">Not enough years of data to run analysis. Upload energy data spanning at least 2 years.</p>
        </div>
      )}
    </div>
  );
}

function SummaryCard({ title, predicted, actual, savingsPct, color }: any) {
  const diff = predicted - actual;
  const isSaving = diff > 0;
  return (
    <div className="rounded-xl border border-ink-100 bg-white p-5">
      <h3 className="font-semibold text-ink-900">{title}</h3>
      <div className="mt-3 flex items-baseline gap-2">
        <span className="text-3xl font-bold text-ink-900">{actual?.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
        <span className="text-sm text-ink-500">kWh actual</span>
      </div>
      <p className="mt-1 text-sm text-ink-400">vs {predicted?.toLocaleString(undefined, { maximumFractionDigits: 0 })} predicted</p>
      <span className={`mt-2 inline-block rounded-full px-2 py-0.5 text-sm font-medium ${isSaving ? 'bg-brand-50 text-brand-700' : 'bg-red-50 text-red-700'}`}>
        {isSaving ? '↓' : '↑'} {Math.abs(savingsPct)}% {isSaving ? 'savings' : 'overuse'}
      </span>
    </div>
  );
}

function MeterTable({ title, data }: any) {
  return (
    <div className="mb-6 overflow-hidden rounded-xl border border-ink-100 bg-white">
      <div className="border-b border-ink-100 bg-ink-50 px-4 py-3">
        <h2 className="font-semibold text-ink-900">{title}</h2>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="border-b border-ink-100">
            <tr className="text-left text-ink-500">
              <th className="px-4 py-2 font-medium">Meter</th>
              <th className="px-4 py-2 font-medium">SEU</th>
              <th className="px-4 py-2 font-medium text-right">Baseline</th>
              <th className="px-4 py-2 font-medium text-right">Predicted</th>
              <th className="px-4 py-2 font-medium text-right">Actual</th>
              <th className="px-4 py-2 font-medium text-right">Savings</th>
              <th className="px-4 py-2 font-medium text-right">% Savings</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-ink-50">
            {data.map((row: any, i: number) => (
              <tr key={i} className="hover:bg-ink-50">
                <td className="px-4 py-2 font-medium text-ink-900">{row.meter}</td>
                <td className="px-4 py-2 text-ink-600">{row.seu_category}</td>
                <td className="px-4 py-2 text-right text-ink-600">{row.baseline?.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                <td className="px-4 py-2 text-right text-ink-600">{row.predicted?.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                <td className="px-4 py-2 text-right text-ink-600">{row.actual?.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                <td className="px-4 py-2 text-right text-ink-600">{row.estimated_savings?.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
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
    </div>
  );
}
