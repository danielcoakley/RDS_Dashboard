'use client';

import dynamic from 'next/dynamic';
import { useAnalysis } from '@/lib/analysis';
import { LineChart, Loader2 } from 'lucide-react';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false }) as any;

export default function BaselinePage() {
  const { selectedSiteId, availableYears, baselineYear, comparisonYear, bundle, status, error } = useAnalysis();

  const analysis = bundle?.analysis;
  const monthly = bundle?.monthly;

  return (
    <div>
      <h1 className="mb-2 text-2xl font-bold text-ink-900">Baseline & EnPI Analysis</h1>
      <p className="mb-8 text-sm text-ink-500">Climate-normalized energy baseline with regression models per meter (ISO 50001 §6.4–6.5). Select a site &amp; years in the sidebar to run.</p>

      {!selectedSiteId && (
        <div className="rounded-xl border border-dashed border-ink-200 bg-white p-12 text-center">
          <LineChart className="mx-auto mb-3 h-10 w-10 text-ink-300" />
          <p className="text-ink-500">Select a site in the sidebar to begin analysis.</p>
        </div>
      )}

      {selectedSiteId && availableYears.length < 2 && (
        <div className="rounded-xl border border-dashed border-ink-200 bg-white p-12 text-center">
          <p className="text-ink-500">Not enough years of data to run analysis. Upload energy data spanning at least 2 years.</p>
        </div>
      )}

      {status === 'loading' && selectedSiteId && availableYears.length >= 2 && (
        <div className="flex items-center gap-2 rounded-xl border border-ink-100 bg-white p-8 text-ink-500">
          <Loader2 className="h-5 w-5 animate-spin text-brand-600" /> Calculating baseline &amp; EnPI…
        </div>
      )}

      {error && <p className="mb-4 text-sm text-red-600">{error}</p>}

      {status === 'ready' && analysis && (
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
