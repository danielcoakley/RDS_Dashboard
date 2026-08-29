'use client';

import dynamic from 'next/dynamic';
import { useAnalysis } from '@/lib/analysis';
import { BarChart3, Loader2 } from 'lucide-react';
import SeuDrilldown from './SeuDrilldown';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false }) as any;

export default function SEUAnalysisPage() {
  const { selectedSiteId, availableYears, baselineYear, comparisonYear, bundle, status, error } = useAnalysis();

  const analysis = bundle?.analysis;
  const sankey = bundle?.sankey;
  const seuSummary = bundle?.seuSummary;
  const seuMonthly = bundle?.seuMonthly;

  return (
    <div>
      <h1 className="mb-2 text-2xl font-bold text-ink-900">SEU Analysis</h1>
      <p className="mb-8 text-sm text-ink-500">Significant Energy Use breakdown and energy flow analysis (ISO 50001 §6.3). Select a site &amp; years in the sidebar to run.</p>

      {!selectedSiteId && (
        <div className="rounded-xl border border-dashed border-ink-200 bg-white p-12 text-center">
          <BarChart3 className="mx-auto mb-3 h-10 w-10 text-ink-300" />
          <p className="text-ink-500">Select a site in the sidebar to begin analysis.</p>
        </div>
      )}

      {selectedSiteId && availableYears.length < 2 && (
        <div className="rounded-xl border border-dashed border-ink-200 bg-white p-12 text-center">
          <p className="text-ink-500">Not enough years of data. Upload energy data spanning at least 2 years.</p>
        </div>
      )}

      {status === 'loading' && selectedSiteId && availableYears.length >= 2 && (
        <div className="flex items-center gap-2 rounded-xl border border-ink-100 bg-white p-8 text-ink-500">
          <Loader2 className="h-5 w-5 animate-spin text-brand-600" /> Calculating SEU analysis…
        </div>
      )}

      {error && <p className="mb-4 text-sm text-red-600">{error}</p>}

      {status === 'ready' && (
        <>
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

          {/* SEU Meter Drill-down */}
          {analysis && (
            <SeuDrilldown
              gasMeters={analysis.gas}
              electricityMeters={analysis.electricity}
              comparisonYear={comparisonYear!}
            />
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
        </>
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
