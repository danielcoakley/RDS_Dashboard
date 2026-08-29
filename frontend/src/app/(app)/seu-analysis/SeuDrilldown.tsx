'use client';

import { useState, useMemo } from 'react';
import dynamic from 'next/dynamic';
import { Search } from 'lucide-react';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false }) as any;

interface MeterRow {
  meter: string;
  seu_category: string;
  baseline: number;
  predicted: number;
  actual: number;
  estimated_savings: number;
  pct_savings: number | null;
  baseline_days: number;
  actual_days: number;
}

interface SeuDrilldownProps {
  gasMeters: MeterRow[];
  electricityMeters: MeterRow[];
  comparisonYear: number;
}

export default function SeuDrilldown({ gasMeters, electricityMeters, comparisonYear }: SeuDrilldownProps) {
  const [selectedSeu, setSelectedSeu] = useState('');

  // Build the list of SEU categories that have non-zero actuals
  const seuOptions = useMemo(() => {
    const gasSeus = gasMeters.filter(m => m.actual > 0).map(m => m.seu_category);
    const elecSeus = electricityMeters.filter(m => m.actual > 0).map(m => m.seu_category);
    return Array.from(new Set([...gasSeus, ...elecSeus])).sort();
  }, [gasMeters, electricityMeters]);

  // Filter meters for the selected SEU
  const drilldownData = useMemo(() => {
    if (!selectedSeu) return null;

    const gasFiltered = gasMeters.filter(m => m.seu_category === selectedSeu && m.actual > 0);
    const elecFiltered = electricityMeters.filter(m => m.seu_category === selectedSeu && m.actual > 0);

    const labels = ['Total Energy'];
    const sources: number[] = [];
    const targets: number[] = [];
    const values: number[] = [];

    let idx = 1;

    // Gas meters
    for (const m of gasFiltered) {
      labels.push(m.meter);
      sources.push(0);
      targets.push(idx);
      values.push(m.actual);
      idx++;
    }
    // Electricity meters
    for (const m of elecFiltered) {
      labels.push(m.meter);
      sources.push(0);
      targets.push(idx);
      values.push(m.actual);
      idx++;
    }

    const gasStart = 1;
    const elecStart = 1 + gasFiltered.length;
    const linkColors = targets.map(t =>
      t >= elecStart ? 'rgba(51,153,255,0.4)' : 'rgba(255,153,51,0.4)'
    );
    const nodeColors = ['#888888']
      .concat(gasFiltered.map(() => '#ffe6cc'))
      .concat(elecFiltered.map(() => '#cce6ff'));

    const totalGas = gasFiltered.reduce((s, m) => s + m.actual, 0);
    const totalElec = elecFiltered.reduce((s, m) => s + m.actual, 0);

    return { labels, sources, targets, values, linkColors, nodeColors, totalGas, totalElec, gasFiltered, elecFiltered };
  }, [selectedSeu, gasMeters, electricityMeters]);

  if (seuOptions.length === 0) return null;

  return (
    <div className="mt-6 rounded-xl border border-ink-100 bg-white p-5">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-lg font-semibold text-ink-900">SEU Meter Drill-down</h2>
        <div className="relative">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-ink-400" />
          <select
            value={selectedSeu}
            onChange={(e) => setSelectedSeu(e.target.value)}
            className="appearance-none rounded-lg border border-ink-200 py-2 pl-9 pr-8 text-sm focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-100"
          >
            <option value="">Select SEU…</option>
            {seuOptions.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
      </div>

      {!selectedSeu && (
        <p className="py-8 text-center text-sm text-ink-400">
          Select a SEU category to see individual meter consumption for {comparisonYear}.
        </p>
      )}

      {selectedSeu && drilldownData && (
        <>
          {drilldownData.labels.length <= 1 ? (
            <p className="py-8 text-center text-sm text-ink-400">
              No meter-level consumption found for this SEU in {comparisonYear}.
            </p>
          ) : (
            <>
              <Plot
                data={[{
                  type: 'sankey',
                  node: {
                    pad: 20,
                    thickness: 20,
                    label: drilldownData.labels,
                    color: drilldownData.nodeColors,
                  },
                  link: {
                    source: drilldownData.sources,
                    target: drilldownData.targets,
                    value: drilldownData.values,
                    color: drilldownData.linkColors,
                  },
                }]}
                layout={{
                  height: 400,
                  margin: { l: 10, r: 10, t: 10, b: 10 },
                  font: { size: 13 },
                }}
                config={{ responsive: true, displayModeBar: false }}
                style={{ width: '100%' }}
              />

              {/* Meter-level table */}
              <div className="mt-4 overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-ink-100 text-left text-ink-500">
                      <th className="px-4 py-2 font-medium">Meter</th>
                      <th className="px-4 py-2 font-medium">Utility</th>
                      <th className="px-4 py-2 font-medium text-right">Baseline</th>
                      <th className="px-4 py-2 font-medium text-right">Predicted</th>
                      <th className="px-4 py-2 font-medium text-right">Actual ({comparisonYear})</th>
                      <th className="px-4 py-2 font-medium text-right">% Savings</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-ink-50">
                    {drilldownData.gasFiltered.map((m, i) => (
                      <tr key={`g${i}`} className="hover:bg-ink-50">
                        <td className="px-4 py-2 font-medium text-ink-900">{m.meter}</td>
                        <td className="px-4 py-2">
                          <span className="rounded-full bg-orange-50 px-2 py-0.5 text-xs font-medium text-orange-700">Gas</span>
                        </td>
                        <td className="px-4 py-2 text-right text-ink-600">{m.baseline?.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                        <td className="px-4 py-2 text-right text-ink-600">{m.predicted?.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                        <td className="px-4 py-2 text-right text-ink-600">{m.actual?.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                        <td className="px-4 py-2 text-right">
                          <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${m.pct_savings > 0 ? 'bg-brand-50 text-brand-700' : m.pct_savings < 0 ? 'bg-red-50 text-red-700' : 'bg-ink-50 text-ink-500'}`}>
                            {m.pct_savings > 0 ? '+' : ''}{m.pct_savings}%
                          </span>
                        </td>
                      </tr>
                    ))}
                    {drilldownData.elecFiltered.map((m, i) => (
                      <tr key={`e${i}`} className="hover:bg-ink-50">
                        <td className="px-4 py-2 font-medium text-ink-900">{m.meter}</td>
                        <td className="px-4 py-2">
                          <span className="rounded-full bg-blue-50 px-2 py-0.5 text-xs font-medium text-blue-700">Electricity</span>
                        </td>
                        <td className="px-4 py-2 text-right text-ink-600">{m.baseline?.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                        <td className="px-4 py-2 text-right text-ink-600">{m.predicted?.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                        <td className="px-4 py-2 text-right text-ink-600">{m.actual?.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                        <td className="px-4 py-2 text-right">
                          <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${m.pct_savings > 0 ? 'bg-brand-50 text-brand-700' : m.pct_savings < 0 ? 'bg-red-50 text-red-700' : 'bg-ink-50 text-ink-500'}`}>
                            {m.pct_savings > 0 ? '+' : ''}{m.pct_savings}%
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
}
