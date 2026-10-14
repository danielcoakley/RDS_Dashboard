'use client';

interface Series { baseline: number[]; actual: number[]; predicted: number[] }

/** Lightweight print-friendly SVG line chart: actual vs weather-normalised baseline model. */
export default function TrendChart({ months, series, title }: { months: string[]; series: Series; title: string }) {
  const W = 640, H = 200, P = { l: 56, r: 12, t: 12, b: 28 };
  const max = Math.max(1, ...series.actual, ...series.predicted, ...series.baseline);
  const x = (i: number) => P.l + (i * (W - P.l - P.r)) / 11;
  const y = (v: number) => H - P.b - (v / max) * (H - P.t - P.b);
  const path = (vals: number[]) => vals.map((v, i) => `${i ? 'L' : 'M'}${x(i)},${y(v)}`).join(' ');
  const lines = [
    { key: 'baseline' as const, color: '#94a3b8', dash: '4 3', label: 'Baseline year (actual)' },
    { key: 'predicted' as const, color: '#0ea5e9', dash: '', label: 'Weather-normalised expected' },
    { key: 'actual' as const, color: '#059669', dash: '', label: 'Reporting year (actual)' },
  ];
  return (
    <figure className="mb-4">
      <figcaption className="a4-sans mb-1 text-xs font-semibold text-ink-700">{title}</figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full">
        {[0, 0.25, 0.5, 0.75, 1].map((f) => (
          <g key={f}>
            <line x1={P.l} x2={W - P.r} y1={y(max * f)} y2={y(max * f)} stroke="#e2e8f0" />
            <text x={P.l - 6} y={y(max * f) + 3} fontSize="9" textAnchor="end" fill="#64748b">{Math.round(max * f).toLocaleString()}</text>
          </g>
        ))}
        {months.map((m, i) => <text key={m} x={x(i)} y={H - 10} fontSize="9" textAnchor="middle" fill="#64748b">{m}</text>)}
        {lines.map((l) => <path key={l.key} d={path(series[l.key])} fill="none" stroke={l.color} strokeWidth="2" strokeDasharray={l.dash} />)}
      </svg>
      <div className="a4-sans flex gap-4 text-[10px] text-ink-600">
        {lines.map((l) => (
          <span key={l.key} className="flex items-center gap-1"><span className="inline-block h-0.5 w-4" style={{ background: l.color }} />{l.label}</span>
        ))}
        <span className="ml-auto">kWh / month</span>
      </div>
    </figure>
  );
}
