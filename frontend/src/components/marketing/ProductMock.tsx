// Static, dependency-free product preview for marketing pages (Tailwind + inline SVG).

const expected = [62, 58, 51, 44, 36, 30, 27, 28, 34, 43, 52, 60];
const actual = [60, 55, 47, 40, 32, 26, 23, 24, 29, 37, 45, 52];
const months = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'];

const W = 420;
const H = 170;
const PAD = 8;
const x = (i: number) => PAD + (i * (W - PAD * 2)) / (expected.length - 1);
const y = (v: number) => H - PAD - ((v - 15) / (70 - 15)) * (H - PAD * 2);
const path = (vals: number[]) => vals.map((v, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
const band =
  expected.map((v, i) => `${i === 0 ? 'M' : 'L'}${x(i).toFixed(1)},${y(v + 3).toFixed(1)}`).join(' ') +
  ' ' +
  [...expected].reverse().map((v, j) => `L${x(expected.length - 1 - j).toFixed(1)},${y(v - 3).toFixed(1)}`).join(' ') +
  ' Z';

const clauses = [
  { c: '4', label: 'Context', v: 100 },
  { c: '5', label: 'Leadership', v: 92 },
  { c: '6', label: 'Planning', v: 84 },
  { c: '7', label: 'Support', v: 76 },
  { c: '8', label: 'Operation', v: 81 },
  { c: '9', label: 'Evaluation', v: 68 },
  { c: '10', label: 'Improvement', v: 74 },
];

function ScoreRing({ score }: { score: number }) {
  const r = 34;
  const circ = 2 * Math.PI * r;
  return (
    <div className="relative h-24 w-24">
      <svg viewBox="0 0 80 80" className="h-24 w-24 -rotate-90">
        <circle cx="40" cy="40" r={r} fill="none" stroke="#ebeef3" strokeWidth="7" />
        <circle
          cx="40" cy="40" r={r} fill="none" stroke="url(#ring)" strokeWidth="7" strokeLinecap="round"
          strokeDasharray={`${(score / 100) * circ} ${circ}`}
        />
        <defs>
          <linearGradient id="ring" x1="0" x2="1">
            <stop offset="0%" stopColor="#40d18f" />
            <stop offset="100%" stopColor="#0f9359" />
          </linearGradient>
        </defs>
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-2xl font-bold text-ink-950">{score}%</span>
        <span className="text-[10px] font-medium uppercase tracking-wide text-ink-400">ready</span>
      </div>
    </div>
  );
}

export function ProductMock() {
  return (
    <div className="relative">
      <div className="absolute -inset-6 -z-10 rounded-[2rem] bg-gradient-to-tr from-brand-200/50 via-white to-sky-100/60 blur-2xl" />
      <div className="overflow-hidden rounded-2xl border border-ink-200/80 bg-white shadow-2xl shadow-ink-900/10">
        {/* window chrome */}
        <div className="flex items-center gap-2 border-b border-ink-100 bg-ink-50/70 px-4 py-2.5">
          <span className="h-2.5 w-2.5 rounded-full bg-ink-200" />
          <span className="h-2.5 w-2.5 rounded-full bg-ink-200" />
          <span className="h-2.5 w-2.5 rounded-full bg-ink-200" />
          <span className="ml-3 rounded-md bg-white px-3 py-0.5 text-[11px] text-ink-400 ring-1 ring-ink-100">app.enms.io / dashboard</span>
        </div>

        <div className="grid gap-4 p-4 sm:grid-cols-3 sm:p-5">
          {/* EnPI chart */}
          <div className="rounded-xl border border-ink-100 p-4 sm:col-span-2">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-xs font-medium text-ink-500">EnPI — site electricity (MWh)</p>
                <p className="mt-0.5 text-lg font-semibold text-ink-950">
                  −11.8% <span className="text-xs font-medium text-brand-600">vs weather-normalised baseline</span>
                </p>
              </div>
              <span className="rounded-full bg-brand-50 px-2 py-0.5 text-[10px] font-semibold text-brand-700 ring-1 ring-brand-100">R² 0.94</span>
            </div>
            <svg viewBox={`0 0 ${W} ${H + 16}`} className="mt-3 w-full" role="img" aria-label="Actual consumption tracking below the weather-normalised baseline">
              {[0, 1, 2, 3].map((g) => (
                <line key={g} x1={PAD} x2={W - PAD} y1={PAD + (g * (H - PAD * 2)) / 3} y2={PAD + (g * (H - PAD * 2)) / 3} stroke="#ebeef3" strokeDasharray="3 4" />
              ))}
              <path d={band} fill="#aeb9ca" opacity="0.18" />
              <path d={path(expected)} fill="none" stroke="#8295b0" strokeWidth="2" strokeDasharray="5 4" />
              <path d={`${path(actual)} L${x(11)},${H - PAD} L${x(0)},${H - PAD} Z`} fill="url(#area)" />
              <path d={path(actual)} fill="none" stroke="#0f9359" strokeWidth="2.5" strokeLinejoin="round" />
              {actual.map((v, i) => (i === 7 ? <circle key={i} cx={x(i)} cy={y(v)} r="4" fill="#fff" stroke="#0f9359" strokeWidth="2" /> : null))}
              {months.map((m, i) => (
                <text key={i} x={x(i)} y={H + 12} textAnchor="middle" fontSize="9" fill="#8295b0">{m}</text>
              ))}
              <defs>
                <linearGradient id="area" x1="0" x2="0" y1="0" y2="1">
                  <stop offset="0%" stopColor="#1ab672" stopOpacity="0.22" />
                  <stop offset="100%" stopColor="#1ab672" stopOpacity="0" />
                </linearGradient>
              </defs>
            </svg>
            <div className="mt-2 flex flex-wrap gap-4 text-[11px] text-ink-500">
              <span className="flex items-center gap-1.5"><span className="h-0.5 w-4 rounded bg-brand-600" /> Actual</span>
              <span className="flex items-center gap-1.5"><span className="h-0.5 w-4 rounded border-t-2 border-dashed border-ink-400" /> Expected (HDD/CDD model)</span>
            </div>
          </div>

          {/* Readiness */}
          <div className="rounded-xl border border-ink-100 p-4">
            <p className="text-xs font-medium text-ink-500">ISO 50001 readiness</p>
            <div className="mt-3 flex justify-center"><ScoreRing score={82} /></div>
            <ul className="mt-4 space-y-1.5">
              {clauses.map((c) => (
                <li key={c.c} className="flex items-center gap-2 text-[10px]">
                  <span className="w-5 font-semibold text-ink-400">§{c.c}</span>
                  <span className="h-1.5 flex-1 overflow-hidden rounded-full bg-ink-100">
                    <span className={`block h-full rounded-full ${c.v >= 80 ? 'bg-brand-500' : 'bg-amber-400'}`} style={{ width: `${c.v}%` }} />
                  </span>
                  <span className="w-7 text-right text-ink-500">{c.v}%</span>
                </li>
              ))}
            </ul>
          </div>

          {/* KPI strip */}
          {[
            { k: 'Significant energy uses', v: '4 SEUs', s: '78% of consumption' },
            { k: 'Avoided energy (YTD)', v: '412 MWh', s: '≈ €61k at current tariff' },
            { k: 'Open actions', v: '7', s: '3 due this month' },
          ].map((m) => (
            <div key={m.k} className="rounded-xl border border-ink-100 bg-ink-50/50 px-4 py-3">
              <p className="text-[11px] text-ink-500">{m.k}</p>
              <p className="mt-0.5 text-base font-semibold text-ink-950">{m.v}</p>
              <p className="text-[11px] text-ink-400">{m.s}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
