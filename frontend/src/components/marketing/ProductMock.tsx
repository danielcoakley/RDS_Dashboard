'use client';

import { useState } from 'react';

// Synthetic monthly data: raw consumption (jagged) vs HDD/CDD regression (smooth)
const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const raw = [412, 468, 371, 318, 262, 241, 288, 251, 279, 336, 401, 455];
const model = [430, 440, 380, 320, 270, 245, 262, 258, 285, 340, 392, 438];

const W = 480, H = 200, P = 16;
const x = (i: number) => P + (i * (W - 2 * P)) / 11;
const y = (v: number) => H - P - ((v - 200) / 300) * (H - 2 * P);
const path = (d: number[]) => d.map((v, i) => `${i ? 'L' : 'M'}${x(i)},${y(v)}`).join(' ');

export default function ProductMock() {
  const [hover, setHover] = useState<number | null>(null);
  const score = 94, r = 34, c = 2 * Math.PI * r;

  return (
    <div className="relative">
      <div className="rounded-lg border border-white/10 bg-ink-900/70 p-5 shadow-2xl shadow-black/40 backdrop-blur">
        <div className="flex items-center justify-between">
          <div>
            <p className="font-mono text-[11px] uppercase tracking-widest text-ink-400">EnPI · Main site electricity</p>
            <p className="mt-1 font-mono text-2xl text-white">−7.8% <span className="text-sm text-brand-400">vs baseline</span></p>
          </div>
          <div className="flex gap-4 font-mono text-[11px] text-ink-300">
            <span className="flex items-center gap-1.5"><span className="h-0.5 w-4 bg-ink-300" />Raw kWh</span>
            <span className="flex items-center gap-1.5"><span className="h-0.5 w-4 bg-cyan-400" />HDD/CDD model</span>
          </div>
        </div>
        <svg viewBox={`0 0 ${W} ${H}`} className="mt-4 w-full" onMouseLeave={() => setHover(null)}>
          {[0, 1, 2, 3].map((g) => (
            <line key={g} x1={P} x2={W - P} y1={P + g * ((H - 2 * P) / 3)} y2={P + g * ((H - 2 * P) / 3)} stroke="rgba(255,255,255,0.06)" />
          ))}
          <path d={path(raw)} fill="none" stroke="#aeb9ca" strokeWidth="1.5" strokeOpacity="0.7" />
          <path d={path(model)} fill="none" stroke="#22d3ee" strokeWidth="2.5" className="mk-draw" />
          {raw.map((v, i) => (
            <g key={i} onMouseEnter={() => setHover(i)}>
              <rect x={x(i) - 18} y={0} width={36} height={H} fill="transparent" />
              <circle cx={x(i)} cy={y(v)} r={hover === i ? 5 : 3} fill={hover === i ? '#40d18f' : '#d4dae3'}
                style={{ filter: hover === i ? 'drop-shadow(0 0 6px #40d18f)' : undefined, transition: 'all .15s' }} />
            </g>
          ))}
          {hover !== null && <line x1={x(hover)} x2={x(hover)} y1={P} y2={H - P} stroke="rgba(64,209,143,0.4)" strokeDasharray="3 3" />}
        </svg>
        <div className="mt-2 flex justify-between font-mono text-[10px] text-ink-500">
          {months.map((m) => <span key={m}>{m}</span>)}
        </div>
        <div className="mt-3 h-10 rounded-sm border border-white/5 bg-ink-950/60 px-3 py-2 font-mono text-xs text-ink-300">
          {hover === null ? 'Hover a point to see its normalised value →' : (
            <>{months[hover]} · raw <span className="text-white">{raw[hover]} MWh</span> · normalised <span className="text-cyan-300">{model[hover]} MWh</span> · Δ <span className={raw[hover] <= model[hover] ? 'text-brand-400' : 'text-amber-400'}>{raw[hover] - model[hover]}</span></>
          )}
        </div>
      </div>

      <div className="absolute -bottom-10 -left-4 w-56 rounded-lg border border-white/10 bg-ink-950/90 p-4 shadow-2xl backdrop-blur sm:-left-10">
        <p className="font-mono text-[10px] uppercase tracking-widest text-ink-400">Clause 6.3 · Readiness</p>
        <div className="mt-3 flex items-center gap-4">
          <svg viewBox="0 0 80 80" className="h-16 w-16 -rotate-90">
            <circle cx="40" cy="40" r={r} fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="7" />
            <circle cx="40" cy="40" r={r} fill="none" stroke="#40d18f" strokeWidth="7" strokeLinecap="round"
              strokeDasharray={c} strokeDashoffset={c * (1 - score / 100)} className="mk-glow" />
          </svg>
          <div>
            <p className="font-mono text-2xl text-white">{score}%</p>
            <p className="text-xs text-ink-400">Audit-ready</p>
          </div>
        </div>
      </div>
    </div>
  );
}
