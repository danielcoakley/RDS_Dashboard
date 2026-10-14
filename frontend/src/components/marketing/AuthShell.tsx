import Link from 'next/link';
import { ArrowLeft, CheckCircle2 } from 'lucide-react';
import { Logo } from './Logo';

const points = [
  'Weather-normalised baselines from HDD/CDD regression',
  'Significant energy uses and EnPIs identified automatically',
  'Clause 4–10 readiness score with evidence gaps highlighted',
];

export function AuthShell({ children, maxWidth = 'max-w-md' }: { children: React.ReactNode; maxWidth?: string }) {
  return (
    <div className="grid min-h-screen lg:grid-cols-[1fr_1.1fr]">
      <aside className="relative hidden overflow-hidden bg-ink-950 p-12 text-white lg:flex lg:flex-col lg:justify-between">
        <div className="pointer-events-none absolute inset-0 marketing-grid opacity-30" />
        <div className="pointer-events-none absolute -bottom-32 -left-20 h-96 w-96 rounded-full bg-brand-500/25 blur-3xl" />
        <div className="relative"><Logo dark /></div>
        <div className="relative">
          <h2 className="max-w-md text-3xl font-bold leading-tight tracking-tight">
            Audit-ready ISO 50001 evidence, all year round.
          </h2>
          <ul className="mt-8 space-y-4">
            {points.map((p) => (
              <li key={p} className="flex gap-3 text-ink-200">
                <CheckCircle2 className="mt-0.5 h-5 w-5 flex-none text-brand-400" /> {p}
              </li>
            ))}
          </ul>
        </div>
        <p className="relative text-sm text-ink-400">
          Just looking?{' '}
          <Link href="/demo" className="font-semibold text-brand-300 hover:text-brand-200">Explore the live demo</Link>
        </p>
      </aside>
      <main className="flex items-center justify-center bg-ink-50/60 px-4 py-12">
        <div className={`w-full ${maxWidth}`}>
          <Link href="/" className="mb-8 inline-flex items-center gap-2 text-sm text-ink-500 hover:text-ink-900">
            <ArrowLeft className="h-4 w-4" /> Back to home
          </Link>
          {children}
        </div>
      </main>
    </div>
  );
}
