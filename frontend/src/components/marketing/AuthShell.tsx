import Link from 'next/link';
import { ArrowLeft } from 'lucide-react';
import Logo from './Logo';

export default function AuthShell({ title, sub, wide = false, children }: { title: string; sub: string; wide?: boolean; children: React.ReactNode }) {
  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-ink-950 px-4 py-12">
      <div className="mk-grid pointer-events-none absolute inset-0" aria-hidden />
      <div className="pointer-events-none absolute left-1/2 top-0 h-80 w-[40rem] -translate-x-1/2 rounded-full bg-brand-500/10 blur-3xl" aria-hidden />
      <div className={`relative w-full ${wide ? 'max-w-lg' : 'max-w-md'}`}>
        <Link href="/" className="mb-8 inline-flex items-center gap-2 text-sm text-ink-400 hover:text-white">
          <ArrowLeft className="h-4 w-4" /> Back to home
        </Link>
        <div className="rounded-lg border border-white/10 bg-white p-8 shadow-2xl shadow-black/50">
          <div className="mb-8 [&_span:last-child]:text-ink-950"><Logo /></div>
          <h1 className="text-2xl font-semibold tracking-tight text-ink-950">{title}</h1>
          <p className="mt-1 text-sm text-ink-500">{sub}</p>
          {children}
        </div>
      </div>
    </div>
  );
}
