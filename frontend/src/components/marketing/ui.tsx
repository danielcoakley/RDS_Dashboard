import Link from 'next/link';
import { ArrowRight } from 'lucide-react';

export function Eyebrow({ children }: { children: React.ReactNode }) {
  return <p className="font-mono text-xs uppercase tracking-[0.2em] text-brand-400">{children}</p>;
}

export function SectionHeading({ eyebrow, title, sub }: { eyebrow: string; title: string; sub?: string }) {
  return (
    <div className="max-w-2xl">
      <Eyebrow>{eyebrow}</Eyebrow>
      <h2 className="mt-4 text-3xl font-semibold leading-tight tracking-tight text-white sm:text-4xl">{title}</h2>
      {sub && <p className="mt-4 text-base leading-relaxed text-ink-300">{sub}</p>}
    </div>
  );
}

export function CTAButtons({ center = false }: { center?: boolean }) {
  return (
    <div className={`flex flex-col gap-3 sm:flex-row ${center ? 'sm:justify-center' : ''}`}>
      <Link href="/signup" className="inline-flex items-center justify-center gap-2 rounded-sm bg-brand-400 px-6 py-3 font-semibold text-ink-950 hover:bg-brand-300">
        Start free trial <ArrowRight className="h-4 w-4" />
      </Link>
      <Link href="/demo" className="inline-flex items-center justify-center rounded-sm border border-white/15 px-6 py-3 font-medium text-white hover:border-white/30 hover:bg-white/5">
        Explore live demo
      </Link>
    </div>
  );
}

export function MarketingShell({ children }: { children: React.ReactNode }) {
  return <div className="min-h-screen bg-ink-950 text-ink-100">{children}</div>;
}
