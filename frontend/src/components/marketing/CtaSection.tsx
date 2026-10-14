import Link from 'next/link';
import { ArrowRight } from 'lucide-react';

export function CtaButtons({ center = false, dark = false }: { center?: boolean; dark?: boolean }) {
  return (
    <div className={`flex flex-col gap-3 sm:flex-row ${center ? 'justify-center' : ''}`}>
      <Link
        href="/signup"
        className="group inline-flex items-center justify-center gap-2 rounded-xl bg-brand-600 px-6 py-3.5 text-sm font-semibold text-white shadow-lg shadow-brand-600/25 transition hover:bg-brand-700"
      >
        Start free trial <ArrowRight className="h-4 w-4 transition group-hover:translate-x-0.5" />
      </Link>
      <Link
        href="/demo"
        className={`inline-flex items-center justify-center gap-2 rounded-xl px-6 py-3.5 text-sm font-semibold transition ${
          dark
            ? 'border border-white/20 text-white hover:bg-white/10'
            : 'border border-ink-200 bg-white text-ink-800 hover:border-ink-300 hover:bg-ink-50'
        }`}
      >
        Explore live demo
      </Link>
    </div>
  );
}

export function FinalCta() {
  return (
    <section className="px-6 py-20">
      <div className="relative mx-auto max-w-6xl overflow-hidden rounded-3xl bg-ink-950 px-8 py-16 text-center sm:px-16">
        <div className="pointer-events-none absolute inset-0 marketing-grid opacity-40" />
        <div className="pointer-events-none absolute -top-24 left-1/2 h-64 w-[36rem] -translate-x-1/2 rounded-full bg-brand-500/30 blur-3xl" />
        <div className="relative">
          <h2 className="mx-auto max-w-2xl text-3xl font-bold tracking-tight text-white sm:text-4xl">
            Walk into your next ISO 50001 audit with the evidence already in place
          </h2>
          <p className="mx-auto mt-4 max-w-xl text-ink-300">
            Upload a year of meter data and see your weather-normalised baseline, SEUs and readiness score in minutes.
            14-day free trial. No credit card required.
          </p>
          <div className="mt-8"><CtaButtons center dark /></div>
        </div>
      </div>
    </section>
  );
}
