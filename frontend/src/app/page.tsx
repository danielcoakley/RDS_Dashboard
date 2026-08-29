'use client';

import Link from 'next/link';
import { useEffect } from 'react';
import { useAuth } from '@/lib/auth';
import {
  Activity, Building2, CloudSun, LineChart, ShieldCheck,
  Target, ArrowRight, CheckCircle2, BarChart3, Workflow, Zap,
} from 'lucide-react';

export default function LandingPage() {
  const { user } = useAuth();

  return (
    <div className="min-h-screen bg-white">
      {/* Nav */}
      <nav className="sticky top-0 z-50 border-b border-ink-100 bg-white/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-2">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand-600">
              <Zap className="h-5 w-5 text-white" />
            </div>
            <span className="text-xl font-bold text-ink-900">EnMS</span>
          </div>
          <div className="flex items-center gap-4">
            {user ? (
              <Link href="/dashboard" className="rounded-lg bg-brand-600 px-5 py-2 text-sm font-semibold text-white hover:bg-brand-700">
                Dashboard
              </Link>
            ) : (
              <>
                <Link href="/login" className="text-sm font-medium text-ink-600 hover:text-ink-900">Sign in</Link>
                <Link href="/signup" className="rounded-lg bg-brand-600 px-5 py-2 text-sm font-semibold text-white hover:bg-brand-700">
                  Get Started
                </Link>
              </>
            )}
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-brand-50 via-white to-ink-50" />
        <div className="relative mx-auto max-w-7xl px-6 py-24 lg:py-32">
          <div className="mx-auto max-w-3xl text-center">
            <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-brand-200 bg-brand-50 px-4 py-1.5 text-sm font-medium text-brand-700">
              <ShieldCheck className="h-4 w-4" />
              ISO 50001:2018 Compliant
            </div>
            <h1 className="text-5xl font-bold tracking-tight text-ink-950 sm:text-6xl">
              Energy management,<br />simplified for certification.
            </h1>
            <p className="mt-6 text-lg leading-relaxed text-ink-600">
              Automate your energy baseline, track performance with climate-normalized analytics,
              and manage the full ISO 50001 PDCA cycle — from energy review to audit readiness.
            </p>
            <div className="mt-10 flex items-center justify-center gap-4">
              <Link href="/signup" className="inline-flex items-center gap-2 rounded-xl bg-brand-600 px-7 py-3.5 text-base font-semibold text-white shadow-lg shadow-brand-600/25 hover:bg-brand-700">
                Start free trial
                <ArrowRight className="h-4 w-4" />
              </Link>
              <Link href="/dashboard" className="inline-flex items-center gap-2 rounded-xl border border-ink-200 px-7 py-3.5 text-base font-semibold text-ink-700 hover:bg-ink-50">
                See it in action
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="border-t border-ink-100 py-24">
        <div className="mx-auto max-w-7xl px-6">
          <div className="mx-auto max-w-2xl text-center">
            <h2 className="text-3xl font-bold text-ink-950">Everything you need for ISO 50001</h2>
            <p className="mt-4 text-ink-600">
              From data collection to certification audit — one platform covering the entire Plan-Do-Check-Act cycle.
            </p>
          </div>
          <div className="mt-16 grid gap-8 sm:grid-cols-2 lg:grid-cols-3">
            <FeatureCard icon={CloudSun} title="Automated Weather Data" desc="HDD and CDD fetched automatically from Open-Meteo based on your site location. No manual uploads — weather normalization built in." />
            <FeatureCard icon={LineChart} title="Baseline Modeling" desc="Regression-based energy baselines per meter. Climate-normalized predicted vs actual consumption with savings tracking." />
            <FeatureCard icon={Building2} title="Multi-Site Management" desc="Create organizations, sites, and meters. Assign SEU categories. Upload energy data with flexible CSV parsing." />
            <FeatureCard icon={Workflow} title="Energy Review" desc="Guided ISO 50001 §6.3 workflow: identify energy sources, significant energy uses, and current performance." />
            <FeatureCard icon={Target} title="Objectives & Targets" desc="Set energy reduction targets, track progress against baselines, and monitor status across your organization." />
            <FeatureCard icon={ShieldCheck} title="Compliance Tracking" desc="Clause-by-clause ISO 50001 compliance checklist with evidence tracking and audit readiness score." />
          </div>
        </div>
      </section>

      {/* How it works */}
      <section className="bg-ink-50 py-24">
        <div className="mx-auto max-w-7xl px-6">
          <div className="mx-auto max-w-2xl text-center">
            <h2 className="text-3xl font-bold text-ink-950">How it works</h2>
            <p className="mt-4 text-ink-600">Get from raw metering data to certification-ready analytics in four steps.</p>
          </div>
          <div className="mt-16 grid gap-8 md:grid-cols-4">
            <StepCard num="1" icon={Building2} title="Set up your organization" desc="Create your org, add sites with addresses (auto-geocoded), and define your meters with SEU categories." />
            <StepCard num="2" icon={Activity} title="Upload energy data" desc="Drag-and-drop CSV upload. Our flexible parser handles your existing export format automatically." />
            <StepCard num="3" icon={BarChart3} title="Run analytics" desc="Baseline models, EnPIs, Sankey energy flow, and SEU breakdown — all climate-normalized." />
            <StepCard num="4" icon={ShieldCheck} title="Track compliance" desc="Energy review, objectives, and clause-by-clause compliance checklist for audit readiness." />
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-24">
        <div className="mx-auto max-w-4xl px-6">
          <div className="rounded-3xl bg-gradient-to-br from-brand-600 to-brand-800 px-8 py-16 text-center shadow-2xl shadow-brand-600/20">
            <h2 className="text-3xl font-bold text-white">Ready to achieve ISO 50001?</h2>
            <p className="mt-4 text-brand-50">Start your free trial today. No credit card required.</p>
            <Link href="/signup" className="mt-8 inline-flex items-center gap-2 rounded-xl bg-white px-8 py-3.5 text-base font-semibold text-brand-700 hover:bg-brand-50">
              Get started free
              <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-ink-100 py-12">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6">
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-brand-600">
              <Zap className="h-4 w-4 text-white" />
            </div>
            <span className="font-bold text-ink-900">EnMS</span>
          </div>
          <p className="text-sm text-ink-400">ISO 50001 Energy Management Platform</p>
        </div>
      </footer>
    </div>
  );
}

function FeatureCard({ icon: Icon, title, desc }: { icon: any; title: string; desc: string }) {
  return (
    <div className="rounded-2xl border border-ink-100 bg-white p-8 transition hover:border-brand-200 hover:shadow-lg">
      <div className="mb-5 flex h-12 w-12 items-center justify-center rounded-xl bg-brand-50">
        <Icon className="h-6 w-6 text-brand-600" />
      </div>
      <h3 className="text-lg font-semibold text-ink-900">{title}</h3>
      <p className="mt-2 text-sm leading-relaxed text-ink-500">{desc}</p>
    </div>
  );
}

function StepCard({ num, icon: Icon, title, desc }: { num: string; icon: any; title: string; desc: string }) {
  return (
    <div className="relative rounded-2xl border border-ink-100 bg-white p-8">
      <div className="absolute -top-3 -left-3 flex h-8 w-8 items-center justify-center rounded-full bg-brand-600 text-sm font-bold text-white">
        {num}
      </div>
      <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-lg bg-brand-50">
        <Icon className="h-5 w-5 text-brand-600" />
      </div>
      <h3 className="font-semibold text-ink-900">{title}</h3>
      <p className="mt-2 text-sm text-ink-500">{desc}</p>
    </div>
  );
}
