'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { api } from '@/lib/api';
import { Building2, ShieldCheck, Zap, ArrowRight, Plus, BarChart3 } from 'lucide-react';

export default function DashboardPage() {
  const [sites, setSites] = useState<any[]>([]);
  const [score, setScore] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([api.listSites(), api.complianceScore().catch(() => null)])
      .then(([s, sc]) => {
        setSites(s);
        setScore(sc);
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="text-ink-400">Loading...</div>;

  const totalMeters = sites.reduce((sum, s) => sum + (s.meter_count || 0), 0);

  return (
    <div>
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-ink-900">Dashboard</h1>
          <p className="text-sm text-ink-500">Overview of your energy management performance</p>
        </div>
        <Link href="/sites" className="inline-flex items-center gap-2 rounded-lg bg-brand-600 px-4 py-2 text-sm font-semibold text-white hover:bg-brand-700">
          <Plus className="h-4 w-4" /> Add Site
        </Link>
      </div>

      {/* KPI Cards */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <KPICard icon={Building2} label="Sites" value={sites.length} color="brand" />
        <KPICard icon={Zap} label="Meters" value={totalMeters} color="ink" />
        <KPICard icon={ShieldCheck} label="Compliance Score" value={score ? `${score.score}%` : '—'} color="brand" />
        <KPICard icon={BarChart3} label="Clauses Complete" value={score ? `${score.complete}/${score.total_clauses}` : '—'} color="ink" />
      </div>

      {/* Quick Links */}
      <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <QuickLink href="/baseline" title="Run Baseline Analysis" desc="Climate-normalized predicted vs actual consumption" />
        <QuickLink href="/energy-review" title="Energy Review" desc="Document your ISO 50001 §6.3 energy review" />
        <QuickLink href="/compliance" title="Compliance Checklist" desc="Track clause-by-clause audit readiness" />
      </div>

      {/* Sites list */}
      <div className="mt-8">
        <h2 className="mb-4 text-lg font-semibold text-ink-900">Your Sites</h2>
        {sites.length === 0 ? (
          <div className="rounded-xl border border-dashed border-ink-200 bg-white p-12 text-center">
            <Building2 className="mx-auto mb-3 h-10 w-10 text-ink-300" />
            <p className="text-ink-500">No sites yet. Create your first site to get started.</p>
            <Link href="/sites" className="mt-4 inline-flex items-center gap-2 rounded-lg bg-brand-600 px-4 py-2 text-sm font-semibold text-white hover:bg-brand-700">
              Create Site <ArrowRight className="h-4 w-4" />
            </Link>
          </div>
        ) : (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {sites.map((site) => (
              <Link key={site.id} href={`/sites/${site.id}`}
                className="rounded-xl border border-ink-100 bg-white p-5 transition hover:border-brand-200 hover:shadow-lg">
                <h3 className="font-semibold text-ink-900">{site.name}</h3>
                <p className="mt-1 text-sm text-ink-500">{site.address || 'No address'}</p>
                <div className="mt-3 flex items-center gap-4 text-sm text-ink-400">
                  <span>{site.meter_count} meters</span>
                </div>
              </Link>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function KPICard({ icon: Icon, label, value, color }: { icon: any; label: string; value: any; color: string }) {
  return (
    <div className="rounded-xl border border-ink-100 bg-white p-5">
      <div className="flex items-center justify-between">
        <div className={`flex h-10 w-10 items-center justify-center rounded-lg ${color === 'brand' ? 'bg-brand-50' : 'bg-ink-100'}`}>
          <Icon className={`h-5 w-5 ${color === 'brand' ? 'text-brand-600' : 'text-ink-600'}`} />
        </div>
      </div>
      <p className="mt-3 text-2xl font-bold text-ink-900">{value}</p>
      <p className="text-sm text-ink-500">{label}</p>
    </div>
  );
}

function QuickLink({ href, title, desc }: { href: string; title: string; desc: string }) {
  return (
    <Link href={href} className="group rounded-xl border border-ink-100 bg-white p-5 transition hover:border-brand-200 hover:shadow-lg">
      <h3 className="font-semibold text-ink-900 group-hover:text-brand-600">{title}</h3>
      <p className="mt-1 text-sm text-ink-500">{desc}</p>
      <span className="mt-3 inline-flex items-center gap-1 text-sm font-medium text-brand-600">
        Go <ArrowRight className="h-3 w-3" />
      </span>
    </Link>
  );
}
