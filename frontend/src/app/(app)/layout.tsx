'use client';

import { useEffect } from 'react';
import { usePathname, useRouter } from 'next/navigation';
import Link from 'next/link';
import { useAuth } from '@/lib/auth';
import { useAnalysis } from '@/lib/analysis';
import {
  Zap, LayoutDashboard, Building2, BarChart3, Workflow,
  Target, ShieldCheck, LogOut, LineChart, CheckCircle2, Loader2, AlertCircle,
} from 'lucide-react';

const navItems = [
  { href: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { href: '/sites', label: 'Sites & Meters', icon: Building2 },
  { href: '/baseline', label: 'Baseline & EnPI', icon: LineChart },
  { href: '/seu-analysis', label: 'SEU Analysis', icon: BarChart3 },
  { href: '/energy-review', label: 'Energy Review', icon: Workflow },
  { href: '/objectives', label: 'Objectives', icon: Target },
  { href: '/compliance', label: 'Compliance', icon: ShieldCheck },
];

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const { user, loading, logout } = useAuth();
  const {
    sites, sitesLoading, selectedSiteId, availableYears,
    baselineYear, comparisonYear, status, error,
    selectSite, setBaselineYear, setComparisonYear,
  } = useAnalysis();
  const pathname = usePathname();
  const router = useRouter();

  useEffect(() => {
    if (!loading && !user) {
      router.push('/login');
    }
  }, [user, loading, router]);

  if (loading || !user) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="text-ink-400">Loading...</div>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen bg-ink-50">
      {/* Sidebar */}
      <aside className="fixed inset-y-0 left-0 z-40 flex w-64 flex-col border-r border-ink-100 bg-white">
        <div className="flex items-center gap-2 border-b border-ink-100 px-6 py-4">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-brand-600">
            <Zap className="h-4 w-4 text-white" />
          </div>
          <span className="font-bold text-ink-900">EnMS</span>
        </div>

        <nav className="flex-1 space-y-1 overflow-y-auto p-3">
          {navItems.map((item) => {
            const active = pathname === item.href || pathname.startsWith(item.href + '/');
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition ${
                  active ? 'bg-brand-50 text-brand-700' : 'text-ink-600 hover:bg-ink-50 hover:text-ink-900'
                }`}
              >
                <Icon className="h-4 w-4" />
                {item.label}
              </Link>
            );
          })}
        </nav>

        {/* Analysis context — always visible in lower-left */}
        <div className="border-t border-ink-100 p-3">
          <div className="mb-2 flex items-center gap-1.5 px-1 text-xs font-semibold uppercase tracking-wide text-ink-400">
            <LineChart className="h-3.5 w-3.5" /> Analysis
          </div>
          <div className="space-y-2">
            <select
              value={selectedSiteId ?? ''}
              onChange={(e) => selectSite(e.target.value ? Number(e.target.value) : null)}
              className="w-full rounded-lg border border-ink-200 px-2.5 py-2 text-sm focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-100"
            >
              <option value="">{sitesLoading ? 'Loading sites…' : 'Select site…'}</option>
              {sites.map((s) => (
                <option key={s.id} value={s.id}>{s.name}</option>
              ))}
            </select>
            {selectedSiteId && availableYears.length >= 2 && (
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="mb-1 block text-[11px] font-medium text-ink-400">Baseline</label>
                  <select
                    value={baselineYear ?? ''}
                    onChange={(e) => setBaselineYear(Number(e.target.value))}
                    className="w-full rounded-lg border border-ink-200 px-2 py-1.5 text-sm focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-100"
                  >
                    {availableYears.map((y) => <option key={y} value={y}>{y}</option>)}
                  </select>
                </div>
                <div>
                  <label className="mb-1 block text-[11px] font-medium text-ink-400">Compare</label>
                  <select
                    value={comparisonYear ?? ''}
                    onChange={(e) => setComparisonYear(Number(e.target.value))}
                    className="w-full rounded-lg border border-ink-200 px-2 py-1.5 text-sm focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-100"
                  >
                    {availableYears.map((y) => <option key={y} value={y}>{y}</option>)}
                  </select>
                </div>
              </div>
            )}
            {selectedSiteId && availableYears.length < 2 && (
              <p className="px-1 text-xs text-ink-400">Not enough years of data yet.</p>
            )}
            {selectedSiteId && availableYears.length >= 2 && (
              <div className="flex items-center gap-1.5 px-1 text-xs">
                {status === 'loading' && (<><Loader2 className="h-3.5 w-3.5 animate-spin text-brand-600" /><span className="text-ink-500">Calculating…</span></>)}
                {status === 'ready' && (<><CheckCircle2 className="h-3.5 w-3.5 text-brand-600" /><span className="text-ink-500">Results ready</span></>)}
                {status === 'error' && (<><AlertCircle className="h-3.5 w-3.5 text-red-500" /><span className="truncate text-red-500" title={error}>{error}</span></>)}
                {status === 'idle' && <span className="text-ink-400">Pick years to run.</span>}
              </div>
            )}
          </div>
        </div>

        <div className="border-t border-ink-100 p-3">
          <div className="mb-2 rounded-lg bg-ink-50 px-3 py-2">
            <p className="text-sm font-medium text-ink-900">{user.name}</p>
            <p className="truncate text-xs text-ink-400">{user.email}</p>
          </div>
          <button
            onClick={logout}
            className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium text-ink-600 hover:bg-red-50 hover:text-red-600"
          >
            <LogOut className="h-4 w-4" />
            Sign out
          </button>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 pl-64">
        <div className="mx-auto max-w-7xl px-6 py-8">{children}</div>
      </main>
    </div>
  );
}
