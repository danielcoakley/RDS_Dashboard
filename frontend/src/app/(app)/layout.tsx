'use client';

import { useEffect, useState } from 'react';
import { usePathname, useRouter } from 'next/navigation';
import Link from 'next/link';
import { useAuth } from '@/lib/auth';
import { useAnalysis } from '@/lib/analysis';
import { isDemoUser } from '@/lib/api';
import {
  Zap, LayoutDashboard, Building2, BarChart3, Workflow,
  Target, ShieldCheck, LogOut, LineChart, CheckCircle2, Loader2, AlertCircle, FileText,
  Menu, X, Sparkles, ArrowRight,
} from 'lucide-react';

const navItems = [
  { href: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { href: '/sites', label: 'Sites & Meters', icon: Building2 },
  { href: '/baseline', label: 'Baseline & EnPI', icon: LineChart },
  { href: '/seu-analysis', label: 'SEU Analysis', icon: BarChart3 },
  { href: '/energy-review', label: 'Energy Review', icon: Workflow },
  { href: '/objectives', label: 'Objectives', icon: Target },
  { href: '/compliance', label: 'Compliance', icon: ShieldCheck },
  { href: '/reports', label: 'Reports & Audit', icon: FileText },
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
  const [mobileOpen, setMobileOpen] = useState(false);

  // Close the mobile drawer whenever the route changes.
  useEffect(() => { setMobileOpen(false); }, [pathname]);

  useEffect(() => {
    if (!loading && !user) {
      router.push('/login');
    }
  }, [user, loading, router]);

  if (loading || !user) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-brand-600" />
      </div>
    );
  }

  const demo = isDemoUser(user);
  const initials = user.name?.split(' ').map((p) => p[0]).slice(0, 2).join('').toUpperCase() || '?';

  return (
    <div className="min-h-screen bg-ink-50">
      {/* Mobile top bar */}
      <header className="sticky top-0 z-30 flex items-center justify-between border-b border-ink-100 bg-white/90 px-4 py-3 backdrop-blur lg:hidden print:hidden">
        <Brand />
        <button
          onClick={() => setMobileOpen(true)}
          className="rounded-lg p-2 text-ink-600 hover:bg-ink-100"
          aria-label="Open navigation"
        >
          <Menu className="h-5 w-5" />
        </button>
      </header>

      {/* Mobile backdrop */}
      {mobileOpen && (
        <div className="fixed inset-0 z-40 bg-ink-950/40 backdrop-blur-sm lg:hidden" onClick={() => setMobileOpen(false)} />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed inset-y-0 left-0 z-50 flex w-72 flex-col border-r border-ink-100 bg-white shadow-xl transition-transform duration-200 lg:w-64 lg:translate-x-0 lg:shadow-none print:hidden ${
          mobileOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="flex items-center justify-between border-b border-ink-100 px-5 py-4">
          <Brand />
          <button
            onClick={() => setMobileOpen(false)}
            className="rounded-lg p-1.5 text-ink-500 hover:bg-ink-100 lg:hidden"
            aria-label="Close navigation"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="px-5 pb-1 pt-4 text-[11px] font-semibold uppercase tracking-wider text-ink-400">Workspace</div>
        <nav className="flex-1 space-y-0.5 overflow-y-auto px-3 pb-3">
          {navItems.map((item) => {
            const active = pathname === item.href || pathname.startsWith(item.href + '/');
            const Icon = item.icon;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`group relative flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition ${
                  active ? 'bg-brand-50 text-brand-800' : 'text-ink-600 hover:bg-ink-50 hover:text-ink-900'
                }`}
              >
                {active && <span className="absolute inset-y-1.5 left-0 w-1 rounded-r bg-brand-600" />}
                <Icon className={`h-4 w-4 ${active ? 'text-brand-600' : 'text-ink-400 group-hover:text-ink-600'}`} />
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
          <div className="mb-2 flex items-center gap-3 rounded-lg bg-ink-50 px-3 py-2">
            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-brand-500 to-brand-700 text-xs font-bold text-white">
              {initials}
            </div>
            <div className="min-w-0">
              <p className="truncate text-sm font-medium text-ink-900">{user.name}</p>
              <p className="truncate text-xs text-ink-400">{demo ? 'Demo workspace' : `${user.role} | ${user.email}`}</p>
            </div>
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
      <main className="lg:pl-64 print:pl-0">
        {demo && (
          <div className="border-b border-brand-200 bg-gradient-to-r from-brand-50 via-white to-brand-50 print:hidden">
            <div className="mx-auto flex max-w-7xl flex-col gap-2 px-4 py-2.5 sm:flex-row sm:items-center sm:justify-between sm:px-6">
              <p className="flex items-center gap-2 text-sm text-ink-700">
                <Sparkles className="h-4 w-4 shrink-0 text-brand-600" />
                <span>
                  <span className="font-semibold text-ink-900">Demo workspace</span> &mdash; explore sample data freely. Changes are not kept.
                </span>
              </p>
              <Link
                href="/signup"
                className="inline-flex items-center justify-center gap-1.5 rounded-lg bg-brand-600 px-3 py-1.5 text-sm font-semibold text-white shadow-sm hover:bg-brand-700"
              >
                Start free trial <ArrowRight className="h-3.5 w-3.5" />
              </Link>
            </div>
          </div>
        )}
        <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 sm:py-8 print:max-w-none print:p-0">{children}</div>
      </main>
    </div>
  );
}

function Brand() {
  return (
    <Link href="/dashboard" className="flex items-center gap-2.5">
      <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-brand-500 to-brand-700 shadow-sm shadow-brand-600/30">
        <Zap className="h-5 w-5 text-white" />
      </div>
      <div className="leading-tight">
        <p className="font-bold tracking-tight text-ink-900">EnMS</p>
        <p className="text-[11px] text-ink-400">ISO 50001 energy management</p>
      </div>
    </Link>
  );
}
