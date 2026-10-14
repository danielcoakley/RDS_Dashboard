'use client';

import { useEffect, useState } from 'react';
import { usePathname, useRouter } from 'next/navigation';
import Link from 'next/link';
import { useAuth } from '@/lib/auth';
import { useAnalysis } from '@/lib/analysis';
import { clearToken } from '@/lib/api';
import {
  Zap, LayoutDashboard, Building2, BarChart3, Workflow, Target, ShieldCheck,
  LogOut, LineChart, CheckCircle2, Loader2, AlertCircle, Menu, X, Sparkles,
} from 'lucide-react';

const navGroups = [
  {
    label: 'Workspace',
    items: [
      { href: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
      { href: '/sites', label: 'Sites & Meters', icon: Building2 },
    ],
  },
  {
    label: 'Analytics',
    items: [
      { href: '/baseline', label: 'Baseline & EnPI', icon: LineChart },
      { href: '/seu-analysis', label: 'SEU Analysis', icon: BarChart3 },
    ],
  },
  {
    label: 'ISO 50001',
    items: [
      { href: '/energy-review', label: 'Energy Review', icon: Workflow },
      { href: '/objectives', label: 'Objectives', icon: Target },
      { href: '/compliance', label: 'Compliance', icon: ShieldCheck },
    ],
  },
];

const selectCls =
  'w-full rounded-lg border border-white/10 bg-white/5 px-2.5 py-2 text-sm text-white focus:border-brand-400 focus:outline-none focus:ring-2 focus:ring-brand-500/30 [&>option]:text-ink-900';

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

  useEffect(() => {
    if (!loading && !user) {
      router.push('/login');
    }
  }, [user, loading, router]);

  useEffect(() => { setMobileOpen(false); }, [pathname]);

  if (loading || !user) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="text-ink-400">Loading...</div>
      </div>
    );
  }

  const isDemo = user.email.endsWith('@demo.enms.app');
  const initials = user.name.split(' ').map((p) => p[0]).join('').slice(0, 2).toUpperCase();

  return (
    <div className="min-h-screen bg-ink-50">
      {/* Mobile top bar */}
      <header className="no-print sticky top-0 z-30 flex items-center justify-between border-b border-ink-100 bg-white/90 px-4 py-3 backdrop-blur lg:hidden">
        <Brand dark={false} />
        <button
          onClick={() => setMobileOpen(true)}
          aria-label="Open navigation"
          className="rounded-lg p-2 text-ink-600 hover:bg-ink-100"
        >
          <Menu className="h-5 w-5" />
        </button>
      </header>

      {mobileOpen && (
        <div className="no-print fixed inset-0 z-40 bg-ink-950/50 backdrop-blur-sm lg:hidden" onClick={() => setMobileOpen(false)} />
      )}

      {/* Sidebar */}
      <aside
        className={`no-print fixed inset-y-0 left-0 z-50 flex w-72 flex-col bg-ink-950 text-ink-300 shadow-2xl transition-transform duration-200 lg:w-64 lg:translate-x-0 lg:shadow-none ${
          mobileOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="flex items-center justify-between px-5 py-5">
          <Brand dark />
          <button
            onClick={() => setMobileOpen(false)}
            aria-label="Close navigation"
            className="rounded-lg p-1.5 text-ink-400 hover:bg-white/10 hover:text-white lg:hidden"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <nav className="flex-1 space-y-6 overflow-y-auto px-3 pb-4">
          {navGroups.map((group) => (
            <div key={group.label}>
              <p className="mb-1.5 px-3 text-[11px] font-semibold uppercase tracking-wider text-ink-500">{group.label}</p>
              <div className="space-y-0.5">
                {group.items.map((item) => {
                  const active = pathname === item.href || pathname.startsWith(item.href + '/');
                  const Icon = item.icon;
                  return (
                    <Link
                      key={item.href}
                      href={item.href}
                      className={`group relative flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition ${
                        active ? 'bg-white/10 text-white' : 'text-ink-300 hover:bg-white/5 hover:text-white'
                      }`}
                    >
                      {active && <span className="absolute inset-y-1.5 left-0 w-0.5 rounded-full bg-brand-400" />}
                      <Icon className={`h-4 w-4 ${active ? 'text-brand-400' : 'text-ink-500 group-hover:text-ink-300'}`} />
                      {item.label}
                    </Link>
                  );
                })}
              </div>
            </div>
          ))}
        </nav>

        {/* Analysis context */}
        <div className="border-t border-white/10 p-3">
          <div className="mb-2 flex items-center gap-1.5 px-1 text-[11px] font-semibold uppercase tracking-wider text-ink-500">
            <LineChart className="h-3.5 w-3.5" /> Analysis context
          </div>
          <div className="space-y-2">
            <select
              value={selectedSiteId ?? ''}
              onChange={(e) => selectSite(e.target.value ? Number(e.target.value) : null)}
              className={selectCls}
            >
              <option value="">{sitesLoading ? 'Loading sites…' : 'Select site…'}</option>
              {sites.map((s) => (
                <option key={s.id} value={s.id}>{s.name}</option>
              ))}
            </select>
            {selectedSiteId && availableYears.length >= 2 && (
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="mb-1 block text-[11px] font-medium text-ink-500">Baseline</label>
                  <select value={baselineYear ?? ''} onChange={(e) => setBaselineYear(Number(e.target.value))} className={selectCls}>
                    {availableYears.map((y) => <option key={y} value={y}>{y}</option>)}
                  </select>
                </div>
                <div>
                  <label className="mb-1 block text-[11px] font-medium text-ink-500">Compare</label>
                  <select value={comparisonYear ?? ''} onChange={(e) => setComparisonYear(Number(e.target.value))} className={selectCls}>
                    {availableYears.map((y) => <option key={y} value={y}>{y}</option>)}
                  </select>
                </div>
              </div>
            )}
            {selectedSiteId && availableYears.length < 2 && (
              <p className="px-1 text-xs text-ink-500">Not enough years of data yet.</p>
            )}
            {selectedSiteId && availableYears.length >= 2 && (
              <div className="flex items-center gap-1.5 px-1 text-xs">
                {status === 'loading' && (<><Loader2 className="h-3.5 w-3.5 animate-spin text-brand-400" /><span className="text-ink-400">Calculating…</span></>)}
                {status === 'ready' && (<><CheckCircle2 className="h-3.5 w-3.5 text-brand-400" /><span className="text-ink-400">Results ready</span></>)}
                {status === 'error' && (<><AlertCircle className="h-3.5 w-3.5 text-red-400" /><span className="truncate text-red-400" title={error}>{error}</span></>)}
                {status === 'idle' && <span className="text-ink-500">Pick years to run.</span>}
              </div>
            )}
          </div>
        </div>

        {/* User */}
        <div className="flex items-center gap-3 border-t border-white/10 p-3">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-brand-400 to-brand-700 text-xs font-bold text-white">
            {initials}
          </div>
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-medium text-white">{user.name}</p>
            <p className="truncate text-xs text-ink-500">{isDemo ? 'Demo workspace' : user.email}</p>
          </div>
          <button
            onClick={logout}
            aria-label="Sign out"
            title="Sign out"
            className="rounded-lg p-2 text-ink-400 hover:bg-white/10 hover:text-red-400"
          >
            <LogOut className="h-4 w-4" />
          </button>
        </div>
      </aside>

      {/* Main content */}
      <main className="app-main lg:pl-64">
        {isDemo && (
          <div className="no-print flex flex-wrap items-center justify-center gap-x-4 gap-y-2 bg-gradient-to-r from-brand-700 to-brand-600 px-4 py-2.5 text-sm text-white">
            <span className="flex items-center gap-2">
              <Sparkles className="h-4 w-4" />
              <span><b>Demo workspace</b> — explore with sample data from Demo Manufacturing Ltd.</span>
            </span>
            <a
              href="/signup"
              onClick={() => clearToken()}
              className="rounded-full bg-white px-3 py-1 text-xs font-semibold text-brand-700 shadow-sm hover:bg-brand-50"
            >
              Start free trial
            </a>
          </div>
        )}
        <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 sm:py-8">{children}</div>
      </main>
    </div>
  );
}

function Brand({ dark }: { dark: boolean }) {
  return (
    <Link href="/dashboard" className="flex items-center gap-2.5">
      <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-brand-400 to-brand-700 shadow-sm">
        <Zap className="h-4 w-4 text-white" />
      </div>
      <div className="leading-tight">
        <span className={`block text-sm font-bold ${dark ? 'text-white' : 'text-ink-900'}`}>EnMS</span>
        <span className={`block text-[10px] font-medium uppercase tracking-wider ${dark ? 'text-ink-500' : 'text-ink-400'}`}>ISO 50001</span>
      </div>
    </Link>
  );
}
