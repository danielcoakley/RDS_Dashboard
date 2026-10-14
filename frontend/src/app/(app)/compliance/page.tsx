'use client';

import { useEffect, useMemo, useState } from 'react';
import { api } from '@/lib/api';
import { useAuth } from '@/lib/auth';
import { CheckCircle2, Clock, Circle, Printer, ChevronDown, Save, Loader2 } from 'lucide-react';
import { RadialScore, SectionBar } from '@/components/app/ReadinessCard';
import { SECTIONS, SECTION_TITLES, computeReadiness, sectionOf, type ComplianceItem } from '@/components/app/readiness';

const STATUS = {
  not_started: { icon: Circle, chip: 'bg-ink-100 text-ink-600', iconCls: 'text-ink-300', label: 'Not started' },
  in_progress: { icon: Clock, chip: 'bg-amber-50 text-amber-700', iconCls: 'text-amber-500', label: 'In progress' },
  complete: { icon: CheckCircle2, chip: 'bg-brand-50 text-brand-700', iconCls: 'text-brand-600', label: 'Complete' },
} as const;
type StatusKey = keyof typeof STATUS;
const statusOf = (s: string): StatusKey => (s in STATUS ? (s as StatusKey) : 'not_started');

export default function CompliancePage() {
  const { user } = useAuth();
  const [items, setItems] = useState<ComplianceItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState<string | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const [drafts, setDrafts] = useState<Record<string, string>>({});

  useEffect(() => {
    api.listCompliance()
      .then((data) => setItems(data as ComplianceItem[]))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  // Open the clause linked from the dashboard (e.g. /compliance#clause-6.3).
  useEffect(() => {
    if (loading || typeof window === 'undefined') return;
    const ref = window.location.hash.replace('#clause-', '');
    if (ref) {
      setOpen(ref);
      document.getElementById(`clause-${ref}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [loading]);

  const readiness = useMemo(() => computeReadiness(items), [items]);

  const save = async (item: ComplianceItem, patch: Partial<ComplianceItem>) => {
    const next = { ...item, ...patch };
    setSaving(item.clause_ref);
    setItems((prev) => prev.map((i) => (i.clause_ref === item.clause_ref ? next : i))); // optimistic
    try {
      await api.updateCompliance(item.clause_ref, { status: next.status, evidence: next.evidence || null });
      setError(null);
    } catch (e) {
      setItems((prev) => prev.map((i) => (i.clause_ref === item.clause_ref ? item : i)));
      setError((e as Error).message);
    } finally {
      setSaving(null);
    }
  };

  if (loading) {
    return <div className="flex items-center gap-2 text-ink-400"><Loader2 className="h-4 w-4 animate-spin" /> Loading checklist…</div>;
  }

  // Viewers can read the checklist; the API enforces RBAC, this just avoids dead controls.
  const canEdit = user?.role !== 'viewer';

  return (
    <div className="compliance-print">
      <div className="mb-6 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-ink-900">ISO 50001 compliance</h1>
          <p className="text-sm text-ink-500">Clause-by-clause audit readiness across ISO 50001:2018 §4–§10</p>
          <p className="hidden text-xs text-ink-500 print:block">
            Audit evidence summary · prepared by {user?.name} · {new Date().toLocaleDateString()}
          </p>
        </div>
        <button
          onClick={() => window.print()}
          className="inline-flex items-center gap-2 rounded-lg border border-ink-200 bg-white px-4 py-2 text-sm font-semibold text-ink-700 shadow-sm hover:bg-ink-50 print:hidden"
        >
          <Printer className="h-4 w-4" /> Export audit evidence
        </button>
      </div>

      {error && <div className="mb-4 rounded-lg bg-red-50 px-4 py-3 text-sm text-red-700 print:hidden">{error}</div>}

      {/* Summary */}
      <section className="mb-6 grid gap-6 rounded-2xl border border-ink-100 bg-white p-6 shadow-sm md:grid-cols-[auto_1fr] print:break-inside-avoid print:shadow-none">
        <div className="flex items-center gap-5">
          <RadialScore score={readiness.score} />
          <div className="space-y-1 text-sm">
            <p><span className="font-semibold text-brand-700">{readiness.complete}</span> <span className="text-ink-500">complete</span></p>
            <p><span className="font-semibold text-amber-600">{readiness.in_progress}</span> <span className="text-ink-500">in progress</span></p>
            <p><span className="font-semibold text-ink-500">{readiness.not_started}</span> <span className="text-ink-500">not started</span></p>
          </div>
        </div>
        <div className="grid gap-x-6 gap-y-2.5 sm:grid-cols-2">
          {readiness.by_section.map((s) => <SectionBar key={s.section} {...s} />)}
        </div>
      </section>

      {/* Checklist grouped by clause section */}
      <div className="space-y-5">
        {SECTIONS.map((section) => {
          const list = items.filter((i) => sectionOf(i.clause_ref) === section);
          if (list.length === 0) return null;
          const s = readiness.by_section.find((b) => b.section === section);
          return (
            <section key={section} className="overflow-hidden rounded-xl border border-ink-100 bg-white print:break-inside-avoid">
              <header className="flex items-center justify-between gap-3 border-b border-ink-100 bg-ink-50 px-5 py-3">
                <h2 className="font-semibold text-ink-900">§{section} {SECTION_TITLES[section]}</h2>
                <div className="flex items-center gap-3">
                  <div className="hidden h-2 w-28 overflow-hidden rounded-full bg-ink-200 sm:block">
                    <div className="h-full rounded-full bg-brand-500" style={{ width: `${s?.score ?? 0}%` }} />
                  </div>
                  <span className="text-xs font-medium text-ink-500">{s?.complete ?? 0}/{list.length}</span>
                </div>
              </header>
              <ul className="divide-y divide-ink-100">
                {list.map((item) => {
                  const st = STATUS[statusOf(item.status)];
                  const Icon = st.icon;
                  const expanded = open === item.clause_ref;
                  const draft = drafts[item.clause_ref] ?? item.evidence ?? '';
                  return (
                    <li key={item.clause_ref} id={`clause-${item.clause_ref}`} className="px-5 py-3">
                      <div className="flex flex-wrap items-center gap-3">
                        <button
                          onClick={() => setOpen(expanded ? null : item.clause_ref)}
                          className="flex min-w-0 flex-1 items-center gap-3 text-left"
                        >
                          <Icon className={`h-5 w-5 shrink-0 ${st.iconCls}`} />
                          <span className="min-w-0">
                            <span className="block text-sm font-medium text-ink-900">
                              <span className="text-ink-400">§{item.clause_ref}</span> {item.clause_title}
                            </span>
                            {item.evidence && !expanded && (
                              <span className="block truncate text-xs text-ink-500 print:whitespace-normal">Evidence: {item.evidence}</span>
                            )}
                          </span>
                          <ChevronDown className={`ml-auto h-4 w-4 shrink-0 text-ink-300 transition print:hidden ${expanded ? 'rotate-180' : ''}`} />
                        </button>

                        {/* Status toggle */}
                        <div className="flex rounded-lg bg-ink-50 p-0.5 print:hidden" role="group" aria-label={`Status for ${item.clause_ref}`}>
                          {(Object.keys(STATUS) as StatusKey[]).map((key) => (
                            <button
                              key={key}
                              disabled={!canEdit || saving === item.clause_ref}
                              onClick={() => key !== item.status && save(item, { status: key })}
                              className={`rounded-md px-2.5 py-1 text-xs font-medium transition disabled:cursor-not-allowed ${
                                item.status === key ? `${STATUS[key].chip} shadow-sm` : 'text-ink-400 hover:text-ink-700'
                              }`}
                            >
                              {STATUS[key].label}
                            </button>
                          ))}
                        </div>
                        <span className={`hidden rounded-full px-2.5 py-0.5 text-xs font-medium print:inline ${st.chip}`}>{st.label}</span>
                      </div>

                      {expanded && (
                        <div className="mt-3 pl-8 print:hidden">
                          <label className="mb-1 block text-xs font-medium text-ink-500">Evidence notes</label>
                          <textarea
                            value={draft}
                            disabled={!canEdit}
                            onChange={(e) => setDrafts((d) => ({ ...d, [item.clause_ref]: e.target.value }))}
                            rows={3}
                            placeholder="e.g. Energy policy v2 approved by top management on 12 March; stored in EnMS/Policy.pdf"
                            className="w-full rounded-lg border border-ink-200 px-3 py-2 text-sm focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-100"
                          />
                          {canEdit && (
                            <button
                              disabled={saving === item.clause_ref || draft === (item.evidence ?? '')}
                              onClick={() => save(item, { evidence: draft })}
                              className="mt-2 inline-flex items-center gap-1.5 rounded-lg bg-brand-600 px-3 py-1.5 text-xs font-semibold text-white hover:bg-brand-700 disabled:opacity-50"
                            >
                              {saving === item.clause_ref ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Save className="h-3.5 w-3.5" />}
                              Save evidence
                            </button>
                          )}
                        </div>
                      )}
                    </li>
                  );
                })}
              </ul>
            </section>
          );
        })}
      </div>

      <style jsx global>{`
        @media print {
          @page { margin: 16mm; }
          body { background: #fff !important; }
          .compliance-print { font-size: 12px; }
        }
      `}</style>
    </div>
  );
}
