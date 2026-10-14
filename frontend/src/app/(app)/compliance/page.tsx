'use client';

import { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import { useAuth } from '@/lib/auth';
import { ShieldCheck, CheckCircle2, Clock, Circle, Download, ChevronDown } from 'lucide-react';

const STATUS: Record<string, { icon: any; label: string; active: string; text: string }> = {
  not_started: { icon: Circle, label: 'Not started', active: 'bg-ink-700 text-white', text: 'text-ink-400' },
  in_progress: { icon: Clock, label: 'In progress', active: 'bg-amber-500 text-white', text: 'text-amber-600' },
  complete: { icon: CheckCircle2, label: 'Complete', active: 'bg-brand-600 text-white', text: 'text-brand-600' },
};

export default function CompliancePage() {
  const { user } = useAuth();
  const [items, setItems] = useState<any[]>([]);
  const [readiness, setReadiness] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState<string | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const [drafts, setDrafts] = useState<Record<string, string>>({});

  const load = async () => {
    const [list, rd]: any = await Promise.all([api.listCompliance(), api.complianceReadiness()]);
    setItems(list);
    setReadiness(rd);
    setLoading(false);
  };

  useEffect(() => { load(); }, []);

  useEffect(() => {
    if (loading || typeof window === 'undefined') return;
    const ref = window.location.hash.replace('#clause-', '');
    if (ref) {
      setOpen(ref);
      document.getElementById(`clause-${ref}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [loading]);

  const save = async (item: any, status: string, evidence: string | null) => {
    setSaving(item.clause_ref);
    try {
      await api.updateCompliance(item.clause_ref, { status, evidence });
      await load();
    } finally {
      setSaving(null);
    }
  };

  if (loading) return <div className="text-ink-400">Loading...</div>;

  return (
    <div className="print-area">
      <div className="mb-8 flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-ink-900">ISO 50001 Compliance</h1>
          <p className="text-sm text-ink-500">Clause-by-clause audit readiness across ISO 50001:2018 §4–§10</p>
          <p className="print-only mt-1 text-xs text-ink-500">
            Audit evidence export · {user?.email} · {new Date().toLocaleDateString()}
          </p>
        </div>
        <button
          onClick={() => window.print()}
          className="no-print inline-flex items-center gap-2 rounded-lg bg-brand-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-brand-700"
        >
          <Download className="h-4 w-4" /> Export audit evidence
        </button>
      </div>

      {readiness && (
        <div className="mb-8 rounded-2xl border border-ink-100 bg-white p-6 shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-6">
            <div className="flex items-center gap-3">
              <ShieldCheck className="h-8 w-8 text-brand-600" />
              <div>
                <p className="text-3xl font-bold text-ink-900">{readiness.score}%</p>
                <p className="text-sm text-ink-500">Audit readiness</p>
              </div>
            </div>
            <div className="flex gap-6 text-center">
              <Stat value={readiness.counts.complete} label="Complete" cls="text-brand-600" />
              <Stat value={readiness.counts.in_progress} label="In progress" cls="text-amber-600" />
              <Stat value={readiness.counts.not_started} label="Not started" cls="text-ink-400" />
            </div>
          </div>
          <div className="mt-4 h-2.5 overflow-hidden rounded-full bg-ink-100">
            <div className="h-full rounded-full bg-brand-500 transition-all" style={{ width: `${readiness.score}%` }} />
          </div>
        </div>
      )}

      <div className="space-y-6">
        {readiness?.by_section.map((sec: any) => {
          const sectionItems = items.filter((i) => i.clause_ref.split('.')[0] === sec.section);
          return (
            <section key={sec.section} className="avoid-break overflow-hidden rounded-xl border border-ink-100 bg-white">
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-ink-100 bg-ink-50 px-5 py-3">
                <h2 className="font-semibold text-ink-900">§{sec.section} — {sec.title}</h2>
                <div className="flex items-center gap-3 text-xs text-ink-500">
                  <span>{sec.complete}/{sec.total} complete</span>
                  <div className="h-1.5 w-28 overflow-hidden rounded-full bg-ink-200">
                    <div className="h-full rounded-full bg-brand-500" style={{ width: `${sec.score}%` }} />
                  </div>
                  <span className="w-9 text-right font-semibold text-ink-900">{sec.score}%</span>
                </div>
              </div>
              <div className="divide-y divide-ink-100">
                {sectionItems.map((item) => {
                  const sc = STATUS[item.status] || STATUS.not_started;
                  const Icon = sc.icon;
                  const isOpen = open === item.clause_ref;
                  const draft = drafts[item.clause_ref] ?? item.evidence ?? '';
                  return (
                    <div key={item.clause_ref} id={`clause-${item.clause_ref}`} className="px-5 py-3">
                      <div className="flex flex-wrap items-center justify-between gap-3">
                        <button
                          onClick={() => setOpen(isOpen ? null : item.clause_ref)}
                          className="flex min-w-0 flex-1 items-center gap-3 text-left"
                        >
                          <Icon className={`h-5 w-5 shrink-0 ${sc.text}`} />
                          <div className="min-w-0">
                            <p className="text-sm font-medium text-ink-900">§{item.clause_ref} {item.clause_title}</p>
                            <p className="print-only text-xs text-ink-500">Status: {sc.label}</p>
                            {item.evidence && !isOpen && <p className="evidence truncate text-xs text-ink-500">{item.evidence}</p>}
                          </div>
                          <ChevronDown className={`no-print h-4 w-4 shrink-0 text-ink-300 transition ${isOpen ? 'rotate-180' : ''}`} />
                        </button>
                        <div className="no-print inline-flex rounded-lg border border-ink-200 p-0.5" role="group" aria-label={`Status for ${item.clause_ref}`}>
                          {Object.entries(STATUS).map(([key, cfg]) => (
                            <button
                              key={key}
                              disabled={saving === item.clause_ref}
                              onClick={() => key !== item.status && save(item, key, item.evidence ?? null)}
                              aria-pressed={item.status === key}
                              className={`rounded-md px-2.5 py-1 text-xs font-medium transition ${
                                item.status === key ? cfg.active : 'text-ink-500 hover:bg-ink-50'
                              }`}
                            >
                              {cfg.label}
                            </button>
                          ))}
                        </div>
                      </div>
                      {isOpen && (
                        <div className="no-print mt-3 pl-8">
                          <label className="mb-1 block text-xs font-medium text-ink-500">Evidence notes</label>
                          <textarea
                            value={draft}
                            onChange={(e) => setDrafts({ ...drafts, [item.clause_ref]: e.target.value })}
                            rows={3}
                            placeholder="Document references, owners, dates, links to records…"
                            className="w-full rounded-lg border border-ink-200 px-3 py-2 text-sm focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-100"
                          />
                          <div className="mt-2 flex justify-end">
                            <button
                              disabled={saving === item.clause_ref || draft === (item.evidence ?? '')}
                              onClick={() => save(item, item.status, draft || null)}
                              className="rounded-lg bg-ink-900 px-3 py-1.5 text-xs font-semibold text-white disabled:opacity-40"
                            >
                              {saving === item.clause_ref ? 'Saving…' : 'Save evidence'}
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </section>
          );
        })}
      </div>
    </div>
  );
}

function Stat({ value, label, cls }: { value: number; label: string; cls: string }) {
  return (
    <div>
      <p className={`text-2xl font-bold ${cls}`}>{value}</p>
      <p className="text-xs text-ink-500">{label}</p>
    </div>
  );
}
