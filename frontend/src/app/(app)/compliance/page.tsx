'use client';

import { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import { ShieldCheck, CheckCircle2, Clock, Circle } from 'lucide-react';

export default function CompliancePage() {
  const [items, setItems] = useState<any[]>([]);
  const [score, setScore] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [updating, setUpdating] = useState<string | null>(null);

  const load = async () => {
    const [items, sc] = await Promise.all([api.listCompliance(), api.complianceScore().catch(() => null)]);
    setItems(items);
    setScore(sc);
    setLoading(false);
  };

  useEffect(() => { load(); }, []);

  const updateStatus = async (clauseRef: string, status: string) => {
    setUpdating(clauseRef);
    try {
      await api.updateCompliance(clauseRef, { status, evidence: null });
      await load();
    } finally {
      setUpdating(null);
    }
  };

  if (loading) return <div className="text-ink-400">Loading...</div>;

  const statusConfig: any = {
    complete: { icon: CheckCircle2, bg: 'bg-brand-50', text: 'text-brand-700', label: 'Complete' },
    in_progress: { icon: Clock, bg: 'bg-amber-50', text: 'text-amber-700', label: 'In Progress' },
    not_started: { icon: Circle, bg: 'bg-ink-50', text: 'text-ink-500', label: 'Not Started' },
  };

  // Group by clause section
  const sections = ['4', '5', '6', '7', '8', '9', '10'];
  const sectionTitles: any = {
    '4': 'Context of the Organization',
    '5': 'Leadership',
    '6': 'Planning',
    '7': 'Support',
    '8': 'Operation',
    '9': 'Performance Evaluation',
    '10': 'Improvement',
  };

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-ink-900">ISO 50001 Compliance</h1>
        <p className="text-sm text-ink-500">Track clause-by-clause audit readiness across all ISO 50001:2018 requirements</p>
      </div>

      {/* Score card */}
      {score && (
        <div className="mb-8 rounded-2xl border border-ink-100 bg-white p-6">
          <div className="flex items-center justify-between">
            <div>
              <div className="flex items-center gap-3">
                <ShieldCheck className="h-8 w-8 text-brand-600" />
                <div>
                  <p className="text-3xl font-bold text-ink-900">{score.score}%</p>
                  <p className="text-sm text-ink-500">Compliance Score</p>
                </div>
              </div>
            </div>
            <div className="flex gap-6">
              <div className="text-center">
                <p className="text-2xl font-bold text-brand-600">{score.complete}</p>
                <p className="text-xs text-ink-500">Complete</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-amber-600">{score.in_progress}</p>
                <p className="text-xs text-ink-500">In Progress</p>
              </div>
              <div className="text-center">
                <p className="text-2xl font-bold text-ink-400">{score.not_started}</p>
                <p className="text-xs text-ink-500">Not Started</p>
              </div>
            </div>
          </div>
          {/* Progress bar */}
          <div className="mt-4 h-3 overflow-hidden rounded-full bg-ink-100">
            <div className="h-full rounded-full bg-brand-500 transition-all" style={{ width: `${score.score}%` }} />
          </div>
        </div>
      )}

      {/* Clause sections */}
      <div className="space-y-6">
        {sections.map((section) => {
          const sectionItems = items.filter(i => i.clause_ref.startsWith(section + '.') || i.clause_ref === section);
          if (sectionItems.length === 0) return null;
          return (
            <div key={section} className="overflow-hidden rounded-xl border border-ink-100 bg-white">
              <div className="border-b border-ink-100 bg-ink-50 px-5 py-3">
                <h2 className="font-semibold text-ink-900">§{section} — {sectionTitles[section]}</h2>
              </div>
              <div className="divide-y divide-ink-50">
                {sectionItems.map((item) => {
                  const sc = statusConfig[item.status] || statusConfig.not_started;
                  const Icon = sc.icon;
                  return (
                    <div key={item.clause_ref} className="flex items-center justify-between px-5 py-3 hover:bg-ink-50">
                      <div className="flex items-center gap-3">
                        <Icon className={`h-5 w-5 ${sc.text}`} />
                        <div>
                          <p className="text-sm font-medium text-ink-900">§{item.clause_ref} {item.clause_title}</p>
                          {item.evidence && <p className="text-xs text-ink-400">{item.evidence}</p>}
                        </div>
                      </div>
                      <select
                        value={item.status}
                        onChange={(e) => updateStatus(item.clause_ref, e.target.value)}
                        disabled={updating === item.clause_ref}
                        className={`rounded-full border-0 px-3 py-1 text-xs font-medium ${sc.bg} ${sc.text}`}
                      >
                        <option value="not_started">Not Started</option>
                        <option value="in_progress">In Progress</option>
                        <option value="complete">Complete</option>
                      </select>
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
