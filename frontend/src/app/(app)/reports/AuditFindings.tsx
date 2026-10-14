'use client';

import { useEffect, useState } from 'react';
import { Plus, Trash2 } from 'lucide-react';
import { api } from '@/lib/api';

export interface Finding {
  id: number;
  clause_ref: string;
  finding: string;
  severity: 'major' | 'minor' | 'observation';
  corrective_action: string | null;
  owner: string | null;
  due_date: string | null;
  status: 'open' | 'in_progress' | 'closed';
}

const empty = { clause_ref: '', finding: '', severity: 'minor', corrective_action: '', owner: '', due_date: '', status: 'open' };

export const severityStyle: Record<string, string> = {
  major: 'bg-red-100 text-red-700',
  minor: 'bg-amber-100 text-amber-700',
  observation: 'bg-sky-100 text-sky-700',
};

const input = 'w-full rounded-lg border border-ink-200 px-3 py-2 text-sm focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-100';

export default function AuditFindings() {
  const [items, setItems] = useState<Finding[]>([]);
  const [form, setForm] = useState<any>(empty);
  const [error, setError] = useState('');
  const [saving, setSaving] = useState(false);

  const load = () => api.listFindings().then((d: any) => setItems(d)).catch((e) => setError(e.message));
  useEffect(() => { load(); }, []);

  const toPayload = (f: any) => ({ ...f, due_date: f.due_date || null, corrective_action: f.corrective_action || null, owner: f.owner || null });

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setError('');
    try {
      await api.createFinding(toPayload(form));
      setForm(empty);
      await load();
    } catch (err: any) {
      setError(err.message);
    } finally {
      setSaving(false);
    }
  };

  const setStatus = async (f: Finding, status: string) => {
    await api.updateFinding(f.id, toPayload({ ...f, status }));
    load();
  };

  const remove = async (f: Finding) => {
    if (!confirm('Delete this finding?')) return;
    await api.deleteFinding(f.id);
    load();
  };

  return (
    <div className="space-y-6">
      <form onSubmit={submit} className="rounded-xl border border-ink-100 bg-white p-5">
        <h2 className="mb-4 font-semibold text-ink-900">Log audit finding</h2>
        <div className="grid gap-3 md:grid-cols-4">
          <input className={input} placeholder="Clause (e.g. 6.3)" required value={form.clause_ref} onChange={(e) => setForm({ ...form, clause_ref: e.target.value })} />
          <select className={input} value={form.severity} onChange={(e) => setForm({ ...form, severity: e.target.value })}>
            <option value="major">Major nonconformity</option>
            <option value="minor">Minor nonconformity</option>
            <option value="observation">Observation</option>
          </select>
          <input className={input} placeholder="Owner" value={form.owner} onChange={(e) => setForm({ ...form, owner: e.target.value })} />
          <input className={input} type="date" value={form.due_date} onChange={(e) => setForm({ ...form, due_date: e.target.value })} />
          <textarea className={`${input} md:col-span-2`} rows={2} placeholder="Finding" required value={form.finding} onChange={(e) => setForm({ ...form, finding: e.target.value })} />
          <textarea className={`${input} md:col-span-2`} rows={2} placeholder="Corrective action" value={form.corrective_action} onChange={(e) => setForm({ ...form, corrective_action: e.target.value })} />
        </div>
        <div className="mt-3 flex items-center gap-3">
          <button disabled={saving} className="flex items-center gap-2 rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-700 disabled:opacity-50">
            <Plus className="h-4 w-4" /> Add finding
          </button>
          {error && <span className="text-sm text-red-600">{error}</span>}
        </div>
      </form>

      <div className="overflow-x-auto rounded-xl border border-ink-100 bg-white">
        <table className="w-full text-sm">
          <thead className="bg-ink-50 text-left text-xs uppercase text-ink-500">
            <tr>
              {['Clause', 'Finding', 'Severity', 'Corrective action', 'Owner', 'Due', 'Status', ''].map((h) => <th key={h} className="px-4 py-3">{h}</th>)}
            </tr>
          </thead>
          <tbody className="divide-y divide-ink-100">
            {items.length === 0 && <tr><td colSpan={8} className="px-4 py-8 text-center text-ink-400">No audit findings logged yet.</td></tr>}
            {items.map((f) => (
              <tr key={f.id} className="align-top">
                <td className="px-4 py-3 font-medium">§{f.clause_ref}</td>
                <td className="px-4 py-3">{f.finding}</td>
                <td className="px-4 py-3"><span className={`rounded-full px-2 py-0.5 text-xs font-medium capitalize ${severityStyle[f.severity]}`}>{f.severity}</span></td>
                <td className="px-4 py-3 text-ink-600">{f.corrective_action || '—'}</td>
                <td className="px-4 py-3">{f.owner || '—'}</td>
                <td className="whitespace-nowrap px-4 py-3">{f.due_date || '—'}</td>
                <td className="px-4 py-3">
                  <select value={f.status} onChange={(e) => setStatus(f, e.target.value)} className="rounded border border-ink-200 px-2 py-1 text-xs">
                    <option value="open">Open</option>
                    <option value="in_progress">In progress</option>
                    <option value="closed">Closed</option>
                  </select>
                </td>
                <td className="px-4 py-3">
                  <button onClick={() => remove(f)} aria-label="Delete finding" className="text-ink-400 hover:text-red-600"><Trash2 className="h-4 w-4" /></button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
