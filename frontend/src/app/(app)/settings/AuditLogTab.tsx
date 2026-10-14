'use client';

import { useEffect, useMemo, useState } from 'react';
import { Download } from 'lucide-react';
import { api } from '@/lib/api';

const input = 'rounded-lg border border-ink-200 px-3 py-2 text-sm focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-100';

function formatDetails(d: Record<string, any>) {
  return Object.entries(d || {})
    .map(([k, v]) => `${k}: ${typeof v === 'object' && v !== null ? JSON.stringify(v) : v}`)
    .join(' · ');
}

export default function AuditLogTab() {
  const [filters, setFilters] = useState({ action: '', user_id: '', start: '', end: '' });
  const [events, setEvents] = useState<any[]>([]);
  const [total, setTotal] = useState(0);
  const [actions, setActions] = useState<string[]>([]);
  const [members, setMembers] = useState<any[]>([]);
  const [error, setError] = useState('');

  const params = useMemo(
    () => Object.fromEntries(Object.entries(filters).filter(([, v]) => v)) as Record<string, string>,
    [filters],
  );

  useEffect(() => {
    api.auditActions().then((a: any) => setActions(a));
    api.getTeam().then((t: any) => setMembers(t.members));
  }, []);

  useEffect(() => {
    api.listAudit(params)
      .then((r: any) => { setEvents(r.events); setTotal(r.total); setError(''); })
      .catch((e) => setError(e.message));
  }, [params]);

  const exportCsv = async () => {
    const blob = await api.exportAudit(params);
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `audit-log-${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const set = (k: keyof typeof filters) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) =>
    setFilters({ ...filters, [k]: e.target.value });

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end gap-3">
        <div>
          <label className="mb-1 block text-xs font-medium text-ink-500">Action</label>
          <select value={filters.action} onChange={set('action')} className={input}>
            <option value="">All actions</option>
            {actions.map((a) => <option key={a} value={a}>{a}</option>)}
          </select>
        </div>
        <div>
          <label className="mb-1 block text-xs font-medium text-ink-500">User</label>
          <select value={filters.user_id} onChange={set('user_id')} className={input}>
            <option value="">All users</option>
            {members.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
          </select>
        </div>
        <div>
          <label className="mb-1 block text-xs font-medium text-ink-500">From</label>
          <input type="date" value={filters.start} onChange={set('start')} className={input} />
        </div>
        <div>
          <label className="mb-1 block text-xs font-medium text-ink-500">To</label>
          <input type="date" value={filters.end} onChange={set('end')} className={input} />
        </div>
        <button onClick={exportCsv} className="ml-auto inline-flex items-center gap-2 rounded-lg border border-ink-200 bg-white px-4 py-2 text-sm font-medium text-ink-700 hover:bg-ink-50">
          <Download className="h-4 w-4" /> Export CSV
        </button>
      </div>

      {error && <div className="rounded-lg bg-red-50 px-4 py-3 text-sm text-red-600">{error}</div>}

      <div className="overflow-x-auto rounded-xl border border-ink-100 bg-white">
        <table className="w-full text-sm">
          <thead className="bg-ink-50 text-left text-xs font-semibold uppercase tracking-wide text-ink-500">
            <tr>
              <th className="px-4 py-3">Time (UTC)</th>
              <th className="px-4 py-3">User</th>
              <th className="px-4 py-3">Action</th>
              <th className="px-4 py-3">Entity</th>
              <th className="px-4 py-3">Details</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-ink-100">
            {events.map((e) => (
              <tr key={e.id} className="align-top">
                <td className="whitespace-nowrap px-4 py-2.5 text-ink-500">{e.created_at.replace('T', ' ').slice(0, 19)}</td>
                <td className="px-4 py-2.5 text-ink-700">{e.user_email}</td>
                <td className="px-4 py-2.5"><span className="rounded bg-brand-50 px-2 py-0.5 font-mono text-xs text-brand-700">{e.action}</span></td>
                <td className="whitespace-nowrap px-4 py-2.5 text-ink-500">{e.entity_type ? `${e.entity_type} #${e.entity_id}` : '—'}</td>
                <td className="max-w-md break-words px-4 py-2.5 text-xs text-ink-500">{formatDetails(e.details)}</td>
              </tr>
            ))}
            {events.length === 0 && (
              <tr><td colSpan={5} className="px-4 py-8 text-center text-ink-400">No events match these filters.</td></tr>
            )}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-ink-400">Showing {events.length} of {total} events</p>
    </div>
  );
}
