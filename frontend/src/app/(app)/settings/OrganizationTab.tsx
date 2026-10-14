'use client';

import { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import { useAuth } from '@/lib/auth';

const input = 'w-full rounded-lg border border-ink-200 px-3 py-2 text-sm focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-100 disabled:bg-ink-50 disabled:text-ink-500';

export default function OrganizationTab() {
  const { user } = useAuth();
  const isOwner = user?.role === 'owner';
  const [org, setOrg] = useState({ name: '', sector: '', country: '' });
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);

  useEffect(() => {
    api.getOrganization().then((o: any) => setOrg({ name: o.name ?? '', sector: o.sector ?? '', country: o.country ?? '' }));
  }, []);

  const save = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await api.updateOrganization(org);
      setMsg({ ok: true, text: 'Organization saved' });
    } catch (err: any) {
      setMsg({ ok: false, text: err.message });
    }
  };

  const fields = [
    { key: 'name', label: 'Organization name' },
    { key: 'sector', label: 'Sector' },
    { key: 'country', label: 'Country' },
  ] as const;

  return (
    <form onSubmit={save} className="max-w-xl space-y-4 rounded-xl border border-ink-100 bg-white p-5">
      {fields.map((f) => (
        <div key={f.key}>
          <label className="mb-1.5 block text-sm font-medium text-ink-700">{f.label}</label>
          <input value={org[f.key]} disabled={!isOwner} required={f.key === 'name'}
            onChange={(e) => setOrg({ ...org, [f.key]: e.target.value })} className={input} />
        </div>
      ))}
      {msg && <p className={`text-sm ${msg.ok ? 'text-brand-700' : 'text-red-600'}`}>{msg.text}</p>}
      {isOwner ? (
        <button className="rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-700">Save</button>
      ) : (
        <p className="text-xs text-ink-400">Only owners can edit organization details.</p>
      )}
    </form>
  );
}
