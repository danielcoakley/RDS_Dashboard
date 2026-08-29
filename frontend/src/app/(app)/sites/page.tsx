'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { api } from '@/lib/api';
import { Plus, Building2, ArrowRight, X } from 'lucide-react';

export default function SitesPage() {
  const [sites, setSites] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState({ name: '', address: '' });
  const [creating, setCreating] = useState(false);

  const load = () => api.listSites().then(setSites).finally(() => setLoading(false));

  useEffect(() => { load(); }, []);

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    setCreating(true);
    try {
      await api.createSite({ name: form.name, address: form.address });
      setForm({ name: '', address: '' });
      setShowForm(false);
      await load();
    } catch (err: any) {
      alert(err.message);
    } finally {
      setCreating(false);
    }
  };

  if (loading) return <div className="text-ink-400">Loading...</div>;

  return (
    <div>
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-ink-900">Sites & Meters</h1>
          <p className="text-sm text-ink-500">Manage your sites, meters, and energy data</p>
        </div>
        <button onClick={() => setShowForm(true)}
          className="inline-flex items-center gap-2 rounded-lg bg-brand-600 px-4 py-2 text-sm font-semibold text-white hover:bg-brand-700">
          <Plus className="h-4 w-4" /> New Site
        </button>
      </div>

      {sites.length === 0 ? (
        <div className="rounded-xl border border-dashed border-ink-200 bg-white p-12 text-center">
          <Building2 className="mx-auto mb-3 h-10 w-10 text-ink-300" />
          <p className="text-ink-500">No sites yet. Create your first site to begin.</p>
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {sites.map((site) => (
            <Link key={site.id} href={`/sites/${site.id}`}
              className="group rounded-xl border border-ink-100 bg-white p-5 transition hover:border-brand-200 hover:shadow-lg">
              <h3 className="font-semibold text-ink-900 group-hover:text-brand-600">{site.name}</h3>
              <p className="mt-1 text-sm text-ink-500">{site.address || 'No address set'}</p>
              <div className="mt-3 flex items-center justify-between">
                <span className="text-sm text-ink-400">{site.meter_count} meters</span>
                <ArrowRight className="h-4 w-4 text-ink-300 group-hover:text-brand-600" />
              </div>
            </Link>
          ))}
        </div>
      )}

      {/* Create modal */}
      {showForm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
          <div className="w-full max-w-md rounded-2xl bg-white p-6 shadow-2xl">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-lg font-bold text-ink-900">Create Site</h2>
              <button onClick={() => setShowForm(false)}><X className="h-5 w-5 text-ink-400" /></button>
            </div>
            <form onSubmit={handleCreate} className="space-y-4">
              <div>
                <label className="mb-1.5 block text-sm font-medium text-ink-700">Site name</label>
                <input required value={form.name} onChange={(e) => setForm(f => ({ ...f, name: e.target.value }))}
                  className="w-full rounded-lg border border-ink-200 px-4 py-2.5 focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-100"
                  placeholder="RDS Main Campus" />
              </div>
              <div>
                <label className="mb-1.5 block text-sm font-medium text-ink-700">Address (auto-geocoded for weather)</label>
                <input value={form.address} onChange={(e) => setForm(f => ({ ...f, address: e.target.value }))}
                  className="w-full rounded-lg border border-ink-200 px-4 py-2.5 focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-100"
                  placeholder="Dublin, Ireland" />
                <p className="mt-1 text-xs text-ink-400">We'll geocode this to fetch weather data automatically.</p>
              </div>
              <button type="submit" disabled={creating}
                className="w-full rounded-lg bg-brand-600 px-4 py-2.5 font-semibold text-white hover:bg-brand-700 disabled:opacity-50">
                {creating ? 'Creating...' : 'Create Site'}
              </button>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
