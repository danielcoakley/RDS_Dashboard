'use client';

import { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import { Plus, Target, Trash2, X, Calendar, TrendingUp, AlertCircle, CheckCircle2 } from 'lucide-react';

export default function ObjectivesPage() {
  const [objectives, setObjectives] = useState<any[]>([]);
  const [sites, setSites] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState({ title: '', description: '', target_pct: '', baseline_value: '', deadline: '', site_id: '' });
  const [saving, setSaving] = useState(false);

  const load = async () => {
    const [objs, s] = await Promise.all([api.listObjectives(), api.listSites()]);
    setObjectives(objs);
    setSites(s);
    setLoading(false);
  };

  useEffect(() => { load(); }, []);

  const create = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    try {
      await api.createObjective({
        title: form.title,
        description: form.description || null,
        target_pct: form.target_pct ? Number(form.target_pct) : null,
        baseline_value: form.baseline_value ? Number(form.baseline_value) : null,
        deadline: form.deadline || null,
        site_id: form.site_id ? Number(form.site_id) : null,
      });
      setForm({ title: '', description: '', target_pct: '', baseline_value: '', deadline: '', site_id: '' });
      setShowForm(false);
      await load();
    } catch (err: any) {
      alert(err.message);
    } finally {
      setSaving(false);
    }
  };

  const updateStatus = async (id: number, status: string) => {
    await api.updateObjective(id, { status });
    await load();
  };

  const del = async (id: number) => {
    if (!confirm('Delete this objective?')) return;
    await api.deleteObjective(id);
    await load();
  };

  if (loading) return <div className="text-ink-400">Loading...</div>;

  const statusConfig: any = {
    active: { icon: TrendingUp, color: 'blue', bg: 'bg-blue-50', text: 'text-blue-700', label: 'Active' },
    completed: { icon: CheckCircle2, color: 'brand', bg: 'bg-brand-50', text: 'text-brand-700', label: 'Completed' },
    at_risk: { icon: AlertCircle, color: 'red', bg: 'bg-red-50', text: 'text-red-700', label: 'At Risk' },
  };

  return (
    <div>
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-ink-900">Objectives & Targets</h1>
          <p className="text-sm text-ink-500">Set and track energy reduction goals (ISO 50001 §6.2)</p>
        </div>
        <button onClick={() => setShowForm(true)}
          className="inline-flex items-center gap-2 rounded-lg bg-brand-600 px-4 py-2 text-sm font-semibold text-white hover:bg-brand-700">
          <Plus className="h-4 w-4" /> New Objective
        </button>
      </div>

      {objectives.length === 0 ? (
        <div className="rounded-xl border border-dashed border-ink-200 bg-white p-12 text-center">
          <Target className="mx-auto mb-3 h-10 w-10 text-ink-300" />
          <p className="text-ink-500">No objectives yet. Create energy reduction targets to track your progress.</p>
        </div>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2">
          {objectives.map((obj) => {
            const sc = statusConfig[obj.status] || statusConfig.active;
            const Icon = sc.icon;
            return (
              <div key={obj.id} className="rounded-xl border border-ink-100 bg-white p-5">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <h3 className="font-semibold text-ink-900">{obj.title}</h3>
                    {obj.description && <p className="mt-1 text-sm text-ink-500">{obj.description}</p>}
                  </div>
                  <button onClick={() => del(obj.id)} className="ml-2 text-ink-400 hover:text-red-600"><Trash2 className="h-4 w-4" /></button>
                </div>
                <div className="mt-3 flex flex-wrap items-center gap-3 text-sm">
                  {obj.target_pct != null && (
                    <span className="rounded-full bg-ink-50 px-2.5 py-0.5 text-xs font-medium text-ink-600">Target: {obj.target_pct}%</span>
                  )}
                  {obj.baseline_value != null && (
                    <span className="rounded-full bg-ink-50 px-2.5 py-0.5 text-xs font-medium text-ink-600">Baseline: {obj.baseline_value}</span>
                  )}
                  {obj.deadline && (
                    <span className="inline-flex items-center gap-1 text-xs text-ink-400">
                      <Calendar className="h-3 w-3" /> {obj.deadline}
                    </span>
                  )}
                </div>
                <div className="mt-4 flex items-center gap-2">
                  <span className={`inline-flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium ${sc.bg} ${sc.text}`}>
                    <Icon className="h-3 w-3" /> {sc.label}
                  </span>
                  <select value={obj.status} onChange={(e) => updateStatus(obj.id, e.target.value)}
                    className="ml-auto rounded-lg border border-ink-200 px-2 py-1 text-xs text-ink-600">
                    <option value="active">Active</option>
                    <option value="at_risk">At Risk</option>
                    <option value="completed">Completed</option>
                  </select>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Create modal */}
      {showForm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
          <div className="w-full max-w-md rounded-2xl bg-white p-6 shadow-2xl">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-lg font-bold text-ink-900">New Objective</h2>
              <button onClick={() => setShowForm(false)}><X className="h-5 w-5 text-ink-400" /></button>
            </div>
            <form onSubmit={create} className="space-y-4">
              <div>
                <label className="mb-1.5 block text-sm font-medium text-ink-700">Title</label>
                <input required value={form.title} onChange={(e) => setForm(f => ({ ...f, title: e.target.value }))}
                  className="w-full rounded-lg border border-ink-200 px-4 py-2.5 focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-100"
                  placeholder="Reduce gas consumption by 10%" />
              </div>
              <div>
                <label className="mb-1.5 block text-sm font-medium text-ink-700">Description</label>
                <textarea value={form.description} onChange={(e) => setForm(f => ({ ...f, description: e.target.value }))}
                  rows={2} className="w-full rounded-lg border border-ink-200 px-4 py-2.5 focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-100"
                  placeholder="Details about this objective..." />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="mb-1.5 block text-sm font-medium text-ink-700">Target (%)</label>
                  <input type="number" value={form.target_pct} onChange={(e) => setForm(f => ({ ...f, target_pct: e.target.value }))}
                    className="w-full rounded-lg border border-ink-200 px-4 py-2.5" placeholder="-10" />
                </div>
                <div>
                  <label className="mb-1.5 block text-sm font-medium text-ink-700">Baseline value</label>
                  <input type="number" value={form.baseline_value} onChange={(e) => setForm(f => ({ ...f, baseline_value: e.target.value }))}
                    className="w-full rounded-lg border border-ink-200 px-4 py-2.5" placeholder="50000" />
                </div>
              </div>
              <div>
                <label className="mb-1.5 block text-sm font-medium text-ink-700">Deadline</label>
                <input type="date" value={form.deadline} onChange={(e) => setForm(f => ({ ...f, deadline: e.target.value }))}
                  className="w-full rounded-lg border border-ink-200 px-4 py-2.5" />
              </div>
              <div>
                <label className="mb-1.5 block text-sm font-medium text-ink-700">Site (optional)</label>
                <select value={form.site_id} onChange={(e) => setForm(f => ({ ...f, site_id: e.target.value }))}
                  className="w-full rounded-lg border border-ink-200 px-4 py-2.5">
                  <option value="">All sites</option>
                  {sites.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}
                </select>
              </div>
              <button type="submit" disabled={saving}
                className="w-full rounded-lg bg-brand-600 px-4 py-2.5 font-semibold text-white hover:bg-brand-700 disabled:opacity-50">
                {saving ? 'Creating...' : 'Create Objective'}
              </button>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
