'use client';

import { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import { Workflow, Plus, Save, Trash2 } from 'lucide-react';

export default function EnergyReviewPage() {
  const [sites, setSites] = useState<any[]>([]);
  const [selectedSite, setSelectedSite] = useState<number | null>(null);
  const [reviews, setReviews] = useState<any[]>([]);
  const [editing, setEditing] = useState<any>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => { api.listSites().then(setSites); }, []);

  useEffect(() => {
    if (selectedSite) {
      api.listReviews(selectedSite).then(setReviews).catch(() => setReviews([]));
      setEditing(null);
    }
  }, [selectedSite]);

  const startNew = () => {
    setEditing({
      review_data: {
        energy_sources: '',
        significant_uses: '',
        current_performance: '',
        influencing_variables: '',
        consumption_patterns: '',
        opportunities: '',
      },
      period_start: '',
      period_end: '',
    });
  };

  const update = (key: string, val: string) => {
    setEditing((prev: any) => ({
      ...prev,
      review_data: { ...prev.review_data, [key]: val },
    }));
  };

  const save = async () => {
    setSaving(true);
    try {
      const payload = {
        site_id: selectedSite,
        review_data: editing.review_data,
        period_start: editing.period_start || null,
        period_end: editing.period_end || null,
      };
      if (editing.id) {
        await api.updateReview(editing.id, payload);
      } else {
        await api.createReview(payload);
      }
      await api.listReviews(selectedSite!).then(setReviews);
      setEditing(null);
    } catch (err: any) {
      alert(err.message);
    } finally {
      setSaving(false);
    }
  };

  const del = async (id: number) => {
    if (!confirm('Delete this energy review?')) return;
    await api.deleteReview(id);
    await api.listReviews(selectedSite!).then(setReviews);
  };

  return (
    <div>
      <h1 className="mb-2 text-2xl font-bold text-ink-900">Energy Review</h1>
      <p className="mb-8 text-sm text-ink-500">Document your ISO 50001 §6.3 energy review — identify energy sources, SEUs, and performance</p>

      <div className="mb-6 flex flex-wrap items-end gap-4">
        <div>
          <label className="mb-1.5 block text-sm font-medium text-ink-700">Site</label>
          <select value={selectedSite || ''} onChange={(e) => setSelectedSite(Number(e.target.value))}
            className="rounded-lg border border-ink-200 px-4 py-2.5 focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-100">
            <option value="">Select site...</option>
            {sites.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}
          </select>
        </div>
        {selectedSite && !editing && (
          <button onClick={startNew}
            className="inline-flex items-center gap-2 rounded-lg bg-brand-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-brand-700">
            <Plus className="h-4 w-4" /> New Energy Review
          </button>
        )}
      </div>

      {selectedSite && !editing && (
        <div className="space-y-4">
          {reviews.length === 0 ? (
            <div className="rounded-xl border border-dashed border-ink-200 bg-white p-12 text-center">
              <Workflow className="mx-auto mb-3 h-10 w-10 text-ink-300" />
              <p className="text-ink-500">No energy reviews yet. Create one to document your ISO 50001 §6.3 analysis.</p>
            </div>
          ) : (
            reviews.map((r) => (
              <div key={r.id} className="rounded-xl border border-ink-100 bg-white p-5">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="font-semibold text-ink-900">
                      {r.period_start ? `${r.period_start} – ${r.period_end || 'present'}` : 'Energy Review'}
                    </p>
                    <p className="text-xs text-ink-400">Created {r.created_at.split('T')[0]}</p>
                  </div>
                  <div className="flex gap-2">
                    <button onClick={() => setEditing({ ...r })} className="text-sm font-medium text-brand-600 hover:text-brand-700">Edit</button>
                    <button onClick={() => del(r.id)} className="text-ink-400 hover:text-red-600"><Trash2 className="h-4 w-4" /></button>
                  </div>
                </div>
                <div className="mt-4 grid gap-3 sm:grid-cols-2">
                  {Object.entries(r.review_data).map(([key, val]: any) => val && (
                    <div key={key} className="rounded-lg bg-ink-50 p-3">
                      <p className="text-xs font-medium uppercase text-ink-400">{key.replace(/_/g, ' ')}</p>
                      <p className="mt-1 text-sm text-ink-700">{val}</p>
                    </div>
                  ))}
                </div>
              </div>
            ))
          )}
        </div>
      )}

      {editing && selectedSite && (
        <div className="rounded-xl border border-ink-100 bg-white p-6">
          <h2 className="mb-4 text-lg font-semibold text-ink-900">{editing.id ? 'Edit Energy Review' : 'New Energy Review'}</h2>
          <div className="mb-4 grid gap-4 sm:grid-cols-2">
            <div>
              <label className="mb-1.5 block text-sm font-medium text-ink-700">Period start</label>
              <input type="date" value={editing.period_start || ''} onChange={(e) => setEditing({ ...editing, period_start: e.target.value })}
                className="w-full rounded-lg border border-ink-200 px-4 py-2.5" />
            </div>
            <div>
              <label className="mb-1.5 block text-sm font-medium text-ink-700">Period end</label>
              <input type="date" value={editing.period_end || ''} onChange={(e) => setEditing({ ...editing, period_end: e.target.value })}
                className="w-full rounded-lg border border-ink-200 px-4 py-2.5" />
            </div>
          </div>

          <div className="space-y-4">
            <ReviewField label="Energy Sources" value={editing.review_data.energy_sources || ''} onChange={(v) => update('energy_sources', v)}
              placeholder="Identify all energy sources (natural gas, electricity, etc.) and their proportions..." />
            <ReviewField label="Significant Energy Uses (SEUs)" value={editing.review_data.significant_uses || ''} onChange={(v) => update('significant_uses', v)}
              placeholder="List SEUs that account for major consumption (boilers, HVAC, lighting, etc.)..." />
            <ReviewField label="Current Energy Performance" value={editing.review_data.current_performance || ''} onChange={(v) => update('current_performance', v)}
              placeholder="Describe current energy performance levels and EnPIs..." />
            <ReviewField label="Influencing Variables" value={editing.review_data.influencing_variables || ''} onChange={(v) => update('influencing_variables', v)}
              placeholder="Weather (HDD/CDD), production volume, occupancy, operating hours..." />
            <ReviewField label="Consumption Patterns" value={editing.review_data.consumption_patterns || ''} onChange={(v) => update('consumption_patterns', v)}
              placeholder="Describe consumption patterns and trends observed..." />
            <ReviewField label="Improvement Opportunities" value={editing.review_data.opportunities || ''} onChange={(v) => update('opportunities', v)}
              placeholder="Identify potential energy saving opportunities and recommendations..." />
          </div>

          <div className="mt-6 flex gap-3">
            <button onClick={save} disabled={saving}
              className="inline-flex items-center gap-2 rounded-lg bg-brand-600 px-6 py-2.5 font-semibold text-white hover:bg-brand-700 disabled:opacity-50">
              <Save className="h-4 w-4" /> {saving ? 'Saving...' : 'Save Review'}
            </button>
            <button onClick={() => setEditing(null)}
              className="rounded-lg border border-ink-200 px-6 py-2.5 font-medium text-ink-700 hover:bg-ink-50">
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function ReviewField({ label, value, onChange, placeholder }: any) {
  return (
    <div>
      <label className="mb-1.5 block text-sm font-medium text-ink-700">{label}</label>
      <textarea value={value} onChange={(e) => onChange(e.target.value)} rows={3} placeholder={placeholder}
        className="w-full rounded-lg border border-ink-200 px-4 py-2.5 text-sm focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-100" />
    </div>
  );
}
