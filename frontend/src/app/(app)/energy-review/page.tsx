'use client';

import { useEffect, useState } from 'react';
import { api } from '@/lib/api';
import { useAnalysis } from '@/lib/analysis';
import { Workflow, Save, Trash2, Loader2, Eye, RefreshCw } from 'lucide-react';
import ReviewReport from './ReviewReport';

async function generateReview(siteId: number, baselineYear: number, reportingYear: number) {
  const token = localStorage.getItem('enms_token');
  const res = await fetch(
    `/api/energy-review/generate/${siteId}?baseline_year=${baselineYear}&reporting_year=${reportingYear}`,
    { headers: token ? { Authorization: `Bearer ${token}` } : {} },
  );
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || 'Failed to generate review');
  return res.json();
}

export default function EnergyReviewPage() {
  const { sites, selectedSiteId, baselineYear, comparisonYear } = useAnalysis();
  const [review, setReview] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [snapshots, setSnapshots] = useState<any[]>([]);
  const [viewing, setViewing] = useState<any>(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  const siteName = sites.find(s => s.id === selectedSiteId)?.name;
  const ready = selectedSiteId && baselineYear && comparisonYear && baselineYear !== comparisonYear;

  const loadSnapshots = () => {
    if (selectedSiteId) api.listReviews(selectedSiteId).then((r: any) => setSnapshots(r)).catch(() => setSnapshots([]));
  };

  const run = () => {
    if (!ready) return;
    setLoading(true); setError(''); setSaved(false); setViewing(null);
    generateReview(selectedSiteId!, baselineYear!, comparisonYear!)
      .then(setReview)
      .catch((e) => { setError(e.message); setReview(null); })
      .finally(() => setLoading(false));
  };

  useEffect(() => { setReview(null); setSnapshots([]); loadSnapshots(); run(); /* eslint-disable-next-line */ }, [selectedSiteId, baselineYear, comparisonYear]);

  const saveSnapshot = async () => {
    if (!review) return;
    setSaving(true);
    try {
      await api.createReview({
        site_id: selectedSiteId,
        review_data: { type: 'auto', generated_at: new Date().toISOString(), ...review },
        period_start: `${review.reporting_year}-01-01`,
        period_end: `${review.reporting_year}-12-31`,
      });
      setSaved(true);
      loadSnapshots();
    } catch (e: any) {
      alert(e.message);
    } finally {
      setSaving(false);
    }
  };

  const del = async (id: number) => {
    if (!confirm('Delete this review snapshot?')) return;
    await api.deleteReview(id);
    if (viewing?.id === id) setViewing(null);
    loadSnapshots();
  };

  const shown = viewing ? viewing.review_data : review;

  return (
    <div>
      <div className="mb-8 flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="mb-2 text-2xl font-bold text-ink-900">Energy Review</h1>
          <p className="text-sm text-ink-500">
            Auto-generated ISO 50001 §6.3 energy review with SEU identification, EnPIs (§6.4) and baseline comparison (§6.5).
            Select a site &amp; years in the sidebar.
          </p>
          {ready && (
            <p className="mt-2 text-sm font-medium text-ink-700">
              {siteName} · Baseline {baselineYear} → Reporting {comparisonYear}
              {viewing && <span className="ml-2 rounded-full bg-brand-50 px-2 py-0.5 text-xs text-brand-700">Snapshot from {viewing.created_at.split('T')[0]}</span>}
            </p>
          )}
        </div>
        {ready && (
          <div className="flex gap-2">
            {viewing ? (
              <button onClick={() => setViewing(null)}
                className="inline-flex items-center gap-2 rounded-lg border border-ink-200 bg-white px-4 py-2.5 text-sm font-semibold text-ink-700 hover:bg-ink-50">
                <RefreshCw className="h-4 w-4" /> Back to live review
              </button>
            ) : (
              <button onClick={saveSnapshot} disabled={!review || saving || saved}
                className="inline-flex items-center gap-2 rounded-lg bg-brand-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-brand-700 disabled:opacity-50">
                <Save className="h-4 w-4" /> {saving ? 'Saving...' : saved ? 'Snapshot saved' : 'Save review snapshot'}
              </button>
            )}
          </div>
        )}
      </div>

      {!ready && (
        <div className="rounded-xl border border-dashed border-ink-200 bg-white p-12 text-center">
          <Workflow className="mx-auto mb-3 h-10 w-10 text-ink-300" />
          <p className="text-ink-500">Select a site with at least two years of data, plus baseline and reporting years, in the sidebar.</p>
        </div>
      )}

      {ready && loading && !viewing && (
        <div className="flex items-center justify-center gap-2 rounded-xl border border-ink-100 bg-white p-12 text-ink-500">
          <Loader2 className="h-5 w-5 animate-spin" /> Generating energy review...
        </div>
      )}

      {ready && error && !viewing && <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">{error}</div>}

      {ready && shown && !(loading && !viewing) && (
        shown.seus?.length ? <ReviewReport review={shown} /> : (
          <div className="rounded-xl border border-dashed border-ink-200 bg-white p-12 text-center text-ink-500">
            No consumption data for {comparisonYear}.
          </div>
        )
      )}

      {selectedSiteId && snapshots.length > 0 && (
        <section className="mt-8 rounded-xl border border-ink-100 bg-white p-6">
          <h2 className="mb-4 text-lg font-semibold text-ink-900">Saved review snapshots</h2>
          <ul className="divide-y divide-ink-50">
            {snapshots.map((r) => (
              <li key={r.id} className="flex items-center justify-between py-3">
                <div>
                  <p className="font-medium text-ink-900">
                    {r.review_data?.type === 'auto'
                      ? `Baseline ${r.review_data.baseline_year} → Reporting ${r.review_data.reporting_year}`
                      : r.period_start ? `${r.period_start} – ${r.period_end || 'present'}` : 'Energy Review'}
                  </p>
                  <p className="text-xs text-ink-400">Saved {r.created_at.split('T')[0]}</p>
                </div>
                <div className="flex items-center gap-3">
                  {r.review_data?.type === 'auto' && (
                    <button onClick={() => { setViewing(r); window.scrollTo({ top: 0, behavior: 'smooth' }); }}
                      className="inline-flex items-center gap-1 text-sm font-medium text-brand-600 hover:text-brand-700">
                      <Eye className="h-4 w-4" /> View
                    </button>
                  )}
                  <button onClick={() => del(r.id)} className="text-ink-400 hover:text-red-600"><Trash2 className="h-4 w-4" /></button>
                </div>
              </li>
            ))}
          </ul>
        </section>
      )}
    </div>
  );
}
