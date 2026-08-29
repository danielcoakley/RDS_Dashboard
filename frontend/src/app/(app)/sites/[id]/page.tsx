'use client';

import { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import Link from 'next/link';
import { api } from '@/lib/api';
import { useAnalysis } from '@/lib/analysis';
import { Plus, Upload, CloudSun, Trash2, ArrowLeft, Zap, FileText, Database, CalendarRange, CheckCircle2, Activity } from 'lucide-react';

const SEU_CATEGORIES = [
  'Boiler', 'AHU', 'Catering',
  'Lighting', 'AC / Refrigeration', 'Heater',
  'Electric Vehicles', 'Solar', 'Mains',
  'Other', 'Unknown',
];

export default function SiteDetailPage() {
  const params = useParams();
  const router = useRouter();
  const siteId = Number(params.id);
  const [site, setSite] = useState<any>(null);
  const [meters, setMeters] = useState<any[]>([]);
  const [weather, setWeather] = useState<any>(null);
  const [summary, setSummary] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [showMeterForm, setShowMeterForm] = useState(false);
  const [meterForm, setMeterForm] = useState({ name: '', utility_type: 'electricity', seu_category: 'Unknown' });
  const [fetchingWeather, setFetchingWeather] = useState(false);

  const { selectedSiteId, selectSite, invalidate } = useAnalysis();
  const isActiveSite = selectedSiteId === siteId;

  const load = async () => {
    try {
      const [s, m, w, sum] = await Promise.all([
        api.getSite(siteId),
        api.listMeters(siteId),
        api.weatherStatus(siteId).catch(() => null),
        api.dataSummary(siteId).catch(() => null),
      ]);
      setSite(s);
      setMeters(m);
      setWeather(w);
      setSummary(sum);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, [siteId]);

  const createMeter = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await api.createMeter(siteId, meterForm);
      setMeterForm({ name: '', utility_type: 'electricity', seu_category: 'Unknown' });
      setShowMeterForm(false);
      await load();
    } catch (err: any) {
      alert(err.message);
    }
  };

  const deleteMeter = async (meterId: number) => {
    if (!confirm('Delete this meter and all its readings?')) return;
    await api.deleteMeter(siteId, meterId);
    await load();
  };

  const refreshWeather = async () => {
    setFetchingWeather(true);
    try {
      await api.refreshWeather(siteId);
      await load();
    } catch (err: any) {
      alert(err.message);
    } finally {
      setFetchingWeather(false);
    }
  };

  if (loading) return <div className="text-ink-400">Loading...</div>;
  if (!site) return <div className="text-ink-400">Site not found</div>;

  return (
    <div>
      <Link href="/sites" className="mb-4 inline-flex items-center gap-2 text-sm text-ink-500 hover:text-ink-900">
        <ArrowLeft className="h-4 w-4" /> Sites
      </Link>
      <div className="mb-8 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-ink-900">{site.name}</h1>
          <p className="text-sm text-ink-500">{site.address || 'No address set'}</p>
          {site.latitude && site.longitude && (
            <p className="mt-1 text-xs text-ink-400">{site.latitude.toFixed(3)}, {site.longitude.toFixed(3)}</p>
          )}
        </div>
        <div className="flex gap-2">
          <Link href={`/sites/${siteId}/upload`}
            className="inline-flex items-center gap-2 rounded-lg border border-ink-200 px-4 py-2 text-sm font-semibold text-ink-700 hover:bg-ink-50">
            <Upload className="h-4 w-4" /> Upload Data
          </Link>
          <button onClick={() => setShowMeterForm(true)}
            className="inline-flex items-center gap-2 rounded-lg bg-brand-600 px-4 py-2 text-sm font-semibold text-white hover:bg-brand-700">
            <Plus className="h-4 w-4" /> Add Meter
          </button>
        </div>
      </div>

      {/* Weather card */}
      <div className="mb-6 rounded-xl border border-ink-100 bg-white p-5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <CloudSun className="h-6 w-6 text-brand-600" />
            <div>
              <h3 className="font-semibold text-ink-900">Weather Data</h3>
              <p className="text-sm text-ink-500">
                {weather?.status === 'available'
                  ? `${weather.total_days} days of HDD/CDD data (latest: ${weather.latest_date})`
                  : 'No weather data yet — fetch to enable climate-normalized analytics'}
              </p>
            </div>
          </div>
          <button onClick={refreshWeather} disabled={fetchingWeather || !site.latitude}
            className="rounded-lg border border-ink-200 px-4 py-2 text-sm font-medium text-ink-700 hover:bg-ink-50 disabled:opacity-50">
            {fetchingWeather ? 'Fetching...' : 'Fetch Weather'}
          </button>
        </div>
      </div>

      {/* Data coverage & uploaded files */}
      {summary && (
        <div className="mb-6 rounded-xl border border-ink-100 bg-white p-5">
          <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
            <h3 className="font-semibold text-ink-900">Data &amp; Analysis</h3>
            {isActiveSite ? (
              <span className="inline-flex items-center gap-1.5 rounded-full bg-brand-50 px-3 py-1 text-sm font-medium text-brand-700">
                <CheckCircle2 className="h-4 w-4" /> Active analysis site
              </span>
            ) : (
              <button onClick={() => selectSite(siteId)}
                className="inline-flex items-center gap-2 rounded-lg bg-brand-600 px-4 py-2 text-sm font-semibold text-white hover:bg-brand-700">
                <Activity className="h-4 w-4" /> Set as analysis site
              </button>
            )}
          </div>

          {/* Coverage stats */}
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Stat icon={Database} label="Meters" value={summary.meters} />
            <Stat icon={Zap} label="Readings" value={summary.total_readings?.toLocaleString()} />
            <Stat icon={CalendarRange} label="Data range"
              value={summary.date_start ? `${summary.date_start} → ${summary.date_end}` : '—'} small />
            <Stat icon={CloudSun} label="Weather days" value={summary.weather_days?.toLocaleString()} />
          </div>

          {/* Uploaded files */}
          <div className="mt-5">
            <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-ink-400">Uploaded files</p>
            {summary.uploads && summary.uploads.length > 0 ? (
              <ul className="divide-y divide-ink-50">
                {summary.uploads.map((u: any) => (
                  <li key={u.id} className="flex items-center gap-3 py-2">
                    <FileText className="h-4 w-4 shrink-0 text-ink-400" />
                    <span className="min-w-0 flex-1 truncate text-sm font-medium text-ink-900">{u.filename}</span>
                    <span className={`shrink-0 rounded-full px-2 py-0.5 text-xs font-medium ${
                      u.kind === 'energy' ? 'bg-blue-50 text-blue-700' : 'bg-purple-50 text-purple-700'
                    }`}>
                      {u.kind === 'energy' ? 'Energy data' : 'SEU map'}
                    </span>
                    <span className="hidden shrink-0 text-xs text-ink-400 sm:inline">{u.records?.toLocaleString()} records</span>
                    <span className="shrink-0 text-xs text-ink-400">{new Date(u.uploaded_at).toLocaleDateString()}</span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-ink-400">No files uploaded yet. Use “Upload Data” to add energy data and SEU mappings.</p>
            )}
          </div>
        </div>
      )}

      {/* Meters */}
      <h2 className="mb-4 text-lg font-semibold text-ink-900">Meters ({meters.length})</h2>
      {meters.length === 0 ? (
        <div className="rounded-xl border border-dashed border-ink-200 bg-white p-12 text-center">
          <Zap className="mx-auto mb-3 h-10 w-10 text-ink-300" />
          <p className="text-ink-500">No meters yet. Add meters or upload energy data to create them automatically.</p>
        </div>
      ) : (
        <div className="overflow-hidden rounded-xl border border-ink-100 bg-white">
          <table className="w-full">
            <thead className="border-b border-ink-100 bg-ink-50">
              <tr>
                <th className="px-4 py-3 text-left text-sm font-medium text-ink-600">Meter Name</th>
                <th className="px-4 py-3 text-left text-sm font-medium text-ink-600">Utility</th>
                <th className="px-4 py-3 text-left text-sm font-medium text-ink-600">SEU Category</th>
                <th className="px-4 py-3 text-left text-sm font-medium text-ink-600">Readings</th>
                <th className="px-4 py-3"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-ink-50">
              {meters.map((m) => (
                <tr key={m.id} className="hover:bg-ink-50">
                  <td className="px-4 py-3 text-sm font-medium text-ink-900">{m.name}</td>
                  <td className="px-4 py-3 text-sm text-ink-600">
                    <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ${m.utility_type === 'gas' ? 'bg-orange-50 text-orange-700' : 'bg-blue-50 text-blue-700'}`}>
                      {m.utility_type}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-sm text-ink-600">{m.seu_category}</td>
                  <td className="px-4 py-3 text-sm text-ink-600">{m.reading_count}</td>
                  <td className="px-4 py-3 text-right">
                    <button onClick={() => deleteMeter(m.id)} className="text-ink-400 hover:text-red-600">
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Meter modal */}
      {showMeterForm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4">
          <div className="w-full max-w-md rounded-2xl bg-white p-6 shadow-2xl">
            <h2 className="mb-4 text-lg font-bold text-ink-900">Add Meter</h2>
            <form onSubmit={createMeter} className="space-y-4">
              <div>
                <label className="mb-1.5 block text-sm font-medium text-ink-700">Meter name</label>
                <input required value={meterForm.name} onChange={(e) => setMeterForm(f => ({ ...f, name: e.target.value }))}
                  className="w-full rounded-lg border border-ink-200 px-4 py-2.5 focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-100"
                  placeholder="GM01 - Main Gas" />
              </div>
              <div>
                <label className="mb-1.5 block text-sm font-medium text-ink-700">Utility type</label>
                <select value={meterForm.utility_type} onChange={(e) => setMeterForm(f => ({ ...f, utility_type: e.target.value }))}
                  className="w-full rounded-lg border border-ink-200 px-4 py-2.5 focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-100">
                  <option value="electricity">Electricity</option>
                  <option value="gas">Gas</option>
                </select>
              </div>
              <div>
                <label className="mb-1.5 block text-sm font-medium text-ink-700">SEU Category</label>
                <select value={meterForm.seu_category} onChange={(e) => setMeterForm(f => ({ ...f, seu_category: e.target.value }))}
                  className="w-full rounded-lg border border-ink-200 px-4 py-2.5 focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-100">
                  {SEU_CATEGORIES.map(c => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
              <button type="submit" className="w-full rounded-lg bg-brand-600 px-4 py-2.5 font-semibold text-white hover:bg-brand-700">
                Add Meter
              </button>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}

function Stat({ icon: Icon, label, value, small }: { icon: any; label: string; value: any; small?: boolean }) {
  return (
    <div className="rounded-lg bg-ink-50 px-3 py-2.5">
      <div className="flex items-center gap-1.5 text-ink-400">
        <Icon className="h-3.5 w-3.5" />
        <span className="text-xs font-medium">{label}</span>
      </div>
      <p className={`mt-1 font-semibold text-ink-900 ${small ? 'text-xs' : 'text-lg'}`}>{value ?? '—'}</p>
    </div>
  );
}
