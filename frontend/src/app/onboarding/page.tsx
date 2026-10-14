'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { api } from '@/lib/api';
import { useAuth } from '@/lib/auth';
import { SEU_CATEGORIES } from '@/lib/seu';
import SmartEnergyUpload from '@/app/(app)/sites/[id]/upload/SmartEnergyUpload';
import { Zap, Building2, MapPin, Gauge, Upload, Check, Loader2, Plus, Trash2, Sparkles, ArrowRight } from 'lucide-react';

const STEPS = [
  { label: 'Organization', icon: Building2 },
  { label: 'First site', icon: MapPin },
  { label: 'Meters', icon: Gauge },
  { label: 'Energy data', icon: Upload },
];

const SECTORS = ['Manufacturing', 'Commercial offices', 'Retail', 'Healthcare', 'Education', 'Hospitality', 'Public sector', 'Data centre', 'Other'];

const inputCls = 'w-full rounded-lg border border-ink-200 px-4 py-2.5 focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-100';
const primaryBtn = 'inline-flex items-center gap-2 rounded-lg bg-brand-600 px-5 py-2.5 font-semibold text-white hover:bg-brand-700 disabled:opacity-50';

export default function OnboardingPage() {
  const { user, loading } = useAuth();
  const router = useRouter();
  const [checking, setChecking] = useState(true);
  const [step, setStep] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  const [org, setOrg] = useState({ sector: '', country: '', energy_cost_per_kwh: '' });
  const [siteForm, setSiteForm] = useState({ name: '', address: '' });
  const [geo, setGeo] = useState<any>(null);
  const [site, setSite] = useState<any>(null);
  const [meters, setMeters] = useState<any[]>([]);
  const [meterForm, setMeterForm] = useState({ name: '', utility_type: 'electricity', seu_category: 'Unknown' });
  const [dataDone, setDataDone] = useState('');

  useEffect(() => {
    if (loading) return;
    if (!user) { router.push('/login'); return; }
    api.getOrgProfile().then((p: any) => {
      if (p.site_count > 0) { router.replace('/dashboard'); return; }
      setOrg({ sector: p.sector || '', country: p.country || '', energy_cost_per_kwh: p.energy_cost_per_kwh?.toString() || '' });
      setChecking(false);
    }).catch(() => setChecking(false));
  }, [user, loading, router]);

  const run = async (fn: () => Promise<void>) => {
    setBusy(true); setError('');
    try { await fn(); } catch (err: any) { setError(err.message); } finally { setBusy(false); }
  };

  const saveOrg = (e: React.FormEvent) => {
    e.preventDefault();
    run(async () => {
      await api.updateOrgProfile({
        sector: org.sector, country: org.country,
        energy_cost_per_kwh: org.energy_cost_per_kwh ? Number(org.energy_cost_per_kwh) : null,
      });
      if (!siteForm.address && org.country) setSiteForm((f) => ({ ...f, address: org.country }));
      setStep(1);
    });
  };

  const lookup = () => run(async () => { setGeo(null); setGeo(await api.geocode(siteForm.address)); });

  const createSite = (e: React.FormEvent) => {
    e.preventDefault();
    run(async () => {
      const s = site ?? await api.createSite({
        name: siteForm.name, address: siteForm.address || null,
        ...(geo && { latitude: geo.latitude, longitude: geo.longitude, timezone: geo.timezone }),
      });
      setSite(s);
      setStep(2);
    });
  };

  const addMeter = (e: React.FormEvent) => {
    e.preventDefault();
    run(async () => {
      const m = await api.createMeter(site.id, { ...meterForm, units: 'kWh' });
      setMeters((ms) => [...ms, m]);
      setMeterForm({ name: '', utility_type: 'electricity', seu_category: 'Unknown' });
    });
  };

  const removeMeter = (id: number) => run(async () => {
    await api.deleteMeter(site.id, id);
    setMeters((ms) => ms.filter((m) => m.id !== id));
  });

  const useSample = () => run(async () => {
    const res: any = await api.loadSampleData(site.id);
    setDataDone(res.message);
  });

  // Full load so the app shell picks up the new site
  const finish = () => { window.location.href = site ? `/sites/${site.id}` : '/dashboard'; };

  if (loading || checking) {
    return <div className="flex min-h-screen items-center justify-center text-ink-400">Loading...</div>;
  }

  return (
    <div className="min-h-screen bg-ink-50 px-4 py-10">
      <div className="mx-auto max-w-3xl">
        <div className="mb-8 flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-brand-600"><Zap className="h-4 w-4 text-white" /></div>
          <span className="font-bold text-ink-900">EnMS</span>
          <button onClick={() => router.push('/dashboard')} className="ml-auto text-sm text-ink-400 hover:text-ink-700">Skip setup</button>
        </div>

        <h1 className="text-2xl font-bold text-ink-900">Welcome{user?.name ? `, ${user.name.split(' ')[0]}` : ''} — let's set up your EnMS</h1>
        <p className="mt-1 text-sm text-ink-500">Four quick steps and you'll have a baseline-ready site.</p>

        {/* Stepper */}
        <ol className="my-8 grid grid-cols-4 gap-2">
          {STEPS.map((s, i) => (
            <li key={s.label} className="flex flex-col items-center gap-2 text-center">
              <div className={`flex h-10 w-10 items-center justify-center rounded-full border-2 ${
                i < step ? 'border-brand-600 bg-brand-600 text-white' : i === step ? 'border-brand-600 bg-white text-brand-600' : 'border-ink-200 bg-white text-ink-300'
              }`}>
                {i < step ? <Check className="h-5 w-5" /> : <s.icon className="h-5 w-5" />}
              </div>
              <span className={`text-xs font-medium ${i <= step ? 'text-ink-900' : 'text-ink-400'}`}>{i + 1}. {s.label}</span>
            </li>
          ))}
        </ol>

        <div className="rounded-2xl border border-ink-100 bg-white p-6 shadow-sm sm:p-8">
          {step === 0 && (
            <form onSubmit={saveOrg} className="space-y-4">
              <h2 className="text-lg font-bold text-ink-900">Organization profile</h2>
              <div>
                <label className="mb-1.5 block text-sm font-medium text-ink-700">Sector</label>
                <select required value={org.sector} onChange={(e) => setOrg({ ...org, sector: e.target.value })} className={inputCls}>
                  <option value="">Select a sector…</option>
                  {Array.from(new Set([...SECTORS, ...(org.sector ? [org.sector] : [])])).map((s) => <option key={s}>{s}</option>)}
                </select>
              </div>
              <div>
                <label className="mb-1.5 block text-sm font-medium text-ink-700">Country</label>
                <input required value={org.country} onChange={(e) => setOrg({ ...org, country: e.target.value })} className={inputCls} placeholder="Ireland" />
              </div>
              <div>
                <label className="mb-1.5 block text-sm font-medium text-ink-700">Energy cost per kWh</label>
                <input required type="number" step="0.001" min="0" value={org.energy_cost_per_kwh}
                  onChange={(e) => setOrg({ ...org, energy_cost_per_kwh: e.target.value })} className={inputCls} placeholder="0.25" />
                <p className="mt-1 text-xs text-ink-400">Used to express savings in money as well as kWh.</p>
              </div>
              <Footer busy={busy} error={error} label="Continue" />
            </form>
          )}

          {step === 1 && (
            <form onSubmit={createSite} className="space-y-4">
              <h2 className="text-lg font-bold text-ink-900">Create your first site</h2>
              <div>
                <label className="mb-1.5 block text-sm font-medium text-ink-700">Site name</label>
                <input required disabled={!!site} value={siteForm.name} onChange={(e) => setSiteForm({ ...siteForm, name: e.target.value })} className={inputCls} placeholder="Main Campus" />
              </div>
              <div>
                <label className="mb-1.5 block text-sm font-medium text-ink-700">Address</label>
                <div className="flex gap-2">
                  <input disabled={!!site} value={siteForm.address} onChange={(e) => { setSiteForm({ ...siteForm, address: e.target.value }); setGeo(null); }}
                    className={inputCls} placeholder="Dublin, Ireland" />
                  <button type="button" onClick={lookup} disabled={busy || !siteForm.address || !!site}
                    className="shrink-0 rounded-lg border border-ink-200 px-4 text-sm font-semibold text-ink-700 hover:bg-ink-50 disabled:opacity-50">
                    Locate
                  </button>
                </div>
                {geo && (
                  <p className="mt-2 flex items-center gap-1.5 text-sm text-brand-700">
                    <MapPin className="h-4 w-4" /> {geo.name}{geo.country ? `, ${geo.country}` : ''} — lat {geo.latitude?.toFixed(4)}, lon {geo.longitude?.toFixed(4)}
                  </p>
                )}
                <p className="mt-1 text-xs text-ink-400">Coordinates are used to fetch local weather (HDD/CDD) for normalized baselines.</p>
              </div>
              <Footer busy={busy} error={error} label={site ? 'Continue' : 'Create site'} onBack={() => setStep(0)} />
            </form>
          )}

          {step === 2 && (
            <div className="space-y-5">
              <h2 className="text-lg font-bold text-ink-900">Add meters to {site?.name}</h2>
              <form onSubmit={addMeter} className="grid gap-3 sm:grid-cols-[2fr_1fr_1.5fr_auto]">
                <input required value={meterForm.name} onChange={(e) => setMeterForm({ ...meterForm, name: e.target.value })} className={inputCls} placeholder="Main incomer" />
                <select value={meterForm.utility_type} onChange={(e) => setMeterForm({ ...meterForm, utility_type: e.target.value })} className={inputCls}>
                  <option value="electricity">Electricity</option>
                  <option value="gas">Gas</option>
                </select>
                <select value={meterForm.seu_category} onChange={(e) => setMeterForm({ ...meterForm, seu_category: e.target.value })} className={inputCls} aria-label="SEU category">
                  {SEU_CATEGORIES.map((c) => <option key={c}>{c}</option>)}
                </select>
                <button type="submit" disabled={busy} className={primaryBtn}><Plus className="h-4 w-4" /> Add</button>
              </form>
              {meters.length === 0 ? (
                <div className="rounded-xl border border-dashed border-ink-200 p-6 text-center text-sm text-ink-500">
                  No meters yet. Add at least one — or skip and let your upload or sample data create them.
                </div>
              ) : (
                <ul className="divide-y divide-ink-100 rounded-xl border border-ink-100">
                  {meters.map((m) => (
                    <li key={m.id} className="flex items-center gap-3 px-4 py-3 text-sm">
                      <Gauge className="h-4 w-4 text-ink-400" />
                      <span className="flex-1 font-medium text-ink-900">{m.name}</span>
                      <span className="rounded-full bg-ink-50 px-2 py-0.5 text-xs capitalize text-ink-600">{m.utility_type}</span>
                      <span className="text-xs text-ink-500">{m.seu_category}</span>
                      <button onClick={() => removeMeter(m.id)} aria-label="Remove meter"><Trash2 className="h-4 w-4 text-ink-300 hover:text-red-500" /></button>
                    </li>
                  ))}
                </ul>
              )}
              <Footer busy={busy} error={error} label={meters.length ? 'Continue' : 'Skip for now'} onBack={() => setStep(1)} onNext={() => setStep(3)} />
            </div>
          )}

          {step === 3 && (
            <div className="space-y-5">
              <h2 className="text-lg font-bold text-ink-900">Add energy data</h2>
              {dataDone ? (
                <div className="flex items-start gap-3 rounded-xl border border-brand-200 bg-brand-50 p-4">
                  <Check className="mt-0.5 h-5 w-5 text-brand-600" />
                  <p className="font-medium text-brand-800">{dataDone}</p>
                </div>
              ) : (
                <>
                  <div className="flex flex-wrap items-center gap-3 rounded-xl border border-brand-100 bg-brand-50/50 p-4">
                    <Sparkles className="h-5 w-5 text-brand-600" />
                    <p className="flex-1 text-sm text-ink-700">Just exploring? Load 2 years of realistic synthetic daily readings.</p>
                    <button onClick={useSample} disabled={busy} className={primaryBtn}>
                      {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />} Use sample data
                    </button>
                  </div>
                  <SmartEnergyUpload siteId={site.id} onImported={(r) => setDataDone(r.message)} />
                </>
              )}
              {error && <p className="text-sm text-red-600">{error}</p>}
              <div className="flex justify-between border-t border-ink-100 pt-5">
                <button onClick={() => setStep(2)} className="text-sm text-ink-500 hover:text-ink-900">Back</button>
                <button onClick={finish} className={primaryBtn}>
                  {dataDone ? 'Go to my site' : 'Finish without data'} <ArrowRight className="h-4 w-4" />
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function Footer({ busy, error, label, onBack, onNext }: { busy: boolean; error: string; label: string; onBack?: () => void; onNext?: () => void }) {
  return (
    <div className="border-t border-ink-100 pt-5">
      {error && <p className="mb-3 text-sm text-red-600">{error}</p>}
      <div className="flex items-center justify-between">
        {onBack ? <button type="button" onClick={onBack} className="text-sm text-ink-500 hover:text-ink-900">Back</button> : <span />}
        <button type={onNext ? 'button' : 'submit'} onClick={onNext} disabled={busy} className={primaryBtn}>
          {busy && <Loader2 className="h-4 w-4 animate-spin" />}{label} <ArrowRight className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}
