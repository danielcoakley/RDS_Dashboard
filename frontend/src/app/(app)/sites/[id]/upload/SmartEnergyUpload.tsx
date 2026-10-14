'use client';

import { useRef, useState } from 'react';
import { api, uploadWithProgress } from '@/lib/api';
import { Upload, FileText, Download, CheckCircle2, AlertTriangle, Loader2, CloudSun, X } from 'lucide-react';

type Mapping = { date_col: string; value_col: string; meter_col: string };

interface Props {
  siteId: number;
  onImported?: (result: any) => void;
}

const selectCls = 'w-full rounded-lg border border-ink-200 bg-white px-3 py-2 text-sm focus:border-brand-500 focus:outline-none';

export default function SmartEnergyUpload({ siteId, onImported }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [preview, setPreview] = useState<any>(null);
  const [mapping, setMapping] = useState<Mapping>({ date_col: '', value_col: '', meter_col: '' });
  const [meterName, setMeterName] = useState('Main Meter');
  const [analyzing, setAnalyzing] = useState(false);
  const [progress, setProgress] = useState<{ pct: number; stage: string } | null>(null);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState('');

  const runPreview = async (f: File, m?: Mapping, name = meterName) => {
    setAnalyzing(true);
    setError('');
    try {
      const fields: Record<string, string> = { meter_name: name };
      if (m) Object.assign(fields, m);
      const res: any = await api.previewUpload(siteId, f, fields);
      setPreview(res);
      if (res.mapping) {
        setMapping({ date_col: res.mapping.date_col || '', value_col: res.mapping.value_col || '', meter_col: res.mapping.meter_col || '' });
      }
    } catch (err: any) {
      setError(err.message);
      setPreview(null);
    } finally {
      setAnalyzing(false);
    }
  };

  const handleFile = (f: File) => {
    if (!/\.(csv|xlsx|xls)$/i.test(f.name)) {
      setError('Please choose a .csv or .xlsx file');
      return;
    }
    setFile(f);
    setResult(null);
    runPreview(f);
  };

  const updateMapping = (key: keyof Mapping, value: string) => {
    const next = { ...mapping, [key]: value };
    setMapping(next);
    if (file && next.date_col && next.value_col) runPreview(file, next);
  };

  const reset = () => {
    setFile(null); setPreview(null); setResult(null); setError(''); setProgress(null);
    if (inputRef.current) inputRef.current.value = '';
  };

  const handleImport = async () => {
    if (!file || !preview) return;
    setError('');
    setProgress({ pct: 0, stage: 'Uploading file' });
    const fd = new FormData();
    fd.append('file', file);
    let path = `/api/data/upload/${siteId}`;
    if (preview.format === 'tabular') {
      path = `/api/data/import/${siteId}`;
      fd.append('date_col', mapping.date_col);
      fd.append('value_col', mapping.value_col);
      if (mapping.meter_col) fd.append('meter_col', mapping.meter_col);
      fd.append('meter_name', meterName);
    }
    try {
      const res = await uploadWithProgress(path, fd, (pct) =>
        setProgress(pct < 100 ? { pct: pct * 0.6, stage: 'Uploading file' } : { pct: 60, stage: 'Importing readings & fetching weather' }));
      setProgress({ pct: 100, stage: 'Done' });
      setResult(res);
      onImported?.(res);
    } catch (err: any) {
      setError(err.message);
      setProgress(null);
    }
  };

  const q = preview?.quality;
  const importing = !!progress && progress.pct < 100;
  const mappedCols = [mapping.date_col, mapping.value_col, mapping.meter_col].filter(Boolean);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-sm text-ink-500">CSV or Excel (.xlsx). Date and value columns are detected automatically.</p>
        <a href={api.templateUrl} download className="inline-flex items-center gap-1.5 text-sm font-medium text-brand-600 hover:underline">
          <Download className="h-4 w-4" /> Download CSV template
        </a>
      </div>

      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => { e.preventDefault(); setDragOver(false); if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]); }}
        onClick={() => !file && inputRef.current?.click()}
        className={`rounded-2xl border-2 border-dashed p-10 text-center transition ${file ? '' : 'cursor-pointer'} ${
          dragOver ? 'border-brand-400 bg-brand-50' : 'border-ink-200 bg-white hover:border-brand-300'
        }`}
      >
        <input ref={inputRef} type="file" accept=".csv,.xlsx,.xls" className="hidden"
          onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])} />
        {file ? (
          <div className="flex items-center justify-center gap-3">
            <FileText className="h-8 w-8 text-brand-600" />
            <div className="text-left">
              <p className="font-medium text-ink-900">{file.name}</p>
              <p className="text-sm text-ink-400">{(file.size / 1024).toFixed(1)} KB</p>
            </div>
            {!importing && (
              <button onClick={(e) => { e.stopPropagation(); reset(); }} className="ml-2 rounded p-1 hover:bg-ink-100" aria-label="Remove file">
                <X className="h-4 w-4 text-ink-400" />
              </button>
            )}
          </div>
        ) : (
          <div className="flex flex-col items-center gap-2">
            <Upload className="h-10 w-10 text-ink-300" />
            <p className="font-medium text-ink-700">Drag & drop your CSV or XLSX here, or click to browse</p>
            <p className="text-xs text-ink-400">Long format (date, meter, value) or the original RDS export</p>
          </div>
        )}
      </div>

      {analyzing && (
        <p className="flex items-center gap-2 text-sm text-ink-500"><Loader2 className="h-4 w-4 animate-spin" /> Analyzing file…</p>
      )}
      {error && <p className="text-sm text-red-600">{error}</p>}

      {preview?.format === 'rds' && !result && (
        <div className="rounded-xl border border-blue-100 bg-blue-50 p-4 text-sm text-blue-800">
          Original RDS format detected ({preview.rows} rows). Meters and dates will be read automatically.
        </div>
      )}

      {/* Column mapping */}
      {preview?.format === 'tabular' && !result && (
        <div className="rounded-xl border border-ink-100 bg-white p-5">
          <h3 className="mb-3 font-semibold text-ink-900">Column mapping</h3>
          <div className="grid gap-4 sm:grid-cols-3">
            <label className="text-sm">
              <span className="mb-1 block font-medium text-ink-700">Date column</span>
              <select className={selectCls} value={mapping.date_col} onChange={(e) => updateMapping('date_col', e.target.value)}>
                <option value="">Select…</option>
                {preview.columns.map((c: string) => <option key={c} value={c}>{c}</option>)}
              </select>
            </label>
            <label className="text-sm">
              <span className="mb-1 block font-medium text-ink-700">Value column (kWh)</span>
              <select className={selectCls} value={mapping.value_col} onChange={(e) => updateMapping('value_col', e.target.value)}>
                <option value="">Select…</option>
                {preview.columns.map((c: string) => <option key={c} value={c}>{c}</option>)}
              </select>
            </label>
            <label className="text-sm">
              <span className="mb-1 block font-medium text-ink-700">Meter column</span>
              <select className={selectCls} value={mapping.meter_col} onChange={(e) => updateMapping('meter_col', e.target.value)}>
                <option value="">— Single meter —</option>
                {preview.columns.map((c: string) => <option key={c} value={c}>{c}</option>)}
              </select>
            </label>
          </div>
          {!mapping.meter_col && (
            <label className="mt-3 block text-sm sm:w-1/3">
              <span className="mb-1 block font-medium text-ink-700">Meter name</span>
              <input className={selectCls} value={meterName} onChange={(e) => setMeterName(e.target.value)}
                onBlur={() => file && mapping.date_col && mapping.value_col && runPreview(file, mapping)} />
            </label>
          )}

          <div className="mt-5 overflow-x-auto rounded-lg border border-ink-100">
            <table className="w-full text-sm">
              <thead className="bg-ink-50">
                <tr>
                  {preview.columns.map((c: string) => (
                    <th key={c} className={`whitespace-nowrap px-3 py-2 text-left font-medium ${mappedCols.includes(c) ? 'bg-brand-50 text-brand-700' : 'text-ink-500'}`}>
                      {c}
                      {c === mapping.date_col && <span className="ml-1 text-xs">(date)</span>}
                      {c === mapping.value_col && <span className="ml-1 text-xs">(value)</span>}
                      {c === mapping.meter_col && <span className="ml-1 text-xs">(meter)</span>}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-ink-50">
                {preview.preview.map((row: any, i: number) => (
                  <tr key={i}>
                    {preview.columns.map((c: string) => (
                      <td key={c} className={`whitespace-nowrap px-3 py-1.5 ${mappedCols.includes(c) ? 'text-ink-900' : 'text-ink-400'}`}>{row[c] ?? ''}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-2 text-xs text-ink-400">Showing first {preview.preview.length} rows</p>
        </div>
      )}

      {/* Data quality report */}
      {q && !result && (
        <div className="rounded-xl border border-ink-100 bg-white p-5">
          <div className="mb-3 flex items-center gap-2">
            {q.issue_count === 0
              ? <CheckCircle2 className="h-5 w-5 text-brand-600" />
              : <AlertTriangle className="h-5 w-5 text-amber-500" />}
            <h3 className="font-semibold text-ink-900">Data quality report</h3>
          </div>
          <p className="mb-4 text-sm text-ink-500">
            {q.valid_rows.toLocaleString()} valid rows of {q.total_rows.toLocaleString()}
            {q.unparseable_rows > 0 && ` · ${q.unparseable_rows} rows skipped (unreadable date/value)`}
          </p>
          <div className="space-y-3">
            {q.meters.map((m: any) => (
              <div key={m.meter} className="rounded-lg border border-ink-100 p-4">
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <p className="font-medium text-ink-900">{m.meter}</p>
                  <p className="text-xs text-ink-400">{m.start} → {m.end} · {m.rows} rows</p>
                </div>
                <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
                  <Issue label="Missing days" count={m.missing_days} detail={m.missing_sample.join(', ')} />
                  <Issue label="Duplicates" count={m.duplicates} detail={m.duplicate_sample.join(', ')} />
                  <Issue label="Negative values" count={m.negatives} detail={m.negative_sample.map((x: any) => `${x.date}: ${x.value}`).join(', ')} />
                  <Issue label="Outliers (z > 4)" count={m.outliers} detail={m.outlier_sample.map((x: any) => `${x.date}: ${x.value} (z=${x.z})`).join(', ')} />
                </div>
              </div>
            ))}
          </div>
          {q.issue_count > 0 && (
            <p className="mt-3 text-xs text-ink-400">Duplicates keep the last value per day. Other flagged rows are imported as-is.</p>
          )}
        </div>
      )}

      {/* Progress */}
      {progress && (
        <div>
          <div className="mb-1 flex justify-between text-sm">
            <span className="flex items-center gap-2 text-ink-600">
              {importing && <Loader2 className="h-4 w-4 animate-spin" />}{progress.stage}
            </span>
            <span className="text-ink-400">{Math.round(progress.pct)}%</span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-ink-100">
            <div className={`h-full rounded-full bg-brand-600 transition-all duration-500 ${progress.pct === 60 ? 'animate-pulse' : ''}`}
              style={{ width: `${progress.pct}%` }} />
          </div>
        </div>
      )}

      {result && (
        <div className="flex items-start gap-3 rounded-xl border border-brand-200 bg-brand-50 p-4">
          <CheckCircle2 className="mt-0.5 h-5 w-5 text-brand-600" />
          <div>
            <p className="font-medium text-brand-800">{result.message}</p>
            <p className="text-sm text-brand-600">{result.meters_created ?? 0} new meters created</p>
            <div className="mt-2 flex items-center gap-1.5 text-sm text-brand-600">
              <CloudSun className="h-4 w-4" /> Weather data auto-fetched for the uploaded date range
            </div>
          </div>
        </div>
      )}

      {preview && !result && (
        <button onClick={handleImport}
          disabled={importing || analyzing || (preview.format === 'tabular' && (!mapping.date_col || !mapping.value_col))}
          className="inline-flex items-center gap-2 rounded-lg bg-brand-600 px-6 py-3 font-semibold text-white hover:bg-brand-700 disabled:opacity-50">
          <Upload className="h-4 w-4" />
          {importing ? 'Importing…' : q ? `Import ${q.valid_rows.toLocaleString()} rows` : 'Import'}
        </button>
      )}
    </div>
  );
}

function Issue({ label, count, detail }: { label: string; count: number; detail: string }) {
  return (
    <div className={`rounded-md px-3 py-2 ${count ? 'bg-amber-50' : 'bg-ink-50'}`} title={detail || undefined}>
      <p className={`text-lg font-semibold ${count ? 'text-amber-700' : 'text-ink-400'}`}>{count}</p>
      <p className="text-xs text-ink-500">{label}</p>
      {count > 0 && detail && <p className="mt-1 truncate text-[11px] text-amber-700">{detail}</p>}
    </div>
  );
}
