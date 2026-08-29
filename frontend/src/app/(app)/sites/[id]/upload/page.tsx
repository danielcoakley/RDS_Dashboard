'use client';

import { useState, useRef } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { api } from '@/lib/api';
import { Upload, FileText, CheckCircle2, ArrowLeft, CloudSun, Tags } from 'lucide-react';

export default function UploadPage() {
  const params = useParams();
  const router = useRouter();
  const siteId = Number(params.id);
  const [file, setFile] = useState<File | null>(null);
  const [seuFile, setSeuFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [seuResult, setSeuResult] = useState<any>(null);
  const [error, setError] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const [seuDragOver, setSeuDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const seuInputRef = useRef<HTMLInputElement>(null);

  const handleFile = (f: File) => {
    setFile(f);
    setResult(null);
    setError('');
  };

  const handleSeuFile = (f: File) => {
    setSeuFile(f);
    setSeuResult(null);
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError('');
    try {
      const res = await api.uploadData(siteId, file);
      setResult(res);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  };

  const handleSeuUpload = async () => {
    if (!seuFile) return;
    setUploading(true);
    setError('');
    try {
      const res = await api.uploadSeuMapping(siteId, seuFile);
      setSeuResult(res);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div>
      <button onClick={() => router.back()} className="mb-4 inline-flex items-center gap-2 text-sm text-ink-500 hover:text-ink-900">
        <ArrowLeft className="h-4 w-4" /> Back
      </button>
      <h1 className="mb-2 text-2xl font-bold text-ink-900">Upload Energy Data</h1>
      <p className="mb-8 text-sm text-ink-500">Upload a CSV file with your energy consumption data. Weather data (HDD/CDD) is fetched automatically after upload.</p>

      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => { e.preventDefault(); setDragOver(false); if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]); }}
        onClick={() => inputRef.current?.click()}
        className={`cursor-pointer rounded-2xl border-2 border-dashed p-12 text-center transition ${
          dragOver ? 'border-brand-400 bg-brand-50' : 'border-ink-200 bg-white hover:border-brand-300'
        }`}
      >
        <input ref={inputRef} type="file" accept=".csv" className="hidden"
          onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])} />
        {file ? (
          <div className="flex flex-col items-center gap-2">
            <FileText className="h-10 w-10 text-brand-600" />
            <p className="font-medium text-ink-900">{file.name}</p>
            <p className="text-xs text-ink-400">{(file.size / 1024).toFixed(1)} KB</p>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-2">
            <Upload className="h-10 w-10 text-ink-300" />
            <p className="font-medium text-ink-700">Drop your CSV here or click to browse</p>
            <p className="text-xs text-ink-400">Supports both simple (date, meter, consumption) and original RDS formats</p>
          </div>
        )}
      </div>

      {error && <p className="mt-4 text-sm text-red-600">{error}</p>}

      {result && (
        <div className="mt-4 flex items-start gap-3 rounded-xl border border-brand-200 bg-brand-50 p-4">
          <CheckCircle2 className="mt-0.5 h-5 w-5 text-brand-600" />
          <div>
            <p className="font-medium text-brand-800">{result.message}</p>
            <p className="text-sm text-brand-600">{result.meters_created} new meters created</p>
            <div className="mt-2 flex items-center gap-1.5 text-sm text-brand-600">
              <CloudSun className="h-4 w-4" /> Weather data auto-fetched for the uploaded date range
            </div>
          </div>
        </div>
      )}

      {file && (
        <button onClick={handleUpload} disabled={uploading}
          className="mt-6 inline-flex items-center gap-2 rounded-lg bg-brand-600 px-6 py-3 font-semibold text-white hover:bg-brand-700 disabled:opacity-50">
          <Upload className="h-4 w-4" />
          {uploading ? 'Uploading...' : 'Upload & Process'}
        </button>
      )}

      {/* SEU Mapping Upload */}
      <div className="mt-10 border-t border-ink-100 pt-8">
        <h2 className="mb-2 text-xl font-bold text-ink-900">SEU Mapping (Optional)</h2>
        <p className="mb-6 text-sm text-ink-500">Upload a CSV mapping meter names to SEU categories for ISO 50001 §6.3 analysis. Use "Ignore" as the SEU_Category to exclude a meter.</p>

        <div
          onDragOver={(e) => { e.preventDefault(); setSeuDragOver(true); }}
          onDragLeave={() => setSeuDragOver(false)}
          onDrop={(e) => { e.preventDefault(); setSeuDragOver(false); if (e.dataTransfer.files[0]) handleSeuFile(e.dataTransfer.files[0]); }}
          onClick={() => seuInputRef.current?.click()}
          className={`cursor-pointer rounded-2xl border-2 border-dashed p-8 text-center transition ${
            seuDragOver ? 'border-brand-400 bg-brand-50' : 'border-ink-200 bg-white hover:border-brand-300'
          }`}
        >
          <input ref={seuInputRef} type="file" accept=".csv" className="hidden"
            onChange={(e) => e.target.files?.[0] && handleSeuFile(e.target.files[0])} />
          {seuFile ? (
            <div className="flex flex-col items-center gap-2">
              <Tags className="h-8 w-8 text-brand-600" />
              <p className="font-medium text-ink-900">{seuFile.name}</p>
              <p className="text-xs text-ink-400">{(seuFile.size / 1024).toFixed(1)} KB</p>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-2">
              <Tags className="h-8 w-8 text-ink-300" />
              <p className="font-medium text-ink-700">Drop SEU mapping CSV here or click to browse</p>
              <p className="text-xs text-ink-400">Columns: Meter, SEU_Category</p>
            </div>
          )}
        </div>

        {seuResult && (
          <div className="mt-4 flex items-start gap-3 rounded-xl border border-brand-200 bg-brand-50 p-4">
            <CheckCircle2 className="mt-0.5 h-5 w-5 text-brand-600" />
            <p className="font-medium text-brand-800">{seuResult.message}</p>
          </div>
        )}

        {seuFile && (
          <button onClick={handleSeuUpload} disabled={uploading}
            className="mt-4 inline-flex items-center gap-2 rounded-lg bg-brand-600 px-6 py-3 font-semibold text-white hover:bg-brand-700 disabled:opacity-50">
            <Tags className="h-4 w-4" />
            {uploading ? 'Uploading...' : 'Upload SEU Mapping'}
          </button>
        )}
      </div>

      {/* Format help */}
      <div className="mt-8 rounded-xl border border-ink-100 bg-ink-50 p-5">
        <h3 className="text-sm font-semibold text-ink-700">Supported CSV formats</h3>
        <div className="mt-3 space-y-3 text-xs text-ink-500">
          <div>
            <p className="font-medium text-ink-700">Simple format:</p>
            <code className="mt-1 block rounded bg-white px-3 py-2 text-ink-600">date,meter,consumption</code>
          </div>
          <div>
            <p className="font-medium text-ink-700">Original RDS format:</p>
            <p>Metered Sector, Utility, Units, Period, then date columns (DD/MM/YYYY). Each meter has a "Meter" row followed by a "Day" row with consumption values.</p>
          </div>
          <div>
            <p className="font-medium text-ink-700">SEU mapping format:</p>
            <code className="mt-1 block rounded bg-white px-3 py-2 text-ink-600">Meter,SEU_Category</code>
            <p className="mt-1">Use "Ignore" as SEU_Category to exclude a meter from analysis.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
