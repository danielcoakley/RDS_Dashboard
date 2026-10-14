'use client';

import { useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { Zap, Loader2, AlertCircle, CheckCircle2 } from 'lucide-react';
import { useAuth } from '@/lib/auth';

const STEPS = [
  'Creating an isolated demo organization',
  'Seeding a site, meters and two years of readings',
  'Loading weather data and running baselines',
  'Preparing objectives and ISO 50001 checklist',
];

export default function DemoPage() {
  const { demoLogin } = useAuth();
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);
  const [step, setStep] = useState(0);
  const started = useRef(false);

  useEffect(() => {
    // Guard against React strict-mode double invocation creating two demo orgs.
    if (started.current) return;
    started.current = true;

    const timer = setInterval(() => setStep((s) => Math.min(s + 1, STEPS.length - 1)), 1200);
    demoLogin()
      .then(() => router.replace('/dashboard'))
      .catch((e: Error) => setError(e.message || 'Could not create a demo workspace.'))
      .finally(() => clearInterval(timer));
  }, [demoLogin, router]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-brand-950 via-ink-950 to-ink-900 px-4">
      <div className="w-full max-w-md rounded-2xl border border-white/10 bg-white p-8 shadow-2xl">
        <div className="mb-6 flex items-center gap-2">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand-600">
            <Zap className="h-5 w-5 text-white" />
          </div>
          <div>
            <p className="font-bold leading-tight text-ink-900">EnMS</p>
            <p className="text-xs text-ink-400">ISO 50001 energy management</p>
          </div>
        </div>

        {error ? (
          <div>
            <div className="mb-4 flex items-start gap-3 rounded-lg bg-red-50 p-4 text-sm text-red-700">
              <AlertCircle className="mt-0.5 h-5 w-5 shrink-0" />
              <div>
                <p className="font-semibold">We couldn&apos;t start the demo</p>
                <p className="mt-1 text-red-600">{error}</p>
              </div>
            </div>
            <div className="flex flex-col gap-2 sm:flex-row">
              <button
                onClick={() => window.location.reload()}
                className="flex-1 rounded-lg border border-ink-200 px-4 py-2 text-sm font-semibold text-ink-700 hover:bg-ink-50"
              >
                Try again
              </button>
              <Link
                href="/signup"
                className="flex-1 rounded-lg bg-brand-600 px-4 py-2 text-center text-sm font-semibold text-white hover:bg-brand-700"
              >
                Start a free trial
              </Link>
            </div>
          </div>
        ) : (
          <div>
            <h1 className="text-xl font-bold text-ink-900">Preparing your demo workspace…</h1>
            <p className="mt-1 text-sm text-ink-500">This takes a few seconds. Your demo data is private to you.</p>
            <ul className="mt-6 space-y-3">
              {STEPS.map((label, i) => (
                <li key={label} className="flex items-center gap-3 text-sm">
                  {i < step ? (
                    <CheckCircle2 className="h-4 w-4 text-brand-600" />
                  ) : i === step ? (
                    <Loader2 className="h-4 w-4 animate-spin text-brand-600" />
                  ) : (
                    <span className="h-4 w-4 rounded-full border border-ink-200" />
                  )}
                  <span className={i <= step ? 'text-ink-800' : 'text-ink-400'}>{label}</span>
                </li>
              ))}
            </ul>
            <div className="mt-6 h-1.5 overflow-hidden rounded-full bg-ink-100">
              <div
                className="h-full rounded-full bg-brand-500 transition-all duration-700"
                style={{ width: `${((step + 1) / STEPS.length) * 90}%` }}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
