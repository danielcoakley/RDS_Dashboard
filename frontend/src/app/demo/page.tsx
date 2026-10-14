'use client';

import { useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import { Loader2, Zap } from 'lucide-react';
import { useAuth } from '@/lib/auth';

export default function DemoPage() {
  const { demoLogin } = useAuth();
  const [error, setError] = useState('');
  const started = useRef(false);

  useEffect(() => {
    if (started.current) return; // avoid double org creation in React strict mode
    started.current = true;
    demoLogin().catch((e) => setError(e.message || 'Could not start the demo'));
  }, [demoLogin]);

  return (
    <div className="flex min-h-screen items-center justify-center bg-ink-50 px-6">
      <div className="w-full max-w-sm rounded-2xl border border-ink-100 bg-white p-8 text-center shadow-sm">
        <div className="mx-auto mb-5 flex h-12 w-12 items-center justify-center rounded-xl bg-brand-600">
          <Zap className="h-6 w-6 text-white" />
        </div>
        {error ? (
          <>
            <p className="font-semibold text-ink-900">Something went wrong</p>
            <p className="mt-1 text-sm text-red-600">{error}</p>
            <Link href="/" className="mt-5 inline-block text-sm font-medium text-brand-600">Back to home</Link>
          </>
        ) : (
          <>
            <Loader2 className="mx-auto mb-3 h-5 w-5 animate-spin text-brand-600" />
            <p className="font-semibold text-ink-900">Preparing your demo workspace...</p>
            <p className="mt-1 text-sm text-ink-500">Loading two years of sample data for Demo Manufacturing Ltd.</p>
          </>
        )}
      </div>
    </div>
  );
}
