'use client';

import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';
import { Zap } from 'lucide-react';
import { api } from '@/lib/api';
import { useAuth } from '@/lib/auth';

const input = 'w-full rounded-lg border border-ink-200 px-3 py-2.5 text-sm focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-100';

export default function AcceptInvitePage() {
  const { token } = useParams<{ token: string }>();
  const { startSession } = useAuth();
  const [invite, setInvite] = useState<any>(null);
  const [loadError, setLoadError] = useState('');
  const [name, setName] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    api.getInvite(token).then(setInvite).catch((e) => setLoadError(e.message));
  }, [token]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      startSession(await api.acceptInvite(token, { name, password }));
    } catch (err: any) {
      setError(err.message);
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-ink-50 px-4">
      <div className="w-full max-w-md rounded-2xl border border-ink-100 bg-white p-8 shadow-xl">
        <div className="mb-8 flex items-center gap-2">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand-600">
            <Zap className="h-5 w-5 text-white" />
          </div>
          <span className="text-xl font-bold">EnMS</span>
        </div>

        {loadError ? (
          <>
            <h1 className="text-2xl font-bold text-ink-900">Invite unavailable</h1>
            <p className="mt-2 text-sm text-red-600">{loadError}</p>
            <p className="mt-2 text-sm text-ink-500">Ask your organization owner for a new invite link.</p>
          </>
        ) : !invite ? (
          <p className="text-ink-400">Loading invite…</p>
        ) : (
          <>
            <h1 className="text-2xl font-bold text-ink-900">Join {invite.organization}</h1>
            <p className="mt-1 text-sm text-ink-500">
              You&apos;ve been invited as <span className="font-medium capitalize text-ink-700">{invite.role}</span>.
            </p>
            <form onSubmit={submit} className="mt-8 space-y-5">
              <div>
                <label className="mb-1.5 block text-sm font-medium text-ink-700">Email</label>
                <input value={invite.email} disabled className={`${input} bg-ink-50 text-ink-500`} />
              </div>
              <div>
                <label className="mb-1.5 block text-sm font-medium text-ink-700">Your name</label>
                <input required value={name} onChange={(e) => setName(e.target.value)} className={input} />
              </div>
              <div>
                <label className="mb-1.5 block text-sm font-medium text-ink-700">Password</label>
                <input type="password" required minLength={8} value={password}
                  onChange={(e) => setPassword(e.target.value)} className={input} placeholder="At least 8 characters" />
              </div>
              {error && <div className="rounded-lg bg-red-50 px-4 py-3 text-sm text-red-600">{error}</div>}
              <button disabled={loading} className="w-full rounded-lg bg-brand-600 py-2.5 text-sm font-semibold text-white hover:bg-brand-700 disabled:opacity-50">
                {loading ? 'Joining…' : 'Join organization'}
              </button>
            </form>
          </>
        )}
      </div>
    </div>
  );
}
