'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useAuth } from '@/lib/auth';
import AuthShell from '@/components/marketing/AuthShell';

export default function SignupPage() {
  const { signup } = useAuth();
  const [form, setForm] = useState({
    name: '', email: '', password: '',
    orgName: '', sector: '', country: '',
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const update = (key: string, val: string) => setForm(f => ({ ...f, [key]: val }));

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      await signup({
        name: form.name,
        email: form.email,
        password: form.password,
        organization: { name: form.orgName, sector: form.sector, country: form.country },
      });
    } catch (err: any) {
      setError(err.message || 'Signup failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthShell title="Create your account" sub="Start managing your energy performance today" wide>

          <form onSubmit={handleSubmit} className="mt-8 space-y-5">
            <div className="rounded-xl bg-ink-50 p-4">
              <h3 className="mb-3 text-sm font-semibold text-ink-700">Organization</h3>
              <div className="space-y-4">
                <div>
                  <label className="mb-1.5 block text-sm font-medium text-ink-700">Organization name</label>
                  <input required value={form.orgName} onChange={(e) => update('orgName', e.target.value)}
                    className="w-full rounded-lg border border-ink-200 px-4 py-2.5 focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-100"
                    placeholder="Acme Manufacturing Ltd" />
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="mb-1.5 block text-sm font-medium text-ink-700">Sector</label>
                    <input value={form.sector} onChange={(e) => update('sector', e.target.value)}
                      className="w-full rounded-lg border border-ink-200 px-4 py-2.5 focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-100"
                      placeholder="Manufacturing" />
                  </div>
                  <div>
                    <label className="mb-1.5 block text-sm font-medium text-ink-700">Country</label>
                    <input value={form.country} onChange={(e) => update('country', e.target.value)}
                      className="w-full rounded-lg border border-ink-200 px-4 py-2.5 focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-100"
                      placeholder="United Kingdom" />
                  </div>
                </div>
              </div>
            </div>

            <h3 className="text-sm font-semibold text-ink-700">Your account</h3>
            <div>
              <label className="mb-1.5 block text-sm font-medium text-ink-700">Full name</label>
              <input required value={form.name} onChange={(e) => update('name', e.target.value)}
                className="w-full rounded-lg border border-ink-200 px-4 py-2.5 focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-100"
                placeholder="Jane Smith" />
            </div>
            <div>
              <label className="mb-1.5 block text-sm font-medium text-ink-700">Email</label>
              <input type="email" required value={form.email} onChange={(e) => update('email', e.target.value)}
                className="w-full rounded-lg border border-ink-200 px-4 py-2.5 focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-100"
                placeholder="you@company.com" />
            </div>
            <div>
              <label className="mb-1.5 block text-sm font-medium text-ink-700">Password</label>
              <input type="password" required minLength={6} value={form.password} onChange={(e) => update('password', e.target.value)}
                className="w-full rounded-lg border border-ink-200 px-4 py-2.5 focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-100"
                placeholder="At least 6 characters" />
            </div>

            {error && <p className="text-sm text-red-600">{error}</p>}
            <button type="submit" disabled={loading}
              className="w-full rounded-sm bg-ink-950 px-4 py-3 font-semibold text-white hover:bg-ink-900 disabled:opacity-50">
              {loading ? 'Creating account...' : 'Create account'}
            </button>
          </form>

          <p className="mt-6 text-center text-sm text-ink-500">
            Already have an account?{' '}
            <Link href="/login" className="font-medium text-brand-600 hover:text-brand-700">Sign in</Link>
          </p>
    </AuthShell>
  );
}
