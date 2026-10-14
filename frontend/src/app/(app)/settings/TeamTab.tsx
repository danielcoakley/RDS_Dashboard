'use client';

import { useCallback, useEffect, useState } from 'react';
import { Copy, Check, Trash2, UserPlus } from 'lucide-react';
import { api } from '@/lib/api';
import { useAuth } from '@/lib/auth';
import { ROLES, TEAM_ROLES } from '@/lib/roles';

const input = 'rounded-lg border border-ink-200 px-3 py-2 text-sm focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-100 disabled:bg-ink-50 disabled:text-ink-400';

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      onClick={() => { navigator.clipboard.writeText(text); setCopied(true); setTimeout(() => setCopied(false), 1500); }}
      className="inline-flex items-center gap-1 rounded-lg border border-ink-200 px-2.5 py-1.5 text-xs font-medium text-ink-700 hover:bg-ink-50"
    >
      {copied ? <Check className="h-3.5 w-3.5 text-brand-600" /> : <Copy className="h-3.5 w-3.5" />}
      {copied ? 'Copied' : 'Copy link'}
    </button>
  );
}

export default function TeamTab() {
  const { user, refreshUser } = useAuth();
  const isOwner = user?.role === 'owner';
  const [team, setTeam] = useState<{ members: any[]; invites: any[] }>({ members: [], invites: [] });
  const [error, setError] = useState('');
  const [form, setForm] = useState({ email: '', role: 'viewer', team_role: '' });
  const [busy, setBusy] = useState(false);

  const load = useCallback(() => {
    api.getTeam().then((t: any) => setTeam(t)).catch((e) => setError(e.message));
  }, []);
  useEffect(load, [load]);

  const run = async (fn: () => Promise<any>) => {
    setError('');
    try {
      await fn();
      load();
    } catch (e: any) {
      setError(e.message);
    }
  };

  const inviteLink = (token: string) => `${window.location.origin}/invite/${token}`;

  const submitInvite = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    await run(async () => {
      await api.createInvite({ ...form, team_role: form.team_role || null });
      setForm({ email: '', role: 'viewer', team_role: '' });
    });
    setBusy(false);
  };

  const updateMember = (id: number, data: any) =>
    run(async () => {
      await api.updateMember(id, data);
      if (id === user?.id) await refreshUser();
    });

  return (
    <div className="space-y-6">
      {error && <div className="rounded-lg bg-red-50 px-4 py-3 text-sm text-red-600">{error}</div>}

      {isOwner && (
        <form onSubmit={submitInvite} className="rounded-xl border border-ink-100 bg-white p-5">
          <h2 className="flex items-center gap-2 font-semibold text-ink-900"><UserPlus className="h-4 w-4" /> Invite a teammate</h2>
          <p className="mt-1 text-xs text-ink-500">Email sending isn&apos;t configured — copy the invite link and share it. Links expire after 7 days.</p>
          <div className="mt-4 flex flex-wrap gap-3">
            <input type="email" required placeholder="name@company.com" value={form.email}
              onChange={(e) => setForm({ ...form, email: e.target.value })} className={`${input} min-w-[240px] flex-1`} />
            <select value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value })} className={input}>
              {ROLES.map((r) => <option key={r.value} value={r.value}>{r.label}</option>)}
            </select>
            <select value={form.team_role} onChange={(e) => setForm({ ...form, team_role: e.target.value })} className={input}>
              <option value="">No EnMS team role</option>
              {TEAM_ROLES.map((r) => <option key={r.value} value={r.value}>{r.label}</option>)}
            </select>
            <button disabled={busy} className="rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white hover:bg-brand-700 disabled:opacity-50">
              Create invite
            </button>
          </div>
        </form>
      )}

      {isOwner && team.invites.length > 0 && (
        <div className="rounded-xl border border-ink-100 bg-white">
          <h2 className="border-b border-ink-100 px-5 py-3 font-semibold text-ink-900">Pending invites</h2>
          <ul className="divide-y divide-ink-100">
            {team.invites.map((i) => (
              <li key={i.id} className="flex flex-wrap items-center gap-3 px-5 py-3 text-sm">
                <span className="font-medium text-ink-900">{i.email}</span>
                <span className="rounded bg-ink-50 px-2 py-0.5 text-xs capitalize text-ink-600">{i.role}</span>
                <span className={`text-xs ${i.expired ? 'text-red-500' : 'text-ink-400'}`}>
                  {i.expired ? 'Expired' : `Expires ${new Date(i.expires_at + 'Z').toLocaleDateString()}`}
                </span>
                <div className="ml-auto flex items-center gap-2">
                  {!i.expired && (
                    <>
                      <input readOnly value={inviteLink(i.token)} className={`${input} w-72 text-xs`} onFocus={(e) => e.target.select()} />
                      <CopyButton text={inviteLink(i.token)} />
                    </>
                  )}
                  <button onClick={() => run(() => api.revokeInvite(i.id))} className="rounded-lg p-1.5 text-ink-400 hover:bg-red-50 hover:text-red-600" title="Revoke">
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="overflow-hidden rounded-xl border border-ink-100 bg-white">
        <table className="w-full text-sm">
          <thead className="bg-ink-50 text-left text-xs font-semibold uppercase tracking-wide text-ink-500">
            <tr>
              <th className="px-5 py-3">Member</th>
              <th className="px-5 py-3">Access role</th>
              <th className="px-5 py-3">EnMS team role (§5.3)</th>
              <th className="px-5 py-3" />
            </tr>
          </thead>
          <tbody className="divide-y divide-ink-100">
            {team.members.map((m) => {
              const self = m.id === user?.id;
              return (
                <tr key={m.id}>
                  <td className="px-5 py-3">
                    <p className="font-medium text-ink-900">{m.name}{self && <span className="ml-1 text-xs text-ink-400">(you)</span>}</p>
                    <p className="text-xs text-ink-400">{m.email}</p>
                  </td>
                  <td className="px-5 py-3">
                    <select value={m.role} disabled={!isOwner} onChange={(e) => updateMember(m.id, { role: e.target.value })} className={input}>
                      {ROLES.map((r) => <option key={r.value} value={r.value}>{r.label}</option>)}
                    </select>
                  </td>
                  <td className="px-5 py-3">
                    <select value={m.team_role ?? ''} disabled={!isOwner && !self}
                      onChange={(e) => updateMember(m.id, { team_role: e.target.value })} className={input}>
                      <option value="">—</option>
                      {TEAM_ROLES.map((r) => <option key={r.value} value={r.value}>{r.label}</option>)}
                    </select>
                  </td>
                  <td className="px-5 py-3 text-right">
                    {isOwner && !self && (
                      <button
                        onClick={() => confirm(`Remove ${m.name} from the organization?`) && run(() => api.removeMember(m.id))}
                        className="rounded-lg p-1.5 text-ink-400 hover:bg-red-50 hover:text-red-600" title="Remove member"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="grid gap-3 sm:grid-cols-3">
        {ROLES.map((r) => (
          <div key={r.value} className="rounded-lg border border-ink-100 bg-white px-4 py-3">
            <p className="text-sm font-semibold text-ink-900">{r.label}</p>
            <p className="text-xs text-ink-500">{r.help}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
