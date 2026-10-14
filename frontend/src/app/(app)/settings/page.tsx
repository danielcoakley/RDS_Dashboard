'use client';

import { useState } from 'react';
import { Users, Building, ScrollText } from 'lucide-react';
import TeamTab from './TeamTab';
import OrganizationTab from './OrganizationTab';
import AuditLogTab from './AuditLogTab';

const tabs = [
  { id: 'team', label: 'Team', icon: Users },
  { id: 'organization', label: 'Organization', icon: Building },
  { id: 'audit', label: 'Audit log', icon: ScrollText },
] as const;

export default function SettingsPage() {
  const [tab, setTab] = useState<(typeof tabs)[number]['id']>('team');

  return (
    <div>
      <h1 className="text-2xl font-bold text-ink-900">Settings</h1>
      <p className="mt-1 text-sm text-ink-500">
        Team roles and authorities (ISO 50001 §5.3) and documented information (§7.5).
      </p>

      <div className="mt-6 flex gap-1 border-b border-ink-100">
        {tabs.map((t) => {
          const Icon = t.icon;
          return (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`-mb-px flex items-center gap-2 border-b-2 px-4 py-2.5 text-sm font-medium ${
                tab === t.id ? 'border-brand-600 text-brand-700' : 'border-transparent text-ink-500 hover:text-ink-900'
              }`}
            >
              <Icon className="h-4 w-4" /> {t.label}
            </button>
          );
        })}
      </div>

      <div className="mt-6">
        {tab === 'team' && <TeamTab />}
        {tab === 'organization' && <OrganizationTab />}
        {tab === 'audit' && <AuditLogTab />}
      </div>
    </div>
  );
}
