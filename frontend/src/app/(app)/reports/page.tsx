'use client';

import { useState } from 'react';
import { FileText, ClipboardList } from 'lucide-react';
import ManagementReport from './ManagementReport';
import AuditFindings from './AuditFindings';
import './report.css';

export default function ReportsPage() {
  const [tab, setTab] = useState<'review' | 'audit'>('review');
  const tabs = [
    { id: 'review' as const, label: 'Management Review (9.1, 9.3)', icon: FileText },
    { id: 'audit' as const, label: 'Internal Audit (9.2)', icon: ClipboardList },
  ];
  return (
    <div>
      <div className="no-print mb-6">
        <h1 className="text-2xl font-bold text-ink-900">Reports & Internal Audit</h1>
        <p className="text-sm text-ink-500">Board-ready ISO 50001 management review and internal audit findings log</p>
        <div className="mt-4 flex gap-2">
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium ${
                tab === t.id ? 'bg-brand-600 text-white' : 'bg-white text-ink-600 border border-ink-200 hover:bg-ink-50'
              }`}
            >
              <t.icon className="h-4 w-4" /> {t.label}
            </button>
          ))}
        </div>
      </div>
      {tab === 'review' ? <ManagementReport /> : <AuditFindings />}
    </div>
  );
}
