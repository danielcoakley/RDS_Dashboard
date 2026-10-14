import { api, type ComplianceReadiness } from '@/lib/api';

export const SECTION_TITLES: Record<string, string> = {
  '4': 'Context of the organization',
  '5': 'Leadership',
  '6': 'Planning',
  '7': 'Support',
  '8': 'Operation',
  '9': 'Performance evaluation',
  '10': 'Improvement',
};

export const SECTIONS = Object.keys(SECTION_TITLES);

export interface ComplianceItem {
  clause_ref: string;
  clause_title: string;
  status: 'not_started' | 'in_progress' | 'complete' | string;
  evidence?: string | null;
}

export function sectionOf(clauseRef: string): string {
  return clauseRef.split('.')[0];
}

/** Partial credit for in-progress clauses, matching an auditor's "partially implemented" view. */
function pct(complete: number, inProgress: number, total: number): number {
  return total ? Math.round(((complete + inProgress * 0.5) / total) * 100) : 0;
}

/** Client-side readiness, used when the backend readiness endpoint is unavailable. */
export function computeReadiness(items: ComplianceItem[]): ComplianceReadiness {
  const count = (list: ComplianceItem[], s: string) => list.filter((i) => i.status === s).length;
  const complete = count(items, 'complete');
  const inProgress = count(items, 'in_progress');
  return {
    score: pct(complete, inProgress, items.length),
    complete,
    in_progress: inProgress,
    not_started: items.length - complete - inProgress,
    total: items.length,
    by_section: SECTIONS.map((section) => {
      const list = items.filter((i) => sectionOf(i.clause_ref) === section);
      const c = count(list, 'complete');
      return { section, title: SECTION_TITLES[section], score: pct(c, count(list, 'in_progress'), list.length), complete: c, total: list.length };
    }).filter((s) => s.total > 0),
    next_actions: items
      .filter((i) => i.status !== 'complete')
      .sort((a, b) => (a.status === 'in_progress' ? -1 : 0) - (b.status === 'in_progress' ? -1 : 0))
      .slice(0, 5)
      .map(({ clause_ref, clause_title, status }) => ({ clause_ref, clause_title, status })),
  };
}

/** Prefer the backend readiness endpoint; fall back to computing it from the checklist. */
export async function fetchReadiness(): Promise<ComplianceReadiness> {
  try {
    return await api.complianceReadiness();
  } catch {
    const items = (await api.listCompliance()) as ComplianceItem[];
    return computeReadiness(items);
  }
}
