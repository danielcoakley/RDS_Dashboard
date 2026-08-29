'use client';

import React, { createContext, useContext, useState, useEffect, useCallback, useRef } from 'react';
import { api } from './api';
import { useAuth } from './auth';

export interface SiteOption {
  id: number;
  name: string;
  address?: string;
  meter_count?: number;
}

export type AnalysisStatus = 'idle' | 'loading' | 'ready' | 'error';

export interface AnalysisBundle {
  analysis: any;
  monthly: any;
  sankey: any;
  seuSummary: any;
  seuMonthly: any;
}

interface AnalysisCtx {
  sites: SiteOption[];
  sitesLoading: boolean;
  selectedSiteId: number | null;
  availableYears: number[];
  baselineYear: number | null;
  comparisonYear: number | null;
  bundle: AnalysisBundle | null;
  status: AnalysisStatus;
  error: string;
  selectSite: (id: number | null) => void;
  setBaselineYear: (y: number) => void;
  setComparisonYear: (y: number) => void;
  /** Re-run analysis for the current selection (e.g. after new data uploaded). */
  refresh: () => void;
  /** Mark that data changed for a site; if it's the active site, re-run. */
  invalidate: (siteId: number) => void;
}

const Ctx = createContext<AnalysisCtx>(null!);

const LS_KEY = 'enms_analysis_selection';

function loadSaved(): { siteId: number | null; baseline: number | null; comparison: number | null } {
  if (typeof window === 'undefined') return { siteId: null, baseline: null, comparison: null };
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return { siteId: null, baseline: null, comparison: null };
    const p = JSON.parse(raw);
    return {
      siteId: p.siteId ?? null,
      baseline: p.baseline ?? null,
      comparison: p.comparison ?? null,
    };
  } catch {
    return { siteId: null, baseline: null, comparison: null };
  }
}

export function AnalysisProvider({ children }: { children: React.ReactNode }) {
  const { user } = useAuth();
  const [sites, setSites] = useState<SiteOption[]>([]);
  const [sitesLoading, setSitesLoading] = useState(false);
  const [selectedSiteId, setSelectedSiteId] = useState<number | null>(null);
  const [availableYears, setAvailableYears] = useState<number[]>([]);
  const [baselineYear, setBaselineYearState] = useState<number | null>(null);
  const [comparisonYear, setComparisonYearState] = useState<number | null>(null);
  const [bundle, setBundle] = useState<AnalysisBundle | null>(null);
  const [status, setStatus] = useState<AnalysisStatus>('idle');
  const [error, setError] = useState('');
  const reqIdRef = useRef(0);

  // Load sites when user logs in
  useEffect(() => {
    if (!user) return;
    setSitesLoading(true);
    api.listSites().then((s) => {
      setSites(s);
      const saved = loadSaved();
      // Only restore saved selection if the site still exists
      if (saved.siteId && s.some((site: any) => site.id === saved.siteId)) {
        setSelectedSiteId(saved.siteId);
      }
    }).finally(() => setSitesLoading(false));
  }, [user]);

  // Persist selection
  useEffect(() => {
    if (typeof window === 'undefined') return;
    localStorage.setItem(LS_KEY, JSON.stringify({
      siteId: selectedSiteId, baseline: baselineYear, comparison: comparisonYear,
    }));
  }, [selectedSiteId, baselineYear, comparisonYear]);

  // When site changes, fetch available years and pick sensible defaults
  useEffect(() => {
    if (!selectedSiteId) {
      setAvailableYears([]);
      setBundle(null);
      setStatus('idle');
      return;
    }
    let cancelled = false;
    api.availableYears(selectedSiteId).then((res: any) => {
      if (cancelled) return;
      const yrs: number[] = res.years || [];
      setAvailableYears(yrs);
      const saved = loadSaved();
      if (saved.baseline && yrs.includes(saved.baseline) && saved.comparison && yrs.includes(saved.comparison)) {
        setBaselineYearState(saved.baseline);
        setComparisonYearState(saved.comparison);
      } else if (yrs.length >= 2) {
        setBaselineYearState(yrs[0]);
        setComparisonYearState(yrs[yrs.length - 1]);
      } else {
        setBaselineYearState(null);
        setComparisonYearState(null);
      }
    }).catch(() => {
      if (cancelled) return;
      setAvailableYears([]);
      setBaselineYearState(null);
      setComparisonYearState(null);
    });
    setBundle(null);
    return () => { cancelled = true; };
  }, [selectedSiteId]);

  // Run analysis automatically whenever site + years are all set
  useEffect(() => {
    if (!selectedSiteId || !baselineYear || !comparisonYear) {
      setBundle(null);
      setStatus(selectedSiteId ? 'idle' : 'idle');
      return;
    }
    if (baselineYear === comparisonYear) return;

    const myId = ++reqIdRef.current;
    setStatus('loading');
    setError('');
    api.analysisBundle(selectedSiteId, baselineYear, comparisonYear)
      .then((res: any) => {
        if (reqIdRef.current !== myId) return; // stale
        setBundle(res);
        setStatus('ready');
      })
      .catch((err: any) => {
        if (reqIdRef.current !== myId) return;
        setError(err.message || 'Analysis failed');
        setStatus('error');
      });
  }, [selectedSiteId, baselineYear, comparisonYear]);

  const selectSite = useCallback((id: number | null) => {
    setSelectedSiteId(id);
    if (id === null) {
      setBaselineYearState(null);
      setComparisonYearState(null);
    }
  }, []);

  const setBaselineYear = useCallback((y: number) => {
    setBaselineYearState(y);
  }, []);

  const setComparisonYear = useCallback((y: number) => {
    setComparisonYearState(y);
  }, []);

  const refresh = useCallback(() => {
    if (!selectedSiteId || !baselineYear || !comparisonYear) return;
    // Force re-run by bumping the effect dependency via state toggle
    const myId = ++reqIdRef.current;
    setStatus('loading');
    setError('');
    api.analysisBundle(selectedSiteId, baselineYear, comparisonYear)
      .then((res: any) => {
        if (reqIdRef.current !== myId) return;
        setBundle(res);
        setStatus('ready');
      })
      .catch((err: any) => {
        if (reqIdRef.current !== myId) return;
        setError(err.message || 'Analysis failed');
        setStatus('error');
      });
  }, [selectedSiteId, baselineYear, comparisonYear]);

  const invalidate = useCallback((siteId: number) => {
    // After data changes on a site: if it's the active site, re-fetch years + re-run
    if (siteId !== selectedSiteId) return;
    api.availableYears(siteId).then((res: any) => {
      const yrs: number[] = res.years || [];
      setAvailableYears(yrs);
      // Keep current years if still valid, else re-pick defaults
      if (!baselineYear || !yrs.includes(baselineYear) || !comparisonYear || !yrs.includes(comparisonYear)) {
        if (yrs.length >= 2) {
          setBaselineYearState(yrs[0]);
          setComparisonYearState(yrs[yrs.length - 1]);
        }
      } else {
        // years unchanged — explicitly re-run
        refresh();
      }
    });
  }, [selectedSiteId, baselineYear, comparisonYear, refresh]);

  return (
    <Ctx.Provider value={{
      sites, sitesLoading, selectedSiteId, availableYears,
      baselineYear, comparisonYear, bundle, status, error,
      selectSite, setBaselineYear, setComparisonYear, refresh, invalidate,
    }}>
      {children}
    </Ctx.Provider>
  );
}

export function useAnalysis() {
  return useContext(Ctx);
}
