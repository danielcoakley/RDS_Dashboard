export const clauses = [
  { n: '4', title: 'Context of the organisation', feature: 'EnMS scope & boundaries', desc: 'Register sites, meters and boundaries once; scope statements stay in sync with your data.', evidence: 'Scope & boundaries register, interested-parties log', status: 'done' },
  { n: '5', title: 'Leadership', feature: 'Energy policy & roles', desc: 'Store the signed energy policy and assign the energy team with responsibilities.', evidence: 'Versioned energy policy, roles & responsibilities matrix', status: 'done' },
  { n: '6', title: 'Planning', feature: 'Baselines, SEUs & EnPIs', desc: 'Automatic HDD/CDD regression baselines, SEU identification and objectives tracking.', evidence: '6.3 energy review, 6.4 EnPIs, 6.5 EnB report, 6.2 objectives table (PDF)', status: 'done' },
  { n: '7', title: 'Support', feature: 'Documented information', desc: 'Every upload, model and decision is versioned and attributable — no lost spreadsheets.', evidence: 'Document control log, competence records', status: 'progress' },
  { n: '8', title: 'Operation', feature: 'Operational control', desc: 'Set operating criteria for each SEU and flag deviations automatically.', evidence: 'SEU operating criteria, deviation alerts', status: 'done' },
  { n: '9', title: 'Performance evaluation', feature: 'Monitoring & audit', desc: 'Weather-normalised EnPI tracking against baseline, internal audit checklists, management review packs.', evidence: 'EnPI trend reports, internal audit records, management review pack', status: 'progress' },
  { n: '10', title: 'Improvement', feature: 'Nonconformity & actions', desc: 'Log nonconformities, corrective actions and demonstrate continual improvement.', evidence: 'Corrective action register, improvement evidence', status: 'done' },
] as const;
