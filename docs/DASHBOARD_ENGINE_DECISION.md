# Dashboard Engine Decision

## Decision

Use **Next.js with React and TypeScript** for the SaaS frontend/dashboard shell, with **Apache ECharts** as the primary visualization engine for rich interactive analytics.

Streamlit remains legacy UI only. Future work may replace `app.py` entirely once the SaaS frontend covers the required workflows.

## Why This Fits

- Next.js provides a mature React application structure for landing pages, authenticated app pages, routing, layouts, and future server/client rendering choices.
- React gives flexibility for custom tenant workflows, reusable design components, and a richer product UX than a notebook-style dashboard.
- Apache ECharts supports a broad set of interactive chart types, good performance, responsive dashboards, and future portfolio-level visualizations.
- The stack pairs cleanly with the existing FastAPI backend boundary and keeps analytics execution in Python services.

## Initial Frontend Scope

- Public landing page oriented to ISO 50001 outcomes.
- Auth-ready app shell with tenant switcher, navigation, and user menu placeholders.
- Tenant dashboard views for sites, meters, uploads, runs, reports, and analytics summaries.
- Visualization components that consume backend API data and render tenant-scoped charts.

## Guardrails

- Do not couple frontend state to local files or Streamlit session state.
- Fetch SaaS data through API contracts, not by importing Python modules into the frontend.
- Keep route/page structure aligned with tenant boundaries.
- Build visualizations as replaceable components so chart libraries can evolve if needed.
