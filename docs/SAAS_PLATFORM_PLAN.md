# SaaS Expansion Plan for ISO 50001 Energy Analytics Platform

## 1) Current-State Review

### Architecture and product shape today
- The app is currently a **single-tenant Streamlit dashboard** with all core UI and workflow in one entry-point (`app.py`).
- The app allows users to upload CSVs for energy, HDD, CDD, and SEU mapping, then performs baseline and comparison analysis.
- Client selection is file-based from `config/clients/*.yaml`, which is a good foundation for future tenant templates.

### Strengths to preserve
- Existing domain logic for ISO 50001 support is already modularized in `src/` (`config_loader`, `data_standardisation`, `data_quality`, `reporting`, `enpi_model`).
- Existing tests cover config and standardization modules.
- Data processing and EnPI-style reporting already exist and can be wrapped into reusable tenant services.

### Gaps blocking SaaS readiness
- No authentication, authorization, or account model.
- No tenant isolation model (data/files currently handled in-process and partly local-file based).
- No persistent backend for users, organizations, assets, uploads, runs, and reports.
- No marketing/landing experience (product currently opens directly into internal analytics UI).
- No billing/subscription model, onboarding flow, or role-based access control.
- No production deployment architecture for multi-client scale, auditability, and security.

---

## 2) Target SaaS Scope (MVP then Growth)

## MVP (Phase 1)
- Public landing page + product marketing pages.
- User signup/login/logout/password reset.
- Organization (tenant) creation and invite flow.
- Basic RBAC roles: Owner, Manager, Viewer.
- Per-tenant data upload storage and run history.
- Reuse current analytics pipeline for tenant-specific project runs.
- Basic admin views: user management and tenant profile.

## Phase 2
- Report scheduling, exports, and collaboration comments.
- Improved data connectors (beyond CSV: API/SFTP/utility integrations).
- Auditable compliance trail for ISO 50001 evidence packages.
- Subscription billing + trial management.

## Phase 3
- Benchmarking, portfolio dashboards, and advanced M&V workflows.
- White-label options and enterprise SSO/SCIM.

---

## 3) Proposed Technical Architecture

### Frontend
- Keep Streamlit for internal analytics workflows short-term.
- Add a dedicated **web frontend** (e.g., Next.js) for landing page + auth flows + tenant/admin UX.
- Optionally embed or link Streamlit analytics until migration to a unified web app is complete.

### Backend/API
- Introduce an API layer (FastAPI recommended) to own:
  - Auth/session/token handling
  - Tenant/user/project/report CRUD
  - Orchestration of analysis runs
  - Audit events and activity logs

### Data layer
- PostgreSQL for users/tenants/projects/runs/metadata.
- Object storage (S3-compatible) for raw uploads and generated reports.
- Background job queue (Celery/RQ/Arq) for longer analytics tasks.

### Security baseline
- Per-tenant access checks at API/service boundary.
- Secrets management and encrypted storage.
- Comprehensive audit logging for upload, config changes, run execution, and report access.

---

## 4) Issue Backlog (Suggested Epics and Stories)

## Epic A — Product Foundation & Discovery
1. Define ICP, packaging, and tier boundaries.
2. Finalize MVP KPI targets (activation, first-report time, retention).
3. Produce SaaS information architecture and user journeys.

## Epic B — Identity, Access, Tenancy
1. Implement auth provider integration (Clerk/Auth0/Supabase/Auth.js).
2. Create user and organization data models.
3. Implement invite + role assignment flows.
4. Enforce RBAC and tenant scoping on all resources.

## Epic C — Platform Backend
1. Scaffold API service and environment configuration.
2. Implement projects/uploads/runs/report endpoints.
3. Add background job execution for heavy analyses.
4. Add observability (structured logs, tracing, metrics).

## Epic D — Analytics Service Refactor
1. Extract current analytics functions into service layer callable by API jobs.
2. Remove local-file coupling (`seu_mapping.csv`) in favor of tenant-scoped storage.
3. Standardize data contract between upload ingestion and analytics pipeline.
4. Expand tests from unit to integration for full run lifecycle.

## Epic E — SaaS Web UX
1. Build landing page and feature pages.
2. Build signup/login/reset pages.
3. Build app shell (tenant switcher, nav, user menu).
4. Build upload/run/report history views.

## Epic F — Compliance, Governance, and Billing
1. Add audit log UI and export.
2. Add consent/privacy controls and retention policies.
3. Implement subscription tiers, checkout, and dunning flows.

---

## 5) Milestone Plan (Recommended)

## Milestone 0 (1–2 weeks): Alignment & Foundations
- Deliverables:
  - Product requirements doc for SaaS MVP.
  - Canonical domain model (User, Organization, Site, Meter, Upload, Run, Report).
  - Target architecture decision record.
- Exit criteria:
  - Team sign-off on MVP scope and implementation stack.

## Milestone 1 (2–4 weeks): Auth + Tenant Skeleton
- Deliverables:
  - Functional signup/login and organization creation.
  - RBAC seed roles and tenant-scoped database schema.
  - Basic dashboard shell after login.
- Exit criteria:
  - Two test tenants can log in and cannot see each other’s data.

## Milestone 2 (3–5 weeks): Ingestion + Run Orchestration
- Deliverables:
  - Tenant-scoped uploads persisted to object storage.
  - API-triggered analytics runs with status lifecycle.
  - Initial report artifacts stored and retrievable.
- Exit criteria:
  - End-to-end flow from upload to result works for at least one tenant.

## Milestone 3 (3–5 weeks): SaaS UX + Landing Experience
- Deliverables:
  - Public landing page and conversion CTAs.
  - In-app pages for uploads, runs, reports, settings.
  - Email notifications for run completion.
- Exit criteria:
  - Pilot users can self-serve core workflow without operator intervention.

## Milestone 4 (2–4 weeks): Hardening + Pilot Readiness
- Deliverables:
  - Audit logs, backups, monitoring, alerting.
  - Security review and threat model outcomes.
  - Pilot onboarding toolkit and support playbook.
- Exit criteria:
  - Platform ready for controlled multi-client pilot.

---

## 6) Immediate Next Actions (next 10 working days)
1. Split current repo into clear boundaries: `frontend/`, `backend/`, `analytics/` (or equivalent).
2. Create initial DB schema and migration setup.
3. Implement auth and organization models.
4. Refactor file I/O paths so no shared local mutable files are needed.
5. Add integration tests for tenant isolation and run lifecycle.
6. Draft landing page copy oriented to ISO 50001 pain points and outcomes.

---

## 7) Risks and Mitigations
- Risk: analytics pipeline assumptions tied to one client format.
  - Mitigation: enforce a tenant-aware canonical ingestion schema and mapping layer.
- Risk: Streamlit-centric UX limits long-term SaaS UX flexibility.
  - Mitigation: treat Streamlit as transitional analytics UI behind authenticated shell.
- Risk: compliance/security debt during rapid build.
  - Mitigation: add security and audit acceptance criteria to every milestone.

---

## 8) Definition of Done for SaaS MVP
- Users can self-register, create/join an organization, and authenticate securely.
- Tenant data is isolated end-to-end (API, DB, storage, jobs).
- Users can upload energy/weather data and run analysis per tenant.
- Users can view/download reports and key ISO 50001-relevant summaries.
- Landing page clearly explains value proposition and includes conversion path.
