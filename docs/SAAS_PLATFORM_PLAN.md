# SaaS Expansion Plan for ISO 50001 Energy Analytics Platform

## 1) Current-State Review

### Architecture and product shape today
- The repo currently contains a **single-tenant Streamlit dashboard** with all legacy UI and workflow in one entry-point (`app.py`).
- The SaaS direction no longer depends on Streamlit as the visualization or dashboard engine. The current UI can be overhauled, replaced, or retired.
- Current backend SaaS scaffolding exists in `backend/`, including FastAPI app setup, tenant domain models, RBAC guards, SQLite-compatible migration scaffolding, onboarding, tenant profile updates, tenant report/evidence-package branding and section-template labels/order settings, tenant privacy settings and privacy request tracking, tenant pilot readiness tracking, tenant analysis settings with weather-model threshold controls, subscription/trial metadata with member/site/run entitlement enforcement, billing checkout/portal/webhook boundaries, report retention metadata and Settings-page application controls, report JSON exports and evidence packages, report schedule metadata and worker-backed scheduled notifications, report review decisions and collaboration comments, artifact deletion enforcement, trusted-origin/security-header middleware, tenant-safe request logging, invite/member administration, tenant-scoped listing routes, and environment-gated development vs Clerk JWT auth modes.
- The Next.js dashboard now renders tenant-derived ECharts visualizations for source readiness, run lifecycle, meter mix, generated report artifact evidence, tenant-scoped report review decisions, and report review comments.
- The legacy app allows users to upload CSVs for energy, HDD, CDD, and SEU mapping, then performs baseline and comparison analysis.
- Tenant upload ingestion now enforces canonical source-file categories (`energy`, `hdd`, `cdd`, and `seu_mapping`), validates source CSV shape before artifact storage, exposes per-site source readiness for the SaaS upload workflow, includes an initial tenant-scoped data-connector registry plus import-to-upload and sync-attempt lifecycle metadata for API/SFTP/utility-portal source imports, and applies tenant-owned weather-normalized M&V settings when generating legacy source-package report artifacts.
- Client selection is file-based from `config/clients/*.yaml`, which is a good foundation for future tenant templates.

### Strengths to preserve
- Existing domain logic for ISO 50001 support is now exposed through the canonical `analytics/` package (`config_loader`, `data_standardisation`, `data_quality`, `reporting`, `enpi_model`, and `service`), with `src/` retained as a compatibility layer for the legacy Streamlit entry point.
- Existing tests cover config and standardization modules.
- Data processing and EnPI-style reporting already exist and can be wrapped into reusable tenant services.
- New backend scaffolding has started establishing tenant isolation, RBAC, API boundaries, and local schema contracts.

### Gaps blocking SaaS readiness
- Auth, authorization, and account models are still MVP scaffolding; opt-in Clerk frontend/backend wiring, account recovery entry page, initial active-organization metadata sync, signed webhook/profile synchronization code, sanitized Clerk replay evidence, production readiness gating for recent Clerk webhook replay, and fresh auth-flow verification evidence readiness exist, but deployment keys, Clerk dashboard webhook setup, auth-flow/account recovery verification execution, and production validation are not yet live.
- Tenant isolation has initial domain, API, RBAC, metadata database, tenant analysis-config, artifact-store, worker, deployment-boundary scaffolding, artifact-store governance evidence readiness, support-protected synthetic tenant-isolation and run-lifecycle smoke evidence, and production readiness evidence gating, but still needs production database/storage deployment, live smoke execution, and broader live end-to-end verification.
- Persistent backend models for users, organizations, assets, uploads, runs, reports, jobs, notifications, subscriptions, and audit events exist in migration scaffolding with an env-selected SQLite/PostgreSQL metadata boundary, plus a support-protected database smoke endpoint that records tenant-safe production readiness evidence, but live PostgreSQL deployment, smoke execution, backups, and restore drills remain outstanding.
- Initial marketing/landing experience exists in the Next.js app with ISO 50001 conversion copy, a public create-only pilot inquiry flow, duplicate-window/honeypot/hashed-source rate-limit spam hardening, configurable support handoff email, signed CRM webhook handoff, and protected support follow-up APIs with queue filters, aggregate summaries, assignment/notes metadata, and sanitized activity history; production analytics, richer abuse analytics, and full sales/support dashboards still need hardening.
- Initial subscription tier, tenant subscription, default trial, owner-only plan update, settings-page subscription visibility, backend entitlement checks for member seats, sites, and monthly runs, owner-only checkout/portal session creation, idempotent billing webhook records, checkout completion, invoice-failure dunning reconciliation, Stripe webhook replay-tolerance validation, sanitized billing replay evidence, production readiness gating for recent billing webhook replay, and fresh billing-operations evidence readiness exist; live Stripe dashboard setup, invoicing/tax settings, replay testing, provider lifecycle verification, and billing evidence capture still remain. Onboarding/account recovery still need production validation; initial role-based access control and member administration exist but need live Clerk write-back or operational ownership decisions.
- Initial production-shaped service architecture exists for API, worker, frontend, metadata database, artifact storage, trusted-origin API security, frontend edge hardening headers with report-only CSP, sanitized CSP report collection, support-protected CSP report summary/list review and retention cleanup, aggregate operational metrics, auth-flow/backup-job/alerting-rules/worker/security-review/tenant-isolation/provider-replay/artifact-store-test/artifact-store-governance/database-test/run-lifecycle-test/worker-run-lifecycle-test/log-aggregation/worker-supervision/billing-operations/email-delivery readiness gates, support-owner readiness checks, support-protected alert-smoke, database smoke-test, artifact-store smoke-test, tenant-isolation smoke-test, run-lifecycle smoke-test, worker run-lifecycle smoke-test, provider replay, and scheduled-report dispatch evidence, support-protected worker heartbeat supervision without tenant ids, alert-smoke/provider-replay/artifact-store-test/database-test/tenant-isolation-test/run-lifecycle-test/worker-run-lifecycle-test production recency checks, and metadata-backed worker heartbeat validation, but live infrastructure provisioning, live auth-flow execution, live worker supervisor verification, monitoring-stack integration, backup job/restore execution, live bucket governance/database/billing verification, live tenant-isolation smoke execution, live run-lifecycle smoke execution, live worker run-lifecycle smoke execution, live email provider execution, and the signed-off security and tenant-isolation reviews remain.
- Legacy visualization and workflow are tightly coupled to Streamlit and `app.py`; this should be treated as replaceable UI code, not a constraint.

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
- Report scheduling, exports, approvals, and collaboration comments. Initial tenant-scoped report schedule metadata, worker-backed scheduled report notifications with safe artifact digests, delivery history visibility, tenant-safe schedule ownership transfer on member removal, JSON report exports, tenant-configurable branded/section-templated dynamic JSON/PDF evidence packages with client-facing section labels/order, approval-gated locked JSON evidence package snapshots, report approval assignments with digest-backed retryable outbox notifications, aggregate outbox and report-schedule execution health metrics, support-listable scheduled-dispatch and provider-replay operation evidence, signed timestamp-bound external provider delivery-event ingestion, report review decisions, and report comments exist; fully custom rendered template layouts, attachment delivery, live provider replay execution, and deeper multi-step approval workflows remain.
- Improved data connectors (beyond CSV: API/SFTP/utility integrations). Initial tenant-scoped connector metadata registry, due-connector selection, connector-supplied import-to-upload API, provider-backed due-connector worker boundary, sync-attempt history, upload-link validation, Settings UI, support-visible worker heartbeat counts, and sanitized audit trail exist; concrete API/SFTP/utility provider adapters, secret-manager integration, and live provider validation still remain.
- Auditable compliance trail for ISO 50001 evidence packages. Initial tenant-configurable branded/section-templated dynamic JSON/PDF evidence package exports with section labels/order, approval-gated locked JSON evidence package snapshots, approval assignment evidence, owner-managed evidence-lock policy settings, and automatic locked snapshots after final approval exist; fully custom evidence-pack layout rendering and deeper multi-step approval policy rules remain.
- Subscription billing + trial management.

## Phase 3
- Benchmarking, portfolio dashboards, and advanced M&V workflows.
- White-label options and enterprise SSO/SCIM.

---

## 3) Proposed Technical Architecture

### Frontend
- Select the dashboard/frontend engine that best serves the SaaS product; Streamlit is optional, not required.
- Prefer a dedicated web frontend/dashboard architecture for landing pages, auth flows, tenant/admin UX, uploads, run history, report review, and rich visual analytics.
- Selected direction: Next.js with React/TypeScript for the SaaS frontend/dashboard shell and Apache ECharts for rich tenant-scoped visualizations.
- The existing Streamlit `app.py` can be replaced rather than embedded if replacement produces a cleaner SaaS architecture.

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
- Comprehensive audit logging for upload, config changes, run execution, report access/export/scheduling, report approval decisions, and report collaboration.

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
2. Implement projects/uploads/runs/report endpoints. Initial tenant-scoped report comments, review decisions, data connector metadata registry, connector import-to-upload route, connector sync-attempt lifecycle routes, and provider-backed connector worker boundary exist through tenant-guarded routes.
3. Add background job execution for heavy analyses.
4. Add observability (structured logs, tracing, metrics). Initial aggregate `/metrics` output, including notification outbox due/retry/failed counts, oldest due-notification presence/age, provider delivery-event status counts, report schedule execution-phase counts, oldest due-report-schedule presence/age, report schedule delivery-attempt status counts, scheduled-dispatch platform operation success/failure plus latest-success age metrics, provider replay operation success/failure counts, artifact-store smoke-test operation success/failure plus latest-success age metrics, database smoke-test operation success/failure plus latest-success age metrics, tenant-isolation smoke-test operation success/failure plus latest-success age metrics, run-lifecycle smoke-test operation success/failure plus latest-success age metrics, worker run-lifecycle smoke-test operation success/failure plus latest-success age metrics, public marketing inquiry lifecycle counts, accepted/rate-limited submission outcomes, sanitized rejection reasons without contact details or source hashes, worker heartbeat scope/presence/age metrics without tenant or worker ids, alert-smoke platform operation success/failure plus latest-success age metrics without request ids or detail text, CSP violation report counts by disposition and sanitized effective directive without document origins or blocked URLs, support-protected sanitized CSP report summary/list review and retention cleanup, support-protected worker heartbeat summary/list review with stable worker fingerprints and no organization ids, structured request logs with `X-Request-Id` and `traceparent` propagation, auth-flow verification readiness checks, log aggregation URL/freshness/retention readiness checks, worker-supervision evidence readiness checks, billing-operations evidence readiness checks, email-delivery verification readiness checks, artifact-store governance evidence readiness checks, alert-rule deployment evidence readiness checks, notification and report-schedule alert-threshold readiness checks, signed notification-provider webhook readiness checks with replay-tolerance validation, provider webhook replay evidence readiness checks, artifact-store smoke-test evidence readiness checks, database smoke-test evidence readiness checks, tenant-isolation smoke-test evidence readiness checks, run-lifecycle smoke-test evidence readiness checks, worker run-lifecycle smoke-test evidence readiness checks, CSP report retention-window readiness checks, support-protected generic alert delivery, database smoke, tenant-isolation smoke, run-lifecycle smoke, worker run-lifecycle smoke, and artifact-store smoke endpoints with platform operation evidence, support-listable scheduled-report dispatch, database smoke, tenant-isolation smoke, run-lifecycle smoke, worker run-lifecycle smoke, artifact-store smoke, and provider-replay operation evidence, recent successful alert-smoke readiness checks, and production worker-readiness checks exist for pilot monitoring without tenant identifiers; live log aggregation verification, live worker supervisor verification, live provider replay execution, live artifact-bucket validation, live database smoke execution, live tenant-isolation smoke execution, live run-lifecycle smoke execution, live worker run-lifecycle smoke execution, and monitoring-stack wiring remain.

## Epic D — Analytics Service Refactor
1. Extract current analytics functions into service layer callable by API jobs.
2. Remove local-file coupling (`seu_mapping.csv`) in favor of tenant-scoped storage.
3. Standardize data contract between upload ingestion and analytics pipeline. Initial source-file category normalization, pre-storage CSV shape validation, tenant/site readiness metadata, tenant-scoped data connector source-category metadata, connector import-to-upload validation, provider-backed due connector import worker boundary, sync-to-upload validation, stored-run source package lineage, legacy energy export resolution into the canonical meter dataframe, aggregate legacy source-package report metrics, and tenant-configurable HDD/CDD weather-normalized model threshold summaries now exist; concrete connector provider adapters plus broader production M&V/model-family selection workflows remain.
4. Expand tests from unit to integration for full run lifecycle. Initial stored-upload, report retrieval/export, support-protected synthetic run-lifecycle smoke tests, and support-protected queued-worker run-lifecycle smoke tests exist; broader live pilot tenant validation remains.

## Epic E — SaaS Web UX
1. Evaluate and choose the dashboard/frontend engine for the SaaS UX.
2. Build landing page and feature pages.
3. Build signup/login/reset pages. Initial local and Clerk-backed pages exist; live provider email recovery verification remains.
4. Build app shell (tenant switcher, nav, user menu).
5. Build upload/run/report history views. Initial report-detail scheduling controls, approval assignment routing, tenant-configurable evidence-locking policy and evidence-section template label/order settings, automatic evidence snapshot locking after final approval, management review decisions, JSON artifact downloads, evidence package downloads, and collaboration comments exist; richer review/export workflows remain.
6. Build rich tenant-scoped visualization views using the selected dashboard engine. Initial dashboard, report-detail ECharts, weather-normalized report model views, and skipped-model diagnostics exist; richer portfolio and M&V visualizations remain.

## Epic F — Compliance, Governance, and Billing
1. Add audit log UI and export.
2. Add consent/privacy controls and retention policies. Initial privacy settings with data-processing acknowledgement, owner-managed privacy request tracking for export/deletion/correction, report retention metadata, read-path filtering, tenant artifact deletion, retention application API, and audit events exist; live legal fulfillment, automated privacy operations, and provider-side lifecycle/backup verification remain.
3. Implement subscription tiers, checkout, and dunning flows. Initial tenant subscription/trial metadata, backend entitlement checks, owner-only checkout/portal sessions, idempotent billing webhooks, invoice-failure dunning reconciliation, and production billing-operations evidence readiness exist; live provider operations and invoicing setup remain.

---

## 5) Milestone Plan (Recommended)

## Milestone 0 (1–2 weeks): Alignment & Foundations
- Deliverables:
  - Product requirements doc for SaaS MVP.
  - Canonical domain model (User, Organization, Site, Meter, Upload, Run, Report).
  - Target architecture decision record.
  - Dashboard/frontend engine decision criteria and preferred direction.
- Exit criteria:
  - Team sign-off on MVP scope and implementation stack.

## Milestone 1 (2–4 weeks): Auth + Tenant Skeleton
- Deliverables:
  - Functional signup/login and organization creation.
  - RBAC seed roles and tenant-scoped database schema.
  - Basic dashboard shell after login using the selected frontend/dashboard direction.
- Exit criteria:
  - Two test tenants can log in and cannot see each other’s data.

## Milestone 2 (3–5 weeks): Ingestion + Run Orchestration
- Deliverables:
  - Tenant-scoped uploads persisted to object storage.
  - Tenant source-file category contract and per-site upload readiness visibility.
  - Tenant-scoped connector registry, connector import-to-upload route, provider-backed connector worker boundary, and sync-attempt history for future API/SFTP/utility imports.
  - API-triggered analytics runs with status lifecycle and tenant source package lineage.
  - Support-protected synthetic upload-to-report run-lifecycle smoke evidence.
  - Initial report artifacts stored, retrievable, and downloadable as tenant-scoped JSON exports, dynamic JSON/PDF evidence packages, and locked evidence package snapshots.
  - Initial tenant-scoped report comments and review decisions stored, listed, and audited.
- Exit criteria:
  - End-to-end flow from upload to result works for at least one tenant.

## Milestone 3 (3–5 weeks): SaaS UX + Landing Experience
- Deliverables:
  - Public landing page and conversion CTAs.
  - In-app pages for uploads, runs, reports, settings, including initial report review decisions and comments.
  - Tenant-scoped analytics visualizations in the selected dashboard engine.
  - Email notifications for run completion and report approval routing.
- Exit criteria:
  - Pilot users can self-serve core workflow without operator intervention.

## Milestone 4 (2–4 weeks): Hardening + Pilot Readiness
- Deliverables:
  - Audit logs, backups, monitoring, alerting.
  - Security review, tenant-isolation review, and threat model outcomes. Production readiness now requires fresh HTTPS evidence URLs and signed-off timestamps, but the live reviews themselves still remain.
  - Pilot onboarding toolkit and support playbook. Initial owner-managed tenant pilot readiness tracking and production support-owner readiness checks exist; live evidence and external sign-off still remain.
- Exit criteria:
  - Platform ready for controlled multi-client pilot.

---

## 6) Immediate Next Actions (next 10 working days)
1. Split current repo into clear boundaries: `frontend/`, `backend/`, `analytics/` (or equivalent). Initial boundaries exist, with backend orchestration calling the canonical `analytics/` service and `src/` retained only for legacy compatibility.
2. Evaluate dashboard/frontend engine options and choose the best fit for SaaS workflows and future visual enhancements.
3. Create initial DB schema and migration setup.
4. Implement auth and organization models.
5. Refactor file I/O paths so no shared local mutable files are needed.
6. Add integration tests for tenant isolation and run lifecycle. Initial service/API, support-protected synthetic smoke coverage, and queued-worker run-lifecycle smoke coverage exist; live production smoke execution remains.
7. Draft landing page copy oriented to ISO 50001 pain points and outcomes.

---

## 7) Risks and Mitigations
- Risk: analytics pipeline assumptions tied to one client format.
  - Mitigation: enforce a tenant-aware canonical ingestion schema and mapping layer.
- Risk: over-preserving the legacy Streamlit UI slows SaaS architecture and visualization improvements.
  - Mitigation: treat Streamlit as replaceable legacy UI and choose the dashboard engine that best supports the product roadmap.
- Risk: compliance/security debt during rapid build.
  - Mitigation: add security and audit acceptance criteria to every milestone.

---

## 8) Definition of Done for SaaS MVP
- Users can self-register, create/join an organization, and authenticate securely.
- Tenant data is isolated end-to-end (API, DB, storage, jobs).
- Users can upload energy/weather data and run analysis per tenant.
- Users can view/download report JSON artifacts, current JSON/PDF evidence packages, and locked evidence package snapshots, record management review decisions, collaborate on report review notes, and see key ISO 50001-relevant summaries.
- Landing page clearly explains value proposition and includes conversion path.
