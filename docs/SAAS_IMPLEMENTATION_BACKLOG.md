# SaaS Implementation Backlog

This backlog is ordered by dependency and aligned to `docs/SAAS_PLATFORM_PLAN.md`.

## Milestone 0: Foundations

1. Keep MVP product requirements and domain model documented.
2. Keep architecture decisions documented and update them when stack choices are made.
3. Define dashboard/frontend engine selection criteria and preferred direction. Next.js with React/TypeScript and Apache ECharts is selected in `docs/DASHBOARD_ENGINE_DECISION.md`.
4. Add local domain and RBAC scaffolding with tests.
5. Identify current shared local-file couplings that must become tenant-scoped.

## Milestone 1: Auth and Tenant Skeleton

1. Choose auth provider and session approach. Clerk is selected in `docs/AUTH_PROVIDER_DECISION.md`; temporary request-user context exists in `backend/auth_context.py` and should be replaced by verified Clerk session claims.
2. Choose API framework and database migration tool. FastAPI is the API boundary, with the local migration runner retained until PostgreSQL migrations are introduced.
3. Add database schema for users, organizations, memberships, sites, and meters. Initial local SQLite-compatible scaffolding exists in `backend/migrations/001_tenant_skeleton.sql`, with repository helpers in `backend/store.py` and owner-organization onboarding in `backend/onboarding.py`.
4. Add tenant-scoped API/service access checks. Initial service-boundary guards exist in `backend/access_control.py`.
5. Add invite and membership management flow. Initial local invite schema, invite acceptance service, audit events, and tenant-guarded invite listing/creation routes now exist in `backend/migrations/004_invites.sql`, `backend/invitations.py`, and `backend/api.py`.
6. Add basic dashboard shell after login. Use the selected Next.js frontend direction; the API now exposes local owner-organization onboarding, invite management, user organization listing, and tenant-guarded site/meter listing routes that can back the first authenticated shell.
7. Add integration tests proving two tenants cannot see each other's resources.

## Milestone 2: Ingestion and Run Orchestration

1. Replace shared `seu_mapping.csv` behavior with tenant-scoped upload storage.
2. Add upload records and storage-key generation. Initial upload/run/report metadata tables exist in `backend/migrations/002_runs_and_reports.sql`, with tenant-scoped key helpers in `backend/storage_keys.py`.
3. Wrap current analytics pipeline behind a run service. Initial analytics service boundary exists in `analytics/service.py`.
4. Add run status lifecycle and report artifact records. Initial local orchestration exists in `backend/run_orchestration.py` and is exposed through a FastAPI local execution route.
5. Add end-to-end tests from upload to report retrieval. Local execution and tenant-guarded report listing are covered in API/service tests.

## Milestone 3: SaaS UX and Landing Experience

1. Draft landing page copy around ISO 50001 pain points and outcomes.
2. Build landing page and conversion CTA.
3. Build app pages for uploads, runs, reports, tenant settings, and user management.
4. Build rich tenant-scoped visualization views in the selected dashboard engine.
5. Add notifications for run completion.

## Milestone 4: Pilot Readiness

1. Add audit events for upload, config change, run execution, and report access. Initial audit-event schema, run orchestration events, and tenant-guarded audit listing API exist.
2. Add monitoring, backup, and alerting plan.
3. Run security review and threat model.
4. Prepare pilot onboarding and support playbook.
