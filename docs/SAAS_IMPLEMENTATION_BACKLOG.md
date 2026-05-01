# SaaS Implementation Backlog

This backlog is ordered by dependency and aligned to `docs/SAAS_PLATFORM_PLAN.md`.

## Milestone 0: Foundations

1. Keep MVP product requirements and domain model documented.
2. Keep architecture decisions documented and update them when stack choices are made.
3. Add local domain and RBAC scaffolding with tests.
4. Identify current shared local-file couplings that must become tenant-scoped.

## Milestone 1: Auth and Tenant Skeleton

1. Choose auth provider and session approach.
2. Choose API framework and database migration tool. FastAPI is the API boundary, with the local migration runner retained until PostgreSQL migrations are introduced.
3. Add database schema for users, organizations, memberships, sites, and meters. Initial local SQLite-compatible scaffolding exists in `backend/migrations/001_tenant_skeleton.sql`, with repository helpers in `backend/store.py` and owner-organization onboarding in `backend/onboarding.py`.
4. Add tenant-scoped API/service access checks. Initial service-boundary guards exist in `backend/access_control.py`.
5. Add basic dashboard shell after login. The API now exposes local owner-organization onboarding, user organization listing, and tenant-guarded site/meter listing routes that can back the first authenticated shell.
6. Add integration tests proving two tenants cannot see each other's resources.

## Milestone 2: Ingestion and Run Orchestration

1. Replace shared `seu_mapping.csv` behavior with tenant-scoped upload storage.
2. Add upload records and storage-key generation.
3. Wrap current analytics pipeline behind a run service.
4. Add run status lifecycle and report artifact records.
5. Add end-to-end tests from upload to report retrieval.

## Milestone 3: SaaS UX and Landing Experience

1. Draft landing page copy around ISO 50001 pain points and outcomes.
2. Build landing page and conversion CTA.
3. Build app pages for uploads, runs, reports, tenant settings, and user management.
4. Add notifications for run completion.

## Milestone 4: Pilot Readiness

1. Add audit events for upload, config change, run execution, and report access.
2. Add monitoring, backup, and alerting plan.
3. Run security review and threat model.
4. Prepare pilot onboarding and support playbook.
