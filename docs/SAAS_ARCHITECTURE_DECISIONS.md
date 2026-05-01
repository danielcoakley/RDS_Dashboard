# SaaS Architecture Decisions

This records initial Milestone 0 architecture decisions for the SaaS expansion.

## ADR-001: Keep Streamlit Analytics During SaaS Shell Build

Decision: keep the current Streamlit dashboard as the short-term analytics UI while platform boundaries are added around it.

Rationale:
- Existing analytics behavior is valuable and should not be disrupted during tenant/auth foundation work.
- The SaaS shell can authenticate users and route them to analytics while the analytics code is gradually extracted.

Implications:
- Do not rewrite `app.py` as part of Milestone 0 or early Milestone 1 work.
- New tenant-aware services should wrap or call analytics functions rather than duplicating them.

## ADR-002: Add Python Backend Boundary Before External Services

Decision: introduce a local Python backend boundary for domain models, RBAC, and service contracts before adding FastAPI, PostgreSQL, object storage, or auth-provider dependencies.

Rationale:
- The repo is currently a Python Streamlit project.
- Standard-library scaffolding can be validated locally and keeps the current app runnable.
- External dependencies should be added only when their interfaces and tests are clear.

Implications:
- Early backend code lives in a simple Python package and avoids live credentials.
- API routes, migrations, and storage clients come after domain and access rules are tested.

## ADR-003: Organization Is the Tenant Boundary

Decision: organization id is the required tenant boundary for users, sites, meters, uploads, runs, reports, jobs, and storage keys.

Rationale:
- The plan defines organization creation, invite flow, RBAC, and tenant-scoped uploads/run history as MVP requirements.
- Tenant isolation must be designed before database, API, and storage implementation.

Implications:
- SaaS-facing models must include organization ownership directly or through a parent resource.
- Tests must cover cross-tenant access denial.
- Shared mutable files such as `seu_mapping.csv` must be replaced by tenant-scoped storage in future slices.

## ADR-004: Use Local Contracts Before Framework Commitments

Decision: define local domain and access contracts first; defer final choices for auth provider, job queue, object storage SDK, and web frontend framework until Milestone 1 planning.

Rationale:
- The plan recommends FastAPI, PostgreSQL, S3-compatible storage, and a dedicated web frontend, but provider choices still need implementation context.
- Local contracts let the team validate tenant isolation without committing to provider-specific APIs too early.

Implications:
- Milestone 1 should introduce the first concrete API/database/auth choices.
- Contracts added in Milestone 0 should stay provider-neutral.
