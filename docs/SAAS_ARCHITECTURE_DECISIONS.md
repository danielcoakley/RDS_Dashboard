# SaaS Architecture Decisions

This records initial Milestone 0 architecture decisions for the SaaS expansion.

## ADR-001: Treat Streamlit as Replaceable Legacy UI

Decision: do not require Streamlit as the SaaS dashboard or visualization engine. The current `app.py` Streamlit dashboard is legacy UI that may be overhauled, replaced, or retired.

Rationale:
- Existing analytics logic is valuable, but the Streamlit UI should not constrain SaaS architecture, UX, or visualization capabilities.
- A purpose-built dashboard/frontend can better support tenant workflows, richer visualization, collaboration, and future enhancements.

Implications:
- Future work may replace `app.py` entirely when a better SaaS frontend/dashboard engine is selected.
- Preserve useful behavior by extracting analytics and reporting logic, not by preserving the current Streamlit workflow.
- Frontend decisions should optimize maintainability, interactivity, tenant UX, and long-term extensibility.

## ADR-002: Add Python Backend Boundary Before External Services

Decision: introduce a local Python backend boundary for domain models, RBAC, and service contracts before adding FastAPI, PostgreSQL, object storage, or auth-provider dependencies.

Rationale:
- The repo now has Python backend scaffolding around the existing analytics modules.
- Standard-library scaffolding can be validated locally while frontend/dashboard choices are evaluated separately.
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

## ADR-004: Use Local Contracts Before Provider Commitments

Decision: define local domain and access contracts first; defer final choices for auth provider, job queue, object storage SDK, and dashboard/frontend framework until Milestone 1 planning.

Rationale:
- The plan recommends FastAPI, PostgreSQL, S3-compatible storage, and a dedicated SaaS frontend/dashboard experience, but provider and visualization-engine choices still need implementation context.
- Local contracts let the team validate tenant isolation without committing to provider-specific APIs too early.

Implications:
- Milestone 1 should introduce the first concrete API/database/auth choices.
- Contracts added in Milestone 0 should stay provider-neutral.
- Dashboard/frontend selection should be made intentionally and should not default to Streamlit.

## ADR-005: Use FastAPI for the Backend API Boundary

Decision: use FastAPI for the SaaS backend API boundary.

Rationale:
- The SaaS plan recommends FastAPI for auth/session handling, tenant CRUD, run orchestration, and audit events.
- FastAPI fits the current Python codebase and can call the existing analytics modules without a language boundary.

Implications:
- Early API work starts in `backend/api.py`.
- API routes must call tenant access guards before reading or mutating tenant resources.
- Route handlers should stay thin and delegate persistence to store/service modules.

## ADR-006: Use Next.js and Apache ECharts for the SaaS Dashboard

Decision: use Next.js with React and TypeScript for the SaaS frontend/dashboard shell, with Apache ECharts as the primary visualization engine.

Rationale:
- Next.js provides a mature React structure for landing pages, authenticated app pages, routing, and future rendering choices.
- ECharts supports rich interactive dashboards and future portfolio-level visualizations without tying the product to Streamlit.
- This split keeps the Python backend focused on API, tenant security, orchestration, and analytics services.

Implications:
- Frontend work should live under `frontend/`.
- The frontend should consume FastAPI contracts rather than importing Python analytics code directly.
- `app.py` can be retired once the new frontend covers the required workflows.
