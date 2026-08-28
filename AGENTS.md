You are an autonomous coding agent working on the SaaS expansion of an ISO 50001 energy analytics platform.

The roadmap lives in `docs/SAAS_PLATFORM_PLAN.md`. Read the relevant sections before planning or implementing SaaS-related work.

Your job is to:
- use `docs/SAAS_PLATFORM_PLAN.md` as the product and implementation roadmap
- plan tasks
- implement code
- run tests or checks
- fix errors
- repeat until complete

SaaS product direction:
- preserve the existing analytics strengths in `src/` while moving toward a multi-tenant SaaS platform
- prioritize MVP capabilities first: landing pages, auth, organization/tenant creation, invite flow, Owner/Manager/Viewer RBAC, tenant-scoped uploads, run history, and basic admin views
- build toward the target architecture: dedicated SaaS frontend/dashboard experience, API layer, PostgreSQL metadata store, object storage for uploads/reports, background jobs for analyses, and audit logging
- choose the best dashboard and visualization engine for the SaaS product; Streamlit is optional and should not constrain architecture, UX, or future enhancements
- the existing `app.py` Streamlit dashboard may be overhauled, replaced, or retired when doing so better supports the SaaS roadmap

Rules:
- align changes with the relevant milestone, epic, or immediate action in `docs/SAAS_PLATFORM_PLAN.md`
- prioritize tenant isolation, secure access, user workflows, maintainability, deployability, and auditability
- do not preserve existing single-tenant Streamlit behavior by default; preserve useful analytics behavior by extracting or reusing domain logic where it still fits
- prefer clean SaaS architecture over compatibility with the current `app.py` UI
- avoid shared mutable local files for tenant data; prefer tenant-scoped storage boundaries and clear data contracts
- enforce tenant and RBAC checks at service/API boundaries when adding platform features
- keep changes small and safe
- do not break existing functionality
- always run or simulate validation steps
- fix errors before stopping
- prefer editing existing files over creating unnecessary new ones
- keep code simple and readable

Workflow:
1. Review `docs/SAAS_PLATFORM_PLAN.md` for the relevant goal, epic, milestone, or immediate next action.
2. Identify the safest small slice that advances the roadmap.
3. Plan the task and note any tenant, auth, data, or deployment impact.
4. Implement changes.
5. Run checks (tests / app / lint) or clearly simulate validation when a check cannot be run.
6. Fix any errors.
7. Repeat until task is complete.

Current roadmap priorities:
- Milestone 0: product requirements, canonical domain model, and architecture decisions
- Milestone 1: auth, organization creation, RBAC seed roles, tenant-scoped schema, and dashboard shell
- Milestone 2: tenant-scoped uploads, run orchestration, report artifacts, and end-to-end run lifecycle
- Milestone 3: landing page, app pages for uploads/runs/reports/settings, and notifications
- Milestone 4: audit logs, backups, monitoring, security review, and pilot readiness

Immediate next actions from the plan:
- split repo boundaries into `frontend/`, `backend/`, and `analytics/` or an equivalent structure
- evaluate and choose the dashboard/frontend engine that best supports SaaS workflows, rich visualization, maintainability, and future enhancements
- create initial database schema and migration setup
- implement auth and organization models
- refactor file I/O paths away from shared local mutable files
- add integration tests for tenant isolation and run lifecycle
- draft landing page copy for ISO 50001 pain points and outcomes

Definition of done:
- changes match the relevant goal in `docs/SAAS_PLATFORM_PLAN.md`
- tenant isolation, auth/RBAC, and audit/security implications are considered for SaaS-facing changes
- code runs without errors
- feature works as described
- no obvious regressions
