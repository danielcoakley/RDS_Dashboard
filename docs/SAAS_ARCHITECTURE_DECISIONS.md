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
- The repo now has Python backend scaffolding around the canonical `analytics/` package, with legacy `src/` imports retained for the Streamlit compatibility layer.
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
- FastAPI fits the current Python codebase and can call the canonical `analytics/` package without a language boundary.

Implications:
- Early API work starts in `backend/api.py`.
- API routes must call tenant access guards before reading or mutating tenant resources.
- Route handlers should stay thin and delegate persistence to store/service modules.

## ADR-006: Use Next.js and Apache ECharts for the SaaS Dashboard

Decision: use Next.js with React and TypeScript for the SaaS frontend/dashboard shell, with Apache ECharts as the primary visualization engine.

Rationale:
- Next.js provides a mature React structure for landing pages, authenticated app pages, routing, and future rendering choices.
- ECharts supports rich interactive dashboards and future portfolio-level visualizations without tying the product to Streamlit.
- This split keeps the Python backend focused on API, tenant security, orchestration, and calls into the separate `analytics/` service boundary.

Implications:
- Frontend work should live under `frontend/`.
- The frontend should consume FastAPI contracts rather than importing Python analytics code directly.
- `app.py` can be retired once the new frontend covers the required workflows.
- The initial dashboard ECharts view visualizes tenant-derived source readiness, run/job lifecycle, and meter/SEU mix data from FastAPI-backed workspace state.

## ADR-007: Use Clerk for MVP Auth and Organizations

Decision: use Clerk as the preferred MVP authentication and organization provider.

Rationale:
- Clerk aligns with the selected Next.js frontend direction and provides organization-oriented flows.
- Organization roles can map to the platform Owner, Manager, and Viewer RBAC model.
- Clerk session tokens can be verified by FastAPI through JWT/JWKS validation while keeping the backend authorization checks authoritative.

Implications:
- `backend/auth_context.py` keeps a credential-free `dev` mode for tests and local flows, and adds `clerk` mode for verified Clerk session JWTs.
- Clerk mode requires `CLERK_ISSUER` and either `CLERK_JWT_KEY` or `CLERK_JWKS_URL`; optional audience and authorized-party allowlists can further constrain accepted tokens.
- Backend API checks remain authoritative for tenant data access.
- Tenant profile updates are owner-only through backend RBAC and create audit events.
- Organization-scoped routes should require an active Clerk organization claim and reject a token whose organization claim does not match the requested organization path.
- The frontend wraps the app with `ClerkProvider`, runs `clerkMiddleware()` from `proxy.ts`, uses Clerk sign-in/sign-up components, sends `auth().getToken()` to FastAPI, and exposes Clerk organization switching when Clerk keys are configured.
- Local development remains credential-free when Clerk keys are absent.
- Initial Clerk user/organization activation uses `POST /auth/clerk/sync` to populate backend tenant metadata from verified active-organization session claims.
- Signed Clerk webhooks use `POST /webhooks/clerk` plus Svix verification to synchronize user, organization, and organization membership create/update/delete events into backend tenant metadata. Live Clerk dashboard endpoint setup and replay validation are still required before production tenant workflows are fully self-service.
- Production readiness requires fresh HTTPS auth-flow verification evidence through `SAAS_AUTH_FLOW_VERIFICATION_URL`, `SAAS_AUTH_FLOW_VERIFIED_AT`, and `SAAS_AUTH_FLOW_VERIFICATION_MAX_AGE_DAYS`, which defaults to 30 days. This evidence must cover sign-up, sign-in, organization activation/switching, invite acceptance, and account recovery with live Clerk settings.

## ADR-008: Use S3-Compatible Storage Behind the Tenant Artifact Store

Decision: use the tenant artifact-store boundary as the upload/report storage contract, with local and in-memory stores for development/tests and an S3-compatible adapter for production-shaped deployments.

Rationale:
- The SaaS plan requires tenant-scoped uploads and report artifacts to move away from shared mutable local files.
- S3-compatible storage matches the target architecture while keeping provider choice flexible.
- A tested interface lets run orchestration and API routes persist artifacts without knowing the storage provider.

Implications:
- Runtime selection uses `SAAS_ARTIFACT_STORE_BACKEND` with `local`, `memory`, or `s3`.
- S3 deployments require `SAAS_S3_BUCKET` and can use `SAAS_S3_PREFIX`, `SAAS_S3_ENDPOINT_URL`, and `AWS_REGION` or `AWS_DEFAULT_REGION`.
- Credentials stay outside the repo, and `/ready` should continue to block production readiness when a local-only artifact store is configured.
- A support-protected artifact-store smoke test now verifies write/read/delete behavior and records tenant-safe `rds.artifact_store_test` evidence for production readiness.
- Production readiness also requires fresh HTTPS bucket-governance evidence through `SAAS_ARTIFACT_STORE_GOVERNANCE_URL`, `SAAS_ARTIFACT_STORE_GOVERNANCE_VERIFIED_AT`, and `SAAS_ARTIFACT_STORE_GOVERNANCE_MAX_AGE_DAYS`, which defaults to 30 days.
- Production bucket provisioning, access-policy review, encryption/versioning/lifecycle verification, backup policy, support smoke execution, and live deployment validation still need to be performed and recorded before pilot readiness.

## ADR-009: Use PostgreSQL for Production Metadata

Decision: keep SQLite as the local development/test metadata store, but require PostgreSQL-shaped metadata configuration for production readiness through `SAAS_DATABASE_URL`.

Rationale:
- The SaaS plan requires a deployable metadata store for users, organizations, memberships, uploads, runs, reports, jobs, notifications, and audit events.
- SQLite is useful for fast local tests but is not the production tenant metadata boundary.
- An env-selected database boundary lets the API, migration runner, and readiness checks move toward deployment without hard-coding local file state.

Implications:
- `SAAS_DATABASE_URL` defaults to `sqlite:///:memory:` for development and tests.
- Production deployments should set `SAAS_DATABASE_URL` to a PostgreSQL URL and install the `psycopg` driver dependency.
- The current migration runner can apply migrations through the PostgreSQL adapter path, but live PostgreSQL execution and operational validation remain required before pilot readiness.
- `/ready` should fail production readiness when SQLite metadata storage is configured.
- A support-protected database smoke test now verifies metadata write/read behavior and records tenant-safe `rds.database_test` evidence for production readiness.

## ADR-010: Package API, Worker, and Frontend as Separate Services

Decision: package the FastAPI API, background analysis worker, and Next.js frontend as separate deployable services that share the same metadata database and tenant artifact-store boundary.

Rationale:
- The SaaS plan separates request/response API work from long-running analysis jobs.
- A dedicated worker process lets uploads and run requests return promptly while queued analyses are drained independently.
- Keeping frontend, API, and worker as separate services makes it possible to scale the dashboard and analysis workload differently during pilot rollout.

Implications:
- `Dockerfile.backend` builds the Python runtime used by both the API and worker.
- `frontend/Dockerfile` builds and runs the Next.js dashboard shell.
- `docker-compose.yml` provides a local pilot-shaped topology with PostgreSQL and a shared artifact volume.
- `backend.worker_app` is the worker process entrypoint and supports `SAAS_WORKER_RUN_ONCE=true` for smoke checks.
- Production deployments must replace local compose secrets, use PostgreSQL plus S3-compatible storage, supervise worker processes with fresh readiness evidence, and validate `/ready` before onboarding pilot tenants.

## ADR-011: Use a Configurable Email Delivery Boundary

Decision: keep tenant notification state in the metadata database and deliver email through a runtime-selected sender boundary, with in-memory delivery for local development/tests and SMTP for production-shaped deployments.

Rationale:
- The SaaS plan requires notifications for run completion and report approval routing, but delivery providers should not be hard-coded into workflow services.
- A metadata-backed outbox lets the API and worker track sent/failed delivery state per tenant.
- SMTP is a broadly supported first production adapter while keeping the code open to a provider API later.

Implications:
- `backend.notifications.email_sender_from_env()` chooses the sender through `SAAS_EMAIL_BACKEND`.
- Public marketing inquiry handoff can use the same email boundary and/or a signed CRM webhook configured with `SAAS_MARKETING_INQUIRY_CRM_WEBHOOK_URL`; webhook failures remain best-effort after durable inquiry storage.
- Public marketing inquiry abuse throttling stores only an HMAC source hash in `marketing_inquiry_submission_events`; `SAAS_MARKETING_INQUIRY_RATE_LIMIT_SECRET` is required in production so raw IP/user-agent data is not persisted for rate limiting.
- `SAAS_EMAIL_BACKEND=memory` remains local/test-only.
- Run orchestration creates pending email notifications for run success/failure; report approval assignment creation/update creates pending email notifications for routing; delivery state is tracked separately from workflow state.
- Normalized notification-provider delivery callbacks are signed with `X-RDS-Timestamp` plus `X-RDS-Signature` over `<timestamp>.<raw JSON body>`, and production readiness validates the configured replay tolerance.
- Production readiness fails unless `SAAS_EMAIL_BACKEND=smtp` has required SMTP settings plus fresh HTTPS email delivery verification evidence from `SAAS_EMAIL_DELIVERY_VERIFICATION_URL` and `SAAS_EMAIL_DELIVERY_VERIFIED_AT`.
- `SAAS_EMAIL_DELIVERY_VERIFICATION_MAX_AGE_DAYS` defaults to 30 and rejects stale sender-domain/deliverability evidence.
- Live sender-domain verification, bounce handling, retry policy, provider replay testing, and delivery monitoring still need to be performed and recorded before pilot readiness.

## ADR-012: Expose Tenant-Safe Aggregate Operational Metrics

Decision: expose Prometheus-compatible operational metrics at `GET /metrics`, protected by `RDS_METRICS_TOKEN` in production and limited to aggregate platform counts.

Rationale:
- The SaaS plan requires monitoring and alerting before a controlled multi-client pilot.
- Early pilot operations need visibility into tenant volume, upload/run/job status, and notification delivery health without exposing tenant identifiers.
- Keeping metrics labels restricted to finite lifecycle statuses avoids high-cardinality labels and prevents organization or user IDs from leaving the API boundary through monitoring.

Implications:
- `RDS_METRICS_TOKEN` is required by production readiness and must be supplied to monitoring systems as a bearer token.
- Development can leave the token unset for local smoke checks.
- `/metrics` should continue to report aggregate counts only unless a future privacy review approves richer observability data.
- Public inquiry submission metrics report only finite outcomes (`accepted`, `rate_limited`) and sanitized rejection reasons such as `source_rate_limit`; they never include source hashes, emails, messages, names, or tenant identifiers.
- Report schedule metrics report only finite schedule phases and delivery statuses; they never include organization ids, report ids, schedule ids, schedule delivery ids, recipient emails, or delivery error text.
- Platform operation metrics report only finite event types (`alert_delivery_test`, `report_schedule_dispatch`, `clerk_webhook_replay`, `billing_webhook_replay`, `notification_provider_webhook_replay`, `artifact_store_test`, `database_test`) and statuses; they never include request ids, worker ids, organization ids, schedule ids, report ids, recipients, webhook URLs, database URLs, storage keys, object keys, or operation detail text.
- Support-protected worker heartbeat endpoints may expose worker fingerprints, finite scopes, batch settings, counters, and timestamps, but they must not expose raw worker ids, organization ids, tenant names, queued resource ids, artifact keys, or recipient values.
- Production readiness requires `SAAS_SUPPORT_OWNER_EMAIL` alongside support-token protection so pilot support workflows have an accountable owner before multi-client launch.
- Live monitoring stack integration, alert-rule deployment, log retention, and backup verification remain required before pilot readiness.

## ADR-013: Store Analysis Assumptions as Tenant Metadata

Decision: persist each organization's analysis configuration in the metadata store and snapshot the resolved config into queued analysis jobs.

Rationale:
- The SaaS plan requires reusing the analytics pipeline for multiple clients without hard-coding one client's meter assumptions or ISO 50001 notes in the frontend.
- The analytics contract already expects a structured config with client, site, meter, analysis, and ISO 50001 sections.
- Saving the config behind tenant/RBAC-guarded API routes lets tenant managers own the assumptions used for future runs.

Implications:
- `analysis_configs` stores the current organization-level config, including tenant-managed weather-normalized M&V controls for enablement plus minimum baseline/comparison record thresholds. Updates require asset-management permission and create `analysis_config_updated` audit events with only meter count, client id, and weather-model setting summaries.
- If no saved config exists, the backend can generate a config from tenant organization, site, and meter records.
- Queued jobs store the resolved config JSON so future edits do not mutate the assumptions behind an already-requested run, and stored-upload reports include the effective weather-model settings in the generated artifact summary.
- Future UI should continue replacing the JSON editor with field-level controls and add config version history before broad rollout.

## ADR-014: Keep Backend RBAC Authoritative for Tenant Metadata

Decision: support owner-only backend membership administration for role changes and member removal, with audit events and lockout protection.

Rationale:
- The SaaS MVP needs Owner, Manager, and Viewer roles to be manageable from the tenant admin workflow, not only assigned at invite time.
- Backend authorization remains the enforcement point for tenant metadata, uploads, runs, reports, and settings.
- Role changes and removals need an auditable trail for ISO 50001 governance and pilot support.

Implications:
- `PATCH/DELETE /organizations/{organization_id}/memberships/{member_user_id}` require `MANAGE_MEMBERS`.
- The backend rejects self-removal and self-role changes that would risk owner lockout, and it rejects removal or demotion of the last owner.
- Membership role changes create `membership_role_updated` audit events; removals create `membership_removed` audit events.
- Removing a member transfers any report schedules they created in the tenant to the acting owner before the membership is deleted. If a transferred schedule was addressed to the removed member, the recipient is reassigned to the acting owner; audit metadata records counts only and does not copy recipient emails.
- In production Clerk mode, Clerk webhooks still synchronize provider-side organization memberships. A future provider write-back integration should mirror backend-initiated role/member changes into Clerk or move the UI action to Clerk-managed flows.

## ADR-015: Model Tenant Subscriptions Before Payment Provider Reconciliation

Decision: store subscription plans and each organization's current subscription in backend metadata, with default trial creation during tenant onboarding, owner-only plan updates, and provider webhook reconciliation.

Rationale:
- The SaaS plan requires subscription tiers and trial management, and the tenant metadata model must remain authoritative even when a payment provider drives lifecycle changes.
- Tenant plan state belongs behind the same RBAC and audit boundaries as other organization metadata.
- A local metadata model lets the frontend display plan status and usage during pilots without hard-coding tier assumptions in React.
- Backend enforcement is needed before checkout so pilot tenants cannot accidentally exceed trial/team boundaries through API calls.

Implications:
- `subscription_plans` seeds the `trial`, `team`, and `enterprise` tiers.
- `organization_subscriptions` stores one subscription row per tenant and is initialized by local onboarding, Clerk tenant sync, or Clerk organization webhooks.
- `GET /organizations/{organization_id}/subscription` is tenant-guarded; `PATCH /organizations/{organization_id}/subscription` requires owner billing permission and creates `subscription_updated` audit events.
- Entitlement helpers block new invite seats, invite acceptance, sites, and run creation when the tenant status or plan limits do not allow the action.
- Payment-provider checkout sessions, portal sessions, idempotent billing webhook records, checkout completion, subscription updates, and invoice-failure dunning reconciliation now exist at the backend boundary.

## ADR-016: Use a Provider Boundary for Billing Checkout and Webhooks

Decision: route billing checkout, portal, and webhook work through a backend billing provider boundary with a local development adapter and a Stripe-shaped production adapter.

Rationale:
- Billing has to be tenant/RBAC scoped, auditable, and idempotent before the app can safely serve multiple paying clients.
- Provider webhooks are the reliable source for subscription lifecycle, dunning, and customer/subscription ids.
- A local adapter keeps tests and pilot demos deterministic while production readiness can reject local-only billing.

Implications:
- `POST /organizations/{organization_id}/billing/checkout` and `/billing/portal` require owner billing permission and create audit events.
- `POST /webhooks/billing` records provider events in `billing_events` before applying subscription lifecycle changes, so duplicate events are ignored.
- `invoice.payment_failed` moves the tenant to `past_due`, which blocks new invite seats, invite acceptance, sites, and analysis runs through entitlement checks.
- Production readiness fails when `SAAS_BILLING_PROVIDER=local`, when Stripe mode lacks `STRIPE_SECRET_KEY`, `SAAS_STRIPE_PRICE_ID_TEAM`, or `SAAS_BILLING_WEBHOOK_SECRET`, when the Stripe webhook replay tolerance is invalid, or when live billing operations evidence is missing, insecure, future-dated, timezone-less, or stale.
- Stripe webhook verification validates the signed timestamp against `SAAS_BILLING_WEBHOOK_TOLERANCE_SECONDS` before applying subscription changes.
- Live billing operations evidence is configured with `SAAS_BILLING_OPERATIONS_URL`, `SAAS_BILLING_OPERATIONS_VERIFIED_AT`, and `SAAS_BILLING_OPERATIONS_MAX_AGE_DAYS`, which defaults to 30 days.
- Live Stripe dashboard setup, replay execution, invoice emails, tax/accounting settings, and operational dunning playbooks still need to be performed and recorded before pilot billing is fully production-ready.

## ADR-017: Require Trusted Origins and API Security Headers

Decision: configure API CORS from `SAAS_ALLOWED_ORIGINS`, add baseline hardening headers to every FastAPI response, apply baseline hardening headers plus report-only CSP from the Next.js edge configuration, collect sanitized CSP violation reports through the API, and fail production readiness when trusted HTTPS origins or HSTS are not configured.

Rationale:
- The SaaS frontend and API may be deployed on separate origins, so browser access must be explicit and tenant-safe.
- Wildcard CORS is not acceptable for a multi-client platform with authenticated tenant data.
- Security headers provide a low-cost guardrail while deeper live infrastructure review is still pending.

Implications:
- Development defaults to localhost frontend origins; production must set explicit HTTPS origins.
- FastAPI responses include `nosniff`, frame denial, strict referrer policy, a restrictive permissions policy, and same-site resource policy.
- Next.js responses include matching frontend hardening headers plus `Content-Security-Policy-Report-Only`; the initial CSP allows the configured API origin and Clerk origins needed by authenticated tenant flows.
- When a configured API origin exists, CSP reports are posted to `POST /security/csp-reports`. The collector stores only disposition, sanitized directive bucket, blocked-URI category, document origin, and status code, then exposes aggregate metrics by finite disposition/directive labels without raw URLs, referrers, script samples, tenant ids, user ids, or session details.
- Support operators can inspect aggregate counts through `GET /support/csp-reports/summary`, recent sanitized records through `GET /support/csp-reports`, and apply retention through `POST /support/csp-reports/retention/apply`, protected by `SAAS_SUPPORT_ACCESS_TOKEN`, so report-only CSP can be reviewed before enforcement without introducing tenant-facing CSP report APIs or unbounded operational storage.
- Production responses include HSTS using `SAAS_HSTS_MAX_AGE_SECONDS`, which must be positive in production readiness.
- Live edge/proxy verification, CSP report review before enforcement, penetration testing, and tenant isolation review remain required before pilot readiness.

## ADR-018: Gate Production on Backup and Alerting Evidence

Decision: add production readiness checks for backup policy documentation, recent backup-job evidence, recent restore-test evidence, alert delivery configuration, deployed alert-rule evidence, notification outbox alert thresholds, and the incident runbook.

Rationale:
- A multi-client SaaS pilot needs a recoverable metadata database and tenant artifact store before customer data is trusted to the platform.
- Backup policy documents the expected shape, backup-job evidence proves the scheduled job has recently succeeded, and restore-test evidence proves the recovery path has been exercised.
- Alerting must have a delivery destination, deployed rule evidence, and a runbook so incidents are detected and routed to an accountable response path.

Implications:
- `/ready` includes `backups` and `alerting` checks.
- Production requires `SAAS_BACKUP_POLICY_URL`, `SAAS_BACKUP_JOB_URL`, `SAAS_BACKUP_JOB_VERIFIED_AT`, `SAAS_BACKUP_RESTORE_TESTED_AT`, `SAAS_ALERT_WEBHOOK_URL`, `SAAS_ALERT_RUNBOOK_URL`, `SAAS_ALERT_RULES_URL`, `SAAS_ALERT_RULES_VERIFIED_AT`, `SAAS_ALERT_NOTIFICATION_DUE_AGE_SECONDS`, `SAAS_ALERT_NOTIFICATION_FAILED_COUNT`, `SAAS_ALERT_REPORT_SCHEDULE_DUE_AGE_SECONDS`, and `SAAS_ALERT_REPORT_SCHEDULE_FAILED_COUNT`.
- `SAAS_BACKUP_MAX_JOB_AGE_DAYS` defaults to 7 and rejects stale backup-job evidence; `SAAS_BACKUP_MAX_RESTORE_TEST_AGE_DAYS` defaults to 90 and rejects stale restore-test evidence.
- `SAAS_ALERT_RULES_MAX_AGE_DAYS` defaults to 30 and rejects stale alert-rule deployment evidence.
- `POST /support/alert-test` provides a support-token-protected generic alert delivery smoke test that includes request-id correlation without tenant identifiers and records platform operation evidence for success/failure outcomes.
- Worker batches that attempt due report schedules record `rds.report_schedule_dispatch` platform operation evidence with scope and count fields only, without organization ids, worker ids, schedule ids, report ids, recipient emails, or delivery error text.
- `GET /support/operation-events` exposes platform operation evidence to support operators using finite event-type and status filters.
- Production readiness requires a recent successful alert-smoke event, with max age controlled by `SAAS_ALERT_DELIVERY_TEST_MAX_AGE_DAYS`.
- These checks prove operational configuration, recent backup-job evidence, restore-test evidence, alert-rule evidence, and recent alert-smoke evidence are present, but live backup provisioning, restore execution, on-call ownership, and incident exercises remain required before pilot launch.

## ADR-019: Enforce Report Retention and Delete Artifacts Through Tenant Store

Decision: compute `expires_at` for generated reports from the tenant subscription plan, hide expired reports from normal tenant reads, and delete expired report artifacts plus locked evidence package artifacts through the tenant artifact-store boundary when retention is applied.

Rationale:
- ISO 50001 evidence should have explicit retention metadata per tenant plan.
- Metadata enforcement can be tested locally and protects report reads even before cloud object lifecycle policies are live.
- Artifact deletion must go through the same tenant-key validation used for reads and writes so retention work cannot cross tenant storage boundaries.
- Provider-side lifecycle policies can still provide a defense-in-depth cleanup path, but API-driven retention application gives operators an auditable control before pilot rollout.

Implications:
- Reports now store `expires_at` and `retention_applied_at`.
- `GET /organizations/{organization_id}/reports` and report artifact reads exclude expired or retention-applied reports.
- `POST /organizations/{organization_id}/reports/retention/apply` deletes expired report artifacts and locked evidence package artifacts, removes locked evidence package metadata for retained reports, marks expired reports unpublished, returns deleted/missing artifact counts including evidence-package cleanup counts, and creates `report_retention_applied` audit events.
- Production object-storage lifecycle policies, backup retention, and restore behavior must still be configured and tested for defense-in-depth cleanup.

## ADR-020: Store Tenant Privacy Settings in Metadata

Decision: add tenant-scoped privacy settings to the metadata store, expose them through owner-managed API routes, and audit changes without copying contact details into audit metadata.

Rationale:
- Multi-client pilots need an explicit record of data-processing acknowledgement before tenant data is trusted to the platform.
- Privacy preferences should sit behind the same tenant/RBAC boundary as profile, billing, and analysis settings.
- Audit events must prove settings changed without unnecessarily spreading personal contact data through the audit log.

Implications:
- `organization_privacy_settings` stores acknowledgement timestamp, acknowledging user, privacy contact, subprocessor notification preference, and anonymized analytics sharing preference.
- `GET /organizations/{organization_id}/privacy-settings` is tenant-guarded; `PATCH /organizations/{organization_id}/privacy-settings` requires organization-management permission.
- Privacy changes create `privacy_settings_updated` audit events with preference flags only.
- Live legal review, DPA language, privacy notice publication, subprocessor list publication, and privacy-request fulfillment operations remain required before production pilot readiness.

## ADR-021: Track Privacy Requests Before Automating Fulfillment

Decision: add owner-managed, tenant-scoped privacy request records for data export, deletion, and correction workflows, and audit lifecycle changes without copying request text into audit metadata.

Rationale:
- The SaaS plan requires broader privacy operations beyond acknowledgement settings before pilot tenants trust the platform with multi-client data.
- Data deletion and export fulfillment can have legal and operational consequences, so the MVP should first create a controlled tracking workflow rather than automatically deleting tenant data.
- Request records need to be tenant-scoped and owner-managed so cross-tenant subject references cannot be introduced through the admin surface.
- Audit metadata should prove the request changed while avoiding duplication of sensitive descriptions or resolution notes.

Implications:
- `organization_privacy_requests` stores request type, status, requester, optional same-tenant subject user, description, resolution notes, timestamps, and completion time.
- `GET/POST /organizations/{organization_id}/privacy-requests` and `PATCH /organizations/{organization_id}/privacy-requests/{privacy_request_id}` require organization-management permission.
- The API rejects subject users that are not members of the same organization and rejects cross-tenant request updates through resource-scope checks.
- Request create/update actions write `privacy_request_created` and `privacy_request_updated` audit events with status flags only.
- Automated export packages, deletion execution, legal approval workflow, retention backups, and provider-side lifecycle verification remain future work before pilot readiness.

## ADR-022: Track Tenant Pilot Readiness Without Replacing External Verification

Decision: add an owner-managed pilot readiness checklist per tenant for launch evidence, with known checklist keys, status tracking, evidence links, notes, and sanitized audit events.

Rationale:
- The SaaS roadmap calls for a pilot onboarding toolkit and support playbook, but several readiness gates depend on live providers and legal/operations sign-off outside the local codebase.
- Tenant owners and operators still need a tenant-scoped place to track whether launch gates are pending, in progress, verified, or blocked.
- A controlled checklist lets the app surface readiness progress without turning unverified external setup into a false green production signal.
- Audit events should prove a readiness gate changed without spreading evidence URLs or internal notes through the audit log.

Implications:
- `organization_pilot_readiness_items` stores one row per tenant checklist gate when a gate is updated.
- `GET /organizations/{organization_id}/pilot-readiness` and `PATCH /organizations/{organization_id}/pilot-readiness/{item_key}` require organization-management permission.
- The API only accepts known gates for DPA, privacy notice, production auth, production billing, email delivery, storage/backups, tenant isolation review, and support ownership.
- Checklist updates write `pilot_readiness_updated` audit events with item key, previous status, new status, and evidence/notes presence flags only.
- Production readiness still depends on live Clerk, Stripe, SMTP, object storage, backup/restore, monitoring, legal, security, and support verification outside this metadata checklist.

## ADR-023: Emit Tenant-Safe API Request Logs

Decision: add API request logging middleware that propagates `X-Request-Id` and W3C `traceparent`, then emits one structured JSON log event per request using route templates instead of raw request paths.

Rationale:
- The SaaS roadmap requires observability before a controlled multi-client pilot.
- Operators need request correlation for API, frontend, and incident investigations even before a full tracing stack is deployed.
- Raw paths, query strings, request bodies, headers, user ids, organization ids, and storage keys can carry tenant-sensitive data and should not be logged by default.
- Route templates preserve enough context to identify the code path while keeping tenant identifiers out of operational logs.

Implications:
- Every API response includes `X-Request-Id` and `traceparent`; callers may supply safe values or the API will generate them.
- The `rds.api` logger emits JSON with `event`, `request_id`, `trace_id`, `span_id`, `method`, `route`, `status_code`, and `duration_ms`.
- CORS allows and exposes `X-Request-Id` and `traceparent` so browser clients can correlate support reports with API logs and tracing tools.
- This remains request-log scaffolding. Production readiness separately requires configured log aggregation, retention evidence, deployed alert-rule evidence, and live incident-review exercises.

## ADR-024: Scope Report Comments to Visible Tenant Reports

Decision: store report review comments as tenant-scoped metadata records under report subresource routes, allow all member roles to add comments, and gate reads/writes through the same tenant and visibility checks used for report artifacts.

Rationale:
- ISO 50001 report review needs lightweight collaboration alongside the scheduling, approval, and export surfaces.
- Viewers should be able to participate in evidence review without gaining upload, run, report-management, member, billing, or organization-management permissions.
- Comments must not revive access to unpublished, expired, or retention-applied reports.
- Audit events should prove collaboration happened without duplicating comment text into the audit log.

Implications:
- `organization_report_comments` stores comment body, author, tenant, report, and created timestamp.
- `GET/POST /organizations/{organization_id}/reports/{report_id}/comments` require tenant membership, report scope, and report visibility.
- `comment_reports` is a narrow RBAC action granted to Owner, Manager, and Viewer roles.
- `report_comment_created` audit events include report id, report type, and body length only.
- Branded scheduled packages, richer export packages, multi-step approval routing, and comment moderation remain separate workflow additions.

## ADR-025: Proxy Tenant Report Downloads Through the SaaS Frontend

Decision: expose tenant-guarded backend report export routes for JSON artifacts, dynamic JSON/PDF evidence packages, and locked evidence package snapshots, and proxy browser downloads through Next.js route handlers that read the current app session server-side.

Rationale:
- The SaaS MVP requires users to download report artifacts and ISO 50001 evidence packages without bypassing tenant checks.
- ISO 50001 evidence packs need an auditable point-in-time archive instead of only a regenerated current-state download.
- Local and Clerk sessions use HTTP-only cookies or server-held tokens in the frontend, so a direct browser link to FastAPI would not reliably carry backend authorization.
- Export reads must honor the same tenant, publication, expiry, and retention checks as artifact views.
- Audit metadata should distinguish downloads from ordinary views without copying artifact content, storage keys, review notes, comments, or delivery recipients.

Implications:
- `GET /organizations/{organization_id}/reports/{report_id}/export` returns an `application/json` attachment for visible report artifacts.
- `GET /organizations/{organization_id}/reports/{report_id}/evidence-package` returns a JSON evidence package with organization profile, report branding, template metadata, and the evidence sections enabled by the tenant template.
- `GET /organizations/{organization_id}/reports/{report_id}/evidence-package/pdf` returns an app-generated tenant-branded PDF summary rendered from the same current evidence package source, saved report-branding settings, and tenant evidence section template.
- `GET/POST /organizations/{organization_id}/reports/{report_id}/evidence-packages` lists and locks tenant-scoped JSON evidence package snapshots. Managers create snapshots only after pending approval assignments are resolved, the latest management review is approved, and the tenant approval policy's minimum approved-assignment threshold is met; viewers can list snapshots for visible reports. Completing the final pending assignment as approved now creates the same locked JSON snapshot automatically through the tenant artifact store when the tenant policy enables auto-locking.
- `GET /organizations/{organization_id}/reports/{report_id}/evidence-packages/{package_id}/download` returns a locked package from tenant object storage.
- `GET /reports/{reportId}/download` in the Next.js app bridges the authenticated browser request to the backend export endpoint.
- `GET /reports/{reportId}/evidence-package` bridges authenticated browser requests to the backend evidence-package endpoint.
- `GET /reports/{reportId}/evidence-package/pdf` bridges authenticated browser requests to the backend PDF evidence-package endpoint without exposing the app session to client JavaScript.
- `GET /reports/{reportId}/evidence-packages/{packageId}/download` bridges authenticated browser requests to locked evidence-package snapshots.
- Artifact downloads create `report_exported` audit events with report id, report type, format, and content length only.
- Evidence-package downloads create `report_evidence_exported` audit events with report id, report type, package type, format, content length, enabled-template-section count, and included evidence section counts only.
- Locked evidence-package creation creates `report_evidence_package_created` audit events with report/package ids, report type, package type, format, content length, approval status/review id, approval counts/policy values, and section counts only. Locked evidence-package downloads create `report_evidence_package_downloaded` audit events with report/package ids, report type, package type, format, content length, and section counts only.
- Fully custom rendered template layouts, scheduled attachments, deeper multi-step approval chains, and escalation routing remain future work.

## ADR-026: Process Report Delivery Schedules Through the Background Worker

Decision: add tenant-scoped report schedule metadata and process due schedules through the background worker as authenticated-access email notifications.

Rationale:
- The SaaS roadmap calls for report scheduling, but automated delivery has provider, retry, notification, and audit implications that should stay behind explicit service boundaries.
- A metadata-backed contract lets tenants configure intended delivery cadence while keeping report visibility, retention, and RBAC checks at the API boundary.
- Owners and managers should control scheduled delivery; viewers can read schedules for visible reports without managing them.
- Audit records need to prove schedule changes and delivery attempts without spreading recipient email addresses, storage keys, or artifact content through audit metadata.

Implications:
- `organization_report_schedules` stores tenant, report, creator, frequency, recipient, next run time, active state, timestamps, and last-delivery state.
- `organization_report_schedule_deliveries` records each worker delivery attempt with tenant, schedule, report, recipient, status, error, and timestamps.
- `GET/POST /organizations/{organization_id}/reports/{report_id}/schedules` and `PATCH /organizations/{organization_id}/reports/{report_id}/schedules/{schedule_id}` gate schedule operations through report visibility and RBAC.
- `GET /organizations/{organization_id}/reports/{report_id}/schedules/{schedule_id}/deliveries` exposes recorded delivery attempts through the same tenant/report visibility guard.
- Schedule creation/update writes `report_schedule_created` and `report_schedule_updated` audit events with frequency, state, report, and timing metadata only.
- Member removal transfers schedules owned by the removed member to the acting owner within the same tenant and reassigns matching removed-member recipients, so active delivery does not remain tied to an inactive member.
- The report detail page surfaces schedule listing, creation, pause controls, last-delivery state, and recent delivery attempts.
- The worker validates due report artifacts through the tenant artifact-store boundary, sends email notifications through the configured sender, advances the next run time, and writes `report_schedule_delivered` or `report_schedule_failed` audit events.
- The worker records sanitized scheduled-dispatch platform operation evidence for batches that attempt due report schedules, and `/metrics` exposes aggregate scheduled-dispatch success/failure plus latest-success age without tenant labels.
- Provider retry policy, branded packages, attachment delivery, live provider replay validation, and external provider lifecycle evidence remain future work.

## ADR-027: Record Tenant Report Review Decisions

Decision: store report approval decisions and approval assignments as tenant-scoped records under report subresource routes, allow Owners and Managers to route or record approvals, allow all tenant members to read assignment and decision history for visible reports, and audit workflow actions without duplicating note text.

Rationale:
- ISO 50001 evidence workflows need a clear management review state beyond free-form comments.
- Approval decisions should be visible to Viewers without giving Viewers report-management power.
- Named approver assignment gives pilot tenants a lightweight routing workflow before deeper approval automation exists.
- Review decisions must not revive access to unpublished, expired, or retention-applied reports.
- Audit records should prove the status decision and assignment state without copying optional management notes into audit metadata.

Implications:
- `organization_report_reviews` stores tenant, report, reviewer, status, optional note, and created timestamp.
- `organization_report_approval_assignments` stores tenant, report, assigned approver, assigner, pending/final status, optional due date, optional note, completion timestamp, and timestamps.
- `GET /organizations/{organization_id}/report-approvers` exposes active Owner/Manager candidates to users with report-management permission.
- `GET/POST/PATCH /organizations/{organization_id}/reports/{report_id}/approval-assignments...` require tenant membership, report scope, and report visibility; writes require `manage_reports`, and assignment creation validates that the assignee is an active Owner or Manager.
- Assignment creation queues `report_approval_assigned`; completion or cancellation queues `report_approval_updated`. Notification bodies include report and assignment identifiers/status only, not assignment notes.
- Completing an assignment as `approved` or `changes_requested` also creates a report review decision so existing review history, evidence packages, and review status remain coherent.
- When the final pending assignment is completed as `approved`, the backend automatically creates a locked evidence-package snapshot with sanitized audit metadata when the tenant approval policy enables auto-locking and its approved-assignment threshold is satisfied, so the approval trail can immediately produce point-in-time ISO 50001 evidence without making that behavior mandatory for every tenant.
- `GET/POST /organizations/{organization_id}/reports/{report_id}/reviews` require tenant membership, report scope, and report visibility; writes require `manage_reports`.
- Supported initial statuses are `approved` and `changes_requested`.
- `report_approval_assigned` and `report_approval_updated` audit events include report id/type, assignment status, assignee user id, due date, generated review id, and note length only.
- `report_review_recorded` audit events include report id, report type, review status, and note length only.
- The report detail page surfaces current review status, approval assignments, review history, Owner/Manager assignment controls, and direct decision controls.
- `organization_report_approval_policies` stores tenant-level evidence-lock behavior: whether final approved assignment completion should auto-lock evidence and how many approved assignments are required before a lock is allowed.
- `GET/PATCH /organizations/{organization_id}/report-approval-policy` lets tenant members read the policy and tenant owners update it through the Settings page.
- `report_approval_policy_updated` audit events include only the auto-lock flag and minimum approved-assignment threshold.
- Configurable multi-step approval chains and escalation routing remain future work.

## ADR-028: Store Tenant Evidence Package Branding in Metadata

Decision: store lightweight report/evidence-package branding and section template settings as tenant metadata and apply them when dynamic evidence packages are generated.

Rationale:
- Multi-client evidence packs need client-specific presentation without duplicating report artifacts or weakening tenant isolation.
- Branding and evidence section choices should be managed by tenant administrators, not hard-coded in shared report rendering code.
- Audit records should prove that branding or template settings changed without copying optional logo URLs, support contacts, template names, or package content into audit metadata.

Implications:
- `organization_report_branding` stores the tenant evidence pack title, primary PDF accent color, optional logo/reference URL, optional support email, and timestamps.
- `GET /organizations/{organization_id}/report-branding` is tenant-guarded; `PATCH /organizations/{organization_id}/report-branding` requires organization-management permission.
- `organization_evidence_package_templates` stores the tenant evidence template name plus section flags, client-facing labels, and render order for report artifact, upload lineage, approval assignments, review decisions, comments, and schedules.
- `GET /organizations/{organization_id}/evidence-package-template` is tenant-guarded; `PATCH /organizations/{organization_id}/evidence-package-template` requires organization-management permission.
- Evidence package JSON includes a `branding` block so PDF rendering and future templates use the same source data.
- Evidence package JSON includes a `template` block with labels/order and omits disabled sections from dynamic JSON/PDF exports and locked snapshots.
- Dynamic evidence-package PDFs use the saved title, accent color, enabled evidence sections, section labels, and render order immediately; logo rendering, attachment composition, and fully custom layout rendering remain future template-renderer concerns.
- `report_branding_updated` audit events include title-configured, logo-configured, support-email-configured, and primary-color metadata only.
- `report_evidence_template_updated` audit events include section flags, enabled-section count, and configured/customized flags for template name, labels, and order only.

## ADR-029: Gate Production on Security and Tenant-Isolation Review Evidence

Decision: require production `/ready` to validate fresh external security-review evidence through `SAAS_SECURITY_REVIEW_URL`, `SAAS_SECURITY_REVIEWED_AT`, and `SAAS_SECURITY_REVIEW_MAX_AGE_DAYS`, plus fresh tenant-isolation review evidence through `SAAS_TENANT_ISOLATION_REVIEW_URL`, `SAAS_TENANT_ISOLATION_REVIEWED_AT`, and `SAAS_TENANT_ISOLATION_REVIEW_MAX_AGE_DAYS`.

Rationale:
- The SaaS roadmap requires a live security review and threat model before a multi-client pilot.
- Tenant isolation is the core multi-client invariant, so it needs a distinct production readiness signal instead of being hidden inside a generic security checklist.
- Readiness gates prevent pilot deployments from appearing healthy when the reviews have not been completed or have gone stale.
- The app should track evidence freshness without storing sensitive review findings, penetration-test details, or internal remediation notes.

Implications:
- Production readiness requires HTTPS evidence URLs and ISO-8601 signed-off timestamps with timezones for both review gates.
- `SAAS_SECURITY_REVIEW_MAX_AGE_DAYS` and `SAAS_TENANT_ISOLATION_REVIEW_MAX_AGE_DAYS` default to `90`, and stale, future, missing, or timezone-ambiguous evidence fails readiness.
- Development can run without review evidence, but configured evidence is still syntax-validated.
- The tenant-isolation review should cover API, database, object storage, background jobs, scheduled notifications, audit exports, metrics, logs, and support-only views across at least two pilot-shaped tenants.
- Live dependency review, secret handling review, edge/proxy verification, CSP report review before enforcement, penetration testing, tenant isolation review execution, and deployment hardening remain external pilot prerequisites.

## ADR-030: Record Tenant-Safe Provider Replay Evidence

Decision: record successful Clerk, billing, and notification-provider webhook processing as finite `platform_operation_events` and require recent successful replay evidence in production readiness.

Rationale:
- Clerk, Stripe, and notification-provider callbacks are external control-plane dependencies for multi-client onboarding, billing, and delivery assurance.
- Signature configuration alone does not prove that live provider replay reaches the deployed API and mutates the expected backend records.
- Replay evidence must help operators prove provider wiring without spreading tenant identifiers, user ids, provider event ids, customer ids, subscription ids, notification ids, recipient emails, or payload content into support surfaces or metrics.

Implications:
- `POST /webhooks/clerk` records `rds.clerk_webhook_replay` after a signed event is verified and processed.
- `POST /webhooks/billing` records `rds.billing_webhook_replay` after a billing webhook is verified, stored, and reconciled or idempotently ignored.
- `POST /webhooks/notifications/provider` records `rds.notification_provider_webhook_replay` after a signed provider delivery callback is verified and processed.
- `SAAS_PROVIDER_WEBHOOK_REPLAY_MAX_AGE_DAYS` defaults to `30`; production `/ready` fails when successful replay evidence for any provider family is missing, stale, future-dated, or unreadable.
- `/metrics` reports only finite provider replay operation type/status counts. `/support/operation-events` exposes only event id, finite type/status, request id, sanitized generic detail, and timestamp.
- Live provider dashboard setup, webhook endpoint registration, replay execution, and operational dunning/bounce handling remain required before pilot launch.

## ADR-031: Record Tenant-Safe Artifact Store Smoke Evidence

Decision: add a support-protected artifact-store smoke endpoint that writes, reads, and deletes a tiny synthetic object through the configured tenant artifact-store boundary, records `rds.artifact_store_test` operation evidence, and requires recent successful evidence in production readiness.

Rationale:
- Multi-client pilots depend on object storage for raw tenant uploads, generated report artifacts, and locked evidence packages.
- A configured S3 bucket name does not prove the deployed API can write, read, and delete objects with the active credentials and network policy.
- Operators need a repeatable validation step before launch, but support surfaces and metrics must not expose tenant ids, probe storage keys, bucket keys, payload content, or provider error details.

Implications:
- `POST /support/artifact-store-test` requires `SAAS_SUPPORT_ACCESS_TOKEN`, runs against the app's configured artifact store, and returns only event id, event type, timestamp, and byte count.
- The probe uses a synthetic platform namespace, deletes the object in the same request, and stores only generic success byte counts or generic failure phase details.
- `SAAS_ARTIFACT_STORE_TEST_MAX_AGE_DAYS` defaults to `30`; production `/ready` fails when successful smoke evidence is missing, stale, future-dated, or unreadable.
- `/metrics` reports artifact-store smoke-test operation success/failure counts and latest-success freshness without request ids or details.
- Live bucket provisioning, IAM/access-policy review, backup/versioning configuration, provider lifecycle policy validation, and support-runbook execution remain external pilot prerequisites.

## ADR-032: Record Tenant-Safe Database Smoke Evidence

Decision: add a support-protected metadata database smoke endpoint that writes and reads a finite `rds.database_test` operation event through the configured store, then require recent successful evidence in production readiness.

Rationale:
- Multi-client pilots depend on PostgreSQL for tenant identity, RBAC, uploads, runs, reports, subscriptions, notifications, audit events, and operational evidence.
- A PostgreSQL-shaped URL and applied migrations prove configuration shape, but they do not give operators a repeatable support action that verifies deployed write/read behavior after networking, credentials, pooling, or migration changes.
- Smoke evidence must demonstrate the metadata boundary is writable without exposing database URLs, credentials, SQL text, tenant ids, user ids, provider errors, or row contents in support surfaces or metrics.

Implications:
- `POST /support/database-test` requires `SAAS_SUPPORT_ACCESS_TOKEN`, writes and reads a platform operation event, and returns only event id, event type, and timestamp.
- `SAAS_DATABASE_TEST_MAX_AGE_DAYS` defaults to `30`; production `/ready` fails when successful database smoke evidence is missing, stale, future-dated, or unreadable.
- `/metrics` reports database smoke-test operation success/failure counts and latest-success freshness without request ids or details.
- Live PostgreSQL provisioning, connection-pool sizing, backup jobs, restore drills, migration execution against production, and operational log/monitoring integration remain external pilot prerequisites.

## ADR-033: Gate Production on Log Aggregation Evidence

Decision: require production `/ready` to validate log aggregation evidence through `SAAS_LOG_AGGREGATION_URL`, `SAAS_LOG_AGGREGATION_VERIFIED_AT`, `SAAS_LOG_AGGREGATION_MAX_AGE_DAYS`, and `SAAS_LOG_RETENTION_DAYS`.

Rationale:
- Multi-client pilots need request-log correlation for support, incident response, and tenant-safety reviews.
- Emitting structured logs is not enough; operators need proof that deployed logs are reaching an aggregation system with a bounded retention policy.
- The readiness gate should remain vendor-neutral while preventing pilot deployments from appearing ready when logs are only written to local process output.
- Evidence must not store tenant ids, user ids, raw paths, query strings, request bodies, authorization headers, or log payloads inside app metadata.

Implications:
- Production readiness requires an HTTPS aggregation dashboard, saved search, or runbook URL, an ISO-8601 verification timestamp with a timezone, and a positive retention window no greater than `3650` days.
- `SAAS_LOG_AGGREGATION_MAX_AGE_DAYS` defaults to `30`; stale, future-dated, missing, timezone-ambiguous, or malformed evidence fails readiness in production.
- Development can run without log aggregation evidence, but configured values are still syntax-validated.
- Live log pipeline deployment, provider access controls, alert routing from aggregated logs, and incident-review exercises remain external pilot prerequisites.

## ADR-034: Gate Production on Worker Supervisor Evidence

Decision: require production `/ready` to validate external worker supervision evidence through `SAAS_WORKER_SUPERVISOR_URL`, `SAAS_WORKER_SUPERVISOR_VERIFIED_AT`, and `SAAS_WORKER_SUPERVISOR_MAX_AGE_DAYS`.

Rationale:
- The SaaS worker drains queued analyses, notifications, and scheduled report deliveries for all tenants, so it must be operated as a supervised long-running service.
- A recent heartbeat proves the worker has checked in, but it does not prove restart policy, crash visibility, or alert routing are configured.
- The readiness gate should remain deployment-platform neutral while preventing a multi-client pilot from relying on an unsupervised shell session or one-off worker run.
- Evidence should reference external operational proof without storing tenant ids, worker raw ids, schedule ids, report ids, recipient emails, logs, or process output inside tenant metadata.

Implications:
- Production readiness now reports `worker_supervision` separately from worker heartbeat liveness.
- Production requires an HTTPS supervisor dashboard, process-manager service, or runbook URL and an ISO-8601 verification timestamp with a timezone.
- `SAAS_WORKER_SUPERVISOR_MAX_AGE_DAYS` defaults to `30`; stale, future-dated, missing, timezone-ambiguous, or malformed evidence fails readiness in production.
- Development can run without worker supervisor evidence, but configured values are still syntax-validated.
- Live supervisor deployment, restart-policy testing, worker failure alerting, and incident drills remain external pilot prerequisites.

## ADR-035: Record Tenant-Safe Tenant Isolation Smoke Evidence

Decision: add a support-protected tenant-isolation smoke endpoint that creates synthetic tenant A/B metadata inside a rollback-only transaction, verifies tenant-scoped list boundaries, records `rds.tenant_isolation_test` operation evidence, and requires recent successful evidence in production readiness.

Rationale:
- Multi-client pilots require more than static code tests; operators need a repeatable deployed check that the live metadata store and service methods still isolate tenants after migrations, configuration changes, and deployment moves.
- The smoke test should exercise user organization membership, site, meter, upload, and cross-tenant site-filter boundaries without leaving synthetic organizations behind.
- Evidence must prove the check ran while avoiding tenant ids, user ids, synthetic probe ids, emails, storage keys, database URLs, or row contents in support surfaces and metrics.

Implications:
- `POST /support/tenant-isolation-test` requires `SAAS_SUPPORT_ACCESS_TOKEN`, rolls back synthetic tenant records before returning, and stores only event id, type, timestamp, checks passed, synthetic record count, request id, and generic failure detail.
- `SAAS_TENANT_ISOLATION_TEST_MAX_AGE_DAYS` defaults to `30`; production `/ready` fails when successful tenant-isolation smoke evidence is missing, stale, future-dated, or unreadable.
- `/metrics` reports tenant-isolation smoke-test operation success/failure counts and latest-success freshness without request ids or details.
- The smoke test complements, but does not replace, the external tenant-isolation review that covers API routes, object storage, background workers, notifications, exports, logs, metrics, support-only views, and at least two pilot-shaped tenants.

## ADR-036: Record Tenant-Safe Run Lifecycle Smoke Evidence

Decision: add a support-protected run-lifecycle smoke endpoint that creates a synthetic tenant, writes a tiny energy CSV through the configured artifact store, executes the stored-upload analysis path, verifies run/job/report metadata plus report artifact content, deletes synthetic metadata and objects, records `rds.run_lifecycle_smoke_test` operation evidence, and requires recent successful evidence in production readiness.

Rationale:
- Milestone 2 depends on the upload-to-run-to-report path, not only metadata isolation and storage reachability.
- Operators need a repeatable deployed check that the API, metadata store, artifact store, subscription/run entitlement path, analysis config derivation, queued job metadata, analytics service, report artifact generation, and cleanup behavior still work together after migrations or deployment changes.
- Evidence must prove the lifecycle ran while avoiding tenant ids, user ids, synthetic probe ids, emails, storage keys, source content, report content, provider errors, or database details in support surfaces and metrics.

Implications:
- `POST /support/run-lifecycle-test` requires `SAAS_SUPPORT_ACCESS_TOKEN`, executes the synthetic lifecycle, removes synthetic rows and artifacts before returning, and stores only event id, type, timestamp, checks passed, artifact count, synthetic record count, request id, and generic failure detail.
- `SAAS_RUN_LIFECYCLE_TEST_MAX_AGE_DAYS` defaults to `30`; production `/ready` fails when successful run-lifecycle smoke evidence is missing, stale, future-dated, or unreadable.
- `/metrics` reports run-lifecycle smoke-test operation success/failure counts and latest-success freshness without request ids or details.
- The smoke test complements, but does not replace, live pilot tenant validation with real uploads, worker processing, report review, evidence packages, provider notifications, and support sign-off.

## ADR-037: Record Tenant-Safe Worker Run Lifecycle Smoke Evidence

Decision: add a support-protected worker run-lifecycle smoke endpoint that creates a synthetic tenant, writes a tiny energy CSV through the configured artifact store, queues a stored-upload analysis run, drains one queued job through the worker job processor, verifies run/job/report metadata plus report artifact content, deletes synthetic metadata and objects, records `rds.worker_run_lifecycle_smoke_test` operation evidence, and requires recent successful evidence in production readiness.

Rationale:
- Milestone 2 depends on queued background processing, not only inline API execution of the stored-upload lifecycle.
- Operators need a repeatable deployed check that queue creation, job claiming, worker processing, metadata updates, artifact writes, report generation, and cleanup behavior still work together after migrations or deployment changes.
- The smoke test should exercise the worker queue path without creating a synthetic worker heartbeat; heartbeat and supervisor readiness must continue to reflect the real long-running worker process.
- Evidence must prove the queued lifecycle ran while avoiding tenant ids, user ids, synthetic probe ids, emails, storage keys, source content, report content, provider errors, or database details in support surfaces and metrics.

Implications:
- `POST /support/worker-run-lifecycle-test` requires `SAAS_SUPPORT_ACCESS_TOKEN`, executes the synthetic queued lifecycle, removes synthetic rows and artifacts before returning, and stores only event id, type, timestamp, checks passed, job counts, artifact count, synthetic record count, request id, and generic failure detail.
- `SAAS_WORKER_RUN_LIFECYCLE_TEST_MAX_AGE_DAYS` defaults to `30`; production `/ready` fails when successful worker run-lifecycle smoke evidence is missing, stale, future-dated, or unreadable.
- `/metrics` reports worker run-lifecycle smoke-test operation success/failure counts and latest-success freshness without request ids or details.
- The smoke test complements, but does not replace, external worker supervision, recent all-tenant worker heartbeat evidence, live scheduled-report dispatch evidence, and pilot tenant validation with real uploads and notifications.

## ADR-038: Store Data Connector Registry Metadata Before Provider Execution

Decision: add a tenant-scoped data connector registry and sync-attempt history for API, SFTP, and utility-portal source metadata before implementing provider-specific sync execution.

Rationale:
- The SaaS roadmap calls for improved data connectors beyond CSV uploads, but production integrations need secret management, provider validation, scheduling, and import lifecycle controls.
- A metadata registry lets tenants model intended source systems, categories, cadence, ownership, due state, and sync outcomes now without storing provider secrets or weakening tenant boundaries.
- Connector setup and changes need an audit trail that proves governance activity without copying connector names, provider paths, credential references, or secrets into audit metadata.

Implications:
- `organization_data_connectors` stores connector id, organization, site, type, canonical source category, draft/active/paused status, cadence, optional external reference, optional credential reference, sync timestamps/status, creator/updater, and timestamps.
- `organization_data_connector_syncs` stores bounded sync-attempt history with tenant, connector, site, source category, finite status, optional same-site/same-category upload link, record count, sanitized status reason, trigger user, and timestamps.
- `POST /organizations/{organization_id}/data-connectors/{connector_id}/imports` lets a provider adapter or admin tool submit connector-supplied CSV content through the same tenant upload validation and artifact-storage contract used by manual uploads, then records a successful connector sync linked to the created upload.
- `backend.data_connector_sync.process_due_data_connectors` defines the provider-backed worker boundary: a runtime-supplied provider fetches content for active due connectors, the service stores valid content through the connector import path, and provider/import failures record sanitized failed sync attempts.
- `GET /organizations/{organization_id}/data-connectors` requires tenant read access; `GET /organizations/{organization_id}/data-connectors/due` provides a tenant-scoped due selection contract; `GET /organizations/{organization_id}/data-connectors/{connector_id}/syncs` lists sync attempts for tenant members; `POST/PATCH /organizations/{organization_id}/data-connectors...` require asset-management permission and tenant-scope both sites and connector ids.
- Sync attempt recording requires asset-management permission, requires an active connector, and validates any linked upload against the connector organization, site, and source category before updating last/next sync state. Connector imports additionally require upload-management permission and reject source content that fails the connector category contract before storing an artifact or upload row.
- Settings exposes a connector registry for Owners and Managers to create/update metadata while Viewers can inspect configured connectors and last-sync evidence.
- Worker heartbeat summaries include connector attempted/succeeded/failed/skipped counts so support can see connector activity without connector ids, organization ids, provider references, storage keys, or error details.
- `data_connector_created` and `data_connector_updated` audit metadata includes only finite connector type/category/status/schedule values and configured-reference flags.
- `data_connector_sync_recorded` audit metadata includes only finite connector type/category/schedule/status values, records processed, and configured upload/reason flags.
- Concrete API/SFTP/utility provider adapters, secret-manager adapters, provider-specific fetch/parse validation, deployment wiring for live scheduled connector syncs, and live provider monitoring remain future work.
