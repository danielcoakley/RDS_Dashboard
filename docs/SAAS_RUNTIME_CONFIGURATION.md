# SaaS Runtime Configuration

This file records the current runtime environment contract for the SaaS backend. Keep secrets out of the repo and inject them through the deployment environment.

## Runtime Mode

- `RDS_RUNTIME_ENV=development|production`
- `development` is the default for local tests and demo flows.
- `production` enables readiness gates that reject local-only metadata and artifact storage.
- Production readiness also rejects local-only billing configuration.

## API Security

- `SAAS_ALLOWED_ORIGINS` is a comma-separated list of trusted frontend origins for API CORS, for example `https://app.example.com`.
- Development defaults to `http://localhost:3000,http://127.0.0.1:3000` when `SAAS_ALLOWED_ORIGINS` is not set.
- Production readiness requires at least one explicit HTTPS origin and rejects wildcard origins.
- API responses include baseline hardening headers:
  - `X-Content-Type-Options: nosniff`
  - `X-Frame-Options: DENY`
  - `Referrer-Policy: no-referrer`
  - `Permissions-Policy: camera=(), microphone=(), geolocation=(), payment=()`
  - `Cross-Origin-Resource-Policy: same-site`
- `SAAS_HSTS_MAX_AGE_SECONDS` defaults to `31536000`; production readiness requires a positive value and production API responses include `Strict-Transport-Security`.
- `SAAS_SECURITY_REVIEW_URL` links to the latest external security-review evidence, such as a signed-off threat model, review ticket, or report.
- `SAAS_SECURITY_REVIEWED_AT` records when that review was completed as an ISO-8601 datetime with a timezone.
- `SAAS_SECURITY_REVIEW_MAX_AGE_DAYS` defaults to `90`; production readiness fails when the review evidence is missing, not HTTPS, timestamped without a timezone, in the future, or older than the configured max age.
- `SAAS_TENANT_ISOLATION_REVIEW_URL` links to the latest external tenant-isolation review evidence covering API, database, artifact storage, worker, logs, and support surfaces.
- `SAAS_TENANT_ISOLATION_REVIEWED_AT` records when that isolation review was completed as an ISO-8601 datetime with a timezone.
- `SAAS_TENANT_ISOLATION_REVIEW_MAX_AGE_DAYS` defaults to `90`; production readiness fails when tenant-isolation review evidence is missing, not HTTPS, timestamped without a timezone, in the future, or older than the configured max age.

## Frontend Security

- The Next.js frontend applies hardening headers from `frontend/next.config.ts` to all routes.
- Frontend responses include `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, `Permissions-Policy`, `Cross-Origin-Opener-Policy`, `Cross-Origin-Resource-Policy`, and `Strict-Transport-Security`, and the default `X-Powered-By` header is disabled.
- The frontend currently emits `Content-Security-Policy-Report-Only` so pilot deployments can observe violations before enforcing CSP. The policy allows the configured `NEXT_PUBLIC_API_BASE_URL` origin plus Clerk origins for authenticated mode.
- When `NEXT_PUBLIC_API_BASE_URL` is set, the frontend CSP includes `report-uri <api-origin>/security/csp-reports`. The public API endpoint accepts browser CSP JSON, stores only sanitized disposition/directive/blocked-URI-category/document-origin/status-code fields, and does not store source files, referrers, blocked URL hosts/paths, script samples, users, sessions, or tenant ids.
- Support operators can review sanitized CSP evidence through `GET /support/csp-reports/summary` for aggregate counts and `GET /support/csp-reports` for recent records, both protected by `SAAS_SUPPORT_ACCESS_TOKEN`. The list endpoint accepts optional `disposition`, `directive`, and `limit` query filters. Responses include only finite disposition/directive counts or stored sanitized fields and remain separate from tenant-facing APIs.
- `SAAS_CSP_REPORT_RETENTION_DAYS` controls sanitized CSP report retention and defaults to `30`. `POST /support/csp-reports/retention/apply` applies the policy and returns the cutoff, deleted count, and remaining count.
- Review CSP reports, Clerk auth behavior, ECharts rendering, report downloads, and the hosting provider's edge/proxy headers before switching from report-only to enforced CSP.

## Authentication

Frontend Clerk mode is enabled when both `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY` and `CLERK_SECRET_KEY` are present in the Next.js environment.

- `NEXT_PUBLIC_CLERK_SIGN_IN_URL=/login`
- `NEXT_PUBLIC_CLERK_SIGN_UP_URL=/signup`
- `NEXT_PUBLIC_CLERK_SIGN_IN_FALLBACK_REDIRECT_URL=/dashboard`
- `NEXT_PUBLIC_CLERK_SIGN_UP_FALLBACK_REDIRECT_URL=/dashboard`

Backend authentication is controlled separately:

- `RDS_AUTH_MODE=dev|clerk`
- `dev` is the default local mode. It accepts `X-User-Id` and local `dev:` bearer tokens.
- `clerk` verifies Clerk session JWTs and rejects development credentials.
- `CLERK_ISSUER` is required in Clerk mode.
- Set either `CLERK_JWT_KEY` for networkless PEM public-key verification or `CLERK_JWKS_URL` for JWKS verification.
- Optional hardening:
  - `CLERK_AUDIENCE`
  - `CLERK_AUTHORIZED_PARTIES`
  - `CLERK_CLOCK_SKEW_SECONDS`
- `SAAS_AUTH_FLOW_VERIFICATION_URL` links to HTTPS evidence that deployed sign-up, sign-in, organization activation/switching, invite acceptance, and account recovery have been verified with production Clerk settings.
- `SAAS_AUTH_FLOW_VERIFIED_AT` records the latest auth-flow verification as an ISO-8601 datetime with a timezone.
- `SAAS_AUTH_FLOW_VERIFICATION_MAX_AGE_DAYS` controls how recent auth-flow verification must be for production readiness; it defaults to `30`.
- Clerk-authenticated frontend requests send the session token from `auth().getToken()` to the FastAPI API.
- `/reset` is the frontend account-recovery entry point. In Clerk mode it renders the managed Clerk sign-in/recovery component; in local development it documents the seeded-session recovery path. Production rollout must verify the Clerk email/password recovery flow with live keys.
- `POST /auth/clerk/sync` activates the current Clerk workspace in backend metadata. It requires a verified Clerk token with an active organization context and upserts `users`, `organizations`, and `memberships` from trusted token IDs and role claims.
- `POST /webhooks/clerk` receives signed Clerk webhook events and verifies the raw request body with Svix before processing the event.
- `CLERK_WEBHOOK_SIGNING_SECRET` is required with at least 32 characters for Clerk webhook verification in production.
- Subscribe the Clerk endpoint to `user.created`, `user.updated`, `user.deleted`, `organization.created`, `organization.updated`, `organization.deleted`, `organizationMembership.created`, `organizationMembership.updated`, and `organizationMembership.deleted`.
- Profile fields such as email, display name, and organization name are synchronized from webhooks when available. The active-session sync endpoint still falls back to stable placeholder metadata when those claims are absent.
- Tenant owners can update member roles and remove members through `PATCH/DELETE /organizations/{organization_id}/memberships/{member_user_id}`.
- Membership administration writes `membership_role_updated` and `membership_removed` audit events and rejects self-removal or self-demotion style lockouts. Removing a member transfers any report schedules they created in that tenant to the acting owner before deleting the membership; if a transferred schedule was addressed to the removed member's email, the recipient is reassigned to the acting owner's email. Audit metadata records only transfer/reassignment counts.
- Tenant owners create invites through `POST /organizations/{organization_id}/invites`; the public create schema accepts email/role only, generates opaque URL-safe invite ids, and ignores legacy caller-supplied `invite_id` values so Settings/API-created invites do not use predictable local ids.
- Invite acceptance still requires the accepting user's normalized email to match the invite email, writes `invite_accepted` audit events, and consumes tenant member capacity.
- Invite create, accept, and revoke audit metadata records role/status plus a configured-email flag only; invitee email addresses are not copied into new audit metadata, and timestamped audit list/export responses sanitize legacy invite metadata before it reaches the Settings UI.
- In production Clerk mode, live Clerk organization role/member changes should remain synchronized through Clerk webhook events; direct backend membership edits are metadata-local until a provider write-back integration is added.

## Marketing Inquiries

- `POST /marketing/inquiries` is a public create-only endpoint used by the landing page pilot request form.
- Inquiry records are stored in `marketing_inquiries` with normalized contact details, intended use, source page, `new` status, and creation timestamp.
- Accepted intended-use values are `audit_readiness`, `consultant_delivery`, `estate_portfolio`, `measurement_verification`, and `other`.
- Abuse protection rejects honeypot-filled submissions, duplicate requests for the same normalized email within the short public inquiry window, and excessive successful submissions from the same hashed source key.
- The public API returns only the inquiry id, status, and creation timestamp. There is no public inquiry listing route.
- Source rate limiting is controlled by `SAAS_MARKETING_INQUIRY_RATE_LIMIT_ENABLED`, `SAAS_MARKETING_INQUIRY_RATE_LIMIT_MAX`, and `SAAS_MARKETING_INQUIRY_RATE_LIMIT_WINDOW_MINUTES`. The submission ledger stores only an HMAC source hash plus inquiry id/timestamp for accepted attempts or a null inquiry id for rate-limited attempts, not raw IP addresses or user agents.
- Rate-limited public inquiry responses return `429` with a `Retry-After` header based on the configured rate-limit window, and the landing page maps that response to a user-facing retry message without exposing raw source details.
- `SAAS_MARKETING_INQUIRY_RATE_LIMIT_SECRET` signs source hashes. Production readiness requires rate limiting to remain enabled and this secret to be configured with at least 32 characters.
- Set `SAAS_MARKETING_INQUIRY_RECIPIENT_EMAIL` to send a best-effort support handoff email through the configured email sender after the inquiry is stored.
- Set `SAAS_MARKETING_INQUIRY_CRM_WEBHOOK_URL` to also POST the stored inquiry to a CRM or workflow handoff endpoint. `SAAS_MARKETING_INQUIRY_CRM_WEBHOOK_SECRET` signs the JSON payload with `X-RDS-Signature: sha256=<digest>` and is required with at least 32 characters for CRM webhook handoff in production; `SAAS_MARKETING_INQUIRY_CRM_WEBHOOK_TIMEOUT_SECONDS` defaults to `5`.
- Email and CRM delivery failures do not drop the saved inquiry or change the public response.
- Production readiness requires either `SAAS_MARKETING_INQUIRY_RECIPIENT_EMAIL` or a signed `SAAS_MARKETING_INQUIRY_CRM_WEBHOOK_URL` so public pilot requests have an operational owner.
- Advanced abuse controls and internal sales/support dashboards remain future production handoff work.

## Support Operations

- `GET /support/marketing-inquiries?status=new&limit=100` lists saved marketing inquiries for operational follow-up. It also supports `assigned_to=<assignee>` and `unassigned=true` filters for support queues.
- `GET /support/marketing-inquiries/summary` returns aggregate queue counts by lifecycle status and assignee without inquiry contact details, messages, or follow-up notes.
- `PATCH /support/marketing-inquiries/{inquiry_id}` updates inquiry status to `new`, `contacted`, `qualified`, or `closed`, optional support assignee, and optional follow-up notes. Email-shaped support assignees are normalized to lowercase so queue filters do not split the same owner across case variants.
- `GET /support/marketing-inquiries/{inquiry_id}/activity` lists support follow-up activity for the inquiry.
- Public creation and support updates write activity records with action, status, assignee, changed-field names, note-change flag, actor label, and timestamp; activity records do not copy inquiry contact details, inquiry messages, or follow-up note bodies.
- Support endpoints include inquiry contact details and message text, are not tenant-scoped, and must never be exposed as public frontend pages without platform-admin auth.
- Support endpoints require `Authorization: Bearer <SAAS_SUPPORT_ACCESS_TOKEN>` and fail closed when the token is not configured.
- Production readiness requires `SAAS_SUPPORT_ACCESS_TOKEN` and `SAAS_SUPPORT_OWNER_EMAIL`; production support tokens must be at least 32 characters.

## Metadata Database

- `SAAS_DATABASE_URL` defaults to `sqlite:///:memory:`.
- Local development can use SQLite, for example `sqlite:///./.saas/metadata.db`.
- Production readiness requires a PostgreSQL URL, for example `postgresql://user:password@host:5432/database`.
- PostgreSQL uses the `psycopg` driver path. Live database provisioning, credentials, migrations, backup policy, and connection-pool sizing must be validated before pilot rollout.
- `POST /support/database-test` requires `SAAS_SUPPORT_ACCESS_TOKEN`, writes and reads a tenant-safe `rds.database_test` operation event through the configured metadata store, and returns only event id, event type, and timestamp.
- Production readiness rejects missing or stale successful database smoke-test evidence using `SAAS_DATABASE_TEST_MAX_AGE_DAYS`, which defaults to `30`.

## Artifact Storage

- `SAAS_ARTIFACT_STORE_BACKEND=local|memory|s3`
- `local` defaults to `.saas-artifacts` and is for development.
- `memory` is for tests only.
- `s3` requires `SAAS_S3_BUCKET`.
- Optional S3 settings:
  - `SAAS_S3_PREFIX`
  - `SAAS_S3_ENDPOINT_URL`
  - `SAAS_S3_REGION`
- `SAAS_ARTIFACT_STORE_GOVERNANCE_URL` links to HTTPS evidence that the production bucket configuration has been reviewed for tenant isolation, least-privilege access policy, encryption, versioning or replication, lifecycle/retention behavior, backup alignment, and recovery ownership.
- `SAAS_ARTIFACT_STORE_GOVERNANCE_VERIFIED_AT` records the latest artifact-store governance verification as an ISO-8601 datetime with a timezone.
- `SAAS_ARTIFACT_STORE_GOVERNANCE_MAX_AGE_DAYS` controls how recent artifact-store governance evidence must be for production readiness; it defaults to `30`.
- `POST /support/artifact-store-test` requires `SAAS_SUPPORT_ACCESS_TOKEN`, writes a tiny synthetic object under a platform probe namespace, verifies read-after-write, deletes it, and records tenant-safe `rds.artifact_store_test` platform operation evidence. The response includes only event id, event type, timestamp, and byte count.
- Production readiness rejects missing or stale successful artifact-store smoke-test evidence using `SAAS_ARTIFACT_STORE_TEST_MAX_AGE_DAYS`, which defaults to `30`.
- Production readiness also rejects missing, insecure, future-dated, timezone-less, stale, or invalid artifact-store governance evidence so tenant uploads and reports cannot launch on a bucket that has not had its security/lifecycle controls reviewed.

## Upload Source Data Contract

- Tenant upload writes accept the canonical source-file categories `energy`, `hdd`, `cdd`, and `seu_mapping`.
- Upload categories are normalized at the domain boundary, so hyphenated or mixed-case values such as `SEU-Mapping` are stored as `seu_mapping`.
- Upload content is validated before tenant artifact storage is written. Energy uploads must match either the SaaS meter CSV shape with a date/timestamp column plus numeric meter values or the legacy `Metered Sector`, `Utility`, `Units`, `Period` export shape with dated columns. HDD/CDD files may include metadata preambles but must expose a `Date` header row plus a matching numeric degree-day column. SEU mapping files must include `Meter` and `SEU_Category` rows.
- `GET /organizations/{organization_id}/uploads/readiness` returns a metadata-only source-file readiness checklist for each tenant site, including the latest stored upload for each canonical category.
- The readiness endpoint requires tenant read access and accepts an optional `site_id` filter that is still tenant-scoped.
- `GET/POST/PATCH /organizations/{organization_id}/data-connectors` manages the tenant data-connector registry for API, SFTP, and utility-portal source metadata. Connector records are scoped to an organization and site, use the same source categories as upload ingestion, and track draft/active/paused status plus manual/daily/weekly/monthly cadence.
- `GET /organizations/{organization_id}/data-connectors/due` lists active scheduled connectors whose `next_sync_at` is due, giving a future worker a tenant-scoped selection contract.
- `GET/POST /organizations/{organization_id}/data-connectors/{connector_id}/syncs` lists or records bounded connector sync attempts. A successful sync may link to a stored tenant upload only when that upload belongs to the same organization, site, and source category as the connector.
- `POST /organizations/{organization_id}/data-connectors/{connector_id}/imports` accepts connector-supplied CSV content, validates it against the connector's canonical source category, stores it as a normal tenant upload under the connector site, then records a successful sync linked to that upload. The import route requires both upload-management and asset-management permission and only accepts active connectors.
- Connector reads require tenant read access; create/update operations require asset-management permission and reject cross-tenant sites or connector ids.
- Sync recording and connector imports update the connector's last-sync status, last-sync timestamp, sanitized last-sync error, and next scheduled sync timestamp from the connector cadence. These routes do not contact external providers or store provider secrets by themselves; live provider execution and secret-manager integration remain future work.
- Connector create/update actions write `data_connector_created` and `data_connector_updated` audit events with connector type, source category, status, schedule, and configured-reference flags only. Connector names, external references, credential references, provider paths, and secrets are not copied into audit metadata.
- Connector imports still write the standard sanitized `upload_stored` audit event for the stored upload, with source marked as `data_connector` and no storage key in audit metadata.
- Connector sync attempts and imports write `data_connector_sync_recorded` audit events with connector type, source category, schedule, status, record count, and configured upload/reason flags only. Connector names, external references, credential references, provider paths, upload storage keys, status reasons, imported content, and secrets are not copied into audit metadata.
- Routine upload, report, report-artifact, and synchronous run execution responses omit raw tenant artifact storage keys. The uploads/runs/reports UI presents upload ids, report ids, run ids, checksums, statuses, and authenticated download actions instead of object-storage paths.
- Upload storage writes create `upload_stored` audit events with category/site and an artifact-location configured flag only; raw storage keys are not copied into new upload audit metadata, and audit list/export responses sanitize legacy upload metadata.
- Stored-upload analysis runs use a stored `energy` CSV as the primary execution input and may include same-site supporting `hdd`, `cdd`, and `seu_mapping` upload ids through `source_upload_ids`.
- Supporting source files are recorded in `run_uploads` with the primary energy upload so report artifacts and evidence packages can show the full tenant source package lineage. Stored-upload runs resolve either the SaaS meter CSV shape directly or convert a legacy energy export into the meter dataframe shape before running the canonical analytics service.
- Report artifacts include a metadata-only `source_package` block with upload ids, categories, checksums, statuses, primary marker, site id, counts, resolved meter columns, and source summaries; raw upload content, storage keys, and internal analysis input paths are not copied into the report JSON. Report read, JSON export, evidence-package, and locked-snapshot paths also sanitize legacy artifact JSON through this public contract before returning it. Complete legacy source packages emit an aggregate `legacy_baseline` block with meter counts, energy/weather record coverage, mapped SEU categories, grouped consumption totals, and a `weather_normalization` model summary that regresses gas meters against HDD and electricity meters against CDD when tenant settings enable weather normalization and enough baseline/comparison records exist. The report detail UI renders modeled rows as predicted-vs-actual charts and a model table, and it shows skipped-meter diagnostics when records, weather data, utility mappings, or tenant thresholds prevent model creation. Broader production M&V workflows, custom model families, and portfolio-scale weather-normalized analytics remain later analytics expansions.
- `POST /support/run-lifecycle-test` requires `SAAS_SUPPORT_ACCESS_TOKEN`, creates a synthetic tenant, stores a tiny energy CSV through the configured artifact store, executes the stored-upload analysis path, verifies run/job/report metadata and report artifact content, removes synthetic rows and objects, and records tenant-safe `rds.run_lifecycle_smoke_test` platform operation evidence.
- Production readiness rejects missing or stale successful run-lifecycle smoke-test evidence using `SAAS_RUN_LIFECYCLE_TEST_MAX_AGE_DAYS`, which defaults to `30`.
- `POST /support/worker-run-lifecycle-test` requires `SAAS_SUPPORT_ACCESS_TOKEN`, creates a synthetic tenant, stores a tiny energy CSV, queues a stored-upload analysis run, drains one queued job through the worker job processor, verifies run/job/report metadata and report artifact content, removes synthetic rows and objects, and records tenant-safe `rds.worker_run_lifecycle_smoke_test` platform operation evidence.
- Production readiness rejects missing or stale successful worker run-lifecycle smoke-test evidence using `SAAS_WORKER_RUN_LIFECYCLE_TEST_MAX_AGE_DAYS`, which defaults to `30`.

## Tenant Analysis Settings

- Tenant analysis settings are stored in metadata through `GET/PATCH /organizations/{organization_id}/analysis-config`.
- The saved config uses the existing analytics contract with `client`, `site`, `meters`, `analysis`, and `iso50001` sections.
- The `analysis` section supports tenant-managed weather-normalized M&V controls: `weather_normalisation`, `weather_model_min_baseline_records`, and `weather_model_min_comparison_records`. Stored-upload reports snapshot these values and include the effective model settings in generated weather-normalization summaries.
- SaaS backend orchestration imports analytics behavior through the canonical `analytics/` package. The legacy `src/` modules are compatibility shims for the old Streamlit entry point.
- If a tenant has not saved analysis settings, the backend can generate a default config from the organization's selected site and meter records.
- Queued and stored-upload runs may omit `client_config`; the backend snapshots the saved or tenant-asset-derived config into the analysis job when the run is requested.
- Analysis config updates require asset-management permission and create `analysis_config_updated` audit events with meter count, client id, and weather-model setting summaries only; the full config is not stored in audit metadata.

## Billing and Trials

- `SAAS_BILLING_PROVIDER=local|stripe`
- `local` is the default development adapter. It creates local checkout/portal URLs and accepts the test webhook envelope.
- `stripe` is the production-shaped payment-provider adapter. It requires:
  - `STRIPE_SECRET_KEY`
  - `SAAS_STRIPE_PRICE_ID_TEAM`
  - `SAAS_BILLING_WEBHOOK_SECRET` with at least 32 characters in production
  - `SAAS_BILLING_WEBHOOK_TOLERANCE_SECONDS`, which defaults to `300` and must be between 1 and 86400 seconds
  - `SAAS_BILLING_OPERATIONS_URL`, an HTTPS link to live billing lifecycle evidence covering checkout, portal access, subscription reconciliation, dunning, invoice email, tax, and accounting configuration
  - `SAAS_BILLING_OPERATIONS_VERIFIED_AT`, an ISO-8601 datetime with timezone for the latest live billing operations verification
  - `SAAS_BILLING_OPERATIONS_MAX_AGE_DAYS`, which defaults to `30` and controls the production freshness window for billing operations evidence
- Optional billing settings:
  - `SAAS_BILLING_APP_BASE_URL`
  - `SAAS_STRIPE_API_BASE_URL`
- `GET /billing/plans` exposes the seeded plan tiers: `trial`, `team`, and `enterprise`.
- Each organization has one `organization_subscriptions` row. Local onboarding, Clerk tenant sync, and Clerk organization webhooks create a default `trialing` subscription when none exists.
- `GET /organizations/{organization_id}/subscription` returns the tenant plan, period/trial timestamps, and usage counts for members, active users, pending invite seats, sites, and runs in the current period.
- `PATCH /organizations/{organization_id}/subscription` is owner-only and records `subscription_updated` audit events. It remains a metadata-local operator path.
- `POST /organizations/{organization_id}/billing/checkout` creates an owner-only payment-provider checkout session for paid plans and records `billing_checkout_created`.
- `POST /organizations/{organization_id}/billing/portal` creates an owner-only payment-provider billing portal session after the tenant has an external customer id and records `billing_portal_created`.
- `POST /webhooks/billing` records idempotent provider events in `billing_events`, reconciles checkout/subscription/invoice lifecycle changes into `organization_subscriptions`, and writes system-actor `subscription_updated` audit events. Stripe webhook signatures must include a timestamp within the configured replay tolerance.
- `invoice.payment_failed` events set the tenant subscription to `past_due`; entitlement checks then block new tenant activity until billing is reconciled.
- Backend service boundaries enforce plan limits before creating new invite seats, accepting invites, adding sites, or creating/queueing runs. Entitlement failures return HTTP 402 from FastAPI routes.
- Production readiness rejects missing, insecure, future-dated, timezone-less, stale, or invalid billing operations evidence so multi-client pilots cannot proceed on provider credentials and webhook replay alone.

## Report Retention

- Subscription plan `report_retention_days` values are applied when generated report metadata is created.
- `ReportSummary` responses include `expires_at` and `retention_applied_at`.
- Normal report listing and artifact reads hide expired reports.
- `POST /organizations/{organization_id}/reports/retention/apply` requires report-management permission, deletes expired report artifacts and any locked evidence package artifacts through the tenant artifact-store boundary, removes locked evidence package metadata for retained reports, marks expired reports unpublished, records `retention_applied_at`, returns deleted/missing artifact counts including evidence-package cleanup counts, and writes `report_retention_applied` audit events.
- The Next.js Settings page exposes a report-retention action for users with report-management permission and surfaces the returned cleanup counts after the action completes.
- Enterprise/custom plans can use `report_retention_days=NULL` to keep report metadata open-ended.

## Report Exports

- `GET /organizations/{organization_id}/report-branding` returns tenant evidence-package branding defaults or saved settings.
- `PATCH /organizations/{organization_id}/report-branding` requires organization-management permission and updates the tenant evidence pack title, PDF accent color, optional logo/reference URL, and optional support email.
- `GET /organizations/{organization_id}/report-approval-policy` returns tenant evidence-locking policy defaults or saved settings.
- `PATCH /organizations/{organization_id}/report-approval-policy` requires organization-management permission and updates whether final approval assignments automatically lock evidence plus the minimum approved-assignment count required before any locked evidence package can be created.
- `GET /organizations/{organization_id}/evidence-package-template` returns tenant evidence-package section defaults or saved settings, including client-facing section labels and render order.
- `PATCH /organizations/{organization_id}/evidence-package-template` requires organization-management permission and updates the tenant evidence template name, inclusion flags, client-facing labels, and render order for report artifact, upload lineage, approval assignments, reviews, comments, and schedules. The default template includes every section in the standard ISO 50001 review order.
- `GET /organizations/{organization_id}/reports/{report_id}/export` downloads the generated report artifact as a JSON attachment.
- `GET /organizations/{organization_id}/reports/{report_id}/evidence-package` downloads a tenant evidence package with organization profile, report branding, template metadata, and the evidence sections enabled by the tenant template. Report and upload lineage in evidence packages omits raw tenant artifact storage keys.
- `GET /organizations/{organization_id}/reports/{report_id}/evidence-package/pdf` downloads an app-generated, tenant-branded PDF summary of the current evidence package for ISO 50001 review packs. The initial renderer applies the saved title/accent color, respects tenant section-template labels/order, includes weather-normalized model and skipped-meter diagnostics when artifact evidence is enabled, and records optional logo/contact fields in the package for later template rendering.
- `GET/POST /organizations/{organization_id}/reports/{report_id}/evidence-packages` lists or locks tenant-scoped point-in-time JSON evidence package snapshots. Manual snapshot creation requires report-management permission, no pending approval assignments, a latest management review status of `approved`, and the tenant policy's minimum approved-assignment threshold; snapshot listing requires read permission.
- Completing the final pending approval assignment as `approved` automatically creates a locked JSON evidence package snapshot through the tenant artifact store when the tenant report-approval policy enables final-approval auto-locking and the policy approval threshold is satisfied. The created package uses the same approval gate, storage boundary, and sanitized `report_evidence_package_created` audit metadata as manual locks, with only aggregate counts, policy values, and the source assignment id recorded.
- `GET /organizations/{organization_id}/reports/{report_id}/evidence-packages/{package_id}/download` downloads a locked evidence package from tenant object storage.
- Report export and locked-package download require tenant membership, report scope, and the same published/not-expired/not-retention-applied visibility checks used for artifact reads.
- Export filenames are generated from safe report id and report type parts.
- Locked evidence packages are stored under tenant-scoped artifact keys generated from the organization, report, package id, and filename.
- Report branding updates write `report_branding_updated` audit events with configured-field flags and accent color only; logo URL and support email are not copied into audit metadata.
- Report approval policy updates write `report_approval_policy_updated` audit events with only the auto-lock flag and minimum approved-assignment threshold.
- Evidence package template updates write `report_evidence_template_updated` audit events with section-inclusion flags, enabled-section count, and whether the template name, section labels, or section order are customized. Template names, section labels, render order, and package content are not copied into audit metadata.
- Report artifact reads write `report_accessed` audit events with report id/type and an artifact-location configured flag only. Report artifact export writes `report_exported` audit events with report id, report type, export format, and content length only. Dynamic JSON/PDF evidence package export writes `report_evidence_exported`; locked snapshot creation writes `report_evidence_package_created` with report id, report type, package type, content length, package id, approval status/review id, approval counts/policy values, and evidence section counts only. Locked snapshot downloads write `report_evidence_package_downloaded` with report/package ids, report type, package type, content length, and evidence section counts only.
- Artifact content, storage keys, approval notes, review notes, comments, branding URLs/contact emails, template names, and recipient emails are not copied into report access/export/retention audit metadata, and audit list/export responses sanitize legacy report-access metadata before it reaches the Settings UI.
- The Next.js report detail page proxies downloads through `/reports/{reportId}/download`, `/reports/{reportId}/evidence-package`, `/reports/{reportId}/evidence-package/pdf`, and `/reports/{reportId}/evidence-packages/{packageId}/download` so HTTP-only app sessions can be used for backend authorization.

## Report Scheduling

- `GET /organizations/{organization_id}/reports/{report_id}/schedules` lists delivery schedules for a visible tenant report.
- `POST /organizations/{organization_id}/reports/{report_id}/schedules` creates a schedule record with frequency, recipient, and next delivery time.
- `PATCH /organizations/{organization_id}/reports/{report_id}/schedules/{schedule_id}` updates an existing schedule, including pausing it with `is_active=false`.
- `GET /organizations/{organization_id}/reports/{report_id}/schedules/{schedule_id}/deliveries` lists recorded worker delivery attempts for a visible tenant report schedule.
- Schedule and delivery-history reads require tenant membership and visible report access. Schedule creates/updates require report-management permission.
- Expired, unpublished, or retention-applied reports reject schedule reads and writes through the same report visibility checks used for artifact access.
- Schedule create/update actions write `report_schedule_created` and `report_schedule_updated` audit events without copying recipient email addresses into audit metadata.
- The background worker now processes due active schedules, validates the report artifact behind the tenant artifact-store boundary, sends an email notification through the configured sender, advances `next_run_at`, records delivery attempts in `organization_report_schedule_deliveries`, and writes `report_schedule_delivered` or `report_schedule_failed` audit events.
- The report detail page now shows recent recorded delivery attempts for each schedule alongside last-delivery state.
- Scheduled notifications prompt recipients to sign in and download the authenticated report artifact. Scheduled and approval-assignment email bodies include only an allowlisted artifact digest built from the sanitized public artifact contract (`total_records`, source-package counts/categories, and weather-normalized model/skipped counts and estimated savings when present); arbitrary artifact fields, storage keys, internal analysis input paths, and raw upload content are not copied into email bodies or audit metadata.
- Provider retry policy, branded report packages, attachment delivery, and live provider replay testing still remain before production pilot rollout.

## Report Collaboration

- `GET /organizations/{organization_id}/reports/{report_id}/comments` lists comments for a visible tenant report.
- `POST /organizations/{organization_id}/reports/{report_id}/comments` creates a tenant-scoped report review comment.
- Owner, Manager, and Viewer roles can add report comments, but comment reads and writes still require membership in the tenant path and the report must belong to that tenant.
- Expired, unpublished, or retention-applied reports reject comment reads and writes through the same report visibility checks used for artifact access.
- Comment creation writes `report_comment_created` audit events with report id, report type, and body length only. The comment body is not copied into audit metadata.
- `GET /organizations/{organization_id}/report-approvers` lets Owners and Managers list active Owner/Manager approver candidates for report routing without exposing full member administration.
- `GET/POST /organizations/{organization_id}/reports/{report_id}/approval-assignments` lists and creates tenant-scoped report approval assignments. Reads require visible report access; creates require report-management permission and the assignee must be an active Owner or Manager in the tenant.
- `PATCH /organizations/{organization_id}/reports/{report_id}/approval-assignments/{assignment_id}` completes or cancels an approval assignment. Approving or requesting changes also records a management review decision for the report. When an approval completes the final pending assignment and the latest review is approved, the backend automatically locks the current evidence package snapshot only when the tenant report-approval policy enables auto-locking and the approved-assignment threshold is satisfied.
- Approval assignment creation queues a `report_approval_assigned` email notification for the assigned approver. Completion or cancellation queues a `report_approval_updated` email notification for the assigner, or for the assignee when the assigner updates their own assignment. Assigned and updated approval notifications include the same allowlisted report digest when the current artifact is available.
- `GET /organizations/{organization_id}/reports/{report_id}/reviews` lists management review decisions for a visible tenant report.
- `POST /organizations/{organization_id}/reports/{report_id}/reviews` records an `approved` or `changes_requested` decision and optional note. Recording a decision requires report-management permission; Viewers can read the history but cannot approve reports.
- Approval assignment create/update actions write `report_approval_assigned` and `report_approval_updated` audit events with status, assignee user id, due date, report id/type, generated review id, and note length only. Assignment notes are not copied into audit metadata.
- Review creation writes `report_review_recorded` audit events with report id, report type, review status, and note length only. Review notes are not copied into audit metadata.

## Privacy Settings

- `GET /organizations/{organization_id}/privacy-settings` returns tenant privacy settings for authenticated tenant members.
- `PATCH /organizations/{organization_id}/privacy-settings` requires organization-management permission, records data-processing acknowledgement details, stores the tenant privacy contact, and controls subprocessor notifications plus anonymized product analytics sharing.
- Privacy settings updates write `privacy_settings_updated` audit events without storing the privacy contact email in audit metadata.
- `GET/POST /organizations/{organization_id}/privacy-requests` and `PATCH /organizations/{organization_id}/privacy-requests/{privacy_request_id}` require organization-management permission and track tenant data export, deletion, and correction requests.
- Privacy request create/update actions write `privacy_request_created` and `privacy_request_updated` audit events without copying free-text descriptions or resolution notes into audit metadata.
- Live legal review, DPA text, privacy notice publication, provider subprocessors, and operational fulfillment procedures for export/deletion/correction requests remain required before production pilot rollout.

## Pilot Readiness Tracking

- `GET /organizations/{organization_id}/pilot-readiness` returns the tenant owner-managed launch checklist with status, evidence link, notes, updater, and timestamp for each gate.
- `PATCH /organizations/{organization_id}/pilot-readiness/{item_key}` requires organization-management permission and updates one known checklist gate.
- Checklist gates cover DPA signature, privacy notice review, production auth, production billing, email delivery, storage/backups, tenant isolation review, and support ownership.
- Pilot readiness updates write `pilot_readiness_updated` audit events without copying notes or evidence URLs into audit metadata.
- This checklist is an internal tracking workflow only. It does not replace live Clerk, Stripe, SMTP, storage, backup, monitoring, security, legal, or support verification before pilot launch.

## Notification Delivery

- `SAAS_EMAIL_BACKEND=memory|smtp`
- `memory` is the default for local development and tests.
- `smtp` sends notification outbox records through an SMTP provider.
- Notification outbox records are currently created for run completion/failure and report approval assignment routing.
- SMTP settings:
  - `SAAS_SMTP_HOST`
  - `SAAS_SMTP_PORT`
  - `SAAS_SMTP_USERNAME`
  - `SAAS_SMTP_PASSWORD`
  - `SAAS_SMTP_FROM_EMAIL`
  - `SAAS_SMTP_USE_TLS`
  - `SAAS_SMTP_TIMEOUT_SECONDS`
- `SAAS_EMAIL_DELIVERY_VERIFICATION_URL` links to HTTPS evidence that the production SMTP/sender-domain setup has been verified, including DNS authentication, sender-domain status, delivery smoke test, bounce/complaint handling, retry policy, and provider dashboard or runbook ownership.
- `SAAS_EMAIL_DELIVERY_VERIFIED_AT` records the latest email delivery verification as an ISO-8601 datetime with a timezone.
- `SAAS_EMAIL_DELIVERY_VERIFICATION_MAX_AGE_DAYS` controls how recent email delivery verification must be for production readiness; it defaults to `30`.
- `SAAS_NOTIFICATION_MAX_ATTEMPTS` controls how many email delivery attempts the worker makes before marking a notification `failed`; it defaults to `3`.
- `SAAS_NOTIFICATION_RETRY_DELAY_SECONDS` controls how long a failed notification remains pending before the next worker retry; it defaults to `300`.
- Tenant notification summaries include attempt count, last attempt timestamp, next retry timestamp, sent timestamp, and sanitized error text so users can see delivery progress without exposing provider credentials.
- `SAAS_NOTIFICATION_PROVIDER_WEBHOOK_SECRET` verifies signed provider delivery callbacks at `POST /webhooks/notifications/provider`. Providers must send `X-RDS-Timestamp` as an ISO-8601 timestamp with a timezone and `X-RDS-Signature: sha256=<digest>`, where the digest is the HMAC-SHA256 of `<timestamp>.<raw JSON body>`. Production readiness requires this secret to be at least 32 characters.
- `SAAS_NOTIFICATION_PROVIDER_WEBHOOK_TOLERANCE_SECONDS` controls replay protection for provider delivery callbacks; it defaults to `300` and must be between 1 and 86400 seconds.
- Provider webhook payloads use a normalized contract with `provider`, `event_id`, `notification_id`, `status`, `occurred_at`, and optional sanitized `detail`. Accepted statuses are `delivered`, `bounced`, `complained`, `deferred`, and `dropped`.
- Provider callbacks are idempotent by `(provider, event_id)`. `delivered` marks the notification sent; `bounced`, `complained`, and `dropped` mark it failed with a generic delivery-status error; `deferred` is recorded as provider evidence without changing tenant notification status.
- Successfully processed Clerk, billing, and notification-provider webhook callbacks record tenant-safe platform operation evidence for production replay validation. Details include only finite action/status/duplicate fields and omit provider event ids, organization ids, user ids, notification ids, email addresses, and payload content.
- `SAAS_PROVIDER_WEBHOOK_REPLAY_MAX_AGE_DAYS` defaults to `30`; production readiness fails when recent successful Clerk, billing, or notification-provider replay evidence is missing or stale.
- `SAAS_MARKETING_INQUIRY_RECIPIENT_EMAIL` receives public landing-page pilot inquiries.
- `SAAS_MARKETING_INQUIRY_CRM_WEBHOOK_URL` optionally receives signed public landing-page pilot inquiry payloads for CRM/workflow handoff.
- `SAAS_MARKETING_INQUIRY_CRM_WEBHOOK_SECRET` signs CRM webhook payloads and is required with at least 32 characters when CRM handoff is configured in production.
- `SAAS_MARKETING_INQUIRY_CRM_WEBHOOK_TIMEOUT_SECONDS` controls the CRM webhook POST timeout.
- `SAAS_MARKETING_INQUIRY_RATE_LIMIT_ENABLED`, `SAAS_MARKETING_INQUIRY_RATE_LIMIT_MAX`, and `SAAS_MARKETING_INQUIRY_RATE_LIMIT_WINDOW_MINUTES` control public inquiry source throttling.
- `SAAS_MARKETING_INQUIRY_RATE_LIMIT_SECRET` signs privacy-preserving source hashes. Production readiness requires rate limiting to remain enabled and this secret to be configured with at least 32 characters.
- Production readiness requires `SAAS_EMAIL_BACKEND=smtp`, `SAAS_SMTP_HOST`, `SAAS_SMTP_FROM_EMAIL`, fresh HTTPS email delivery verification evidence, valid provider webhook replay settings, and `SAAS_NOTIFICATION_PROVIDER_WEBHOOK_SECRET`.

## Monitoring

- `GET /metrics` exposes Prometheus-compatible aggregate platform metrics for monitoring systems.
- Metrics include total organizations, users, active users, sites, meters, status counts for uploads, runs, analysis jobs, notifications, notification provider events, report schedule deliveries, and public marketing inquiries, notification outbox delivery-phase counts, report schedule execution-phase counts, oldest due-notification presence/age, oldest due-report-schedule presence/age, accepted/rate-limited public inquiry submission outcomes, sanitized rejection reasons, worker heartbeat counts by all-tenant vs tenant-scoped processing scope, latest all-tenant worker heartbeat presence/age, alert-smoke platform operation success/failure counts, latest alert-smoke success presence/age, scheduled-dispatch platform operation success/failure counts, latest scheduled-dispatch success presence/age, provider replay operation success/failure counts, artifact-store smoke-test operation success/failure counts, latest artifact-store smoke-test success presence/age, database smoke-test operation success/failure counts, latest database smoke-test success presence/age, tenant-isolation smoke-test operation success/failure counts, latest tenant-isolation smoke-test success presence/age, run-lifecycle smoke-test operation success/failure counts, latest run-lifecycle smoke-test success presence/age, worker run-lifecycle smoke-test operation success/failure counts, latest worker run-lifecycle smoke-test success presence/age, and CSP violation report counts by disposition and sanitized effective directive.
- Notification outbox phases are limited to `pending_due`, `pending_retry_scheduled`, and `failed` so operators can alert on retry backlog and terminal failures without tenant labels.
- Report schedule phases are limited to `active_due`, `active_future`, `paused`, and `failed_last_delivery` so operators can alert on schedule backlog and failed scheduled delivery without tenant labels.
- Metrics labels are limited to known lifecycle status, provider delivery status, delivery/schedule phase, outcome, reason, worker-scope, platform-operation type/status, CSP disposition, or CSP directive values; organization IDs, user IDs, recipient emails, worker IDs, provider names/event ids, notification ids, report ids, schedule ids, schedule delivery ids, alert-test request IDs/details, database URLs, CSP document origins, CSP blocked URLs, storage keys, and other tenant identifiers are not emitted.
- `RDS_METRICS_TOKEN` protects the metrics endpoint with `Authorization: Bearer <token>` when configured.
- In development, `RDS_METRICS_TOKEN` is optional so local smoke checks can run without secrets.
- Production readiness requires `RDS_METRICS_TOKEN`; production tokens must be at least 32 characters.
- `SAAS_SUPPORT_ACCESS_TOKEN` protects support-only operational inquiry follow-up APIs and must be at least 32 characters in production.
- `SAAS_SUPPORT_OWNER_EMAIL` identifies the accountable support owner for pilot operations. Production readiness requires this to be a valid email-shaped value so alert and support handoff workflows have a named owner.
- API responses include `X-Request-Id` and `traceparent`. Requests may provide a safe `X-Request-Id` and valid W3C `traceparent`; otherwise the API generates them.
- The API writes one JSON request log event per request to the `rds.api` logger. Events include request id, trace id, span id, method, route template, status code, and duration.
- Request logs intentionally use route templates such as `/organizations/{organization_id}` and do not include raw paths, query strings, request bodies, user ids, organization ids, storage keys, or authorization headers.
- `SAAS_LOG_AGGREGATION_URL` points to the HTTPS log aggregation dashboard, saved search, or runbook evidence for the deployed `rds.api` logs.
- `SAAS_LOG_AGGREGATION_VERIFIED_AT` records the most recent successful verification that structured request logs are reaching the aggregation stack as an ISO-8601 datetime with a timezone.
- `SAAS_LOG_AGGREGATION_MAX_AGE_DAYS` controls how recent the verification must be for production readiness; it defaults to `30`.
- `SAAS_LOG_RETENTION_DAYS` records the operational log retention window and must be a positive value no greater than `3650`.
- Production readiness requires HTTPS log aggregation evidence, a fresh verification timestamp, and a valid retention window.

## Backups and Alerting

- `SAAS_BACKUP_POLICY_URL` points to the approved metadata/artifact backup policy or runbook.
- `SAAS_BACKUP_JOB_URL` points to the latest successful backup job dashboard, backup-service record, or runbook evidence.
- `SAAS_BACKUP_JOB_VERIFIED_AT` records the most recent successful backup job verification as an ISO-8601 datetime with a timezone.
- `SAAS_BACKUP_MAX_JOB_AGE_DAYS` defaults to `7`; production readiness fails when backup-job evidence is older than this value.
- `SAAS_BACKUP_RESTORE_TESTED_AT` records the most recent successful restore test as an ISO-8601 datetime.
- `SAAS_BACKUP_MAX_RESTORE_TEST_AGE_DAYS` defaults to `90`; production readiness fails when restore-test evidence is older than this value.
- `SAAS_ALERT_WEBHOOK_URL` points to the production monitoring/incident destination.
- `SAAS_ALERT_RUNBOOK_URL` points to the incident response runbook.
- `SAAS_ALERT_RULES_URL` points to the deployed alert-rule dashboard, saved rule group, or runbook evidence for the pilot monitoring stack.
- `SAAS_ALERT_RULES_VERIFIED_AT` records the most recent verification that alert rules for API errors, worker heartbeat age, due notification backlog, terminal notification failures, CSP report spikes, and report-schedule backlog are deployed as an ISO-8601 datetime with a timezone.
- `SAAS_ALERT_RULES_MAX_AGE_DAYS` controls how recent alert-rule deployment evidence must be for production readiness; it defaults to `30`.
- `SAAS_ALERT_WEBHOOK_TIMEOUT_SECONDS` controls the alert webhook delivery timeout for the support smoke-test endpoint; it defaults to `5`.
- `SAAS_ALERT_DELIVERY_TEST_MAX_AGE_DAYS` controls how recent the latest successful alert delivery smoke-test event must be for production readiness; it defaults to `30`.
- `SAAS_ALERT_NOTIFICATION_DUE_AGE_SECONDS` sets the alert threshold for the oldest due notification waiting in the outbox; it defaults to `900` in development and must be configured as a positive integer in production.
- `SAAS_ALERT_NOTIFICATION_FAILED_COUNT` sets the alert threshold for terminally failed notification outbox records after retries are exhausted; it defaults to `1` in development and must be configured as a positive integer in production.
- `SAAS_ALERT_REPORT_SCHEDULE_DUE_AGE_SECONDS` sets the alert threshold for the oldest due report schedule waiting for worker delivery; it defaults to `900` in development and must be configured as a positive integer in production.
- `SAAS_ALERT_REPORT_SCHEDULE_FAILED_COUNT` sets the alert threshold for schedules whose last delivery failed; it defaults to `1` in development and must be configured as a positive integer in production.
- `SAAS_PROVIDER_WEBHOOK_REPLAY_MAX_AGE_DAYS` controls how recent successful Clerk, billing, and notification-provider webhook replay evidence must be for production readiness; it defaults to `30`.
- `SAAS_ARTIFACT_STORE_TEST_MAX_AGE_DAYS` controls how recent successful artifact-store smoke-test evidence must be for production readiness; it defaults to `30`.
- `SAAS_DATABASE_TEST_MAX_AGE_DAYS` controls how recent successful database smoke-test evidence must be for production readiness; it defaults to `30`.
- `SAAS_TENANT_ISOLATION_TEST_MAX_AGE_DAYS` controls how recent successful synthetic tenant-isolation smoke-test evidence must be for production readiness; it defaults to `30`.
- `SAAS_RUN_LIFECYCLE_TEST_MAX_AGE_DAYS` controls how recent successful synthetic upload-to-report smoke-test evidence must be for production readiness; it defaults to `30`.
- `SAAS_WORKER_RUN_LIFECYCLE_TEST_MAX_AGE_DAYS` controls how recent successful synthetic queued-worker upload-to-report smoke-test evidence must be for production readiness; it defaults to `30`.
- `POST /support/alert-test` requires `SAAS_SUPPORT_ACCESS_TOKEN`, sends a generic `rds.alert_delivery_test` event to `SAAS_ALERT_WEBHOOK_URL`, and records a platform operation event so operators can verify alert delivery before a pilot. The payload includes service, severity, timestamp, message, and request id only; it does not include tenant, user, recipient, storage, or support-token values.
- `POST /support/database-test` requires `SAAS_SUPPORT_ACCESS_TOKEN`, verifies metadata database write/read behavior through the platform operation evidence table, and records `rds.database_test` evidence. It does not return or record database URLs, credentials, SQL text, tenant ids, user ids, or provider error details.
- `POST /support/artifact-store-test` requires `SAAS_SUPPORT_ACCESS_TOKEN`, verifies the configured artifact store with a synthetic write/read/delete cycle, and records `rds.artifact_store_test` evidence. It does not return or record tenant ids, storage keys, bucket keys, payload content, or provider error details.
- `POST /support/tenant-isolation-test` requires `SAAS_SUPPORT_ACCESS_TOKEN`, creates synthetic tenant A/B metadata inside a rollback-only transaction, verifies user organization, site, meter, upload, and cross-tenant site-filter boundaries, removes the synthetic records, and records `rds.tenant_isolation_test` evidence. The response and stored operation event include only event metadata, checks passed, and synthetic record count.
- `POST /support/run-lifecycle-test` requires `SAAS_SUPPORT_ACCESS_TOKEN`, verifies the synthetic stored-upload to generated-report lifecycle through metadata and artifact storage, removes synthetic rows and objects, and records `rds.run_lifecycle_smoke_test` evidence. It does not return or record tenant ids, user ids, site ids, storage keys, email addresses, source content, report content, or synthetic probe ids.
- `POST /support/worker-run-lifecycle-test` requires `SAAS_SUPPORT_ACCESS_TOKEN`, verifies the synthetic stored-upload to generated-report lifecycle through the queued worker job processor, removes synthetic rows and objects, and records `rds.worker_run_lifecycle_smoke_test` evidence. It does not return or record tenant ids, user ids, site ids, storage keys, email addresses, source content, report content, or synthetic probe ids.
- `GET /support/operation-events` requires `SAAS_SUPPORT_ACCESS_TOKEN` and lists recent platform operation evidence with optional `event_type=rds.alert_delivery_test|rds.report_schedule_dispatch|rds.clerk_webhook_replay|rds.billing_webhook_replay|rds.notification_provider_webhook_replay|rds.artifact_store_test|rds.database_test|rds.tenant_isolation_test|rds.run_lifecycle_smoke_test|rds.worker_run_lifecycle_smoke_test` and `status=succeeded|failed` filters. Events record event id, type, status, request id, sanitized detail, and timestamp only.
- `GET /support/worker-heartbeats/summary` and `GET /support/worker-heartbeats?scope=all|organization` require `SAAS_SUPPORT_ACCESS_TOKEN` and expose recent worker heartbeat supervision data. Responses include last analysis, report-schedule, and connector-import batch counts, finite scope labels, and stable worker fingerprints; they do not include raw worker ids, organization ids, tenant names, artifact keys, connector ids, provider references, recipient emails, or queued resource ids.
- Alert-test configuration failures return `503`; non-2xx webhook responses return `502`.
- In production, backup policy, recent backup-job evidence, restore-test evidence, alert webhook URL, alert runbook URL, recent alert-rule evidence, notification and report-schedule alert thresholds, support owner email, recent successful alert-smoke evidence, recent successful artifact-store smoke-test evidence, recent successful database smoke-test evidence, recent successful tenant-isolation smoke-test evidence, recent successful run-lifecycle smoke-test evidence, and recent successful worker run-lifecycle smoke-test evidence must all be configured or recorded.

## Background Worker

- Run queued analysis jobs, due report-schedule notifications, and provider-backed due connector imports with `python -m backend.worker_app`.
- `SAAS_WORKER_ENABLED=true` asserts that a long-running worker process is deployed for production readiness.
- `SAAS_WORKER_ID` optionally names the worker instance recorded in metadata heartbeats; it defaults to `default`.
- `SAAS_WORKER_BATCH_SIZE` controls how many queued jobs, due report schedules, and due data connectors the worker attempts per polling cycle.
- `SAAS_WORKER_POLL_SECONDS` controls the delay between polling cycles.
- `SAAS_WORKER_HEARTBEAT_MAX_AGE_SECONDS` controls how recent the latest all-tenant worker heartbeat must be for production readiness; it defaults to `120`.
- `SAAS_WORKER_SUPERVISOR_URL` points to the HTTPS supervisor dashboard, process-manager service, or runbook evidence for the long-running worker deployment.
- `SAAS_WORKER_SUPERVISOR_VERIFIED_AT` records the most recent verification that the worker process is externally supervised as an ISO-8601 datetime with a timezone.
- `SAAS_WORKER_SUPERVISOR_MAX_AGE_DAYS` controls how recent the supervisor verification must be for production readiness; it defaults to `30`.
- `SAAS_WORKER_ORGANIZATION_ID` optionally limits a worker process to one tenant; scoped workers process only that tenant's queued analysis jobs, due report schedules, due data connectors when a connector provider is supplied, and pending notification delivery.
- `SAAS_WORKER_RUN_ONCE=true` runs one batch and exits for smoke tests.
- Each worker batch records a `worker_heartbeats` row with the worker id, scope, poll settings, last analysis/report-schedule/connector-import batch counts, and check-in timestamp. Tenant-scoped workers record scoped heartbeats for operations visibility, but the production `/ready` gate requires a recent all-tenant heartbeat.
- Worker batches that attempt due report schedules also record `rds.report_schedule_dispatch` platform operation evidence with scope and scheduled attempt/sent/failed/skipped counts only. These support events intentionally exclude worker ids, organization ids, report ids, schedule ids, recipient emails, artifact keys, and delivery error text.
- Support operators can inspect recent worker heartbeat summaries through `/support/worker-heartbeats` without exposing tenant ids; use `scope=all` to verify the primary production worker and `scope=organization` only for targeted worker diagnostics.
- Connector import processing is opt-in at the service boundary: deployments or tests must provide a provider adapter that fetches connector content. If no provider is supplied, the worker leaves due connectors untouched rather than marking them failed.
- The worker must use the same `SAAS_DATABASE_URL`, artifact-store configuration, and email sender configuration as the API.
- Production readiness requires `SAAS_WORKER_ENABLED=true`, a positive poll interval, `SAAS_WORKER_RUN_ONCE=false` or unset, no `SAAS_WORKER_ORGANIZATION_ID` tenant filter, recent HTTPS worker-supervisor evidence, and a recent all-tenant worker heartbeat so queued work for all tenants can be processed.

## Production Readiness

For `RDS_RUNTIME_ENV=production`, `/ready` should only report ready when:

- database migrations are applied and `SAAS_DATABASE_URL` is PostgreSQL-shaped,
- `RDS_AUTH_MODE=clerk` has valid Clerk verifier configuration and fresh HTTPS auth-flow verification evidence,
- `SAAS_ALLOWED_ORIGINS` lists trusted HTTPS frontend origins and HSTS is enabled,
- `SAAS_SECURITY_REVIEW_URL` and `SAAS_SECURITY_REVIEWED_AT` point to fresh HTTPS security-review evidence,
- `SAAS_TENANT_ISOLATION_REVIEW_URL` and `SAAS_TENANT_ISOLATION_REVIEWED_AT` point to fresh HTTPS tenant-isolation review evidence,
- `CLERK_WEBHOOK_SIGNING_SECRET` is configured with at least 32 characters for signed Clerk webhook processing,
- `SAAS_EMAIL_BACKEND=smtp` has required SMTP delivery settings and fresh HTTPS email delivery verification evidence,
- `SAAS_NOTIFICATION_PROVIDER_WEBHOOK_SECRET` is configured with at least 32 characters for signed provider delivery callbacks,
- `SAAS_MARKETING_INQUIRY_RECIPIENT_EMAIL` or `SAAS_MARKETING_INQUIRY_CRM_WEBHOOK_URL` with a signing secret of at least 32 characters is configured for public pilot inquiry handoff,
- `SAAS_MARKETING_INQUIRY_RATE_LIMIT_SECRET` is configured with at least 32 characters for privacy-preserving public inquiry abuse controls,
- `SAAS_BILLING_PROVIDER` is a production payment adapter with required checkout/webhook configuration, including a billing webhook secret of at least 32 characters and a valid webhook replay tolerance,
- backup policy, recent backup-job evidence, and recent restore-test evidence are configured,
- alerting webhook, incident runbook, deployed alert-rule evidence, notification thresholds, and report-schedule thresholds are configured,
- recent successful Clerk, billing, and notification-provider webhook replay evidence is recorded,
- recent successful artifact-store smoke-test evidence is recorded,
- recent successful database smoke-test evidence is recorded,
- recent successful synthetic tenant-isolation smoke-test evidence is recorded,
- recent successful synthetic upload-to-report run-lifecycle smoke-test evidence is recorded,
- `RDS_METRICS_TOKEN` is configured with at least 32 characters for protected operational metrics,
- log aggregation evidence and retention are configured through `SAAS_LOG_AGGREGATION_URL`, `SAAS_LOG_AGGREGATION_VERIFIED_AT`, and `SAAS_LOG_RETENTION_DAYS`,
- `SAAS_SUPPORT_ACCESS_TOKEN` is configured with at least 32 characters for protected support operations,
- `SAAS_SUPPORT_OWNER_EMAIL` is configured for pilot support ownership,
- `SAAS_CSP_REPORT_RETENTION_DAYS` is a valid bounded CSP report evidence retention window,
- `SAAS_WORKER_ENABLED=true` confirms a long-running all-tenant worker deployment is configured, recent worker supervisor evidence exists, and a recent all-tenant worker heartbeat is recorded,
- artifact storage is not local-only and has fresh HTTPS bucket-governance evidence.

See `docs/PILOT_DEPLOYMENT_RUNBOOK.md` for the API, worker, frontend, PostgreSQL, and artifact-store deployment shape.
