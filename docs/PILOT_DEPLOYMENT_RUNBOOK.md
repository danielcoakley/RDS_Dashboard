# Pilot Deployment Runbook

This runbook turns the current SaaS scaffolding into a repeatable pilot-shaped deployment. It is not a substitute for live infrastructure provisioning, but it keeps the API, worker, frontend, metadata database, and artifact store boundaries aligned with `docs/SAAS_PLATFORM_PLAN.md`.

## Services

- `frontend`: Next.js SaaS shell and landing experience.
- `api`: FastAPI tenant/auth/upload/run/report boundary.
- `worker`: background analysis job processor using the same metadata database and artifact store as the API.
- `db`: PostgreSQL metadata store for pilot-shaped local compose runs.
- Object storage: local shared compose volume for development; S3-compatible storage for production.

## Local Compose Smoke Run

1. Copy `.env.example` to a local secret file when real credentials are available. Do not commit secrets.
2. For local development without live Clerk, keep `RDS_RUNTIME_ENV=development` and `RDS_AUTH_MODE=dev`.
3. Start the stack:

```powershell
docker compose up --build
```

4. Check the API:

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/ready
```

5. Open the frontend at `http://localhost:3000`.

The compose stack uses PostgreSQL for metadata and a shared local artifact volume between API and worker. This catches more multi-service issues than running the API alone, while still avoiding production secrets.

## Production Environment

Set these before pilot rollout:

- `RDS_RUNTIME_ENV=production`
- `RDS_AUTH_MODE=clerk`
- `SAAS_DATABASE_URL=postgresql://...`
- `SAAS_ALLOWED_ORIGINS=https://app.example.com`
- `SAAS_HSTS_MAX_AGE_SECONDS=31536000`
- `SAAS_SECURITY_REVIEW_URL=https://ops.example.com/security-review`
- `SAAS_SECURITY_REVIEWED_AT=...`
- `SAAS_TENANT_ISOLATION_REVIEW_URL=https://ops.example.com/tenant-isolation-review`
- `SAAS_TENANT_ISOLATION_REVIEWED_AT=...`
- `SAAS_ARTIFACT_STORE_BACKEND=s3`
- `SAAS_S3_BUCKET=...`
- `SAAS_ARTIFACT_STORE_GOVERNANCE_URL=https://ops.example.com/artifacts/governance`
- `SAAS_ARTIFACT_STORE_GOVERNANCE_VERIFIED_AT=...`
- `CLERK_ISSUER=...`
- `CLERK_JWT_KEY` or `CLERK_JWKS_URL`
- `CLERK_WEBHOOK_SIGNING_SECRET=...` (at least 32 characters)
- `SAAS_AUTH_FLOW_VERIFICATION_URL=https://ops.example.com/auth/flow-verification`
- `SAAS_AUTH_FLOW_VERIFIED_AT=...`
- `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=...`
- `CLERK_SECRET_KEY=...`
- `NEXT_PUBLIC_API_BASE_URL=https://api.example.com`
- `SAAS_EMAIL_BACKEND=smtp`
- `SAAS_SMTP_HOST=...`
- `SAAS_SMTP_FROM_EMAIL=...`
- `SAAS_EMAIL_DELIVERY_VERIFICATION_URL=https://ops.example.com/email/delivery`
- `SAAS_EMAIL_DELIVERY_VERIFIED_AT=...`
- `SAAS_NOTIFICATION_PROVIDER_WEBHOOK_SECRET=...` (at least 32 characters)
- `SAAS_MARKETING_INQUIRY_RECIPIENT_EMAIL=...` or signed CRM webhook handoff:
- `SAAS_MARKETING_INQUIRY_CRM_WEBHOOK_URL=https://crm.example.com/rds`
- `SAAS_MARKETING_INQUIRY_CRM_WEBHOOK_SECRET=...` (at least 32 characters)
- `SAAS_MARKETING_INQUIRY_RATE_LIMIT_SECRET=...` (at least 32 characters)
- `SAAS_BILLING_PROVIDER=stripe`
- `STRIPE_SECRET_KEY=...`
- `SAAS_STRIPE_PRICE_ID_TEAM=...`
- `SAAS_BILLING_WEBHOOK_SECRET=...` (at least 32 characters)
- `SAAS_BILLING_WEBHOOK_TOLERANCE_SECONDS=300`
- `SAAS_BILLING_OPERATIONS_URL=https://ops.example.com/billing/lifecycle`
- `SAAS_BILLING_OPERATIONS_VERIFIED_AT=...`
- `RDS_METRICS_TOKEN=...`
- `SAAS_SUPPORT_ACCESS_TOKEN=...`
- `SAAS_SUPPORT_OWNER_EMAIL=support.owner@example.com`
- `SAAS_WORKER_ENABLED=true`
- `SAAS_WORKER_SUPERVISOR_URL=https://ops.example.com/workers/rds`
- `SAAS_WORKER_SUPERVISOR_VERIFIED_AT=...`
- `SAAS_BACKUP_POLICY_URL=https://ops.example.com/backups`
- `SAAS_BACKUP_JOB_URL=https://ops.example.com/backups/latest`
- `SAAS_BACKUP_JOB_VERIFIED_AT=...`
- `SAAS_BACKUP_RESTORE_TESTED_AT=...`
- `SAAS_ALERT_WEBHOOK_URL=https://alerts.example.com/rds`
- `SAAS_ALERT_RUNBOOK_URL=https://ops.example.com/rds-runbook`
- `SAAS_ALERT_RULES_URL=https://monitoring.example.com/rds/alerts`
- `SAAS_ALERT_RULES_VERIFIED_AT=...`
- `SAAS_ALERT_NOTIFICATION_DUE_AGE_SECONDS=900`
- `SAAS_ALERT_NOTIFICATION_FAILED_COUNT=1`
- `SAAS_ALERT_REPORT_SCHEDULE_DUE_AGE_SECONDS=900`
- `SAAS_ALERT_REPORT_SCHEDULE_FAILED_COUNT=1`
- `SAAS_ALERT_DELIVERY_TEST_MAX_AGE_DAYS=30`
- `SAAS_PROVIDER_WEBHOOK_REPLAY_MAX_AGE_DAYS=30`
- `SAAS_ARTIFACT_STORE_TEST_MAX_AGE_DAYS=30`
- `SAAS_DATABASE_TEST_MAX_AGE_DAYS=30`

Optional hardening:

- `CLERK_AUDIENCE`
- `CLERK_AUTHORIZED_PARTIES`
- `CLERK_CLOCK_SKEW_SECONDS`
- `SAAS_AUTH_FLOW_VERIFICATION_MAX_AGE_DAYS`
- `SAAS_S3_PREFIX`
- `SAAS_S3_ENDPOINT_URL`
- `SAAS_S3_REGION`
- `SAAS_ARTIFACT_STORE_GOVERNANCE_MAX_AGE_DAYS`
- `SAAS_WORKER_BATCH_SIZE`
- `SAAS_WORKER_POLL_SECONDS`
- `SAAS_SMTP_PORT`
- `SAAS_SMTP_USERNAME`
- `SAAS_SMTP_PASSWORD`
- `SAAS_SMTP_USE_TLS`
- `SAAS_SMTP_TIMEOUT_SECONDS`
- `SAAS_EMAIL_DELIVERY_VERIFICATION_MAX_AGE_DAYS`
- `SAAS_ALERT_WEBHOOK_TIMEOUT_SECONDS`
- `SAAS_ALERT_RULES_MAX_AGE_DAYS`
- `SAAS_NOTIFICATION_MAX_ATTEMPTS`
- `SAAS_NOTIFICATION_RETRY_DELAY_SECONDS`
- `SAAS_BILLING_APP_BASE_URL`
- `SAAS_STRIPE_API_BASE_URL`
- `SAAS_BILLING_OPERATIONS_MAX_AGE_DAYS`
- `SAAS_BACKUP_MAX_JOB_AGE_DAYS`
- `SAAS_BACKUP_MAX_RESTORE_TEST_AGE_DAYS`
- `SAAS_SECURITY_REVIEW_MAX_AGE_DAYS`
- `SAAS_TENANT_ISOLATION_REVIEW_MAX_AGE_DAYS`

## Clerk Webhook Setup

Configure a Clerk webhook endpoint pointing at:

```text
https://api.example.com/webhooks/clerk
```

Subscribe to:

- `user.created`
- `user.updated`
- `user.deleted`
- `organization.created`
- `organization.updated`
- `organization.deleted`
- `organizationMembership.created`
- `organizationMembership.updated`
- `organizationMembership.deleted`

Use Clerk/Svix replay tooling to verify that a signed test event reaches the API and updates the backend metadata tables. Successful processing records a tenant-safe `rds.clerk_webhook_replay` platform operation event without user ids, organization ids, or email addresses. The `/ready` endpoint should fail production readiness when `CLERK_WEBHOOK_SIGNING_SECRET` is missing or shorter than 32 characters, or when no recent successful Clerk replay evidence exists.

For pilot administration, verify member role changes and removals in Settings against the backend membership APIs. In Clerk production mode, decide whether Clerk or the backend UI is the operational source of truth until provider write-back is implemented, because Clerk webhooks will continue to synchronize provider-side membership changes into backend metadata.

Before setting production auth ready, verify the deployed user journey with live Clerk settings: sign-up, sign-in, sign-out, password/account recovery through `/reset`, organization creation or activation, organization switching, invite acceptance, and a backend `POST /auth/clerk/sync` call using the active organization. Record the evidence in an HTTPS operator document or dashboard through `SAAS_AUTH_FLOW_VERIFICATION_URL`, set `SAAS_AUTH_FLOW_VERIFIED_AT` with a timezone-aware ISO timestamp, and tune `SAAS_AUTH_FLOW_VERIFICATION_MAX_AGE_DAYS` if the default 30-day freshness window is not appropriate. The readiness detail exposes only configured env names and evidence age.

## Billing Webhook Setup

Configure a Stripe webhook endpoint pointing at:

```text
https://api.example.com/webhooks/billing
```

Subscribe to checkout, subscription, and invoice payment failure lifecycle events that map to `checkout.session.completed`, `customer.subscription.created`, `customer.subscription.updated`, `customer.subscription.deleted`, and `invoice.payment_failed`.

Use Stripe replay tooling to verify that signed events are recorded in `billing_events`, update `organization_subscriptions`, and create `subscription_updated` audit events. Successful processing records a tenant-safe `rds.billing_webhook_replay` platform operation event without billing event ids, customer ids, subscription ids, organization ids, or invoice payloads. Stripe signatures must include a timestamp inside `SAAS_BILLING_WEBHOOK_TOLERANCE_SECONDS`, which defaults to five minutes.

Before setting production billing ready, verify the live lifecycle outside the app: paid checkout, billing portal access, subscription update/cancel reconciliation, invoice email delivery, tax behavior, accounting handoff, and dunning ownership. Record the evidence in an HTTPS operator document or dashboard through `SAAS_BILLING_OPERATIONS_URL`, set `SAAS_BILLING_OPERATIONS_VERIFIED_AT` with a timezone-aware ISO timestamp, and tune `SAAS_BILLING_OPERATIONS_MAX_AGE_DAYS` if the default 30-day freshness window is not appropriate. The `/ready` endpoint should fail production readiness when local billing is configured, required Stripe billing secrets/price ids are missing, the replay tolerance is invalid, billing operations evidence is missing/stale/insecure, or no recent successful billing replay evidence exists.

## Monitoring

The API exposes Prometheus-compatible aggregate metrics at:

```text
https://api.example.com/metrics
```

Call the endpoint with the configured bearer token:

```powershell
Invoke-WebRequest https://api.example.com/metrics -Headers @{ Authorization = "Bearer $env:RDS_METRICS_TOKEN" }
```

The metrics output intentionally avoids tenant labels and identifiers. It reports platform totals and status counts only, including notification outbox due/retry/failed counts plus oldest due-notification age, provider delivery event counts by normalized status, report schedule execution phases, oldest due-report-schedule age, schedule delivery-attempt status counts, scheduled-dispatch operation event counts and freshness, public pilot inquiry lifecycle counts, accepted/rate-limited public inquiry submission outcomes, sanitized rejection reasons without inquiry contact details or source hashes, worker heartbeat counts by all-tenant vs tenant-scoped processing scope, latest all-tenant worker heartbeat presence/age without worker ids, alert-smoke evidence freshness, artifact-store smoke-test evidence freshness, database smoke-test evidence freshness, tenant-isolation smoke-test evidence freshness, run-lifecycle smoke-test evidence freshness, worker run-lifecycle smoke-test evidence freshness, and CSP violation report counts by disposition and sanitized effective directive.

Ship the `rds.api` JSON request logs to the pilot aggregation stack and configure:

- `SAAS_LOG_AGGREGATION_URL` with the HTTPS dashboard, saved search, or runbook evidence,
- `SAAS_LOG_AGGREGATION_VERIFIED_AT` after confirming a smoke request appears with matching `X-Request-Id` and `traceparent`,
- `SAAS_LOG_AGGREGATION_MAX_AGE_DAYS` for the acceptable freshness window, defaulting to `30`,
- `SAAS_LOG_RETENTION_DAYS` for the operational log retention period.

Support-only inquiry follow-up APIs require a separate bearer token because they expose public pilot-request contact details:

```powershell
Invoke-WebRequest https://api.example.com/support/marketing-inquiries -Headers @{ Authorization = "Bearer $env:SAAS_SUPPORT_ACCESS_TOKEN" }
```

Use the support summary endpoint for aggregate queue counts by status and assignee. The public form can send a best-effort signed CRM webhook after storing the inquiry and throttles repeated successful submissions by hashed source key; tune `SAAS_MARKETING_INQUIRY_RATE_LIMIT_MAX` and `SAAS_MARKETING_INQUIRY_RATE_LIMIT_WINDOW_MINUTES` before launch. Rate-limited callers receive `429` plus `Retry-After`, the landing page shows an approximate retry message, and the blocked attempt is counted with a sanitized `source_rate_limit` rejection reason. Use the support PATCH endpoint to record inquiry status, assignee, and follow-up notes for any inquiries that still need manual follow-up. Filter the list with `unassigned=true` to find new leads without an owner, or `assigned_to=<assignee>` for a support queue. Use the inquiry activity endpoint to verify follow-up history without exposing inquiry messages or note bodies in activity records. Rotate `SAAS_SUPPORT_ACCESS_TOKEN` separately from monitoring credentials and do not expose these endpoints through the tenant frontend until platform-admin auth exists.

## Backups and Alerting

Before onboarding pilot tenants:

- provision automated PostgreSQL backups,
- provision object-storage versioning or backup replication for tenant artifacts,
- verify object-storage tenant-prefix access policy, encryption, versioning or replication, lifecycle/retention behavior, backup alignment, and recovery ownership, then configure `SAAS_ARTIFACT_STORE_GOVERNANCE_URL` plus `SAAS_ARTIFACT_STORE_GOVERNANCE_VERIFIED_AT`,
- verify the latest scheduled backup job and configure `SAAS_BACKUP_JOB_URL` plus `SAAS_BACKUP_JOB_VERIFIED_AT`,
- perform and record a restore test,
- configure `SAAS_BACKUP_RESTORE_TESTED_AT` with the successful restore-test timestamp,
- connect the monitoring stack to the alert webhook,
- deploy the pilot alert rules and configure `SAAS_ALERT_RULES_URL` plus `SAAS_ALERT_RULES_VERIFIED_AT`,
- configure `SAAS_ALERT_NOTIFICATION_DUE_AGE_SECONDS`, `SAAS_ALERT_NOTIFICATION_FAILED_COUNT`, `SAAS_ALERT_REPORT_SCHEDULE_DUE_AGE_SECONDS`, and `SAAS_ALERT_REPORT_SCHEDULE_FAILED_COUNT` to match the pilot support SLA,
- alert on `rds_notification_oldest_due_age_seconds` and `rds_notification_outbox_total{phase="failed"}`,
- alert on `rds_report_schedule_oldest_due_age_seconds` and `rds_report_schedules_total{phase="failed_last_delivery"}`,
- alert on failed scheduled-dispatch batches with `rds_platform_operation_events_total{event_type="report_schedule_dispatch",status="failed"}` and stale successful dispatch evidence through `rds_platform_report_schedule_dispatch_latest_success_age_seconds`,
- alert on stale or missing alert-smoke evidence through `rds_platform_alert_delivery_latest_success_present`, `rds_platform_alert_delivery_latest_success_age_seconds`, and `rds_platform_operation_events_total{event_type="alert_delivery_test",status="failed"}`,
- alert on stale or missing artifact-store smoke evidence through `rds_platform_artifact_store_test_latest_success_present`, `rds_platform_artifact_store_test_latest_success_age_seconds`, and `rds_platform_operation_events_total{event_type="artifact_store_test",status="failed"}`,
- alert on stale or missing database smoke evidence through `rds_platform_database_test_latest_success_present`, `rds_platform_database_test_latest_success_age_seconds`, and `rds_platform_operation_events_total{event_type="database_test",status="failed"}`,
- alert on stale or missing tenant-isolation smoke evidence through `rds_platform_tenant_isolation_test_latest_success_present`, `rds_platform_tenant_isolation_test_latest_success_age_seconds`, and `rds_platform_operation_events_total{event_type="tenant_isolation_test",status="failed"}`,
- alert on stale or missing run-lifecycle smoke evidence through `rds_platform_run_lifecycle_test_latest_success_present`, `rds_platform_run_lifecycle_test_latest_success_age_seconds`, and `rds_platform_operation_events_total{event_type="run_lifecycle_smoke_test",status="failed"}`,
- alert on stale or missing worker run-lifecycle smoke evidence through `rds_platform_worker_run_lifecycle_test_latest_success_present`, `rds_platform_worker_run_lifecycle_test_latest_success_age_seconds`, and `rds_platform_operation_events_total{event_type="worker_run_lifecycle_smoke_test",status="failed"}`,
- send a support-protected alert smoke test:

```powershell
Invoke-RestMethod https://api.example.com/support/alert-test `
  -Method POST `
  -Headers @{
    Authorization = "Bearer $env:SAAS_SUPPORT_ACCESS_TOKEN"
    "X-Request-Id" = "pilot-alert-smoke-001"
  }
```

Then verify the recorded platform evidence:

```powershell
Invoke-RestMethod "https://api.example.com/support/operation-events?event_type=rds.alert_delivery_test&status=succeeded" `
  -Headers @{ Authorization = "Bearer $env:SAAS_SUPPORT_ACCESS_TOKEN" }
```

- send a support-protected database smoke test:

```powershell
Invoke-RestMethod https://api.example.com/support/database-test `
  -Method POST `
  -Headers @{
    Authorization = "Bearer $env:SAAS_SUPPORT_ACCESS_TOKEN"
    "X-Request-Id" = "pilot-database-smoke-001"
  }
```

Then verify the recorded platform evidence:

```powershell
Invoke-RestMethod "https://api.example.com/support/operation-events?event_type=rds.database_test&status=succeeded" `
  -Headers @{ Authorization = "Bearer $env:SAAS_SUPPORT_ACCESS_TOKEN" }
```

The database smoke test writes and reads back a platform operation event through the configured metadata store. The response and stored operation evidence must not include database URLs, credentials, SQL text, tenant ids, user ids, or provider error details.

- send a support-protected tenant-isolation smoke test:

```powershell
Invoke-RestMethod https://api.example.com/support/tenant-isolation-test `
  -Method POST `
  -Headers @{
    Authorization = "Bearer $env:SAAS_SUPPORT_ACCESS_TOKEN"
    "X-Request-Id" = "pilot-tenant-isolation-smoke-001"
  }
```

Then verify the recorded platform evidence:

```powershell
Invoke-RestMethod "https://api.example.com/support/operation-events?event_type=rds.tenant_isolation_test&status=succeeded" `
  -Headers @{ Authorization = "Bearer $env:SAAS_SUPPORT_ACCESS_TOKEN" }
```

The tenant-isolation smoke test creates synthetic tenant A/B metadata inside a rollback-only transaction, verifies user organization, site, meter, upload, and cross-tenant site-filter boundaries, removes the synthetic records, and records only aggregate evidence. The response and stored operation evidence must not include tenant ids, user ids, site ids, storage keys, email addresses, or synthetic probe ids.

- send a support-protected run-lifecycle smoke test:

```powershell
Invoke-RestMethod https://api.example.com/support/run-lifecycle-test `
  -Method POST `
  -Headers @{
    Authorization = "Bearer $env:SAAS_SUPPORT_ACCESS_TOKEN"
    "X-Request-Id" = "pilot-run-lifecycle-smoke-001"
  }
```

Then verify the recorded platform evidence:

```powershell
Invoke-RestMethod "https://api.example.com/support/operation-events?event_type=rds.run_lifecycle_smoke_test&status=succeeded" `
  -Headers @{ Authorization = "Bearer $env:SAAS_SUPPORT_ACCESS_TOKEN" }
```

The run-lifecycle smoke test creates a synthetic tenant, writes a tiny energy CSV through the configured artifact store, executes the stored-upload analysis path, verifies run/job/report metadata and the generated report artifact, deletes the synthetic metadata and objects, and records only aggregate evidence. The response and stored operation evidence must not include tenant ids, user ids, site ids, storage keys, email addresses, source content, report content, or synthetic probe ids.

- send a support-protected worker run-lifecycle smoke test:

```powershell
Invoke-RestMethod https://api.example.com/support/worker-run-lifecycle-test `
  -Method POST `
  -Headers @{
    Authorization = "Bearer $env:SAAS_SUPPORT_ACCESS_TOKEN"
    "X-Request-Id" = "pilot-worker-run-lifecycle-smoke-001"
  }
```

Then verify the recorded platform evidence:

```powershell
Invoke-RestMethod "https://api.example.com/support/operation-events?event_type=rds.worker_run_lifecycle_smoke_test&status=succeeded" `
  -Headers @{ Authorization = "Bearer $env:SAAS_SUPPORT_ACCESS_TOKEN" }
```

The worker run-lifecycle smoke test creates a synthetic tenant, writes a tiny energy CSV through the configured artifact store, queues the stored-upload analysis run, drains one queued job through the worker job processor, verifies run/job/report metadata and the generated report artifact, deletes the synthetic metadata and objects, and records only aggregate evidence. The response and stored operation evidence must not include tenant ids, user ids, site ids, storage keys, email addresses, source content, report content, or synthetic probe ids.

- send a support-protected artifact-store smoke test:

```powershell
Invoke-RestMethod https://api.example.com/support/artifact-store-test `
  -Method POST `
  -Headers @{
    Authorization = "Bearer $env:SAAS_SUPPORT_ACCESS_TOKEN"
    "X-Request-Id" = "pilot-artifact-store-smoke-001"
  }
```

Then verify the recorded platform evidence:

```powershell
Invoke-RestMethod "https://api.example.com/support/operation-events?event_type=rds.artifact_store_test&status=succeeded" `
  -Headers @{ Authorization = "Bearer $env:SAAS_SUPPORT_ACCESS_TOKEN" }
```

The artifact smoke test writes, reads, and deletes a tiny synthetic object. The response and stored operation evidence must not include tenant ids, storage keys, bucket keys, object contents, or provider error details. The smoke test complements `SAAS_ARTIFACT_STORE_GOVERNANCE_URL`; it proves the API can use the store, while the governance evidence proves the bucket policy and lifecycle controls have been reviewed.

- confirm the alert runbook names ownership, escalation, and tenant communication steps.

The `/ready` endpoint fails production readiness when artifact-store governance evidence, backup policy, recent backup-job evidence, recent restore-test evidence, alert webhook, alert runbook, recent alert-rule evidence, notification alert-threshold configuration, recent successful alert-smoke evidence, recent successful database smoke evidence, recent successful tenant-isolation smoke evidence, recent successful run-lifecycle smoke evidence, recent successful worker run-lifecycle smoke evidence, or recent successful artifact-store smoke evidence is missing.

## Report Retention

Generated reports receive an `expires_at` value from the tenant subscription plan. The API hides expired report metadata from normal listings, rejects expired artifact reads, and `POST /organizations/{organization_id}/reports/retention/apply` deletes expired report artifacts plus locked evidence package artifacts through the tenant artifact-store boundary before marking reports retention-applied. Before pilot launch, test the retention route against a non-production tenant, confirm the returned evidence-package cleanup counts, and configure provider-side object lifecycle, backup retention, and archive policies as defense-in-depth controls.

## Privacy Settings

Before onboarding each pilot tenant, verify `GET/PATCH /organizations/{organization_id}/privacy-settings` in Settings. Record the tenant privacy contact, confirm whether subprocessor notifications should be enabled, decide whether anonymized product analytics sharing is allowed, and have an owner record data-processing acknowledgement. Keep legal DPA wording, privacy notice publication, subprocessor list publication, and deletion-request handling outside this technical smoke until reviewed by counsel.

For privacy operations, verify the owner-only Settings workflow can create and update `GET/POST/PATCH /organizations/{organization_id}/privacy-requests` records for export, deletion, and correction requests. Confirm request descriptions and resolution notes are visible only through the owner-managed request list and are not copied into audit metadata. Treat actual export delivery, account/data deletion, legal approval, backup retention impact, and provider-side lifecycle execution as manual runbook items until counsel and operations sign off.

## Security Headers and CORS

Verify the API only allows trusted frontend origins and emits hardening headers:

```powershell
Invoke-WebRequest https://api.example.com/health -Headers @{ Origin = "https://app.example.com" }
```

Confirm the response includes `Access-Control-Allow-Origin: https://app.example.com`, `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, and `Strict-Transport-Security`.

Verify the frontend origin also emits its edge hardening headers:

```powershell
Invoke-WebRequest https://app.example.com/
```

Confirm the response includes `Content-Security-Policy-Report-Only`, `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, `Permissions-Policy`, `Cross-Origin-Opener-Policy: same-origin`, and `Strict-Transport-Security`. When `NEXT_PUBLIC_API_BASE_URL` is configured, confirm the CSP includes `report-uri https://api.example.com/security/csp-reports`.

Send a synthetic CSP report to verify the collector and metrics path:

```powershell
Invoke-WebRequest https://api.example.com/security/csp-reports `
  -Method POST `
  -ContentType "application/csp-report" `
  -Body '{"csp-report":{"document-uri":"https://app.example.com/dashboard?tenant=redacted","effective-directive":"script-src","violated-directive":"script-src ''self''","blocked-uri":"https://blocked.example.com/app.js","disposition":"report","status-code":200}}'
```

Confirm `/metrics` increments `rds_csp_violation_reports_total{disposition="report"}` and `rds_csp_violation_reports_by_directive_total{directive="script-src"}` without exposing the document origin, blocked URL, query string, tenant id, or script sample. Review CSP reports with live Clerk sign-in, dashboard charts, and report download routes before enforcing CSP.

Use the support-protected summary endpoint to inspect aggregate CSP evidence before enforcement:

```powershell
Invoke-WebRequest https://api.example.com/support/csp-reports/summary `
  -Headers @{ Authorization = "Bearer $env:SAAS_SUPPORT_ACCESS_TOKEN" }
```

Then inspect recent sanitized records for the busiest disposition/directive buckets:

```powershell
Invoke-WebRequest "https://api.example.com/support/csp-reports?disposition=report&directive=script-src" `
  -Headers @{ Authorization = "Bearer $env:SAAS_SUPPORT_ACCESS_TOKEN" }
```

The summary response contains total count, latest report timestamp, and finite counts by disposition/directive only. The list response contains sanitized ids, disposition, directive bucket, blocked-URI category, document origin, status code, and timestamp only. Neither response may contain raw document paths, blocked URL hosts/paths, referrers, script samples, query strings, tenant ids, user ids, or session details.

After the review window, apply sanitized CSP report retention so operational evidence does not accumulate indefinitely:

```powershell
Invoke-WebRequest https://api.example.com/support/csp-reports/retention/apply `
  -Method POST `
  -Headers @{ Authorization = "Bearer $env:SAAS_SUPPORT_ACCESS_TOKEN" }
```

Set `SAAS_CSP_REPORT_RETENTION_DAYS` to the approved review window before launch. The retention response includes the cutoff timestamp, deleted count, and remaining count only.

## Security Review Evidence

Run the live security review and threat model after production Clerk, Stripe, SMTP, object storage, frontend, API, worker, and monitoring settings are in place. The review should include dependency review, secret-handling review, tenant isolation review, CORS origin review, response-header verification, edge/proxy verification, CSP report review, and deployment hardening.

After the review is signed off, configure `SAAS_SECURITY_REVIEW_URL` with an HTTPS link to the evidence record and set `SAAS_SECURITY_REVIEWED_AT` to the signed-off timestamp as an ISO-8601 datetime with a timezone. `SAAS_SECURITY_REVIEW_MAX_AGE_DAYS` defaults to `90`; `/ready` fails production readiness when the evidence is missing, insecure, timestamped without a timezone, in the future, or older than the configured window.

Record the tenant-isolation review separately in `SAAS_TENANT_ISOLATION_REVIEW_URL` and `SAAS_TENANT_ISOLATION_REVIEWED_AT`. This evidence should verify at least two live pilot-shaped tenants cannot cross-read or mutate each other's API data, database rows, object-storage artifacts, worker jobs, scheduled notifications, audit exports, metrics, logs, or support-only views. `SAAS_TENANT_ISOLATION_REVIEW_MAX_AGE_DAYS` defaults to `90`, and `/ready` reports a separate `tenant_isolation_review` failure when that evidence is missing or stale.

## Worker

Run the worker as a separate long-running process:

```powershell
python -m backend.worker_app
```

The worker reads the same `SAAS_DATABASE_URL` and artifact storage variables as the API. For production, run at least one worker instance after confirming the artifact store is shared object storage, not a local filesystem. Each processed batch records a `worker_heartbeats` row with the worker id, scope, batch settings, last attempt counts, and check-in time. Batches that attempt due report schedules also record `rds.report_schedule_dispatch` platform operation evidence with aggregate counts only; review it through `/support/operation-events?event_type=rds.report_schedule_dispatch` when checking scheduled delivery health.

Set `SAAS_WORKER_ENABLED=true` in the production API/worker environment once that long-running worker is deployed. Optionally set `SAAS_WORKER_ID` to a stable instance name. Keep `SAAS_WORKER_RUN_ONCE=false` or unset, leave `SAAS_WORKER_ORGANIZATION_ID` unset for the primary production worker so queued analysis and report-schedule work can run for all tenants, and keep `SAAS_WORKER_HEARTBEAT_MAX_AGE_SECONDS` aligned to your monitoring window.

Configure external process supervision before pilot launch:

- `SAAS_WORKER_SUPERVISOR_URL` with the HTTPS supervisor dashboard, process-manager service, or runbook evidence,
- `SAAS_WORKER_SUPERVISOR_VERIFIED_AT` after confirming restart policy, crash visibility, and alert routing for the worker process,
- `SAAS_WORKER_SUPERVISOR_MAX_AGE_DAYS` for the acceptable verification freshness window, defaulting to `30`.

The `/ready` endpoint fails production readiness when these worker deployment assumptions are not met, no recent supervisor evidence exists, or no recent all-tenant worker heartbeat is recorded.

Use the support heartbeat endpoints during pilot checks without exposing tenant ids:

```powershell
Invoke-WebRequest https://api.example.com/support/worker-heartbeats/summary `
  -Headers @{ Authorization = "Bearer $env:SAAS_SUPPORT_ACCESS_TOKEN" }

Invoke-WebRequest "https://api.example.com/support/worker-heartbeats?scope=all" `
  -Headers @{ Authorization = "Bearer $env:SAAS_SUPPORT_ACCESS_TOKEN" }
```

Set `SAAS_NOTIFICATION_MAX_ATTEMPTS` and `SAAS_NOTIFICATION_RETRY_DELAY_SECONDS` to match the SMTP provider's retry guidance. Failed notification sends stay pending until the next retry window, then become `failed` only after the max attempt count. The Runs page notification outbox shows attempt count, last attempt, next retry, sent time, and sanitized delivery error text.

Before setting production email ready, verify sender-domain/DNS authentication, live outbound SMTP delivery to a pilot inbox, bounce and complaint handling, retry behavior, provider dashboard access, and operational ownership. Record the evidence in an HTTPS operator document or dashboard through `SAAS_EMAIL_DELIVERY_VERIFICATION_URL`, set `SAAS_EMAIL_DELIVERY_VERIFIED_AT` with a timezone-aware ISO timestamp, and tune `SAAS_EMAIL_DELIVERY_VERIFICATION_MAX_AGE_DAYS` if the default 30-day freshness window is not appropriate.

Configure the SMTP provider or workflow relay to POST normalized delivery events to:

```text
https://api.example.com/webhooks/notifications/provider
```

Send `X-RDS-Timestamp` as an ISO-8601 timestamp with a timezone and sign `<timestamp>.<raw JSON body>` with `SAAS_NOTIFICATION_PROVIDER_WEBHOOK_SECRET`. Put the digest in `X-RDS-Signature: sha256=<digest>`. `SAAS_NOTIFICATION_PROVIDER_WEBHOOK_TOLERANCE_SECONDS` defaults to a five-minute replay window; keep provider clock skew well below that tolerance. The normalized payload includes `provider`, `event_id`, `notification_id`, `status`, `occurred_at`, and optional sanitized `detail`; use only `delivered`, `bounced`, `complained`, `deferred`, or `dropped` for status. Successful processing records a tenant-safe `rds.notification_provider_webhook_replay` platform operation event without provider event ids, notification ids, organization ids, or recipient emails. Alert on `rds_notification_provider_events_total{status="bounced"}`, `rds_notification_provider_events_total{status="complained"}`, and `rds_notification_provider_events_total{status="dropped"}`.

Before onboarding pilot tenants, save or verify each tenant's analysis settings at `GET/PATCH /organizations/{organization_id}/analysis-config`. Queued jobs snapshot the saved or tenant-asset-derived config when the run is requested, so update assumptions before starting a pilot baseline run.

For a smoke test that exits after one batch:

```powershell
$env:SAAS_WORKER_RUN_ONCE = "true"
python -m backend.worker_app
```

## Pilot Readiness Checklist

Before opening a tenant pilot, verify the owner-only Settings checklist backed by `GET/PATCH /organizations/{organization_id}/pilot-readiness`. Record status and evidence for DPA signature, privacy notice review, production auth, production billing, email delivery, storage/backups, tenant isolation review, and support ownership. Confirm `pilot_readiness_updated` audit events are created without copying notes or evidence URLs into audit metadata.

Treat the checklist as a tracking surface. Do not mark an item verified until the matching live provider, legal, security, backup, monitoring, or support evidence exists outside the app.

## Readiness Gate

For production, `/ready` should report `ready` only when:

- database migrations are applied against PostgreSQL,
- trusted HTTPS frontend origins and HSTS are configured,
- fresh HTTPS security-review evidence is configured,
- fresh HTTPS tenant-isolation review evidence is configured,
- Clerk session JWT verification is configured with fresh HTTPS auth-flow evidence,
- Clerk webhook signature verification is configured,
- SMTP notification delivery is configured with fresh HTTPS sender-domain/deliverability evidence,
- public pilot inquiry handoff is configured through support email or a CRM webhook signed with a secret of at least 32 characters,
- public inquiry rate-limit hashing secret is configured,
- Stripe billing checkout and webhook configuration is present with a billing webhook secret of at least 32 characters and a valid replay tolerance,
- backup policy, recent backup-job evidence, and recent restore-test evidence are configured,
- alert webhook, incident runbook, and recent alert-rule evidence are configured,
- a recent successful alert-smoke event is recorded in platform operation evidence,
- recent successful Clerk, billing, and notification-provider webhook replay evidence is recorded in platform operation evidence,
- recent successful artifact-store smoke-test evidence is recorded in platform operation evidence,
- recent successful database smoke-test evidence is recorded in platform operation evidence,
- recent successful tenant-isolation smoke-test evidence is recorded in platform operation evidence,
- recent successful run-lifecycle smoke-test evidence is recorded in platform operation evidence,
- recent successful worker run-lifecycle smoke-test evidence is recorded in platform operation evidence,
- metrics bearer-token protection is configured,
- log aggregation evidence is configured with a fresh verification timestamp and valid retention window,
- support bearer-token protection is configured,
- support owner email is configured for pilot incident ownership,
- CSP report retention is configured with a valid bounded review window,
- long-running all-tenant worker deployment is asserted with `SAAS_WORKER_ENABLED=true`, fresh worker supervisor evidence, and a recent all-tenant worker heartbeat,
- artifact storage is not local-only and has fresh HTTPS bucket-governance evidence.

Treat any `not_ready` response as a deployment blocker for a multi-client pilot.

## Remaining Pilot Work

- Provision live PostgreSQL and S3-compatible storage with backups, configure fresh `SAAS_ARTIFACT_STORE_GOVERNANCE_*` and `SAAS_BACKUP_JOB_*` evidence, run the support database, tenant-isolation, run-lifecycle, worker run-lifecycle, and artifact-store smoke tests, and confirm `rds.database_test`, `rds.tenant_isolation_test`, `rds.run_lifecycle_smoke_test`, `rds.worker_run_lifecycle_smoke_test`, and `rds.artifact_store_test` evidence is fresh.
- Configure real Clerk keys, account recovery, and webhook replay validation, then confirm `rds.clerk_webhook_replay` evidence is fresh.
- Provision and test the production SMTP provider, including sender-domain verification, retry policy, timestamp-bound provider webhook signatures, and provider replay into `POST /webhooks/notifications/provider`, then configure `SAAS_EMAIL_DELIVERY_VERIFICATION_URL`/`SAAS_EMAIL_DELIVERY_VERIFIED_AT` and confirm `rds.notification_provider_webhook_replay` evidence is fresh.
- Configure and replay-test the CRM/workflow webhook for public pilot inquiries when CRM handoff is used.
- Configure Stripe products/prices, invoice/tax/accounting settings, live webhook replay testing, dunning operations, and fresh `rds.billing_webhook_replay` evidence.
- Connect `/metrics` and the `rds.api` structured request logs to the pilot monitoring stack, add alerts for worker heartbeat age, due notification backlog, terminal notification failures, CSP report spikes, and API errors, configure operational log retention, set the `SAAS_LOG_AGGREGATION_*` and `SAAS_ALERT_RULES_*` evidence values, and verify `X-Request-Id` plus `traceparent` correlation during an API smoke test.
- Configure external worker process supervision, set the `SAAS_WORKER_SUPERVISOR_*` evidence values, and verify the supervisor restarts or alerts on worker failure.
- Run the live security review and tenant isolation review, then configure both signed-off evidence URLs and timestamps used by `/ready`.
