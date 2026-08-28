# RDS SaaS Frontend

Next.js dashboard shell for the SaaS energy analytics platform.

## Commands

- `npm install`
- `npm run dev`
- `npm run typecheck`
- `npm run build`
- `npm run start`

The frontend consumes backend API contracts. It must not import Python analytics modules directly.

Set `NEXT_PUBLIC_API_BASE_URL` to the FastAPI service URL when running against a live backend. The local default is `http://localhost:8000`.

The production container in `frontend/Dockerfile` runs `npm run start` after `npm run build`. For the local pilot-shaped compose stack, run `docker compose up --build` from the repository root.

`next.config.ts` applies frontend hardening headers to all routes: `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, `Permissions-Policy`, `Cross-Origin-Opener-Policy`, `Cross-Origin-Resource-Policy`, `Strict-Transport-Security`, and a report-only Content Security Policy. It also disables the default `X-Powered-By` header. The CSP includes the configured `NEXT_PUBLIC_API_BASE_URL` origin plus Clerk origins for authenticated deployments; when `NEXT_PUBLIC_API_BASE_URL` is set, the policy sends violation reports to `POST /security/csp-reports` on the API. Keep CSP report-only until live Clerk/API traffic has been reviewed in the target hosting environment.

For local session-style testing, you can also set:

- `DEMO_USER_ID`
- `DEMO_ORGANIZATION_ID`
- `DEMO_AUTH_TOKEN` or `NEXT_PUBLIC_DEMO_AUTH_TOKEN`

`DEMO_AUTH_TOKEN` should be a bearer token payload created from Clerk-style claims using the local development token seam in `backend/auth_context.py`. If no auth token is set, the frontend falls back to the temporary `X-User-Id` header.

## Clerk Auth Mode

When Clerk environment keys are present, the frontend switches from local dev sessions to Clerk:

- `NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY`
- `CLERK_SECRET_KEY`
- `NEXT_PUBLIC_CLERK_SIGN_IN_URL=/login`
- `NEXT_PUBLIC_CLERK_SIGN_UP_URL=/signup`
- `NEXT_PUBLIC_CLERK_SIGN_IN_FALLBACK_REDIRECT_URL=/dashboard`
- `NEXT_PUBLIC_CLERK_SIGN_UP_FALLBACK_REDIRECT_URL=/dashboard`

In Clerk mode, `/login` and `/signup` render Clerk auth components, app pages require a signed-in user, `/organizations` renders Clerk's organization switcher, and server-side API calls use `auth().getToken()` as the bearer token for FastAPI.

The `/reset` page provides account recovery. In Clerk mode it renders the managed sign-in/recovery component; in local development it explains the seeded-session recovery path. The `/organizations` page activates the selected Clerk organization by calling `POST /auth/clerk/sync`, which upserts the verified user, active organization, and membership into the backend tenant metadata tables before opening the workspace. The backend also exposes `POST /webhooks/clerk` for signed Clerk user, organization, and membership sync. Production still needs live Clerk keys, account recovery email verification, and Clerk dashboard webhook endpoint setup.

## Local Auth and Onboarding

- `/signup` creates an owner user, creates an organization, and opens a local dev session.
- `/login` creates a local dev session for an existing user and optional organization membership.
- `/reset` documents local account recovery and links back into the local sign-in flow.
- `/organizations` lets a signed-in user choose which tenant context to open when they belong to more than one organization.
- `/join/[inviteId]` accepts an invite and then opens a local dev session. Settings asks the backend to generate opaque invite ids rather than creating predictable client-side tokens.
- `/logout` clears the local session cookies.

The frontend stores the local session in HTTP-only cookies and uses that session when loading tenant-scoped app pages.

The public landing page includes a pilot request form that submits to `POST /marketing/inquiries` without requiring a tenant session. The backend stores normalized inquiry metadata, rejects honeypot-filled submissions, recent duplicate email requests, and excessive submissions from the same hashed source key, returns only the created id/status/timestamp, can send a best-effort support handoff email when `SAAS_MARKETING_INQUIRY_RECIPIENT_EMAIL` is configured, can POST a signed CRM handoff when `SAAS_MARKETING_INQUIRY_CRM_WEBHOOK_URL` is configured, and exposes token-protected support follow-up APIs for queue filters, aggregate summaries, status, assignment, notes, and sanitized activity history. A browser support dashboard waits for platform-admin auth; internal sales dashboards remain future work.

The `/dashboard` page uses Apache ECharts to visualize tenant-scoped workflow data from the backend: required source-file readiness, run/job lifecycle status, and the configured meter/SEU mix.

Report pages show plan-derived report expiry metadata. Expired or retention-applied reports are filtered by the backend and cannot be opened through the artifact endpoint. Report detail pages visualize generated artifact evidence fields and legacy weather-normalized model outputs with ECharts, show skipped-meter model diagnostics and the raw artifact table for review, support tenant-scoped approval assignments, management review decisions, collaboration comments, delivery schedule controls with last-delivery state and recent delivery-attempt history, current JSON/PDF tenant-branded and section-templated evidence package downloads with tenant section labels/order, and locked evidence package snapshots gated by approved review state plus tenant policy. Managers can lock snapshots manually after approval, and the backend automatically locks the current JSON evidence package when the final pending approval assignment is completed as approved and the tenant approval policy enables it. JSON report downloads proxy through `/reports/[reportId]/download`; dynamic evidence packages proxy through `/reports/[reportId]/evidence-package` and `/reports/[reportId]/evidence-package/pdf`, with the PDF summary including model and skipped-meter diagnostics when artifact evidence is enabled; locked evidence package downloads proxy through `/reports/[reportId]/evidence-packages/[packageId]/download`. Due schedules and report approval assignments are processed through authenticated-access email notifications, and those emails include only allowlisted report digest metrics rather than arbitrary artifact fields.

The `/settings` page shows tenant subscription status and usage from `GET /organizations/{organization_id}/subscription`, including active users, pending invite seats, sites, runs this period, and report-retention days. Tenant owners can start billing checkout sessions for paid plans and open the provider billing portal after a customer id exists; local mode returns deterministic development URLs while production uses the configured payment adapter. Owners and managers can apply report retention from Settings and see expired-report, deleted/missing artifact, and locked evidence-package cleanup counts from `POST /organizations/{organization_id}/reports/retention/apply`. Settings also surfaces `GET/PATCH /organizations/{organization_id}/report-branding` for tenant evidence-pack title, accent color, logo URL, and support contact, `GET/PATCH /organizations/{organization_id}/evidence-package-template` for evidence section inclusion, labels, and render order, `GET/PATCH /organizations/{organization_id}/report-approval-policy` for evidence auto-lock and approval-threshold rules, `GET/POST/PATCH /organizations/{organization_id}/data-connectors` for tenant API/SFTP/utility-portal connector metadata and last-sync state, `POST /organizations/{organization_id}/data-connectors/{connector_id}/imports` for connector-supplied CSV content that becomes a tenant upload plus sync record, `GET/PATCH /organizations/{organization_id}/privacy-settings` so owners can record data-processing acknowledgement and tenant privacy preferences, `GET/POST/PATCH /organizations/{organization_id}/privacy-requests` for owner-managed export, deletion, and correction request tracking, and `GET/PATCH /organizations/{organization_id}/pilot-readiness` for tenant launch checklist evidence.

Tenant analysis settings include field-level controls for weather-normalized M&V enablement and minimum baseline/comparison record thresholds, while retaining the full JSON editor for the wider analytics config. Stored-upload report artifacts include the effective weather-model settings in the generated legacy source-package summary.

## Dependency Audit

`npm audit --audit-level=moderate` currently reports remaining moderate PostCSS/Next.js advisories with no non-force fix available from npm. `echarts` is pinned to `^6.1.0` to avoid the older XSS advisory. Recheck after Next publishes a safe patched release.
