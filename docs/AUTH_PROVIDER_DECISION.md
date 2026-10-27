# Auth Provider Decision

## Decision

Use **Clerk** as the preferred MVP authentication and organization provider for the SaaS platform.

## Why This Fits

- The selected frontend stack is Next.js, and Clerk has first-class Next.js organization flows.
- Clerk supports organization creation/switching patterns and organization role checks that map well to Owner, Manager, and Viewer.
- Clerk session tokens are validated by the FastAPI backend using JWT/JWKS verification through `RDS_AUTH_MODE=clerk`.
- It lets the project avoid building password reset, invite acceptance, and account security flows from scratch during MVP.

## Backend Integration Direction

- Keep the local `backend/auth_context.py` development seam for tests and local signup/login, but do not deploy it in production.
- Replace the temporary `X-User-Id` header with verified Clerk session claims by running with `RDS_AUTH_MODE=clerk`.
- Map Clerk user id to `users.id`.
- Map Clerk organization id to `organizations.id`.
- Map Clerk organization role/permission claims to platform RBAC roles.
- Initial claim-to-user mapping exists in `backend/auth_context.py`; the backend now verifies Clerk session JWTs with `CLERK_ISSUER` plus either `CLERK_JWT_KEY` or `CLERK_JWKS_URL`, and supports both legacy `org_id`/`org_role` and current nested `o` organization claim shapes.
- The Next.js frontend now supports opt-in Clerk mode: `/login`, `/signup`, and `/reset` render Clerk auth/account recovery components, the app is wrapped in `ClerkProvider`, `proxy.ts` runs `clerkMiddleware()` when keys are present, `/organizations` exposes Clerk organization switching, and server-side API calls use `auth().getToken()`.
- Local onboarding and auth entry pages still use the development seam when Clerk keys are absent, creating HTTP-only dev sessions for signup, login, and invite acceptance without introducing live Clerk credentials into the repo.
- Backend auth is now gated by `RDS_AUTH_MODE`. The default `dev` mode allows local `dev:` bearer tokens and `X-User-Id` headers for tests and development; `clerk` mode rejects those development credentials and requires verified Clerk JWT/JWKS configuration.
- The API readiness report remains `not_ready` for production auth until `RDS_AUTH_MODE=clerk`, valid Clerk verifier configuration, and fresh HTTPS auth-flow verification evidence are present.
- Organization-scoped API routes require a verified token active organization claim and reject requests when it does not match the requested organization path.
- Initial live Clerk tenant activation exists through `POST /auth/clerk/sync`, which upserts the verified Clerk user, active organization, and membership role into the backend `users`, `organizations`, and `memberships` tables. The frontend organization page calls this endpoint before opening the workspace.
- Signed Clerk webhook provisioning exists through `POST /webhooks/clerk`, backed by Svix signature verification and event handlers for user, organization, and organization membership create/update/delete events.
- Live Clerk production keys, Clerk dashboard webhook endpoint configuration, event replay validation, auth-flow/account recovery verification execution, and deployment configuration still need to be completed before production tenant data calls are fully self-service.

## Guardrails

- Do not hard-code Clerk secrets or tenant ids.
- Keep API authorization checks in backend service/API boundaries.
- Do not trust frontend-only RBAC for tenant data access.
- Keep local tests credential-free by generating test JWTs locally rather than calling live Clerk environments.
- Do not deploy with `RDS_AUTH_MODE=dev`; production must use verified provider tokens.
