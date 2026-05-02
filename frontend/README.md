# RDS SaaS Frontend

Next.js dashboard shell for the SaaS energy analytics platform.

## Commands

- `npm install`
- `npm run dev`
- `npm run typecheck`
- `npm run build`

The frontend consumes backend API contracts. It must not import Python analytics modules directly.

Set `NEXT_PUBLIC_API_BASE_URL` to the FastAPI service URL when running against a live backend. The local default is `http://localhost:8000`.

For local session-style testing, you can also set:

- `DEMO_USER_ID`
- `DEMO_ORGANIZATION_ID`
- `DEMO_AUTH_TOKEN` or `NEXT_PUBLIC_DEMO_AUTH_TOKEN`

`DEMO_AUTH_TOKEN` should be a bearer token payload created from Clerk-style claims using the local development token seam in `backend/auth_context.py`. If no auth token is set, the frontend falls back to the temporary `X-User-Id` header.

## Dependency Audit

`npm audit --audit-level=moderate` currently reports a transitive PostCSS advisory through Next.js. NPM suggests `npm audit fix --force`, but that would downgrade Next to an old breaking version, so do not apply it blindly. Recheck after Next publishes a safe patched release.
