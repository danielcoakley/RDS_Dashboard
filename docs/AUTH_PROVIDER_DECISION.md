# Auth Provider Decision

## Decision

Use **Clerk** as the preferred MVP authentication and organization provider for the SaaS platform.

## Why This Fits

- The selected frontend stack is Next.js, and Clerk has first-class Next.js organization flows.
- Clerk supports organization creation/switching patterns and organization role checks that map well to Owner, Manager, and Viewer.
- Clerk session tokens can be validated by the FastAPI backend using JWT/JWKS verification in a later implementation slice.
- It lets the project avoid building password reset, invite acceptance, and account security flows from scratch during MVP.

## Backend Integration Direction

- Keep the local `backend/auth_context.py` request-user seam until Clerk credentials and environments exist.
- Replace the temporary `X-User-Id` header with verified Clerk session claims.
- Map Clerk user id to `users.id`.
- Map Clerk organization id to `organizations.id`.
- Map Clerk organization role/permission claims to platform RBAC roles.

## Guardrails

- Do not hard-code Clerk secrets or tenant ids.
- Keep API authorization checks in backend service/API boundaries.
- Do not trust frontend-only RBAC for tenant data access.
- Keep local tests credential-free by testing token-claim mapping separately from live Clerk calls.
