# RDS SaaS Frontend

Next.js dashboard shell for the SaaS energy analytics platform.

## Commands

- `npm install`
- `npm run dev`
- `npm run typecheck`
- `npm run build`

The frontend consumes backend API contracts. It must not import Python analytics modules directly.

Set `NEXT_PUBLIC_API_BASE_URL` to the FastAPI service URL when running against a live backend. The local default is `http://localhost:8000`.

## Dependency Audit

`npm audit --audit-level=moderate` currently reports a transitive PostCSS advisory through Next.js. NPM suggests `npm audit fix --force`, but that would downgrade Next to an old breaking version, so do not apply it blindly. Recheck after Next publishes a safe patched release.
