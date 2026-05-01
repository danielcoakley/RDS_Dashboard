# RDS SaaS Frontend

Next.js dashboard shell for the SaaS energy analytics platform.

## Commands

- `npm install`
- `npm run dev`
- `npm run typecheck`
- `npm run build`

The frontend consumes backend API contracts. It must not import Python analytics modules directly.

## Dependency Audit

`npm audit --audit-level=moderate` currently reports a transitive PostCSS advisory through Next.js. NPM suggests `npm audit fix --force`, but that would downgrade Next to an old breaking version, so do not apply it blindly. Recheck after Next publishes a safe patched release.
