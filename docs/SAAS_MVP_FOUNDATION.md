# SaaS MVP Foundation

This document turns `docs/SAAS_PLATFORM_PLAN.md` into the Milestone 0 foundation for the first SaaS implementation slices.

## Product Requirements

### Target users
- Energy managers who need ISO 50001 evidence and repeatable energy performance reporting.
- Consultants who manage analysis for multiple client organizations.
- Client stakeholders who need read-only access to reports and summaries.

### MVP outcome
Users can sign up, create or join an organization, upload tenant-scoped energy and weather data, run the existing analytics pipeline for that tenant, and retrieve ISO 50001-relevant summaries and report artifacts.

### MVP capabilities
- Public landing page with ISO 50001 value proposition and conversion path.
- Authentication flows for signup, login, logout, and password reset.
- Organization creation, invitations, and membership management.
- RBAC roles: Owner, Manager, Viewer.
- Tenant-scoped sites, meters, uploads, analysis runs, and reports.
- Run history with status, timestamps, and report artifact links.
- Basic admin views for tenant profile and user management.

### Non-goals for MVP
- Billing, dunning, SSO/SCIM, white-labeling, and enterprise connectors.
- Full migration away from Streamlit analytics UI.
- Production object storage or auth providers without explicit credentials and deployment decisions.

## Canonical Domain Model

### User
Represents a human account. Authentication may be external, but the platform stores a stable user id, email, display name, status, and timestamps.

### Organization
The tenant boundary. Every site, meter, upload, run, and report belongs to exactly one organization.

### Membership
Connects a user to an organization with one role: Owner, Manager, or Viewer. Access decisions must use membership plus tenant id.

### Site
A physical location or operating boundary inside an organization. Sites group meters and analysis runs.

### Meter
A configured energy meter for one site. It captures commodity, unit, source column, and whether it is a significant energy use meter.

### Upload
A tenant-scoped uploaded file or batch. Uploads record file category, storage key, checksum, status, and uploader.

### Run
An analytics execution for one organization and site. Runs record lifecycle status, input upload ids, timestamps, and error messages.

### Report
A generated artifact from a run. Reports record report type, storage key, creation time, and publishing state.

## Access Model

- Owner can manage organization settings, memberships, uploads, runs, and reports.
- Manager can manage sites, meters, uploads, runs, and reports.
- Viewer can read sites, meters, run history, and reports.
- All access checks must include organization id.
- Background jobs must carry organization id through every storage, database, and analytics call.

## Acceptance Criteria

- Two organizations can contain similarly named sites, meters, uploads, runs, and reports without id collisions or shared file paths.
- A user with membership in one organization cannot access resources from another organization.
- The current Streamlit dashboard still runs for the existing single-tenant workflow.
- New SaaS scaffolding is testable locally without live auth, database, storage, or billing services.
