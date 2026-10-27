import base64
import hmac
import json
import logging
from datetime import datetime, timedelta, timezone
from hashlib import sha256

from fastapi import HTTPException
from fastapi.testclient import TestClient
from fastapi.routing import APIRoute

from backend.api import (
    API_TITLE,
    AnalysisConfigUpdate,
    BillingCheckoutCreate,
    BillingPortalCreate,
    DataConnectorCreate,
    DataConnectorImportCreate,
    DataConnectorSyncRecord,
    DataConnectorUpdate,
    DevSessionCreate,
    LocalAnalysisRunCreate,
    OwnerOrganizationCreate,
    OrganizationReportApprovalPolicyUpdate,
    OrganizationEvidencePackageTemplateUpdate,
    OrganizationPilotReadinessUpdate,
    OrganizationProfileUpdate,
    OrganizationPrivacyRequestCreate,
    OrganizationPrivacyRequestStatusUpdate,
    OrganizationPrivacySettingsUpdate,
    OrganizationReportBrandingUpdate,
    OrganizationInviteAccept,
    OrganizationInviteCreate,
    OrganizationSubscriptionUpdate,
    MembershipRoleUpdate,
    MeterCreate,
    ReportApprovalAssignmentCreate,
    ReportApprovalAssignmentUpdate,
    ReportCommentCreate,
    ReportReviewCreate,
    ReportScheduleCreate,
    ReportScheduleUpdate,
    SiteCreate,
    RunCreate,
    StoredUploadAnalysisRunCreate,
    UploadContentCreate,
    UploadCreate,
    accept_invite,
    create_billing_checkout_summary,
    create_billing_portal_summary,
    create_data_connector_summary,
    create_dev_session,
    create_app,
    create_invite,
    create_meter_summary,
    create_privacy_request_summary,
    create_report_approval_assignment_summary,
    create_report_comment_summary,
    create_report_evidence_package_snapshot,
    create_report_review_summary,
    create_report_schedule_summary,
    create_run_summary,
    create_site_summary,
    create_upload_content_summary,
    create_upload_summary,
    execute_local_analysis_run,
    execute_stored_upload_analysis_run,
    export_report_evidence_package,
    export_report_evidence_package_pdf,
    export_audit_events,
    export_report_artifact,
    _report_evidence_package_pdf_sections,
    get_report_artifact,
    health_check,
    get_organization_summary,
    get_pilot_readiness_summary,
    get_privacy_settings_summary,
    get_evidence_package_template_summary,
    get_report_approval_policy_summary,
    get_report_branding_summary,
    get_upload_readiness_summary,
    get_analysis_config_summary,
    get_subscription_summary,
    import_data_connector_content_summary,
    list_data_connector_summaries,
    list_data_connector_sync_summaries,
    list_due_data_connector_summaries,
    list_audit_event_summaries,
    list_analysis_job_summaries,
    list_organization_invite_summaries,
    list_membership_summaries,
    list_meter_summaries,
    list_notification_summaries,
    list_privacy_request_summaries,
    list_report_approval_assignment_summaries,
    list_report_approver_summaries,
    list_report_comment_summaries,
    list_report_evidence_package_summaries,
    list_report_review_summaries,
    list_report_schedule_delivery_summaries,
    list_report_schedule_summaries,
    list_report_summaries,
    list_run_summaries,
    list_site_summaries,
    list_subscription_plan_summaries,
    list_upload_summaries,
    list_user_organization_summaries,
    onboard_owner_organization,
    queue_stored_upload_analysis_run,
    readiness_check,
    revoke_invite,
    sync_clerk_session,
    download_report_evidence_package_snapshot,
    remove_membership_summary,
    update_organization_summary,
    update_pilot_readiness_item_summary,
    update_privacy_request_summary,
    update_privacy_settings_summary,
    update_evidence_package_template_summary,
    update_report_approval_policy_summary,
    update_report_branding_summary,
    update_report_approval_assignment_summary,
    update_report_schedule_summary,
    update_analysis_config_summary,
    update_data_connector_summary,
    update_membership_summary,
    update_subscription_summary,
    user_id_for_organization_route,
    record_data_connector_sync_summary,
)
from backend.access_control import AccessDenied
from backend.auth_context import AuthenticatedUser
from backend.artifacts import InMemoryTenantArtifactStore
from backend.billing import (
    BILLING_PROVIDER_ENV,
    BILLING_WEBHOOK_SECRET_ENV,
    BILLING_WEBHOOK_TOLERANCE_SECONDS_ENV,
    STRIPE_BILLING_PROVIDER,
)
from backend.database import initialize_database
from backend.domain import (
    AuditAction,
    AuditEvent,
    Membership,
    Meter,
    PilotReadinessStatus,
    PrivacyRequestStatus,
    PrivacyRequestType,
    Report,
    ReportApprovalAssignmentStatus,
    ReportEvidencePackage,
    ReportReviewStatus,
    ReportSchedule,
    ReportScheduleDelivery,
    ReportScheduleDeliveryStatus,
    ReportScheduleFrequency,
    Role,
    Run,
    RunStatus,
    Site,
    Upload,
    UploadCategory,
    UploadStatus,
    User,
    normalize_upload_category,
)
from backend.observability import API_LOGGER_NAME, REQUEST_ID_HEADER, TRACEPARENT_HEADER
from backend.operations import BILLING_WEBHOOK_REPLAY_EVENT_TYPE
from backend.runtime_config import RUNTIME_ENV_ENV
from backend.security import ALLOWED_ORIGINS_ENV
from backend.store import SaaSStore
from backend.subscriptions import SubscriptionLimitExceeded
from analytics.config_loader import load_client_config


STRIPE_WEBHOOK_SECRET = "whsec_test_billing_replay_secret_123456"


def stripe_signature_headers(
    body: bytes,
    secret: str = STRIPE_WEBHOOK_SECRET,
    timestamp: int | None = None,
) -> dict[str, str]:
    timestamp_value = timestamp or int(datetime.now(timezone.utc).timestamp())
    signed_payload = f"{timestamp_value}.".encode("utf-8") + body
    digest = hmac.new(secret.encode("utf-8"), signed_payload, sha256).hexdigest()
    return {
        "Content-Type": "application/json",
        "Stripe-Signature": f"t={timestamp_value},v1={digest}",
    }


def test_api_app_registers_system_routes(monkeypatch):
    monkeypatch.delenv(RUNTIME_ENV_ENV, raising=False)
    monkeypatch.delenv(ALLOWED_ORIGINS_ENV, raising=False)
    app = create_app()
    routes = {
        route.path
        for route in app.routes
        if isinstance(route, APIRoute)
    }

    assert app.title == API_TITLE
    assert {
        "/health",
        "/ready",
        "/metrics",
        "/security/csp-reports",
        "/auth/dev/session",
        "/auth/clerk/sync",
        "/webhooks/clerk",
        "/webhooks/billing",
        "/billing/plans",
        "/organizations/onboard-owner",
        "/organizations/{organization_id}",
        "/organizations/{organization_id}/privacy-settings",
        "/organizations/{organization_id}/privacy-requests",
        "/organizations/{organization_id}/privacy-requests/{privacy_request_id}",
        "/organizations/{organization_id}/pilot-readiness",
        "/organizations/{organization_id}/pilot-readiness/{item_key}",
        "/organizations/{organization_id}/subscription",
        "/organizations/{organization_id}/billing/checkout",
        "/organizations/{organization_id}/billing/portal",
        "/organizations/{organization_id}/analysis-config",
        "/organizations/{organization_id}/data-connectors",
        "/organizations/{organization_id}/data-connectors/due",
        "/organizations/{organization_id}/data-connectors/{connector_id}",
        "/organizations/{organization_id}/data-connectors/{connector_id}/imports",
        "/organizations/{organization_id}/data-connectors/{connector_id}/syncs",
        "/organizations/{organization_id}/report-branding",
        "/organizations/{organization_id}/report-approval-policy",
        "/organizations/{organization_id}/evidence-package-template",
        "/organizations/{organization_id}/sites",
        "/organizations/{organization_id}/memberships",
        "/organizations/{organization_id}/memberships/{member_user_id}",
        "/organizations/{organization_id}/report-approvers",
        "/organizations/{organization_id}/invites",
        "/organizations/{organization_id}/invites/{invite_id}/revoke",
        "/organizations/{organization_id}/meters",
        "/organizations/{organization_id}/uploads",
        "/organizations/{organization_id}/uploads/readiness",
        "/organizations/{organization_id}/uploads/content",
        "/organizations/{organization_id}/runs",
        "/organizations/{organization_id}/analysis-jobs",
        "/organizations/{organization_id}/notifications",
        "/organizations/{organization_id}/reports",
        "/organizations/{organization_id}/reports/retention/apply",
        "/organizations/{organization_id}/reports/{report_id}/artifact",
        "/organizations/{organization_id}/reports/{report_id}/evidence-package",
        "/organizations/{organization_id}/reports/{report_id}/evidence-package/pdf",
        "/organizations/{organization_id}/reports/{report_id}/evidence-packages",
        "/organizations/{organization_id}/reports/{report_id}/evidence-packages/{package_id}/download",
        "/organizations/{organization_id}/reports/{report_id}/export",
        "/organizations/{organization_id}/reports/{report_id}/approval-assignments",
        "/organizations/{organization_id}/reports/{report_id}/approval-assignments/{assignment_id}",
        "/organizations/{organization_id}/reports/{report_id}/reviews",
        "/organizations/{organization_id}/reports/{report_id}/schedules",
        "/organizations/{organization_id}/reports/{report_id}/schedules/{schedule_id}",
        "/organizations/{organization_id}/reports/{report_id}/schedules/{schedule_id}/deliveries",
        "/organizations/{organization_id}/reports/{report_id}/comments",
        "/organizations/{organization_id}/audit-events",
        "/organizations/{organization_id}/audit-events/export",
        "/organizations/{organization_id}/runs/execute-local",
        "/organizations/{organization_id}/runs/execute-upload",
        "/organizations/{organization_id}/runs/queue-upload",
        "/support/alert-test",
        "/support/database-test",
        "/support/artifact-store-test",
        "/support/operation-events",
        "/support/worker-heartbeats",
        "/support/worker-heartbeats/summary",
        "/support/csp-reports",
        "/support/csp-reports/summary",
        "/support/csp-reports/retention/apply",
        "/invites/{invite_id}/accept",
        "/me/organizations",
    }.issubset(routes)


def test_invite_create_schema_does_not_advertise_client_supplied_ids(monkeypatch):
    monkeypatch.delenv(RUNTIME_ENV_ENV, raising=False)
    monkeypatch.delenv(ALLOWED_ORIGINS_ENV, raising=False)
    client = TestClient(
        create_app(
            store=SaaSStore(initialize_database()),
            artifact_store=InMemoryTenantArtifactStore(),
        )
    )

    response = client.get("/openapi.json")
    schema = response.json()["components"]["schemas"]["OrganizationInviteCreate"]

    assert response.status_code == 200
    assert "invite_id" not in schema["properties"]
    assert set(schema["required"]) == {"email", "role"}


def test_frontend_summary_schemas_do_not_advertise_storage_keys(monkeypatch):
    monkeypatch.delenv(RUNTIME_ENV_ENV, raising=False)
    monkeypatch.delenv(ALLOWED_ORIGINS_ENV, raising=False)
    client = TestClient(
        create_app(
            store=SaaSStore(initialize_database()),
            artifact_store=InMemoryTenantArtifactStore(),
        )
    )

    response = client.get("/openapi.json")
    schemas = response.json()["components"]["schemas"]

    assert response.status_code == 200
    assert "storage_key" not in schemas["UploadSummary"]["properties"]
    assert "storage_key" not in schemas["ReportSummary"]["properties"]
    assert "storage_key" not in schemas["ReportArtifactResponse"]["properties"]
    assert "report_storage_key" not in schemas["LocalAnalysisRunResponse"]["properties"]


def test_api_adds_security_headers_and_development_cors(monkeypatch):
    monkeypatch.delenv(RUNTIME_ENV_ENV, raising=False)
    monkeypatch.delenv(ALLOWED_ORIGINS_ENV, raising=False)
    client = TestClient(
        create_app(
            store=SaaSStore(initialize_database()),
            artifact_store=InMemoryTenantArtifactStore(),
        )
    )

    response = client.get("/health", headers={"Origin": "http://localhost:3000"})
    preflight = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["cross-origin-resource-policy"] == "same-site"
    assert "x-request-id" in response.headers
    assert TRACEPARENT_HEADER in response.headers
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert response.headers["access-control-expose-headers"] == "X-Request-Id, traceparent"
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "http://localhost:3000"
    assert "X-Request-Id" in preflight.headers["access-control-allow-headers"]
    assert TRACEPARENT_HEADER in preflight.headers["access-control-allow-headers"]


def test_api_adds_hsts_for_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv(ALLOWED_ORIGINS_ENV, "https://app.example.com")
    client = TestClient(
        create_app(
            store=SaaSStore(initialize_database()),
            artifact_store=InMemoryTenantArtifactStore(),
        )
    )

    response = client.get("/health", headers={"Origin": "https://app.example.com"})

    assert response.headers["strict-transport-security"] == "max-age=31536000; includeSubDomains"
    assert response.headers["access-control-allow-origin"] == "https://app.example.com"


def test_api_logs_structured_request_events_without_tenant_identifiers(caplog):
    client = TestClient(
        create_app(
            store=SaaSStore(initialize_database()),
            artifact_store=InMemoryTenantArtifactStore(),
        )
    )

    traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    with caplog.at_level(logging.INFO, logger=API_LOGGER_NAME):
        response = client.get(
            "/health",
            headers={REQUEST_ID_HEADER: "trace-123", TRACEPARENT_HEADER: traceparent},
        )

    records = [record for record in caplog.records if record.name == API_LOGGER_NAME]
    assert response.headers[REQUEST_ID_HEADER] == "trace-123"
    assert len(records) == 1
    payload = json.loads(records[0].message)
    assert payload["event"] == "api_request"
    assert payload["request_id"] == "trace-123"
    assert response.headers[TRACEPARENT_HEADER] == traceparent
    assert payload["trace_id"] == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert payload["span_id"] == "00f067aa0ba902b7"
    assert payload["method"] == "GET"
    assert payload["route"] == "/health"
    assert payload["status_code"] == 200
    assert payload["duration_ms"] >= 0

    caplog.clear()
    with caplog.at_level(logging.INFO, logger=API_LOGGER_NAME):
        denied = client.get(
            "/organizations/org_sensitive?token=secret",
            headers={
                REQUEST_ID_HEADER: "bad id with spaces",
                TRACEPARENT_HEADER: "bad trace with spaces",
                "X-User-Id": "user_sensitive",
            },
        )

    denied_records = [record for record in caplog.records if record.name == API_LOGGER_NAME]
    assert denied.status_code == 403
    assert denied.headers[REQUEST_ID_HEADER] != "bad id with spaces"
    assert denied.headers[TRACEPARENT_HEADER] != "bad trace with spaces"
    assert denied.headers[TRACEPARENT_HEADER].startswith("00-")
    assert len(denied_records) == 1
    denied_payload = json.loads(denied_records[0].message)
    assert denied_payload["route"] == "/organizations/{organization_id}"
    assert denied_payload["status_code"] == 403
    assert len(denied_payload["trace_id"]) == 32
    assert len(denied_payload["span_id"]) == 16
    assert "org_sensitive" not in denied_records[0].message
    assert "user_sensitive" not in denied_records[0].message
    assert "secret" not in denied_records[0].message
    assert "bad trace with spaces" not in denied_records[0].message


def test_system_endpoint_handlers_are_side_effect_free():
    assert health_check() == {"status": "ok"}
    readiness = readiness_check()
    assert readiness["status"] == "ready"
    assert readiness["service"] == "rds-saas-api"
    assert readiness["runtime_env"] == "development"


def test_user_id_for_organization_route_rejects_mismatched_auth_context():
    assert (
        user_id_for_organization_route(
            AuthenticatedUser(id="user_1", organization_id="org_1", auth_mode="clerk"),
            "org_1",
        )
        == "user_1"
    )
    assert user_id_for_organization_route(AuthenticatedUser(id="user_1"), "org_2") == "user_1"

    try:
        user_id_for_organization_route(
            AuthenticatedUser(id="user_1", organization_id="org_1"),
            "org_2",
        )
        assert False, "Expected organization context mismatch to fail"
    except HTTPException as exc:
        assert exc.status_code == 403
        assert "organization context" in exc.detail

    try:
        user_id_for_organization_route(
            AuthenticatedUser(id="user_1", auth_mode="clerk"),
            "org_1",
        )
        assert False, "Expected missing Clerk organization context to fail"
    except HTTPException as exc:
        assert exc.status_code == 403
        assert "active organization" in exc.detail


def test_create_dev_session_returns_bearer_token_for_known_user_and_membership():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="user_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )

    response = create_dev_session(
        DevSessionCreate(user_id="user_1", organization_id="org_1"),
        store,
    )

    assert response.user_id == "user_1"
    assert response.organization_id == "org_1"
    assert response.role == "owner"
    assert response.auth_token.startswith("dev:")


def test_create_dev_session_rejects_unknown_membership():
    store = SaaSStore(initialize_database())
    store.create_user(User(id="user_1", email="owner@example.com", display_name="Owner"))

    try:
        create_dev_session(DevSessionCreate(user_id="user_1", organization_id="org_1"), store)
        assert False, "Expected missing membership to fail"
    except ValueError as exc:
        assert "no membership" in str(exc)


def test_sync_clerk_session_creates_backend_metadata():
    store = SaaSStore(initialize_database())

    response = sync_clerk_session(
        AuthenticatedUser(
            id="user_123",
            organization_id="org_123",
            role="admin",
            auth_mode="clerk",
            email="owner@example.com",
            display_name="Owner User",
            organization_slug="example-energy",
            organization_name="Example Energy",
        ),
        store,
    )

    memberships = store.list_memberships(user_id="user_123", organization_id="org_123")

    assert response.user_id == "user_123"
    assert response.organization_id == "org_123"
    assert response.organization_slug == "example-energy"
    assert response.role == "owner"
    assert memberships[0].role == Role.OWNER


def test_sync_clerk_session_rejects_dev_auth_context():
    store = SaaSStore(initialize_database())

    try:
        sync_clerk_session(AuthenticatedUser(id="user_123", auth_mode="dev"), store)
        assert False, "Expected non-Clerk sync request to fail"
    except HTTPException as exc:
        assert exc.status_code == 401
        assert "Clerk session" in exc.detail


def test_onboard_owner_organization_endpoint_logic_creates_owner_membership():
    store = SaaSStore(initialize_database())
    payload = OwnerOrganizationCreate(
        user_id="user_1",
        email="owner@example.com",
        display_name="Owner",
        organization_id="org_1",
        organization_name="Example Energy",
        organization_slug="example-energy",
    )

    response = onboard_owner_organization(payload, store)
    memberships = store.list_memberships(user_id="user_1", organization_id="org_1")

    assert response.user_id == "user_1"
    assert response.organization_id == "org_1"
    assert response.role == "owner"
    assert memberships[0].role == Role.OWNER
    subscription = store.get_organization_subscription("org_1")
    assert subscription is not None
    assert subscription["plan_id"] == "trial"
    assert subscription["status"] == "trialing"


def test_subscription_plan_summaries_expose_seeded_active_tiers():
    store = SaaSStore(initialize_database())

    plans = list_subscription_plan_summaries(store)

    assert [plan.id for plan in plans] == ["trial", "team", "enterprise"]
    assert plans[0].monthly_price_cents == 0
    assert plans[1].included_sites == 10
    assert plans[2].monthly_run_limit is None


def test_subscription_summary_is_tenant_guarded_and_includes_usage():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_2",
            email="other@example.com",
            display_name="Other",
            organization_id="org_2",
            organization_name="Other Energy",
            organization_slug="other-energy",
        ),
        store,
    )
    store.create_site(Site(id="site_1", organization_id="org_1", name="Main Site", timezone="Europe/London"))

    summary = get_subscription_summary("owner_1", "org_1", store)

    assert summary.organization_id == "org_1"
    assert summary.plan.id == "trial"
    assert summary.status == "trialing"
    assert summary.usage.member_count == 1
    assert summary.usage.active_user_count == 1
    assert summary.usage.pending_invite_count == 0
    assert summary.usage.site_count == 1
    assert summary.usage.runs_this_period == 0

    try:
        get_subscription_summary("owner_2", "org_1", store)
        assert False, "Expected cross-tenant subscription read to be denied"
    except AccessDenied:
        pass


def test_subscription_seat_limit_reserves_pending_invites():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    store.conn.execute(
        "UPDATE subscription_plans SET included_users = ? WHERE id = ?",
        (2, "trial"),
    )

    create_invite(
        "owner_1",
        "org_1",
        OrganizationInviteCreate(
            email="viewer@example.com",
            role=Role.VIEWER,
        ),
        store,
    )
    summary = get_subscription_summary("owner_1", "org_1", store)

    assert summary.usage.active_user_count == 1
    assert summary.usage.pending_invite_count == 1
    try:
        create_invite(
            "owner_1",
            "org_1",
            OrganizationInviteCreate(
                email="second-viewer@example.com",
                role=Role.VIEWER,
            ),
            store,
        )
        assert False, "Expected pending invite seat reservation to enforce plan limit"
    except SubscriptionLimitExceeded as exc:
        assert "Member seat limit exceeded" in str(exc)


def test_subscription_site_limit_blocks_new_sites_and_returns_402_from_api():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    store.conn.execute(
        "UPDATE subscription_plans SET included_sites = ? WHERE id = ?",
        (1, "trial"),
    )

    created = create_site_summary(
        "owner_1",
        "org_1",
        SiteCreate(site_id="site_1", name="Main Site", timezone="Europe/London"),
        store,
    )

    assert created.id == "site_1"
    try:
        create_site_summary(
            "owner_1",
            "org_1",
            SiteCreate(site_id="site_2", name="Second Site", timezone="Europe/London"),
            store,
        )
        assert False, "Expected site creation over plan limit to fail"
    except SubscriptionLimitExceeded as exc:
        assert "Site limit exceeded" in str(exc)

    client = TestClient(create_app(store=store))
    response = client.post(
        "/organizations/org_1/sites",
        headers={"X-User-Id": "owner_1"},
        json={
            "site_id": "site_3",
            "name": "Third Site",
            "timezone": "Europe/London",
        },
    )

    assert response.status_code == 402
    assert "Site limit exceeded" in response.json()["detail"]


def test_subscription_monthly_run_limit_blocks_new_runs():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    store.conn.execute(
        "UPDATE subscription_plans SET monthly_run_limit = ? WHERE id = ?",
        (1, "trial"),
    )
    store.create_site(Site(id="site_1", organization_id="org_1", name="Main Site", timezone="Europe/London"))
    store.create_upload(
        Upload(
            id="upload_1",
            organization_id="org_1",
            site_id="site_1",
            uploaded_by_user_id="owner_1",
            category="energy",
            storage_key="tenants/org_1/sites/site_1/uploads/upload_1/energy.csv",
            checksum="abc123",
            status=UploadStatus.STORED,
        )
    )

    create_run_summary(
        "owner_1",
        "org_1",
        RunCreate(run_id="run_1", site_id="site_1", upload_ids=["upload_1"]),
        store,
    )

    try:
        create_run_summary(
            "owner_1",
            "org_1",
            RunCreate(run_id="run_2", site_id="site_1", upload_ids=["upload_1"]),
            store,
        )
        assert False, "Expected run creation over plan limit to fail"
    except SubscriptionLimitExceeded as exc:
        assert "Monthly run limit exceeded" in str(exc)


def test_subscription_plan_update_is_owner_only_and_audited():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    store.create_user(User(id="manager_1", email="manager@example.com", display_name="Manager"))
    store.add_membership(Membership(user_id="manager_1", organization_id="org_1", role=Role.MANAGER))

    updated = update_subscription_summary(
        "owner_1",
        "org_1",
        OrganizationSubscriptionUpdate(plan_id="team"),
        store,
    )

    subscription = store.get_organization_subscription("org_1")
    audit_events = store.list_audit_events("org_1")
    assert updated.plan.id == "team"
    assert updated.status == "active"
    assert subscription is not None
    assert subscription["plan_id"] == "team"
    assert [event["action"] for event in audit_events] == ["subscription_updated"]
    assert json.loads(audit_events[0]["metadata_json"]) == {
        "plan_id": "team",
        "previous_plan_id": "trial",
        "status": "active",
    }

    try:
        update_subscription_summary(
            "manager_1",
            "org_1",
            OrganizationSubscriptionUpdate(plan_id="enterprise"),
            store,
        )
        assert False, "Expected manager subscription update to be denied"
    except AccessDenied:
        pass


def test_billing_checkout_session_is_owner_only_and_audited(monkeypatch):
    monkeypatch.delenv("SAAS_BILLING_WEBHOOK_SECRET", raising=False)
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    store.create_user(User(id="manager_1", email="manager@example.com", display_name="Manager"))
    store.add_membership(Membership(user_id="manager_1", organization_id="org_1", role=Role.MANAGER))

    checkout = create_billing_checkout_summary(
        "owner_1",
        "org_1",
        BillingCheckoutCreate(plan_id="team"),
        store,
    )

    audit_events = store.list_audit_events("org_1")
    assert checkout.provider == "local"
    assert checkout.plan_id == "team"
    assert checkout.checkout_session_id.startswith("local_checkout_")
    assert "billing-checkout-created" in checkout.checkout_url
    assert [event["action"] for event in audit_events] == ["billing_checkout_created"]
    assert json.loads(audit_events[0]["metadata_json"])["plan_id"] == "team"

    try:
        create_billing_checkout_summary(
            "manager_1",
            "org_1",
            BillingCheckoutCreate(plan_id="team"),
            store,
        )
        assert False, "Expected manager checkout creation to be denied"
    except AccessDenied:
        pass


def test_billing_webhook_completion_updates_subscription_and_is_idempotent(monkeypatch):
    monkeypatch.delenv("SAAS_BILLING_WEBHOOK_SECRET", raising=False)
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    client = TestClient(create_app(store=store))
    event = {
        "id": "evt_checkout_completed",
        "type": "checkout.session.completed",
        "data": {
            "object": {
                "organization_id": "org_1",
                "plan_id": "team",
                "customer_id": "cus_org_1",
                "subscription_id": "sub_org_1",
                "status": "active",
                "current_period_start": "2026-07-20T00:00:00+00:00",
                "current_period_end": "2026-08-19T00:00:00+00:00",
            }
        },
    }

    response = client.post("/webhooks/billing", json=event)

    subscription = store.get_organization_subscription("org_1")
    billing_events = store.conn.execute("SELECT * FROM billing_events").fetchall()
    audit_events = store.list_audit_events("org_1")
    assert response.status_code == 200
    assert response.json()["action"] == "subscription_updated"
    assert response.json()["duplicate"] is False
    assert subscription is not None
    assert subscription["plan_id"] == "team"
    assert subscription["status"] == "active"
    assert subscription["external_customer_id"] == "cus_org_1"
    assert subscription["external_subscription_id"] == "sub_org_1"
    assert len(billing_events) == 1
    assert billing_events[0]["id"] == "evt_checkout_completed"
    assert [event["action"] for event in audit_events] == ["subscription_updated"]
    assert json.loads(audit_events[0]["metadata_json"])["source"] == "billing_webhook"
    operation_events = store.list_platform_operation_events(
        event_type=BILLING_WEBHOOK_REPLAY_EVENT_TYPE,
        event_status="succeeded",
    )
    assert len(operation_events) == 1
    assert operation_events[0]["detail"] == "provider=local action=subscription_updated duplicate=false"
    assert "org_1" not in json.dumps(dict(operation_events[0]))
    assert "evt_checkout_completed" not in json.dumps(dict(operation_events[0]))

    duplicate = client.post("/webhooks/billing", json=event)

    assert duplicate.status_code == 200
    assert duplicate.json()["action"] == "duplicate_ignored"
    assert duplicate.json()["duplicate"] is True
    assert len(store.conn.execute("SELECT * FROM billing_events").fetchall()) == 1
    assert len(store.list_audit_events("org_1")) == 1
    operation_events = store.list_platform_operation_events(
        event_type=BILLING_WEBHOOK_REPLAY_EVENT_TYPE,
        event_status="succeeded",
    )
    assert len(operation_events) == 2
    assert operation_events[0]["detail"] == "provider=local action=duplicate_ignored duplicate=true"


def test_stripe_billing_webhook_accepts_timestamp_bound_signature(monkeypatch):
    monkeypatch.setenv(BILLING_PROVIDER_ENV, STRIPE_BILLING_PROVIDER)
    monkeypatch.setenv(BILLING_WEBHOOK_SECRET_ENV, STRIPE_WEBHOOK_SECRET)
    monkeypatch.delenv(BILLING_WEBHOOK_TOLERANCE_SECONDS_ENV, raising=False)
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    client = TestClient(create_app(store=store))
    event = {
        "id": "evt_stripe_checkout_completed",
        "type": "checkout.session.completed",
        "data": {
            "object": {
                "metadata": {"organization_id": "org_1", "plan_id": "team"},
                "customer": "cus_org_1",
                "subscription": "sub_org_1",
                "status": "active",
            }
        },
    }
    body = json.dumps(event, separators=(",", ":"), sort_keys=True).encode("utf-8")

    response = client.post(
        "/webhooks/billing",
        content=body,
        headers=stripe_signature_headers(body),
    )

    subscription = store.get_organization_subscription("org_1")
    assert response.status_code == 200
    assert response.json()["provider"] == "stripe"
    assert response.json()["action"] == "subscription_updated"
    assert subscription is not None
    assert subscription["plan_id"] == "team"
    assert subscription["external_customer_id"] == "cus_org_1"
    assert subscription["external_subscription_id"] == "sub_org_1"


def test_stripe_billing_webhook_rejects_replayed_timestamp(monkeypatch):
    monkeypatch.setenv(BILLING_PROVIDER_ENV, STRIPE_BILLING_PROVIDER)
    monkeypatch.setenv(BILLING_WEBHOOK_SECRET_ENV, STRIPE_WEBHOOK_SECRET)
    monkeypatch.setenv(BILLING_WEBHOOK_TOLERANCE_SECONDS_ENV, "300")
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    client = TestClient(create_app(store=store))
    event = {
        "id": "evt_stripe_replayed",
        "type": "checkout.session.completed",
        "data": {
            "object": {
                "metadata": {"organization_id": "org_1", "plan_id": "team"},
                "customer": "cus_org_1",
                "subscription": "sub_org_1",
                "status": "active",
            }
        },
    }
    body = json.dumps(event, separators=(",", ":"), sort_keys=True).encode("utf-8")
    now = datetime.now(timezone.utc)

    for timestamp in (
        int((now - timedelta(seconds=3600)).timestamp()),
        int((now + timedelta(seconds=3600)).timestamp()),
    ):
        response = client.post(
            "/webhooks/billing",
            content=body,
            headers=stripe_signature_headers(body, timestamp=timestamp),
        )

        assert response.status_code == 400
        assert "outside tolerance" in response.json()["detail"]

    assert store.conn.execute("SELECT COUNT(*) FROM billing_events").fetchone()[0] == 0
    assert store.get_organization_subscription("org_1")["plan_id"] == "trial"


def test_billing_webhook_rejects_invalid_signature(monkeypatch):
    monkeypatch.setenv("SAAS_BILLING_WEBHOOK_SECRET", "secret")
    store = SaaSStore(initialize_database())
    client = TestClient(create_app(store=store))
    event = {
        "id": "evt_signed",
        "type": "checkout.session.completed",
        "data": {"object": {"organization_id": "org_1", "plan_id": "team"}},
    }
    body = json.dumps(event).encode("utf-8")

    response = client.post(
        "/webhooks/billing",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-RDS-Billing-Signature": "bad",
        },
    )

    assert response.status_code == 400
    assert "Invalid billing webhook signature" in response.json()["detail"]


def test_billing_dunning_marks_subscription_past_due_and_blocks_activity(monkeypatch):
    monkeypatch.delenv("SAAS_BILLING_WEBHOOK_SECRET", raising=False)
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    client = TestClient(create_app(store=store))
    completed_event = {
        "id": "evt_checkout_completed",
        "type": "checkout.session.completed",
        "data": {
            "object": {
                "organization_id": "org_1",
                "plan_id": "team",
                "customer_id": "cus_org_1",
                "subscription_id": "sub_org_1",
            }
        },
    }
    past_due_event = {
        "id": "evt_invoice_failed",
        "type": "invoice.payment_failed",
        "data": {
            "object": {
                "customer": "cus_org_1",
                "subscription": "sub_org_1",
            }
        },
    }

    assert client.post("/webhooks/billing", json=completed_event).status_code == 200
    response = client.post("/webhooks/billing", json=past_due_event)

    subscription = store.get_organization_subscription("org_1")
    assert response.status_code == 200
    assert response.json()["action"] == "subscription_updated"
    assert response.json()["status"] == "past_due"
    assert subscription is not None
    assert subscription["status"] == "past_due"
    assert subscription["plan_id"] == "team"
    try:
        create_site_summary(
            "owner_1",
            "org_1",
            SiteCreate(site_id="site_1", name="Main Site", timezone="Europe/London"),
            store,
        )
        assert False, "Expected past-due subscriptions to block new tenant activity"
    except SubscriptionLimitExceeded as exc:
        assert "past_due" in str(exc)


def test_billing_portal_session_requires_provider_customer_and_owner(monkeypatch):
    monkeypatch.delenv("SAAS_BILLING_WEBHOOK_SECRET", raising=False)
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    store.create_user(User(id="manager_1", email="manager@example.com", display_name="Manager"))
    store.add_membership(Membership(user_id="manager_1", organization_id="org_1", role=Role.MANAGER))

    try:
        create_billing_portal_summary(
            "owner_1",
            "org_1",
            BillingPortalCreate(return_url="/settings"),
            store,
        )
        assert False, "Expected portal creation without a provider customer to fail"
    except ValueError as exc:
        assert "payment-provider customer" in str(exc)

    client = TestClient(create_app(store=store))
    assert client.post(
        "/webhooks/billing",
        json={
            "id": "evt_checkout_completed",
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "organization_id": "org_1",
                    "plan_id": "team",
                    "customer_id": "cus_org_1",
                    "subscription_id": "sub_org_1",
                }
            },
        },
    ).status_code == 200

    portal = create_billing_portal_summary(
        "owner_1",
        "org_1",
        BillingPortalCreate(return_url="/settings"),
        store,
    )

    assert portal.provider == "local"
    assert portal.portal_session_id.startswith("local_portal_")
    assert "billing-portal-created" in portal.portal_url
    try:
        create_billing_portal_summary(
            "manager_1",
            "org_1",
            BillingPortalCreate(return_url="/settings"),
            store,
        )
        assert False, "Expected manager portal creation to be denied"
    except AccessDenied:
        pass


def test_list_user_organization_summaries_is_tenant_scoped():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="user_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="user_2",
            email="other@example.com",
            display_name="Other",
            organization_id="org_2",
            organization_name="Other Energy",
            organization_slug="other-energy",
        ),
        store,
    )

    organizations = list_user_organization_summaries("user_1", store)

    assert [organization.id for organization in organizations] == ["org_1"]


def test_get_organization_summary_requires_tenant_membership():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_2",
            email="other@example.com",
            display_name="Other",
            organization_id="org_2",
            organization_name="Other Energy",
            organization_slug="other-energy",
        ),
        store,
    )

    summary = get_organization_summary("owner_1", "org_1", store)

    assert summary.name == "Example Energy"
    assert summary.slug == "example-energy"
    try:
        get_organization_summary("owner_2", "org_1", store)
        assert False, "Expected cross-tenant organization profile read to be denied"
    except AccessDenied:
        pass


def test_update_organization_summary_is_owner_only_and_audited():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    store.create_user(User(id="manager_1", email="manager@example.com", display_name="Manager"))
    store.add_membership(Membership(user_id="manager_1", organization_id="org_1", role=Role.MANAGER))

    updated = update_organization_summary(
        "owner_1",
        "org_1",
        OrganizationProfileUpdate(name="Example Energy Group", slug="Example Energy Group"),
        store,
    )

    organization = store.get_organization("org_1")
    audit_events = store.list_audit_events("org_1")
    assert updated.name == "Example Energy Group"
    assert updated.slug == "example-energy-group"
    assert organization is not None
    assert organization["name"] == "Example Energy Group"
    assert [event["action"] for event in audit_events] == ["organization_updated"]
    assert json.loads(audit_events[0]["metadata_json"]) == {
        "name": "Example Energy Group",
        "previous_name": "Example Energy",
        "previous_slug": "example-energy",
        "slug": "example-energy-group",
    }

    try:
        update_organization_summary(
            "manager_1",
            "org_1",
            OrganizationProfileUpdate(name="Manager Rename", slug="manager-rename"),
            store,
        )
        assert False, "Expected manager organization profile update to be denied"
    except AccessDenied:
        pass


def test_update_organization_summary_rejects_duplicate_slug():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_2",
            email="other@example.com",
            display_name="Other",
            organization_id="org_2",
            organization_name="Other Energy",
            organization_slug="other-energy",
        ),
        store,
    )

    try:
        update_organization_summary(
            "owner_1",
            "org_1",
            OrganizationProfileUpdate(name="Example Energy", slug="other-energy"),
            store,
        )
        assert False, "Expected duplicate organization slug to fail"
    except ValueError as exc:
        assert "already in use" in str(exc)


def test_privacy_settings_are_tenant_guarded_owner_only_and_audited():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_2",
            email="other@example.com",
            display_name="Other",
            organization_id="org_2",
            organization_name="Other Energy",
            organization_slug="other-energy",
        ),
        store,
    )
    store.create_user(User(id="manager_1", email="manager@example.com", display_name="Manager"))
    store.add_membership(Membership(user_id="manager_1", organization_id="org_1", role=Role.MANAGER))

    default_settings = get_privacy_settings_summary("owner_1", "org_1", store)

    assert default_settings.organization_id == "org_1"
    assert default_settings.data_processing_acknowledged_at is None
    assert default_settings.data_processing_acknowledged_by_user_id is None
    assert default_settings.data_processing_contact_email is None
    assert default_settings.subprocessor_notifications_enabled is True
    assert default_settings.analytics_data_sharing_enabled is False

    updated = update_privacy_settings_summary(
        "owner_1",
        "org_1",
        OrganizationPrivacySettingsUpdate(
            data_processing_contact_email=" privacy@example.com ",
            subprocessor_notifications_enabled=False,
            analytics_data_sharing_enabled=True,
            acknowledge_data_processing=True,
        ),
        store,
    )
    row = store.get_organization_privacy_settings("org_1")
    audit_events = store.list_audit_events("org_1")
    audit_metadata = json.loads(audit_events[-1]["metadata_json"])

    assert updated.data_processing_contact_email == "privacy@example.com"
    assert updated.data_processing_acknowledged_at is not None
    assert updated.data_processing_acknowledged_by_user_id == "owner_1"
    assert updated.subprocessor_notifications_enabled is False
    assert updated.analytics_data_sharing_enabled is True
    assert row is not None
    assert row["data_processing_contact_email"] == "privacy@example.com"
    assert row["analytics_data_sharing_enabled"] == 1
    assert audit_events[-1]["action"] == "privacy_settings_updated"
    assert audit_events[-1]["resource_type"] == "privacy_settings"
    assert audit_metadata == {
        "analytics_data_sharing_enabled": True,
        "data_processing_acknowledged": True,
        "data_processing_contact_email_configured": True,
        "subprocessor_notifications_enabled": False,
    }
    assert "privacy@example.com" not in audit_events[-1]["metadata_json"]

    try:
        update_privacy_settings_summary(
            "manager_1",
            "org_1",
            OrganizationPrivacySettingsUpdate(
                data_processing_contact_email=None,
                subprocessor_notifications_enabled=True,
                analytics_data_sharing_enabled=False,
            ),
            store,
        )
        assert False, "Expected manager privacy settings update to be denied"
    except AccessDenied:
        pass

    try:
        get_privacy_settings_summary("owner_2", "org_1", store)
        assert False, "Expected cross-tenant privacy settings read to be denied"
    except AccessDenied:
        pass


def test_report_branding_is_tenant_guarded_owner_only_and_audited():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_2",
            email="other@example.com",
            display_name="Other",
            organization_id="org_2",
            organization_name="Other Energy",
            organization_slug="other-energy",
        ),
        store,
    )
    store.create_user(User(id="manager_1", email="manager@example.com", display_name="Manager"))
    store.add_membership(Membership(user_id="manager_1", organization_id="org_1", role=Role.MANAGER))

    default_branding = get_report_branding_summary("owner_1", "org_1", store)

    assert default_branding.organization_id == "org_1"
    assert default_branding.evidence_pack_title == "ISO 50001 Evidence Package"
    assert default_branding.primary_color == "#0f766e"
    assert default_branding.logo_url is None
    assert default_branding.support_email is None
    assert default_branding.updated_at is None

    updated = update_report_branding_summary(
        "owner_1",
        "org_1",
        OrganizationReportBrandingUpdate(
            evidence_pack_title=" Example Energy Audit Pack ",
            primary_color=" #123ABC ",
            logo_url=" https://cdn.example.com/logo.png ",
            support_email=" Support@Example.com ",
        ),
        store,
    )
    row = store.get_organization_report_branding("org_1")
    audit_events = store.list_audit_events("org_1")
    audit_metadata = json.loads(audit_events[-1]["metadata_json"])

    assert updated.evidence_pack_title == "Example Energy Audit Pack"
    assert updated.primary_color == "#123abc"
    assert updated.logo_url == "https://cdn.example.com/logo.png"
    assert updated.support_email == "support@example.com"
    assert row is not None
    assert row["evidence_pack_title"] == "Example Energy Audit Pack"
    assert row["primary_color"] == "#123abc"
    assert audit_events[-1]["action"] == "report_branding_updated"
    assert audit_events[-1]["resource_type"] == "report_branding"
    assert audit_metadata == {
        "evidence_pack_title_configured": True,
        "logo_url_configured": True,
        "primary_color": "#123abc",
        "support_email_configured": True,
    }
    assert "https://cdn.example.com/logo.png" not in audit_events[-1]["metadata_json"]
    assert "support@example.com" not in audit_events[-1]["metadata_json"]

    client = TestClient(create_app(store=store))
    route_get = client.get(
        "/organizations/org_1/report-branding",
        headers={"X-User-Id": "owner_1"},
    )
    route_update = client.patch(
        "/organizations/org_1/report-branding",
        headers={"X-User-Id": "owner_1"},
        json={
            "evidence_pack_title": "Tenant Board Pack",
            "primary_color": "#0a5f44",
            "logo_url": None,
            "support_email": None,
        },
    )
    route_denied = client.patch(
        "/organizations/org_1/report-branding",
        headers={"X-User-Id": "manager_1"},
        json={
            "evidence_pack_title": "Manager Pack",
            "primary_color": "#0a5f44",
            "logo_url": None,
            "support_email": None,
        },
    )
    route_invalid = client.patch(
        "/organizations/org_1/report-branding",
        headers={"X-User-Id": "owner_1"},
        json={
            "evidence_pack_title": "Invalid Pack",
            "primary_color": "green",
            "logo_url": None,
            "support_email": None,
        },
    )

    assert route_get.status_code == 200
    assert route_get.json()["evidence_pack_title"] == "Example Energy Audit Pack"
    assert route_update.status_code == 200
    assert route_update.json()["evidence_pack_title"] == "Tenant Board Pack"
    assert route_update.json()["primary_color"] == "#0a5f44"
    assert route_denied.status_code == 403
    assert route_invalid.status_code == 400

    try:
        get_report_branding_summary("owner_2", "org_1", store)
        assert False, "Expected cross-tenant report branding read to be denied"
    except AccessDenied:
        pass
    try:
        update_report_branding_summary(
            "manager_1",
            "org_1",
            OrganizationReportBrandingUpdate(
                evidence_pack_title="Manager Pack",
                primary_color="#0a5f44",
            ),
            store,
        )
        assert False, "Expected manager report branding update to be denied"
    except AccessDenied:
        pass


def test_report_approval_policy_is_tenant_guarded_owner_only_and_audited():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_2",
            email="other@example.com",
            display_name="Other",
            organization_id="org_2",
            organization_name="Other Energy",
            organization_slug="other-energy",
        ),
        store,
    )
    store.create_user(User(id="manager_1", email="manager@example.com", display_name="Manager"))
    store.add_membership(Membership(user_id="manager_1", organization_id="org_1", role=Role.MANAGER))

    default_policy = get_report_approval_policy_summary("owner_1", "org_1", store)

    assert default_policy.organization_id == "org_1"
    assert default_policy.auto_lock_evidence_on_final_approval is True
    assert default_policy.minimum_approval_assignments_for_lock == 0
    assert default_policy.updated_by_user_id is None
    assert default_policy.updated_at is None

    updated = update_report_approval_policy_summary(
        "owner_1",
        "org_1",
        OrganizationReportApprovalPolicyUpdate(
            auto_lock_evidence_on_final_approval=False,
            minimum_approval_assignments_for_lock=2,
        ),
        store,
    )
    row = store.get_organization_report_approval_policy("org_1")
    audit_events = store.list_audit_events("org_1")
    audit_metadata = json.loads(audit_events[-1]["metadata_json"])

    assert updated.auto_lock_evidence_on_final_approval is False
    assert updated.minimum_approval_assignments_for_lock == 2
    assert updated.updated_by_user_id == "owner_1"
    assert row is not None
    assert row["auto_lock_evidence_on_final_approval"] == 0
    assert row["minimum_approval_assignments_for_lock"] == 2
    assert audit_events[-1]["action"] == "report_approval_policy_updated"
    assert audit_events[-1]["resource_type"] == "report_approval_policy"
    assert audit_metadata == {
        "auto_lock_evidence_on_final_approval": False,
        "minimum_approval_assignments_for_lock": 2,
    }

    client = TestClient(create_app(store=store))
    route_get = client.get(
        "/organizations/org_1/report-approval-policy",
        headers={"X-User-Id": "manager_1"},
    )
    route_update = client.patch(
        "/organizations/org_1/report-approval-policy",
        headers={"X-User-Id": "owner_1"},
        json={
            "auto_lock_evidence_on_final_approval": True,
            "minimum_approval_assignments_for_lock": 1,
        },
    )
    route_denied = client.patch(
        "/organizations/org_1/report-approval-policy",
        headers={"X-User-Id": "manager_1"},
        json={
            "auto_lock_evidence_on_final_approval": True,
            "minimum_approval_assignments_for_lock": 1,
        },
    )
    route_invalid = client.patch(
        "/organizations/org_1/report-approval-policy",
        headers={"X-User-Id": "owner_1"},
        json={
            "auto_lock_evidence_on_final_approval": True,
            "minimum_approval_assignments_for_lock": -1,
        },
    )

    assert route_get.status_code == 200
    assert route_get.json()["minimum_approval_assignments_for_lock"] == 2
    assert route_update.status_code == 200
    assert route_update.json()["auto_lock_evidence_on_final_approval"] is True
    assert route_update.json()["minimum_approval_assignments_for_lock"] == 1
    assert route_denied.status_code == 403
    assert route_invalid.status_code == 422

    try:
        get_report_approval_policy_summary("owner_2", "org_1", store)
        assert False, "Expected cross-tenant report approval policy read to be denied"
    except AccessDenied:
        pass
    try:
        update_report_approval_policy_summary(
            "manager_1",
            "org_1",
            OrganizationReportApprovalPolicyUpdate(
                auto_lock_evidence_on_final_approval=True,
                minimum_approval_assignments_for_lock=0,
            ),
            store,
        )
        assert False, "Expected manager report approval policy update to be denied"
    except AccessDenied:
        pass


def test_evidence_package_template_is_tenant_guarded_owner_only_and_audited():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_2",
            email="other@example.com",
            display_name="Other",
            organization_id="org_2",
            organization_name="Other Energy",
            organization_slug="other-energy",
        ),
        store,
    )
    store.create_user(User(id="manager_1", email="manager@example.com", display_name="Manager"))
    store.add_membership(Membership(user_id="manager_1", organization_id="org_1", role=Role.MANAGER))

    default_template = get_evidence_package_template_summary("owner_1", "org_1", store)

    assert default_template.organization_id == "org_1"
    assert default_template.template_name == "Full ISO 50001 Evidence Package"
    assert default_template.include_report_artifact is True
    assert default_template.include_uploads is True
    assert default_template.include_approval_assignments is True
    assert default_template.include_reviews is True
    assert default_template.include_comments is True
    assert default_template.include_schedules is True
    assert default_template.section_labels == {
        "approval_assignments": "Approval Assignments",
        "comments": "Collaboration Evidence",
        "report_artifact": "Report Artifact Summary",
        "reviews": "Management Review Decisions",
        "schedules": "Scheduled Delivery Evidence",
        "uploads": "Upload Lineage",
    }
    assert default_template.section_order == [
        "report_artifact",
        "uploads",
        "approval_assignments",
        "reviews",
        "comments",
        "schedules",
    ]
    assert default_template.updated_by_user_id is None
    assert default_template.updated_at is None

    section_labels = {
        "approval_assignments": "Board sign-off",
        "comments": "Review discussion",
        "report_artifact": "Board report appendix",
        "reviews": "Management review record",
        "schedules": "Delivery register",
        "uploads": "Source file register",
    }
    section_order = [
        "approval_assignments",
        "reviews",
        "uploads",
        "report_artifact",
        "comments",
        "schedules",
    ]
    updated = update_evidence_package_template_summary(
        "owner_1",
        "org_1",
        OrganizationEvidencePackageTemplateUpdate(
            template_name=" Board summary pack ",
            include_report_artifact=False,
            include_uploads=True,
            include_approval_assignments=True,
            include_reviews=True,
            include_comments=False,
            include_schedules=False,
            section_labels=section_labels,
            section_order=section_order,
        ),
        store,
    )
    row = store.get_organization_evidence_package_template("org_1")
    audit_events = store.list_audit_events("org_1")
    audit_metadata = json.loads(audit_events[-1]["metadata_json"])

    assert updated.template_name == "Board summary pack"
    assert updated.include_report_artifact is False
    assert updated.include_uploads is True
    assert updated.include_comments is False
    assert updated.include_schedules is False
    assert updated.section_labels == section_labels
    assert updated.section_order == section_order
    assert updated.updated_by_user_id == "owner_1"
    assert row is not None
    assert row["template_name"] == "Board summary pack"
    assert row["include_report_artifact"] == 0
    assert row["include_comments"] == 0
    assert json.loads(row["section_labels_json"]) == section_labels
    assert json.loads(row["section_order_json"]) == section_order
    assert audit_events[-1]["action"] == "report_evidence_template_updated"
    assert audit_events[-1]["resource_type"] == "report_evidence_template"
    assert audit_metadata == {
        "enabled_section_count": 3,
        "include_approval_assignments": True,
        "include_comments": False,
        "include_report_artifact": False,
        "include_reviews": True,
        "include_schedules": False,
        "include_uploads": True,
        "section_labels_customized": True,
        "section_order_customized": True,
        "template_name_configured": True,
    }
    assert "Board summary pack" not in audit_events[-1]["metadata_json"]
    assert "Board sign-off" not in audit_events[-1]["metadata_json"]
    assert "Source file register" not in audit_events[-1]["metadata_json"]

    client = TestClient(create_app(store=store))
    route_get = client.get(
        "/organizations/org_1/evidence-package-template",
        headers={"X-User-Id": "manager_1"},
    )
    route_update = client.patch(
        "/organizations/org_1/evidence-package-template",
        headers={"X-User-Id": "owner_1"},
        json={
            "template_name": "Full pilot pack",
            "include_report_artifact": True,
            "include_uploads": True,
            "include_approval_assignments": True,
            "include_reviews": True,
            "include_comments": True,
            "include_schedules": False,
        },
    )
    route_denied = client.patch(
        "/organizations/org_1/evidence-package-template",
        headers={"X-User-Id": "manager_1"},
        json={
            "template_name": "Manager pack",
            "include_report_artifact": True,
            "include_uploads": True,
            "include_approval_assignments": True,
            "include_reviews": True,
            "include_comments": True,
            "include_schedules": True,
        },
    )
    route_invalid = client.patch(
        "/organizations/org_1/evidence-package-template",
        headers={"X-User-Id": "owner_1"},
        json={
            "template_name": " ",
            "include_report_artifact": True,
            "include_uploads": True,
            "include_approval_assignments": True,
            "include_reviews": True,
            "include_comments": True,
            "include_schedules": True,
        },
    )

    assert route_get.status_code == 200
    assert route_get.json()["template_name"] == "Board summary pack"
    assert route_get.json()["section_labels"]["approval_assignments"] == "Board sign-off"
    assert route_get.json()["section_order"][0] == "approval_assignments"
    assert route_update.status_code == 200
    assert route_update.json()["template_name"] == "Full pilot pack"
    assert route_update.json()["include_schedules"] is False
    assert route_update.json()["section_labels"]["report_artifact"] == "Report Artifact Summary"
    assert route_update.json()["section_order"][0] == "report_artifact"
    assert route_denied.status_code == 403
    assert route_invalid.status_code == 400

    try:
        get_evidence_package_template_summary("owner_2", "org_1", store)
        assert False, "Expected cross-tenant evidence template read to be denied"
    except AccessDenied:
        pass
    try:
        update_evidence_package_template_summary(
            "manager_1",
            "org_1",
            OrganizationEvidencePackageTemplateUpdate(
                template_name="Manager pack",
                include_report_artifact=True,
                include_uploads=True,
                include_approval_assignments=True,
                include_reviews=True,
                include_comments=True,
                include_schedules=True,
            ),
            store,
        )
        assert False, "Expected manager evidence template update to be denied"
    except AccessDenied:
        pass


def test_privacy_requests_are_tenant_guarded_owner_only_and_audited():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_2",
            email="other@example.com",
            display_name="Other",
            organization_id="org_2",
            organization_name="Other Energy",
            organization_slug="other-energy",
        ),
        store,
    )
    store.create_user(User(id="manager_1", email="manager@example.com", display_name="Manager"))
    store.add_membership(Membership(user_id="manager_1", organization_id="org_1", role=Role.MANAGER))

    assert list_privacy_request_summaries("owner_1", "org_1", store) == []

    created = create_privacy_request_summary(
        "owner_1",
        "org_1",
        OrganizationPrivacyRequestCreate(
            request_type=PrivacyRequestType.DELETION,
            subject_user_id="manager_1",
            description=" Delete old employee data ",
        ),
        store,
    )
    audit_events = store.list_audit_events("org_1")
    created_metadata = json.loads(audit_events[-1]["metadata_json"])

    assert created.organization_id == "org_1"
    assert created.requested_by_user_id == "owner_1"
    assert created.subject_user_id == "manager_1"
    assert created.request_type == PrivacyRequestType.DELETION
    assert created.status == PrivacyRequestStatus.OPEN
    assert created.description == "Delete old employee data"
    assert created.resolution_notes is None
    assert created.completed_at is None
    assert audit_events[-1]["action"] == "privacy_request_created"
    assert audit_events[-1]["resource_type"] == "privacy_request"
    assert created_metadata == {
        "description_configured": True,
        "request_type": "deletion",
        "status": "open",
        "subject_user_id_configured": True,
    }
    assert "Delete old employee data" not in audit_events[-1]["metadata_json"]

    listed = list_privacy_request_summaries("owner_1", "org_1", store)
    assert [request.id for request in listed] == [created.id]

    updated = update_privacy_request_summary(
        "owner_1",
        "org_1",
        created.id,
        OrganizationPrivacyRequestStatusUpdate(
            status=PrivacyRequestStatus.COMPLETED,
            resolution_notes=" Completed manually after legal review ",
        ),
        store,
    )
    updated_metadata = json.loads(store.list_audit_events("org_1")[-1]["metadata_json"])

    assert updated.status == PrivacyRequestStatus.COMPLETED
    assert updated.resolution_notes == "Completed manually after legal review"
    assert updated.completed_at is not None
    assert updated_metadata == {
        "previous_status": "open",
        "resolution_notes_configured": True,
        "status": "completed",
    }
    assert "Completed manually after legal review" not in store.list_audit_events("org_1")[-1]["metadata_json"]

    try:
        list_privacy_request_summaries("manager_1", "org_1", store)
        assert False, "Expected manager privacy request listing to be denied"
    except AccessDenied:
        pass

    try:
        create_privacy_request_summary(
            "owner_1",
            "org_1",
            OrganizationPrivacyRequestCreate(
                request_type=PrivacyRequestType.DATA_EXPORT,
                subject_user_id="owner_2",
                description=None,
            ),
            store,
        )
        assert False, "Expected cross-tenant subject user to be denied"
    except AccessDenied:
        pass

    try:
        update_privacy_request_summary(
            "owner_2",
            "org_2",
            created.id,
            OrganizationPrivacyRequestStatusUpdate(status=PrivacyRequestStatus.REJECTED),
            store,
        )
        assert False, "Expected cross-tenant privacy request update to be denied"
    except AccessDenied:
        pass


def test_pilot_readiness_is_tenant_guarded_owner_only_and_audited():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_2",
            email="other@example.com",
            display_name="Other",
            organization_id="org_2",
            organization_name="Other Energy",
            organization_slug="other-energy",
        ),
        store,
    )
    store.create_user(User(id="manager_1", email="manager@example.com", display_name="Manager"))
    store.add_membership(Membership(user_id="manager_1", organization_id="org_1", role=Role.MANAGER))

    default_summary = get_pilot_readiness_summary("owner_1", "org_1", store)

    assert default_summary.organization_id == "org_1"
    assert default_summary.total_count == 8
    assert default_summary.verified_count == 0
    assert [item.status for item in default_summary.items] == [PilotReadinessStatus.PENDING] * 8

    updated = update_pilot_readiness_item_summary(
        "owner_1",
        "org_1",
        "storage_backup_verified",
        OrganizationPilotReadinessUpdate(
            status=PilotReadinessStatus.VERIFIED,
            evidence_url=" https://ops.example.com/restore-test ",
            notes=" Restore tested with tenant-safe backup ",
        ),
        store,
    )
    row = store.get_organization_pilot_readiness_item("org_1", "storage_backup_verified")
    audit_events = store.list_audit_events("org_1")
    audit_metadata = json.loads(audit_events[-1]["metadata_json"])
    updated_item = next(
        item for item in updated.items if item.item_key == "storage_backup_verified"
    )

    assert updated.verified_count == 1
    assert updated_item.status == PilotReadinessStatus.VERIFIED
    assert updated_item.evidence_url == "https://ops.example.com/restore-test"
    assert updated_item.notes == "Restore tested with tenant-safe backup"
    assert row is not None
    assert row["status"] == "verified"
    assert row["updated_by_user_id"] == "owner_1"
    assert audit_events[-1]["action"] == "pilot_readiness_updated"
    assert audit_events[-1]["resource_type"] == "pilot_readiness"
    assert audit_metadata == {
        "evidence_url_configured": True,
        "item_key": "storage_backup_verified",
        "notes_configured": True,
        "previous_status": "pending",
        "status": "verified",
    }
    assert "Restore tested with tenant-safe backup" not in audit_events[-1]["metadata_json"]
    assert "https://ops.example.com/restore-test" not in audit_events[-1]["metadata_json"]

    try:
        get_pilot_readiness_summary("manager_1", "org_1", store)
        assert False, "Expected manager pilot readiness read to be denied"
    except AccessDenied:
        pass

    try:
        get_pilot_readiness_summary("owner_2", "org_1", store)
        assert False, "Expected cross-tenant pilot readiness read to be denied"
    except AccessDenied:
        pass

    try:
        update_pilot_readiness_item_summary(
            "owner_1",
            "org_1",
            "unsupported_gate",
            OrganizationPilotReadinessUpdate(status=PilotReadinessStatus.BLOCKED),
            store,
        )
        assert False, "Expected unsupported pilot readiness item to fail"
    except ValueError as exc:
        assert "Unsupported pilot readiness item" in str(exc)


def test_analysis_config_summaries_are_tenant_guarded_and_audited():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_2",
            email="other@example.com",
            display_name="Other",
            organization_id="org_2",
            organization_name="Other Energy",
            organization_slug="other-energy",
        ),
        store,
    )
    store.create_site(Site(id="site_1", organization_id="org_1", name="Main Site", timezone="Europe/London"))
    store.create_meter(
        Meter(
            id="meter_1",
            organization_id="org_1",
            site_id="site_1",
            display_name="Main Electricity",
            commodity="electricity",
            unit="kWh",
            source_column="Main Electricity",
            is_seu=True,
        )
    )
    config = get_analysis_config_summary("owner_1", "org_1", store, site_id="site_1")

    assert config.source == "tenant_assets"
    assert config.config["client"]["id"] == "org_1"
    assert config.config["meters"][0]["source_column"] == "Main Electricity"

    saved = update_analysis_config_summary(
        "owner_1",
        "org_1",
        AnalysisConfigUpdate(config=config.config),
        store,
    )

    assert saved.source == "saved"
    assert saved.updated_by_user_id == "owner_1"
    assert [event["action"] for event in store.list_audit_events("org_1")] == [
        "analysis_config_updated"
    ]

    try:
        get_analysis_config_summary("owner_2", "org_1", store, site_id="site_1")
        assert False, "Expected cross-tenant analysis config read to be denied"
    except AccessDenied:
        pass


def test_data_connectors_are_tenant_guarded_asset_managed_and_audited():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_2",
            email="other@example.com",
            display_name="Other",
            organization_id="org_2",
            organization_name="Other Energy",
            organization_slug="other-energy",
        ),
        store,
    )
    store.create_user(User(id="manager_1", email="manager@example.com", display_name="Manager"))
    store.create_user(User(id="viewer_1", email="viewer@example.com", display_name="Viewer"))
    store.add_membership(Membership(user_id="manager_1", organization_id="org_1", role=Role.MANAGER))
    store.add_membership(Membership(user_id="viewer_1", organization_id="org_1", role=Role.VIEWER))
    store.create_site(Site(id="site_1", organization_id="org_1", name="Main Site", timezone="Europe/London"))
    store.create_site(Site(id="site_2", organization_id="org_2", name="Other Site", timezone="Europe/London"))
    store.create_site(Site(id="site_3", organization_id="org_1", name="Second Site", timezone="Europe/London"))
    store.create_upload(
        Upload(
            id="upload_1",
            organization_id="org_1",
            site_id="site_1",
            uploaded_by_user_id="manager_1",
            category="energy",
            storage_key="tenants/org_1/sites/site_1/uploads/upload_1/energy.csv",
            checksum="abc123",
            status=UploadStatus.STORED,
        )
    )
    store.create_upload(
        Upload(
            id="upload_hdd",
            organization_id="org_1",
            site_id="site_1",
            uploaded_by_user_id="manager_1",
            category="hdd",
            storage_key="tenants/org_1/sites/site_1/uploads/upload_hdd/hdd.csv",
            checksum="def456",
            status=UploadStatus.STORED,
        )
    )
    store.create_upload(
        Upload(
            id="upload_other_site",
            organization_id="org_1",
            site_id="site_3",
            uploaded_by_user_id="manager_1",
            category="energy",
            storage_key="tenants/org_1/sites/site_3/uploads/upload_other_site/energy.csv",
            checksum="ghi789",
            status=UploadStatus.STORED,
        )
    )
    store.create_upload(
        Upload(
            id="upload_other_tenant",
            organization_id="org_2",
            site_id="site_2",
            uploaded_by_user_id="owner_2",
            category="energy",
            storage_key="tenants/org_2/sites/site_2/uploads/upload_other_tenant/energy.csv",
            checksum="jkl012",
            status=UploadStatus.STORED,
        )
    )

    created = create_data_connector_summary(
        "manager_1",
        "org_1",
        DataConnectorCreate(
            connector_id=" connector_1 ",
            site_id="site_1",
            name=" Main utility SFTP ",
            connector_type="sftp",
            source_category="energy",
            schedule="weekly",
            external_reference="sftp://example.com/private/path",
            credential_reference="secret/data-connectors/org_1/connector_1",
            next_sync_at="2026-08-01T09:00:00+00:00",
        ),
        store,
    )
    listed = list_data_connector_summaries("viewer_1", "org_1", store)
    audit_events = store.list_audit_events("org_1")
    create_metadata = json.loads(audit_events[-1]["metadata_json"])

    assert created.id == "connector_1"
    assert created.name == "Main utility SFTP"
    assert created.connector_type.value == "sftp"
    assert created.source_category == "energy"
    assert created.status.value == "draft"
    assert created.schedule.value == "weekly"
    assert created.external_reference == "sftp://example.com/private/path"
    assert created.credential_reference_configured is True
    assert listed[0].id == "connector_1"
    assert audit_events[-1]["action"] == "data_connector_created"
    assert create_metadata == {
        "connector_type": "sftp",
        "credential_reference_configured": True,
        "external_reference_configured": True,
        "next_sync_configured": True,
        "schedule": "weekly",
        "source_category": "energy",
        "status": "draft",
    }
    assert "Main utility SFTP" not in audit_events[-1]["metadata_json"]
    assert "sftp://example.com/private/path" not in audit_events[-1]["metadata_json"]
    assert "secret/data-connectors" not in audit_events[-1]["metadata_json"]

    updated = update_data_connector_summary(
        "owner_1",
        "org_1",
        "connector_1",
        DataConnectorUpdate(
            name="Utility portal handoff",
            status="active",
            schedule="monthly",
            external_reference=None,
            next_sync_at=None,
        ),
        store,
    )
    update_metadata = json.loads(store.list_audit_events("org_1")[-1]["metadata_json"])

    assert updated.name == "Utility portal handoff"
    assert updated.status.value == "active"
    assert updated.schedule.value == "monthly"
    assert updated.external_reference is None
    assert updated.credential_reference_configured is True
    assert update_metadata == {
        "connector_type": "sftp",
        "credential_reference_configured": True,
        "external_reference_configured": False,
        "next_sync_configured": False,
        "schedule": "monthly",
        "source_category": "energy",
        "status": "active",
    }
    assert "Utility portal handoff" not in store.list_audit_events("org_1")[-1]["metadata_json"]

    sync = record_data_connector_sync_summary(
        "manager_1",
        "org_1",
        "connector_1",
        DataConnectorSyncRecord(
            status="succeeded",
            upload_id="upload_1",
            records_processed=42,
            status_reason="Imported interval data",
            started_at="2026-08-05T10:00:00+00:00",
            completed_at="2026-08-05T10:01:00+00:00",
        ),
        store,
    )
    syncs = list_data_connector_sync_summaries("viewer_1", "org_1", "connector_1", store)
    due_connectors = list_due_data_connector_summaries(
        "viewer_1",
        "org_1",
        store,
        due_at="2026-09-05T10:01:00+00:00",
    )
    synced_connector = store.get_data_connector("connector_1")
    sync_metadata = json.loads(store.list_audit_events("org_1")[-1]["metadata_json"])

    assert sync.connector_id == "connector_1"
    assert sync.status.value == "succeeded"
    assert sync.upload_id == "upload_1"
    assert sync.records_processed == 42
    assert sync.status_reason == "Imported interval data"
    assert syncs[0].id == sync.id
    assert due_connectors[0].id == "connector_1"
    assert synced_connector["last_sync_status"] == "succeeded"
    assert synced_connector["last_sync_at"] == "2026-08-05T10:01:00+00:00"
    assert synced_connector["next_sync_at"] == "2026-09-05T10:01:00+00:00"
    assert store.list_audit_events("org_1")[-1]["action"] == "data_connector_sync_recorded"
    assert sync_metadata == {
        "connector_type": "sftp",
        "records_processed": 42,
        "schedule": "monthly",
        "source_category": "energy",
        "status": "succeeded",
        "status_reason_configured": True,
        "upload_configured": True,
    }
    assert "Imported interval data" not in store.list_audit_events("org_1")[-1]["metadata_json"]
    assert "sftp://example.com/private/path" not in store.list_audit_events("org_1")[-1]["metadata_json"]
    assert "secret/data-connectors" not in store.list_audit_events("org_1")[-1]["metadata_json"]

    try:
        record_data_connector_sync_summary(
            "manager_1",
            "org_1",
            "connector_1",
            DataConnectorSyncRecord(status="succeeded", upload_id="upload_hdd"),
            store,
        )
        assert False, "Expected connector sync upload category mismatch to fail"
    except ValueError as exc:
        assert "category" in str(exc)
    try:
        record_data_connector_sync_summary(
            "manager_1",
            "org_1",
            "connector_1",
            DataConnectorSyncRecord(status="succeeded", upload_id="upload_other_site"),
            store,
        )
        assert False, "Expected connector sync upload site mismatch to fail"
    except ValueError as exc:
        assert "site" in str(exc)
    try:
        record_data_connector_sync_summary(
            "manager_1",
            "org_1",
            "connector_1",
            DataConnectorSyncRecord(status="succeeded", upload_id="upload_other_tenant"),
            store,
        )
        assert False, "Expected cross-tenant connector sync upload to be denied"
    except AccessDenied:
        pass

    client = TestClient(create_app(store=store))
    route_list = client.get(
        "/organizations/org_1/data-connectors",
        headers={"X-User-Id": "viewer_1"},
    )
    route_create = client.post(
        "/organizations/org_1/data-connectors",
        headers={"X-User-Id": "manager_1"},
        json={
            "connector_id": "connector_2",
            "site_id": "site_1",
            "name": "Utility API",
            "connector_type": "api",
            "source_category": "hdd",
            "status": "active",
            "schedule": "daily",
            "next_sync_at": "2026-08-01T09:00:00+00:00",
        },
    )
    route_due = client.get(
        "/organizations/org_1/data-connectors/due?due_at=2026-08-01T09:05:00%2B00:00",
        headers={"X-User-Id": "viewer_1"},
    )
    route_sync_list = client.get(
        "/organizations/org_1/data-connectors/connector_1/syncs",
        headers={"X-User-Id": "viewer_1"},
    )
    route_sync_record = client.post(
        "/organizations/org_1/data-connectors/connector_1/syncs",
        headers={"X-User-Id": "manager_1"},
        json={
            "status": "failed",
            "records_processed": 0,
            "status_reason": "Provider timeout",
            "completed_at": "2026-08-06T10:00:00+00:00",
        },
    )
    route_sync_denied = client.post(
        "/organizations/org_1/data-connectors/connector_1/syncs",
        headers={"X-User-Id": "viewer_1"},
        json={
            "status": "skipped",
            "status_reason": "Viewer cannot record sync state",
        },
    )
    route_update = client.patch(
        "/organizations/org_1/data-connectors/connector_2",
        headers={"X-User-Id": "manager_1"},
        json={
            "name": "Utility API paused",
            "status": "paused",
            "schedule": "manual",
            "external_reference": "api:utility",
        },
    )
    route_denied = client.post(
        "/organizations/org_1/data-connectors",
        headers={"X-User-Id": "viewer_1"},
        json={
            "connector_id": "connector_denied",
            "site_id": "site_1",
            "name": "Denied connector",
            "connector_type": "api",
            "source_category": "energy",
        },
    )
    route_cross_site = client.post(
        "/organizations/org_1/data-connectors",
        headers={"X-User-Id": "manager_1"},
        json={
            "connector_id": "connector_cross",
            "site_id": "site_2",
            "name": "Cross tenant connector",
            "connector_type": "api",
            "source_category": "energy",
        },
    )

    assert route_list.status_code == 200
    assert route_list.json()[0]["id"] == "connector_1"
    assert route_create.status_code == 200
    assert route_create.json()["connector_type"] == "api"
    assert route_create.json()["credential_reference_configured"] is False
    assert route_due.status_code == 200
    assert [connector["id"] for connector in route_due.json()] == ["connector_2"]
    assert route_sync_list.status_code == 200
    assert route_sync_list.json()[0]["id"] == sync.id
    assert route_sync_record.status_code == 200
    assert route_sync_record.json()["status"] == "failed"
    assert route_sync_denied.status_code == 403
    assert route_update.status_code == 200
    assert route_update.json()["status"] == "paused"
    assert route_denied.status_code == 403
    assert route_cross_site.status_code == 403

    try:
        list_data_connector_summaries("owner_2", "org_1", store)
        assert False, "Expected cross-tenant connector read to be denied"
    except AccessDenied:
        pass
    try:
        create_data_connector_summary(
            "viewer_1",
            "org_1",
            DataConnectorCreate(
                connector_id="connector_viewer",
                site_id="site_1",
                name="Viewer connector",
                connector_type="api",
                source_category="energy",
            ),
            store,
        )
        assert False, "Expected viewer connector creation to be denied"
    except AccessDenied:
        pass


def test_data_connector_import_stores_tenant_upload_and_records_sync_evidence():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    store.create_user(User(id="manager_1", email="manager@example.com", display_name="Manager"))
    store.create_user(User(id="viewer_1", email="viewer@example.com", display_name="Viewer"))
    store.add_membership(Membership(user_id="manager_1", organization_id="org_1", role=Role.MANAGER))
    store.add_membership(Membership(user_id="viewer_1", organization_id="org_1", role=Role.VIEWER))
    store.create_site(Site(id="site_1", organization_id="org_1", name="Main Site", timezone="Europe/London"))
    create_data_connector_summary(
        "manager_1",
        "org_1",
        DataConnectorCreate(
            connector_id="connector_1",
            site_id="site_1",
            name="Utility API",
            connector_type="api",
            source_category="energy",
            status="active",
            schedule="daily",
            external_reference="api://utility/private",
            credential_reference="secret/data-connectors/org_1/connector_1",
            next_sync_at="2026-08-01T09:00:00+00:00",
        ),
        store,
    )
    create_data_connector_summary(
        "manager_1",
        "org_1",
        DataConnectorCreate(
            connector_id="connector_draft",
            site_id="site_1",
            name="Draft Utility API",
            connector_type="api",
            source_category="energy",
            status="draft",
            schedule="daily",
        ),
        store,
    )
    energy_csv = "Date,Main Electricity\n2026-08-01,100\n2026-08-02,105\n"
    imported = import_data_connector_content_summary(
        "manager_1",
        "org_1",
        "connector_1",
        DataConnectorImportCreate(
            upload_id="connector_upload_1",
            filename="interval-energy.csv",
            content_base64=base64.b64encode(energy_csv.encode("utf-8")).decode("ascii"),
            records_processed=2,
            status_reason="Imported from provider export",
            started_at="2026-08-01T09:00:00+00:00",
            completed_at="2026-08-01T09:01:00+00:00",
        ),
        store,
        artifact_store,
    )
    upload = store.get_upload("connector_upload_1")
    connector = store.get_data_connector("connector_1")
    audit_events = store.list_audit_events("org_1")
    upload_metadata = json.loads(audit_events[-2]["metadata_json"])
    sync_metadata = json.loads(audit_events[-1]["metadata_json"])

    assert imported.upload.id == "connector_upload_1"
    assert imported.upload.category == "energy"
    assert imported.sync.status.value == "succeeded"
    assert imported.sync.upload_id == "connector_upload_1"
    assert imported.sync.records_processed == 2
    assert imported.connector.last_sync_status.value == "succeeded"
    assert upload["site_id"] == "site_1"
    assert artifact_store.get_text("org_1", upload["storage_key"]) == energy_csv
    assert connector["last_sync_at"] == "2026-08-01T09:01:00+00:00"
    assert connector["next_sync_at"] == "2026-08-02T09:01:00+00:00"
    assert upload_metadata == {
        "artifact_location_configured": "true",
        "category": "energy",
        "site_id": "site_1",
        "source": "data_connector",
    }
    assert sync_metadata == {
        "connector_type": "api",
        "records_processed": 2,
        "schedule": "daily",
        "source_category": "energy",
        "status": "succeeded",
        "status_reason_configured": True,
        "upload_configured": True,
    }
    assert "Imported from provider export" not in audit_events[-1]["metadata_json"]
    assert "api://utility/private" not in audit_events[-1]["metadata_json"]
    assert "secret/data-connectors" not in audit_events[-1]["metadata_json"]
    assert "storage_key" not in audit_events[-2]["metadata_json"]

    try:
        import_data_connector_content_summary(
            "manager_1",
            "org_1",
            "connector_1",
            DataConnectorImportCreate(
                upload_id="connector_upload_bad",
                filename="hdd.csv",
                content_base64=base64.b64encode(
                    b"Date,HDD\n2026-08-01,12\n"
                ).decode("ascii"),
            ),
            store,
            artifact_store,
        )
        assert False, "Expected source-contract mismatch to fail before storage"
    except ValueError as exc:
        assert "energy connector import failed" in str(exc)
    assert store.get_upload("connector_upload_bad") is None

    client = TestClient(create_app(store=store, artifact_store=artifact_store))
    route_import = client.post(
        "/organizations/org_1/data-connectors/connector_1/imports",
        headers={"X-User-Id": "manager_1"},
        json={
            "filename": "route-energy.csv",
            "content_base64": base64.b64encode(
                b"Date,Main Electricity\n2026-08-03,110\n"
            ).decode("ascii"),
            "records_processed": 1,
            "completed_at": "2026-08-03T09:00:00+00:00",
        },
    )
    route_denied = client.post(
        "/organizations/org_1/data-connectors/connector_1/imports",
        headers={"X-User-Id": "viewer_1"},
        json={
            "filename": "viewer-energy.csv",
            "content_base64": base64.b64encode(
                b"Date,Main Electricity\n2026-08-03,110\n"
            ).decode("ascii"),
        },
    )
    route_inactive = client.post(
        "/organizations/org_1/data-connectors/connector_draft/imports",
        headers={"X-User-Id": "manager_1"},
        json={
            "filename": "draft-energy.csv",
            "content_base64": base64.b64encode(
                b"Date,Main Electricity\n2026-08-03,110\n"
            ).decode("ascii"),
        },
    )

    assert route_import.status_code == 200
    assert route_import.json()["upload"]["category"] == "energy"
    assert route_import.json()["sync"]["status"] == "succeeded"
    assert route_import.json()["connector"]["last_sync_status"] == "succeeded"
    assert route_denied.status_code == 403
    assert route_inactive.status_code == 400


def test_site_and_meter_summaries_require_tenant_membership():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="user_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="user_2",
            email="other@example.com",
            display_name="Other",
            organization_id="org_2",
            organization_name="Other Energy",
            organization_slug="other-energy",
        ),
        store,
    )
    store.create_site(Site(id="site_1", organization_id="org_1", name="Main Site", timezone="Europe/London"))
    store.create_meter(
        Meter(
            id="meter_1",
            organization_id="org_1",
            site_id="site_1",
            display_name="Main Electricity",
            commodity="electricity",
            unit="kWh",
            source_column="Main Electricity",
            is_seu=True,
        )
    )

    sites = list_site_summaries("user_1", "org_1", store)
    meters = list_meter_summaries("user_1", "org_1", store)

    assert [site.id for site in sites] == ["site_1"]
    assert [meter.id for meter in meters] == ["meter_1"]

    try:
        list_site_summaries("user_2", "org_1", store)
        assert False, "Expected cross-tenant site listing to be denied"
    except AccessDenied:
        pass


def test_membership_summaries_require_membership_management_permission():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_2",
            email="other@example.com",
            display_name="Other Owner",
            organization_id="org_2",
            organization_name="Other Energy",
            organization_slug="other-energy",
        ),
        store,
    )
    store.create_user(User(id="viewer_1", email="viewer@example.com", display_name="Viewer"))
    store.create_user(User(id="manager_1", email="manager@example.com", display_name="Manager"))
    store.add_membership(
        Membership(
            user_id="manager_1",
            organization_id="org_1",
            role=Role.MANAGER,
            invited_by_user_id="owner_1",
        )
    )
    store.add_membership(
        Membership(
            user_id="viewer_1",
            organization_id="org_1",
            role=Role.VIEWER,
            invited_by_user_id="owner_1",
        )
    )

    members = list_membership_summaries("owner_1", "org_1", store)

    assert [member.user_id for member in members] == ["manager_1", "owner_1", "viewer_1"]

    try:
        list_membership_summaries("manager_1", "org_1", store)
        assert False, "Expected manager membership access to be denied"
    except AccessDenied:
        pass

    try:
        list_membership_summaries("viewer_1", "org_1", store)
        assert False, "Expected viewer membership access to be denied"
    except AccessDenied:
        pass


def test_membership_role_updates_and_removal_are_owner_only_and_audited():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    store.create_user(User(id="manager_1", email="manager@example.com", display_name="Manager"))
    store.create_user(User(id="viewer_1", email="viewer@example.com", display_name="Viewer"))
    store.add_membership(
        Membership(
            user_id="manager_1",
            organization_id="org_1",
            role=Role.MANAGER,
            invited_by_user_id="owner_1",
        )
    )
    store.add_membership(
        Membership(
            user_id="viewer_1",
            organization_id="org_1",
            role=Role.VIEWER,
            invited_by_user_id="owner_1",
        )
    )

    updated = update_membership_summary(
        "owner_1",
        "org_1",
        "viewer_1",
        MembershipRoleUpdate(role=Role.MANAGER),
        store,
    )
    removed = remove_membership_summary("owner_1", "org_1", "manager_1", store)

    assert updated.user_id == "viewer_1"
    assert updated.role == "manager"
    assert removed.user_id == "manager_1"
    assert removed.role == "manager"
    assert store.list_memberships(user_id="manager_1", organization_id="org_1") == []
    assert store.list_memberships(user_id="viewer_1", organization_id="org_1")[0].role == Role.MANAGER
    audit_events = store.list_audit_events("org_1")
    assert [event["action"] for event in audit_events] == [
        "membership_role_updated",
        "membership_removed",
    ]
    assert json.loads(audit_events[0]["metadata_json"]) == {
        "previous_role": "viewer",
        "role": "manager",
        "target_user_id": "viewer_1",
    }
    assert json.loads(audit_events[1]["metadata_json"]) == {
        "previous_role": "manager",
        "report_schedule_recipient_reassignment_count": 0,
        "report_schedule_transfer_count": 0,
        "target_user_id": "manager_1",
    }

    try:
        update_membership_summary(
            "viewer_1",
            "org_1",
            "owner_1",
            MembershipRoleUpdate(role=Role.VIEWER),
            store,
        )
        assert False, "Expected manager role update to be denied"
    except AccessDenied:
        pass

    try:
        remove_membership_summary("owner_1", "org_1", "owner_1", store)
        assert False, "Expected owner self-removal to be denied"
    except ValueError as exc:
        assert "own membership" in str(exc)


def test_member_removal_transfers_owned_report_schedules_to_actor():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_2",
            email="other@example.com",
            display_name="Other Owner",
            organization_id="org_2",
            organization_name="Other Energy",
            organization_slug="other-energy",
        ),
        store,
    )
    store.create_user(User(id="manager_1", email="manager@example.com", display_name="Manager"))
    store.add_membership(
        Membership(
            user_id="manager_1",
            organization_id="org_1",
            role=Role.MANAGER,
            invited_by_user_id="owner_1",
        )
    )
    store.add_membership(
        Membership(
            user_id="manager_1",
            organization_id="org_2",
            role=Role.MANAGER,
            invited_by_user_id="owner_2",
        )
    )
    for organization_id, owner_id, site_id, upload_id, run_id, report_id, schedule_id in [
        ("org_1", "manager_1", "site_1", "upload_1", "run_1", "report_1", "schedule_1"),
        ("org_2", "manager_1", "site_2", "upload_2", "run_2", "report_2", "schedule_2"),
    ]:
        store.create_site(
            Site(
                id=site_id,
                organization_id=organization_id,
                name=f"{organization_id} Site",
                timezone="UTC",
            )
        )
        store.create_upload(
            Upload(
                id=upload_id,
                organization_id=organization_id,
                site_id=site_id,
                uploaded_by_user_id=owner_id,
                category=UploadCategory.ENERGY,
                storage_key=f"tenants/{organization_id}/sites/{site_id}/uploads/{upload_id}/energy.csv",
                checksum=f"checksum-{upload_id}",
                status=UploadStatus.STORED,
            )
        )
        store.create_run(
            Run(
                id=run_id,
                organization_id=organization_id,
                site_id=site_id,
                requested_by_user_id=owner_id,
                upload_ids=[upload_id],
                status=RunStatus.SUCCEEDED,
            )
        )
        store.create_report(
            Report(
                id=report_id,
                organization_id=organization_id,
                run_id=run_id,
                report_type="iso_summary",
                storage_key=f"tenants/{organization_id}/reports/{report_id}.json",
                is_published=True,
            )
        )
        store.create_report_schedule(
            ReportSchedule(
                id=schedule_id,
                organization_id=organization_id,
                report_id=report_id,
                created_by_user_id=owner_id,
                frequency=ReportScheduleFrequency.MONTHLY,
                recipient_email="manager@example.com",
                next_run_at=datetime(2026, 8, 1, 9, 0, tzinfo=timezone.utc),
            )
        )

    removed = remove_membership_summary("owner_1", "org_1", "manager_1", store)

    org_1_schedule = store.get_report_schedule("schedule_1")
    org_2_schedule = store.get_report_schedule("schedule_2")
    audit_event = store.list_audit_events("org_1")[-1]
    audit_metadata = json.loads(audit_event["metadata_json"])
    assert removed.user_id == "manager_1"
    assert org_1_schedule["created_by_user_id"] == "owner_1"
    assert org_1_schedule["recipient_email"] == "owner@example.com"
    assert org_1_schedule["updated_at"] != org_1_schedule["created_at"]
    assert org_2_schedule["created_by_user_id"] == "manager_1"
    assert org_2_schedule["recipient_email"] == "manager@example.com"
    assert store.list_report_schedules_by_creator("org_1", "manager_1") == []
    assert [schedule["id"] for schedule in store.list_report_schedules_by_creator("org_2", "manager_1")] == [
        "schedule_2"
    ]
    assert audit_event["action"] == "membership_removed"
    assert audit_metadata == {
        "previous_role": "manager",
        "report_schedule_recipient_reassignment_count": 1,
        "report_schedule_transfer_count": 1,
        "target_user_id": "manager_1",
    }
    assert "reports@example.com" not in audit_event["metadata_json"]


def test_invite_summaries_and_acceptance_flow_are_tenant_safe():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_2",
            email="other@example.com",
            display_name="Other Owner",
            organization_id="org_2",
            organization_name="Other Energy",
            organization_slug="other-energy",
        ),
        store,
    )

    created = create_invite(
        "owner_1",
        "org_1",
        OrganizationInviteCreate(
            email="invitee@example.com",
            role="viewer",
        ),
        store,
    )

    invites = list_organization_invite_summaries("owner_1", "org_1", store)
    accepted = accept_invite(
        created.id,
        OrganizationInviteAccept(
            user_id="user_2",
            email="invitee@example.com",
            display_name="Invitee",
        ),
        store,
    )

    assert created.id.startswith("invite_")
    assert created.id != "invite_1"
    assert [invite.id for invite in invites] == [created.id]
    assert accepted.status == "accepted"
    assert accepted.accepted_by_user_id == "user_2"

    try:
        list_organization_invite_summaries("owner_2", "org_1", store)
        assert False, "Expected cross-tenant invite listing to be denied"
    except AccessDenied:
        pass


def test_invite_create_route_generates_server_side_invite_id():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    client = TestClient(
        create_app(
            store=store,
            artifact_store=InMemoryTenantArtifactStore(),
        )
    )

    response = client.post(
        "/organizations/org_1/invites",
        headers={"X-User-Id": "owner_1"},
        json={
            "invite_id": "invite_predictable",
            "email": "invitee@example.com",
            "role": "viewer",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    invite_row = store.get_organization_invite(payload["id"])

    assert payload["id"].startswith("invite_")
    assert len(payload["id"]) > len("invite_") + 20
    assert payload["id"] != "invite_1"
    assert payload["id"] != "invite_predictable"
    assert payload["email"] == "invitee@example.com"
    assert invite_row["organization_id"] == "org_1"
    assert invite_row["status"] == "pending"
    assert store.get_organization_invite("invite_predictable") is None


def test_revoke_invite_is_tenant_guarded():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_2",
            email="other@example.com",
            display_name="Other Owner",
            organization_id="org_2",
            organization_name="Other Energy",
            organization_slug="other-energy",
        ),
        store,
    )

    created = create_invite(
        "owner_1",
        "org_1",
        OrganizationInviteCreate(
            email="invitee@example.com",
            role="viewer",
        ),
        store,
    )

    revoked = revoke_invite("owner_1", "org_1", created.id, store)

    assert revoked.status == "revoked"

    try:
        revoke_invite("owner_2", "org_1", created.id, store)
        assert False, "Expected cross-tenant revoke to be denied"
    except AccessDenied:
        pass

    try:
        list_membership_summaries("owner_2", "org_1", store)
        assert False, "Expected cross-tenant membership access to be denied"
    except AccessDenied:
        pass


def test_execute_local_analysis_run_returns_run_metadata_and_iso_summary():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="user_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    store.create_site(Site(id="site_1", organization_id="org_1", name="Main Site", timezone="Europe/London"))
    payload = LocalAnalysisRunCreate(
        site_id="site_1",
        upload_id="upload_1",
        run_id="run_1",
        report_id="report_1",
        filename="energy.csv",
        rows=[
            {"Date": "2025-01-01", "Main Electricity": 100, "Main Gas": 80},
            {"Date": "2025-02-01", "Main Electricity": 120, "Main Gas": 70},
        ],
        client_config=load_client_config("config/clients/example_client.yaml"),
    )

    response = execute_local_analysis_run("user_1", "org_1", payload, store, artifact_store)
    report_row = store.get_report("report_1")

    assert response.run_id == "run_1"
    assert response.run_status == "succeeded"
    assert "report_storage_key" not in response.model_dump()
    assert report_row is not None
    assert report_row["storage_key"] == "tenants/org_1/sites/site_1/runs/run_1/reports/iso-summary.json"
    assert response.iso_summary["total_records"] == 4
    assert "\"total_records\": 4" in artifact_store.get_text("org_1", report_row["storage_key"])


def test_execute_stored_upload_analysis_run_uses_existing_upload_artifact():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="user_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    store.create_site(Site(id="site_1", organization_id="org_1", name="Main Site", timezone="Europe/London"))
    upload = create_upload_content_summary(
        "user_1",
        "org_1",
        UploadContentCreate(
            upload_id="upload_1",
            site_id="site_1",
            category="energy",
            filename="energy.csv",
            content_base64=base64.b64encode(
                b"Date,Main Electricity,Main Gas\n2025-01-01,100,80\n2025-02-01,120,70\n"
            ).decode("ascii"),
        ),
        store,
        artifact_store,
    )

    response = execute_stored_upload_analysis_run(
        "user_1",
        "org_1",
        StoredUploadAnalysisRunCreate(
            upload_id=upload.id,
            run_id="run_1",
            report_id="report_1",
            client_config=load_client_config("config/clients/example_client.yaml"),
        ),
        store,
        artifact_store,
    )
    report_row = store.get_report("report_1")

    assert response.upload_id == "upload_1"
    assert response.run_status == "succeeded"
    assert "report_storage_key" not in response.model_dump()
    assert report_row is not None
    assert report_row["storage_key"] == "tenants/org_1/sites/site_1/runs/run_1/reports/iso-summary.json"
    assert response.iso_summary["total_records"] == 4
    assert [run["id"] for run in store.list_runs("org_1")] == ["run_1"]


def test_queue_stored_upload_analysis_run_returns_queued_job():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="user_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    store.create_site(Site(id="site_1", organization_id="org_1", name="Main Site", timezone="Europe/London"))
    upload = create_upload_content_summary(
        "user_1",
        "org_1",
        UploadContentCreate(
            upload_id="upload_1",
            site_id="site_1",
            category="energy",
            filename="energy.csv",
            content_base64=base64.b64encode(
                b"Date,Main Electricity,Main Gas\n2025-01-01,100,80\n2025-02-01,120,70\n"
            ).decode("ascii"),
        ),
        store,
        artifact_store,
    )
    hdd_upload = create_upload_content_summary(
        "user_1",
        "org_1",
        UploadContentCreate(
            upload_id="upload_hdd",
            site_id="site_1",
            category="hdd",
            filename="hdd.csv",
            content_base64=base64.b64encode(
                b"Date,HDD\n2025-01-01,12\n2025-02-01,18\n"
            ).decode("ascii"),
        ),
        store,
        artifact_store,
    )

    response = queue_stored_upload_analysis_run(
        "user_1",
        "org_1",
        StoredUploadAnalysisRunCreate(
            upload_id=upload.id,
            run_id="run_1",
            report_id="report_1",
            source_upload_ids=[hdd_upload.id],
            client_config=load_client_config("config/clients/example_client.yaml"),
        ),
        store,
    )

    assert response.job_id == "job_run_1"
    assert response.run_id == "run_1"
    assert response.run_upload_ids == ["upload_1", "upload_hdd"]
    assert [upload["id"] for upload in store.list_run_uploads("org_1", "run_1")] == [
        "upload_1",
        "upload_hdd",
    ]
    assert response.run_status == "queued"
    assert store.list_runs("org_1")[0]["status"] == "queued"
    assert store.list_analysis_jobs("org_1")[0]["status"] == "queued"
    assert store.list_reports("org_1") == []


def test_analysis_job_summaries_are_tenant_guarded():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_2",
            email="other@example.com",
            display_name="Other Owner",
            organization_id="org_2",
            organization_name="Other Energy",
            organization_slug="other-energy",
        ),
        store,
    )
    store.create_site(Site(id="site_1", organization_id="org_1", name="Main Site", timezone="Europe/London"))
    upload = create_upload_content_summary(
        "owner_1",
        "org_1",
        UploadContentCreate(
            upload_id="upload_1",
            site_id="site_1",
            category="energy",
            filename="energy.csv",
            content_base64=base64.b64encode(
                b"Date,Main Electricity,Main Gas\n2025-01-01,100,80\n2025-02-01,120,70\n"
            ).decode("ascii"),
        ),
        store,
        artifact_store,
    )
    queue_stored_upload_analysis_run(
        "owner_1",
        "org_1",
        StoredUploadAnalysisRunCreate(
            upload_id=upload.id,
            run_id="run_1",
            report_id="report_1",
            client_config=load_client_config("config/clients/example_client.yaml"),
        ),
        store,
    )

    jobs = list_analysis_job_summaries("owner_1", "org_1", store)

    assert [job.id for job in jobs] == ["job_run_1"]
    assert jobs[0].status == "queued"
    assert jobs[0].upload_id == "upload_1"

    try:
        list_analysis_job_summaries("owner_2", "org_1", store)
        assert False, "Expected cross-tenant analysis job listing to be denied"
    except AccessDenied:
        pass


def test_notification_summaries_are_recipient_and_tenant_scoped():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_2",
            email="other@example.com",
            display_name="Other Owner",
            organization_id="org_2",
            organization_name="Other Energy",
            organization_slug="other-energy",
        ),
        store,
    )
    store.create_user(User(id="viewer_1", email="viewer@example.com", display_name="Viewer"))
    store.add_membership(
        Membership(
            user_id="viewer_1",
            organization_id="org_1",
            role=Role.VIEWER,
            invited_by_user_id="owner_1",
        )
    )
    store.conn.executemany(
        """
        INSERT INTO notifications (
            id,
            organization_id,
            recipient_user_id,
            channel,
            event_type,
            subject,
            body,
            resource_type,
            resource_id,
            attempt_count,
            last_attempt_at,
            next_attempt_at
        )
        VALUES (?, ?, ?, 'email', 'run_succeeded', ?, ?, 'run', ?, ?, ?, ?)
        """,
        [
            (
                "notification_owner",
                "org_1",
                "owner_1",
                "Run complete",
                "Owner run ready",
                "run_1",
                1,
                "2026-07-20T12:00:00+00:00",
                "2026-07-20T12:05:00+00:00",
            ),
            (
                "notification_viewer",
                "org_1",
                "viewer_1",
                "Run complete",
                "Viewer run ready",
                "run_2",
                0,
                None,
                None,
            ),
            (
                "notification_other",
                "org_2",
                "owner_2",
                "Run complete",
                "Other run ready",
                "run_3",
                2,
                "2026-07-20T12:01:00+00:00",
                "2026-07-20T12:06:00+00:00",
            ),
        ],
    )

    owner_notifications = list_notification_summaries("owner_1", "org_1", store)
    viewer_notifications = list_notification_summaries("viewer_1", "org_1", store)

    assert [notification.id for notification in owner_notifications] == ["notification_owner"]
    assert [notification.id for notification in viewer_notifications] == ["notification_viewer"]
    assert owner_notifications[0].attempt_count == 1
    assert owner_notifications[0].last_attempt_at == "2026-07-20T12:00:00+00:00"
    assert owner_notifications[0].next_attempt_at == "2026-07-20T12:05:00+00:00"
    assert viewer_notifications[0].attempt_count == 0
    assert viewer_notifications[0].last_attempt_at is None
    assert viewer_notifications[0].next_attempt_at is None

    try:
        list_notification_summaries("owner_2", "org_1", store)
        assert False, "Expected cross-tenant notification listing to be denied"
    except AccessDenied:
        pass


def test_report_summaries_are_tenant_guarded():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="user_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    store.create_site(Site(id="site_1", organization_id="org_1", name="Main Site", timezone="Europe/London"))
    payload = LocalAnalysisRunCreate(
        site_id="site_1",
        upload_id="upload_1",
        run_id="run_1",
        report_id="report_1",
        filename="energy.csv",
        rows=[
            {"Date": "2025-01-01", "Main Electricity": 100, "Main Gas": 80},
            {"Date": "2025-02-01", "Main Electricity": 120, "Main Gas": 70},
        ],
        client_config=load_client_config("config/clients/example_client.yaml"),
    )
    execute_local_analysis_run("user_1", "org_1", payload, store, artifact_store)

    reports = list_report_summaries("user_1", "org_1", store, run_id="run_1")

    assert [report.id for report in reports] == ["report_1"]
    assert reports[0].is_published
    assert reports[0].expires_at is not None
    assert reports[0].retention_applied_at is None


def test_report_artifact_is_read_from_tenant_scoped_storage():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="user_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    store.create_site(Site(id="site_1", organization_id="org_1", name="Main Site", timezone="Europe/London"))
    payload = LocalAnalysisRunCreate(
        site_id="site_1",
        upload_id="upload_1",
        run_id="run_1",
        report_id="report_1",
        filename="energy.csv",
        rows=[
            {"Date": "2025-01-01", "Main Electricity": 100, "Main Gas": 80},
            {"Date": "2025-02-01", "Main Electricity": 120, "Main Gas": 70},
        ],
        client_config=load_client_config("config/clients/example_client.yaml"),
    )
    execute_local_analysis_run("user_1", "org_1", payload, store, artifact_store)

    expected_storage_key = "tenants/org_1/sites/site_1/runs/run_1/reports/iso-summary.json"
    report_row = store.get_report("report_1")
    assert report_row is not None
    legacy_artifact_content = json.loads(artifact_store.get_text("org_1", expected_storage_key))
    legacy_artifact_content["storage_key"] = expected_storage_key
    legacy_artifact_content["source_package"]["storage_key"] = expected_storage_key
    legacy_artifact_content["source_package"]["uploads"][0]["storage_key"] = expected_storage_key
    legacy_artifact_content["source_package"]["analysis_context"]["storage_key"] = expected_storage_key
    legacy_artifact_content["source_package"]["analysis_context"]["analysis_input_path"] = (
        expected_storage_key
    )
    legacy_artifact_content["source_package"]["analysis_context"]["sources"][0]["storage_key"] = (
        expected_storage_key
    )
    legacy_artifact_content["source_package"]["analysis_context"]["sources"][0]["analysis_input_path"] = (
        expected_storage_key
    )
    artifact_store.put_text(
        "org_1",
        expected_storage_key,
        json.dumps(legacy_artifact_content),
    )

    artifact = get_report_artifact("user_1", "org_1", "report_1", store, artifact_store)
    audit_events = store.list_audit_events("org_1")

    assert artifact.report_id == "report_1"
    assert "storage_key" not in artifact.model_dump()
    assert report_row["storage_key"] == expected_storage_key
    assert artifact.content["total_records"] == 4
    assert "storage_key" not in artifact.content
    assert artifact.content["source_package"]["primary_upload_id"] == "upload_1"
    assert artifact.content["source_package"]["upload_count"] == 1
    assert artifact.content["source_package"]["categories"] == ["energy"]
    assert "storage_key" not in artifact.content["source_package"]
    assert "storage_key" not in artifact.content["source_package"]["uploads"][0]
    assert "storage_key" not in artifact.content["source_package"]["analysis_context"]
    assert "analysis_input_path" not in artifact.content["source_package"]["analysis_context"]
    assert (
        "storage_key"
        not in artifact.content["source_package"]["analysis_context"]["sources"][0]
    )
    assert (
        "analysis_input_path"
        not in artifact.content["source_package"]["analysis_context"]["sources"][0]
    )
    assert expected_storage_key not in json.dumps(artifact.content)
    assert [event["action"] for event in audit_events][-1] == "report_accessed"
    assert audit_events[-1]["resource_id"] == "report_1"
    assert json.loads(audit_events[-1]["metadata_json"]) == {
        "artifact_location_configured": "true",
        "report_type": "iso_summary",
    }
    assert "storage_key" not in audit_events[-1]["metadata_json"]
    assert expected_storage_key not in audit_events[-1]["metadata_json"]


def test_report_export_download_is_tenant_guarded_visible_and_audited():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_2",
            email="other@example.com",
            display_name="Other Owner",
            organization_id="org_2",
            organization_name="Other Energy",
            organization_slug="other-energy",
        ),
        store,
    )
    store.create_user(User(id="viewer_1", email="viewer@example.com", display_name="Viewer"))
    store.add_membership(
        Membership(
            user_id="viewer_1",
            organization_id="org_1",
            role=Role.VIEWER,
            invited_by_user_id="owner_1",
        )
    )
    store.create_site(Site(id="site_1", organization_id="org_1", name="Main Site", timezone="Europe/London"))
    payload = LocalAnalysisRunCreate(
        site_id="site_1",
        upload_id="upload_1",
        run_id="run_1",
        report_id="report_1",
        filename="energy.csv",
        rows=[
            {"Date": "2025-01-01", "Main Electricity": 100, "Main Gas": 80},
            {"Date": "2025-02-01", "Main Electricity": 120, "Main Gas": 70},
        ],
        client_config=load_client_config("config/clients/example_client.yaml"),
    )
    execute_local_analysis_run("owner_1", "org_1", payload, store, artifact_store)
    report_storage_key = "tenants/org_1/sites/site_1/runs/run_1/reports/iso-summary.json"
    artifact_store.put_text(
        "org_1",
        report_storage_key,
        json.dumps(
            {
                "client": "Example Energy",
                "site": "Main Site",
                "storage_key": report_storage_key,
                "total_records": 4,
                "seu_total_consumption": 370,
                "source_package": {
                    "primary_upload_id": "upload_1",
                    "site_id": "site_1",
                    "upload_count": 1,
                    "categories": ["energy"],
                    "storage_key": report_storage_key,
                    "uploads": [
                        {
                            "id": "upload_1",
                            "category": "energy",
                            "checksum": "abc123",
                            "status": "stored",
                            "is_primary": True,
                            "storage_key": report_storage_key,
                        }
                    ],
                    "analysis_context": {
                        "execution_input": "legacy_energy_export",
                        "resolved_meter_columns": ["Main Electricity"],
                        "resolved_record_count": 2,
                        "analysis_input_path": report_storage_key,
                        "storage_key": report_storage_key,
                        "sources": [
                            {
                                "upload_id": "upload_1",
                                "category": "energy",
                                "is_primary": True,
                                "parse_status": "parsed",
                                "shape": "saas_meter_csv",
                                "row_count": 2,
                                "storage_key": report_storage_key,
                                "analysis_input_path": report_storage_key,
                            }
                        ],
                    },
                },
                "legacy_baseline": {
                    "weather_normalization": {
                        "status": "modeled",
                        "baseline_year": 2024,
                        "comparison_year": 2025,
                        "model_count": 1,
                        "skipped_meter_count": 1,
                        "estimated_savings": 42.5,
                        "models": [
                            {
                                "meter": "Boiler gas",
                                "utility": "gas",
                                "seu_category": "Boiler",
                                "normalization": "hdd",
                                "actual_consumption": 1200.0,
                                "predicted_consumption": 1242.5,
                                "estimated_savings": 42.5,
                                "r2": 0.91,
                            }
                        ],
                        "skipped_meters": [
                            {
                                "meter": "Backup generator",
                                "utility": "diesel",
                                "seu_category": "Unknown",
                                "reason": "unsupported_utility",
                            }
                        ],
                    }
                },
            },
            indent=2,
            sort_keys=True,
        ),
    )
    create_report_comment_summary(
        "owner_1",
        "org_1",
        "report_1",
        ReportCommentCreate(body=" Package review note "),
        store,
    )
    create_report_review_summary(
        "owner_1",
        "org_1",
        "report_1",
        ReportReviewCreate(
            status=ReportReviewStatus.APPROVED,
            notes=" Approved for evidence package ",
        ),
        store,
    )
    approval_assignment = create_report_approval_assignment_summary(
        "owner_1",
        "org_1",
        "report_1",
        ReportApprovalAssignmentCreate(
            assigned_to_user_id="owner_1",
            notes=" Confirm package evidence before audit ",
        ),
        store,
    )
    next_run_at = (datetime.now(timezone.utc) + timedelta(days=30)).replace(microsecond=0)
    create_report_schedule_summary(
        "owner_1",
        "org_1",
        "report_1",
        ReportScheduleCreate(
            frequency=ReportScheduleFrequency.MONTHLY,
            recipient_email="recipient@example.com",
            next_run_at=next_run_at.isoformat(),
        ),
        store,
    )
    update_report_branding_summary(
        "owner_1",
        "org_1",
        OrganizationReportBrandingUpdate(
            evidence_pack_title="Example Energy Audit Pack",
            primary_color="#123abc",
            logo_url="https://cdn.example.com/logo.png",
            support_email="support@example.com",
        ),
        store,
    )

    export = export_report_artifact("owner_1", "org_1", "report_1", store, artifact_store)
    audit_events = store.list_audit_events("org_1")
    audit_metadata = json.loads(audit_events[-1]["metadata_json"])
    export_content = export.body.decode("utf-8")
    export_json = json.loads(export_content)

    assert export.status_code == 200
    assert export.media_type == "application/json"
    assert export.headers["content-disposition"] == 'attachment; filename="report_1-iso_summary.json"'
    assert export_json["total_records"] == 4
    assert export_json["source_package"]["primary_upload_id"] == "upload_1"
    assert "storage_key" not in export_content
    assert "analysis_input_path" not in export_content
    assert report_storage_key not in export_content
    assert audit_events[-1]["action"] == "report_exported"
    assert audit_events[-1]["resource_type"] == "report"
    assert audit_events[-1]["resource_id"] == "report_1"
    assert audit_metadata == {
        "content_length": len(export_content),
        "format": "json",
        "report_id": "report_1",
        "report_type": "iso_summary",
    }
    assert "storage_key" not in audit_events[-1]["metadata_json"]
    assert "total_records" not in audit_events[-1]["metadata_json"]

    client = TestClient(create_app(store=store, artifact_store=artifact_store))
    route_export = client.get(
        "/organizations/org_1/reports/report_1/export",
        headers={"X-User-Id": "owner_1"},
    )

    assert route_export.status_code == 200
    assert route_export.headers["content-disposition"] == (
        'attachment; filename="report_1-iso_summary.json"'
    )
    assert route_export.headers["content-type"].startswith("application/json")
    assert route_export.json()["total_records"] == 4

    evidence_export = export_report_evidence_package(
        "owner_1",
        "org_1",
        "report_1",
        store,
        artifact_store,
    )
    evidence = json.loads(evidence_export.body.decode("utf-8"))
    evidence_audit = store.list_audit_events("org_1")[-1]
    evidence_metadata = json.loads(evidence_audit["metadata_json"])

    assert evidence_export.status_code == 200
    assert evidence_export.media_type == "application/json"
    assert evidence_export.headers["content-disposition"] == (
        'attachment; filename="report_1-iso_summary-evidence-package.json"'
    )
    assert evidence["package_type"] == "iso50001_report_evidence"
    assert evidence["organization_id"] == "org_1"
    assert evidence["organization"] == {
        "id": "org_1",
        "name": "Example Energy",
        "slug": "example-energy",
    }
    assert evidence["branding"] == {
        "evidence_pack_title": "Example Energy Audit Pack",
        "logo_url": "https://cdn.example.com/logo.png",
        "primary_color": "#123abc",
        "support_email": "support@example.com",
    }
    assert evidence["report"]["id"] == "report_1"
    assert "storage_key" not in evidence["report"]
    assert evidence["run"]["id"] == "run_1"
    assert evidence["site"]["name"] == "Main Site"
    assert evidence["uploads"][0]["id"] == "upload_1"
    assert evidence["uploads"][0]["checksum"]
    assert "storage_key" not in evidence["uploads"][0]
    assert evidence["artifact"]["total_records"] == 4
    assert evidence["artifact"]["source_package"]["primary_upload_id"] == "upload_1"
    assert "storage_key" not in json.dumps(evidence["artifact"])
    assert "analysis_input_path" not in json.dumps(evidence["artifact"])
    weather_evidence = evidence["artifact"]["legacy_baseline"]["weather_normalization"]
    assert weather_evidence["model_count"] == 1
    assert weather_evidence["skipped_meter_count"] == 1
    assert weather_evidence["models"][0]["meter"] == "Boiler gas"
    assert weather_evidence["skipped_meters"][0] == {
        "meter": "Backup generator",
        "reason": "unsupported_utility",
        "seu_category": "Unknown",
        "utility": "diesel",
    }
    evidence_pdf_sections = dict(
        _report_evidence_package_pdf_sections(
            evidence,
            {
                "comment_count": 1,
                "delivery_count": 0,
                "review_count": 1,
                "schedule_count": 1,
                "upload_count": 1,
            },
        )
    )
    assert "Weather Normalized Evidence" in evidence_pdf_sections
    weather_pdf_lines = evidence_pdf_sections["Weather Normalized Evidence"]
    assert any("Model 1: Boiler gas" in line for line in weather_pdf_lines)
    assert any("Skipped 1: Backup generator" in line for line in weather_pdf_lines)
    assert any("reason unsupported_utility" in line for line in weather_pdf_lines)
    assert evidence["approval_assignments"][0]["notes"] == "Confirm package evidence before audit"
    assert evidence["approval_assignments"][0]["status"] == "pending"
    assert evidence["comments"][0]["body"] == "Package review note"
    assert evidence["reviews"][0]["notes"] == "Approved for evidence package"
    assert evidence["schedules"][0]["recipient_email"] == "recipient@example.com"
    assert evidence_audit["action"] == "report_evidence_exported"
    assert evidence_audit["resource_type"] == "report"
    assert evidence_audit["resource_id"] == "report_1"
    assert evidence_metadata == {
        "comment_count": 1,
        "content_length": len(evidence_export.body.decode("utf-8")),
        "delivery_count": 0,
        "format": "json",
        "package_type": "iso50001_report_evidence",
        "report_id": "report_1",
        "report_type": "iso_summary",
        "review_count": 1,
        "schedule_count": 1,
        "template_enabled_section_count": 6,
        "upload_count": 1,
    }
    assert "Package review note" not in evidence_audit["metadata_json"]
    assert "Confirm package evidence before audit" not in evidence_audit["metadata_json"]
    assert "Approved for evidence package" not in evidence_audit["metadata_json"]
    assert "recipient@example.com" not in evidence_audit["metadata_json"]
    assert "https://cdn.example.com/logo.png" not in evidence_audit["metadata_json"]
    assert "support@example.com" not in evidence_audit["metadata_json"]
    assert "storage_key" not in evidence_audit["metadata_json"]
    assert "total_records" not in evidence_audit["metadata_json"]
    assert report_storage_key not in evidence_export.body.decode("utf-8")
    assert "tenants/org_1/sites/site_1/uploads/upload_1/energy.csv" not in evidence_export.body.decode(
        "utf-8"
    )

    route_evidence = client.get(
        "/organizations/org_1/reports/report_1/evidence-package",
        headers={"X-User-Id": "owner_1"},
    )

    assert route_evidence.status_code == 200
    assert route_evidence.headers["content-disposition"] == (
        'attachment; filename="report_1-iso_summary-evidence-package.json"'
    )
    assert route_evidence.headers["content-type"].startswith("application/json")
    assert route_evidence.json()["package_type"] == "iso50001_report_evidence"

    pdf_export = export_report_evidence_package_pdf(
        "owner_1",
        "org_1",
        "report_1",
        store,
        artifact_store,
    )
    pdf_audit = store.list_audit_events("org_1")[-1]
    pdf_metadata = json.loads(pdf_audit["metadata_json"])

    assert pdf_export.status_code == 200
    assert pdf_export.media_type == "application/pdf"
    assert pdf_export.headers["content-disposition"] == (
        'attachment; filename="report_1-iso_summary-evidence-package.pdf"'
    )
    assert pdf_export.body.startswith(b"%PDF")
    assert pdf_audit["action"] == "report_evidence_exported"
    assert pdf_audit["resource_type"] == "report"
    assert pdf_audit["resource_id"] == "report_1"
    assert pdf_metadata == {
        "comment_count": 1,
        "content_length": len(pdf_export.body),
        "delivery_count": 0,
        "format": "pdf",
        "package_type": "iso50001_report_evidence",
        "report_id": "report_1",
        "report_type": "iso_summary",
        "review_count": 1,
        "schedule_count": 1,
        "template_enabled_section_count": 6,
        "upload_count": 1,
    }
    assert "Package review note" not in pdf_audit["metadata_json"]
    assert "Confirm package evidence before audit" not in pdf_audit["metadata_json"]
    assert "Approved for evidence package" not in pdf_audit["metadata_json"]
    assert "recipient@example.com" not in pdf_audit["metadata_json"]
    assert "https://cdn.example.com/logo.png" not in pdf_audit["metadata_json"]
    assert "support@example.com" not in pdf_audit["metadata_json"]
    assert "storage_key" not in pdf_audit["metadata_json"]
    assert "total_records" not in pdf_audit["metadata_json"]

    route_evidence_pdf = client.get(
        "/organizations/org_1/reports/report_1/evidence-package/pdf",
        headers={"X-User-Id": "owner_1"},
    )

    assert route_evidence_pdf.status_code == 200
    assert route_evidence_pdf.headers["content-disposition"] == (
        'attachment; filename="report_1-iso_summary-evidence-package.pdf"'
    )
    assert route_evidence_pdf.headers["content-type"].startswith("application/pdf")
    assert route_evidence_pdf.content.startswith(b"%PDF")

    update_evidence_package_template_summary(
        "owner_1",
        "org_1",
        OrganizationEvidencePackageTemplateUpdate(
            template_name="Approval-only board pack",
            include_report_artifact=False,
            include_uploads=False,
            include_approval_assignments=True,
            include_reviews=False,
            include_comments=False,
            include_schedules=False,
            section_labels={
                "approval_assignments": "Board sign-off",
                "comments": "Review discussion",
                "report_artifact": "Board report appendix",
                "reviews": "Management review record",
                "schedules": "Delivery register",
                "uploads": "Source file register",
            },
            section_order=[
                "approval_assignments",
                "report_artifact",
                "uploads",
                "reviews",
                "comments",
                "schedules",
            ],
        ),
        store,
    )
    custom_evidence_export = export_report_evidence_package(
        "owner_1",
        "org_1",
        "report_1",
        store,
        artifact_store,
    )
    custom_evidence = json.loads(custom_evidence_export.body.decode("utf-8"))
    custom_evidence_metadata = json.loads(store.list_audit_events("org_1")[-1]["metadata_json"])
    custom_pdf_sections = dict(
        _report_evidence_package_pdf_sections(
            custom_evidence,
            {
                "comment_count": 0,
                "delivery_count": 0,
                "review_count": 0,
                "schedule_count": 0,
                "template_enabled_section_count": 1,
                "upload_count": 0,
            },
        )
    )
    custom_pdf_export = export_report_evidence_package_pdf(
        "owner_1",
        "org_1",
        "report_1",
        store,
        artifact_store,
    )
    custom_pdf_metadata = json.loads(store.list_audit_events("org_1")[-1]["metadata_json"])

    assert custom_evidence["template"] == {
        "include_approval_assignments": True,
        "include_comments": False,
        "include_report_artifact": False,
        "include_reviews": False,
        "include_schedules": False,
        "include_uploads": False,
        "section_labels": {
            "approval_assignments": "Board sign-off",
            "comments": "Review discussion",
            "report_artifact": "Board report appendix",
            "reviews": "Management review record",
            "schedules": "Delivery register",
            "uploads": "Source file register",
        },
        "section_order": [
            "approval_assignments",
            "report_artifact",
            "uploads",
            "reviews",
            "comments",
            "schedules",
        ],
        "template_name": "Approval-only board pack",
    }
    assert custom_evidence["artifact"] == {}
    assert custom_evidence["uploads"] == []
    assert custom_evidence["reviews"] == []
    assert custom_evidence["comments"] == []
    assert custom_evidence["schedules"] == []
    assert custom_evidence["approval_assignments"][0]["notes"] == (
        "Confirm package evidence before audit"
    )
    assert custom_evidence_metadata["comment_count"] == 0
    assert custom_evidence_metadata["review_count"] == 0
    assert custom_evidence_metadata["schedule_count"] == 0
    assert custom_evidence_metadata["template_enabled_section_count"] == 1
    assert custom_evidence_metadata["upload_count"] == 0
    assert "Board sign-off" in custom_pdf_sections
    assert "Approval And Review Evidence" not in custom_pdf_sections
    assert "Collaboration Evidence" not in custom_pdf_sections
    assert "Scheduled Delivery Evidence" not in custom_pdf_sections
    assert "Weather Normalized Evidence" not in custom_pdf_sections
    assert "Report Artifact Summary" not in custom_pdf_sections
    assert custom_pdf_export.body.startswith(b"%PDF")
    assert custom_pdf_metadata["comment_count"] == 0
    assert custom_pdf_metadata["template_enabled_section_count"] == 1
    assert "Approval-only board pack" not in store.list_audit_events("org_1")[-1]["metadata_json"]
    assert "Board sign-off" not in store.list_audit_events("org_1")[-1]["metadata_json"]
    assert "Package review note" not in store.list_audit_events("org_1")[-1]["metadata_json"]
    assert "recipient@example.com" not in store.list_audit_events("org_1")[-1]["metadata_json"]

    update_evidence_package_template_summary(
        "owner_1",
        "org_1",
        OrganizationEvidencePackageTemplateUpdate(
            template_name="Full ISO 50001 Evidence Package",
            include_report_artifact=True,
            include_uploads=True,
            include_approval_assignments=True,
            include_reviews=True,
            include_comments=True,
            include_schedules=True,
        ),
        store,
    )

    assert list_report_evidence_package_summaries("viewer_1", "org_1", "report_1", store) == []
    try:
        create_report_evidence_package_snapshot(
            "owner_1",
            "org_1",
            "report_1",
            store,
            artifact_store,
        )
        assert False, "Expected pending approval assignment to block locked evidence package"
    except ValueError as exc:
        assert "Pending report approval assignments" in str(exc)

    approved_assignment = update_report_approval_assignment_summary(
        "owner_1",
        "org_1",
        "report_1",
        approval_assignment.id,
        ReportApprovalAssignmentUpdate(
            status=ReportApprovalAssignmentStatus.APPROVED,
            notes=" Evidence lock approved ",
        ),
        store,
    )
    assert approved_assignment.status == ReportApprovalAssignmentStatus.APPROVED

    snapshot = create_report_evidence_package_snapshot(
        "owner_1",
        "org_1",
        "report_1",
        store,
        artifact_store,
    )
    package_row = store.get_report_evidence_package(snapshot.id)
    assert package_row is not None
    stored_package_content = artifact_store.get_text("org_1", package_row["storage_key"])
    stored_package = json.loads(stored_package_content)
    created_audit = store.list_audit_events("org_1")[-1]
    created_metadata = json.loads(created_audit["metadata_json"])

    assert snapshot.report_id == "report_1"
    assert snapshot.created_by_user_id == "owner_1"
    assert snapshot.filename == "report_1-iso_summary-evidence-package.json"
    assert snapshot.content_type == "application/json"
    assert snapshot.package_type == "iso50001_report_evidence"
    assert snapshot.format == "json"
    assert snapshot.content_length == len(stored_package_content)
    assert snapshot.comment_count == 1
    assert snapshot.review_count == 2
    assert snapshot.schedule_count == 1
    assert snapshot.delivery_count == 0
    assert snapshot.upload_count == 1
    assert package_row["storage_key"].startswith("tenants/org_1/reports/report_1/evidence-packages/")
    assert stored_package["comments"][0]["body"] == "Package review note"
    assert "storage_key" not in stored_package["report"]
    assert "storage_key" not in stored_package["uploads"][0]
    assert "storage_key" not in json.dumps(stored_package["artifact"])
    assert "analysis_input_path" not in json.dumps(stored_package["artifact"])
    assert report_storage_key not in stored_package_content
    assert stored_package["approval_assignments"][0]["notes"] == "Evidence lock approved"
    assert stored_package["approval_assignments"][0]["status"] == "approved"
    assert stored_package["reviews"][0]["notes"] == "Evidence lock approved"
    assert stored_package["reviews"][1]["notes"] == "Approved for evidence package"
    assert stored_package["artifact"]["total_records"] == 4
    assert created_audit["action"] == "report_evidence_package_created"
    assert created_audit["resource_type"] == "report_evidence_package"
    assert created_audit["resource_id"] == snapshot.id
    assert created_metadata == {
        "comment_count": 1,
        "content_length": len(stored_package_content),
        "delivery_count": 0,
        "format": "json",
        "package_id": snapshot.id,
        "package_type": "iso50001_report_evidence",
        "approval_assignment_count": 1,
        "approval_assignment_approved_count": 1,
        "approval_policy_auto_lock_evidence_on_final_approval": True,
        "approval_policy_minimum_assignments_for_lock": 0,
        "approval_review_id": stored_package["reviews"][0]["id"],
        "approval_status": "approved",
        "report_id": "report_1",
        "report_type": "iso_summary",
        "review_count": 2,
        "schedule_count": 1,
        "storage_key_configured": True,
        "template_enabled_section_count": 6,
        "upload_count": 1,
    }
    assert "Package review note" not in created_audit["metadata_json"]
    assert "Confirm package evidence before audit" not in created_audit["metadata_json"]
    assert "Evidence lock approved" not in created_audit["metadata_json"]
    assert "Approved for evidence package" not in created_audit["metadata_json"]
    assert "recipient@example.com" not in created_audit["metadata_json"]
    assert "https://cdn.example.com/logo.png" not in created_audit["metadata_json"]
    assert "support@example.com" not in created_audit["metadata_json"]
    assert '"storage_key":' not in created_audit["metadata_json"]
    assert "total_records" not in created_audit["metadata_json"]
    assert "tenants/org_1/sites/site_1/runs/run_1/reports/iso-summary.json" not in stored_package_content
    assert "tenants/org_1/sites/site_1/uploads/upload_1/energy.csv" not in stored_package_content

    listed_snapshots = list_report_evidence_package_summaries(
        "viewer_1",
        "org_1",
        "report_1",
        store,
    )
    downloaded_snapshot = download_report_evidence_package_snapshot(
        "viewer_1",
        "org_1",
        "report_1",
        snapshot.id,
        store,
        artifact_store,
    )
    download_audit = store.list_audit_events("org_1")[-1]
    download_metadata = json.loads(download_audit["metadata_json"])

    assert [item.id for item in listed_snapshots] == [snapshot.id]
    assert downloaded_snapshot.status_code == 200
    assert downloaded_snapshot.headers["content-disposition"] == (
        'attachment; filename="report_1-iso_summary-evidence-package.json"'
    )
    assert downloaded_snapshot.body.decode("utf-8") == stored_package_content
    assert download_audit["action"] == "report_evidence_package_downloaded"
    assert download_audit["resource_id"] == snapshot.id
    assert download_metadata == {
        "comment_count": 1,
        "content_length": len(stored_package_content),
        "delivery_count": 0,
        "format": "json",
        "package_id": snapshot.id,
        "package_type": "iso50001_report_evidence",
        "report_id": "report_1",
        "report_type": "iso_summary",
        "review_count": 2,
        "schedule_count": 1,
        "upload_count": 1,
    }
    assert "Package review note" not in download_audit["metadata_json"]
    assert "Approved for evidence package" not in download_audit["metadata_json"]
    assert "recipient@example.com" not in download_audit["metadata_json"]
    assert "https://cdn.example.com/logo.png" not in download_audit["metadata_json"]
    assert "support@example.com" not in download_audit["metadata_json"]
    assert "storage_key" not in download_audit["metadata_json"]
    assert "total_records" not in download_audit["metadata_json"]

    route_snapshot_list = client.get(
        "/organizations/org_1/reports/report_1/evidence-packages",
        headers={"X-User-Id": "viewer_1"},
    )
    route_snapshot_denied = client.post(
        "/organizations/org_1/reports/report_1/evidence-packages",
        headers={"X-User-Id": "viewer_1"},
    )
    route_snapshot_create = client.post(
        "/organizations/org_1/reports/report_1/evidence-packages",
        headers={"X-User-Id": "owner_1"},
    )
    route_snapshot_id = route_snapshot_create.json()["id"]
    route_snapshot_download = client.get(
        f"/organizations/org_1/reports/report_1/evidence-packages/{route_snapshot_id}/download",
        headers={"X-User-Id": "viewer_1"},
    )

    assert route_snapshot_list.status_code == 200
    assert route_snapshot_list.json()[0]["id"] == snapshot.id
    assert "storage_key" not in route_snapshot_list.json()[0]
    assert route_snapshot_denied.status_code == 403
    assert route_snapshot_create.status_code == 201
    assert route_snapshot_create.json()["report_id"] == "report_1"
    assert "storage_key" not in route_snapshot_create.json()
    assert route_snapshot_download.status_code == 200
    assert route_snapshot_download.headers["content-disposition"] == (
        'attachment; filename="report_1-iso_summary-evidence-package.json"'
    )
    assert route_snapshot_download.json()["package_type"] == "iso50001_report_evidence"

    try:
        export_report_artifact("owner_2", "org_1", "report_1", store, artifact_store)
        assert False, "Expected cross-tenant report export to be denied"
    except AccessDenied:
        pass
    try:
        export_report_evidence_package("owner_2", "org_1", "report_1", store, artifact_store)
        assert False, "Expected cross-tenant evidence package export to be denied"
    except AccessDenied:
        pass
    try:
        export_report_evidence_package_pdf("owner_2", "org_1", "report_1", store, artifact_store)
        assert False, "Expected cross-tenant evidence package PDF export to be denied"
    except AccessDenied:
        pass
    try:
        list_report_evidence_package_summaries("owner_2", "org_1", "report_1", store)
        assert False, "Expected cross-tenant evidence package list to be denied"
    except AccessDenied:
        pass
    try:
        create_report_evidence_package_snapshot("owner_2", "org_1", "report_1", store, artifact_store)
        assert False, "Expected cross-tenant evidence package snapshot create to be denied"
    except AccessDenied:
        pass
    try:
        download_report_evidence_package_snapshot(
            "owner_2",
            "org_1",
            "report_1",
            snapshot.id,
            store,
            artifact_store,
        )
        assert False, "Expected cross-tenant evidence package snapshot download to be denied"
    except AccessDenied:
        pass

    store.create_report(
        Report(
            id="report_expired_export",
            organization_id="org_1",
            run_id="run_1",
            report_type="iso_summary",
            storage_key="tenants/org_1/sites/site_1/runs/run_1/reports/expired-export.json",
            is_published=True,
            expires_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
    )
    artifact_store.put_text(
        "org_1",
        "tenants/org_1/sites/site_1/runs/run_1/reports/expired-export.json",
        "{\"expired\": true}",
    )
    try:
        export_report_artifact(
            "owner_1",
            "org_1",
            "report_expired_export",
            store,
            artifact_store,
        )
        assert False, "Expected expired report export to fail"
    except ValueError as exc:
        assert "expired" in str(exc)
    try:
        export_report_evidence_package(
            "owner_1",
            "org_1",
            "report_expired_export",
            store,
            artifact_store,
        )
        assert False, "Expected expired report evidence package export to fail"
    except ValueError as exc:
        assert "expired" in str(exc)
    try:
        export_report_evidence_package_pdf(
            "owner_1",
            "org_1",
            "report_expired_export",
            store,
            artifact_store,
        )
        assert False, "Expected expired report evidence package PDF export to fail"
    except ValueError as exc:
        assert "expired" in str(exc)
    try:
        create_report_evidence_package_snapshot(
            "owner_1",
            "org_1",
            "report_expired_export",
            store,
            artifact_store,
        )
        assert False, "Expected expired report evidence package snapshot create to fail"
    except ValueError as exc:
        assert "expired" in str(exc)
    try:
        download_report_evidence_package_snapshot(
            "owner_1",
            "org_1",
            "report_expired_export",
            snapshot.id,
            store,
            artifact_store,
        )
        assert False, "Expected expired report evidence package snapshot download to fail"
    except ValueError as exc:
        assert "expired" in str(exc)


def test_report_schedules_are_tenant_guarded_manager_only_and_audited():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_2",
            email="other@example.com",
            display_name="Other Owner",
            organization_id="org_2",
            organization_name="Other Energy",
            organization_slug="other-energy",
        ),
        store,
    )
    store.create_user(User(id="manager_1", email="manager@example.com", display_name="Manager"))
    store.create_user(User(id="viewer_1", email="viewer@example.com", display_name="Viewer"))
    store.add_membership(
        Membership(
            user_id="manager_1",
            organization_id="org_1",
            role=Role.MANAGER,
            invited_by_user_id="owner_1",
        )
    )
    store.add_membership(
        Membership(
            user_id="viewer_1",
            organization_id="org_1",
            role=Role.VIEWER,
            invited_by_user_id="owner_1",
        )
    )
    store.create_site(Site(id="site_1", organization_id="org_1", name="Main Site", timezone="Europe/London"))
    payload = LocalAnalysisRunCreate(
        site_id="site_1",
        upload_id="upload_1",
        run_id="run_1",
        report_id="report_1",
        filename="energy.csv",
        rows=[
            {"Date": "2025-01-01", "Main Electricity": 100, "Main Gas": 80},
            {"Date": "2025-02-01", "Main Electricity": 120, "Main Gas": 70},
        ],
        client_config=load_client_config("config/clients/example_client.yaml"),
    )
    execute_local_analysis_run("owner_1", "org_1", payload, store, artifact_store)

    assert list_report_schedule_summaries("viewer_1", "org_1", "report_1", store) == []

    next_run_at = (datetime.now(timezone.utc) + timedelta(days=30)).replace(microsecond=0)
    created = create_report_schedule_summary(
        "manager_1",
        "org_1",
        "report_1",
        ReportScheduleCreate(
            frequency=ReportScheduleFrequency.MONTHLY,
            recipient_email=" Energy.Manager@Example.com ",
            next_run_at=next_run_at.isoformat(),
        ),
        store,
    )
    audit_events = store.list_audit_events("org_1")
    created_metadata = json.loads(audit_events[-1]["metadata_json"])

    assert created.organization_id == "org_1"
    assert created.report_id == "report_1"
    assert created.created_by_user_id == "manager_1"
    assert created.frequency == ReportScheduleFrequency.MONTHLY
    assert created.recipient_email == "energy.manager@example.com"
    assert created.next_run_at == next_run_at.isoformat()
    assert created.is_active is True
    assert audit_events[-1]["action"] == "report_schedule_created"
    assert audit_events[-1]["resource_type"] == "report_schedule"
    assert created_metadata == {
        "frequency": "monthly",
        "next_run_at": next_run_at.isoformat(),
        "recipient_email_configured": True,
        "report_id": "report_1",
        "report_type": "iso_summary",
    }
    assert "energy.manager@example.com" not in audit_events[-1]["metadata_json"]

    client = TestClient(create_app(store=store, artifact_store=artifact_store))
    route_schedules = client.get(
        "/organizations/org_1/reports/report_1/schedules",
        headers={"X-User-Id": "owner_1"},
    )

    assert route_schedules.status_code == 200
    assert [schedule["id"] for schedule in route_schedules.json()] == [created.id]

    delivery_attempted_at = datetime(2026, 1, 20, 9, 0, tzinfo=timezone.utc)
    store.create_report_schedule_delivery(
        ReportScheduleDelivery(
            id="report_schedule_delivery_1",
            organization_id="org_1",
            schedule_id=created.id,
            report_id="report_1",
            recipient_email=created.recipient_email,
            status=ReportScheduleDeliveryStatus.SENT,
            attempted_at=delivery_attempted_at,
            sent_at=delivery_attempted_at,
        )
    )
    delivery_summaries = list_report_schedule_delivery_summaries(
        "viewer_1",
        "org_1",
        "report_1",
        created.id,
        store,
    )

    assert len(delivery_summaries) == 1
    assert delivery_summaries[0].id == "report_schedule_delivery_1"
    assert delivery_summaries[0].organization_id == "org_1"
    assert delivery_summaries[0].schedule_id == created.id
    assert delivery_summaries[0].report_id == "report_1"
    assert delivery_summaries[0].recipient_email == "energy.manager@example.com"
    assert delivery_summaries[0].status == ReportScheduleDeliveryStatus.SENT
    assert delivery_summaries[0].attempted_at == delivery_attempted_at.isoformat()
    assert delivery_summaries[0].sent_at == delivery_attempted_at.isoformat()

    route_deliveries = client.get(
        f"/organizations/org_1/reports/report_1/schedules/{created.id}/deliveries",
        headers={"X-User-Id": "viewer_1"},
    )

    assert route_deliveries.status_code == 200
    assert route_deliveries.json()[0]["id"] == "report_schedule_delivery_1"
    assert route_deliveries.json()[0]["status"] == "sent"

    updated = update_report_schedule_summary(
        "manager_1",
        "org_1",
        "report_1",
        created.id,
        ReportScheduleUpdate(is_active=False),
        store,
    )
    updated_metadata = json.loads(store.list_audit_events("org_1")[-1]["metadata_json"])

    assert updated.is_active is False
    assert updated_metadata["is_active"] is False
    assert updated_metadata["previous_is_active"] is True
    assert updated_metadata["recipient_email_configured"] is True
    assert "energy.manager@example.com" not in store.list_audit_events("org_1")[-1]["metadata_json"]

    try:
        create_report_schedule_summary(
            "viewer_1",
            "org_1",
            "report_1",
            ReportScheduleCreate(
                frequency=ReportScheduleFrequency.WEEKLY,
                recipient_email="viewer@example.com",
                next_run_at=next_run_at.isoformat(),
            ),
            store,
        )
        assert False, "Expected viewer report scheduling to be denied"
    except AccessDenied:
        pass

    try:
        list_report_schedule_summaries("owner_2", "org_1", "report_1", store)
        assert False, "Expected cross-tenant report schedule listing to be denied"
    except AccessDenied:
        pass

    try:
        list_report_schedule_delivery_summaries(
            "owner_2",
            "org_1",
            "report_1",
            created.id,
            store,
        )
        assert False, "Expected cross-tenant report schedule delivery listing to be denied"
    except AccessDenied:
        pass

    store.create_report(
        Report(
            id="report_other_schedule",
            organization_id="org_1",
            run_id="run_1",
            report_type="iso_summary",
            storage_key="tenants/org_1/sites/site_1/runs/run_1/reports/other-schedule.json",
            is_published=True,
        )
    )
    try:
        list_report_schedule_delivery_summaries(
            "owner_1",
            "org_1",
            "report_other_schedule",
            created.id,
            store,
        )
        assert False, "Expected mismatched report schedule delivery listing to be denied"
    except AccessDenied:
        pass

    try:
        update_report_schedule_summary(
            "owner_2",
            "org_2",
            "report_1",
            created.id,
            ReportScheduleUpdate(is_active=True),
            store,
        )
        assert False, "Expected cross-tenant report schedule update to be denied"
    except AccessDenied:
        pass

    store.create_report(
        Report(
            id="report_expired_schedule",
            organization_id="org_1",
            run_id="run_1",
            report_type="iso_summary",
            storage_key="tenants/org_1/sites/site_1/runs/run_1/reports/expired-schedule.json",
            is_published=True,
            expires_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
    )
    try:
        create_report_schedule_summary(
            "manager_1",
            "org_1",
            "report_expired_schedule",
            ReportScheduleCreate(
                frequency=ReportScheduleFrequency.QUARTERLY,
                recipient_email="energy.manager@example.com",
                next_run_at=next_run_at.isoformat(),
            ),
            store,
        )
        assert False, "Expected expired report schedule creation to fail"
    except ValueError as exc:
        assert "expired" in str(exc)

    store.create_report(
        Report(
            id="report_expired_delivery",
            organization_id="org_1",
            run_id="run_1",
            report_type="iso_summary",
            storage_key="tenants/org_1/sites/site_1/runs/run_1/reports/expired-delivery.json",
            is_published=True,
            expires_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
    )
    store.create_report_schedule(
        ReportSchedule(
            id="report_schedule_expired_delivery",
            organization_id="org_1",
            report_id="report_expired_delivery",
            created_by_user_id="manager_1",
            frequency=ReportScheduleFrequency.MONTHLY,
            recipient_email="energy.manager@example.com",
            next_run_at=next_run_at,
        )
    )
    try:
        list_report_schedule_delivery_summaries(
            "manager_1",
            "org_1",
            "report_expired_delivery",
            "report_schedule_expired_delivery",
            store,
        )
        assert False, "Expected expired report schedule delivery listing to fail"
    except ValueError as exc:
        assert "expired" in str(exc)


def test_report_approval_assignments_are_tenant_guarded_manager_only_and_audited():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_2",
            email="other@example.com",
            display_name="Other Owner",
            organization_id="org_2",
            organization_name="Other Energy",
            organization_slug="other-energy",
        ),
        store,
    )
    store.create_user(User(id="manager_1", email="manager@example.com", display_name="Manager"))
    store.create_user(User(id="manager_2", email="manager2@example.com", display_name="Manager Two"))
    store.create_user(User(id="viewer_1", email="viewer@example.com", display_name="Viewer"))
    store.add_membership(
        Membership(
            user_id="manager_1",
            organization_id="org_1",
            role=Role.MANAGER,
            invited_by_user_id="owner_1",
        )
    )
    store.add_membership(
        Membership(
            user_id="manager_2",
            organization_id="org_1",
            role=Role.MANAGER,
            invited_by_user_id="owner_1",
        )
    )
    store.add_membership(
        Membership(
            user_id="viewer_1",
            organization_id="org_1",
            role=Role.VIEWER,
            invited_by_user_id="owner_1",
        )
    )
    store.create_site(Site(id="site_1", organization_id="org_1", name="Main Site", timezone="Europe/London"))
    payload = LocalAnalysisRunCreate(
        site_id="site_1",
        upload_id="upload_1",
        run_id="run_1",
        report_id="report_1",
        filename="energy.csv",
        rows=[
            {"Date": "2025-01-01", "Main Electricity": 100, "Main Gas": 80},
            {"Date": "2025-02-01", "Main Electricity": 120, "Main Gas": 70},
        ],
        client_config=load_client_config("config/clients/example_client.yaml"),
    )
    execute_local_analysis_run("owner_1", "org_1", payload, store, artifact_store)
    report = store.get_report("report_1")
    assert report is not None
    artifact_store.put_text(
        "org_1",
        report["storage_key"],
        json.dumps(
            {
                "annual_kwh": 12345,
                "storage_key": report["storage_key"],
                "legacy_baseline": {
                    "weather_normalization": {
                        "estimated_savings": 42.5,
                        "model_count": 1,
                        "skipped_meter_count": 1,
                        "status": "modeled",
                    }
                },
                "source_package": {
                    "categories": ["energy", "hdd", "cdd", "seu_mapping"],
                    "storage_key": report["storage_key"],
                    "upload_count": 4,
                    "analysis_context": {
                        "execution_input": "saas_meter_csv",
                        "resolved_record_count": 2,
                        "analysis_input_path": report["storage_key"],
                        "storage_key": report["storage_key"],
                        "sources": [
                            {
                                "upload_id": "upload_1",
                                "category": "energy",
                                "storage_key": report["storage_key"],
                                "analysis_input_path": report["storage_key"],
                            }
                        ],
                    },
                },
                "tenant_sensitive_note": "org_1 management review",
                "total_records": 2,
            },
            sort_keys=True,
        ),
    )

    approvers = list_report_approver_summaries("manager_1", "org_1", store)
    assert [approver.user_id for approver in approvers] == ["manager_1", "manager_2", "owner_1"]
    assert {approver.role for approver in approvers} == {Role.OWNER, Role.MANAGER}
    try:
        list_report_approver_summaries("viewer_1", "org_1", store)
        assert False, "Expected viewer approver listing to be denied"
    except AccessDenied:
        pass

    assert list_report_approval_assignment_summaries("viewer_1", "org_1", "report_1", store) == []

    due_at = (datetime.now(timezone.utc) + timedelta(days=7)).replace(microsecond=0)
    created = create_report_approval_assignment_summary(
        "manager_1",
        "org_1",
        "report_1",
        ReportApprovalAssignmentCreate(
            assigned_to_user_id="manager_2",
            due_at=due_at.isoformat(),
            notes=" Confirm the final SEU evidence before lock ",
        ),
        store,
        artifact_store,
    )
    created_audit = store.list_audit_events("org_1")[-1]
    created_metadata = json.loads(created_audit["metadata_json"])

    assert created.organization_id == "org_1"
    assert created.report_id == "report_1"
    assert created.assigned_to_user_id == "manager_2"
    assert created.assigned_by_user_id == "manager_1"
    assert created.status == ReportApprovalAssignmentStatus.PENDING
    assert created.due_at == due_at.isoformat()
    assert created.notes == "Confirm the final SEU evidence before lock"
    assert created.completed_at is None
    assert created_audit["action"] == "report_approval_assigned"
    assert created_audit["resource_type"] == "report_approval_assignment"
    assert created_metadata == {
        "assigned_to_user_id": "manager_2",
        "due_at": due_at.isoformat(),
        "notes_length": len("Confirm the final SEU evidence before lock"),
        "report_id": "report_1",
        "report_type": "iso_summary",
        "status": "pending",
    }
    assert "Confirm the final SEU evidence before lock" not in created_audit["metadata_json"]
    assignee_notifications = list_notification_summaries("manager_2", "org_1", store)
    assert len(assignee_notifications) == 1
    assert assignee_notifications[0].event_type == "report_approval_assigned"
    assert assignee_notifications[0].recipient_user_id == "manager_2"
    assert assignee_notifications[0].resource_type == "report_approval_assignment"
    assert assignee_notifications[0].resource_id == created.id
    assert assignee_notifications[0].status == "pending"
    assert "report_1" in assignee_notifications[0].body
    assert due_at.isoformat() in assignee_notifications[0].body
    assert "Report digest:" in assignee_notifications[0].body
    assert "Total records: 2" in assignee_notifications[0].body
    assert "Source files: 4" in assignee_notifications[0].body
    assert "Source categories: energy, hdd, cdd, seu_mapping" in assignee_notifications[0].body
    assert "Weather normalization: modeled" in assignee_notifications[0].body
    assert "Weather models: 1" in assignee_notifications[0].body
    assert "Skipped weather models: 1" in assignee_notifications[0].body
    assert "Estimated normalized savings: 42.5" in assignee_notifications[0].body
    assert "Confirm the final SEU evidence before lock" not in assignee_notifications[0].body
    assert "annual_kwh" not in assignee_notifications[0].body
    assert "12345" not in assignee_notifications[0].body
    assert "tenant_sensitive_note" not in assignee_notifications[0].body
    assert "org_1 management review" not in assignee_notifications[0].body
    assert "storage_key" not in assignee_notifications[0].body
    assert "analysis_input_path" not in assignee_notifications[0].body
    assert report["storage_key"] not in assignee_notifications[0].body

    client = TestClient(create_app(store=store, artifact_store=artifact_store))
    route_assignments = client.get(
        "/organizations/org_1/reports/report_1/approval-assignments",
        headers={"X-User-Id": "viewer_1"},
    )
    route_approvers = client.get(
        "/organizations/org_1/report-approvers",
        headers={"X-User-Id": "manager_1"},
    )

    assert route_assignments.status_code == 200
    assert route_assignments.json()[0]["id"] == created.id
    assert route_approvers.status_code == 200
    assert [approver["user_id"] for approver in route_approvers.json()] == [
        "manager_1",
        "manager_2",
        "owner_1",
    ]

    updated = update_report_approval_assignment_summary(
        "manager_2",
        "org_1",
        "report_1",
        created.id,
        ReportApprovalAssignmentUpdate(
            status=ReportApprovalAssignmentStatus.APPROVED,
            notes=" Approved for locked package ",
        ),
        store,
        artifact_store,
    )
    audit_events = store.list_audit_events("org_1")
    review_audit = audit_events[-3]
    updated_audit = audit_events[-2]
    locked_package_audit = audit_events[-1]
    review_metadata = json.loads(review_audit["metadata_json"])
    updated_metadata = json.loads(updated_audit["metadata_json"])
    locked_package_metadata = json.loads(locked_package_audit["metadata_json"])
    reviews = list_report_review_summaries("viewer_1", "org_1", "report_1", store)
    locked_packages = list_report_evidence_package_summaries(
        "viewer_1",
        "org_1",
        "report_1",
        store,
    )
    locked_package_row = store.get_report_evidence_package(locked_packages[0].id)
    assert locked_package_row is not None
    locked_package_content = artifact_store.get_text("org_1", locked_package_row["storage_key"])
    locked_package = json.loads(locked_package_content)

    assert updated.status == ReportApprovalAssignmentStatus.APPROVED
    assert updated.notes == "Approved for locked package"
    assert updated.completed_at is not None
    assert reviews[0].reviewer_user_id == "manager_2"
    assert reviews[0].status == ReportReviewStatus.APPROVED
    assert reviews[0].notes == "Approved for locked package"
    assert review_audit["action"] == "report_review_recorded"
    assert review_metadata["assignment_id"] == created.id
    assert review_metadata["review_status"] == "approved"
    assert updated_audit["action"] == "report_approval_updated"
    assert updated_metadata["assigned_to_user_id"] == "manager_2"
    assert updated_metadata["previous_status"] == "pending"
    assert updated_metadata["status"] == "approved"
    assert updated_metadata["review_id"] == reviews[0].id
    assert len(locked_packages) == 1
    assert locked_packages[0].created_by_user_id == "manager_2"
    assert locked_packages[0].report_id == "report_1"
    assert locked_packages[0].review_count == 1
    assert locked_packages[0].upload_count == 1
    assert locked_package["approval_assignments"][0]["status"] == "approved"
    assert locked_package["approval_assignments"][0]["notes"] == "Approved for locked package"
    assert locked_package["reviews"][0]["id"] == reviews[0].id
    assert locked_package["reviews"][0]["notes"] == "Approved for locked package"
    assert "storage_key" not in locked_package["report"]
    assert "storage_key" not in json.dumps(locked_package["artifact"])
    assert report["storage_key"] not in locked_package_content
    assert locked_package_audit["action"] == "report_evidence_package_created"
    assert locked_package_audit["resource_type"] == "report_evidence_package"
    assert locked_package_audit["resource_id"] == locked_packages[0].id
    assert locked_package_metadata == {
        "approval_assignment_count": 1,
        "approval_assignment_approved_count": 1,
        "approval_assignment_id": created.id,
        "approval_policy_auto_lock_evidence_on_final_approval": True,
        "approval_policy_minimum_assignments_for_lock": 0,
        "approval_review_id": reviews[0].id,
        "approval_status": "approved",
        "auto_created": True,
        "comment_count": 0,
        "content_length": len(locked_package_content),
        "created_from": "approval_assignment",
        "delivery_count": 0,
        "format": "json",
        "package_id": locked_packages[0].id,
        "package_type": "iso50001_report_evidence",
        "report_id": "report_1",
        "report_type": "iso_summary",
        "review_count": 1,
        "schedule_count": 0,
        "storage_key_configured": True,
        "template_enabled_section_count": 6,
        "upload_count": 1,
    }
    assert "Approved for locked package" not in review_audit["metadata_json"]
    assert "Approved for locked package" not in updated_audit["metadata_json"]
    assert "Approved for locked package" not in locked_package_audit["metadata_json"]
    assert '"storage_key":' not in locked_package_audit["metadata_json"]
    assert "tenant_sensitive_note" not in locked_package_audit["metadata_json"]
    assigner_notifications = list_notification_summaries("manager_1", "org_1", store)
    assert len(assigner_notifications) == 1
    assert assigner_notifications[0].event_type == "report_approval_updated"
    assert assigner_notifications[0].recipient_user_id == "manager_1"
    assert assigner_notifications[0].resource_type == "report_approval_assignment"
    assert assigner_notifications[0].resource_id == created.id
    assert assigner_notifications[0].status == "pending"
    assert "approved" in assigner_notifications[0].body
    assert "Report digest:" in assigner_notifications[0].body
    assert "Source files: 4" in assigner_notifications[0].body
    assert "annual_kwh" not in assigner_notifications[0].body
    assert "storage_key" not in assigner_notifications[0].body
    assert "analysis_input_path" not in assigner_notifications[0].body
    assert report["storage_key"] not in assigner_notifications[0].body
    assert "Approved for locked package" not in assigner_notifications[0].body

    try:
        update_report_approval_assignment_summary(
            "manager_2",
            "org_1",
            "report_1",
            created.id,
            ReportApprovalAssignmentUpdate(status=ReportApprovalAssignmentStatus.CANCELED),
            store,
        )
        assert False, "Expected completed assignment updates to fail"
    except ValueError as exc:
        assert "already complete" in str(exc)

    route_created = client.post(
        "/organizations/org_1/reports/report_1/approval-assignments",
        headers={"X-User-Id": "owner_1"},
        json={"assigned_to_user_id": "manager_1"},
    )
    assert route_created.status_code == 200
    route_created_payload = route_created.json()
    route_assignment_notification = next(
        notification
        for notification in list_notification_summaries("manager_1", "org_1", store)
        if notification.resource_id == route_created_payload["id"]
        and notification.event_type == "report_approval_assigned"
    )
    assert "Report digest:" in route_assignment_notification.body
    assert "Source files: 4" in route_assignment_notification.body
    assert "annual_kwh" not in route_assignment_notification.body
    assert "storage_key" not in route_assignment_notification.body
    assert "analysis_input_path" not in route_assignment_notification.body
    assert report["storage_key"] not in route_assignment_notification.body
    route_updated = client.patch(
        f"/organizations/org_1/reports/report_1/approval-assignments/{route_created_payload['id']}",
        headers={"X-User-Id": "manager_1"},
        json={"status": "changes_requested", "notes": "Clarify the baseline note"},
    )
    assert route_updated.status_code == 200
    assert route_updated.json()["status"] == "changes_requested"
    route_update_notification = next(
        notification
        for notification in list_notification_summaries("owner_1", "org_1", store)
        if notification.resource_id == route_created_payload["id"]
        and notification.event_type == "report_approval_updated"
    )
    assert "Report digest:" in route_update_notification.body
    assert "Source files: 4" in route_update_notification.body
    assert "tenant_sensitive_note" not in route_update_notification.body
    assert "storage_key" not in route_update_notification.body
    assert "analysis_input_path" not in route_update_notification.body
    assert report["storage_key"] not in route_update_notification.body

    try:
        create_report_approval_assignment_summary(
            "manager_1",
            "org_1",
            "report_1",
            ReportApprovalAssignmentCreate(assigned_to_user_id="viewer_1"),
            store,
        )
        assert False, "Expected viewer assignment to be denied"
    except AccessDenied:
        pass

    try:
        create_report_approval_assignment_summary(
            "viewer_1",
            "org_1",
            "report_1",
            ReportApprovalAssignmentCreate(assigned_to_user_id="manager_1"),
            store,
        )
        assert False, "Expected viewer assignment creation to be denied"
    except AccessDenied:
        pass

    try:
        list_report_approval_assignment_summaries("owner_2", "org_1", "report_1", store)
        assert False, "Expected cross-tenant assignment listing to be denied"
    except AccessDenied:
        pass

    try:
        update_report_approval_assignment_summary(
            "owner_2",
            "org_2",
            "report_1",
            created.id,
            ReportApprovalAssignmentUpdate(status=ReportApprovalAssignmentStatus.CANCELED),
            store,
        )
        assert False, "Expected cross-tenant assignment update to be denied"
    except AccessDenied:
        pass

    store.create_report(
        Report(
            id="report_expired_assignment",
            organization_id="org_1",
            run_id="run_1",
            report_type="iso_summary",
            storage_key="tenants/org_1/sites/site_1/runs/run_1/reports/expired-assignment.json",
            is_published=True,
            expires_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
    )
    try:
        create_report_approval_assignment_summary(
            "manager_1",
            "org_1",
            "report_expired_assignment",
            ReportApprovalAssignmentCreate(assigned_to_user_id="manager_2"),
            store,
        )
        assert False, "Expected expired report assignment creation to fail"
    except ValueError as exc:
        assert "expired" in str(exc)


def test_report_approval_assignment_auto_locks_after_final_pending_approval():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    store.create_user(User(id="manager_1", email="manager1@example.com", display_name="Manager One"))
    store.create_user(User(id="manager_2", email="manager2@example.com", display_name="Manager Two"))
    store.create_user(User(id="viewer_1", email="viewer@example.com", display_name="Viewer"))
    for user_id, role in (
        ("manager_1", Role.MANAGER),
        ("manager_2", Role.MANAGER),
        ("viewer_1", Role.VIEWER),
    ):
        store.add_membership(
            Membership(
                user_id=user_id,
                organization_id="org_1",
                role=role,
                invited_by_user_id="owner_1",
            )
        )
    store.create_site(Site(id="site_1", organization_id="org_1", name="Main Site", timezone="Europe/London"))
    execute_local_analysis_run(
        "owner_1",
        "org_1",
        LocalAnalysisRunCreate(
            site_id="site_1",
            upload_id="upload_1",
            run_id="run_1",
            report_id="report_1",
            filename="energy.csv",
            rows=[
                {"Date": "2025-01-01", "Main Electricity": 100, "Main Gas": 80},
                {"Date": "2025-02-01", "Main Electricity": 120, "Main Gas": 70},
            ],
            client_config=load_client_config("config/clients/example_client.yaml"),
        ),
        store,
        artifact_store,
    )

    first_assignment = create_report_approval_assignment_summary(
        "owner_1",
        "org_1",
        "report_1",
        ReportApprovalAssignmentCreate(assigned_to_user_id="manager_1"),
        store,
        artifact_store,
    )
    second_assignment = create_report_approval_assignment_summary(
        "owner_1",
        "org_1",
        "report_1",
        ReportApprovalAssignmentCreate(assigned_to_user_id="manager_2"),
        store,
        artifact_store,
    )

    update_report_approval_assignment_summary(
        "manager_1",
        "org_1",
        "report_1",
        first_assignment.id,
        ReportApprovalAssignmentUpdate(
            status=ReportApprovalAssignmentStatus.APPROVED,
            notes="First approver accepts",
        ),
        store,
        artifact_store,
    )

    assert list_report_evidence_package_summaries("viewer_1", "org_1", "report_1", store) == []

    update_report_approval_assignment_summary(
        "manager_2",
        "org_1",
        "report_1",
        second_assignment.id,
        ReportApprovalAssignmentUpdate(
            status=ReportApprovalAssignmentStatus.APPROVED,
            notes="Second approver accepts",
        ),
        store,
        artifact_store,
    )

    locked_packages = list_report_evidence_package_summaries(
        "viewer_1",
        "org_1",
        "report_1",
        store,
    )
    package_row = store.get_report_evidence_package(locked_packages[0].id)
    assert package_row is not None
    locked_package = json.loads(artifact_store.get_text("org_1", package_row["storage_key"]))
    audit_metadata = json.loads(store.list_audit_events("org_1")[-1]["metadata_json"])

    assert len(locked_packages) == 1
    assert locked_packages[0].created_by_user_id == "manager_2"
    assert locked_packages[0].review_count == 2
    assert [assignment["status"] for assignment in locked_package["approval_assignments"]] == [
        "approved",
        "approved",
    ]
    assert [review["status"] for review in locked_package["reviews"]] == [
        "approved",
        "approved",
    ]
    assert audit_metadata["approval_assignment_count"] == 2
    assert audit_metadata["approval_assignment_approved_count"] == 2
    assert audit_metadata["approval_assignment_id"] == second_assignment.id
    assert audit_metadata["approval_policy_auto_lock_evidence_on_final_approval"] is True
    assert audit_metadata["approval_policy_minimum_assignments_for_lock"] == 0
    assert audit_metadata["auto_created"] is True
    assert audit_metadata["created_from"] == "approval_assignment"
    assert audit_metadata["review_count"] == 2
    assert "First approver accepts" not in store.list_audit_events("org_1")[-1]["metadata_json"]
    assert "Second approver accepts" not in store.list_audit_events("org_1")[-1]["metadata_json"]


def test_report_approval_policy_controls_lock_threshold_and_auto_lock():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    store.create_user(User(id="manager_1", email="manager1@example.com", display_name="Manager One"))
    store.create_user(User(id="manager_2", email="manager2@example.com", display_name="Manager Two"))
    store.create_user(User(id="viewer_1", email="viewer@example.com", display_name="Viewer"))
    for user_id, role in (
        ("manager_1", Role.MANAGER),
        ("manager_2", Role.MANAGER),
        ("viewer_1", Role.VIEWER),
    ):
        store.add_membership(
            Membership(
                user_id=user_id,
                organization_id="org_1",
                role=role,
                invited_by_user_id="owner_1",
            )
        )
    store.create_site(Site(id="site_1", organization_id="org_1", name="Main Site", timezone="Europe/London"))
    execute_local_analysis_run(
        "owner_1",
        "org_1",
        LocalAnalysisRunCreate(
            site_id="site_1",
            upload_id="upload_1",
            run_id="run_1",
            report_id="report_1",
            filename="energy.csv",
            rows=[
                {"Date": "2025-01-01", "Main Electricity": 100, "Main Gas": 80},
                {"Date": "2025-02-01", "Main Electricity": 120, "Main Gas": 70},
            ],
            client_config=load_client_config("config/clients/example_client.yaml"),
        ),
        store,
        artifact_store,
    )
    update_report_approval_policy_summary(
        "owner_1",
        "org_1",
        OrganizationReportApprovalPolicyUpdate(
            auto_lock_evidence_on_final_approval=False,
            minimum_approval_assignments_for_lock=2,
        ),
        store,
    )

    first_assignment = create_report_approval_assignment_summary(
        "owner_1",
        "org_1",
        "report_1",
        ReportApprovalAssignmentCreate(assigned_to_user_id="manager_1"),
        store,
        artifact_store,
    )
    update_report_approval_assignment_summary(
        "manager_1",
        "org_1",
        "report_1",
        first_assignment.id,
        ReportApprovalAssignmentUpdate(
            status=ReportApprovalAssignmentStatus.APPROVED,
            notes="First policy approver accepts",
        ),
        store,
        artifact_store,
    )

    assert list_report_evidence_package_summaries("viewer_1", "org_1", "report_1", store) == []
    try:
        create_report_evidence_package_snapshot(
            "owner_1",
            "org_1",
            "report_1",
            store,
            artifact_store,
        )
        assert False, "Expected policy threshold to block manual lock"
    except ValueError as exc:
        assert "at least 2 approved approval assignments" in str(exc)

    second_assignment = create_report_approval_assignment_summary(
        "owner_1",
        "org_1",
        "report_1",
        ReportApprovalAssignmentCreate(assigned_to_user_id="manager_2"),
        store,
        artifact_store,
    )
    update_report_approval_assignment_summary(
        "manager_2",
        "org_1",
        "report_1",
        second_assignment.id,
        ReportApprovalAssignmentUpdate(
            status=ReportApprovalAssignmentStatus.APPROVED,
            notes="Second policy approver accepts",
        ),
        store,
        artifact_store,
    )

    assert list_report_evidence_package_summaries("viewer_1", "org_1", "report_1", store) == []

    snapshot = create_report_evidence_package_snapshot(
        "owner_1",
        "org_1",
        "report_1",
        store,
        artifact_store,
    )
    audit_metadata = json.loads(store.list_audit_events("org_1")[-1]["metadata_json"])

    assert snapshot.created_by_user_id == "owner_1"
    assert audit_metadata["approval_assignment_count"] == 2
    assert audit_metadata["approval_assignment_approved_count"] == 2
    assert audit_metadata["approval_policy_auto_lock_evidence_on_final_approval"] is False
    assert audit_metadata["approval_policy_minimum_assignments_for_lock"] == 2
    assert audit_metadata["approval_status"] == "approved"
    assert "First policy approver accepts" not in store.list_audit_events("org_1")[-1]["metadata_json"]
    assert "Second policy approver accepts" not in store.list_audit_events("org_1")[-1]["metadata_json"]


def test_report_reviews_are_tenant_guarded_manager_only_visible_and_audited():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_2",
            email="other@example.com",
            display_name="Other Owner",
            organization_id="org_2",
            organization_name="Other Energy",
            organization_slug="other-energy",
        ),
        store,
    )
    store.create_user(User(id="manager_1", email="manager@example.com", display_name="Manager"))
    store.create_user(User(id="viewer_1", email="viewer@example.com", display_name="Viewer"))
    store.add_membership(
        Membership(
            user_id="manager_1",
            organization_id="org_1",
            role=Role.MANAGER,
            invited_by_user_id="owner_1",
        )
    )
    store.add_membership(
        Membership(
            user_id="viewer_1",
            organization_id="org_1",
            role=Role.VIEWER,
            invited_by_user_id="owner_1",
        )
    )
    store.create_site(Site(id="site_1", organization_id="org_1", name="Main Site", timezone="Europe/London"))
    payload = LocalAnalysisRunCreate(
        site_id="site_1",
        upload_id="upload_1",
        run_id="run_1",
        report_id="report_1",
        filename="energy.csv",
        rows=[
            {"Date": "2025-01-01", "Main Electricity": 100, "Main Gas": 80},
            {"Date": "2025-02-01", "Main Electricity": 120, "Main Gas": 70},
        ],
        client_config=load_client_config("config/clients/example_client.yaml"),
    )
    execute_local_analysis_run("owner_1", "org_1", payload, store, artifact_store)

    assert list_report_review_summaries("viewer_1", "org_1", "report_1", store) == []

    created = create_report_review_summary(
        "manager_1",
        "org_1",
        "report_1",
        ReportReviewCreate(
            status=ReportReviewStatus.APPROVED,
            notes=" Approved for management review evidence ",
        ),
        store,
    )
    audit_events = store.list_audit_events("org_1")
    audit_metadata = json.loads(audit_events[-1]["metadata_json"])

    assert created.organization_id == "org_1"
    assert created.report_id == "report_1"
    assert created.reviewer_user_id == "manager_1"
    assert created.status == ReportReviewStatus.APPROVED
    assert created.notes == "Approved for management review evidence"
    assert audit_events[-1]["action"] == "report_review_recorded"
    assert audit_events[-1]["resource_type"] == "report_review"
    assert audit_events[-1]["resource_id"] == created.id
    assert audit_metadata == {
        "notes_length": len("Approved for management review evidence"),
        "report_id": "report_1",
        "report_type": "iso_summary",
        "review_status": "approved",
    }
    assert "Approved for management review evidence" not in audit_events[-1]["metadata_json"]

    client = TestClient(create_app(store=store, artifact_store=artifact_store))
    route_review = client.post(
        "/organizations/org_1/reports/report_1/reviews",
        headers={"X-User-Id": "owner_1"},
        json={"status": "changes_requested", "notes": "Clarify SEU variance"},
    )
    route_reviews = client.get(
        "/organizations/org_1/reports/report_1/reviews",
        headers={"X-User-Id": "viewer_1"},
    )

    assert route_review.status_code == 200
    assert route_review.json()["status"] == "changes_requested"
    assert route_review.json()["notes"] == "Clarify SEU variance"
    assert route_reviews.status_code == 200
    assert [review["status"] for review in route_reviews.json()] == [
        "changes_requested",
        "approved",
    ]

    try:
        create_report_review_summary(
            "viewer_1",
            "org_1",
            "report_1",
            ReportReviewCreate(status=ReportReviewStatus.CHANGES_REQUESTED),
            store,
        )
        assert False, "Expected viewer report review decision to be denied"
    except AccessDenied:
        pass

    try:
        list_report_review_summaries("owner_2", "org_1", "report_1", store)
        assert False, "Expected cross-tenant report review listing to be denied"
    except AccessDenied:
        pass

    store.create_report(
        Report(
            id="report_expired_review",
            organization_id="org_1",
            run_id="run_1",
            report_type="iso_summary",
            storage_key="tenants/org_1/sites/site_1/runs/run_1/reports/expired-review.json",
            is_published=True,
            expires_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
    )
    try:
        create_report_review_summary(
            "manager_1",
            "org_1",
            "report_expired_review",
            ReportReviewCreate(status=ReportReviewStatus.APPROVED),
            store,
        )
        assert False, "Expected expired report review creation to fail"
    except ValueError as exc:
        assert "expired" in str(exc)


def test_report_comments_are_tenant_guarded_visible_and_audited():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_2",
            email="other@example.com",
            display_name="Other Owner",
            organization_id="org_2",
            organization_name="Other Energy",
            organization_slug="other-energy",
        ),
        store,
    )
    store.create_user(User(id="viewer_1", email="viewer@example.com", display_name="Viewer"))
    store.add_membership(
        Membership(
            user_id="viewer_1",
            organization_id="org_1",
            role=Role.VIEWER,
            invited_by_user_id="owner_1",
        )
    )
    store.create_site(Site(id="site_1", organization_id="org_1", name="Main Site", timezone="Europe/London"))
    payload = LocalAnalysisRunCreate(
        site_id="site_1",
        upload_id="upload_1",
        run_id="run_1",
        report_id="report_1",
        filename="energy.csv",
        rows=[
            {"Date": "2025-01-01", "Main Electricity": 100, "Main Gas": 80},
            {"Date": "2025-02-01", "Main Electricity": 120, "Main Gas": 70},
        ],
        client_config=load_client_config("config/clients/example_client.yaml"),
    )
    execute_local_analysis_run("owner_1", "org_1", payload, store, artifact_store)

    assert list_report_comment_summaries("owner_1", "org_1", "report_1", store) == []

    created = create_report_comment_summary(
        "viewer_1",
        "org_1",
        "report_1",
        ReportCommentCreate(body=" Ready for management review "),
        store,
    )
    audit_events = store.list_audit_events("org_1")
    audit_metadata = json.loads(audit_events[-1]["metadata_json"])

    assert created.organization_id == "org_1"
    assert created.report_id == "report_1"
    assert created.author_user_id == "viewer_1"
    assert created.body == "Ready for management review"
    assert audit_events[-1]["action"] == "report_comment_created"
    assert audit_events[-1]["resource_type"] == "report_comment"
    assert audit_events[-1]["resource_id"] == created.id
    assert audit_metadata == {
        "body_length": len("Ready for management review"),
        "report_id": "report_1",
        "report_type": "iso_summary",
    }
    assert "Ready for management review" not in audit_events[-1]["metadata_json"]

    client = TestClient(create_app(store=store, artifact_store=artifact_store))
    route_comment = client.post(
        "/organizations/org_1/reports/report_1/comments",
        headers={"X-User-Id": "viewer_1"},
        json={"body": " Final evidence check complete "},
    )
    route_comments = client.get(
        "/organizations/org_1/reports/report_1/comments",
        headers={"X-User-Id": "owner_1"},
    )

    assert route_comment.status_code == 200
    assert route_comment.json()["body"] == "Final evidence check complete"
    assert route_comments.status_code == 200
    assert [comment["body"] for comment in route_comments.json()] == [
        "Ready for management review",
        "Final evidence check complete",
    ]

    try:
        list_report_comment_summaries("owner_2", "org_1", "report_1", store)
        assert False, "Expected cross-tenant report comment listing to be denied"
    except AccessDenied:
        pass

    try:
        create_report_comment_summary(
            "owner_2",
            "org_2",
            "report_1",
            ReportCommentCreate(body="Cross-tenant note"),
            store,
        )
        assert False, "Expected cross-tenant report comment creation to be denied"
    except AccessDenied:
        pass

    store.create_report(
        Report(
            id="report_expired",
            organization_id="org_1",
            run_id="run_1",
            report_type="iso_summary",
            storage_key="tenants/org_1/sites/site_1/runs/run_1/reports/expired-comments.json",
            is_published=True,
            expires_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
    )
    try:
        create_report_comment_summary(
            "owner_1",
            "org_1",
            "report_expired",
            ReportCommentCreate(body="Expired report note"),
            store,
        )
        assert False, "Expected expired report comment creation to fail"
    except ValueError as exc:
        assert "expired" in str(exc)


def test_report_retention_policy_filters_artifact_access_and_audits():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    store.create_site(Site(id="site_1", organization_id="org_1", name="Main Site", timezone="Europe/London"))
    store.create_upload(
        Upload(
            id="upload_1",
            organization_id="org_1",
            site_id="site_1",
            uploaded_by_user_id="owner_1",
            category="energy",
            storage_key="tenants/org_1/sites/site_1/uploads/upload_1/energy.csv",
            checksum="abc123",
            status=UploadStatus.STORED,
        )
    )
    create_run_summary(
        "owner_1",
        "org_1",
        RunCreate(run_id="run_1", site_id="site_1", upload_ids=["upload_1"]),
        store,
    )
    expired_storage_key = "tenants/org_1/sites/site_1/runs/run_1/reports/expired.json"
    active_storage_key = "tenants/org_1/sites/site_1/runs/run_1/reports/active.json"
    expired_package_storage_key = (
        "tenants/org_1/reports/report_expired/evidence-packages/"
        "package_expired/report-expired-evidence-package.json"
    )
    missing_package_storage_key = (
        "tenants/org_1/reports/report_expired/evidence-packages/"
        "package_missing/report-expired-missing-evidence-package.json"
    )
    active_package_storage_key = (
        "tenants/org_1/reports/report_active/evidence-packages/"
        "package_active/report-active-evidence-package.json"
    )
    artifact_store.put_text("org_1", expired_storage_key, "{\"expired\": true}")
    artifact_store.put_text("org_1", active_storage_key, "{\"active\": true}")
    artifact_store.put_text("org_1", expired_package_storage_key, "{\"expired_package\": true}")
    artifact_store.put_text("org_1", active_package_storage_key, "{\"active_package\": true}")
    now = datetime.now(timezone.utc)
    store.create_report(
        Report(
            id="report_expired",
            organization_id="org_1",
            run_id="run_1",
            report_type="iso_summary",
            storage_key=expired_storage_key,
            is_published=True,
            expires_at=now - timedelta(days=1),
        )
    )
    store.create_report(
        Report(
            id="report_active",
            organization_id="org_1",
            run_id="run_1",
            report_type="iso_summary",
            storage_key=active_storage_key,
            is_published=True,
            expires_at=now + timedelta(days=30),
        )
    )
    store.create_report_evidence_package(
        ReportEvidencePackage(
            id="package_expired",
            organization_id="org_1",
            report_id="report_expired",
            created_by_user_id="owner_1",
            storage_key=expired_package_storage_key,
            filename="report-expired-evidence-package.json",
            content_type="application/json",
            content_length=25,
            package_type="iso50001_report_evidence",
            format="json",
            comment_count=1,
            review_count=1,
            schedule_count=0,
            delivery_count=0,
            upload_count=1,
        )
    )
    store.create_report_evidence_package(
        ReportEvidencePackage(
            id="package_missing",
            organization_id="org_1",
            report_id="report_expired",
            created_by_user_id="owner_1",
            storage_key=missing_package_storage_key,
            filename="report-expired-missing-evidence-package.json",
            content_type="application/json",
            content_length=25,
            package_type="iso50001_report_evidence",
            format="json",
            comment_count=0,
            review_count=0,
            schedule_count=0,
            delivery_count=0,
            upload_count=1,
        )
    )
    store.create_report_evidence_package(
        ReportEvidencePackage(
            id="package_active",
            organization_id="org_1",
            report_id="report_active",
            created_by_user_id="owner_1",
            storage_key=active_package_storage_key,
            filename="report-active-evidence-package.json",
            content_type="application/json",
            content_length=24,
            package_type="iso50001_report_evidence",
            format="json",
            comment_count=1,
            review_count=1,
            schedule_count=0,
            delivery_count=0,
            upload_count=1,
        )
    )

    reports = list_report_summaries("owner_1", "org_1", store)

    assert [report.id for report in reports] == ["report_active"]
    try:
        get_report_artifact("owner_1", "org_1", "report_expired", store, artifact_store)
        assert False, "Expected expired report artifact read to fail"
    except ValueError as exc:
        assert "expired" in str(exc)

    client = TestClient(create_app(store=store, artifact_store=artifact_store))
    response = client.post(
        "/organizations/org_1/reports/retention/apply",
        headers={"X-User-Id": "owner_1"},
    )
    summary = response.json()
    refreshed = store.get_report("report_expired")
    audit_events = store.list_audit_events("org_1")
    retention_metadata = json.loads(audit_events[-1]["metadata_json"])

    assert response.status_code == 200
    assert summary["evaluated_count"] == 2
    assert summary["expired_count"] == 1
    assert summary["deleted_artifact_count"] == 2
    assert summary["missing_artifact_count"] == 1
    assert summary["deleted_evidence_package_artifact_count"] == 1
    assert summary["missing_evidence_package_artifact_count"] == 1
    assert summary["deleted_evidence_package_metadata_count"] == 2
    assert summary["retained_report_ids"] == ["report_expired"]
    assert not artifact_store.exists("org_1", expired_storage_key)
    assert not artifact_store.exists("org_1", expired_package_storage_key)
    assert artifact_store.exists("org_1", active_storage_key)
    assert artifact_store.exists("org_1", active_package_storage_key)
    assert store.list_report_evidence_packages("org_1", "report_expired") == []
    assert [package["id"] for package in store.list_report_evidence_packages("org_1", "report_active")] == [
        "package_active"
    ]
    assert refreshed is not None
    assert refreshed["is_published"] == 0
    assert refreshed["retention_applied_at"] is not None
    assert audit_events[-1]["action"] == "report_retention_applied"
    assert audit_events[-1]["resource_id"] == "report_expired"
    assert retention_metadata["artifact_location_configured"] is True
    assert retention_metadata["artifact_deleted"] is True
    assert retention_metadata["evidence_package_count"] == 2
    assert retention_metadata["evidence_package_artifact_deleted_count"] == 1
    assert retention_metadata["evidence_package_artifact_missing_count"] == 1
    assert retention_metadata["evidence_package_metadata_deleted_count"] == 2
    assert expired_storage_key not in audit_events[-1]["metadata_json"]
    assert expired_package_storage_key not in audit_events[-1]["metadata_json"]
    assert missing_package_storage_key not in audit_events[-1]["metadata_json"]
    assert "storage_key" not in audit_events[-1]["metadata_json"]

    repeat = client.post(
        "/organizations/org_1/reports/retention/apply",
        headers={"X-User-Id": "owner_1"},
    ).json()
    assert repeat["expired_count"] == 0
    assert repeat["deleted_artifact_count"] == 0
    assert repeat["deleted_evidence_package_artifact_count"] == 0
    assert repeat["deleted_evidence_package_metadata_count"] == 0


def test_report_artifact_access_is_tenant_guarded():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_2",
            email="other@example.com",
            display_name="Other Owner",
            organization_id="org_2",
            organization_name="Other Energy",
            organization_slug="other-energy",
        ),
        store,
    )
    store.create_site(Site(id="site_1", organization_id="org_1", name="Main Site", timezone="Europe/London"))
    payload = LocalAnalysisRunCreate(
        site_id="site_1",
        upload_id="upload_1",
        run_id="run_1",
        report_id="report_1",
        filename="energy.csv",
        rows=[
            {"Date": "2025-01-01", "Main Electricity": 100, "Main Gas": 80},
            {"Date": "2025-02-01", "Main Electricity": 120, "Main Gas": 70},
        ],
        client_config=load_client_config("config/clients/example_client.yaml"),
    )
    execute_local_analysis_run("owner_1", "org_1", payload, store, artifact_store)

    try:
        get_report_artifact("owner_2", "org_1", "report_1", store, artifact_store)
        assert False, "Expected cross-tenant report artifact read to be denied"
    except AccessDenied:
        pass


def test_upload_and_run_summaries_are_tenant_guarded():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="user_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    store.create_site(Site(id="site_1", organization_id="org_1", name="Main Site", timezone="Europe/London"))
    payload = LocalAnalysisRunCreate(
        site_id="site_1",
        upload_id="upload_1",
        run_id="run_1",
        report_id="report_1",
        filename="energy.csv",
        rows=[
            {"Date": "2025-01-01", "Main Electricity": 100, "Main Gas": 80},
            {"Date": "2025-02-01", "Main Electricity": 120, "Main Gas": 70},
        ],
        client_config=load_client_config("config/clients/example_client.yaml"),
    )
    execute_local_analysis_run("user_1", "org_1", payload, store)

    uploads = list_upload_summaries("user_1", "org_1", store, site_id="site_1")
    runs = list_run_summaries("user_1", "org_1", store, site_id="site_1")

    assert [upload.id for upload in uploads] == ["upload_1"]
    assert [run.id for run in runs] == ["run_1"]
    assert runs[0].status == "succeeded"


def test_upload_category_contract_normalizes_and_rejects_unsupported_values():
    assert normalize_upload_category("SEU-Mapping") == UploadCategory.SEU_MAPPING.value

    upload = Upload(
        id="upload_1",
        organization_id="org_1",
        site_id="site_1",
        uploaded_by_user_id="user_1",
        category="HDD",
        storage_key="tenants/org_1/sites/site_1/uploads/upload_1/hdd.csv",
        checksum="abc123",
        status=UploadStatus.STORED,
    )

    assert upload.category == UploadCategory.HDD.value

    try:
        Upload(
            id="upload_2",
            organization_id="org_1",
            site_id="site_1",
            uploaded_by_user_id="user_1",
            category="meter-data",
            storage_key="tenants/org_1/sites/site_1/uploads/upload_2/meter-data.csv",
            checksum="abc123",
            status=UploadStatus.STORED,
        )
        assert False, "Expected unsupported upload category to be rejected"
    except ValueError as exc:
        assert "upload.category" in str(exc)


def test_upload_readiness_summary_groups_source_files_by_site():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    store.create_site(Site(id="site_1", organization_id="org_1", name="Main Site", timezone="Europe/London"))
    store.create_site(Site(id="site_2", organization_id="org_1", name="Second Site", timezone="Europe/London"))
    create_upload_summary(
        "owner_1",
        "org_1",
        UploadCreate(
            upload_id="upload_1",
            site_id="site_1",
            category="energy",
            filename="energy.csv",
            checksum="abc123",
        ),
        store,
    )
    create_upload_summary(
        "owner_1",
        "org_1",
        UploadCreate(
            upload_id="upload_2",
            site_id="site_1",
            category="HDD",
            filename="hdd.csv",
            checksum="def456",
        ),
        store,
    )
    create_upload_summary(
        "owner_1",
        "org_1",
        UploadCreate(
            upload_id="upload_3",
            site_id="site_2",
            category="cdd",
            filename="cdd.csv",
            checksum="ghi789",
        ),
        store,
    )

    summary = get_upload_readiness_summary("owner_1", "org_1", store)
    site_1 = next(site for site in summary.sites if site.site_id == "site_1")
    site_2 = next(site for site in summary.sites if site.site_id == "site_2")
    energy_file = next(file for file in site_1.files if file.category == UploadCategory.ENERGY.value)
    hdd_file = next(file for file in site_1.files if file.category == UploadCategory.HDD.value)
    missing_file = next(file for file in site_1.files if file.category == UploadCategory.SEU_MAPPING.value)

    assert summary.required_categories == ["energy"]
    assert summary.site_count == 2
    assert summary.ready_site_count == 1
    assert site_1.ready_count == 1
    assert site_1.required_count == 1
    assert site_1.is_ready
    assert site_2.ready_count == 0
    assert energy_file.is_ready
    assert energy_file.latest_upload is not None
    assert energy_file.latest_upload.id == "upload_1"
    assert hdd_file.latest_upload is not None
    assert hdd_file.latest_upload.category == "hdd"
    assert not missing_file.is_ready
    assert missing_file.latest_upload is None

    filtered = get_upload_readiness_summary("owner_1", "org_1", store, site_id="site_1")

    assert filtered.site_count == 1
    assert filtered.sites[0].site_id == "site_1"


def test_upload_readiness_summary_rejects_cross_tenant_site_filter():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_2",
            email="other@example.com",
            display_name="Other Owner",
            organization_id="org_2",
            organization_name="Other Energy",
            organization_slug="other-energy",
        ),
        store,
    )
    store.create_site(Site(id="site_2", organization_id="org_2", name="Other Site", timezone="Europe/London"))

    try:
        get_upload_readiness_summary("owner_1", "org_1", store, site_id="site_2")
        assert False, "Expected cross-tenant readiness filter to be denied"
    except AccessDenied:
        pass


def test_upload_readiness_route_returns_tenant_site_contract():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    store.create_site(Site(id="site_1", organization_id="org_1", name="Main Site", timezone="Europe/London"))
    create_upload_summary(
        "owner_1",
        "org_1",
        UploadCreate(
            upload_id="upload_1",
            site_id="site_1",
            category="energy",
            filename="energy.csv",
            checksum="abc123",
        ),
        store,
    )
    client = TestClient(create_app(store=store))

    response = client.get(
        "/organizations/org_1/uploads/readiness?site_id=site_1",
        headers={"X-User-Id": "owner_1"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["organization_id"] == "org_1"
    assert body["site_count"] == 1
    assert body["sites"][0]["site_id"] == "site_1"
    assert body["sites"][0]["ready_count"] == 1
    assert body["sites"][0]["files"][0]["latest_upload"]["id"] == "upload_1"


def test_create_upload_summary_requires_upload_permission():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_2",
            email="other@example.com",
            display_name="Other Owner",
            organization_id="org_2",
            organization_name="Other Energy",
            organization_slug="other-energy",
        ),
        store,
    )
    store.create_site(Site(id="site_1", organization_id="org_1", name="Main Site", timezone="Europe/London"))

    summary = create_upload_summary(
        "owner_1",
        "org_1",
        UploadCreate(
            upload_id="upload_1",
            site_id="site_1",
            category="energy",
            filename="energy.csv",
            checksum="abc123",
        ),
        store,
    )

    assert summary.id == "upload_1"
    assert summary.status == "stored"

    try:
        create_upload_summary(
            "owner_2",
            "org_1",
            UploadCreate(
                upload_id="upload_2",
                site_id="site_1",
                category="energy",
                filename="energy.csv",
                checksum="abc123",
            ),
            store,
        )
        assert False, "Expected cross-tenant upload creation to be denied"
    except AccessDenied:
        pass


def test_create_upload_content_summary_persists_tenant_artifact_and_metadata():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    store.create_site(Site(id="site_1", organization_id="org_1", name="Main Site", timezone="Europe/London"))
    content = b"Date,Main Electricity\n2025-01-01,100\n"

    summary = create_upload_content_summary(
        "owner_1",
        "org_1",
        UploadContentCreate(
            upload_id="upload_1",
            site_id="site_1",
            category="energy",
            filename="Energy Data.csv",
            content_base64=base64.b64encode(content).decode("ascii"),
        ),
        store,
        artifact_store,
    )

    assert summary.id == "upload_1"
    assert "storage_key" not in summary.model_dump()
    assert summary.checksum == sha256(content).hexdigest()
    expected_storage_key = "tenants/org_1/sites/site_1/uploads/upload_1/energy-data.csv"
    assert artifact_store.get_bytes("org_1", expected_storage_key) == content
    audit_event = store.list_audit_events("org_1")[0]
    assert audit_event["action"] == "upload_stored"
    assert json.loads(audit_event["metadata_json"]) == {
        "artifact_location_configured": "true",
        "category": "energy",
        "site_id": "site_1",
    }
    assert "storage_key" not in audit_event["metadata_json"]
    assert expected_storage_key not in audit_event["metadata_json"]


def test_create_upload_content_summary_accepts_legacy_source_file_shapes():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    store.create_site(Site(id="site_1", organization_id="org_1", name="Main Site", timezone="Europe/London"))
    legacy_energy = (
        b"Metered Sector,Utility,Units,Period,01/01/2025\n"
        b"Boiler gas,Gas,kWh,Total,10\n"
        b",,,Day,9\n"
    )
    weather_with_preamble = (
        b"Description:,Celsius-based heating degree days\n"
        b"Source:,degreedays.net\n"
        b"\n"
        b"Date,HDD 15.5\n"
        b"01/01/2025,6.2\n"
    )
    seu_mapping = b"Meter,SEU_Category\nBoiler gas,Boiler\n"

    energy_summary = create_upload_content_summary(
        "owner_1",
        "org_1",
        UploadContentCreate(
            upload_id="upload_energy",
            site_id="site_1",
            category="energy",
            filename="legacy-energy.csv",
            content_base64=base64.b64encode(legacy_energy).decode("ascii"),
        ),
        store,
        artifact_store,
    )
    hdd_summary = create_upload_content_summary(
        "owner_1",
        "org_1",
        UploadContentCreate(
            upload_id="upload_hdd",
            site_id="site_1",
            category="hdd",
            filename="hdd.csv",
            content_base64=base64.b64encode(weather_with_preamble).decode("ascii"),
        ),
        store,
        artifact_store,
    )
    mapping_summary = create_upload_content_summary(
        "owner_1",
        "org_1",
        UploadContentCreate(
            upload_id="upload_mapping",
            site_id="site_1",
            category="seu_mapping",
            filename="mapping.csv",
            content_base64=base64.b64encode(seu_mapping).decode("ascii"),
        ),
        store,
        artifact_store,
    )

    assert [summary.category for summary in [energy_summary, hdd_summary, mapping_summary]] == [
        "energy",
        "hdd",
        "seu_mapping",
    ]
    assert sorted(upload["id"] for upload in store.list_uploads("org_1")) == [
        "upload_energy",
        "upload_hdd",
        "upload_mapping",
    ]


def test_create_upload_content_summary_rejects_source_contract_mismatch_before_storage():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    store.create_site(Site(id="site_1", organization_id="org_1", name="Main Site", timezone="Europe/London"))

    try:
        create_upload_content_summary(
            "owner_1",
            "org_1",
            UploadContentCreate(
                upload_id="upload_bad_hdd",
                site_id="site_1",
                category="hdd",
                filename="not-hdd.csv",
                content_base64=base64.b64encode(
                    b"Date,Main Electricity\n2025-01-01,100\n"
                ).decode("ascii"),
            ),
            store,
            artifact_store,
        )
        assert False, "Expected mislabeled source file upload to be rejected"
    except ValueError as exc:
        assert "hdd upload failed source contract validation" in str(exc)

    assert store.list_uploads("org_1") == []
    assert store.list_audit_events("org_1") == []


def test_create_upload_content_summary_rejects_invalid_base64():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    store.create_site(Site(id="site_1", organization_id="org_1", name="Main Site", timezone="Europe/London"))

    try:
        create_upload_content_summary(
            "owner_1",
            "org_1",
            UploadContentCreate(
                upload_id="upload_1",
                site_id="site_1",
                category="energy",
                filename="energy.csv",
                content_base64="not valid base64",
            ),
            store,
            artifact_store,
        )
        assert False, "Expected invalid upload content to be rejected"
    except ValueError as exc:
        assert "base64" in str(exc)

    assert store.list_uploads("org_1") == []


def test_create_upload_content_summary_rejects_cross_tenant_site_id():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_2",
            email="other@example.com",
            display_name="Other Owner",
            organization_id="org_2",
            organization_name="Other Energy",
            organization_slug="other-energy",
        ),
        store,
    )
    store.create_site(Site(id="site_2", organization_id="org_2", name="Other Site", timezone="Europe/London"))

    try:
        create_upload_content_summary(
            "owner_1",
            "org_1",
            UploadContentCreate(
                upload_id="upload_1",
                site_id="site_2",
                category="energy",
                filename="energy.csv",
                content_base64=base64.b64encode(b"Date,Value\n").decode("ascii"),
            ),
            store,
            artifact_store,
        )
        assert False, "Expected cross-tenant site upload content to be denied"
    except AccessDenied:
        pass

    assert store.list_uploads("org_1") == []


def test_create_upload_summary_rejects_cross_tenant_site_id():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_2",
            email="other@example.com",
            display_name="Other Owner",
            organization_id="org_2",
            organization_name="Other Energy",
            organization_slug="other-energy",
        ),
        store,
    )
    store.create_site(Site(id="site_2", organization_id="org_2", name="Other Site", timezone="Europe/London"))

    try:
        create_upload_summary(
            "owner_1",
            "org_1",
            UploadCreate(
                upload_id="upload_1",
                site_id="site_2",
                category="energy",
                filename="energy.csv",
                checksum="abc123",
            ),
            store,
        )
        assert False, "Expected cross-tenant site upload to be denied"
    except AccessDenied:
        pass

    assert store.list_uploads("org_1") == []


def test_create_site_summary_requires_asset_management_permission():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_2",
            email="other@example.com",
            display_name="Other Owner",
            organization_id="org_2",
            organization_name="Other Energy",
            organization_slug="other-energy",
        ),
        store,
    )

    created = create_site_summary(
        "owner_1",
        "org_1",
        SiteCreate(site_id="site_1", name="Main Site", timezone="Europe/London"),
        store,
    )
    assert created.id == "site_1"
    assert created.organization_id == "org_1"
    audit_events = store.list_audit_events("org_1")
    assert [event["action"] for event in audit_events] == ["site_created"]
    assert audit_events[0]["actor_user_id"] == "owner_1"
    assert audit_events[0]["resource_type"] == "site"
    assert audit_events[0]["resource_id"] == "site_1"
    assert json.loads(audit_events[0]["metadata_json"]) == {
        "name": "Main Site",
        "timezone": "Europe/London",
    }

    try:
        create_site_summary(
            "owner_2",
            "org_1",
            SiteCreate(site_id="site_2", name="Blocked Site", timezone="Europe/London"),
            store,
        )
        assert False, "Expected cross-tenant site creation to be denied"
    except AccessDenied:
        pass
    assert [event["action"] for event in store.list_audit_events("org_1")] == ["site_created"]
    assert store.list_audit_events("org_2") == []


def test_create_meter_summary_requires_asset_management_permission():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_2",
            email="other@example.com",
            display_name="Other Owner",
            organization_id="org_2",
            organization_name="Other Energy",
            organization_slug="other-energy",
        ),
        store,
    )
    store.create_site(Site(id="site_1", organization_id="org_1", name="Main Site", timezone="Europe/London"))

    created = create_meter_summary(
        "owner_1",
        "org_1",
        MeterCreate(
            meter_id="meter_1",
            site_id="site_1",
            display_name="Main Electricity",
            commodity="electricity",
            unit="kWh",
            source_column="Main Electricity",
            is_seu=True,
        ),
        store,
    )
    assert created.id == "meter_1"
    assert created.organization_id == "org_1"
    assert created.is_seu is True
    audit_events = store.list_audit_events("org_1")
    assert [event["action"] for event in audit_events] == ["meter_created"]
    assert audit_events[0]["actor_user_id"] == "owner_1"
    assert audit_events[0]["resource_type"] == "meter"
    assert audit_events[0]["resource_id"] == "meter_1"
    assert json.loads(audit_events[0]["metadata_json"]) == {
        "commodity": "electricity",
        "display_name": "Main Electricity",
        "is_seu": "true",
        "site_id": "site_1",
        "source_column": "Main Electricity",
        "unit": "kWh",
    }

    try:
        create_meter_summary(
            "owner_2",
            "org_1",
            MeterCreate(
                meter_id="meter_2",
                site_id="site_1",
                display_name="Blocked Meter",
                commodity="gas",
                unit="kWh",
                source_column="Main Gas",
                is_seu=False,
            ),
            store,
        )
        assert False, "Expected cross-tenant meter creation to be denied"
    except AccessDenied:
        pass
    assert [event["action"] for event in store.list_audit_events("org_1")] == ["meter_created"]


def test_manager_can_create_sites_and_meters():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    store.create_user(User(id="manager_1", email="manager@example.com", display_name="Manager"))
    store.add_membership(
        Membership(
            user_id="manager_1",
            organization_id="org_1",
            role=Role.MANAGER,
            invited_by_user_id="owner_1",
        )
    )

    site = create_site_summary(
        "manager_1",
        "org_1",
        SiteCreate(site_id="site_1", name="Main Site", timezone="Europe/London"),
        store,
    )
    meter = create_meter_summary(
        "manager_1",
        "org_1",
        MeterCreate(
            meter_id="meter_1",
            site_id="site_1",
            display_name="Main Electricity",
            commodity="electricity",
            unit="kWh",
            source_column="Main Electricity",
            is_seu=True,
        ),
        store,
    )

    assert site.organization_id == "org_1"
    assert meter.organization_id == "org_1"
    assert meter.site_id == "site_1"
    audit_events = store.list_audit_events("org_1")
    assert [event["action"] for event in audit_events] == ["site_created", "meter_created"]
    assert [event["actor_user_id"] for event in audit_events] == ["manager_1", "manager_1"]


def test_create_meter_summary_rejects_cross_tenant_site_id():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_2",
            email="other@example.com",
            display_name="Other Owner",
            organization_id="org_2",
            organization_name="Other Energy",
            organization_slug="other-energy",
        ),
        store,
    )
    store.create_site(Site(id="site_2", organization_id="org_2", name="Other Site", timezone="Europe/London"))

    try:
        create_meter_summary(
            "owner_1",
            "org_1",
            MeterCreate(
                meter_id="meter_1",
                site_id="site_2",
                display_name="Blocked Meter",
                commodity="gas",
                unit="kWh",
                source_column="Main Gas",
                is_seu=False,
            ),
            store,
        )
        assert False, "Expected cross-tenant site meter creation to be denied"
    except AccessDenied:
        pass

    assert store.list_meters("org_1") == []


def test_create_run_summary_requires_manage_runs_permission():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_2",
            email="other@example.com",
            display_name="Other Owner",
            organization_id="org_2",
            organization_name="Other Energy",
            organization_slug="other-energy",
        ),
        store,
    )
    store.create_site(Site(id="site_1", organization_id="org_1", name="Main Site", timezone="Europe/London"))
    store.create_upload(
        Upload(
            id="upload_1",
            organization_id="org_1",
            site_id="site_1",
            uploaded_by_user_id="owner_1",
            category="energy",
            storage_key="tenants/org_1/sites/site_1/uploads/upload_1/energy.csv",
            checksum="abc123",
            status=UploadStatus.STORED,
        )
    )

    created = create_run_summary(
        "owner_1",
        "org_1",
        RunCreate(run_id="run_1", site_id="site_1", upload_ids=["upload_1"]),
        store,
    )
    assert created.id == "run_1"
    assert created.organization_id == "org_1"
    assert created.status == "queued"

    try:
        create_run_summary(
            "owner_2",
            "org_1",
            RunCreate(run_id="run_2", site_id="site_1", upload_ids=["upload_1"]),
            store,
        )
        assert False, "Expected cross-tenant run creation to be denied"
    except AccessDenied:
        pass


def test_create_run_summary_rejects_cross_tenant_upload_id():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_2",
            email="other@example.com",
            display_name="Other Owner",
            organization_id="org_2",
            organization_name="Other Energy",
            organization_slug="other-energy",
        ),
        store,
    )
    store.create_site(Site(id="site_1", organization_id="org_1", name="Main Site", timezone="Europe/London"))
    store.create_site(Site(id="site_2", organization_id="org_2", name="Other Site", timezone="Europe/London"))
    store.create_upload(
        Upload(
            id="upload_2",
            organization_id="org_2",
            site_id="site_2",
            uploaded_by_user_id="owner_2",
            category="energy",
            storage_key="tenants/org_2/sites/site_2/uploads/upload_2/energy.csv",
            checksum="abc123",
            status=UploadStatus.STORED,
        )
    )

    try:
        create_run_summary(
            "owner_1",
            "org_1",
            RunCreate(run_id="run_1", site_id="site_1", upload_ids=["upload_2"]),
            store,
        )
        assert False, "Expected cross-tenant upload run creation to be denied"
    except AccessDenied:
        pass

    assert store.list_runs("org_1") == []


def test_create_run_summary_rejects_upload_from_another_site():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    store.create_site(Site(id="site_1", organization_id="org_1", name="Main Site", timezone="Europe/London"))
    store.create_site(Site(id="site_2", organization_id="org_1", name="Second Site", timezone="Europe/London"))
    store.create_upload(
        Upload(
            id="upload_1",
            organization_id="org_1",
            site_id="site_2",
            uploaded_by_user_id="owner_1",
            category="energy",
            storage_key="tenants/org_1/sites/site_2/uploads/upload_1/energy.csv",
            checksum="abc123",
            status=UploadStatus.STORED,
        )
    )

    try:
        create_run_summary(
            "owner_1",
            "org_1",
            RunCreate(run_id="run_1", site_id="site_1", upload_ids=["upload_1"]),
            store,
        )
        assert False, "Expected mismatched site upload to be rejected"
    except ValueError as exc:
        assert "does not belong to site" in str(exc)

    assert store.list_runs("org_1") == []


def test_audit_event_summaries_require_audit_permission():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="user_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    store.create_site(Site(id="site_1", organization_id="org_1", name="Main Site", timezone="Europe/London"))
    payload = LocalAnalysisRunCreate(
        site_id="site_1",
        upload_id="upload_1",
        run_id="run_1",
        report_id="report_1",
        filename="energy.csv",
        rows=[
            {"Date": "2025-01-01", "Main Electricity": 100, "Main Gas": 80},
            {"Date": "2025-02-01", "Main Electricity": 120, "Main Gas": 70},
        ],
        client_config=load_client_config("config/clients/example_client.yaml"),
    )
    execute_local_analysis_run("user_1", "org_1", payload, store)

    events = list_audit_event_summaries("user_1", "org_1", store)

    assert [event.action for event in events] == [
        "upload_stored",
        "run_started",
        "run_succeeded",
        "report_created",
    ]
    assert all(event.created_at for event in events)


def test_audit_export_returns_tenant_csv_and_requires_audit_permission():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    store.create_user(User(id="viewer_1", email="viewer@example.com", display_name="Viewer"))
    store.add_membership(
        Membership(
            user_id="viewer_1",
            organization_id="org_1",
            role=Role.VIEWER,
            invited_by_user_id="owner_1",
        )
    )
    store.create_site(Site(id="site_1", organization_id="org_1", name="Main Site", timezone="Europe/London"))
    payload = LocalAnalysisRunCreate(
        site_id="site_1",
        upload_id="upload_1",
        run_id="run_1",
        report_id="report_1",
        filename="energy.csv",
        rows=[
            {"Date": "2025-01-01", "Main Electricity": 100, "Main Gas": 80},
            {"Date": "2025-02-01", "Main Electricity": 120, "Main Gas": 70},
        ],
        client_config=load_client_config("config/clients/example_client.yaml"),
    )
    execute_local_analysis_run("owner_1", "org_1", payload, store)

    export = export_audit_events("owner_1", "org_1", store)

    assert export.filename == "org_1-audit-events.csv"
    assert export.content_type == "text/csv"
    assert export.event_count == 4
    assert "id,organization_id,created_at,actor_user_id,action,resource_type,resource_id,metadata_json" in export.content
    assert "run_succeeded" in export.content
    assert "report_created" in export.content

    try:
        export_audit_events("viewer_1", "org_1", store)
        assert False, "Expected viewer audit export to be denied"
    except AccessDenied:
        pass


def test_audit_list_and_export_redact_legacy_sensitive_metadata():
    store = SaaSStore(initialize_database())
    onboard_owner_organization(
        OwnerOrganizationCreate(
            user_id="owner_1",
            email="owner@example.com",
            display_name="Owner",
            organization_id="org_1",
            organization_name="Example Energy",
            organization_slug="example-energy",
        ),
        store,
    )
    store.create_audit_event(
        AuditEvent(
            id="audit_legacy_invite",
            organization_id="org_1",
            actor_user_id="owner_1",
            action=AuditAction.INVITE_CREATED,
            resource_type="organization_invite",
            resource_id="invite_1",
            metadata={"email": "invitee@example.com", "role": "manager"},
        )
    )
    legacy_report_storage_key = "tenants/org_1/sites/site_1/runs/run_1/reports/iso-summary.json"
    store.create_audit_event(
        AuditEvent(
            id="audit_legacy_report_access",
            organization_id="org_1",
            actor_user_id="owner_1",
            action=AuditAction.REPORT_ACCESSED,
            resource_type="report",
            resource_id="report_1",
            metadata={
                "report_type": "iso_summary",
                "storage_key": legacy_report_storage_key,
            },
        )
    )
    legacy_upload_storage_key = "tenants/org_1/sites/site_1/uploads/upload_1/energy-data.csv"
    store.create_audit_event(
        AuditEvent(
            id="audit_legacy_upload",
            organization_id="org_1",
            actor_user_id="owner_1",
            action=AuditAction.UPLOAD_STORED,
            resource_type="upload",
            resource_id="upload_1",
            metadata={
                "category": "energy",
                "site_id": "site_1",
                "storage_key": legacy_upload_storage_key,
            },
        )
    )

    summaries = list_audit_event_summaries("owner_1", "org_1", store)
    invite_summary = next(event for event in summaries if event.id == "audit_legacy_invite")
    invite_metadata = json.loads(invite_summary.metadata_json)
    report_summary = next(event for event in summaries if event.id == "audit_legacy_report_access")
    report_metadata = json.loads(report_summary.metadata_json)
    upload_summary = next(event for event in summaries if event.id == "audit_legacy_upload")
    upload_metadata = json.loads(upload_summary.metadata_json)
    export = export_audit_events("owner_1", "org_1", store)

    assert invite_metadata == {
        "invite_email_configured": "true",
        "role": "manager",
        "status": "pending",
    }
    assert report_metadata == {
        "artifact_location_configured": "true",
        "report_type": "iso_summary",
    }
    assert upload_metadata == {
        "artifact_location_configured": "true",
        "category": "energy",
        "site_id": "site_1",
    }
    assert "invitee@example.com" not in invite_summary.metadata_json
    assert "invitee@example.com" not in export.content
    assert "invite_email_configured" in export.content
    assert legacy_report_storage_key not in report_summary.metadata_json
    assert legacy_report_storage_key not in export.content
    assert legacy_upload_storage_key not in upload_summary.metadata_json
    assert legacy_upload_storage_key not in export.content
    assert "storage_key" not in report_summary.metadata_json
    assert "storage_key" not in upload_summary.metadata_json
