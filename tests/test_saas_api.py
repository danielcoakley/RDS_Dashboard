from fastapi.routing import APIRoute

from backend.api import (
    API_TITLE,
    DevSessionCreate,
    LocalAnalysisRunCreate,
    OwnerOrganizationCreate,
    OrganizationInviteAccept,
    OrganizationInviteCreate,
    accept_invite,
    create_dev_session,
    create_app,
    create_invite,
    execute_local_analysis_run,
    health_check,
    list_audit_event_summaries,
    list_organization_invite_summaries,
    list_membership_summaries,
    list_meter_summaries,
    list_report_summaries,
    list_run_summaries,
    list_site_summaries,
    list_upload_summaries,
    list_user_organization_summaries,
    onboard_owner_organization,
    readiness_check,
    revoke_invite,
)
from backend.access_control import AccessDenied
from backend.database import initialize_database
from backend.domain import Membership, Meter, Role, Site, User
from backend.store import SaaSStore
from src.config_loader import load_client_config


def test_api_app_registers_system_routes():
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
        "/auth/dev/session",
        "/organizations/onboard-owner",
        "/organizations/{organization_id}/sites",
        "/organizations/{organization_id}/memberships",
        "/organizations/{organization_id}/invites",
        "/organizations/{organization_id}/invites/{invite_id}/revoke",
        "/organizations/{organization_id}/meters",
        "/organizations/{organization_id}/uploads",
        "/organizations/{organization_id}/runs",
        "/organizations/{organization_id}/reports",
        "/organizations/{organization_id}/audit-events",
        "/organizations/{organization_id}/runs/execute-local",
        "/invites/{invite_id}/accept",
        "/me/organizations",
    }.issubset(routes)


def test_system_endpoint_handlers_are_side_effect_free():
    assert health_check() == {"status": "ok"}
    assert readiness_check() == {"status": "ready", "service": "rds-saas-api"}


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
    store.add_membership(
        Membership(
            user_id="viewer_1",
            organization_id="org_1",
            role=Role.VIEWER,
            invited_by_user_id="owner_1",
        )
    )

    members = list_membership_summaries("owner_1", "org_1", store)

    assert [member.user_id for member in members] == ["owner_1", "viewer_1"]

    try:
        list_membership_summaries("viewer_1", "org_1", store)
        assert False, "Expected viewer membership access to be denied"
    except AccessDenied:
        pass


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
            invite_id="invite_1",
            email="invitee@example.com",
            role="viewer",
        ),
        store,
    )

    invites = list_organization_invite_summaries("owner_1", "org_1", store)
    accepted = accept_invite(
        "invite_1",
        OrganizationInviteAccept(
            user_id="user_2",
            email="invitee@example.com",
            display_name="Invitee",
        ),
        store,
    )

    assert created.id == "invite_1"
    assert [invite.id for invite in invites] == ["invite_1"]
    assert accepted.status == "accepted"
    assert accepted.accepted_by_user_id == "user_2"

    try:
        list_organization_invite_summaries("owner_2", "org_1", store)
        assert False, "Expected cross-tenant invite listing to be denied"
    except AccessDenied:
        pass


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

    create_invite(
        "owner_1",
        "org_1",
        OrganizationInviteCreate(
            invite_id="invite_1",
            email="invitee@example.com",
            role="viewer",
        ),
        store,
    )

    revoked = revoke_invite("owner_1", "org_1", "invite_1", store)

    assert revoked.status == "revoked"

    try:
        revoke_invite("owner_2", "org_1", "invite_1", store)
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

    response = execute_local_analysis_run("user_1", "org_1", payload, store)

    assert response.run_id == "run_1"
    assert response.run_status == "succeeded"
    assert response.report_storage_key == "tenants/org_1/sites/site_1/runs/run_1/reports/iso-summary.json"
    assert response.iso_summary["total_records"] == 4


def test_report_summaries_are_tenant_guarded():
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

    reports = list_report_summaries("user_1", "org_1", store, run_id="run_1")

    assert [report.id for report in reports] == ["report_1"]
    assert reports[0].is_published


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
