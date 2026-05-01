from fastapi.routing import APIRoute

from backend.api import (
    API_TITLE,
    OwnerOrganizationCreate,
    create_app,
    health_check,
    list_user_organization_summaries,
    onboard_owner_organization,
    readiness_check,
)
from backend.database import initialize_database
from backend.domain import Role
from backend.store import SaaSStore


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
        "/organizations/onboard-owner",
        "/users/{user_id}/organizations",
    }.issubset(routes)


def test_system_endpoint_handlers_are_side_effect_free():
    assert health_check() == {"status": "ok"}
    assert readiness_check() == {"status": "ready", "service": "rds-saas-api"}


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
