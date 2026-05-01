from fastapi.routing import APIRoute

from backend.api import (
    API_TITLE,
    OwnerOrganizationCreate,
    create_app,
    health_check,
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
    assert {"/health", "/ready", "/organizations/onboard-owner"}.issubset(routes)


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
