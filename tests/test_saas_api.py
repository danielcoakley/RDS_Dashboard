from fastapi.routing import APIRoute

from backend.api import API_TITLE, health_check, readiness_check, create_app


def test_api_app_registers_system_routes():
    app = create_app()
    routes = {
        route.path
        for route in app.routes
        if isinstance(route, APIRoute)
    }

    assert app.title == API_TITLE
    assert {"/health", "/ready"}.issubset(routes)


def test_system_endpoint_handlers_are_side_effect_free():
    assert health_check() == {"status": "ok"}
    assert readiness_check() == {"status": "ready", "service": "rds-saas-api"}
