from __future__ import annotations

from fastapi import FastAPI, Request
from pydantic import BaseModel

from .access_control import require_permission
from .database import initialize_database
from .domain import Organization, User
from .onboarding import create_owner_organization
from .rbac import Action
from .store import SaaSStore


API_TITLE = "RDS Energy Analytics SaaS API"
API_VERSION = "0.1.0"


class OwnerOrganizationCreate(BaseModel):
    user_id: str
    email: str
    display_name: str
    organization_id: str
    organization_name: str
    organization_slug: str


class OwnerOrganizationResponse(BaseModel):
    user_id: str
    organization_id: str
    role: str


class OrganizationSummary(BaseModel):
    id: str
    name: str
    slug: str


class SiteSummary(BaseModel):
    id: str
    organization_id: str
    name: str
    timezone: str


class MeterSummary(BaseModel):
    id: str
    organization_id: str
    site_id: str
    display_name: str
    commodity: str
    unit: str
    source_column: str
    is_seu: bool


def health_check() -> dict[str, str]:
    return {"status": "ok"}


def readiness_check() -> dict[str, str]:
    return {"status": "ready", "service": "rds-saas-api"}


def onboard_owner_organization(
    payload: OwnerOrganizationCreate,
    store: SaaSStore,
) -> OwnerOrganizationResponse:
    result = create_owner_organization(
        store,
        user=User(
            id=payload.user_id,
            email=payload.email,
            display_name=payload.display_name,
        ),
        organization=Organization(
            id=payload.organization_id,
            name=payload.organization_name,
            slug=payload.organization_slug,
        ),
    )
    return OwnerOrganizationResponse(
        user_id=result.user.id,
        organization_id=result.organization.id,
        role=result.membership.role.value,
    )


def list_user_organization_summaries(
    user_id: str,
    store: SaaSStore,
) -> list[OrganizationSummary]:
    return [
        OrganizationSummary(id=organization.id, name=organization.name, slug=organization.slug)
        for organization in store.list_user_organizations(user_id)
    ]


def list_site_summaries(
    user_id: str,
    organization_id: str,
    store: SaaSStore,
) -> list[SiteSummary]:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.READ,
    )
    return [
        SiteSummary(
            id=site.id,
            organization_id=site.organization_id,
            name=site.name,
            timezone=site.timezone,
        )
        for site in store.list_sites(organization_id)
    ]


def list_meter_summaries(
    user_id: str,
    organization_id: str,
    store: SaaSStore,
    site_id: str | None = None,
) -> list[MeterSummary]:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.READ,
    )
    return [
        MeterSummary(
            id=meter.id,
            organization_id=meter.organization_id,
            site_id=meter.site_id,
            display_name=meter.display_name,
            commodity=meter.commodity,
            unit=meter.unit,
            source_column=meter.source_column,
            is_seu=meter.is_seu,
        )
        for meter in store.list_meters(organization_id=organization_id, site_id=site_id)
    ]


def create_app(store: SaaSStore | None = None) -> FastAPI:
    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description="Backend API boundary for the SaaS energy analytics platform.",
    )
    app.state.store = store or SaaSStore(initialize_database())
    app.add_api_route("/health", health_check, methods=["GET"], tags=["system"])
    app.add_api_route("/ready", readiness_check, methods=["GET"], tags=["system"])

    @app.post(
        "/organizations/onboard-owner",
        response_model=OwnerOrganizationResponse,
        tags=["organizations"],
    )
    def onboard_owner(payload: OwnerOrganizationCreate, request: Request) -> OwnerOrganizationResponse:
        return onboard_owner_organization(payload, request.app.state.store)

    @app.get(
        "/users/{user_id}/organizations",
        response_model=list[OrganizationSummary],
        tags=["organizations"],
    )
    def user_organizations(user_id: str, request: Request) -> list[OrganizationSummary]:
        return list_user_organization_summaries(user_id, request.app.state.store)

    @app.get(
        "/organizations/{organization_id}/sites",
        response_model=list[SiteSummary],
        tags=["organizations"],
    )
    def organization_sites(
        organization_id: str,
        user_id: str,
        request: Request,
    ) -> list[SiteSummary]:
        return list_site_summaries(user_id, organization_id, request.app.state.store)

    @app.get(
        "/organizations/{organization_id}/meters",
        response_model=list[MeterSummary],
        tags=["organizations"],
    )
    def organization_meters(
        organization_id: str,
        user_id: str,
        request: Request,
        site_id: str | None = None,
    ) -> list[MeterSummary]:
        return list_meter_summaries(user_id, organization_id, request.app.state.store, site_id=site_id)

    return app


app = create_app()
