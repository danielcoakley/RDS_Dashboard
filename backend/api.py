from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, Request
import pandas as pd
from pydantic import BaseModel

from .access_control import require_permission
from .auth_context import AuthenticatedUser, request_user_from_header
from .database import initialize_database
from .domain import Organization, User
from .onboarding import create_owner_organization
from .rbac import Action
from .run_orchestration import AnalysisRunRequest, execute_analysis_run
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


class LocalAnalysisRunCreate(BaseModel):
    site_id: str
    upload_id: str
    run_id: str
    report_id: str
    filename: str
    rows: list[dict[str, Any]]
    client_config: dict[str, Any]


class LocalAnalysisRunResponse(BaseModel):
    upload_id: str
    run_id: str
    report_id: str
    run_status: str
    report_storage_key: str
    iso_summary: dict[str, Any]


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


def execute_local_analysis_run(
    user_id: str,
    organization_id: str,
    payload: LocalAnalysisRunCreate,
    store: SaaSStore,
) -> LocalAnalysisRunResponse:
    outcome = execute_analysis_run(
        store,
        AnalysisRunRequest(
            user_id=user_id,
            organization_id=organization_id,
            site_id=payload.site_id,
            upload_id=payload.upload_id,
            run_id=payload.run_id,
            report_id=payload.report_id,
            filename=payload.filename,
            raw_meter_data=pd.DataFrame(payload.rows),
            client_config=payload.client_config,
        ),
    )
    run_row = store.list_runs(organization_id, site_id=payload.site_id)[0]
    return LocalAnalysisRunResponse(
        upload_id=outcome.upload.id,
        run_id=outcome.run.id,
        report_id=outcome.report.id,
        run_status=run_row["status"],
        report_storage_key=outcome.report.storage_key,
        iso_summary=outcome.analytics.iso_summary,
    )


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
        "/me/organizations",
        response_model=list[OrganizationSummary],
        tags=["organizations"],
    )
    def user_organizations(
        request: Request,
        user: AuthenticatedUser = Depends(request_user_from_header),
    ) -> list[OrganizationSummary]:
        return list_user_organization_summaries(user.id, request.app.state.store)

    @app.get(
        "/organizations/{organization_id}/sites",
        response_model=list[SiteSummary],
        tags=["organizations"],
    )
    def organization_sites(
        organization_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_user_from_header),
    ) -> list[SiteSummary]:
        return list_site_summaries(user.id, organization_id, request.app.state.store)

    @app.get(
        "/organizations/{organization_id}/meters",
        response_model=list[MeterSummary],
        tags=["organizations"],
    )
    def organization_meters(
        organization_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_user_from_header),
        site_id: str | None = None,
    ) -> list[MeterSummary]:
        return list_meter_summaries(user.id, organization_id, request.app.state.store, site_id=site_id)

    @app.post(
        "/organizations/{organization_id}/runs/execute-local",
        response_model=LocalAnalysisRunResponse,
        tags=["runs"],
    )
    def organization_execute_local_run(
        organization_id: str,
        payload: LocalAnalysisRunCreate,
        request: Request,
        user: AuthenticatedUser = Depends(request_user_from_header),
    ) -> LocalAnalysisRunResponse:
        return execute_local_analysis_run(user.id, organization_id, payload, request.app.state.store)

    return app


app = create_app()
