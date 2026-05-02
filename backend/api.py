from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
import pandas as pd
from pydantic import BaseModel

from .access_control import require_permission
from .auth_context import AuthenticatedUser, dev_token_from_claims, request_authenticated_user
from .database import initialize_database
from .domain import Organization, Role, User
from .invitations import accept_organization_invite, create_organization_invite
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


class DevSessionCreate(BaseModel):
    user_id: str
    organization_id: str | None = None


class DevSessionResponse(BaseModel):
    user_id: str
    organization_id: str | None
    role: str | None
    auth_token: str


class OrganizationSummary(BaseModel):
    id: str
    name: str
    slug: str


class SiteSummary(BaseModel):
    id: str
    organization_id: str
    name: str
    timezone: str


class MembershipSummary(BaseModel):
    user_id: str
    organization_id: str
    role: str
    invited_by_user_id: str | None
    email: str
    display_name: str
    is_active: bool


class OrganizationInviteCreate(BaseModel):
    invite_id: str
    email: str
    role: Role


class OrganizationInviteAccept(BaseModel):
    user_id: str
    email: str
    display_name: str


class OrganizationInviteSummary(BaseModel):
    id: str
    organization_id: str
    email: str
    role: str
    invited_by_user_id: str
    status: str
    accepted_by_user_id: str | None
    accepted_at: str | None


class MeterSummary(BaseModel):
    id: str
    organization_id: str
    site_id: str
    display_name: str
    commodity: str
    unit: str
    source_column: str
    is_seu: bool


class UploadSummary(BaseModel):
    id: str
    organization_id: str
    site_id: str
    uploaded_by_user_id: str
    category: str
    storage_key: str
    checksum: str
    status: str


class RunSummary(BaseModel):
    id: str
    organization_id: str
    site_id: str
    requested_by_user_id: str
    status: str
    error_message: str | None
    completed_at: str | None


class ReportSummary(BaseModel):
    id: str
    organization_id: str
    run_id: str
    report_type: str
    storage_key: str
    is_published: bool


class AuditEventSummary(BaseModel):
    id: str
    organization_id: str
    actor_user_id: str
    action: str
    resource_type: str
    resource_id: str
    metadata_json: str


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


def create_dev_session(
    payload: DevSessionCreate,
    store: SaaSStore,
) -> DevSessionResponse:
    user_row = store.get_user(payload.user_id)
    if user_row is None:
        raise ValueError(f"User {payload.user_id} was not found")
    if not bool(user_row["is_active"]):
        raise ValueError(f"User {payload.user_id} is inactive")

    role: str | None = None
    claims: dict[str, str] = {"sub": payload.user_id}
    if payload.organization_id is not None:
        memberships = store.list_memberships(
            user_id=payload.user_id,
            organization_id=payload.organization_id,
        )
        if not memberships:
            raise ValueError(
                f"User {payload.user_id} has no membership in organization {payload.organization_id}"
            )
        role = memberships[0].role.value
        claims["org_id"] = payload.organization_id
        claims["org_role"] = role

    return DevSessionResponse(
        user_id=payload.user_id,
        organization_id=payload.organization_id,
        role=role,
        auth_token=dev_token_from_claims(claims),
    )


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


def list_membership_summaries(
    user_id: str,
    organization_id: str,
    store: SaaSStore,
) -> list[MembershipSummary]:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.MANAGE_MEMBERS,
    )
    return [
        MembershipSummary(
            user_id=row["user_id"],
            organization_id=row["organization_id"],
            role=row["role"],
            invited_by_user_id=row["invited_by_user_id"],
            email=row["email"],
            display_name=row["display_name"],
            is_active=bool(row["is_active"]),
        )
        for row in store.list_organization_members(organization_id)
    ]


def list_organization_invite_summaries(
    user_id: str,
    organization_id: str,
    store: SaaSStore,
) -> list[OrganizationInviteSummary]:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.MANAGE_MEMBERS,
    )
    return [
        OrganizationInviteSummary(
            id=row["id"],
            organization_id=row["organization_id"],
            email=row["email"],
            role=row["role"],
            invited_by_user_id=row["invited_by_user_id"],
            status=row["status"],
            accepted_by_user_id=row["accepted_by_user_id"],
            accepted_at=row["accepted_at"],
        )
        for row in store.list_organization_invites(organization_id)
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


def list_upload_summaries(
    user_id: str,
    organization_id: str,
    store: SaaSStore,
    site_id: str | None = None,
) -> list[UploadSummary]:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.READ,
    )
    return [
        UploadSummary(
            id=upload["id"],
            organization_id=upload["organization_id"],
            site_id=upload["site_id"],
            uploaded_by_user_id=upload["uploaded_by_user_id"],
            category=upload["category"],
            storage_key=upload["storage_key"],
            checksum=upload["checksum"],
            status=upload["status"],
        )
        for upload in store.list_uploads(organization_id=organization_id, site_id=site_id)
    ]


def list_run_summaries(
    user_id: str,
    organization_id: str,
    store: SaaSStore,
    site_id: str | None = None,
) -> list[RunSummary]:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.READ,
    )
    return [
        RunSummary(
            id=run["id"],
            organization_id=run["organization_id"],
            site_id=run["site_id"],
            requested_by_user_id=run["requested_by_user_id"],
            status=run["status"],
            error_message=run["error_message"],
            completed_at=run["completed_at"],
        )
        for run in store.list_runs(organization_id=organization_id, site_id=site_id)
    ]


def list_report_summaries(
    user_id: str,
    organization_id: str,
    store: SaaSStore,
    run_id: str | None = None,
) -> list[ReportSummary]:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.READ,
    )
    return [
        ReportSummary(
            id=report["id"],
            organization_id=report["organization_id"],
            run_id=report["run_id"],
            report_type=report["report_type"],
            storage_key=report["storage_key"],
            is_published=bool(report["is_published"]),
        )
        for report in store.list_reports(organization_id=organization_id, run_id=run_id)
    ]


def list_audit_event_summaries(
    user_id: str,
    organization_id: str,
    store: SaaSStore,
) -> list[AuditEventSummary]:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.VIEW_AUDIT,
    )
    return [
        AuditEventSummary(
            id=event["id"],
            organization_id=event["organization_id"],
            actor_user_id=event["actor_user_id"],
            action=event["action"],
            resource_type=event["resource_type"],
            resource_id=event["resource_id"],
            metadata_json=event["metadata_json"],
        )
        for event in store.list_audit_events(organization_id)
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


def create_invite(
    user_id: str,
    organization_id: str,
    payload: OrganizationInviteCreate,
    store: SaaSStore,
) -> OrganizationInviteSummary:
    invite = create_organization_invite(
        store,
        actor_user_id=user_id,
        organization_id=organization_id,
        invite_id=payload.invite_id,
        email=payload.email,
        role=payload.role,
    )
    return OrganizationInviteSummary(
        id=invite.id,
        organization_id=invite.organization_id,
        email=invite.email,
        role=invite.role.value,
        invited_by_user_id=invite.invited_by_user_id,
        status=invite.status.value,
        accepted_by_user_id=invite.accepted_by_user_id,
        accepted_at=invite.accepted_at.isoformat() if invite.accepted_at else None,
    )


def accept_invite(
    invite_id: str,
    payload: OrganizationInviteAccept,
    store: SaaSStore,
) -> OrganizationInviteSummary:
    result = accept_organization_invite(
        store,
        invite_id=invite_id,
        user=User(
            id=payload.user_id,
            email=payload.email,
            display_name=payload.display_name,
        ),
    )
    return OrganizationInviteSummary(
        id=result.invite.id,
        organization_id=result.invite.organization_id,
        email=result.invite.email,
        role=result.invite.role.value,
        invited_by_user_id=result.invite.invited_by_user_id,
        status=result.invite.status.value,
        accepted_by_user_id=result.invite.accepted_by_user_id,
        accepted_at=result.invite.accepted_at.isoformat() if result.invite.accepted_at else None,
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
        "/auth/dev/session",
        response_model=DevSessionResponse,
        tags=["auth"],
    )
    def auth_dev_session(payload: DevSessionCreate, request: Request) -> DevSessionResponse:
        try:
            return create_dev_session(payload, request.app.state.store)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc

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
        user: AuthenticatedUser = Depends(request_authenticated_user),
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
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> list[SiteSummary]:
        return list_site_summaries(user.id, organization_id, request.app.state.store)

    @app.get(
        "/organizations/{organization_id}/memberships",
        response_model=list[MembershipSummary],
        tags=["organizations"],
    )
    def organization_memberships(
        organization_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> list[MembershipSummary]:
        return list_membership_summaries(user.id, organization_id, request.app.state.store)

    @app.get(
        "/organizations/{organization_id}/invites",
        response_model=list[OrganizationInviteSummary],
        tags=["organizations"],
    )
    def organization_invites(
        organization_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> list[OrganizationInviteSummary]:
        return list_organization_invite_summaries(user.id, organization_id, request.app.state.store)

    @app.post(
        "/organizations/{organization_id}/invites",
        response_model=OrganizationInviteSummary,
        tags=["organizations"],
    )
    def organization_create_invite(
        organization_id: str,
        payload: OrganizationInviteCreate,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> OrganizationInviteSummary:
        return create_invite(user.id, organization_id, payload, request.app.state.store)

    @app.get(
        "/organizations/{organization_id}/meters",
        response_model=list[MeterSummary],
        tags=["organizations"],
    )
    def organization_meters(
        organization_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
        site_id: str | None = None,
    ) -> list[MeterSummary]:
        return list_meter_summaries(user.id, organization_id, request.app.state.store, site_id=site_id)

    @app.get(
        "/organizations/{organization_id}/uploads",
        response_model=list[UploadSummary],
        tags=["uploads"],
    )
    def organization_uploads(
        organization_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
        site_id: str | None = None,
    ) -> list[UploadSummary]:
        return list_upload_summaries(user.id, organization_id, request.app.state.store, site_id=site_id)

    @app.get(
        "/organizations/{organization_id}/runs",
        response_model=list[RunSummary],
        tags=["runs"],
    )
    def organization_runs(
        organization_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
        site_id: str | None = None,
    ) -> list[RunSummary]:
        return list_run_summaries(user.id, organization_id, request.app.state.store, site_id=site_id)

    @app.get(
        "/organizations/{organization_id}/reports",
        response_model=list[ReportSummary],
        tags=["reports"],
    )
    def organization_reports(
        organization_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
        run_id: str | None = None,
    ) -> list[ReportSummary]:
        return list_report_summaries(user.id, organization_id, request.app.state.store, run_id=run_id)

    @app.get(
        "/organizations/{organization_id}/audit-events",
        response_model=list[AuditEventSummary],
        tags=["audit"],
    )
    def organization_audit_events(
        organization_id: str,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> list[AuditEventSummary]:
        return list_audit_event_summaries(user.id, organization_id, request.app.state.store)

    @app.post(
        "/organizations/{organization_id}/runs/execute-local",
        response_model=LocalAnalysisRunResponse,
        tags=["runs"],
    )
    def organization_execute_local_run(
        organization_id: str,
        payload: LocalAnalysisRunCreate,
        request: Request,
        user: AuthenticatedUser = Depends(request_authenticated_user),
    ) -> LocalAnalysisRunResponse:
        return execute_local_analysis_run(user.id, organization_id, payload, request.app.state.store)

    @app.post(
        "/invites/{invite_id}/accept",
        response_model=OrganizationInviteSummary,
        tags=["organizations"],
    )
    def organization_accept_invite(
        invite_id: str,
        payload: OrganizationInviteAccept,
        request: Request,
    ) -> OrganizationInviteSummary:
        return accept_invite(invite_id, payload, request.app.state.store)

    return app


app = create_app()
