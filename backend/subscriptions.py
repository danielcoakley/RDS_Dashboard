from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from .access_control import require_permission
from .domain import (
    AuditAction,
    AuditEvent,
    InviteStatus,
    OrganizationSubscription,
    SubscriptionPlan,
    SubscriptionStatus,
)
from .rbac import Action
from .store import SaaSStore


DEFAULT_TRIAL_PLAN_ID = "trial"
TRIAL_DAYS = 30
BILLING_PERIOD_DAYS = 30


@dataclass(frozen=True)
class SubscriptionUsage:
    member_count: int
    active_user_count: int
    pending_invite_count: int
    site_count: int
    runs_this_period: int


@dataclass(frozen=True)
class TenantSubscription:
    organization_id: str
    plan: SubscriptionPlan
    status: SubscriptionStatus
    current_period_start: str | None
    current_period_end: str | None
    trial_ends_at: str | None
    external_customer_id: str | None
    external_subscription_id: str | None
    usage: SubscriptionUsage


class SubscriptionLimitExceeded(ValueError):
    pass


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def build_subscription_for_plan(
    organization_id: str,
    plan_id: str,
    now: datetime | None = None,
    external_customer_id: str | None = None,
    external_subscription_id: str | None = None,
) -> OrganizationSubscription:
    effective_now = now or _now_utc()
    period_end = effective_now + timedelta(days=BILLING_PERIOD_DAYS)
    status = (
        SubscriptionStatus.TRIALING
        if plan_id == DEFAULT_TRIAL_PLAN_ID
        else SubscriptionStatus.ACTIVE
    )
    return OrganizationSubscription(
        organization_id=organization_id,
        plan_id=plan_id,
        status=status,
        current_period_start=effective_now,
        current_period_end=period_end,
        trial_ends_at=(
            effective_now + timedelta(days=TRIAL_DAYS)
            if status == SubscriptionStatus.TRIALING
            else None
        ),
        external_customer_id=external_customer_id,
        external_subscription_id=external_subscription_id,
    )


def list_available_subscription_plans(store: SaaSStore) -> list[SubscriptionPlan]:
    return store.list_subscription_plans(active_only=True)


def ensure_default_subscription(store: SaaSStore, organization_id: str) -> None:
    if store.get_organization_subscription(organization_id) is not None:
        return
    if store.get_organization(organization_id) is None:
        raise ValueError(f"Organization {organization_id} was not found")
    if store.get_subscription_plan(DEFAULT_TRIAL_PLAN_ID) is None:
        raise ValueError(f"Subscription plan {DEFAULT_TRIAL_PLAN_ID} was not found")
    store.upsert_organization_subscription(
        build_subscription_for_plan(organization_id, DEFAULT_TRIAL_PLAN_ID)
    )


def _subscription_usage(
    store: SaaSStore,
    organization_id: str,
    current_period_start: str | None,
) -> SubscriptionUsage:
    period_start = _parse_datetime(current_period_start)
    if period_start is not None:
        period_start = period_start.replace(microsecond=0)
    members = store.list_organization_members(organization_id)
    pending_invites = store.list_organization_invites(organization_id, InviteStatus.PENDING)
    runs = store.list_runs(organization_id)
    return SubscriptionUsage(
        member_count=len(members),
        active_user_count=sum(1 for member in members if bool(member["is_active"])),
        pending_invite_count=len(pending_invites),
        site_count=len(store.list_sites(organization_id)),
        runs_this_period=sum(
            1
            for run in runs
            if period_start is None or _parse_datetime(run["created_at"]) >= period_start
        ),
    )


def _tenant_subscription_from_row(
    store: SaaSStore,
    organization_id: str,
) -> TenantSubscription:
    row = store.get_organization_subscription(organization_id)
    if row is None:
        raise ValueError(f"Subscription for organization {organization_id} was not found")
    return TenantSubscription(
        organization_id=row["organization_id"],
        plan=SubscriptionPlan(
            id=row["plan_id"],
            name=row["plan_name"],
            description=row["plan_description"],
            monthly_price_cents=row["monthly_price_cents"],
            included_users=row["included_users"],
            included_sites=row["included_sites"],
            monthly_run_limit=row["monthly_run_limit"],
            report_retention_days=row["report_retention_days"],
            is_active=bool(row["plan_is_active"]),
        ),
        status=SubscriptionStatus(row["status"]),
        current_period_start=row["current_period_start"],
        current_period_end=row["current_period_end"],
        trial_ends_at=row["trial_ends_at"],
        external_customer_id=row["external_customer_id"],
        external_subscription_id=row["external_subscription_id"],
        usage=_subscription_usage(store, organization_id, row["current_period_start"]),
    )


def get_tenant_subscription(
    store: SaaSStore,
    user_id: str,
    organization_id: str,
) -> TenantSubscription:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.READ,
    )
    with store.conn:
        ensure_default_subscription(store, organization_id)
    return _tenant_subscription_from_row(store, organization_id)


def _tenant_subscription_for_limit_check(
    store: SaaSStore,
    organization_id: str,
) -> TenantSubscription:
    with store.conn:
        ensure_default_subscription(store, organization_id)
    subscription = _tenant_subscription_from_row(store, organization_id)
    if subscription.status not in {SubscriptionStatus.TRIALING, SubscriptionStatus.ACTIVE}:
        raise SubscriptionLimitExceeded(
            f"Subscription status {subscription.status.value} does not allow new tenant activity"
        )
    return subscription


def _require_capacity(
    *,
    resource_name: str,
    current: int,
    additional: int,
    limit: int | None,
) -> None:
    if limit is None:
        return
    if current + additional > limit:
        raise SubscriptionLimitExceeded(
            f"{resource_name} limit exceeded: {current + additional}/{limit}"
        )


def require_member_capacity(
    store: SaaSStore,
    organization_id: str,
    *,
    additional_members: int = 1,
    reserve_pending_invites: bool = True,
) -> None:
    subscription = _tenant_subscription_for_limit_check(store, organization_id)
    current = subscription.usage.active_user_count
    if reserve_pending_invites:
        current += subscription.usage.pending_invite_count
    _require_capacity(
        resource_name="Member seat",
        current=current,
        additional=additional_members,
        limit=subscription.plan.included_users,
    )


def require_site_capacity(
    store: SaaSStore,
    organization_id: str,
    *,
    additional_sites: int = 1,
) -> None:
    subscription = _tenant_subscription_for_limit_check(store, organization_id)
    _require_capacity(
        resource_name="Site",
        current=subscription.usage.site_count,
        additional=additional_sites,
        limit=subscription.plan.included_sites,
    )


def require_run_capacity(
    store: SaaSStore,
    organization_id: str,
    *,
    additional_runs: int = 1,
) -> None:
    subscription = _tenant_subscription_for_limit_check(store, organization_id)
    _require_capacity(
        resource_name="Monthly run",
        current=subscription.usage.runs_this_period,
        additional=additional_runs,
        limit=subscription.plan.monthly_run_limit,
    )


def update_tenant_subscription_plan(
    store: SaaSStore,
    user_id: str,
    organization_id: str,
    plan_id: str,
) -> TenantSubscription:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.MANAGE_BILLING,
    )
    plan = store.get_subscription_plan(plan_id)
    if plan is None or not plan.is_active:
        raise ValueError(f"Subscription plan {plan_id} was not found")

    with store.conn:
        ensure_default_subscription(store, organization_id)
        current = store.get_organization_subscription(organization_id)
        if current is None:
            raise ValueError(f"Subscription for organization {organization_id} was not found")
        subscription = build_subscription_for_plan(
            organization_id=organization_id,
            plan_id=plan.id,
            external_customer_id=current["external_customer_id"],
            external_subscription_id=current["external_subscription_id"],
        )
        store.upsert_organization_subscription(subscription)
        store.create_audit_event(
            AuditEvent(
                id=f"audit_subscription_{uuid4().hex}",
                organization_id=organization_id,
                actor_user_id=user_id,
                action=AuditAction.SUBSCRIPTION_UPDATED,
                resource_type="subscription",
                resource_id=organization_id,
                metadata={
                    "previous_plan_id": current["plan_id"],
                    "plan_id": plan.id,
                    "status": subscription.status.value,
                },
            )
        )

    return _tenant_subscription_from_row(store, organization_id)
