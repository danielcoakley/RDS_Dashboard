from __future__ import annotations

from dataclasses import dataclass

from .domain import Membership, Organization, Role, User
from .store import SaaSStore
from .subscriptions import DEFAULT_TRIAL_PLAN_ID, build_subscription_for_plan


@dataclass(frozen=True)
class OrganizationOnboardingResult:
    user: User
    organization: Organization
    membership: Membership


def create_owner_organization(
    store: SaaSStore,
    user: User,
    organization: Organization,
) -> OrganizationOnboardingResult:
    membership = Membership(
        user_id=user.id,
        organization_id=organization.id,
        role=Role.OWNER,
    )

    with store.conn:
        store.create_user(user)
        store.create_organization(organization)
        store.add_membership(membership)
        if store.get_subscription_plan(DEFAULT_TRIAL_PLAN_ID) is not None:
            store.upsert_organization_subscription(
                build_subscription_for_plan(organization.id, DEFAULT_TRIAL_PLAN_ID)
            )

    return OrganizationOnboardingResult(
        user=user,
        organization=organization,
        membership=membership,
    )
