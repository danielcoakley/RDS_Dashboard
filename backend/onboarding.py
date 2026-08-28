from __future__ import annotations

from dataclasses import dataclass

from .domain import Membership, Organization, Role, User
from .store import SaaSStore


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

    return OrganizationOnboardingResult(
        user=user,
        organization=organization,
        membership=membership,
    )
