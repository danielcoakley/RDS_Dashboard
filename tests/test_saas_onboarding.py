from backend.database import initialize_database
from backend.domain import Organization, Role, User
from backend.onboarding import create_owner_organization
from backend.store import SaaSStore


def test_create_owner_organization_creates_user_org_and_owner_membership():
    store = SaaSStore(initialize_database())
    user = User(id="user_1", email="owner@example.com", display_name="Owner")
    organization = Organization(id="org_1", name="Example Energy", slug="example-energy")

    result = create_owner_organization(store, user, organization)

    memberships = store.list_memberships(user_id=user.id, organization_id=organization.id)
    organizations = store.list_user_organizations(user.id)

    assert result.membership.role == Role.OWNER
    assert [membership.role for membership in memberships] == [Role.OWNER]
    assert [org.id for org in organizations] == [organization.id]


def test_create_owner_organization_rolls_back_if_membership_cannot_be_created():
    store = SaaSStore(initialize_database())
    store.create_user(User(id="inviter", email="inviter@example.com", display_name="Inviter"))
    user = User(id="inviter", email="duplicate@example.com", display_name="Duplicate")
    organization = Organization(id="org_1", name="Example Energy", slug="example-energy")

    try:
        create_owner_organization(store, user, organization)
        assert False, "Expected duplicate user id to roll back onboarding"
    except Exception:
        pass

    organizations = store.list_user_organizations("inviter")
    org_rows = store.conn.execute("SELECT id FROM organizations WHERE id = ?", ("org_1",)).fetchall()

    assert organizations == []
    assert org_rows == []
