from backend.database import initialize_database
from backend.domain import Membership, Meter, Organization, Role, Site, User
from backend.store import SaaSStore


def test_store_creates_organization_with_owner_membership():
    store = SaaSStore(initialize_database())
    user = User(id="user_1", email="owner@example.com", display_name="Owner")
    org = Organization(id="org_1", name="Example Energy", slug="example-energy")

    store.create_user(user)
    store.create_organization(org)
    store.add_membership(Membership(user_id=user.id, organization_id=org.id, role=Role.OWNER))

    organizations = store.list_user_organizations(user.id)
    memberships = store.list_memberships(user_id=user.id)

    assert [organization.id for organization in organizations] == [org.id]
    assert memberships[0].role == Role.OWNER


def test_store_lists_only_user_member_organizations():
    store = SaaSStore(initialize_database())
    store.create_user(User(id="user_1", email="owner@example.com", display_name="Owner"))
    store.create_organization(Organization(id="org_1", name="Example Energy", slug="example-energy"))
    store.create_organization(Organization(id="org_2", name="Other Energy", slug="other-energy"))
    store.add_membership(Membership(user_id="user_1", organization_id="org_2", role=Role.VIEWER))

    organizations = store.list_user_organizations("user_1")

    assert [organization.id for organization in organizations] == ["org_2"]


def test_store_lists_sites_and_meters_by_tenant_scope():
    store = SaaSStore(initialize_database())
    store.create_organization(Organization(id="org_1", name="Example Energy", slug="example-energy"))
    store.create_organization(Organization(id="org_2", name="Other Energy", slug="other-energy"))
    store.create_site(Site(id="site_1", organization_id="org_1", name="Main Site", timezone="Europe/London"))
    store.create_site(Site(id="site_2", organization_id="org_2", name="Other Site", timezone="Europe/London"))
    store.create_meter(
        Meter(
            id="meter_1",
            organization_id="org_1",
            site_id="site_1",
            display_name="Main Electricity",
            commodity="electricity",
            unit="kWh",
            source_column="Main Electricity",
            is_seu=True,
        )
    )
    store.create_meter(
        Meter(
            id="meter_2",
            organization_id="org_2",
            site_id="site_2",
            display_name="Other Electricity",
            commodity="electricity",
            unit="kWh",
            source_column="Other Electricity",
        )
    )

    assert [site.id for site in store.list_sites("org_1")] == ["site_1"]
    assert [meter.id for meter in store.list_meters("org_1")] == ["meter_1"]
    assert [meter.id for meter in store.list_meters("org_2")] == ["meter_2"]


def test_store_memberships_can_feed_access_guards():
    store = SaaSStore(initialize_database())
    store.create_user(User(id="user_1", email="owner@example.com", display_name="Owner"))
    store.create_organization(Organization(id="org_1", name="Example Energy", slug="example-energy"))
    store.add_membership(Membership(user_id="user_1", organization_id="org_1", role=Role.MANAGER))

    memberships = store.list_memberships(user_id="user_1", organization_id="org_1")

    assert len(memberships) == 1
    assert memberships[0].user_id == "user_1"
    assert memberships[0].organization_id == "org_1"
    assert memberships[0].role == Role.MANAGER
