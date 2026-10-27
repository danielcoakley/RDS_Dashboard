from backend.database import initialize_database
from backend.domain import AuditAction, AuditEvent, Membership, Organization, Role, User
from backend.store import SaaSStore


def test_store_records_tenant_scoped_audit_events():
    store = SaaSStore(initialize_database())
    store.create_user(User(id="user_1", email="owner@example.com", display_name="Owner"))
    store.create_organization(Organization(id="org_1", name="Example Energy", slug="example-energy"))
    store.add_membership(Membership(user_id="user_1", organization_id="org_1", role=Role.OWNER))

    store.create_audit_event(
        AuditEvent(
            id="audit_1",
            organization_id="org_1",
            actor_user_id="user_1",
            action=AuditAction.RUN_STARTED,
            resource_type="run",
            resource_id="run_1",
            metadata={"source": "test"},
        )
    )

    events = store.list_audit_events("org_1")

    assert len(events) == 1
    assert events[0]["action"] == "run_started"
    assert events[0]["metadata_json"] == '{"source": "test"}'
