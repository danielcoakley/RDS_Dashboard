from datetime import datetime, timezone
import json

from fastapi.testclient import TestClient
from svix.webhooks import Webhook

from backend.api import create_app
from backend.clerk_webhooks import (
    CLERK_WEBHOOK_SIGNING_SECRET_ENV,
    clerk_webhook_event_from_json,
    clerk_webhook_configuration_status,
    process_clerk_webhook_event,
    verify_clerk_webhook_event,
)
from backend.database import initialize_database
from backend.domain import Role
from backend.operations import CLERK_WEBHOOK_REPLAY_EVENT_TYPE
from backend.store import SaaSStore


WEBHOOK_SECRET = "whsec_test_clerk_webhook_secret_123456"


def signed_headers(payload: str, secret: str = WEBHOOK_SECRET) -> dict[str, str]:
    timestamp = datetime.now(timezone.utc)
    return {
        "svix-id": "msg_test",
        "svix-timestamp": str(int(timestamp.timestamp())),
        "svix-signature": Webhook(secret).sign("msg_test", timestamp, payload),
    }


def user_event(event_type: str = "user.updated") -> dict:
    return {
        "type": event_type,
        "object": "event",
        "data": {
            "id": "user_123",
            "primary_email_address_id": "idn_primary",
            "email_addresses": [
                {"id": "idn_other", "email_address": "other@example.com"},
                {"id": "idn_primary", "email_address": "owner@example.com"},
            ],
            "first_name": "Owner",
            "last_name": "User",
        },
    }


def organization_event(event_type: str = "organization.updated") -> dict:
    return {
        "type": event_type,
        "object": "event",
        "data": {
            "id": "org_123",
            "name": "Example Energy",
            "slug": "example-energy",
        },
    }


def membership_event(event_type: str = "organizationMembership.created", role: str = "org:admin") -> dict:
    return {
        "type": event_type,
        "object": "event",
        "data": {
            "id": "orgmem_123",
            "role": role,
            "organization": {
                "id": "org_123",
                "name": "Example Energy",
                "slug": "example-energy",
            },
            "public_user_data": {
                "user_id": "user_123",
                "identifier": "owner@example.com",
                "first_name": "Owner",
                "last_name": "User",
            },
        },
    }


def test_verify_clerk_webhook_event_accepts_signed_raw_payload():
    payload = json.dumps(user_event(), separators=(",", ":"), sort_keys=True)

    event = verify_clerk_webhook_event(
        payload.encode("utf-8"),
        signed_headers(payload),
        secret=WEBHOOK_SECRET,
    )

    assert event["type"] == "user.updated"
    assert event["data"]["id"] == "user_123"


def test_verify_clerk_webhook_event_rejects_bad_signature():
    payload = json.dumps(user_event(), separators=(",", ":"), sort_keys=True)
    headers = signed_headers(payload)
    headers["svix-signature"] = "v1,not-valid"

    try:
        verify_clerk_webhook_event(payload.encode("utf-8"), headers, secret=WEBHOOK_SECRET)
        assert False, "Expected invalid webhook signature to fail"
    except ValueError as exc:
        assert "signature" in str(exc)


def test_clerk_webhook_configuration_status_rejects_short_secret(monkeypatch):
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, "whsec_test_clerk_webhook_secret")

    configured, detail = clerk_webhook_configuration_status()

    assert configured is False
    assert "at least 32 characters" in detail


def test_process_clerk_user_webhook_upserts_user_and_deletes_memberships_on_delete():
    store = SaaSStore(initialize_database())

    result = process_clerk_webhook_event(user_event(), store)
    user_row = store.get_user("user_123")

    assert result.action == "user_upserted"
    assert user_row is not None
    assert user_row["email"] == "owner@example.com"
    assert user_row["display_name"] == "Owner User"
    assert bool(user_row["is_active"]) is True

    process_clerk_webhook_event(membership_event(), store)
    assert store.list_memberships(user_id="user_123", organization_id="org_123")

    delete_result = process_clerk_webhook_event(
        {"type": "user.deleted", "object": "event", "data": {"id": "user_123"}},
        store,
    )

    user_row = store.get_user("user_123")
    assert delete_result.action == "user_deactivated"
    assert user_row is not None
    assert bool(user_row["is_active"]) is False
    assert store.list_memberships(user_id="user_123", organization_id="org_123") == []


def test_process_clerk_organization_webhook_upserts_org_and_revokes_deleted_org_memberships():
    store = SaaSStore(initialize_database())

    result = process_clerk_webhook_event(organization_event(), store)

    organization_row = store.get_organization("org_123")
    assert result.action == "organization_upserted"
    assert organization_row is not None
    assert organization_row["name"] == "Example Energy"
    assert organization_row["slug"] == "example-energy"

    process_clerk_webhook_event(membership_event(), store)
    assert store.list_memberships(user_id="user_123", organization_id="org_123")

    delete_result = process_clerk_webhook_event(
        {"type": "organization.deleted", "object": "event", "data": {"id": "org_123"}},
        store,
    )

    assert delete_result.action == "organization_memberships_revoked"
    assert store.list_memberships(user_id="user_123", organization_id="org_123") == []


def test_process_clerk_membership_webhook_upserts_and_deletes_membership():
    store = SaaSStore(initialize_database())

    result = process_clerk_webhook_event(membership_event(), store)

    memberships = store.list_memberships(user_id="user_123", organization_id="org_123")
    assert result.action == "membership_upserted"
    assert result.role == "owner"
    assert memberships[0].role == Role.OWNER

    process_clerk_webhook_event(membership_event("organizationMembership.updated", "org:member"), store)
    memberships = store.list_memberships(user_id="user_123", organization_id="org_123")
    assert memberships[0].role == Role.VIEWER

    delete_result = process_clerk_webhook_event(
        membership_event("organizationMembership.deleted", "org:member"),
        store,
    )

    assert delete_result.action == "membership_deleted"
    assert store.list_memberships(user_id="user_123", organization_id="org_123") == []


def test_process_clerk_webhook_ignores_unsubscribed_events():
    store = SaaSStore(initialize_database())

    result = process_clerk_webhook_event(
        {"type": "session.created", "object": "event", "data": {"id": "sess_123"}},
        store,
    )

    assert result.action == "ignored"


def test_clerk_webhook_event_from_json_requires_object_payload():
    assert clerk_webhook_event_from_json(json.dumps(user_event()))["type"] == "user.updated"

    try:
        clerk_webhook_event_from_json("[]")
        assert False, "Expected non-object webhook JSON to fail"
    except ValueError as exc:
        assert "JSON object" in str(exc)


def test_clerk_webhook_route_verifies_signature_and_processes_event(monkeypatch):
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, WEBHOOK_SECRET)
    store = SaaSStore(initialize_database())
    app = create_app(store=store)
    client = TestClient(app)
    payload = json.dumps(membership_event(), separators=(",", ":"), sort_keys=True)

    response = client.post(
        "/webhooks/clerk",
        content=payload,
        headers=signed_headers(payload),
    )

    assert response.status_code == 200
    assert response.json()["action"] == "membership_upserted"
    assert store.list_memberships(user_id="user_123", organization_id="org_123")[0].role == Role.OWNER
    operation_events = store.list_platform_operation_events(
        event_type=CLERK_WEBHOOK_REPLAY_EVENT_TYPE,
        event_status="succeeded",
    )
    assert len(operation_events) == 1
    assert operation_events[0]["detail"] == "action=membership_upserted"
    serialized_event = json.dumps(dict(operation_events[0]))
    assert "org_123" not in serialized_event
    assert "user_123" not in serialized_event
    assert "owner@example.com" not in serialized_event


def test_clerk_webhook_route_rejects_invalid_signature(monkeypatch):
    monkeypatch.setenv(CLERK_WEBHOOK_SIGNING_SECRET_ENV, WEBHOOK_SECRET)
    store = SaaSStore(initialize_database())
    app = create_app(store=store)
    client = TestClient(app)
    payload = json.dumps(user_event(), separators=(",", ":"), sort_keys=True)
    headers = signed_headers(payload)
    headers["svix-signature"] = "v1,not-valid"

    response = client.post("/webhooks/clerk", content=payload, headers=headers)

    assert response.status_code == 400
    assert "signature" in response.json()["detail"]
