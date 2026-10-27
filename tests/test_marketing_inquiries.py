from backend.api import (
    MarketingInquiryCreate,
    create_app,
    create_marketing_inquiry_summary,
)
from backend.database import initialize_database
from backend.domain import MarketingInquiryStatus
from backend.marketing_handoff import (
    MARKETING_INQUIRY_CRM_WEBHOOK_SECRET_ENV,
    MARKETING_INQUIRY_CRM_WEBHOOK_TIMEOUT_SECONDS_ENV,
    MARKETING_INQUIRY_CRM_WEBHOOK_URL_ENV,
)
from backend.marketing_rate_limit import (
    MARKETING_INQUIRY_RATE_LIMIT_ENABLED_ENV,
    MARKETING_INQUIRY_RATE_LIMIT_MAX_ENV,
    MARKETING_INQUIRY_RATE_LIMIT_SECRET_ENV,
    MARKETING_INQUIRY_RATE_LIMIT_WINDOW_MINUTES_ENV,
)
from backend.notifications import InMemoryEmailSender
from backend.runtime_config import RUNTIME_ENV_ENV
from backend.store import SaaSStore
from backend.support import SUPPORT_ACCESS_TOKEN_ENV
from fastapi.testclient import TestClient
import pytest


class FailingInquirySender:
    def send(self, message):
        raise RuntimeError("simulated handoff failure")


class RecordingInquiryCrmSender:
    def __init__(self):
        self.sent_inquiries = []

    def send(self, inquiry):
        self.sent_inquiries.append(inquiry)


class FailingInquiryCrmSender:
    def send(self, inquiry):
        raise RuntimeError("simulated CRM handoff failure")


@pytest.fixture(autouse=True)
def clear_marketing_inquiry_crm_env(monkeypatch):
    monkeypatch.delenv(MARKETING_INQUIRY_CRM_WEBHOOK_URL_ENV, raising=False)
    monkeypatch.delenv(MARKETING_INQUIRY_CRM_WEBHOOK_SECRET_ENV, raising=False)
    monkeypatch.delenv(MARKETING_INQUIRY_CRM_WEBHOOK_TIMEOUT_SECONDS_ENV, raising=False)
    monkeypatch.delenv(MARKETING_INQUIRY_RATE_LIMIT_ENABLED_ENV, raising=False)
    monkeypatch.delenv(MARKETING_INQUIRY_RATE_LIMIT_MAX_ENV, raising=False)
    monkeypatch.delenv(MARKETING_INQUIRY_RATE_LIMIT_SECRET_ENV, raising=False)
    monkeypatch.delenv(MARKETING_INQUIRY_RATE_LIMIT_WINDOW_MINUTES_ENV, raising=False)


def test_create_marketing_inquiry_summary_normalizes_public_landing_request():
    store = SaaSStore(initialize_database())

    summary = create_marketing_inquiry_summary(
        MarketingInquiryCreate(
            name="  Alex Manager  ",
            email=" ALEX@example.COM ",
            organization_name=" Example Estates ",
            role=" Energy Lead ",
            intended_use="consultant-delivery",
            message="  Need a pilot for two client sites.  ",
            source_page="/?utm_source=landing#contact",
        ),
        store,
    )

    row = store.get_marketing_inquiry(summary.id)

    assert summary.status == MarketingInquiryStatus.NEW
    assert row is not None
    assert row["email"] == "alex@example.com"
    assert row["name"] == "Alex Manager"
    assert row["organization_name"] == "Example Estates"
    assert row["role"] == "Energy Lead"
    assert row["intended_use"] == "consultant_delivery"
    assert row["message"] == "Need a pilot for two client sites."
    assert row["source_page"] == "/?utm_source=landing#contact"
    assert row["status"] == "new"
    assert [inquiry["id"] for inquiry in store.list_marketing_inquiries(MarketingInquiryStatus.NEW)] == [
        summary.id
    ]


def test_marketing_inquiry_public_route_requires_no_tenant_session_and_does_not_list():
    store = SaaSStore(initialize_database())
    sender = InMemoryEmailSender()
    crm_sender = RecordingInquiryCrmSender()
    client = TestClient(
        create_app(
            store=store,
            email_sender=sender,
            marketing_inquiry_recipient_email="sales@example.com",
            marketing_inquiry_crm_sender=crm_sender,
        )
    )

    response = client.post(
        "/marketing/inquiries",
        json={
            "name": "Jordan Auditor",
            "email": "jordan@example.com",
            "organization_name": "Audit Partners",
            "role": "Consultant",
            "intended_use": "audit_readiness",
            "message": "We need ISO 50001 reporting evidence for a new client.",
            "source_page": "/",
        },
    )
    list_response = client.get("/marketing/inquiries")

    assert response.status_code == 200
    assert response.json()["status"] == "new"
    assert "id" in response.json()
    assert len(store.list_marketing_inquiries()) == 1
    assert list_response.status_code == 405
    assert len(sender.sent_messages) == 1
    message = sender.sent_messages[0]
    assert message.to_email == "sales@example.com"
    assert message.organization_id == "platform"
    assert message.recipient_user_id == "marketing_inquiries"
    assert response.json()["id"] in message.body
    assert "Jordan Auditor" in message.body
    assert "jordan@example.com" in message.body
    assert "audit readiness" in message.body
    assert "ISO 50001 reporting evidence" in message.body
    assert len(crm_sender.sent_inquiries) == 1
    assert crm_sender.sent_inquiries[0].id == response.json()["id"]
    assert crm_sender.sent_inquiries[0].email == "jordan@example.com"
    assert set(response.json()) == {"id", "status", "created_at"}


def test_marketing_inquiry_handoff_failure_does_not_drop_inquiry():
    store = SaaSStore(initialize_database())
    client = TestClient(
        create_app(
            store=store,
            email_sender=FailingInquirySender(),
            marketing_inquiry_recipient_email="sales@example.com",
        )
    )

    response = client.post(
        "/marketing/inquiries",
        json={
            "name": "Jordan Auditor",
            "email": "jordan@example.com",
            "intended_use": "audit_readiness",
            "message": "We need ISO 50001 reporting evidence for a new client.",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "new"
    assert len(store.list_marketing_inquiries()) == 1


def test_marketing_inquiry_crm_handoff_failure_does_not_drop_inquiry_or_email():
    store = SaaSStore(initialize_database())
    sender = InMemoryEmailSender()
    client = TestClient(
        create_app(
            store=store,
            email_sender=sender,
            marketing_inquiry_recipient_email="sales@example.com",
            marketing_inquiry_crm_sender=FailingInquiryCrmSender(),
        )
    )

    response = client.post(
        "/marketing/inquiries",
        json={
            "name": "Jordan Auditor",
            "email": "jordan@example.com",
            "intended_use": "audit_readiness",
            "message": "We need ISO 50001 reporting evidence for a new client.",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "new"
    assert len(store.list_marketing_inquiries()) == 1
    assert len(sender.sent_messages) == 1


def test_marketing_inquiry_rejects_recent_duplicate_email_without_second_handoff():
    store = SaaSStore(initialize_database())
    sender = InMemoryEmailSender()
    crm_sender = RecordingInquiryCrmSender()
    client = TestClient(
        create_app(
            store=store,
            email_sender=sender,
            marketing_inquiry_recipient_email="sales@example.com",
            marketing_inquiry_crm_sender=crm_sender,
        )
    )
    payload = {
        "name": "Jordan Auditor",
        "email": "jordan@example.com",
        "intended_use": "audit_readiness",
        "message": "Need a pilot for one client.",
    }

    first_response = client.post("/marketing/inquiries", json=payload)
    duplicate_response = client.post(
        "/marketing/inquiries",
        json={**payload, "email": " JORDAN@example.COM "},
    )

    assert first_response.status_code == 200
    assert duplicate_response.status_code == 400
    assert "submitted recently" in duplicate_response.json()["detail"]
    assert len(store.list_marketing_inquiries()) == 1
    assert len(sender.sent_messages) == 1
    assert len(crm_sender.sent_inquiries) == 1


def test_marketing_inquiry_honeypot_rejects_bot_submission_without_handoff():
    store = SaaSStore(initialize_database())
    sender = InMemoryEmailSender()
    crm_sender = RecordingInquiryCrmSender()
    client = TestClient(
        create_app(
            store=store,
            email_sender=sender,
            marketing_inquiry_recipient_email="sales@example.com",
            marketing_inquiry_crm_sender=crm_sender,
        )
    )

    response = client.post(
        "/marketing/inquiries",
        json={
            "name": "Jordan Auditor",
            "email": "jordan@example.com",
            "intended_use": "audit_readiness",
            "message": "Need a pilot.",
            "website": "https://spam.example",
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Marketing inquiry could not be accepted"
    assert store.list_marketing_inquiries() == []
    assert sender.sent_messages == []
    assert crm_sender.sent_inquiries == []


def test_marketing_inquiry_public_route_rate_limits_source_without_storing_raw_source(
    monkeypatch,
):
    monkeypatch.setenv(MARKETING_INQUIRY_RATE_LIMIT_MAX_ENV, "2")
    monkeypatch.setenv(MARKETING_INQUIRY_RATE_LIMIT_WINDOW_MINUTES_ENV, "60")
    monkeypatch.setenv(MARKETING_INQUIRY_RATE_LIMIT_SECRET_ENV, "rate-secret")
    store = SaaSStore(initialize_database())
    sender = InMemoryEmailSender()
    crm_sender = RecordingInquiryCrmSender()
    client = TestClient(
        create_app(
            store=store,
            email_sender=sender,
            marketing_inquiry_recipient_email="sales@example.com",
            marketing_inquiry_crm_sender=crm_sender,
        )
    )
    headers = {"User-Agent": "rate-limit-test-agent"}

    first = client.post(
        "/marketing/inquiries",
        json={
            "name": "Jordan Auditor",
            "email": "jordan1@example.com",
            "intended_use": "audit_readiness",
            "message": "Need a pilot for one client.",
        },
        headers=headers,
    )
    second = client.post(
        "/marketing/inquiries",
        json={
            "name": "Alex Manager",
            "email": "alex2@example.com",
            "intended_use": "estate_portfolio",
            "message": "Need portfolio reporting for three sites.",
        },
        headers=headers,
    )
    limited = client.post(
        "/marketing/inquiries",
        json={
            "name": "Priya Lead",
            "email": "priya3@example.com",
            "intended_use": "measurement_verification",
            "message": "Need M&V evidence for a savings project.",
        },
        headers=headers,
    )

    events = store.conn.execute(
        """
        SELECT source_key_hash, inquiry_id, outcome
        FROM marketing_inquiry_submission_events
        ORDER BY created_at, id
        """
    ).fetchall()

    assert first.status_code == 200
    assert second.status_code == 200
    assert limited.status_code == 429
    assert limited.json()["detail"] == "Marketing inquiry could not be accepted right now"
    assert len(store.list_marketing_inquiries()) == 2
    assert len(sender.sent_messages) == 2
    assert len(crm_sender.sent_inquiries) == 2
    assert len(events) == 3
    assert events[0]["source_key_hash"] == events[1]["source_key_hash"]
    assert events[1]["source_key_hash"] == events[2]["source_key_hash"]
    assert events[0]["inquiry_id"] == first.json()["id"]
    assert events[1]["inquiry_id"] == second.json()["id"]
    assert events[2]["inquiry_id"] is None
    assert [event["outcome"] for event in events] == [
        "accepted",
        "accepted",
        "rate_limited",
    ]
    assert limited.headers["retry-after"] == "3600"
    assert "rate-limit-test-agent" not in events[0]["source_key_hash"]
    assert "testclient" not in events[0]["source_key_hash"]


def test_support_marketing_inquiries_require_support_token(monkeypatch):
    monkeypatch.delenv(SUPPORT_ACCESS_TOKEN_ENV, raising=False)
    store = SaaSStore(initialize_database())
    client = TestClient(create_app(store=store))

    missing_config = client.get("/support/marketing-inquiries")
    missing_summary_config = client.get("/support/marketing-inquiries/summary")

    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, "support-secret")
    missing_auth = client.get("/support/marketing-inquiries")
    missing_summary_auth = client.get("/support/marketing-inquiries/summary")
    bad_auth = client.get(
        "/support/marketing-inquiries",
        headers={"Authorization": "Bearer wrong-secret"},
    )
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, "short-support-token")
    weak_config = client.get(
        "/support/marketing-inquiries",
        headers={"Authorization": "Bearer short-support-token"},
    )

    assert missing_config.status_code == 503
    assert missing_summary_config.status_code == 503
    assert SUPPORT_ACCESS_TOKEN_ENV in missing_config.json()["detail"]
    assert missing_auth.status_code == 401
    assert missing_summary_auth.status_code == 401
    assert bad_auth.status_code == 401
    assert weak_config.status_code == 503
    assert "at least 32 characters" in weak_config.json()["detail"]
    assert missing_auth.headers["www-authenticate"] == "Bearer"


def test_support_marketing_inquiries_list_filter_and_update_status(monkeypatch):
    monkeypatch.setenv(SUPPORT_ACCESS_TOKEN_ENV, "support-secret")
    store = SaaSStore(initialize_database())
    client = TestClient(create_app(store=store))

    first = create_marketing_inquiry_summary(
        MarketingInquiryCreate(
            name="Jordan Auditor",
            email="jordan@example.com",
            intended_use="audit_readiness",
            message="Need a pilot for one client.",
            source_page="/",
        ),
        store,
    )
    second = create_marketing_inquiry_summary(
        MarketingInquiryCreate(
            name="Alex Manager",
            email="alex@example.com",
            intended_use="estate_portfolio",
            message="Need portfolio reporting for three sites.",
            source_page="/?utm_source=partner",
        ),
        store,
    )

    headers = {
        "Authorization": "Bearer support-secret",
        "X-Support-Actor": "Taylor Support",
    }
    list_response = client.get(
        "/support/marketing-inquiries?status=new&limit=1",
        headers=headers,
    )
    update_response = client.patch(
        f"/support/marketing-inquiries/{first.id}",
        json={
            "status": "qualified",
            "assigned_to": "Sales.Owner@Example.com",
            "follow_up_notes": "Booked discovery call for the estates team.",
        },
        headers=headers,
    )
    qualified_response = client.get(
        "/support/marketing-inquiries?status=qualified",
        headers=headers,
    )
    assigned_response = client.get(
        "/support/marketing-inquiries",
        params={"assigned_to": "SALES.OWNER@EXAMPLE.COM"},
        headers=headers,
    )
    unassigned_response = client.get(
        "/support/marketing-inquiries",
        params={"unassigned": "true"},
        headers=headers,
    )
    contradictory_filter_response = client.get(
        "/support/marketing-inquiries",
        params={
            "assigned_to": "sales.owner@example.com",
            "unassigned": "true",
        },
        headers=headers,
    )
    summary_response = client.get("/support/marketing-inquiries/summary", headers=headers)
    clear_notes_response = client.patch(
        f"/support/marketing-inquiries/{first.id}",
        json={"follow_up_notes": None},
        headers=headers,
    )
    invalid_assignee_response = client.patch(
        f"/support/marketing-inquiries/{first.id}",
        json={"assigned_to": "bad\nassignee"},
        headers=headers,
    )
    not_found_response = client.patch(
        "/support/marketing-inquiries/missing",
        json={"status": "closed"},
        headers=headers,
    )
    activity_response = client.get(
        f"/support/marketing-inquiries/{first.id}/activity",
        headers=headers,
    )
    missing_activity_response = client.get(
        "/support/marketing-inquiries/missing/activity",
        headers=headers,
    )

    assert list_response.status_code == 200
    assert len(list_response.json()) == 1
    assert set(list_response.json()[0]) == {
        "id",
        "email",
        "name",
        "organization_name",
        "role",
        "intended_use",
        "message",
        "source_page",
        "status",
        "created_at",
        "assigned_to",
        "follow_up_notes",
        "updated_at",
    }
    assert list_response.json()[0]["status"] == "new"
    assert update_response.status_code == 200
    assert update_response.json()["id"] == first.id
    assert update_response.json()["status"] == "qualified"
    assert update_response.json()["assigned_to"] == "sales.owner@example.com"
    assert update_response.json()["follow_up_notes"] == "Booked discovery call for the estates team."
    assert update_response.json()["updated_at"] is not None
    assert qualified_response.status_code == 200
    assert [inquiry["id"] for inquiry in qualified_response.json()] == [first.id]
    assert assigned_response.status_code == 200
    assert [inquiry["id"] for inquiry in assigned_response.json()] == [first.id]
    assert unassigned_response.status_code == 200
    assert [inquiry["id"] for inquiry in unassigned_response.json()] == [second.id]
    assert contradictory_filter_response.status_code == 400
    assert summary_response.status_code == 200
    summary = summary_response.json()
    assert summary["total_count"] == 2
    assert summary["unassigned_count"] == 1
    assert summary["by_status"] == [
        {"status": "new", "count": 1},
        {"status": "contacted", "count": 0},
        {"status": "qualified", "count": 1},
        {"status": "closed", "count": 0},
    ]
    assert summary["by_assignee"] == [
        {
            "assigned_to": None,
            "total_count": 1,
            "by_status": [
                {"status": "new", "count": 1},
                {"status": "contacted", "count": 0},
                {"status": "qualified", "count": 0},
                {"status": "closed", "count": 0},
            ],
        },
        {
            "assigned_to": "sales.owner@example.com",
            "total_count": 1,
            "by_status": [
                {"status": "new", "count": 0},
                {"status": "contacted", "count": 0},
                {"status": "qualified", "count": 1},
                {"status": "closed", "count": 0},
            ],
        },
    ]
    assert "Jordan Auditor" not in str(summary)
    assert "Booked discovery call" not in str(summary)
    assert clear_notes_response.status_code == 200
    assert clear_notes_response.json()["status"] == "qualified"
    assert clear_notes_response.json()["assigned_to"] == "sales.owner@example.com"
    assert clear_notes_response.json()["follow_up_notes"] is None
    assert invalid_assignee_response.status_code == 400
    stored = store.get_marketing_inquiry(first.id)
    assert stored["status"] == "qualified"
    assert stored["assigned_to"] == "sales.owner@example.com"
    assert stored["follow_up_notes"] is None
    assert stored["updated_at"] is not None
    assert not_found_response.status_code == 404
    assert activity_response.status_code == 200
    activity = activity_response.json()
    assert [event["action"] for event in activity] == ["updated", "updated", "created"]
    assert activity[0]["actor_label"] == "Taylor Support"
    assert activity[0]["changed_fields"] == ["follow_up_notes"]
    assert activity[0]["notes_changed"] is True
    assert activity[0]["status"] == "qualified"
    assert activity[0]["assigned_to"] == "sales.owner@example.com"
    assert activity[1]["actor_label"] == "Taylor Support"
    assert activity[1]["changed_fields"] == ["status", "assigned_to", "follow_up_notes"]
    assert activity[1]["notes_changed"] is True
    assert activity[1]["status"] == "qualified"
    assert activity[2]["actor_label"] == "public_form"
    assert activity[2]["action"] == "created"
    assert activity[2]["changed_fields"] == [
        "email",
        "name",
        "organization_name",
        "role",
        "intended_use",
        "message",
        "source_page",
        "status",
    ]
    assert "Booked discovery call" not in str(activity)
    assert "jordan@example.com" not in str(activity)
    assert missing_activity_response.status_code == 404


def test_marketing_inquiry_rejects_invalid_email_and_unknown_use_case():
    store = SaaSStore(initialize_database())
    client = TestClient(create_app(store=store))

    bad_email = client.post(
        "/marketing/inquiries",
        json={
            "name": "Jordan Auditor",
            "email": "not-an-email",
            "intended_use": "audit_readiness",
            "message": "Need a pilot.",
        },
    )
    bad_use_case = client.post(
        "/marketing/inquiries",
        json={
            "name": "Jordan Auditor",
            "email": "jordan@example.com",
            "intended_use": "raw_meter_export",
            "message": "Need a pilot.",
        },
    )

    assert bad_email.status_code == 400
    assert bad_use_case.status_code == 400
    assert store.list_marketing_inquiries() == []
