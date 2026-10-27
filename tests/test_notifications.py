import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from backend.api import create_app
from backend.artifacts import InMemoryTenantArtifactStore
from backend.database import initialize_database
from backend.domain import (
    Membership,
    Notification,
    NotificationChannel,
    NotificationProviderEventStatus,
    NotificationStatus,
    Organization,
    Role,
    User,
)
from backend.notifications import (
    EMAIL_BACKEND_ENV,
    EMAIL_DELIVERY_VERIFICATION_MAX_AGE_DAYS_ENV,
    EMAIL_DELIVERY_VERIFICATION_URL_ENV,
    EMAIL_DELIVERY_VERIFIED_AT_ENV,
    EmailMessage,
    InMemoryEmailSender,
    MARKETING_INQUIRY_RECIPIENT_EMAIL_ENV,
    NOTIFICATION_MAX_ATTEMPTS_ENV,
    NOTIFICATION_PROVIDER_SIGNATURE_HEADER,
    NOTIFICATION_PROVIDER_TIMESTAMP_HEADER,
    NOTIFICATION_PROVIDER_WEBHOOK_SECRET_ENV,
    NOTIFICATION_PROVIDER_WEBHOOK_TOLERANCE_SECONDS_ENV,
    NOTIFICATION_RETRY_DELAY_SECONDS_ENV,
    NotificationProviderWebhookVerificationError,
    SMTP_FROM_EMAIL_ENV,
    SMTP_HOST_ENV,
    SMTP_PASSWORD_ENV,
    SMTP_PORT_ENV,
    SMTP_USERNAME_ENV,
    SmtpEmailSender,
    email_delivery_configuration_status,
    email_delivery_verification_status,
    email_sender_from_env,
    marketing_inquiry_configuration_status,
    marketing_inquiry_recipient_email_from_env,
    notification_max_attempts_from_env,
    notification_provider_webhook_configuration_status,
    notification_retry_delay_seconds_from_env,
    process_notification_provider_webhook_event,
    process_pending_notifications,
)
from backend.operations import NOTIFICATION_PROVIDER_WEBHOOK_REPLAY_EVENT_TYPE
from backend.store import SaaSStore

PROVIDER_WEBHOOK_SECRET = "notification-provider-webhook-secret-tests"


class CapturingSmtp:
    instances = []

    def __init__(self, host, port, timeout):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.started_tls = False
        self.login_args = None
        self.messages = []
        CapturingSmtp.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def starttls(self):
        self.started_tls = True

    def login(self, username, password):
        self.login_args = (username, password)

    def send_message(self, message):
        self.messages.append(message)


def seed_notification_tenants(store: SaaSStore) -> None:
    store.create_user(User(id="user_1", email="owner@example.com", display_name="Owner"))
    store.create_user(User(id="user_2", email="other@example.com", display_name="Other"))
    store.create_organization(Organization(id="org_1", name="Example Energy", slug="example-energy"))
    store.create_organization(Organization(id="org_2", name="Other Energy", slug="other-energy"))
    store.add_membership(Membership(user_id="user_1", organization_id="org_1", role=Role.OWNER))
    store.add_membership(Membership(user_id="user_2", organization_id="org_2", role=Role.OWNER))


def create_run_notification(
    store: SaaSStore,
    notification_id: str,
    organization_id: str,
    recipient_user_id: str,
    run_id: str,
) -> None:
    store.create_notification(
        Notification(
            id=notification_id,
            organization_id=organization_id,
            recipient_user_id=recipient_user_id,
            channel=NotificationChannel.EMAIL,
            event_type="run_succeeded",
            subject=f"Run {run_id} complete",
            body=f"Run {run_id} is ready.",
            resource_type="run",
            resource_id=run_id,
        )
    )


def provider_webhook_body(**overrides) -> bytes:
    payload = {
        "provider": "smtp-relay",
        "event_id": "evt_1",
        "notification_id": "notification_1",
        "status": "delivered",
        "occurred_at": "2026-07-20T09:05:00+00:00",
        "detail": "smtp 250 owner@example.com",
    }
    payload.update(overrides)
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")


def provider_webhook_headers(
    body: bytes,
    secret: str = PROVIDER_WEBHOOK_SECRET,
    timestamp: str | None = None,
) -> dict[str, str]:
    timestamp_text = timestamp or datetime.now(timezone.utc).isoformat()
    signed_payload = timestamp_text.encode("utf-8") + b"." + body
    digest = hmac.new(secret.encode("utf-8"), signed_payload, hashlib.sha256).hexdigest()
    return {
        NOTIFICATION_PROVIDER_SIGNATURE_HEADER: f"sha256={digest}",
        NOTIFICATION_PROVIDER_TIMESTAMP_HEADER: timestamp_text,
    }


def test_process_pending_notifications_sends_and_marks_sent():
    store = SaaSStore(initialize_database())
    seed_notification_tenants(store)
    create_run_notification(store, "notification_1", "org_1", "user_1", "run_1")
    sender = InMemoryEmailSender()

    summary = process_pending_notifications(store, sender, organization_id="org_1")

    notifications = store.list_notifications("org_1", recipient_user_id="user_1")
    assert summary.attempted == 1
    assert summary.sent == 1
    assert summary.failed == 0
    assert [message.to_email for message in sender.sent_messages] == ["owner@example.com"]
    assert notifications[0]["status"] == NotificationStatus.SENT.value
    assert notifications[0]["sent_at"] is not None
    assert notifications[0]["error_message"] is None


def test_notification_provider_webhook_marks_delivered_and_records_safe_event(monkeypatch):
    monkeypatch.setenv(NOTIFICATION_PROVIDER_WEBHOOK_SECRET_ENV, PROVIDER_WEBHOOK_SECRET)
    store = SaaSStore(initialize_database())
    seed_notification_tenants(store)
    create_run_notification(store, "notification_1", "org_1", "user_1", "run_1")
    body = provider_webhook_body()

    result = process_notification_provider_webhook_event(
        store,
        body,
        provider_webhook_headers(body),
    )

    notification = store.get_notification("notification_1")
    events = store.list_notification_provider_events("org_1", "notification_1")
    assert result.duplicate is False
    assert result.status == NotificationProviderEventStatus.DELIVERED
    assert result.organization_id == "org_1"
    assert result.notification_status == NotificationStatus.SENT
    assert notification["status"] == NotificationStatus.SENT.value
    assert notification["sent_at"] == "2026-07-20T09:05:00+00:00"
    assert notification["error_message"] is None
    assert len(events) == 1
    assert events[0]["provider"] == "smtp-relay"
    assert events[0]["provider_event_id"] == "evt_1"
    assert events[0]["status"] == "delivered"
    assert "@" not in events[0]["detail"]
    assert store.count_notification_provider_events_by_status()["delivered"] == 1


def test_notification_provider_webhook_is_idempotent_by_provider_event_id(monkeypatch):
    monkeypatch.setenv(NOTIFICATION_PROVIDER_WEBHOOK_SECRET_ENV, PROVIDER_WEBHOOK_SECRET)
    store = SaaSStore(initialize_database())
    seed_notification_tenants(store)
    create_run_notification(store, "notification_1", "org_1", "user_1", "run_1")
    body = provider_webhook_body()

    first = process_notification_provider_webhook_event(
        store,
        body,
        provider_webhook_headers(body),
    )
    duplicate = process_notification_provider_webhook_event(
        store,
        body,
        provider_webhook_headers(body),
    )

    assert first.duplicate is False
    assert duplicate.duplicate is True
    assert len(store.list_notification_provider_events("org_1", "notification_1")) == 1
    assert store.count_notification_provider_events_by_status()["delivered"] == 1


def test_notification_provider_webhook_marks_terminal_failure_without_raw_detail(
    monkeypatch,
):
    monkeypatch.setenv(NOTIFICATION_PROVIDER_WEBHOOK_SECRET_ENV, PROVIDER_WEBHOOK_SECRET)
    store = SaaSStore(initialize_database())
    seed_notification_tenants(store)
    create_run_notification(store, "notification_1", "org_1", "user_1", "run_1")
    body = provider_webhook_body(
        event_id="evt_bounce",
        status="bounced",
        detail="hard bounce for owner@example.com",
    )

    result = process_notification_provider_webhook_event(
        store,
        body,
        provider_webhook_headers(body),
    )

    notification = store.get_notification("notification_1")
    events = store.list_notification_provider_events("org_1", "notification_1")
    assert result.status == NotificationProviderEventStatus.BOUNCED
    assert result.notification_status == NotificationStatus.FAILED
    assert notification["status"] == NotificationStatus.FAILED.value
    assert notification["error_message"] == "provider_delivery_status=bounced"
    assert "owner@example.com" not in notification["error_message"]
    assert events[0]["status"] == "bounced"
    assert "@" not in events[0]["detail"]


def test_notification_provider_webhook_records_deferred_without_status_change(monkeypatch):
    monkeypatch.setenv(NOTIFICATION_PROVIDER_WEBHOOK_SECRET_ENV, PROVIDER_WEBHOOK_SECRET)
    store = SaaSStore(initialize_database())
    seed_notification_tenants(store)
    create_run_notification(store, "notification_1", "org_1", "user_1", "run_1")
    body = provider_webhook_body(event_id="evt_deferred", status="deferred")

    result = process_notification_provider_webhook_event(
        store,
        body,
        provider_webhook_headers(body),
    )

    notification = store.get_notification("notification_1")
    assert result.status == NotificationProviderEventStatus.DEFERRED
    assert result.notification_status == NotificationStatus.PENDING
    assert notification["status"] == NotificationStatus.PENDING.value
    assert store.count_notification_provider_events_by_status()["deferred"] == 1


def test_notification_provider_webhook_rejects_invalid_signature(monkeypatch):
    monkeypatch.setenv(NOTIFICATION_PROVIDER_WEBHOOK_SECRET_ENV, PROVIDER_WEBHOOK_SECRET)
    store = SaaSStore(initialize_database())
    seed_notification_tenants(store)
    create_run_notification(store, "notification_1", "org_1", "user_1", "run_1")
    body = provider_webhook_body()
    headers = provider_webhook_headers(body)
    headers[NOTIFICATION_PROVIDER_SIGNATURE_HEADER] = "sha256=bad"

    try:
        process_notification_provider_webhook_event(
            store,
            body,
            headers,
        )
        assert False, "Expected invalid provider webhook signature to fail"
    except NotificationProviderWebhookVerificationError:
        pass


def test_notification_provider_webhook_rejects_missing_timestamp(monkeypatch):
    monkeypatch.setenv(NOTIFICATION_PROVIDER_WEBHOOK_SECRET_ENV, PROVIDER_WEBHOOK_SECRET)
    store = SaaSStore(initialize_database())
    seed_notification_tenants(store)
    create_run_notification(store, "notification_1", "org_1", "user_1", "run_1")
    body = provider_webhook_body()
    headers = provider_webhook_headers(body)
    headers.pop(NOTIFICATION_PROVIDER_TIMESTAMP_HEADER)

    try:
        process_notification_provider_webhook_event(store, body, headers)
        assert False, "Expected missing provider webhook timestamp to fail"
    except NotificationProviderWebhookVerificationError as exc:
        assert NOTIFICATION_PROVIDER_TIMESTAMP_HEADER in str(exc)


def test_notification_provider_webhook_rejects_stale_or_future_timestamp(monkeypatch):
    monkeypatch.setenv(NOTIFICATION_PROVIDER_WEBHOOK_SECRET_ENV, PROVIDER_WEBHOOK_SECRET)
    store = SaaSStore(initialize_database())
    seed_notification_tenants(store)
    create_run_notification(store, "notification_1", "org_1", "user_1", "run_1")
    body = provider_webhook_body()
    now = datetime(2026, 7, 20, 9, 10, tzinfo=timezone.utc)

    for timestamp in (
        (now - timedelta(seconds=301)).isoformat(),
        (now + timedelta(seconds=301)).isoformat(),
    ):
        try:
            process_notification_provider_webhook_event(
                store,
                body,
                provider_webhook_headers(body, timestamp=timestamp),
                now=now,
            )
            assert False, "Expected provider webhook timestamp outside tolerance to fail"
        except NotificationProviderWebhookVerificationError as exc:
            assert "outside tolerance" in str(exc)


def test_notification_provider_webhook_configuration_status(monkeypatch):
    monkeypatch.delenv(NOTIFICATION_PROVIDER_WEBHOOK_SECRET_ENV, raising=False)
    monkeypatch.delenv(NOTIFICATION_PROVIDER_WEBHOOK_TOLERANCE_SECONDS_ENV, raising=False)

    configured, detail = notification_provider_webhook_configuration_status("development")

    assert configured is True
    assert "optional" in detail

    configured, detail = notification_provider_webhook_configuration_status("production")

    assert configured is False
    assert NOTIFICATION_PROVIDER_WEBHOOK_SECRET_ENV in detail

    monkeypatch.setenv(NOTIFICATION_PROVIDER_WEBHOOK_SECRET_ENV, "short")

    configured, detail = notification_provider_webhook_configuration_status("production")

    assert configured is False
    assert "at least 32 characters" in detail

    monkeypatch.setenv(NOTIFICATION_PROVIDER_WEBHOOK_SECRET_ENV, PROVIDER_WEBHOOK_SECRET)

    configured, detail = notification_provider_webhook_configuration_status("production")

    assert configured is True
    assert "configured" in detail

    monkeypatch.setenv(NOTIFICATION_PROVIDER_WEBHOOK_TOLERANCE_SECONDS_ENV, "0")

    configured, detail = notification_provider_webhook_configuration_status("production")

    assert configured is False
    assert NOTIFICATION_PROVIDER_WEBHOOK_TOLERANCE_SECONDS_ENV in detail


def test_notification_provider_webhook_route_accepts_signed_event(monkeypatch):
    monkeypatch.setenv(NOTIFICATION_PROVIDER_WEBHOOK_SECRET_ENV, PROVIDER_WEBHOOK_SECRET)
    store = SaaSStore(initialize_database())
    seed_notification_tenants(store)
    create_run_notification(store, "notification_1", "org_1", "user_1", "run_1")
    app = create_app(
        store=store,
        artifact_store=InMemoryTenantArtifactStore(),
    )
    client = TestClient(app)
    body = provider_webhook_body()

    response = client.post(
        "/webhooks/notifications/provider",
        content=body,
        headers=provider_webhook_headers(body),
    )
    rejected = client.post(
        "/webhooks/notifications/provider",
        content=body,
        headers={
            **provider_webhook_headers(body),
            NOTIFICATION_PROVIDER_SIGNATURE_HEADER: "sha256=bad",
        },
    )

    assert response.status_code == 200
    assert response.json()["event_id"] == "evt_1"
    assert response.json()["status"] == "delivered"
    assert response.json()["notification_status"] == "sent"
    assert "organization_id" not in response.json()
    assert rejected.status_code == 400
    operation_events = store.list_platform_operation_events(
        event_type=NOTIFICATION_PROVIDER_WEBHOOK_REPLAY_EVENT_TYPE,
        event_status="succeeded",
    )
    assert len(operation_events) == 1
    assert operation_events[0]["detail"] == "status=delivered duplicate=false"
    serialized_event = json.dumps(dict(operation_events[0]))
    assert "org_1" not in serialized_event
    assert "notification_1" not in serialized_event
    assert "evt_1" not in serialized_event
    assert "smtp-relay" not in serialized_event


def test_process_pending_notifications_schedules_retry_before_final_failure():
    store = SaaSStore(initialize_database())
    seed_notification_tenants(store)
    create_run_notification(store, "notification_1", "org_1", "user_1", "run_1")
    sender = InMemoryEmailSender(fail_notification_ids={"notification_1"})
    now = datetime(2026, 7, 20, 9, 0, tzinfo=timezone.utc)

    summary = process_pending_notifications(
        store,
        sender,
        organization_id="org_1",
        max_attempts=3,
        retry_delay_seconds=60,
        now=now,
    )

    notifications = store.list_notifications("org_1", recipient_user_id="user_1")
    assert summary.attempted == 1
    assert summary.sent == 0
    assert summary.failed == 0
    assert summary.retry_scheduled == 1
    assert sender.sent_messages == []
    assert notifications[0]["status"] == NotificationStatus.PENDING.value
    assert notifications[0]["attempt_count"] == 1
    assert notifications[0]["last_attempt_at"] == now.isoformat()
    assert notifications[0]["next_attempt_at"] == (now + timedelta(seconds=60)).isoformat()
    assert "Simulated email delivery failure" in notifications[0]["error_message"]


def test_process_pending_notifications_retries_due_failures_and_marks_sent():
    store = SaaSStore(initialize_database())
    seed_notification_tenants(store)
    create_run_notification(store, "notification_1", "org_1", "user_1", "run_1")
    first_sender = InMemoryEmailSender(fail_notification_ids={"notification_1"})
    first_attempt_at = datetime(2026, 7, 20, 9, 0, tzinfo=timezone.utc)

    process_pending_notifications(
        store,
        first_sender,
        organization_id="org_1",
        max_attempts=3,
        retry_delay_seconds=60,
        now=first_attempt_at,
    )
    too_soon = process_pending_notifications(
        store,
        InMemoryEmailSender(),
        organization_id="org_1",
        max_attempts=3,
        retry_delay_seconds=60,
        now=first_attempt_at + timedelta(seconds=30),
    )
    retry_sender = InMemoryEmailSender()
    retry = process_pending_notifications(
        store,
        retry_sender,
        organization_id="org_1",
        max_attempts=3,
        retry_delay_seconds=60,
        now=first_attempt_at + timedelta(seconds=60),
    )

    notifications = store.list_notifications("org_1", recipient_user_id="user_1")
    assert too_soon.attempted == 0
    assert retry.attempted == 1
    assert retry.sent == 1
    assert retry.failed == 0
    assert retry.retry_scheduled == 0
    assert len(retry_sender.sent_messages) == 1
    assert notifications[0]["status"] == NotificationStatus.SENT.value
    assert notifications[0]["attempt_count"] == 2
    assert notifications[0]["sent_at"] == (first_attempt_at + timedelta(seconds=60)).isoformat()
    assert notifications[0]["next_attempt_at"] is None
    assert notifications[0]["error_message"] is None


def test_process_pending_notifications_marks_final_failure_after_max_attempts():
    store = SaaSStore(initialize_database())
    seed_notification_tenants(store)
    create_run_notification(store, "notification_1", "org_1", "user_1", "run_1")
    sender = InMemoryEmailSender(fail_notification_ids={"notification_1"})
    now = datetime(2026, 7, 20, 9, 0, tzinfo=timezone.utc)

    summary = process_pending_notifications(
        store,
        sender,
        organization_id="org_1",
        max_attempts=1,
        retry_delay_seconds=60,
        now=now,
    )

    notifications = store.list_notifications("org_1", recipient_user_id="user_1")
    assert summary.attempted == 1
    assert summary.sent == 0
    assert summary.failed == 1
    assert summary.retry_scheduled == 0
    assert notifications[0]["status"] == NotificationStatus.FAILED.value
    assert notifications[0]["attempt_count"] == 1
    assert notifications[0]["last_attempt_at"] == now.isoformat()
    assert notifications[0]["next_attempt_at"] is None
    assert "Simulated email delivery failure" in notifications[0]["error_message"]


def test_process_pending_notifications_respects_tenant_scope_and_limit():
    store = SaaSStore(initialize_database())
    seed_notification_tenants(store)
    create_run_notification(store, "notification_1", "org_1", "user_1", "run_1")
    create_run_notification(store, "notification_2", "org_1", "user_1", "run_2")
    create_run_notification(store, "notification_3", "org_2", "user_2", "run_3")
    sender = InMemoryEmailSender()

    summary = process_pending_notifications(store, sender, organization_id="org_1", limit=1)

    org_1_statuses = [row["status"] for row in store.list_notifications("org_1")]
    org_2_statuses = [row["status"] for row in store.list_notifications("org_2")]
    assert summary.attempted == 1
    assert summary.sent == 1
    assert org_1_statuses.count(NotificationStatus.SENT.value) == 1
    assert org_1_statuses.count(NotificationStatus.PENDING.value) == 1
    assert org_2_statuses == [NotificationStatus.PENDING.value]


def test_notification_retry_policy_from_env(monkeypatch):
    monkeypatch.delenv(NOTIFICATION_MAX_ATTEMPTS_ENV, raising=False)
    monkeypatch.delenv(NOTIFICATION_RETRY_DELAY_SECONDS_ENV, raising=False)

    assert notification_max_attempts_from_env() == 3
    assert notification_retry_delay_seconds_from_env() == 300

    monkeypatch.setenv(NOTIFICATION_MAX_ATTEMPTS_ENV, "5")
    monkeypatch.setenv(NOTIFICATION_RETRY_DELAY_SECONDS_ENV, "30")

    assert notification_max_attempts_from_env() == 5
    assert notification_retry_delay_seconds_from_env() == 30


def test_smtp_email_sender_sends_message_with_tenant_headers():
    CapturingSmtp.instances = []
    sender = SmtpEmailSender(
        host="smtp.example.com",
        port=587,
        from_email="reports@example.com",
        username="smtp-user",
        password="smtp-password",
        smtp_factory=CapturingSmtp,
    )

    sender.send(
        EmailMessage(
            notification_id="notification_1",
            organization_id="org_1",
            recipient_user_id="user_1",
            to_email="owner@example.com",
            subject="Run complete",
            body="Your report is ready.",
        )
    )

    smtp = CapturingSmtp.instances[0]
    message = smtp.messages[0]
    assert smtp.host == "smtp.example.com"
    assert smtp.port == 587
    assert smtp.started_tls is True
    assert smtp.login_args == ("smtp-user", "smtp-password")
    assert message["From"] == "reports@example.com"
    assert message["To"] == "owner@example.com"
    assert message["Subject"] == "Run complete"
    assert message["X-RDS-Organization-Id"] == "org_1"
    assert message["X-RDS-Notification-Id"] == "notification_1"
    assert "Your report is ready." in message.get_content()


def test_email_sender_from_env_defaults_to_memory(monkeypatch):
    monkeypatch.delenv(EMAIL_BACKEND_ENV, raising=False)
    monkeypatch.delenv(EMAIL_DELIVERY_VERIFICATION_URL_ENV, raising=False)
    monkeypatch.delenv(EMAIL_DELIVERY_VERIFIED_AT_ENV, raising=False)

    sender = email_sender_from_env()
    configured, detail = email_delivery_configuration_status()
    verified, verification_detail = email_delivery_verification_status("development")

    assert isinstance(sender, InMemoryEmailSender)
    assert configured is True
    assert "memory" in detail
    assert verified is True
    assert "optional" in verification_detail


def test_marketing_inquiry_recipient_email_from_env_is_optional_and_validated(monkeypatch):
    monkeypatch.delenv(MARKETING_INQUIRY_RECIPIENT_EMAIL_ENV, raising=False)

    configured, detail = marketing_inquiry_configuration_status()

    assert marketing_inquiry_recipient_email_from_env() is None
    assert configured is False
    assert MARKETING_INQUIRY_RECIPIENT_EMAIL_ENV in detail

    monkeypatch.setenv(MARKETING_INQUIRY_RECIPIENT_EMAIL_ENV, " Sales@Example.COM ")

    configured, detail = marketing_inquiry_configuration_status()

    assert marketing_inquiry_recipient_email_from_env() == "sales@example.com"
    assert configured is True
    assert "configured" in detail

    monkeypatch.setenv(MARKETING_INQUIRY_RECIPIENT_EMAIL_ENV, "not-an-email")

    configured, detail = marketing_inquiry_configuration_status()

    assert configured is False
    assert MARKETING_INQUIRY_RECIPIENT_EMAIL_ENV in detail

    try:
        marketing_inquiry_recipient_email_from_env()
        assert False, "Expected invalid marketing recipient email to fail"
    except ValueError as exc:
        assert MARKETING_INQUIRY_RECIPIENT_EMAIL_ENV in str(exc)


def test_email_sender_from_env_builds_smtp_sender(monkeypatch):
    monkeypatch.setenv(EMAIL_BACKEND_ENV, "smtp")
    monkeypatch.setenv(SMTP_HOST_ENV, "smtp.example.com")
    monkeypatch.setenv(SMTP_FROM_EMAIL_ENV, "reports@example.com")
    monkeypatch.setenv(SMTP_PORT_ENV, "2525")
    monkeypatch.setenv(SMTP_USERNAME_ENV, "smtp-user")
    monkeypatch.setenv(SMTP_PASSWORD_ENV, "smtp-password")
    monkeypatch.setenv(
        EMAIL_DELIVERY_VERIFICATION_URL_ENV,
        "https://ops.example.com/email/delivery",
    )
    monkeypatch.setenv(EMAIL_DELIVERY_VERIFIED_AT_ENV, datetime.now(timezone.utc).isoformat())
    monkeypatch.setenv(EMAIL_DELIVERY_VERIFICATION_MAX_AGE_DAYS_ENV, "30")

    sender = email_sender_from_env()
    configured, detail = email_delivery_configuration_status("production")

    assert isinstance(sender, SmtpEmailSender)
    assert sender.host == "smtp.example.com"
    assert sender.port == 2525
    assert sender.from_email == "reports@example.com"
    assert sender.username == "smtp-user"
    assert sender.password == "smtp-password"
    assert configured is True
    assert SMTP_HOST_ENV in detail
    assert EMAIL_DELIVERY_VERIFICATION_URL_ENV in detail


def test_email_delivery_configuration_status_rejects_missing_production_evidence(
    monkeypatch,
):
    monkeypatch.setenv(EMAIL_BACKEND_ENV, "smtp")
    monkeypatch.setenv(SMTP_HOST_ENV, "smtp.example.com")
    monkeypatch.setenv(SMTP_FROM_EMAIL_ENV, "reports@example.com")
    monkeypatch.delenv(EMAIL_DELIVERY_VERIFICATION_URL_ENV, raising=False)
    monkeypatch.delenv(EMAIL_DELIVERY_VERIFIED_AT_ENV, raising=False)

    configured, detail = email_delivery_configuration_status("production")

    assert configured is False
    assert EMAIL_DELIVERY_VERIFICATION_URL_ENV in detail


def test_email_delivery_verification_status_rejects_stale_production_evidence(
    monkeypatch,
):
    monkeypatch.setenv(
        EMAIL_DELIVERY_VERIFICATION_URL_ENV,
        "https://ops.example.com/email/delivery",
    )
    monkeypatch.setenv(EMAIL_DELIVERY_VERIFICATION_MAX_AGE_DAYS_ENV, "7")
    monkeypatch.setenv(
        EMAIL_DELIVERY_VERIFIED_AT_ENV,
        (datetime.now(timezone.utc) - timedelta(days=8)).isoformat(),
    )

    configured, detail = email_delivery_verification_status("production")

    assert configured is False
    assert "older than 7 days" in detail


def test_email_sender_from_env_rejects_incomplete_smtp_config(monkeypatch):
    monkeypatch.setenv(EMAIL_BACKEND_ENV, "smtp")
    monkeypatch.delenv(SMTP_HOST_ENV, raising=False)
    monkeypatch.setenv(SMTP_FROM_EMAIL_ENV, "reports@example.com")

    configured, detail = email_delivery_configuration_status()

    assert configured is False
    assert SMTP_HOST_ENV in detail

    try:
        email_sender_from_env()
        assert False, "Expected missing SMTP host to fail"
    except ValueError as exc:
        assert SMTP_HOST_ENV in str(exc)
