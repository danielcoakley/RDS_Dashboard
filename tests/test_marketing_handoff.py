import hashlib
import hmac
import json

from backend.domain import MarketingInquiry
from backend.marketing_handoff import (
    CRM_WEBHOOK_INQUIRY_ID_HEADER,
    CRM_WEBHOOK_SIGNATURE_HEADER,
    MARKETING_INQUIRY_CRM_WEBHOOK_SECRET_ENV,
    MARKETING_INQUIRY_CRM_WEBHOOK_TIMEOUT_SECONDS_ENV,
    MARKETING_INQUIRY_CRM_WEBHOOK_URL_ENV,
    WebhookMarketingInquiryCrmSender,
    marketing_inquiry_crm_configuration_status,
    marketing_inquiry_crm_payload,
    marketing_inquiry_crm_sender_from_env,
    marketing_inquiry_handoff_configuration_status,
)
from backend.notifications import MARKETING_INQUIRY_RECIPIENT_EMAIL_ENV
from backend.runtime_config import RUNTIME_ENV_ENV


PRODUCTION_CRM_WEBHOOK_SECRET = "crm-webhook-secret-for-production-tests-123456"


class WebhookResponse:
    status = 202

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class RecordingOpener:
    def __init__(self):
        self.request = None
        self.timeout = None

    def __call__(self, request, timeout):
        self.request = request
        self.timeout = timeout
        return WebhookResponse()


def test_webhook_marketing_inquiry_sender_posts_signed_payload():
    inquiry = MarketingInquiry(
        id="marketing_inquiry_1",
        email="alex@example.com",
        name="Alex Manager",
        organization_name="Example Estates",
        role="Energy Lead",
        intended_use="audit_readiness",
        message="Need a pilot.",
        source_page="/",
    )
    opener = RecordingOpener()
    sender = WebhookMarketingInquiryCrmSender(
        url="https://crm.example.com/hooks/rds",
        secret="crm-secret",
        timeout_seconds=3.5,
        opener=opener,
    )

    sender.send(inquiry)

    assert opener.request is not None
    assert opener.request.full_url == "https://crm.example.com/hooks/rds"
    assert opener.request.get_method() == "POST"
    assert opener.timeout == 3.5
    assert json.loads(opener.request.data.decode("utf-8")) == marketing_inquiry_crm_payload(
        inquiry
    )
    expected_signature = hmac.new(
        b"crm-secret",
        opener.request.data,
        hashlib.sha256,
    ).hexdigest()
    headers = {key.lower(): value for key, value in opener.request.header_items()}
    assert headers[CRM_WEBHOOK_INQUIRY_ID_HEADER.lower()] == inquiry.id
    assert headers[CRM_WEBHOOK_SIGNATURE_HEADER.lower()] == f"sha256={expected_signature}"


def test_marketing_inquiry_crm_sender_from_env_validates_production_config(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv(MARKETING_INQUIRY_CRM_WEBHOOK_URL_ENV, "http://crm.example.com/hook")

    configured, detail = marketing_inquiry_crm_configuration_status("production")

    assert configured is False
    assert "https" in detail

    monkeypatch.setenv(MARKETING_INQUIRY_CRM_WEBHOOK_URL_ENV, "https://crm.example.com/hook")
    monkeypatch.delenv(MARKETING_INQUIRY_CRM_WEBHOOK_SECRET_ENV, raising=False)

    configured, detail = marketing_inquiry_crm_configuration_status("production")

    assert configured is False
    assert MARKETING_INQUIRY_CRM_WEBHOOK_SECRET_ENV in detail

    monkeypatch.setenv(MARKETING_INQUIRY_CRM_WEBHOOK_SECRET_ENV, "crm-secret")
    configured, detail = marketing_inquiry_crm_configuration_status("production")

    assert configured is False
    assert "at least 32 characters" in detail

    monkeypatch.setenv(
        MARKETING_INQUIRY_CRM_WEBHOOK_SECRET_ENV,
        PRODUCTION_CRM_WEBHOOK_SECRET,
    )
    monkeypatch.setenv(MARKETING_INQUIRY_CRM_WEBHOOK_TIMEOUT_SECONDS_ENV, "2.5")

    sender = marketing_inquiry_crm_sender_from_env("production")
    configured, detail = marketing_inquiry_crm_configuration_status("production")

    assert isinstance(sender, WebhookMarketingInquiryCrmSender)
    assert sender.url == "https://crm.example.com/hook"
    assert sender.secret == PRODUCTION_CRM_WEBHOOK_SECRET
    assert sender.timeout_seconds == 2.5
    assert configured is True
    assert MARKETING_INQUIRY_CRM_WEBHOOK_URL_ENV in detail


def test_marketing_inquiry_handoff_configuration_allows_email_or_crm(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.delenv(MARKETING_INQUIRY_RECIPIENT_EMAIL_ENV, raising=False)
    monkeypatch.delenv(MARKETING_INQUIRY_CRM_WEBHOOK_URL_ENV, raising=False)
    monkeypatch.delenv(MARKETING_INQUIRY_CRM_WEBHOOK_SECRET_ENV, raising=False)

    configured, detail = marketing_inquiry_handoff_configuration_status("production")

    assert configured is False
    assert MARKETING_INQUIRY_RECIPIENT_EMAIL_ENV in detail
    assert MARKETING_INQUIRY_CRM_WEBHOOK_URL_ENV in detail

    monkeypatch.setenv(MARKETING_INQUIRY_RECIPIENT_EMAIL_ENV, "sales@example.com")

    configured, detail = marketing_inquiry_handoff_configuration_status("production")

    assert configured is True
    assert MARKETING_INQUIRY_RECIPIENT_EMAIL_ENV in detail

    monkeypatch.delenv(MARKETING_INQUIRY_RECIPIENT_EMAIL_ENV, raising=False)
    monkeypatch.setenv(MARKETING_INQUIRY_CRM_WEBHOOK_URL_ENV, "https://crm.example.com/hook")
    monkeypatch.setenv(
        MARKETING_INQUIRY_CRM_WEBHOOK_SECRET_ENV,
        PRODUCTION_CRM_WEBHOOK_SECRET,
    )

    configured, detail = marketing_inquiry_handoff_configuration_status("production")

    assert configured is True
    assert MARKETING_INQUIRY_CRM_WEBHOOK_URL_ENV in detail
