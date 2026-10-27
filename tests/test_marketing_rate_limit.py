from backend.marketing_rate_limit import (
    MARKETING_INQUIRY_RATE_LIMIT_ENABLED_ENV,
    MARKETING_INQUIRY_RATE_LIMIT_MAX_ENV,
    MARKETING_INQUIRY_RATE_LIMIT_SECRET_ENV,
    MARKETING_INQUIRY_RATE_LIMIT_WINDOW_MINUTES_ENV,
    marketing_inquiry_rate_limit_configuration_status,
    marketing_inquiry_rate_limit_key,
    marketing_inquiry_rate_limit_policy_from_env,
)
from backend.runtime_config import RUNTIME_ENV_ENV

PRODUCTION_RATE_LIMIT_SECRET = "rate-limit-secret-for-production-tests-123456"


def test_marketing_inquiry_rate_limit_key_is_deterministic_and_private():
    source = "203.0.113.10|Test Browser"

    first = marketing_inquiry_rate_limit_key(source, "secret-one")
    second = marketing_inquiry_rate_limit_key(source, "secret-one")
    rotated = marketing_inquiry_rate_limit_key(source, "secret-two")

    assert first == second
    assert first != rotated
    assert "203.0.113.10" not in first
    assert "Test Browser" not in first
    assert len(first) == 64


def test_marketing_inquiry_rate_limit_policy_requires_secret_in_production(monkeypatch):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.delenv(MARKETING_INQUIRY_RATE_LIMIT_SECRET_ENV, raising=False)

    configured, detail = marketing_inquiry_rate_limit_configuration_status("production")

    assert configured is False
    assert MARKETING_INQUIRY_RATE_LIMIT_SECRET_ENV in detail

    monkeypatch.setenv(MARKETING_INQUIRY_RATE_LIMIT_SECRET_ENV, PRODUCTION_RATE_LIMIT_SECRET)
    monkeypatch.setenv(MARKETING_INQUIRY_RATE_LIMIT_MAX_ENV, "3")
    monkeypatch.setenv(MARKETING_INQUIRY_RATE_LIMIT_WINDOW_MINUTES_ENV, "15")

    policy = marketing_inquiry_rate_limit_policy_from_env("production")
    configured, detail = marketing_inquiry_rate_limit_configuration_status("production")

    assert policy.enabled is True
    assert policy.max_submissions == 3
    assert policy.window_minutes == 15
    assert policy.secret == PRODUCTION_RATE_LIMIT_SECRET
    assert configured is True
    assert "3 submissions per 15 minutes" in detail

    monkeypatch.setenv(MARKETING_INQUIRY_RATE_LIMIT_SECRET_ENV, "rate-secret")
    configured, detail = marketing_inquiry_rate_limit_configuration_status("production")

    assert configured is False
    assert "at least 32 characters" in detail


def test_marketing_inquiry_rate_limit_cannot_be_disabled_for_production_readiness(
    monkeypatch,
):
    monkeypatch.setenv(RUNTIME_ENV_ENV, "production")
    monkeypatch.setenv(MARKETING_INQUIRY_RATE_LIMIT_ENABLED_ENV, "false")

    configured, detail = marketing_inquiry_rate_limit_configuration_status("production")

    assert configured is False
    assert MARKETING_INQUIRY_RATE_LIMIT_ENABLED_ENV in detail
