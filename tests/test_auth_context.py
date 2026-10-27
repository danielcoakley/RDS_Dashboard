import json
import time
from datetime import datetime, timezone

from fastapi import HTTPException
import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from backend.auth_context import (
    AUTH_MODE_ENV,
    AUTH_FLOW_VERIFICATION_MAX_AGE_DAYS_ENV,
    AUTH_FLOW_VERIFICATION_URL_ENV,
    AUTH_FLOW_VERIFIED_AT_ENV,
    CLERK_AUTHORIZED_PARTIES_ENV,
    CLERK_ISSUER_ENV,
    CLERK_JWKS_URL_ENV,
    CLERK_JWT_KEY_ENV,
    ClerkAuthConfig,
    auth_flow_verification_status,
    auth_mode_from_env,
    clerk_auth_configuration_status,
    claims_from_dev_token,
    dev_token_from_claims,
    request_authenticated_user,
    request_user_from_header,
    user_from_clerk_claims,
    verify_clerk_session_token,
)


CLERK_TEST_ISSUER = "https://example.clerk.accounts.dev"


def rsa_private_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def public_key_pem(private_key) -> str:
    return (
        private_key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("utf-8")
    )


def clerk_session_token(private_key, claims: dict | None = None, headers: dict | None = None) -> str:
    now = int(time.time())
    payload = {
        "sub": "user_123",
        "iss": CLERK_TEST_ISSUER,
        "iat": now,
        "nbf": now - 1,
        "exp": now + 300,
        "azp": "https://app.example.com",
        "sid": "sess_123",
    }
    if claims:
        payload.update(claims)
    return jwt.encode(payload, private_key, algorithm="RS256", headers=headers)


def test_auth_mode_from_env_defaults_to_dev(monkeypatch):
    monkeypatch.delenv(AUTH_MODE_ENV, raising=False)

    assert auth_mode_from_env() == "dev"


def test_auth_mode_from_env_rejects_unknown_mode(monkeypatch):
    monkeypatch.setenv(AUTH_MODE_ENV, "passwords-in-a-spreadsheet")

    try:
        auth_mode_from_env()
        assert False, "Expected unsupported auth mode to fail"
    except ValueError as exc:
        assert AUTH_MODE_ENV in str(exc)


def test_request_user_from_header_returns_authenticated_user():
    user = request_user_from_header(" user_1 ")

    assert user.id == "user_1"
    assert user.auth_mode == "dev"


def test_request_user_from_header_rejects_missing_user():
    try:
        request_user_from_header(None)
        assert False, "Expected missing user context to fail"
    except HTTPException as exc:
        assert exc.status_code == 401


def test_request_authenticated_user_accepts_bearer_dev_token():
    token = dev_token_from_claims(
        {
            "sub": "user_123",
            "org_id": "org_123",
            "org_role": "org:admin",
        }
    )

    user = request_authenticated_user(f"Bearer {token}", None)

    assert user.id == "user_123"
    assert user.organization_id == "org_123"
    assert user.role == "org:admin"
    assert user.auth_mode == "dev"


def test_request_authenticated_user_falls_back_to_user_id_header():
    user = request_authenticated_user(None, " user_1 ")

    assert user.id == "user_1"


def test_request_authenticated_user_rejects_bad_bearer_token():
    try:
        request_authenticated_user("Bearer not-a-dev-token", None)
        assert False, "Expected invalid bearer token to fail"
    except HTTPException as exc:
        assert exc.status_code == 401


def test_request_authenticated_user_rejects_dev_token_in_clerk_mode(monkeypatch):
    monkeypatch.setenv(AUTH_MODE_ENV, "clerk")
    token = dev_token_from_claims({"sub": "user_123"})

    try:
        request_authenticated_user(f"Bearer {token}", None)
        assert False, "Expected dev token to be rejected in Clerk mode"
    except HTTPException as exc:
        assert exc.status_code == 401
        assert "Clerk JWT" in exc.detail


def test_request_authenticated_user_accepts_verified_clerk_bearer_token(monkeypatch):
    private_key = rsa_private_key()
    monkeypatch.setenv(AUTH_MODE_ENV, "clerk")
    monkeypatch.setenv(CLERK_ISSUER_ENV, CLERK_TEST_ISSUER)
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem(private_key))
    monkeypatch.setenv(CLERK_AUTHORIZED_PARTIES_ENV, "https://app.example.com")
    token = clerk_session_token(
        private_key,
        {
            "email": "owner@example.com",
            "fullName": "Owner User",
            "o": {
                "id": "org_123",
                "rol": "admin",
                "slg": "example-energy",
            }
        },
    )

    user = request_authenticated_user(f"Bearer {token}", None)

    assert user.id == "user_123"
    assert user.organization_id == "org_123"
    assert user.role == "admin"
    assert user.auth_mode == "clerk"
    assert user.email == "owner@example.com"
    assert user.display_name == "Owner User"
    assert user.organization_slug == "example-energy"


def test_request_authenticated_user_accepts_verified_clerk_session_cookie(monkeypatch):
    private_key = rsa_private_key()
    monkeypatch.setenv(AUTH_MODE_ENV, "clerk")
    monkeypatch.setenv(CLERK_ISSUER_ENV, CLERK_TEST_ISSUER)
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem(private_key))
    token = clerk_session_token(private_key, {"org_id": "org_123", "org_role": "org:viewer"})

    user = request_authenticated_user(None, None, token)

    assert user.id == "user_123"
    assert user.organization_id == "org_123"
    assert user.role == "org:viewer"
    assert user.auth_mode == "clerk"


def test_request_authenticated_user_rejects_wrong_authorized_party(monkeypatch):
    private_key = rsa_private_key()
    monkeypatch.setenv(AUTH_MODE_ENV, "clerk")
    monkeypatch.setenv(CLERK_ISSUER_ENV, CLERK_TEST_ISSUER)
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem(private_key))
    monkeypatch.setenv(CLERK_AUTHORIZED_PARTIES_ENV, "https://app.example.com")
    token = clerk_session_token(private_key, {"azp": "https://evil.example.com"})

    try:
        request_authenticated_user(f"Bearer {token}", None)
        assert False, "Expected wrong authorized party to fail"
    except HTTPException as exc:
        assert exc.status_code == 401
        assert "authorized party" in exc.detail


def test_request_authenticated_user_reports_missing_clerk_config(monkeypatch):
    monkeypatch.setenv(AUTH_MODE_ENV, "clerk")
    monkeypatch.delenv(CLERK_ISSUER_ENV, raising=False)
    monkeypatch.delenv(CLERK_JWT_KEY_ENV, raising=False)
    token = "not-a-real-jwt"

    try:
        request_authenticated_user(f"Bearer {token}", None)
        assert False, "Expected missing Clerk config to fail"
    except HTTPException as exc:
        assert exc.status_code == 500
        assert CLERK_ISSUER_ENV in exc.detail


def test_request_authenticated_user_rejects_user_header_in_clerk_mode(monkeypatch):
    monkeypatch.setenv(AUTH_MODE_ENV, "clerk")

    try:
        request_authenticated_user(None, "user_1")
        assert False, "Expected user header to be rejected in Clerk mode"
    except HTTPException as exc:
        assert exc.status_code == 401
        assert "Development user headers" in exc.detail


def test_request_authenticated_user_reports_invalid_auth_mode(monkeypatch):
    monkeypatch.setenv(AUTH_MODE_ENV, "unknown")

    try:
        request_authenticated_user(None, "user_1")
        assert False, "Expected invalid auth mode to fail"
    except HTTPException as exc:
        assert exc.status_code == 500
        assert AUTH_MODE_ENV in exc.detail


def test_dev_token_round_trip_preserves_claims():
    claims = {"sub": "user_123", "org_id": "org_123", "org_role": "org:viewer"}

    token = dev_token_from_claims(claims)

    assert claims_from_dev_token(token) == claims


def test_user_from_clerk_claims_maps_subject_and_org_context():
    user = user_from_clerk_claims(
        {
            "sub": "user_123",
            "org_id": "org_123",
            "org_role": "org:admin",
        }
    )

    assert user.id == "user_123"
    assert user.organization_id == "org_123"
    assert user.role == "org:admin"


def test_user_from_clerk_claims_maps_current_clerk_organization_shape():
    user = user_from_clerk_claims(
        {
            "sub": "user_123",
            "primary_email": "owner@example.com",
            "name": "Owner User",
            "o": {
                "id": "org_123",
                "rol": "admin",
                "slg": "example-energy",
            },
        }
    )

    assert user.id == "user_123"
    assert user.organization_id == "org_123"
    assert user.role == "admin"
    assert user.email == "owner@example.com"
    assert user.display_name == "Owner User"
    assert user.organization_slug == "example-energy"


def test_verify_clerk_session_token_accepts_jwks_url(tmp_path):
    private_key = rsa_private_key()
    key_id = "test-key"
    public_jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key()))
    public_jwk.update({"kid": key_id, "use": "sig", "alg": "RS256"})
    jwks_path = tmp_path / "jwks.json"
    jwks_path.write_text(json.dumps({"keys": [public_jwk]}), encoding="utf-8")
    token = clerk_session_token(private_key, headers={"kid": key_id})

    claims = verify_clerk_session_token(
        token,
        config=ClerkAuthConfig(
            issuer=CLERK_TEST_ISSUER,
            jwks_url=jwks_path.as_uri(),
        ),
    )

    assert claims["sub"] == "user_123"


def test_clerk_auth_configuration_status_requires_real_public_key(monkeypatch):
    monkeypatch.setenv(CLERK_ISSUER_ENV, CLERK_TEST_ISSUER)
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, "not a pem key")

    configured, detail = clerk_auth_configuration_status()

    assert configured is False
    assert CLERK_JWT_KEY_ENV in detail


def test_clerk_auth_configuration_status_requires_http_jwks_url(monkeypatch):
    monkeypatch.setenv(CLERK_ISSUER_ENV, CLERK_TEST_ISSUER)
    monkeypatch.delenv(CLERK_JWT_KEY_ENV, raising=False)
    monkeypatch.setenv(CLERK_JWKS_URL_ENV, "not-a-url")

    configured, detail = clerk_auth_configuration_status()

    assert configured is False
    assert CLERK_JWKS_URL_ENV in detail


def test_auth_flow_verification_status_is_optional_in_development(monkeypatch):
    monkeypatch.delenv(AUTH_FLOW_VERIFICATION_URL_ENV, raising=False)
    monkeypatch.delenv(AUTH_FLOW_VERIFIED_AT_ENV, raising=False)

    configured, detail = auth_flow_verification_status("development")

    assert configured is True
    assert "optional" in detail


def test_auth_flow_verification_status_accepts_fresh_production_evidence(monkeypatch):
    monkeypatch.setenv(
        AUTH_FLOW_VERIFICATION_URL_ENV,
        "https://ops.example.com/auth/flow-verification",
    )
    monkeypatch.setenv(AUTH_FLOW_VERIFIED_AT_ENV, "2026-07-20T12:00:00+00:00")
    monkeypatch.setenv(AUTH_FLOW_VERIFICATION_MAX_AGE_DAYS_ENV, "30")

    configured, detail = auth_flow_verification_status(
        "production",
        now=datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc),
    )

    assert configured is True
    assert AUTH_FLOW_VERIFICATION_URL_ENV in detail
    assert "age 1 day(s)" in detail


def test_auth_flow_verification_status_rejects_stale_production_evidence(monkeypatch):
    monkeypatch.setenv(
        AUTH_FLOW_VERIFICATION_URL_ENV,
        "https://ops.example.com/auth/flow-verification",
    )
    monkeypatch.setenv(AUTH_FLOW_VERIFIED_AT_ENV, "2026-07-12T12:00:00+00:00")
    monkeypatch.setenv(AUTH_FLOW_VERIFICATION_MAX_AGE_DAYS_ENV, "7")

    configured, detail = auth_flow_verification_status(
        "production",
        now=datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc),
    )

    assert configured is False
    assert "older than 7 days" in detail


def test_clerk_auth_configuration_status_requires_auth_flow_evidence_in_production(
    monkeypatch,
):
    private_key = rsa_private_key()
    monkeypatch.setenv(CLERK_ISSUER_ENV, CLERK_TEST_ISSUER)
    monkeypatch.setenv(CLERK_JWT_KEY_ENV, public_key_pem(private_key))
    monkeypatch.delenv(AUTH_FLOW_VERIFICATION_URL_ENV, raising=False)
    monkeypatch.delenv(AUTH_FLOW_VERIFIED_AT_ENV, raising=False)

    configured, detail = clerk_auth_configuration_status("production")

    assert configured is False
    assert AUTH_FLOW_VERIFICATION_URL_ENV in detail


def test_user_from_clerk_claims_requires_subject():
    try:
        user_from_clerk_claims({"org_id": "org_123"})
        assert False, "Expected missing subject to fail"
    except ValueError as exc:
        assert "subject" in str(exc)
