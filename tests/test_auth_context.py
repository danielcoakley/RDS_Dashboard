from fastapi import HTTPException

from backend.auth_context import (
    claims_from_dev_token,
    dev_token_from_claims,
    request_authenticated_user,
    request_user_from_header,
    user_from_clerk_claims,
)


def test_request_user_from_header_returns_authenticated_user():
    user = request_user_from_header(" user_1 ")

    assert user.id == "user_1"


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


def test_request_authenticated_user_falls_back_to_user_id_header():
    user = request_authenticated_user(None, " user_1 ")

    assert user.id == "user_1"


def test_request_authenticated_user_rejects_bad_bearer_token():
    try:
        request_authenticated_user("Bearer not-a-dev-token", None)
        assert False, "Expected invalid bearer token to fail"
    except HTTPException as exc:
        assert exc.status_code == 401


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


def test_user_from_clerk_claims_requires_subject():
    try:
        user_from_clerk_claims({"org_id": "org_123"})
        assert False, "Expected missing subject to fail"
    except ValueError as exc:
        assert "subject" in str(exc)
