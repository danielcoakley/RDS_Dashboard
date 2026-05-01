from fastapi import HTTPException

from backend.auth_context import request_user_from_header, user_from_clerk_claims


def test_request_user_from_header_returns_authenticated_user():
    user = request_user_from_header(" user_1 ")

    assert user.id == "user_1"


def test_request_user_from_header_rejects_missing_user():
    try:
        request_user_from_header(None)
        assert False, "Expected missing user context to fail"
    except HTTPException as exc:
        assert exc.status_code == 401


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
