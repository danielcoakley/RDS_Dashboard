from fastapi import HTTPException

from backend.auth_context import request_user_from_header


def test_request_user_from_header_returns_authenticated_user():
    user = request_user_from_header(" user_1 ")

    assert user.id == "user_1"


def test_request_user_from_header_rejects_missing_user():
    try:
        request_user_from_header(None)
        assert False, "Expected missing user context to fail"
    except HTTPException as exc:
        assert exc.status_code == 401
