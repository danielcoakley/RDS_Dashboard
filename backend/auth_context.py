from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Annotated

from fastapi import Header, HTTPException, status


@dataclass(frozen=True)
class AuthenticatedUser:
    id: str
    organization_id: str | None = None
    role: str | None = None


def request_user_from_header(
    x_user_id: Annotated[str | None, Header(alias="X-User-Id")] = None,
) -> AuthenticatedUser:
    if x_user_id is None or not x_user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authenticated user context",
        )
    return AuthenticatedUser(id=x_user_id.strip())


def dev_token_from_claims(claims: dict) -> str:
    payload = json.dumps(claims, separators=(",", ":"), sort_keys=True).encode("utf-8")
    encoded = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    return f"dev:{encoded}"


def claims_from_dev_token(token: str) -> dict:
    if not token.startswith("dev:"):
        raise ValueError("Unsupported development token format")

    encoded = token[4:]
    padding = "=" * (-len(encoded) % 4)
    try:
        decoded = base64.urlsafe_b64decode(f"{encoded}{padding}".encode("ascii"))
        claims = json.loads(decoded.decode("utf-8"))
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("Invalid development token payload") from exc

    if not isinstance(claims, dict):
        raise ValueError("Development token payload must be a JSON object")
    return claims


def request_authenticated_user(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
    x_user_id: Annotated[str | None, Header(alias="X-User-Id")] = None,
) -> AuthenticatedUser:
    if authorization and authorization.strip():
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization header",
            )
        try:
            claims = claims_from_dev_token(token.strip())
            return user_from_clerk_claims(claims)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(exc),
            ) from exc

    return request_user_from_header(x_user_id)


def user_from_clerk_claims(claims: dict) -> AuthenticatedUser:
    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject.strip():
        raise ValueError("Clerk claims must include a subject")

    organization_id = claims.get("org_id")
    role = claims.get("org_role") or claims.get("org_permissions")
    return AuthenticatedUser(
        id=subject.strip(),
        organization_id=organization_id if isinstance(organization_id, str) else None,
        role=role if isinstance(role, str) else None,
    )
