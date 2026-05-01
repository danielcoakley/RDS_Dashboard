from __future__ import annotations

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
