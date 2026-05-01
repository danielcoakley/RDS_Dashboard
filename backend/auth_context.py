from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Header, HTTPException, status


@dataclass(frozen=True)
class AuthenticatedUser:
    id: str


def request_user_from_header(
    x_user_id: Annotated[str | None, Header(alias="X-User-Id")] = None,
) -> AuthenticatedUser:
    if x_user_id is None or not x_user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authenticated user context",
        )
    return AuthenticatedUser(id=x_user_id.strip())
