from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from typing import Annotated, Any
from urllib.parse import urlparse

from fastapi import Cookie, Header, HTTPException, status


AUTH_MODE_ENV = "RDS_AUTH_MODE"
DEV_AUTH_MODE = "dev"
CLERK_AUTH_MODE = "clerk"
SUPPORTED_AUTH_MODES = {DEV_AUTH_MODE, CLERK_AUTH_MODE}
CLERK_ISSUER_ENV = "CLERK_ISSUER"
CLERK_JWT_KEY_ENV = "CLERK_JWT_KEY"
CLERK_JWKS_URL_ENV = "CLERK_JWKS_URL"
CLERK_AUDIENCE_ENV = "CLERK_AUDIENCE"
CLERK_AUTHORIZED_PARTIES_ENV = "CLERK_AUTHORIZED_PARTIES"
CLERK_CLOCK_SKEW_SECONDS_ENV = "CLERK_CLOCK_SKEW_SECONDS"
AUTH_FLOW_VERIFICATION_URL_ENV = "SAAS_AUTH_FLOW_VERIFICATION_URL"
AUTH_FLOW_VERIFIED_AT_ENV = "SAAS_AUTH_FLOW_VERIFIED_AT"
AUTH_FLOW_VERIFICATION_MAX_AGE_DAYS_ENV = "SAAS_AUTH_FLOW_VERIFICATION_MAX_AGE_DAYS"
DEFAULT_AUTH_FLOW_VERIFICATION_MAX_AGE_DAYS = 30
PRODUCTION_RUNTIME = "production"
CLERK_JWT_ALGORITHMS = ["RS256"]


@dataclass(frozen=True)
class AuthenticatedUser:
    id: str
    organization_id: str | None = None
    role: str | None = None
    auth_mode: str | None = None
    email: str | None = None
    display_name: str | None = None
    organization_slug: str | None = None
    organization_name: str | None = None


@dataclass(frozen=True)
class ClerkAuthConfig:
    issuer: str
    jwt_key: str | None = None
    jwks_url: str | None = None
    audience: tuple[str, ...] = ()
    authorized_parties: tuple[str, ...] = ()
    clock_skew_seconds: int = 5


def auth_mode_from_env() -> str:
    value = os.getenv(AUTH_MODE_ENV, DEV_AUTH_MODE).strip().lower()
    if value not in SUPPORTED_AUTH_MODES:
        raise ValueError(
            f"{AUTH_MODE_ENV} must be one of: {', '.join(sorted(SUPPORTED_AUTH_MODES))}"
        )
    return value


def _configured_auth_mode() -> str:
    try:
        return auth_mode_from_env()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc


def _csv_env(name: str) -> tuple[str, ...]:
    raw_value = os.getenv(name, "")
    return tuple(part.strip() for part in raw_value.split(",") if part.strip())


def _env_text(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _positive_int_env(name: str, default: int) -> int:
    value = _env_text(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def _require_https_url(name: str, runtime_env: str) -> str | None:
    value = _env_text(name)
    if value is None:
        if runtime_env == PRODUCTION_RUNTIME:
            raise ValueError(f"{name} is required in production")
        return None

    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{name} must be a URL")
    if runtime_env == PRODUCTION_RUNTIME and parsed.scheme != "https":
        raise ValueError(f"{name} must use https in production")
    return value


def auth_flow_verification_max_age_days_from_env() -> int:
    return _positive_int_env(
        AUTH_FLOW_VERIFICATION_MAX_AGE_DAYS_ENV,
        DEFAULT_AUTH_FLOW_VERIFICATION_MAX_AGE_DAYS,
    )


def _parse_auth_flow_verified_at(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{AUTH_FLOW_VERIFIED_AT_ENV} must be an ISO-8601 datetime") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{AUTH_FLOW_VERIFIED_AT_ENV} must include a timezone")
    return parsed.astimezone(timezone.utc)


def auth_flow_verification_status(
    runtime_env: str,
    *,
    now: datetime | None = None,
) -> tuple[bool, str]:
    try:
        verification_url = _require_https_url(AUTH_FLOW_VERIFICATION_URL_ENV, runtime_env)
        max_age_days = auth_flow_verification_max_age_days_from_env()
    except ValueError as exc:
        return False, str(exc)

    verified_at = _env_text(AUTH_FLOW_VERIFIED_AT_ENV)
    if runtime_env == PRODUCTION_RUNTIME and verified_at is None:
        return False, f"{AUTH_FLOW_VERIFIED_AT_ENV} is required in production"
    if verified_at is None:
        return True, "auth flow verification optional in development"

    try:
        observed_at = _parse_auth_flow_verified_at(verified_at)
    except ValueError as exc:
        return False, str(exc)

    effective_now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if observed_at > effective_now:
        return False, f"{AUTH_FLOW_VERIFIED_AT_ENV} cannot be in the future"
    age_days = (effective_now - observed_at).days
    if runtime_env == PRODUCTION_RUNTIME and age_days > max_age_days:
        return False, f"{AUTH_FLOW_VERIFIED_AT_ENV} is older than {max_age_days} days"

    detail = f"auth flow verified age {age_days} day(s)"
    if verification_url:
        detail = f"{AUTH_FLOW_VERIFICATION_URL_ENV} configured; {detail}"
    return True, detail


def _validate_clerk_jwt_key(jwt_key: str) -> None:
    try:
        from cryptography.hazmat.primitives.serialization import load_pem_public_key
    except ImportError as exc:
        raise RuntimeError("cryptography is required for Clerk JWT key validation") from exc
    try:
        load_pem_public_key(jwt_key.encode("utf-8"))
    except ValueError as exc:
        raise ValueError(f"{CLERK_JWT_KEY_ENV} must be a PEM public key") from exc


def _validate_jwks_url(jwks_url: str) -> None:
    parsed = urlparse(jwks_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{CLERK_JWKS_URL_ENV} must be an http(s) URL")


def clerk_auth_config_from_env() -> ClerkAuthConfig:
    issuer = _env_text(CLERK_ISSUER_ENV)
    jwt_key = _env_text(CLERK_JWT_KEY_ENV)
    jwks_url = _env_text(CLERK_JWKS_URL_ENV)

    if jwt_key is not None:
        jwt_key = jwt_key.replace("\\n", "\n")
        _validate_clerk_jwt_key(jwt_key)
    if jwks_url is not None:
        _validate_jwks_url(jwks_url)
    if issuer is not None:
        issuer = issuer.rstrip("/")

    if issuer is None:
        raise ValueError(f"{CLERK_ISSUER_ENV} is required when {AUTH_MODE_ENV}=clerk")
    if jwt_key is None and jwks_url is None:
        raise ValueError(
            f"Set {CLERK_JWT_KEY_ENV} or {CLERK_JWKS_URL_ENV} when {AUTH_MODE_ENV}=clerk"
        )

    clock_skew_raw = _env_text(CLERK_CLOCK_SKEW_SECONDS_ENV)
    if clock_skew_raw is None:
        clock_skew_seconds = 5
    else:
        try:
            clock_skew_seconds = int(clock_skew_raw)
        except ValueError as exc:
            raise ValueError(f"{CLERK_CLOCK_SKEW_SECONDS_ENV} must be an integer") from exc
        if clock_skew_seconds < 0:
            raise ValueError(f"{CLERK_CLOCK_SKEW_SECONDS_ENV} must be non-negative")

    return ClerkAuthConfig(
        issuer=issuer,
        jwt_key=jwt_key,
        jwks_url=jwks_url,
        audience=_csv_env(CLERK_AUDIENCE_ENV),
        authorized_parties=_csv_env(CLERK_AUTHORIZED_PARTIES_ENV),
        clock_skew_seconds=clock_skew_seconds,
    )


def clerk_auth_configuration_status(
    runtime_env: str = "development",
    *,
    now: datetime | None = None,
) -> tuple[bool, str]:
    try:
        config = clerk_auth_config_from_env()
    except (RuntimeError, ValueError) as exc:
        return False, str(exc)

    source = CLERK_JWT_KEY_ENV if config.jwt_key is not None else CLERK_JWKS_URL_ENV
    checks = [f"{AUTH_MODE_ENV}={CLERK_AUTH_MODE}", f"{source} configured"]
    if config.audience:
        checks.append(f"{CLERK_AUDIENCE_ENV} configured")
    if config.authorized_parties:
        checks.append(f"{CLERK_AUTHORIZED_PARTIES_ENV} configured")
    verification_ready, verification_detail = auth_flow_verification_status(
        runtime_env,
        now=now,
    )
    if not verification_ready:
        return False, verification_detail
    checks.append(verification_detail)
    return True, "; ".join(checks)


def request_user_from_header(
    x_user_id: Annotated[str | None, Header(alias="X-User-Id")] = None,
    *,
    auth_mode: str | None = None,
) -> AuthenticatedUser:
    mode = auth_mode or _configured_auth_mode()
    if mode != DEV_AUTH_MODE:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Development user headers are disabled outside dev auth mode",
        )
    if x_user_id is None or not x_user_id.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authenticated user context",
        )
    return AuthenticatedUser(id=x_user_id.strip(), auth_mode=mode)


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


def verify_clerk_session_token(
    token: str,
    *,
    config: ClerkAuthConfig | None = None,
) -> dict[str, Any]:
    active_config = config or clerk_auth_config_from_env()
    try:
        import jwt
        from jwt import InvalidTokenError, PyJWKClient, PyJWKClientError
    except ImportError as exc:
        raise RuntimeError("PyJWT is required for Clerk JWT verification") from exc

    try:
        if active_config.jwt_key is not None:
            signing_key: Any = active_config.jwt_key
        else:
            assert active_config.jwks_url is not None
            jwks_client = PyJWKClient(active_config.jwks_url)
            signing_key = jwks_client.get_signing_key_from_jwt(token).key

        claims = jwt.decode(
            token,
            signing_key,
            algorithms=CLERK_JWT_ALGORITHMS,
            audience=list(active_config.audience) or None,
            issuer=active_config.issuer,
            leeway=active_config.clock_skew_seconds,
            options={
                "require": ["sub", "exp", "iss"],
                "verify_aud": bool(active_config.audience),
            },
        )
    except (InvalidTokenError, PyJWKClientError) as exc:
        raise ValueError("Invalid Clerk session token") from exc

    if not isinstance(claims, dict):
        raise ValueError("Clerk session token payload must be a JSON object")

    if active_config.authorized_parties:
        authorized_party = claims.get("azp")
        if (
            not isinstance(authorized_party, str)
            or authorized_party not in active_config.authorized_parties
        ):
            raise ValueError("Clerk token authorized party is not allowed")

    return claims


def request_authenticated_user(
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
    x_user_id: Annotated[str | None, Header(alias="X-User-Id")] = None,
    session_cookie: Annotated[str | None, Cookie(alias="__session")] = None,
) -> AuthenticatedUser:
    mode = _configured_auth_mode()
    bearer_token: str | None = None
    if authorization and authorization.strip():
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authorization header",
            )
        bearer_token = token.strip()
    elif session_cookie is not None and session_cookie.strip():
        bearer_token = session_cookie.strip()

    if bearer_token is not None:
        try:
            if mode == DEV_AUTH_MODE:
                claims = claims_from_dev_token(bearer_token)
            else:
                if bearer_token.startswith("dev:"):
                    raise ValueError(
                        "Development bearer tokens are disabled; use a verified Clerk JWT"
                    )
                config = clerk_auth_config_from_env()
                claims = verify_clerk_session_token(bearer_token, config=config)
            user = user_from_clerk_claims(claims)
            return AuthenticatedUser(
                id=user.id,
                organization_id=user.organization_id,
                role=user.role,
                auth_mode=mode,
                email=user.email,
                display_name=user.display_name,
                organization_slug=user.organization_slug,
                organization_name=user.organization_name,
            )
        except RuntimeError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc
        except ValueError as exc:
            status_code = (
                status.HTTP_500_INTERNAL_SERVER_ERROR
                if mode == CLERK_AUTH_MODE
                and (
                    str(exc).startswith(CLERK_ISSUER_ENV)
                    or str(exc).startswith(CLERK_JWT_KEY_ENV)
                    or str(exc).startswith(CLERK_JWKS_URL_ENV)
                    or str(exc).startswith("Set ")
                    or str(exc).startswith(CLERK_CLOCK_SKEW_SECONDS_ENV)
                )
                else status.HTTP_401_UNAUTHORIZED
            )
            raise HTTPException(
                status_code=status_code,
                detail=str(exc),
            ) from exc

    return request_user_from_header(x_user_id, auth_mode=mode)


def _claim_text(value: Any) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _first_claim_text(claims: dict, *names: str) -> str | None:
    for name in names:
        value = _claim_text(claims.get(name))
        if value is not None:
            return value
    return None


def _email_from_claims(claims: dict) -> str | None:
    email = _first_claim_text(
        claims,
        "email",
        "primary_email",
        "primaryEmail",
        "email_address",
        "emailAddress",
    )
    if email is not None:
        return email

    primary_email_address = claims.get("primary_email_address")
    if isinstance(primary_email_address, dict):
        email = _first_claim_text(
            primary_email_address,
            "email",
            "email_address",
            "emailAddress",
        )
        if email is not None:
            return email

    email_addresses = claims.get("email_addresses")
    if isinstance(email_addresses, list):
        for item in email_addresses:
            if isinstance(item, str):
                email = _claim_text(item)
            elif isinstance(item, dict):
                email = _first_claim_text(item, "email", "email_address", "emailAddress")
            else:
                email = None
            if email is not None:
                return email

    return None


def _display_name_from_claims(claims: dict, email: str | None) -> str | None:
    display_name = _first_claim_text(
        claims,
        "name",
        "full_name",
        "fullName",
        "display_name",
        "displayName",
        "username",
    )
    if display_name is not None:
        return display_name

    first_name = _claim_text(claims.get("first_name") or claims.get("firstName"))
    last_name = _claim_text(claims.get("last_name") or claims.get("lastName"))
    combined_name = " ".join(part for part in (first_name, last_name) if part)
    if combined_name:
        return combined_name

    return email


def user_from_clerk_claims(claims: dict) -> AuthenticatedUser:
    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject.strip():
        raise ValueError("Clerk claims must include a subject")

    organization_id = _claim_text(claims.get("org_id"))
    organization_slug = _claim_text(claims.get("org_slug"))
    organization_name = _first_claim_text(claims, "org_name", "organization_name")
    role = _claim_text(claims.get("org_role")) or _claim_text(claims.get("org_permissions"))
    organization_claim = claims.get("o")
    if isinstance(organization_claim, dict):
        if organization_id is None:
            organization_id = _claim_text(organization_claim.get("id"))
        if role is None:
            role = _claim_text(organization_claim.get("rol"))
        if organization_slug is None:
            organization_slug = _claim_text(organization_claim.get("slg"))
        if organization_name is None:
            organization_name = _first_claim_text(organization_claim, "nam", "name")

    email = _email_from_claims(claims)

    return AuthenticatedUser(
        id=subject.strip(),
        organization_id=organization_id,
        role=role,
        email=email,
        display_name=_display_name_from_claims(claims, email),
        organization_slug=organization_slug,
        organization_name=organization_name,
    )
