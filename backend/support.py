from __future__ import annotations

import hmac
import os

from .runtime_config import PRODUCTION_RUNTIME, runtime_env_from_env


SUPPORT_ACCESS_TOKEN_ENV = "SAAS_SUPPORT_ACCESS_TOKEN"
SUPPORT_OWNER_EMAIL_ENV = "SAAS_SUPPORT_OWNER_EMAIL"
SUPPORT_ACCESS_TOKEN_MIN_LENGTH = 32


class SupportConfigurationError(RuntimeError):
    pass


class SupportAuthenticationError(RuntimeError):
    pass


def _configured_support_token() -> str:
    return os.environ.get(SUPPORT_ACCESS_TOKEN_ENV, "").strip()


def _configured_support_owner_email() -> str:
    return os.environ.get(SUPPORT_OWNER_EMAIL_ENV, "").strip()


def _looks_like_email(value: str) -> bool:
    local, separator, domain = value.partition("@")
    return bool(separator and local.strip() and "." in domain and not value.isspace())


def support_access_configuration_status(runtime_env: str | None = None) -> tuple[bool, str]:
    active_runtime_env = runtime_env or runtime_env_from_env()
    token = _configured_support_token()
    owner_email = _configured_support_owner_email()
    if token:
        if active_runtime_env == PRODUCTION_RUNTIME and len(token) < SUPPORT_ACCESS_TOKEN_MIN_LENGTH:
            return (
                False,
                f"{SUPPORT_ACCESS_TOKEN_ENV} must be at least "
                f"{SUPPORT_ACCESS_TOKEN_MIN_LENGTH} characters in production",
            )
        if active_runtime_env == PRODUCTION_RUNTIME:
            if not owner_email:
                return (
                    False,
                    f"{SUPPORT_OWNER_EMAIL_ENV} must be configured when RDS_RUNTIME_ENV=production",
                )
            if not _looks_like_email(owner_email):
                return False, f"{SUPPORT_OWNER_EMAIL_ENV} must be an email address"
            return True, f"{SUPPORT_ACCESS_TOKEN_ENV} configured; {SUPPORT_OWNER_EMAIL_ENV} configured"
        return True, f"{SUPPORT_ACCESS_TOKEN_ENV} configured"
    if active_runtime_env == PRODUCTION_RUNTIME:
        return (
            False,
            f"{SUPPORT_ACCESS_TOKEN_ENV} must be configured when RDS_RUNTIME_ENV=production",
        )
    return True, f"{SUPPORT_ACCESS_TOKEN_ENV} is optional in development"


def verify_support_request(
    authorization: str | None,
    runtime_env: str | None = None,
) -> None:
    configured, detail = support_access_configuration_status(runtime_env)
    if not configured:
        raise SupportConfigurationError(detail)

    token = _configured_support_token()
    if not token:
        raise SupportConfigurationError(
            f"{SUPPORT_ACCESS_TOKEN_ENV} must be configured for support operations"
        )

    scheme, separator, presented_token = (authorization or "").partition(" ")
    if (
        separator != " "
        or scheme.lower() != "bearer"
        or not hmac.compare_digest(presented_token.strip(), token)
    ):
        raise SupportAuthenticationError("A valid support bearer token is required")
