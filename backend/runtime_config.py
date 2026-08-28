from __future__ import annotations

import os


RUNTIME_ENV_ENV = "RDS_RUNTIME_ENV"
DEVELOPMENT_RUNTIME = "development"
PRODUCTION_RUNTIME = "production"
SUPPORTED_RUNTIME_ENVS = {DEVELOPMENT_RUNTIME, PRODUCTION_RUNTIME}


def runtime_env_from_env() -> str:
    value = os.getenv(RUNTIME_ENV_ENV, DEVELOPMENT_RUNTIME).strip().lower()
    if value not in SUPPORTED_RUNTIME_ENVS:
        raise ValueError(
            f"{RUNTIME_ENV_ENV} must be one of: {', '.join(sorted(SUPPORTED_RUNTIME_ENVS))}"
        )
    return value


def is_production_runtime() -> bool:
    return runtime_env_from_env() == PRODUCTION_RUNTIME
