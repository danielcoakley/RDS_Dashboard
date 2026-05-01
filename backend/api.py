from __future__ import annotations

from fastapi import FastAPI


API_TITLE = "RDS Energy Analytics SaaS API"
API_VERSION = "0.1.0"


def health_check() -> dict[str, str]:
    return {"status": "ok"}


def readiness_check() -> dict[str, str]:
    return {"status": "ready", "service": "rds-saas-api"}


def create_app() -> FastAPI:
    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description="Backend API boundary for the SaaS energy analytics platform.",
    )
    app.add_api_route("/health", health_check, methods=["GET"], tags=["system"])
    app.add_api_route("/ready", readiness_check, methods=["GET"], tags=["system"])
    return app


app = create_app()
