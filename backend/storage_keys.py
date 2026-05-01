from __future__ import annotations

from pathlib import PurePosixPath
import re


_SAFE_PART = re.compile(r"[^a-zA-Z0-9._-]+")


def safe_key_part(value: str) -> str:
    cleaned = _SAFE_PART.sub("-", value.strip()).strip("-._")
    if not cleaned:
        raise ValueError("Storage key part cannot be empty")
    return cleaned.lower()


def upload_storage_key(
    organization_id: str,
    site_id: str,
    upload_id: str,
    filename: str,
) -> str:
    return str(
        PurePosixPath(
            "tenants",
            safe_key_part(organization_id),
            "sites",
            safe_key_part(site_id),
            "uploads",
            safe_key_part(upload_id),
            safe_key_part(filename),
        )
    )


def report_storage_key(
    organization_id: str,
    site_id: str,
    run_id: str,
    filename: str,
) -> str:
    return str(
        PurePosixPath(
            "tenants",
            safe_key_part(organization_id),
            "sites",
            safe_key_part(site_id),
            "runs",
            safe_key_part(run_id),
            "reports",
            safe_key_part(filename),
        )
    )
