"""Backend scaffolding for the SaaS platform boundary."""

from .access_control import AccessDenied, TenantContext, require_permission, require_resource_scope
from .domain import (
    Membership,
    Meter,
    Organization,
    Report,
    Role,
    Run,
    RunStatus,
    Site,
    Upload,
    UploadStatus,
    User,
)
from .database import apply_migrations, connect_database, initialize_database
from .rbac import Action, can_access_organization, can_perform

__all__ = [
    "Action",
    "AccessDenied",
    "Membership",
    "Meter",
    "Organization",
    "Report",
    "Role",
    "Run",
    "RunStatus",
    "Site",
    "TenantContext",
    "Upload",
    "UploadStatus",
    "User",
    "apply_migrations",
    "can_access_organization",
    "can_perform",
    "connect_database",
    "initialize_database",
    "require_permission",
    "require_resource_scope",
]
