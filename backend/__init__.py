"""Backend scaffolding for the SaaS platform boundary."""

from .access_control import AccessDenied, TenantContext, require_permission, require_resource_scope
from .api import create_app
from .auth_context import AuthenticatedUser, request_user_from_header, user_from_clerk_claims
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
from .onboarding import OrganizationOnboardingResult, create_owner_organization
from .run_orchestration import AnalysisRunOutcome, AnalysisRunRequest, execute_analysis_run
from .store import SaaSStore
from .storage_keys import report_storage_key, safe_key_part, upload_storage_key

__all__ = [
    "Action",
    "AccessDenied",
    "AnalysisRunOutcome",
    "AnalysisRunRequest",
    "AuthenticatedUser",
    "Membership",
    "Meter",
    "Organization",
    "OrganizationOnboardingResult",
    "Report",
    "Role",
    "Run",
    "RunStatus",
    "SaaSStore",
    "Site",
    "TenantContext",
    "Upload",
    "UploadStatus",
    "User",
    "apply_migrations",
    "can_access_organization",
    "can_perform",
    "connect_database",
    "create_app",
    "create_owner_organization",
    "execute_analysis_run",
    "initialize_database",
    "require_permission",
    "require_resource_scope",
    "request_user_from_header",
    "report_storage_key",
    "safe_key_part",
    "upload_storage_key",
    "user_from_clerk_claims",
]
