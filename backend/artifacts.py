from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import os
from typing import Any, Protocol
from urllib.parse import urlparse
from uuid import uuid4

from .runtime_config import PRODUCTION_RUNTIME
from .storage_keys import safe_key_part


ARTIFACT_STORE_BACKEND_ENV = "SAAS_ARTIFACT_STORE_BACKEND"
ARTIFACT_STORE_DIR_ENV = "SAAS_ARTIFACT_STORE_DIR"
S3_BUCKET_ENV = "SAAS_S3_BUCKET"
S3_PREFIX_ENV = "SAAS_S3_PREFIX"
S3_ENDPOINT_URL_ENV = "SAAS_S3_ENDPOINT_URL"
S3_REGION_ENV = "SAAS_S3_REGION"
ARTIFACT_STORE_GOVERNANCE_URL_ENV = "SAAS_ARTIFACT_STORE_GOVERNANCE_URL"
ARTIFACT_STORE_GOVERNANCE_VERIFIED_AT_ENV = "SAAS_ARTIFACT_STORE_GOVERNANCE_VERIFIED_AT"
ARTIFACT_STORE_GOVERNANCE_MAX_AGE_DAYS_ENV = (
    "SAAS_ARTIFACT_STORE_GOVERNANCE_MAX_AGE_DAYS"
)
DEFAULT_ARTIFACT_STORE_GOVERNANCE_MAX_AGE_DAYS = 30
ARTIFACT_STORE_SMOKE_TEST_ORGANIZATION_ID = "platform_artifact_probe"


class ArtifactStoreSmokeTestError(RuntimeError):
    pass


class ArtifactStoreGovernanceConfigurationError(ValueError):
    pass


@dataclass(frozen=True)
class ArtifactStoreSmokeTestResult:
    tested_at: str
    bytes_written: int


class TenantArtifactStore(Protocol):
    def put_bytes(self, organization_id: str, storage_key: str, payload: bytes) -> None:
        ...

    def get_bytes(self, organization_id: str, storage_key: str) -> bytes:
        ...

    def exists(self, organization_id: str, storage_key: str) -> bool:
        ...

    def delete(self, organization_id: str, storage_key: str) -> bool:
        ...

    def put_text(self, organization_id: str, storage_key: str, payload: str) -> None:
        self.put_bytes(organization_id, storage_key, payload.encode("utf-8"))

    def get_text(self, organization_id: str, storage_key: str) -> str:
        return self.get_bytes(organization_id, storage_key).decode("utf-8")


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
        raise ArtifactStoreGovernanceConfigurationError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise ArtifactStoreGovernanceConfigurationError(f"{name} must be positive")
    return parsed


def _require_https_url(name: str, runtime_env: str) -> str | None:
    value = _env_text(name)
    if value is None:
        if runtime_env == PRODUCTION_RUNTIME:
            raise ArtifactStoreGovernanceConfigurationError(
                f"{name} is required in production"
            )
        return None

    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ArtifactStoreGovernanceConfigurationError(f"{name} must be a URL")
    if runtime_env == PRODUCTION_RUNTIME and parsed.scheme != "https":
        raise ArtifactStoreGovernanceConfigurationError(f"{name} must use https in production")
    return value


def artifact_store_governance_max_age_days_from_env() -> int:
    return _positive_int_env(
        ARTIFACT_STORE_GOVERNANCE_MAX_AGE_DAYS_ENV,
        DEFAULT_ARTIFACT_STORE_GOVERNANCE_MAX_AGE_DAYS,
    )


def _parse_artifact_store_governance_verified_at(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ArtifactStoreGovernanceConfigurationError(
            f"{ARTIFACT_STORE_GOVERNANCE_VERIFIED_AT_ENV} must be an ISO-8601 datetime"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ArtifactStoreGovernanceConfigurationError(
            f"{ARTIFACT_STORE_GOVERNANCE_VERIFIED_AT_ENV} must include a timezone"
        )
    return parsed.astimezone(timezone.utc)


def artifact_store_governance_status(
    runtime_env: str,
    *,
    now: datetime | None = None,
) -> tuple[bool, str]:
    try:
        governance_url = _require_https_url(ARTIFACT_STORE_GOVERNANCE_URL_ENV, runtime_env)
        max_age_days = artifact_store_governance_max_age_days_from_env()
    except ArtifactStoreGovernanceConfigurationError as exc:
        return False, str(exc)

    verified_at = _env_text(ARTIFACT_STORE_GOVERNANCE_VERIFIED_AT_ENV)
    if runtime_env == PRODUCTION_RUNTIME and verified_at is None:
        return False, f"{ARTIFACT_STORE_GOVERNANCE_VERIFIED_AT_ENV} is required in production"
    if verified_at is None:
        return True, "artifact store governance evidence optional in development"

    try:
        observed_at = _parse_artifact_store_governance_verified_at(verified_at)
    except ArtifactStoreGovernanceConfigurationError as exc:
        return False, str(exc)

    effective_now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if observed_at > effective_now:
        return False, f"{ARTIFACT_STORE_GOVERNANCE_VERIFIED_AT_ENV} cannot be in the future"
    age_days = (effective_now - observed_at).days
    if runtime_env == PRODUCTION_RUNTIME and age_days > max_age_days:
        return (
            False,
            f"{ARTIFACT_STORE_GOVERNANCE_VERIFIED_AT_ENV} is older than {max_age_days} days",
        )

    detail = f"artifact store governance verified age {age_days} day(s)"
    if governance_url:
        detail = f"{ARTIFACT_STORE_GOVERNANCE_URL_ENV} configured; {detail}"
    return True, detail


def validate_tenant_storage_key(organization_id: str, storage_key: str) -> tuple[str, ...]:
    path = PurePosixPath(storage_key)
    parts = path.parts
    expected_prefix = ("tenants", safe_key_part(organization_id))

    if path.is_absolute() or not parts:
        raise ValueError("Storage key must be a relative tenant key")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("Storage key cannot contain traversal segments")
    if parts[:2] != expected_prefix:
        raise ValueError(
            f"Storage key must start with tenants/{safe_key_part(organization_id)}"
        )

    return parts


def artifact_store_smoke_test(
    artifact_store: TenantArtifactStore,
    *,
    test_id: str | None = None,
    now: datetime | None = None,
) -> ArtifactStoreSmokeTestResult:
    tested_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    safe_test_id = safe_key_part(test_id or f"artifact-store-test-{uuid4().hex}")
    organization_id = ARTIFACT_STORE_SMOKE_TEST_ORGANIZATION_ID
    storage_key = str(
        PurePosixPath(
            "tenants",
            safe_key_part(organization_id),
            "platform",
            "artifact-store-tests",
            f"{safe_test_id}.txt",
        )
    )
    payload = f"rds artifact store smoke test\ncreated_at={tested_at}\n".encode("utf-8")

    try:
        artifact_store.put_bytes(organization_id, storage_key, payload)
    except Exception as exc:
        raise ArtifactStoreSmokeTestError("Artifact store smoke test failed during write") from exc

    try:
        if not artifact_store.exists(organization_id, storage_key):
            raise ArtifactStoreSmokeTestError("Artifact store smoke test failed during exists")
        if artifact_store.get_bytes(organization_id, storage_key) != payload:
            raise ArtifactStoreSmokeTestError("Artifact store smoke test failed during read")
    except ArtifactStoreSmokeTestError:
        raise
    except Exception as exc:
        raise ArtifactStoreSmokeTestError("Artifact store smoke test failed during read") from exc
    finally:
        try:
            artifact_store.delete(organization_id, storage_key)
        except Exception:
            pass

    try:
        if artifact_store.exists(organization_id, storage_key):
            raise ArtifactStoreSmokeTestError("Artifact store smoke test failed during delete")
    except ArtifactStoreSmokeTestError:
        raise
    except Exception as exc:
        raise ArtifactStoreSmokeTestError("Artifact store smoke test failed during delete") from exc

    return ArtifactStoreSmokeTestResult(
        tested_at=tested_at,
        bytes_written=len(payload),
    )


class InMemoryTenantArtifactStore:
    def __init__(self) -> None:
        self._objects: dict[tuple[str, str], bytes] = {}

    def put_text(self, organization_id: str, storage_key: str, payload: str) -> None:
        self.put_bytes(organization_id, storage_key, payload.encode("utf-8"))

    def get_text(self, organization_id: str, storage_key: str) -> str:
        return self.get_bytes(organization_id, storage_key).decode("utf-8")

    def put_bytes(self, organization_id: str, storage_key: str, payload: bytes) -> None:
        validate_tenant_storage_key(organization_id, storage_key)
        self._objects[(safe_key_part(organization_id), storage_key)] = payload

    def get_bytes(self, organization_id: str, storage_key: str) -> bytes:
        validate_tenant_storage_key(organization_id, storage_key)
        try:
            return self._objects[(safe_key_part(organization_id), storage_key)]
        except KeyError as exc:
            raise FileNotFoundError(storage_key) from exc

    def exists(self, organization_id: str, storage_key: str) -> bool:
        validate_tenant_storage_key(organization_id, storage_key)
        return (safe_key_part(organization_id), storage_key) in self._objects

    def delete(self, organization_id: str, storage_key: str) -> bool:
        validate_tenant_storage_key(organization_id, storage_key)
        return self._objects.pop((safe_key_part(organization_id), storage_key), None) is not None


class LocalTenantArtifactStore:
    def __init__(self, base_path: str | Path):
        self.base_path = Path(base_path)

    @classmethod
    def from_env(cls) -> "LocalTenantArtifactStore":
        return cls(os.environ.get(ARTIFACT_STORE_DIR_ENV, ".saas-artifacts"))

    def _path_for(self, organization_id: str, storage_key: str) -> Path:
        parts = validate_tenant_storage_key(organization_id, storage_key)
        base = self.base_path.resolve()
        target = (base / Path(*parts)).resolve()
        try:
            target.relative_to(base)
        except ValueError as exc:
            raise ValueError("Storage key resolved outside artifact store") from exc
        return target

    def put_text(self, organization_id: str, storage_key: str, payload: str) -> None:
        self.put_bytes(organization_id, storage_key, payload.encode("utf-8"))

    def get_text(self, organization_id: str, storage_key: str) -> str:
        return self.get_bytes(organization_id, storage_key).decode("utf-8")

    def put_bytes(self, organization_id: str, storage_key: str, payload: bytes) -> None:
        target = self._path_for(organization_id, storage_key)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)

    def get_bytes(self, organization_id: str, storage_key: str) -> bytes:
        return self._path_for(organization_id, storage_key).read_bytes()

    def exists(self, organization_id: str, storage_key: str) -> bool:
        return self._path_for(organization_id, storage_key).exists()

    def delete(self, organization_id: str, storage_key: str) -> bool:
        target = self._path_for(organization_id, storage_key)
        try:
            target.unlink()
        except FileNotFoundError:
            return False
        return True


class S3TenantArtifactStore:
    def __init__(
        self,
        bucket: str,
        *,
        prefix: str = "",
        client: Any | None = None,
        endpoint_url: str | None = None,
        region_name: str | None = None,
    ):
        if not bucket.strip():
            raise ValueError("S3 artifact bucket is required")
        self.bucket = bucket.strip()
        self.prefix = prefix.strip().strip("/")
        self.client = client or self._create_client(endpoint_url=endpoint_url, region_name=region_name)

    @classmethod
    def from_env(cls) -> "S3TenantArtifactStore":
        bucket = os.environ.get(S3_BUCKET_ENV, "")
        return cls(
            bucket,
            prefix=os.environ.get(S3_PREFIX_ENV, ""),
            endpoint_url=os.environ.get(S3_ENDPOINT_URL_ENV),
            region_name=os.environ.get(S3_REGION_ENV),
        )

    @staticmethod
    def _create_client(endpoint_url: str | None = None, region_name: str | None = None) -> Any:
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError(
                "boto3 is required for SAAS_ARTIFACT_STORE_BACKEND=s3"
            ) from exc
        kwargs: dict[str, str] = {}
        if endpoint_url:
            kwargs["endpoint_url"] = endpoint_url
        if region_name:
            kwargs["region_name"] = region_name
        return boto3.client("s3", **kwargs)

    def _object_key(self, organization_id: str, storage_key: str) -> str:
        validate_tenant_storage_key(organization_id, storage_key)
        if self.prefix:
            return f"{self.prefix}/{storage_key}"
        return storage_key

    def put_text(self, organization_id: str, storage_key: str, payload: str) -> None:
        self.put_bytes(organization_id, storage_key, payload.encode("utf-8"))

    def get_text(self, organization_id: str, storage_key: str) -> str:
        return self.get_bytes(organization_id, storage_key).decode("utf-8")

    def put_bytes(self, organization_id: str, storage_key: str, payload: bytes) -> None:
        self.client.put_object(
            Bucket=self.bucket,
            Key=self._object_key(organization_id, storage_key),
            Body=payload,
        )

    def get_bytes(self, organization_id: str, storage_key: str) -> bytes:
        try:
            response = self.client.get_object(
                Bucket=self.bucket,
                Key=self._object_key(organization_id, storage_key),
            )
        except Exception as exc:
            raise FileNotFoundError(storage_key) from exc
        return response["Body"].read()

    def exists(self, organization_id: str, storage_key: str) -> bool:
        try:
            self.client.head_object(
                Bucket=self.bucket,
                Key=self._object_key(organization_id, storage_key),
            )
        except Exception:
            return False
        return True

    def delete(self, organization_id: str, storage_key: str) -> bool:
        object_key = self._object_key(organization_id, storage_key)
        existed = self.exists(organization_id, storage_key)
        self.client.delete_object(Bucket=self.bucket, Key=object_key)
        return existed


def artifact_store_from_env() -> TenantArtifactStore:
    backend = os.environ.get(ARTIFACT_STORE_BACKEND_ENV, "local").strip().lower()
    if backend == "local":
        return LocalTenantArtifactStore.from_env()
    if backend == "memory":
        return InMemoryTenantArtifactStore()
    if backend == "s3":
        return S3TenantArtifactStore.from_env()
    raise ValueError(f"{ARTIFACT_STORE_BACKEND_ENV} must be one of: local, memory, s3")
