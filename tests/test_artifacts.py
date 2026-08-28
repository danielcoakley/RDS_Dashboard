from datetime import datetime, timezone

import pytest

from backend.artifacts import (
    ARTIFACT_STORE_BACKEND_ENV,
    ARTIFACT_STORE_GOVERNANCE_MAX_AGE_DAYS_ENV,
    ARTIFACT_STORE_GOVERNANCE_URL_ENV,
    ARTIFACT_STORE_GOVERNANCE_VERIFIED_AT_ENV,
    ArtifactStoreSmokeTestError,
    InMemoryTenantArtifactStore,
    LocalTenantArtifactStore,
    S3TenantArtifactStore,
    artifact_store_from_env,
    artifact_store_governance_status,
    artifact_store_smoke_test,
    validate_tenant_storage_key,
)


class FakeS3Body:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return self.payload


class FakeS3Client:
    def __init__(self):
        self.objects = {}

    def put_object(self, Bucket, Key, Body):
        self.objects[(Bucket, Key)] = Body

    def get_object(self, Bucket, Key):
        return {"Body": FakeS3Body(self.objects[(Bucket, Key)])}

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise KeyError(Key)
        return {}

    def delete_object(self, Bucket, Key):
        self.objects.pop((Bucket, Key), None)
        return {}


def test_validate_tenant_storage_key_requires_matching_tenant_prefix():
    parts = validate_tenant_storage_key(
        "Org 1",
        "tenants/org-1/sites/main-site/runs/run-1/reports/iso-summary.json",
    )

    assert parts[:2] == ("tenants", "org-1")


def test_validate_tenant_storage_key_rejects_cross_tenant_key():
    try:
        validate_tenant_storage_key(
            "org_1",
            "tenants/org_2/sites/site_1/uploads/upload_1/energy.csv",
        )
        assert False, "Expected cross-tenant artifact key to be rejected"
    except ValueError as exc:
        assert "tenants/org_1" in str(exc)


def test_validate_tenant_storage_key_rejects_traversal_segments():
    try:
        validate_tenant_storage_key(
            "org_1",
            "tenants/org_1/sites/site_1/../upload.csv",
        )
        assert False, "Expected traversal segment to be rejected"
    except ValueError as exc:
        assert "traversal" in str(exc)


def test_in_memory_artifact_store_round_trips_tenant_content():
    store = InMemoryTenantArtifactStore()
    key = "tenants/org_1/sites/site_1/runs/run_1/reports/iso-summary.json"

    store.put_text("org_1", key, "{\"total_records\": 4}")

    assert store.exists("org_1", key)
    assert store.get_text("org_1", key) == "{\"total_records\": 4}"
    assert store.delete("org_1", key)
    assert not store.exists("org_1", key)
    assert not store.delete("org_1", key)


def test_artifact_store_smoke_test_round_trips_and_removes_probe_object():
    store = InMemoryTenantArtifactStore()

    result = artifact_store_smoke_test(
        store,
        test_id="probe-1",
        now=datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc),
    )

    assert result.tested_at == "2026-07-20T12:00:00+00:00"
    assert result.bytes_written > 0
    assert store._objects == {}


def test_artifact_store_smoke_test_reports_generic_read_failures():
    class MismatchedReadArtifactStore(InMemoryTenantArtifactStore):
        def get_bytes(self, organization_id: str, storage_key: str) -> bytes:
            super().get_bytes(organization_id, storage_key)
            return b"wrong payload"

    store = MismatchedReadArtifactStore()

    with pytest.raises(ArtifactStoreSmokeTestError) as exc:
        artifact_store_smoke_test(store, test_id="probe-1")

    assert str(exc.value) == "Artifact store smoke test failed during read"
    assert "probe-1" not in str(exc.value)
    assert "tenants/" not in str(exc.value)
    assert store._objects == {}


def test_artifact_store_smoke_test_reports_generic_write_failures():
    class FailingWriteArtifactStore(InMemoryTenantArtifactStore):
        def put_bytes(self, organization_id: str, storage_key: str, payload: bytes) -> None:
            raise RuntimeError("s3 credentials include secret details")

    with pytest.raises(ArtifactStoreSmokeTestError) as exc:
        artifact_store_smoke_test(FailingWriteArtifactStore(), test_id="probe-1")

    assert str(exc.value) == "Artifact store smoke test failed during write"
    assert "secret" not in str(exc.value)
    assert "probe-1" not in str(exc.value)


def test_artifact_store_governance_status_is_optional_in_development(monkeypatch):
    monkeypatch.delenv(ARTIFACT_STORE_GOVERNANCE_URL_ENV, raising=False)
    monkeypatch.delenv(ARTIFACT_STORE_GOVERNANCE_VERIFIED_AT_ENV, raising=False)

    configured, detail = artifact_store_governance_status("development")

    assert configured is True
    assert "optional" in detail


def test_artifact_store_governance_status_accepts_fresh_production_evidence(monkeypatch):
    monkeypatch.setenv(
        ARTIFACT_STORE_GOVERNANCE_URL_ENV,
        "https://ops.example.com/artifacts/governance",
    )
    monkeypatch.setenv(
        ARTIFACT_STORE_GOVERNANCE_VERIFIED_AT_ENV,
        "2026-07-20T12:00:00+00:00",
    )
    monkeypatch.setenv(ARTIFACT_STORE_GOVERNANCE_MAX_AGE_DAYS_ENV, "30")

    configured, detail = artifact_store_governance_status(
        "production",
        now=datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc),
    )

    assert configured is True
    assert ARTIFACT_STORE_GOVERNANCE_URL_ENV in detail
    assert "age 1 day(s)" in detail


def test_artifact_store_governance_status_rejects_stale_production_evidence(monkeypatch):
    monkeypatch.setenv(
        ARTIFACT_STORE_GOVERNANCE_URL_ENV,
        "https://ops.example.com/artifacts/governance",
    )
    monkeypatch.setenv(
        ARTIFACT_STORE_GOVERNANCE_VERIFIED_AT_ENV,
        "2026-07-12T12:00:00+00:00",
    )
    monkeypatch.setenv(ARTIFACT_STORE_GOVERNANCE_MAX_AGE_DAYS_ENV, "7")

    configured, detail = artifact_store_governance_status(
        "production",
        now=datetime(2026, 7, 21, 12, 0, tzinfo=timezone.utc),
    )

    assert configured is False
    assert "older than 7 days" in detail


def test_local_artifact_store_writes_inside_base_path(tmp_path):
    store = LocalTenantArtifactStore(tmp_path)
    key = "tenants/org_1/sites/site_1/uploads/upload_1/energy.csv"

    store.put_text("org_1", key, "Date,Main Electricity\n2025-01-01,100\n")

    target = tmp_path / "tenants" / "org_1" / "sites" / "site_1" / "uploads" / "upload_1" / "energy.csv"
    assert target.read_text(encoding="utf-8") == "Date,Main Electricity\n2025-01-01,100\n"
    assert store.delete("org_1", key)
    assert not target.exists()
    assert not store.delete("org_1", key)


def test_s3_artifact_store_round_trips_tenant_content_with_prefix():
    client = FakeS3Client()
    store = S3TenantArtifactStore("rds-artifacts", prefix="prod", client=client)
    key = "tenants/org_1/sites/site_1/uploads/upload_1/energy.csv"

    store.put_text("org_1", key, "Date,Main Electricity\n2025-01-01,100\n")

    assert store.exists("org_1", key)
    assert store.get_text("org_1", key) == "Date,Main Electricity\n2025-01-01,100\n"
    assert ("rds-artifacts", f"prod/{key}") in client.objects
    assert store.delete("org_1", key)
    assert not store.exists("org_1", key)
    assert ("rds-artifacts", f"prod/{key}") not in client.objects
    assert not store.delete("org_1", key)


def test_s3_artifact_store_rejects_cross_tenant_key():
    store = S3TenantArtifactStore("rds-artifacts", client=FakeS3Client())

    try:
        store.put_text("org_1", "tenants/org_2/sites/site_1/uploads/upload_1/energy.csv", "bad")
        assert False, "Expected cross-tenant S3 key to be rejected"
    except ValueError as exc:
        assert "tenants/org_1" in str(exc)


def test_artifact_store_from_env_defaults_to_local(monkeypatch, tmp_path):
    monkeypatch.delenv(ARTIFACT_STORE_BACKEND_ENV, raising=False)
    monkeypatch.setenv("SAAS_ARTIFACT_STORE_DIR", str(tmp_path))

    store = artifact_store_from_env()

    assert isinstance(store, LocalTenantArtifactStore)
    assert store.base_path == tmp_path


def test_artifact_store_from_env_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv(ARTIFACT_STORE_BACKEND_ENV, "shoebox")

    try:
        artifact_store_from_env()
        assert False, "Expected unknown artifact backend to fail"
    except ValueError as exc:
        assert ARTIFACT_STORE_BACKEND_ENV in str(exc)
