from datetime import datetime, timezone

from backend.artifacts import InMemoryTenantArtifactStore
from backend.data_connector_sync import DataConnectorFetchResult, process_due_data_connectors
from backend.database import initialize_database
from backend.domain import (
    DataConnector,
    DataConnectorSchedule,
    DataConnectorStatus,
    DataConnectorType,
    Membership,
    Organization,
    Role,
    Site,
    Upload,
    UploadStatus,
    User,
)
from backend.notifications import InMemoryEmailSender
from backend.run_orchestration import StoredUploadAnalysisRunRequest, queue_analysis_run_from_upload
from backend.store import SaaSStore
from backend.worker_app import AnalysisWorkerConfig, process_worker_batch
from backend.workers import process_queued_analysis_jobs
from analytics.config_loader import load_client_config


class StaticDataConnectorProvider:
    def __init__(self, result: DataConnectorFetchResult | Exception) -> None:
        self.result = result
        self.connector_ids: list[str] = []

    def fetch(self, connector):
        self.connector_ids.append(connector["id"])
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def seed_worker_tenant(store: SaaSStore, user_id: str, organization_id: str, site_id: str) -> None:
    store.create_user(User(id=user_id, email=f"{user_id}@example.com", display_name=user_id))
    store.create_organization(
        Organization(id=organization_id, name=f"{organization_id} Energy", slug=organization_id)
    )
    store.add_membership(Membership(user_id=user_id, organization_id=organization_id, role=Role.OWNER))
    store.create_site(Site(id=site_id, organization_id=organization_id, name=site_id, timezone="Europe/London"))


def add_energy_upload(
    store: SaaSStore,
    artifact_store: InMemoryTenantArtifactStore,
    user_id: str,
    organization_id: str,
    site_id: str,
    upload_id: str,
    *,
    write_artifact: bool = True,
) -> None:
    storage_key = f"tenants/{organization_id}/sites/{site_id}/uploads/{upload_id}/energy.csv"
    if write_artifact:
        artifact_store.put_text(
            organization_id,
            storage_key,
            "Date,Main Electricity,Main Gas\n2025-01-01,100,80\n2025-02-01,120,70\n",
        )
    store.create_upload(
        Upload(
            id=upload_id,
            organization_id=organization_id,
            site_id=site_id,
            uploaded_by_user_id=user_id,
            category="energy",
            storage_key=storage_key,
            checksum="abc123",
            status=UploadStatus.STORED,
        )
    )


def queue_job(
    store: SaaSStore,
    user_id: str,
    organization_id: str,
    upload_id: str,
    run_id: str,
    report_id: str,
) -> None:
    queue_analysis_run_from_upload(
        store,
        StoredUploadAnalysisRunRequest(
            user_id=user_id,
            organization_id=organization_id,
            upload_id=upload_id,
            run_id=run_id,
            report_id=report_id,
            client_config=load_client_config("config/clients/example_client.yaml"),
        ),
    )


def add_data_connector(
    store: SaaSStore,
    user_id: str,
    organization_id: str,
    site_id: str,
    connector_id: str,
    *,
    next_sync_at: datetime,
    status: DataConnectorStatus = DataConnectorStatus.ACTIVE,
) -> None:
    store.create_data_connector(
        DataConnector(
            id=connector_id,
            organization_id=organization_id,
            site_id=site_id,
            name=f"{connector_id} utility API",
            connector_type=DataConnectorType.API,
            source_category="energy",
            status=status,
            schedule=DataConnectorSchedule.DAILY,
            external_reference=f"api://{organization_id}/{connector_id}",
            credential_reference=f"secret/data-connectors/{organization_id}/{connector_id}",
            created_by_user_id=user_id,
            updated_by_user_id=user_id,
            next_sync_at=next_sync_at,
        )
    )


def test_worker_processes_only_selected_tenant_jobs_and_sends_notifications():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    sender = InMemoryEmailSender()
    seed_worker_tenant(store, "user_1", "org_1", "site_1")
    seed_worker_tenant(store, "user_2", "org_2", "site_2")
    add_energy_upload(store, artifact_store, "user_1", "org_1", "site_1", "upload_1")
    add_energy_upload(store, artifact_store, "user_2", "org_2", "site_2", "upload_2")
    queue_job(store, "user_1", "org_1", "upload_1", "run_1", "report_1")
    queue_job(store, "user_2", "org_2", "upload_2", "run_2", "report_2")

    summary = process_queued_analysis_jobs(
        store,
        artifact_store,
        email_sender=sender,
        organization_id="org_1",
    )

    assert summary.attempted == 1
    assert summary.succeeded == 1
    assert summary.failed == 0
    assert store.list_analysis_jobs("org_1")[0]["status"] == "succeeded"
    assert store.list_analysis_jobs("org_2")[0]["status"] == "queued"
    assert [message.organization_id for message in sender.sent_messages] == ["org_1"]
    assert store.list_notifications("org_1", recipient_user_id="user_1")[0]["status"] == "sent"


def test_worker_processes_due_data_connectors_for_selected_tenant():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    sender = InMemoryEmailSender()
    seed_worker_tenant(store, "user_1", "org_1", "site_1")
    seed_worker_tenant(store, "user_2", "org_2", "site_2")
    add_data_connector(
        store,
        "user_1",
        "org_1",
        "site_1",
        "connector_1",
        next_sync_at=datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc),
    )
    add_data_connector(
        store,
        "user_2",
        "org_2",
        "site_2",
        "connector_2",
        next_sync_at=datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc),
    )
    provider = StaticDataConnectorProvider(
        DataConnectorFetchResult(
            filename="energy.csv",
            content=b"Date,Main Electricity\n2026-01-01,100\n",
            records_processed=1,
            status_reason="Fetched provider export",
            started_at=datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc),
            completed_at=datetime(2026, 1, 1, 9, 1, tzinfo=timezone.utc),
        )
    )

    summary = process_worker_batch(
        AnalysisWorkerConfig(batch_size=10, organization_id="org_1", worker_id="worker_1"),
        store=store,
        artifact_store=artifact_store,
        email_sender=sender,
        data_connector_provider=provider,
    )

    org_1_uploads = store.list_uploads("org_1")
    org_1_syncs = store.list_data_connector_syncs("org_1", "connector_1")
    org_2_syncs = store.list_data_connector_syncs("org_2", "connector_2")
    connector = store.get_data_connector("connector_1")
    heartbeat = store.get_latest_worker_heartbeat("org_1")

    assert summary.connector_attempted == 1
    assert summary.connector_succeeded == 1
    assert summary.connector_failed == 0
    assert provider.connector_ids == ["connector_1"]
    assert len(org_1_uploads) == 1
    assert artifact_store.get_text("org_1", org_1_uploads[0]["storage_key"]) == (
        "Date,Main Electricity\n2026-01-01,100\n"
    )
    assert org_1_syncs[0]["status"] == "succeeded"
    assert org_1_syncs[0]["upload_id"] == org_1_uploads[0]["id"]
    assert org_2_syncs == []
    assert connector["last_sync_status"] == "succeeded"
    assert connector["next_sync_at"] == "2026-01-02T09:01:00+00:00"
    assert heartbeat["last_connector_attempted"] == 1
    assert heartbeat["last_connector_succeeded"] == 1
    assert heartbeat["last_connector_failed"] == 0


def test_due_data_connector_provider_failure_records_sanitized_failed_sync():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    seed_worker_tenant(store, "user_1", "org_1", "site_1")
    add_data_connector(
        store,
        "user_1",
        "org_1",
        "site_1",
        "connector_1",
        next_sync_at=datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc),
    )
    provider = StaticDataConnectorProvider(RuntimeError("token=secret-provider-error"))

    summary = process_due_data_connectors(
        store,
        artifact_store,
        provider,
        organization_id="org_1",
        now=datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc),
    )

    syncs = store.list_data_connector_syncs("org_1", "connector_1")
    connector = store.get_data_connector("connector_1")
    audit_event = store.list_audit_events("org_1")[-1]

    assert summary.attempted == 1
    assert summary.succeeded == 0
    assert summary.failed == 1
    assert summary.skipped == 0
    assert store.list_uploads("org_1") == []
    assert syncs[0]["status"] == "failed"
    assert syncs[0]["status_reason"] == "Connector provider import failed"
    assert connector["last_sync_status"] == "failed"
    assert connector["last_sync_error"] == "Connector provider import failed"
    assert connector["next_sync_at"] == "2026-01-02T10:00:00+00:00"
    assert audit_event["action"] == "data_connector_sync_recorded"
    assert "secret-provider-error" not in audit_event["metadata_json"]


def test_worker_respects_limit_when_draining_queue():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    seed_worker_tenant(store, "user_1", "org_1", "site_1")
    add_energy_upload(store, artifact_store, "user_1", "org_1", "site_1", "upload_1")
    add_energy_upload(store, artifact_store, "user_1", "org_1", "site_1", "upload_2")
    queue_job(store, "user_1", "org_1", "upload_1", "run_1", "report_1")
    queue_job(store, "user_1", "org_1", "upload_2", "run_2", "report_2")

    summary = process_queued_analysis_jobs(store, artifact_store, organization_id="org_1", limit=1)

    statuses = [job["status"] for job in store.list_analysis_jobs("org_1")]
    assert summary.attempted == 1
    assert summary.succeeded == 1
    assert statuses.count("succeeded") == 1
    assert statuses.count("queued") == 1


def test_worker_marks_failed_jobs_and_continues_processing():
    store = SaaSStore(initialize_database())
    artifact_store = InMemoryTenantArtifactStore()
    sender = InMemoryEmailSender()
    seed_worker_tenant(store, "user_1", "org_1", "site_1")
    add_energy_upload(
        store,
        artifact_store,
        "user_1",
        "org_1",
        "site_1",
        "upload_missing",
        write_artifact=False,
    )
    add_energy_upload(store, artifact_store, "user_1", "org_1", "site_1", "upload_ok")
    queue_job(store, "user_1", "org_1", "upload_missing", "run_missing", "report_missing")
    queue_job(store, "user_1", "org_1", "upload_ok", "run_ok", "report_ok")

    summary = process_queued_analysis_jobs(
        store,
        artifact_store,
        email_sender=sender,
        organization_id="org_1",
    )

    jobs = {job["id"]: job for job in store.list_analysis_jobs("org_1")}
    runs = {run["id"]: run for run in store.list_runs("org_1")}
    assert summary.attempted == 2
    assert summary.succeeded == 1
    assert summary.failed == 1
    assert jobs["job_run_missing"]["status"] == "failed"
    assert jobs["job_run_ok"]["status"] == "succeeded"
    assert runs["run_missing"]["status"] == "failed"
    assert runs["run_ok"]["status"] == "succeeded"
    assert [message.organization_id for message in sender.sent_messages] == ["org_1"]
