from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .artifacts import TenantArtifactStore
from .domain import RunStatus
from .notifications import EmailSender, process_pending_notifications
from .run_orchestration import process_analysis_job
from .store import SaaSStore

if TYPE_CHECKING:
    from .data_connector_sync import DataConnectorProvider


@dataclass(frozen=True)
class AnalysisWorkerSummary:
    attempted: int
    succeeded: int
    failed: int
    skipped: int
    scheduled_attempted: int = 0
    scheduled_sent: int = 0
    scheduled_failed: int = 0
    scheduled_skipped: int = 0
    connector_attempted: int = 0
    connector_succeeded: int = 0
    connector_failed: int = 0
    connector_skipped: int = 0


def process_queued_analysis_jobs(
    store: SaaSStore,
    artifact_store: TenantArtifactStore,
    email_sender: EmailSender | None = None,
    organization_id: str | None = None,
    limit: int | None = None,
) -> AnalysisWorkerSummary:
    queued_jobs = store.list_queued_analysis_jobs(organization_id=organization_id, limit=limit)
    succeeded = 0
    failed = 0
    skipped = 0

    for job in queued_jobs:
        with store.conn:
            claimed = store.claim_analysis_job(job["id"])
        if not claimed:
            skipped += 1
            continue

        try:
            process_analysis_job(store, job["id"], artifact_store)
        except Exception as exc:
            with store.conn:
                store.update_run_status(job["run_id"], RunStatus.FAILED, error_message=str(exc))
                store.update_analysis_job_status(job["id"], RunStatus.FAILED, error_message=str(exc))
            failed += 1
        else:
            succeeded += 1
        finally:
            if email_sender is not None:
                process_pending_notifications(
                    store,
                    email_sender,
                    organization_id=job["organization_id"],
                )

    return AnalysisWorkerSummary(
        attempted=len(queued_jobs),
        succeeded=succeeded,
        failed=failed,
        skipped=skipped,
    )


def process_due_data_connector_imports(
    store: SaaSStore,
    artifact_store: TenantArtifactStore,
    provider: DataConnectorProvider | None = None,
    organization_id: str | None = None,
    limit: int | None = None,
) -> AnalysisWorkerSummary:
    if provider is None:
        return AnalysisWorkerSummary(attempted=0, succeeded=0, failed=0, skipped=0)
    from .data_connector_sync import process_due_data_connectors

    connector_summary = process_due_data_connectors(
        store,
        artifact_store,
        provider,
        organization_id=organization_id,
        limit=limit,
    )
    return AnalysisWorkerSummary(
        attempted=0,
        succeeded=0,
        failed=0,
        skipped=0,
        connector_attempted=connector_summary.attempted,
        connector_succeeded=connector_summary.succeeded,
        connector_failed=connector_summary.failed,
        connector_skipped=connector_summary.skipped,
    )
