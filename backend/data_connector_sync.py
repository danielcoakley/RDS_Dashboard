from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import base64
from typing import Any, Protocol

from .api import (
    DataConnectorImportCreate,
    DataConnectorSyncRecord,
    import_data_connector_content_summary,
    record_data_connector_sync_summary,
)
from .artifacts import TenantArtifactStore
from .store import SaaSStore


@dataclass(frozen=True)
class DataConnectorFetchResult:
    filename: str
    content: bytes
    records_processed: int = 0
    status_reason: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class DataConnectorProvider(Protocol):
    def fetch(self, connector: Any) -> DataConnectorFetchResult:
        ...


@dataclass(frozen=True)
class DataConnectorWorkerSummary:
    attempted: int
    succeeded: int
    failed: int
    skipped: int = 0


def _connector_actor_user_id(connector: Any) -> str:
    return connector["updated_by_user_id"] or connector["created_by_user_id"]


def _isoformat_or_none(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def process_due_data_connectors(
    store: SaaSStore,
    artifact_store: TenantArtifactStore,
    provider: DataConnectorProvider,
    *,
    organization_id: str | None = None,
    limit: int | None = None,
    now: datetime | None = None,
) -> DataConnectorWorkerSummary:
    effective_now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    due_connectors = store.list_due_data_connectors(
        effective_now.isoformat(),
        organization_id=organization_id,
        limit=limit,
    )
    succeeded = 0
    failed = 0
    skipped = 0

    for connector in due_connectors:
        actor_user_id = _connector_actor_user_id(connector)
        try:
            fetched = provider.fetch(connector)
            if not fetched.content:
                raise ValueError("Connector provider returned empty content")
            import_data_connector_content_summary(
                actor_user_id,
                connector["organization_id"],
                connector["id"],
                DataConnectorImportCreate(
                    filename=fetched.filename,
                    content_base64=base64.b64encode(fetched.content).decode("ascii"),
                    records_processed=fetched.records_processed,
                    status_reason=fetched.status_reason,
                    started_at=_isoformat_or_none(fetched.started_at),
                    completed_at=_isoformat_or_none(fetched.completed_at) or effective_now.isoformat(),
                ),
                store,
                artifact_store,
            )
        except Exception:
            failed += 1
            try:
                record_data_connector_sync_summary(
                    actor_user_id,
                    connector["organization_id"],
                    connector["id"],
                    DataConnectorSyncRecord(
                        status="failed",
                        records_processed=0,
                        status_reason="Connector provider import failed",
                        completed_at=effective_now.isoformat(),
                    ),
                    store,
                )
            except Exception:
                skipped += 1
            continue
        succeeded += 1

    return DataConnectorWorkerSummary(
        attempted=len(due_connectors),
        succeeded=succeeded,
        failed=failed,
        skipped=skipped,
    )
