from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import pandas as pd

from analytics.service import AnalyticsRunResult, run_meter_analysis

from .access_control import require_permission
from .domain import Report, Run, RunStatus, Upload, UploadStatus
from .rbac import Action
from .storage_keys import report_storage_key, upload_storage_key
from .store import SaaSStore


@dataclass(frozen=True)
class AnalysisRunRequest:
    user_id: str
    organization_id: str
    site_id: str
    upload_id: str
    run_id: str
    report_id: str
    filename: str
    raw_meter_data: pd.DataFrame
    client_config: dict


@dataclass(frozen=True)
class AnalysisRunOutcome:
    upload: Upload
    run: Run
    report: Report
    analytics: AnalyticsRunResult


def _dataframe_checksum(df: pd.DataFrame) -> str:
    csv_payload = df.to_csv(index=False).encode("utf-8")
    return sha256(csv_payload).hexdigest()


def execute_analysis_run(store: SaaSStore, request: AnalysisRunRequest) -> AnalysisRunOutcome:
    require_permission(
        store.list_memberships(user_id=request.user_id, organization_id=request.organization_id),
        user_id=request.user_id,
        organization_id=request.organization_id,
        action=Action.MANAGE_RUNS,
    )

    upload = Upload(
        id=request.upload_id,
        organization_id=request.organization_id,
        site_id=request.site_id,
        uploaded_by_user_id=request.user_id,
        category="energy",
        storage_key=upload_storage_key(
            request.organization_id,
            request.site_id,
            request.upload_id,
            request.filename,
        ),
        checksum=_dataframe_checksum(request.raw_meter_data),
        status=UploadStatus.STORED,
    )
    run = Run(
        id=request.run_id,
        organization_id=request.organization_id,
        site_id=request.site_id,
        requested_by_user_id=request.user_id,
        upload_ids=[request.upload_id],
        status=RunStatus.QUEUED,
    )

    with store.conn:
        store.create_upload(upload)
        store.create_run(run)
        store.update_run_status(request.run_id, RunStatus.RUNNING)

    try:
        analytics = run_meter_analysis(request.raw_meter_data, request.client_config)
    except Exception as exc:
        with store.conn:
            store.update_run_status(request.run_id, RunStatus.FAILED, error_message=str(exc))
        raise

    report = Report(
        id=request.report_id,
        organization_id=request.organization_id,
        run_id=request.run_id,
        report_type="iso_summary",
        storage_key=report_storage_key(
            request.organization_id,
            request.site_id,
            request.run_id,
            "iso-summary.json",
        ),
        is_published=True,
    )
    with store.conn:
        store.update_run_status(request.run_id, RunStatus.SUCCEEDED)
        store.create_report(report)

    return AnalysisRunOutcome(upload=upload, run=run, report=report, analytics=analytics)
