from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO, StringIO
import json
from typing import Any

import pandas as pd

from analytics.legacy_source_package import build_weather_normalized_summary
from analytics.service import AnalyticsRunResult, run_meter_analysis

from .access_control import require_permission, require_resource_scope
from .analysis_configs import resolve_analysis_config_for_run, weather_model_settings_from_config
from .artifacts import InMemoryTenantArtifactStore, TenantArtifactStore
from .domain import (
    AnalysisJob,
    AuditAction,
    AuditEvent,
    Notification,
    NotificationChannel,
    NotificationStatus,
    Report,
    Run,
    RunStatus,
    Upload,
    UploadCategory,
    UploadStatus,
)
from .rbac import Action
from .retention import report_expires_at_for_organization
from .storage_keys import report_storage_key, upload_storage_key
from .store import SaaSStore
from .subscriptions import require_run_capacity


SOURCE_CATEGORY_ORDER = {
    UploadCategory.ENERGY.value: 0,
    UploadCategory.HDD.value: 1,
    UploadCategory.CDD.value: 2,
    UploadCategory.SEU_MAPPING.value: 3,
}
INTERNAL_REPORT_ARTIFACT_KEYS = {
    "analysis_input_path",
    "artifact_path",
    "report_storage_key",
    "storage_key",
    "upload_storage_key",
}
PUBLIC_SOURCE_CONTEXT_KEYS = (
    "execution_input",
    "resolved_meter_columns",
    "resolved_record_count",
)
PUBLIC_SOURCE_PACKAGE_KEYS = (
    "primary_upload_id",
    "site_id",
    "upload_count",
    "categories",
)
PUBLIC_SOURCE_PACKAGE_UPLOAD_KEYS = (
    "id",
    "category",
    "checksum",
    "status",
    "is_primary",
)
PUBLIC_SOURCE_SUMMARY_KEYS = (
    "upload_id",
    "category",
    "is_primary",
    "parse_status",
    "error",
    "shape",
    "row_count",
    "meter_column_count",
    "meter_columns",
    "value_columns",
    "mapping_count",
    "seu_categories",
    "start_date",
    "end_date",
)


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
    client_config: dict[str, Any] | None = None


@dataclass(frozen=True)
class StoredUploadAnalysisRunRequest:
    user_id: str
    organization_id: str
    upload_id: str
    run_id: str
    report_id: str
    source_upload_ids: tuple[str, ...] = ()
    client_config: dict[str, Any] | None = None


@dataclass(frozen=True)
class AnalysisRunOutcome:
    upload: Upload
    run: Run
    report: Report
    analytics: AnalyticsRunResult


@dataclass(frozen=True)
class QueuedAnalysisRun:
    job: AnalysisJob
    run: Run
    upload: Upload


@dataclass(frozen=True)
class AnalysisInput:
    raw_meter_data: pd.DataFrame
    source_context: dict[str, Any]
    legacy_baseline: dict[str, Any] | None = None


def _dataframe_checksum(df: pd.DataFrame) -> str:
    csv_payload = df.to_csv(index=False).encode("utf-8")
    return sha256(csv_payload).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _upload_from_row(row: Any) -> Upload:
    return Upload(
        id=row["id"],
        organization_id=row["organization_id"],
        site_id=row["site_id"],
        uploaded_by_user_id=row["uploaded_by_user_id"],
        category=row["category"],
        storage_key=row["storage_key"],
        checksum=row["checksum"],
        status=UploadStatus(row["status"]),
    )


def _read_csv_payload(payload: bytes) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "latin1"):
        try:
            return pd.read_csv(BytesIO(payload), encoding=encoding)
        except Exception as exc:  # pandas can raise several parser/codec errors here
            last_error = exc
    raise ValueError(f"Unable to read upload as CSV: {last_error}") from last_error


def _decode_csv_payload(payload: bytes) -> str:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "latin1"):
        try:
            return payload.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError(f"Unable to decode upload as CSV: {last_error}") from last_error


def _csv_key(value: Any) -> str:
    normalized = []
    previous_separator = False
    for character in str(value).strip().lower():
        if character.isalnum():
            normalized.append(character)
            previous_separator = False
        elif not previous_separator:
            normalized.append("_")
            previous_separator = True
    return "".join(normalized).strip("_")


def _csv_rows_from_payload(payload: bytes) -> list[list[str]]:
    try:
        return list(csv.reader(StringIO(_decode_csv_payload(payload))))
    except csv.Error as exc:
        raise ValueError(f"Unable to read upload as CSV: {exc}") from exc


def _header_positions(row: list[str]) -> dict[str, int]:
    return {_csv_key(column): index for index, column in enumerate(row)}


def _read_tabular_csv_payload(
    payload: bytes,
    header_predicate: Any,
    description: str,
) -> pd.DataFrame:
    rows = _csv_rows_from_payload(payload)
    for index, row in enumerate(rows):
        if not header_predicate(row):
            continue
        width = len(row)
        data_rows = []
        for data_row in rows[index + 1 :]:
            if not any(str(value).strip() for value in data_row):
                continue
            padded = [*data_row, *([""] * max(0, width - len(data_row)))]
            data_rows.append(padded[:width])
        return pd.DataFrame(data_rows, columns=row)
    raise ValueError(f"{description} CSV did not include the expected header row")


def _normalise_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    columns = []
    for column in normalized.columns:
        name = str(column)
        for bom in ("\ufeff", "Ã¯Â»Â¿"):
            if name.startswith(bom):
                name = name[len(bom) :]
        columns.append(name.strip().strip("'").strip('"'))
    normalized.columns = columns
    return normalized


def _configured_source_columns(client_config: dict[str, Any]) -> list[str]:
    return [
        meter["source_column"]
        for meter in client_config.get("meters", [])
        if isinstance(meter, dict) and meter.get("source_column")
    ]


def _find_timestamp_column(df: pd.DataFrame) -> str | None:
    normalized_columns = {str(column).strip().lower(): str(column) for column in df.columns}
    for candidate in ("date", "timestamp", "datetime", "time"):
        if candidate in normalized_columns:
            return normalized_columns[candidate]
    return None


def _missing_meter_source_columns(df: pd.DataFrame, client_config: dict[str, Any]) -> list[str]:
    available = {str(column).strip() for column in df.columns}
    return [column for column in _configured_source_columns(client_config) if column not in available]


def _is_legacy_energy_export(df: pd.DataFrame) -> bool:
    columns = {str(column).strip().lower() for column in df.columns}
    return {"metered sector", "utility", "units", "period"}.issubset(columns)


def _date_columns(columns: list[str]) -> list[str]:
    return [
        column
        for column in columns
        if not pd.isna(pd.to_datetime(column, errors="coerce", dayfirst=True))
    ]


def _legacy_energy_export_to_long_dataframe(energy_df: pd.DataFrame) -> pd.DataFrame:
    energy = _normalise_dataframe_columns(energy_df)
    required_columns = ["Metered Sector", "Utility", "Units", "Period"]
    missing = [column for column in required_columns if column not in energy.columns]
    if missing:
        raise ValueError(
            "Legacy energy export is missing required columns: " + ", ".join(missing)
        )
    date_columns = _date_columns([str(column) for column in energy.columns])
    if not date_columns:
        raise ValueError("Legacy energy export does not include dated consumption columns")

    records: list[dict[str, Any]] = []
    previous_meter: str | None = None
    previous_utility: str | None = None
    previous_units: str | None = None
    for _index, row in energy.iterrows():
        meter_value = row.get("Metered Sector")
        if isinstance(meter_value, str) and meter_value.strip():
            previous_meter = meter_value.strip()
        utility_value = row.get("Utility")
        if isinstance(utility_value, str) and utility_value.strip():
            previous_utility = utility_value.strip()
        units_value = row.get("Units")
        if isinstance(units_value, str) and units_value.strip():
            previous_units = units_value.strip()
        if str(row.get("Period", "")).strip().lower() != "day" or previous_meter is None:
            continue
        for date_column in date_columns:
            timestamp = pd.to_datetime(date_column, errors="coerce", dayfirst=True)
            value = pd.to_numeric(row.get(date_column), errors="coerce")
            if pd.isna(timestamp) or pd.isna(value):
                continue
            records.append(
                {
                    "Date": timestamp,
                    "Meter": previous_meter,
                    "Utility": previous_utility or "Unknown",
                    "Units": previous_units or None,
                    "Consumption": value,
                }
            )

    if not records:
        raise ValueError("Legacy energy export does not include daily meter consumption rows")
    return pd.DataFrame(records).sort_values(["Date", "Meter"]).reset_index(drop=True)


def _legacy_energy_export_to_meter_dataframe(energy_df: pd.DataFrame) -> pd.DataFrame:
    long_df = _legacy_energy_export_to_long_dataframe(energy_df)
    wide_df = (
        long_df.pivot_table(
            index="Date",
            columns="Meter",
            values="Consumption",
            aggfunc="sum",
        )
        .reset_index()
        .sort_values("Date")
    )
    wide_df.columns.name = None
    return wide_df


def _date_range_summary(values: pd.Series) -> dict[str, str | None]:
    timestamps = pd.to_datetime(values, errors="coerce", dayfirst=True).dropna()
    if timestamps.empty:
        return {"start_date": None, "end_date": None}
    return {
        "start_date": timestamps.min().date().isoformat(),
        "end_date": timestamps.max().date().isoformat(),
    }


def _column_by_csv_key(df: pd.DataFrame, *keys: str) -> str | None:
    columns_by_key = {_csv_key(column): str(column) for column in df.columns}
    for key in keys:
        if key in columns_by_key:
            return columns_by_key[key]
    return None


def _is_degree_day_value_key(category: str, key: str) -> bool:
    if category == UploadCategory.HDD.value:
        return "hdd" in key or "heating_degree" in key
    if category == UploadCategory.CDD.value:
        return "cdd" in key or "cooling_degree" in key
    return False


def _degree_day_dataframe_from_payload(payload: bytes, category: str) -> pd.DataFrame:
    description = category.upper()

    def _is_header(row: list[str]) -> bool:
        positions = _header_positions(row)
        has_date = any(key in positions for key in ("date", "timestamp", "datetime", "time"))
        has_value = any(_is_degree_day_value_key(category, _csv_key(column)) for column in row)
        return has_date and has_value

    df = _normalise_dataframe_columns(_read_tabular_csv_payload(payload, _is_header, description))
    date_column = _column_by_csv_key(df, "date", "timestamp", "datetime", "time")
    value_column = next(
        (
            column
            for column in df.columns
            if column != date_column and _is_degree_day_value_key(category, _csv_key(column))
        ),
        None,
    )
    if date_column is None or value_column is None:
        raise ValueError(f"{description} CSV must include Date and {description} columns")

    working = df[[date_column, value_column]].copy()
    working["Date"] = pd.to_datetime(working[date_column], errors="coerce", dayfirst=True)
    working[description] = pd.to_numeric(working[value_column], errors="coerce")
    working = working.dropna(subset=["Date", description]).sort_values("Date")
    if working.empty:
        raise ValueError(f"{description} CSV does not include dated numeric values")
    return working[["Date", description]].reset_index(drop=True)


def _seu_mapping_dataframe_from_payload(payload: bytes) -> pd.DataFrame:
    def _is_header(row: list[str]) -> bool:
        positions = _header_positions(row)
        return {"meter", "seu_category"}.issubset(positions)

    df = _normalise_dataframe_columns(
        _read_tabular_csv_payload(payload, _is_header, "SEU mapping")
    )
    meter_column = _column_by_csv_key(df, "meter")
    category_column = _column_by_csv_key(df, "seu_category")
    if meter_column is None or category_column is None:
        raise ValueError("SEU mapping CSV must include Meter and SEU_Category columns")

    mapping = df[[meter_column, category_column]].copy()
    mapping.columns = ["Meter", "SEU_Category"]
    for column in ("Meter", "SEU_Category"):
        mapping[column] = mapping[column].where(mapping[column].notna(), "").astype(str).str.strip()
    mapping = mapping[(mapping["Meter"] != "") & (mapping["SEU_Category"] != "")]
    if mapping.empty:
        raise ValueError("SEU mapping CSV must include at least one meter/category row")
    return mapping.reset_index(drop=True)


def _normalise_utility(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    key = _csv_key(text)
    if "gas" in key:
        return "gas"
    if "elect" in key:
        return "electricity"
    return text.lower() if text else "unknown"


def _grouped_consumption_totals(
    df: pd.DataFrame,
    group_column: str,
    output_key: str,
) -> list[dict[str, Any]]:
    if df.empty or group_column not in df.columns:
        return []
    grouped = (
        df.groupby(group_column, dropna=False)["Consumption"]
        .sum()
        .reset_index()
        .sort_values(group_column)
    )
    records = []
    for _index, row in grouped.iterrows():
        raw_label = row[group_column]
        label = "Unknown" if pd.isna(raw_label) else str(raw_label).strip() or "Unknown"
        records.append({output_key: label, "consumption": float(row["Consumption"])})
    return records


def _normalised_date_values(values: pd.Series) -> set[pd.Timestamp]:
    return set(pd.to_datetime(values, errors="coerce", dayfirst=True).dropna().dt.normalize())


def _matching_date_record_count(df: pd.DataFrame, dates: set[pd.Timestamp]) -> int:
    if df.empty or not dates:
        return 0
    normalized_dates = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True).dt.normalize()
    return int(normalized_dates.isin(dates).sum())


def _weather_metric_summary(df: pd.DataFrame | None) -> dict[str, Any]:
    if df is None:
        return {
            "record_count": 0,
            "start_date": None,
            "end_date": None,
        }
    return {
        "record_count": int(len(df)),
        **_date_range_summary(df["Date"]),
    }


def _upload_by_category(run_uploads: list[Upload]) -> dict[str, Upload]:
    return {upload.category: upload for upload in run_uploads}


def _legacy_baseline_metrics(
    primary_upload: Upload,
    run_uploads: list[Upload],
    payloads: dict[str, bytes],
    client_config: dict[str, Any],
) -> dict[str, Any] | None:
    energy_df = _normalise_dataframe_columns(_read_csv_payload(payloads[primary_upload.id]))
    if not _is_legacy_energy_export(energy_df):
        return None

    energy_records = _legacy_energy_export_to_long_dataframe(energy_df)
    energy_records["Utility"] = energy_records["Utility"].apply(_normalise_utility)
    source_uploads = _upload_by_category(run_uploads)

    hdd_df: pd.DataFrame | None = None
    cdd_df: pd.DataFrame | None = None
    if UploadCategory.HDD.value in source_uploads:
        hdd_upload = source_uploads[UploadCategory.HDD.value]
        hdd_df = _degree_day_dataframe_from_payload(payloads[hdd_upload.id], UploadCategory.HDD.value)
    if UploadCategory.CDD.value in source_uploads:
        cdd_upload = source_uploads[UploadCategory.CDD.value]
        cdd_df = _degree_day_dataframe_from_payload(payloads[cdd_upload.id], UploadCategory.CDD.value)

    active_records = energy_records.copy()
    mapped_meter_count = 0
    ignored_meter_count = 0
    if UploadCategory.SEU_MAPPING.value in source_uploads:
        mapping_upload = source_uploads[UploadCategory.SEU_MAPPING.value]
        mapping_df = _seu_mapping_dataframe_from_payload(payloads[mapping_upload.id])
        ignored_mapping = mapping_df["SEU_Category"].str.casefold() == "ignore"
        ignored_meters = set(mapping_df.loc[ignored_mapping, "Meter"])
        energy_meters = set(active_records["Meter"])
        ignored_meter_count = len(ignored_meters.intersection(energy_meters))
        active_records = active_records[~active_records["Meter"].isin(ignored_meters)].copy()

        active_mapping = mapping_df.loc[~ignored_mapping].drop_duplicates("Meter")
        mapped_meter_count = len(set(active_mapping["Meter"]).intersection(set(active_records["Meter"])))
        active_records = active_records.merge(active_mapping, on="Meter", how="left")
    else:
        active_records["SEU_Category"] = "Unknown"

    active_records["SEU_Category"] = active_records["SEU_Category"].fillna("Unknown")
    gas_records = active_records[active_records["Utility"] == "gas"]
    electricity_records = active_records[active_records["Utility"] == "electricity"]
    hdd_dates = _normalised_date_values(hdd_df["Date"]) if hdd_df is not None else set()
    cdd_dates = _normalised_date_values(cdd_df["Date"]) if cdd_df is not None else set()
    seu_categories = sorted(
        {
            str(category).strip()
            for category in active_records["SEU_Category"].dropna()
            if str(category).strip() and str(category).strip().casefold() != "unknown"
        }
    )

    weather_model_settings = weather_model_settings_from_config(client_config)

    return {
        "source": "legacy_source_package",
        "energy_record_count": int(len(energy_records)),
        "active_energy_record_count": int(len(active_records)),
        "meter_count": int(energy_records["Meter"].nunique()),
        "active_meter_count": int(active_records["Meter"].nunique()),
        "gas_record_count": int(len(gas_records)),
        "electricity_record_count": int(len(electricity_records)),
        "mapped_meter_count": mapped_meter_count,
        "ignored_meter_count": ignored_meter_count,
        "seu_categories": seu_categories,
        "date_range": _date_range_summary(energy_records["Date"]),
        "weather_coverage": {
            "hdd": {
                **_weather_metric_summary(hdd_df),
                "matched_energy_record_count": _matching_date_record_count(gas_records, hdd_dates),
            },
            "cdd": {
                **_weather_metric_summary(cdd_df),
                "matched_energy_record_count": _matching_date_record_count(electricity_records, cdd_dates),
            },
        },
        "weather_normalization": build_weather_normalized_summary(
            active_records,
            hdd_df,
            cdd_df,
            enabled=weather_model_settings.enabled,
            min_baseline_records=weather_model_settings.min_baseline_records,
            min_comparison_records=weather_model_settings.min_comparison_records,
        ),
        "total_consumption": float(active_records["Consumption"].sum()),
        "total_consumption_by_utility": _grouped_consumption_totals(
            active_records,
            "Utility",
            "utility",
        ),
        "total_consumption_by_seu_category": _grouped_consumption_totals(
            active_records,
            "SEU_Category",
            "seu_category",
        ),
    }


def _source_dataframe_summary(
    upload: Upload,
    payload: bytes,
    primary_upload: Upload,
) -> dict[str, Any]:
    try:
        if upload.category in {UploadCategory.HDD.value, UploadCategory.CDD.value}:
            df = _degree_day_dataframe_from_payload(payload, upload.category)
        elif upload.category == UploadCategory.SEU_MAPPING.value:
            df = _seu_mapping_dataframe_from_payload(payload)
        else:
            df = _normalise_dataframe_columns(_read_csv_payload(payload))
    except ValueError as exc:
        return {
            "upload_id": upload.id,
            "category": upload.category,
            "is_primary": upload.id == primary_upload.id,
            "parse_status": "failed",
            "error": str(exc),
        }

    summary: dict[str, Any] = {
        "upload_id": upload.id,
        "category": upload.category,
        "is_primary": upload.id == primary_upload.id,
        "parse_status": "parsed",
        "row_count": int(len(df)),
    }
    if upload.category == UploadCategory.ENERGY.value:
        if _is_legacy_energy_export(df):
            meter_df = _legacy_energy_export_to_meter_dataframe(df)
            meter_columns = [column for column in meter_df.columns if column != "Date"]
            summary.update(
                {
                    "shape": "legacy_energy_export",
                    "meter_column_count": len(meter_columns),
                    "meter_columns": meter_columns,
                    **_date_range_summary(meter_df["Date"]),
                }
            )
        else:
            timestamp_column = _find_timestamp_column(df)
            meter_columns = [column for column in df.columns if column != timestamp_column]
            summary.update(
                {
                    "shape": "saas_meter_csv",
                    "meter_column_count": len(meter_columns),
                    "meter_columns": meter_columns,
                    **(
                        _date_range_summary(df[timestamp_column])
                        if timestamp_column is not None
                        else {"start_date": None, "end_date": None}
                    ),
            }
        )
    elif upload.category in {UploadCategory.HDD.value, UploadCategory.CDD.value}:
        summary.update(
            {
                "shape": "degree_day_csv",
                "value_columns": [upload.category.upper()],
                **_date_range_summary(df["Date"]),
            }
        )
    elif upload.category == UploadCategory.SEU_MAPPING.value:
        categories = (
            sorted(
                str(category)
                for category in df.get("SEU_Category", pd.Series(dtype=str)).dropna().unique()
                if str(category).strip()
            )
            if "SEU_Category" in df.columns
            else []
        )
        summary.update(
            {
                "shape": "seu_mapping_csv",
                "mapping_count": int(len(df)),
                "seu_categories": categories,
            }
        )
    return summary


def _analysis_input_from_source_package(
    organization_id: str,
    primary_upload: Upload,
    run_uploads: list[Upload],
    client_config: dict[str, Any],
    artifact_store: TenantArtifactStore,
) -> AnalysisInput:
    payloads = {
        upload.id: artifact_store.get_bytes(organization_id, upload.storage_key)
        for upload in run_uploads
    }
    raw_meter_data = _normalise_dataframe_columns(_read_csv_payload(payloads[primary_upload.id]))
    execution_input = "saas_meter_csv"
    if _missing_meter_source_columns(raw_meter_data, client_config) and _is_legacy_energy_export(raw_meter_data):
        raw_meter_data = _legacy_energy_export_to_meter_dataframe(raw_meter_data)
        execution_input = "legacy_energy_export"

    missing_columns = _missing_meter_source_columns(raw_meter_data, client_config)
    if missing_columns:
        raise ValueError(
            "Configured meter source columns are missing from resolved analysis input: "
            + ", ".join(missing_columns)
        )

    source_summaries = [
        _source_dataframe_summary(upload, payloads[upload.id], primary_upload)
        for upload in run_uploads
    ]
    return AnalysisInput(
        raw_meter_data=raw_meter_data,
        legacy_baseline=_legacy_baseline_metrics(
            primary_upload,
            run_uploads,
            payloads,
            client_config,
        ),
        source_context={
            "execution_input": execution_input,
            "resolved_meter_columns": [
                column for column in raw_meter_data.columns if column != _find_timestamp_column(raw_meter_data)
            ],
            "resolved_record_count": int(len(raw_meter_data)),
            "sources": source_summaries,
        },
    )


def _stored_upload_ids(primary_upload_id: str, source_upload_ids: tuple[str, ...]) -> list[str]:
    upload_ids = [primary_upload_id]
    seen = {primary_upload_id}
    for upload_id in source_upload_ids:
        if upload_id in seen:
            continue
        upload_ids.append(upload_id)
        seen.add(upload_id)
    return upload_ids


def _ordered_run_uploads(primary_upload: Upload, run_uploads: list[Upload]) -> list[Upload]:
    supporting_uploads = [upload for upload in run_uploads if upload.id != primary_upload.id]
    return [
        primary_upload,
        *sorted(
            supporting_uploads,
            key=lambda upload: (SOURCE_CATEGORY_ORDER.get(upload.category, 99), upload.id),
        ),
    ]


def _resolve_stored_run_uploads(
    store: SaaSStore,
    context: Any,
    request: StoredUploadAnalysisRunRequest,
) -> tuple[Upload, list[Upload]]:
    primary_row = store.get_upload(request.upload_id)
    if primary_row is None:
        raise ValueError(f"Upload {request.upload_id} was not found")
    require_resource_scope(context, primary_row["organization_id"], "upload")
    primary_upload = _upload_from_row(primary_row)
    if primary_upload.status != UploadStatus.STORED:
        raise ValueError(f"Upload {primary_upload.id} must be stored before it can be used in a run")
    if primary_upload.category != UploadCategory.ENERGY.value:
        raise ValueError("Stored upload analysis requires an energy upload as the primary input")

    run_uploads = [primary_upload]
    categories = {primary_upload.category}
    for upload_id in _stored_upload_ids(request.upload_id, tuple(request.source_upload_ids))[1:]:
        upload_row = store.get_upload(upload_id)
        if upload_row is None:
            raise ValueError(f"Upload {upload_id} was not found")
        require_resource_scope(context, upload_row["organization_id"], "upload")
        upload = _upload_from_row(upload_row)
        if upload.status != UploadStatus.STORED:
            raise ValueError(f"Upload {upload.id} must be stored before it can be used in a run")
        if upload.site_id != primary_upload.site_id:
            raise ValueError(
                f"Upload {upload.id} does not belong to site {primary_upload.site_id}"
            )
        if upload.category == UploadCategory.ENERGY.value:
            raise ValueError("Supporting uploads must use hdd, cdd, or seu_mapping categories")
        if upload.category in categories:
            raise ValueError(
                f"Run source package already includes a {upload.category} upload"
            )
        categories.add(upload.category)
        run_uploads.append(upload)

    return primary_upload, _ordered_run_uploads(primary_upload, run_uploads)


def _public_source_summary(source: Any) -> dict[str, Any]:
    if not isinstance(source, dict):
        return {}
    return {
        key: _redact_internal_artifact_fields(source[key])
        for key in PUBLIC_SOURCE_SUMMARY_KEYS
        if key in source
    }


def _public_analysis_context(source_context: dict[str, Any]) -> dict[str, Any]:
    context = {
        key: _redact_internal_artifact_fields(source_context[key])
        for key in PUBLIC_SOURCE_CONTEXT_KEYS
        if key in source_context
    }
    sources = source_context.get("sources")
    if isinstance(sources, list):
        context["sources"] = [
            summary
            for source in sources
            if (summary := _public_source_summary(source))
        ]
    return context


def _public_source_package(source_package: Any) -> dict[str, Any]:
    if not isinstance(source_package, dict):
        return {}
    public = {
        key: _redact_internal_artifact_fields(source_package[key])
        for key in PUBLIC_SOURCE_PACKAGE_KEYS
        if key in source_package
    }
    uploads = source_package.get("uploads")
    if isinstance(uploads, list):
        public["uploads"] = [
            {
                key: _redact_internal_artifact_fields(upload[key])
                for key in PUBLIC_SOURCE_PACKAGE_UPLOAD_KEYS
                if isinstance(upload, dict) and key in upload
            }
            for upload in uploads
            if isinstance(upload, dict)
        ]
    analysis_context = source_package.get("analysis_context")
    if isinstance(analysis_context, dict):
        public["analysis_context"] = _public_analysis_context(analysis_context)
    return public


def _redact_internal_artifact_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _redact_internal_artifact_fields(item)
            for key, item in value.items()
            if str(key) not in INTERNAL_REPORT_ARTIFACT_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_redact_internal_artifact_fields(item) for item in value]
    return _json_ready(value)


def sanitize_report_artifact_content(content: dict[str, Any]) -> dict[str, Any]:
    public_content = _redact_internal_artifact_fields(content)
    source_package = public_content.get("source_package")
    if isinstance(source_package, dict):
        public_content["source_package"] = _public_source_package(source_package)
    return public_content


def _source_package_summary(
    primary_upload: Upload,
    run_uploads: list[Upload],
    source_context: dict[str, Any],
) -> dict[str, Any]:
    summary = {
        "primary_upload_id": primary_upload.id,
        "site_id": primary_upload.site_id,
        "upload_count": len(run_uploads),
        "categories": [upload.category for upload in run_uploads],
        "uploads": [
            {
                "id": upload.id,
                "category": upload.category,
                "checksum": upload.checksum,
                "status": upload.status.value,
                "is_primary": upload.id == primary_upload.id,
            }
            for upload in run_uploads
        ],
    }
    summary["analysis_context"] = source_context
    return _public_source_package(summary)


def _report_artifact_content(
    analytics: AnalyticsRunResult,
    primary_upload: Upload,
    run_uploads: list[Upload],
    source_context: dict[str, Any],
    legacy_baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    content = dict(_json_ready(analytics.iso_summary))
    content["source_package"] = _source_package_summary(primary_upload, run_uploads, source_context)
    if legacy_baseline is not None:
        content["legacy_baseline"] = _json_ready(legacy_baseline)
    return content


def _local_source_context(upload: Upload, raw_meter_data: pd.DataFrame) -> dict[str, Any]:
    timestamp_column = _find_timestamp_column(raw_meter_data)
    meter_columns = [column for column in raw_meter_data.columns if column != timestamp_column]
    return {
        "execution_input": "local_dataframe",
        "resolved_meter_columns": meter_columns,
        "resolved_record_count": int(len(raw_meter_data)),
        "sources": [
            {
                "upload_id": upload.id,
                "category": upload.category,
                "is_primary": True,
                "parse_status": "parsed",
                "shape": "local_dataframe",
                "row_count": int(len(raw_meter_data)),
                "meter_column_count": len(meter_columns),
                "meter_columns": meter_columns,
                **(
                    _date_range_summary(raw_meter_data[timestamp_column])
                    if timestamp_column is not None
                    else {"start_date": None, "end_date": None}
                ),
            }
        ],
    }


def _audit_event(
    request: AnalysisRunRequest | StoredUploadAnalysisRunRequest,
    action: AuditAction,
    resource_type: str,
    resource_id: str,
) -> AuditEvent:
    return AuditEvent(
        id=f"{request.run_id}:{action.value}:{resource_type}:{resource_id}",
        organization_id=request.organization_id,
        actor_user_id=request.user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
    )


def _run_completion_notification(
    request: AnalysisRunRequest | StoredUploadAnalysisRunRequest,
    event_type: str,
    subject: str,
    body: str,
) -> Notification:
    return Notification(
        id=f"{request.run_id}:email:{event_type}",
        organization_id=request.organization_id,
        recipient_user_id=request.user_id,
        channel=NotificationChannel.EMAIL,
        event_type=event_type,
        subject=subject,
        body=body,
        resource_type="run",
        resource_id=request.run_id,
        status=NotificationStatus.PENDING,
    )


def execute_analysis_run(
    store: SaaSStore,
    request: AnalysisRunRequest,
    artifact_store: TenantArtifactStore | None = None,
) -> AnalysisRunOutcome:
    artifacts = artifact_store or InMemoryTenantArtifactStore()
    context = require_permission(
        store.list_memberships(user_id=request.user_id, organization_id=request.organization_id),
        user_id=request.user_id,
        organization_id=request.organization_id,
        action=Action.MANAGE_RUNS,
    )
    site = store.get_site(request.site_id)
    if site is None:
        raise ValueError(f"Site {request.site_id} was not found")
    require_resource_scope(context, site.organization_id, "site")
    require_run_capacity(store, request.organization_id)
    client_config = resolve_analysis_config_for_run(
        store,
        request.organization_id,
        request.site_id,
        request.client_config,
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
    artifacts.put_text(
        request.organization_id,
        upload.storage_key,
        request.raw_meter_data.to_csv(index=False),
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
        store.create_audit_event(
            _audit_event(request, AuditAction.UPLOAD_STORED, "upload", request.upload_id)
        )
        store.create_run(run)
        store.update_run_status(request.run_id, RunStatus.RUNNING)
        store.create_audit_event(_audit_event(request, AuditAction.RUN_STARTED, "run", request.run_id))

    try:
        analytics = run_meter_analysis(request.raw_meter_data, client_config)
    except Exception as exc:
        with store.conn:
            store.update_run_status(request.run_id, RunStatus.FAILED, error_message=str(exc))
            store.create_audit_event(_audit_event(request, AuditAction.RUN_FAILED, "run", request.run_id))
            store.create_notification(
                _run_completion_notification(
                    request,
                    "run_failed",
                    f"Energy analysis {request.run_id} failed",
                    f"Run {request.run_id} could not complete: {exc}",
                )
            )
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
        expires_at=report_expires_at_for_organization(store, request.organization_id),
    )
    artifacts.put_text(
        request.organization_id,
        report.storage_key,
        json.dumps(
            _report_artifact_content(
                analytics,
                upload,
                [upload],
                _local_source_context(upload, request.raw_meter_data),
            ),
            indent=2,
            sort_keys=True,
        ),
    )
    with store.conn:
        store.update_run_status(request.run_id, RunStatus.SUCCEEDED)
        store.create_audit_event(_audit_event(request, AuditAction.RUN_SUCCEEDED, "run", request.run_id))
        store.create_report(report)
        store.create_audit_event(
            _audit_event(request, AuditAction.REPORT_CREATED, "report", request.report_id)
        )
        store.create_notification(
            _run_completion_notification(
                request,
                "run_succeeded",
                f"Energy analysis {request.run_id} completed",
                f"Run {request.run_id} completed successfully. Report {request.report_id} is ready.",
            )
        )

    return AnalysisRunOutcome(upload=upload, run=run, report=report, analytics=analytics)


def queue_analysis_run_from_upload(
    store: SaaSStore,
    request: StoredUploadAnalysisRunRequest,
) -> QueuedAnalysisRun:
    context = require_permission(
        store.list_memberships(user_id=request.user_id, organization_id=request.organization_id),
        user_id=request.user_id,
        organization_id=request.organization_id,
        action=Action.MANAGE_RUNS,
    )
    upload, run_uploads = _resolve_stored_run_uploads(store, context, request)
    require_run_capacity(store, request.organization_id)
    client_config = resolve_analysis_config_for_run(
        store,
        request.organization_id,
        upload.site_id,
        request.client_config,
    )
    run = Run(
        id=request.run_id,
        organization_id=request.organization_id,
        site_id=upload.site_id,
        requested_by_user_id=request.user_id,
        upload_ids=[source_upload.id for source_upload in run_uploads],
        status=RunStatus.QUEUED,
    )
    job = AnalysisJob(
        id=f"job_{request.run_id}",
        organization_id=request.organization_id,
        run_id=request.run_id,
        upload_id=upload.id,
        report_id=request.report_id,
        requested_by_user_id=request.user_id,
        client_config=client_config,
        status=RunStatus.QUEUED,
    )

    with store.conn:
        store.create_run(run)
        store.create_analysis_job(job)

    return QueuedAnalysisRun(job=job, run=run, upload=upload)


def process_analysis_job(
    store: SaaSStore,
    job_id: str,
    artifact_store: TenantArtifactStore,
) -> AnalysisRunOutcome:
    job_row = store.get_analysis_job(job_id)
    if job_row is None:
        raise ValueError(f"Analysis job {job_id} was not found")
    upload_row = store.get_upload(job_row["upload_id"])
    if upload_row is None:
        raise ValueError(f"Upload {job_row['upload_id']} was not found")
    run_row = store.get_run(job_row["run_id"])
    if run_row is None:
        raise ValueError(f"Run {job_row['run_id']} was not found")

    upload = _upload_from_row(upload_row)
    run_uploads = [upload]
    for run_upload_row in store.list_run_uploads(job_row["organization_id"], job_row["run_id"]):
        if run_upload_row["id"] != upload.id:
            run_uploads.append(_upload_from_row(run_upload_row))
    run_uploads = _ordered_run_uploads(upload, run_uploads)
    run_upload_ids = [run_upload.id for run_upload in run_uploads]
    analysis_input = _analysis_input_from_source_package(
        job_row["organization_id"],
        upload,
        run_uploads,
        json.loads(job_row["client_config_json"]),
        artifact_store,
    )
    run = Run(
        id=run_row["id"],
        organization_id=run_row["organization_id"],
        site_id=run_row["site_id"],
        requested_by_user_id=run_row["requested_by_user_id"],
        upload_ids=run_upload_ids,
        status=RunStatus(run_row["status"]),
        error_message=run_row["error_message"],
    )
    request = StoredUploadAnalysisRunRequest(
        user_id=job_row["requested_by_user_id"],
        organization_id=job_row["organization_id"],
        upload_id=job_row["upload_id"],
        run_id=job_row["run_id"],
        report_id=job_row["report_id"],
        source_upload_ids=tuple(upload_id for upload_id in run_upload_ids if upload_id != upload.id),
        client_config=json.loads(job_row["client_config_json"]),
    )

    with store.conn:
        store.update_analysis_job_status(job_id, RunStatus.RUNNING)
        store.update_run_status(request.run_id, RunStatus.RUNNING)
        store.create_audit_event(_audit_event(request, AuditAction.RUN_STARTED, "run", request.run_id))

    try:
        analytics = run_meter_analysis(analysis_input.raw_meter_data, request.client_config)
    except Exception as exc:
        with store.conn:
            store.update_run_status(request.run_id, RunStatus.FAILED, error_message=str(exc))
            store.update_analysis_job_status(job_id, RunStatus.FAILED, error_message=str(exc))
            store.create_audit_event(_audit_event(request, AuditAction.RUN_FAILED, "run", request.run_id))
            store.create_notification(
                _run_completion_notification(
                    request,
                    "run_failed",
                    f"Energy analysis {request.run_id} failed",
                    f"Run {request.run_id} could not complete: {exc}",
                )
            )
        raise

    report = Report(
        id=request.report_id,
        organization_id=request.organization_id,
        run_id=request.run_id,
        report_type="iso_summary",
        storage_key=report_storage_key(
            request.organization_id,
            upload.site_id,
            request.run_id,
            "iso-summary.json",
        ),
        is_published=True,
        expires_at=report_expires_at_for_organization(store, request.organization_id),
    )
    artifact_store.put_text(
        request.organization_id,
        report.storage_key,
        json.dumps(
            _report_artifact_content(
                analytics,
                upload,
                run_uploads,
                analysis_input.source_context,
                analysis_input.legacy_baseline,
            ),
            indent=2,
            sort_keys=True,
        ),
    )
    with store.conn:
        store.update_run_status(request.run_id, RunStatus.SUCCEEDED)
        store.update_analysis_job_status(job_id, RunStatus.SUCCEEDED)
        store.create_audit_event(_audit_event(request, AuditAction.RUN_SUCCEEDED, "run", request.run_id))
        store.create_report(report)
        store.create_audit_event(
            _audit_event(request, AuditAction.REPORT_CREATED, "report", request.report_id)
        )
        store.create_notification(
            _run_completion_notification(
                request,
                "run_succeeded",
                f"Energy analysis {request.run_id} completed",
                f"Run {request.run_id} completed successfully. Report {request.report_id} is ready.",
            )
        )

    return AnalysisRunOutcome(upload=upload, run=run, report=report, analytics=analytics)


def execute_analysis_run_from_upload(
    store: SaaSStore,
    request: StoredUploadAnalysisRunRequest,
    artifact_store: TenantArtifactStore,
) -> AnalysisRunOutcome:
    queued = queue_analysis_run_from_upload(store, request)
    return process_analysis_job(store, queued.job.id, artifact_store)
