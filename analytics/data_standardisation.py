from __future__ import annotations

import pandas as pd


TIMESTAMP_CANDIDATES = ["timestamp", "date", "datetime", "time"]


def load_uploaded_data(file) -> pd.DataFrame:
    try:
        return pd.read_csv(file)
    except Exception as exc:
        raise ValueError(f"Unable to read uploaded file as CSV: {exc}") from exc


def _find_timestamp_column(raw_df: pd.DataFrame) -> str:
    normalized = {c.lower().strip(): c for c in raw_df.columns}
    for candidate in TIMESTAMP_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    return raw_df.columns[0]


def standardise_meter_data(raw_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    if raw_df.empty:
        raise ValueError("Uploaded data is empty.")

    ts_col = _find_timestamp_column(raw_df)
    working = raw_df.copy()
    working[ts_col] = pd.to_datetime(working[ts_col], errors="coerce", dayfirst=True)

    missing_meter_cols = [
        meter["source_column"] for meter in config["meters"] if meter["source_column"] not in working.columns
    ]
    if missing_meter_cols:
        raise ValueError(
            "Configured meter source columns are missing from uploaded data: "
            + ", ".join(missing_meter_cols)
        )

    frames = []
    for meter in config["meters"]:
        src_col = meter["source_column"]
        meter_df = working[[ts_col, src_col]].copy()
        meter_df.rename(columns={ts_col: "timestamp", src_col: "value"}, inplace=True)
        meter_df["meter_id"] = meter["meter_id"]
        meter_df["unit"] = meter["unit"]
        meter_df["commodity"] = meter["commodity"]
        meter_df["source"] = src_col
        meter_df["value"] = pd.to_numeric(meter_df["value"], errors="coerce")
        frames.append(meter_df)

    standard_df = pd.concat(frames, ignore_index=True)
    standard_df.dropna(subset=["timestamp", "value"], inplace=True)
    standard_df = standard_df[["timestamp", "meter_id", "value", "unit", "commodity", "source"]]
    return standard_df
