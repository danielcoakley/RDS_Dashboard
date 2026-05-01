from __future__ import annotations

import pandas as pd


def clean_meter_data(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["value"] = pd.to_numeric(cleaned["value"], errors="coerce")
    cleaned = cleaned.dropna(subset=["timestamp", "meter_id", "value"])
    return cleaned.sort_values(["meter_id", "timestamp"]) 


def detect_data_gaps(df: pd.DataFrame) -> pd.DataFrame:
    gaps = []
    for meter_id, group in df.sort_values("timestamp").groupby("meter_id"):
        ts = group["timestamp"].dropna().sort_values()
        if len(ts) < 2:
            continue
        diffs = ts.diff().dropna()
        expected = diffs.mode().iloc[0]
        for i, delta in diffs.items():
            if delta > expected:
                gaps.append({"meter_id": meter_id, "timestamp": ts.loc[i], "gap": delta})
    return pd.DataFrame(gaps)


def detect_outliers(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    flagged = []
    for meter_id, group in df.groupby("meter_id"):
        q1 = group["value"].quantile(0.25)
        q3 = group["value"].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out = group[(group["value"] < lower) | (group["value"] > upper)]
        if not out.empty:
            flagged.append(out)
    return pd.concat(flagged, ignore_index=True) if flagged else pd.DataFrame(columns=df.columns)
