from __future__ import annotations

import pandas as pd


def build_baseline_model(df: pd.DataFrame, weather_df: pd.DataFrame | None = None, config: dict | None = None) -> dict:
    # Lightweight placeholder baseline: monthly average by meter
    working = df.copy()
    working["month"] = working["timestamp"].dt.to_period("M")
    baseline = working.groupby(["meter_id", "month"], as_index=False)["value"].mean()
    return {"baseline_table": baseline}


def calculate_enpi(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    totals = df.groupby("meter_id", as_index=False)["value"].sum()
    grand_total = totals["value"].sum()
    totals["enpi_share_pct"] = (totals["value"] / grand_total * 100).round(2) if grand_total else 0
    return totals
