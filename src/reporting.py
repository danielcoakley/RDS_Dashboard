from __future__ import annotations

import pandas as pd


def calculate_consumption_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["commodity", "meter_id", "unit"], as_index=False)["value"]
        .sum()
        .rename(columns={"value": "total_consumption"})
        .sort_values("total_consumption", ascending=False)
    )


def generate_iso_summary(df: pd.DataFrame, config: dict) -> dict:
    seu_meters = {m["meter_id"] for m in config.get("meters", []) if m.get("is_seu")}
    seu_total = df[df["meter_id"].isin(seu_meters)]["value"].sum()
    return {
        "client": config.get("client", {}).get("name", "Unknown"),
        "site": config.get("site", {}).get("name", "Unknown"),
        "total_records": int(len(df)),
        "seu_total_consumption": float(seu_total),
    }
