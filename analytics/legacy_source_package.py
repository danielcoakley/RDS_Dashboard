from __future__ import annotations

import math
from typing import Any

import pandas as pd
from sklearn.linear_model import LinearRegression


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return round(parsed, 6)


def _prepare_energy_records(energy_records: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"Date", "Meter", "Utility", "Consumption"}
    missing = required_columns.difference(energy_records.columns)
    if missing:
        raise ValueError(
            "Legacy weather-normalized analysis is missing energy columns: "
            + ", ".join(sorted(missing))
        )

    working = energy_records.copy()
    working["Date"] = pd.to_datetime(working["Date"], errors="coerce", dayfirst=True).dt.normalize()
    working["Consumption"] = pd.to_numeric(working["Consumption"], errors="coerce")
    working["Meter"] = working["Meter"].where(working["Meter"].notna(), "").astype(str).str.strip()
    working["Utility"] = (
        working["Utility"].where(working["Utility"].notna(), "unknown").astype(str).str.strip().str.lower()
    )
    if "SEU_Category" not in working.columns:
        working["SEU_Category"] = "Unknown"
    working["SEU_Category"] = (
        working["SEU_Category"].where(working["SEU_Category"].notna(), "Unknown").astype(str).str.strip()
    )
    working = working.dropna(subset=["Date", "Consumption"])
    working = working[working["Meter"] != ""].copy()
    working["Year"] = working["Date"].dt.year
    working["IsOperational"] = (working["Consumption"] > 0).astype(int)
    return working


def _prepare_weather_data(weather_df: pd.DataFrame | None, value_column: str) -> pd.DataFrame:
    if weather_df is None or weather_df.empty:
        return pd.DataFrame(columns=["Date", value_column])
    if "Date" not in weather_df.columns or value_column not in weather_df.columns:
        raise ValueError(f"Weather data must include Date and {value_column} columns")
    working = weather_df[["Date", value_column]].copy()
    working["Date"] = pd.to_datetime(working["Date"], errors="coerce", dayfirst=True).dt.normalize()
    working[value_column] = pd.to_numeric(working[value_column], errors="coerce")
    working = working.dropna(subset=["Date", value_column])
    return working.groupby("Date", as_index=False)[value_column].sum()


def _normalization_for_utility(utility: str) -> tuple[str, str]:
    if utility == "gas":
        return "hdd", "HDD"
    if utility == "electricity":
        return "cdd", "CDD"
    return "unsupported", ""


def _skip(meter: str, utility: str, seu_category: str, reason: str) -> dict[str, Any]:
    return {
        "meter": meter,
        "utility": utility,
        "seu_category": seu_category or "Unknown",
        "reason": reason,
    }


def _model_meter(
    meter: str,
    meter_records: pd.DataFrame,
    weather_data: dict[str, pd.DataFrame],
    baseline_year: int,
    comparison_year: int,
    min_baseline_records: int,
    min_comparison_records: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    utility = str(meter_records["Utility"].iloc[0])
    seu_category = str(meter_records["SEU_Category"].iloc[0] or "Unknown")
    normalization, weather_column = _normalization_for_utility(utility)
    if normalization == "unsupported":
        return None, _skip(meter, utility, seu_category, "unsupported_utility")

    weather_df = weather_data[weather_column]
    if weather_df.empty:
        return None, _skip(meter, utility, seu_category, f"missing_{normalization}_weather")

    modeled = meter_records.merge(weather_df, on="Date", how="left")
    train = modeled[modeled["Year"] == baseline_year].dropna(subset=[weather_column, "Consumption"])
    test = modeled[modeled["Year"] == comparison_year].dropna(subset=[weather_column, "Consumption"])
    if len(train) < min_baseline_records:
        return None, _skip(meter, utility, seu_category, "insufficient_baseline_records")
    if len(test) < min_comparison_records:
        return None, _skip(meter, utility, seu_category, "insufficient_comparison_records")

    feature_columns = [weather_column, "IsOperational"]
    model = LinearRegression().fit(train[feature_columns], train["Consumption"])
    predictions = model.predict(test[feature_columns])
    predicted_consumption = predictions.sum()
    actual_consumption = test["Consumption"].sum()
    estimated_savings = predicted_consumption - actual_consumption

    return (
        {
            "meter": meter,
            "utility": utility,
            "seu_category": seu_category,
            "normalization": normalization,
            "weather_column": weather_column,
            "baseline_record_count": int(len(train)),
            "comparison_record_count": int(len(test)),
            "baseline_consumption": _finite_float(train["Consumption"].sum()),
            "actual_consumption": _finite_float(actual_consumption),
            "predicted_consumption": _finite_float(predicted_consumption),
            "estimated_savings": _finite_float(estimated_savings),
            "r2": _finite_float(model.score(train[feature_columns], train["Consumption"])),
            "weather_sensitivity": _finite_float(model.coef_[0]),
            "operational_day_sensitivity": _finite_float(model.coef_[1]),
            "intercept": _finite_float(model.intercept_),
        },
        None,
    )


def build_weather_normalized_summary(
    energy_records: pd.DataFrame,
    hdd_df: pd.DataFrame | None = None,
    cdd_df: pd.DataFrame | None = None,
    enabled: bool = True,
    min_baseline_records: int = 10,
    min_comparison_records: int = 1,
) -> dict[str, Any]:
    """Build tenant-safe weather-normalized model summaries for legacy source packages."""

    energy = _prepare_energy_records(energy_records)
    years = sorted(int(year) for year in energy["Year"].dropna().unique())
    baseline_year = years[0] if years else None
    comparison_year = years[-1] if len(years) > 1 else None
    model_settings = {
        "enabled": enabled,
        "min_baseline_records": min_baseline_records,
        "min_comparison_records": min_comparison_records,
    }
    if not enabled:
        return {
            "source": "legacy_source_package",
            "status": "disabled",
            "reason": "tenant_analysis_config_disabled",
            "baseline_year": baseline_year,
            "comparison_year": comparison_year,
            "model_count": 0,
            "skipped_meter_count": 0,
            "model_settings": model_settings,
            "models": [],
            "skipped_meters": [],
        }
    if baseline_year is None or comparison_year is None:
        return {
            "source": "legacy_source_package",
            "status": "not_modeled",
            "reason": "insufficient_years",
            "baseline_year": baseline_year,
            "comparison_year": comparison_year,
            "model_count": 0,
            "skipped_meter_count": int(energy["Meter"].nunique()) if not energy.empty else 0,
            "model_settings": model_settings,
            "models": [],
            "skipped_meters": [
                _skip(str(meter), "unknown", "Unknown", "insufficient_years")
                for meter in sorted(energy["Meter"].dropna().unique())
            ],
        }

    weather_data = {
        "HDD": _prepare_weather_data(hdd_df, "HDD"),
        "CDD": _prepare_weather_data(cdd_df, "CDD"),
    }
    models: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for meter, meter_records in energy.groupby("Meter", sort=True):
        model, skip = _model_meter(
            str(meter),
            meter_records.sort_values("Date"),
            weather_data,
            baseline_year,
            comparison_year,
            min_baseline_records,
            min_comparison_records,
        )
        if model is not None:
            models.append(model)
        if skip is not None:
            skipped.append(skip)

    total_actual = sum(model["actual_consumption"] or 0 for model in models)
    total_predicted = sum(model["predicted_consumption"] or 0 for model in models)
    total_baseline = sum(model["baseline_consumption"] or 0 for model in models)
    estimated_savings = total_predicted - total_actual
    return {
        "source": "legacy_source_package",
        "status": "modeled" if models else "not_modeled",
        "baseline_year": baseline_year,
        "comparison_year": comparison_year,
        "model_count": len(models),
        "skipped_meter_count": len(skipped),
        "model_settings": model_settings,
        "total_baseline_consumption": _finite_float(total_baseline),
        "total_actual_consumption": _finite_float(total_actual),
        "total_predicted_consumption": _finite_float(total_predicted),
        "estimated_savings": _finite_float(estimated_savings),
        "estimated_savings_pct": _finite_float(
            (estimated_savings / total_actual * 100) if total_actual else None
        ),
        "models": models,
        "skipped_meters": skipped,
    }
