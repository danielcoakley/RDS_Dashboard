import pandas as pd
import pytest

from analytics.legacy_source_package import build_weather_normalized_summary


def _modeled_legacy_inputs():
    baseline_dates = pd.date_range("2024-01-01", periods=12, freq="D")
    comparison_dates = pd.date_range("2025-01-01", periods=12, freq="D")
    dates = [*baseline_dates, *comparison_dates]
    hdd_values = [float(index + 1) for index, _date in enumerate(dates)]
    cdd_values = [float((index + 1) * 2) for index, _date in enumerate(dates)]
    gas_values = [100 + 2 * value for value in hdd_values[:12]]
    gas_values.extend([100 + 2 * value - 10 for value in hdd_values[12:]])
    electricity_values = [50 + 3 * value for value in cdd_values[:12]]
    electricity_values.extend([50 + 3 * value - 5 for value in cdd_values[12:]])

    energy_records = pd.DataFrame(
        [
            {
                "Date": date,
                "Meter": "Boiler gas",
                "Utility": "gas",
                "SEU_Category": "Boiler",
                "Consumption": value,
            }
            for date, value in zip(dates, gas_values)
        ]
        + [
            {
                "Date": date,
                "Meter": "Office power",
                "Utility": "electricity",
                "SEU_Category": "Lighting",
                "Consumption": value,
            }
            for date, value in zip(dates, electricity_values)
        ]
    )
    hdd_df = pd.DataFrame({"Date": dates, "HDD": hdd_values})
    cdd_df = pd.DataFrame({"Date": dates, "CDD": cdd_values})
    return energy_records, hdd_df, cdd_df


def test_build_weather_normalized_summary_models_legacy_energy_against_degree_days():
    energy_records, hdd_df, cdd_df = _modeled_legacy_inputs()

    summary = build_weather_normalized_summary(energy_records, hdd_df, cdd_df)

    assert summary["status"] == "modeled"
    assert summary["baseline_year"] == 2024
    assert summary["comparison_year"] == 2025
    assert summary["model_count"] == 2
    assert summary["skipped_meter_count"] == 0
    assert summary["total_actual_consumption"] == pytest.approx(3396.0)
    assert summary["total_predicted_consumption"] == pytest.approx(3576.0)
    assert summary["estimated_savings"] == pytest.approx(180.0)
    models_by_meter = {model["meter"]: model for model in summary["models"]}
    assert models_by_meter["Boiler gas"]["normalization"] == "hdd"
    assert models_by_meter["Boiler gas"]["weather_sensitivity"] == pytest.approx(2.0)
    assert models_by_meter["Boiler gas"]["estimated_savings"] == pytest.approx(120.0)
    assert models_by_meter["Office power"]["normalization"] == "cdd"
    assert models_by_meter["Office power"]["weather_sensitivity"] == pytest.approx(3.0)
    assert models_by_meter["Office power"]["estimated_savings"] == pytest.approx(60.0)


def test_build_weather_normalized_summary_reports_skips_when_years_are_insufficient():
    energy_records, hdd_df, cdd_df = _modeled_legacy_inputs()
    single_year_records = energy_records[energy_records["Date"].dt.year == 2024]

    summary = build_weather_normalized_summary(single_year_records, hdd_df, cdd_df)

    assert summary["status"] == "not_modeled"
    assert summary["reason"] == "insufficient_years"
    assert summary["model_count"] == 0
    assert summary["skipped_meter_count"] == 2
    assert {meter["reason"] for meter in summary["skipped_meters"]} == {"insufficient_years"}
