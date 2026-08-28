import pandas as pd

from analytics.service import AnalyticsRunResult, run_meter_analysis
from src.config_loader import load_client_config


def test_run_meter_analysis_wraps_existing_analytics_pipeline():
    cfg = load_client_config("config/clients/example_client.yaml")
    raw = pd.DataFrame(
        {
            "Date": ["2025-01-01", "2025-02-01"],
            "Main Electricity": [100, 120],
            "Main Gas": [80, 70],
        }
    )

    result = run_meter_analysis(raw, cfg)

    assert isinstance(result, AnalyticsRunResult)
    assert len(result.standardised_data) == 4
    assert set(result.consumption_summary["meter_id"]) == {"elec_main", "gas_main"}
    assert result.iso_summary["total_records"] == 4
    assert "baseline_table" in result.baseline_model


def test_run_meter_analysis_raises_for_missing_configured_columns():
    cfg = load_client_config("config/clients/example_client.yaml")
    raw = pd.DataFrame(
        {
            "Date": ["2025-01-01"],
            "Main Electricity": [100],
        }
    )

    try:
        run_meter_analysis(raw, cfg)
        assert False, "Expected missing meter source column to fail"
    except ValueError as exc:
        assert "missing" in str(exc).lower()
