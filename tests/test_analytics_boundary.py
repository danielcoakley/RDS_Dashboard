import importlib

import pandas as pd

from analytics.config_loader import load_client_config
from analytics.data_quality import clean_meter_data
from analytics.data_standardisation import standardise_meter_data
from analytics.service import run_meter_analysis
from src.config_loader import load_client_config as legacy_load_client_config


def test_analytics_package_exposes_canonical_pipeline_modules():
    config = load_client_config("config/clients/example_client.yaml")
    raw = pd.DataFrame(
        {
            "Date": ["2025-01-01", "2025-02-01"],
            "Main Electricity": [100, 120],
            "Main Gas": [80, 70],
        }
    )

    standardised = standardise_meter_data(raw, config)
    cleaned = clean_meter_data(standardised, config)
    result = run_meter_analysis(raw, config)

    assert len(standardised) == 4
    assert len(cleaned) == 4
    assert result.iso_summary["total_records"] == 4


def test_legacy_src_config_loader_reexports_analytics_boundary():
    assert legacy_load_client_config is load_client_config


def test_backend_analytics_entrypoints_do_not_import_legacy_src_modules():
    for module_name in ["backend.analysis_configs", "backend.run_orchestration"]:
        module = importlib.import_module(module_name)
        assert "src." not in (module.__loader__.get_source(module_name) or "")
