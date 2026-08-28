from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data_quality import clean_meter_data, detect_data_gaps, detect_outliers
from src.data_standardisation import standardise_meter_data
from src.enpi_model import build_baseline_model, calculate_enpi
from src.reporting import calculate_consumption_summary, generate_iso_summary


@dataclass(frozen=True)
class AnalyticsRunResult:
    standardised_data: pd.DataFrame
    consumption_summary: pd.DataFrame
    enpi_summary: pd.DataFrame
    data_gaps: pd.DataFrame
    outliers: pd.DataFrame
    iso_summary: dict
    baseline_model: dict


def run_meter_analysis(raw_meter_data: pd.DataFrame, client_config: dict) -> AnalyticsRunResult:
    standardised = standardise_meter_data(raw_meter_data, client_config)
    cleaned = clean_meter_data(standardised, client_config)

    return AnalyticsRunResult(
        standardised_data=cleaned,
        consumption_summary=calculate_consumption_summary(cleaned),
        enpi_summary=calculate_enpi(cleaned, client_config),
        data_gaps=detect_data_gaps(cleaned),
        outliers=detect_outliers(cleaned, client_config),
        iso_summary=generate_iso_summary(cleaned, client_config),
        baseline_model=build_baseline_model(cleaned, config=client_config),
    )
