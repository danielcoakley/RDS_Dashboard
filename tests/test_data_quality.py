import pandas as pd
from src.config_loader import load_client_config
from src.data_standardisation import standardise_meter_data


def test_missing_configured_columns_raises_error():
    cfg = load_client_config('config/clients/example_client.yaml')
    raw = pd.DataFrame({
        'Date': ['2025-01-01'],
        'Main Electricity': [100],
    })
    try:
        standardise_meter_data(raw, cfg)
        assert False, 'Expected ValueError for missing meter column'
    except ValueError as exc:
        assert 'missing' in str(exc).lower()
