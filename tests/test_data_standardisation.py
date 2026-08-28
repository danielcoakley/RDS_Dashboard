import pandas as pd
from src.config_loader import load_client_config
from src.data_standardisation import standardise_meter_data


def test_standardise_wide_to_long():
    cfg = load_client_config('config/clients/example_client.yaml')
    raw = pd.DataFrame({
        'Date': ['2025-01-01', '2025-02-01'],
        'Main Electricity': [100, 120],
        'Main Gas': [80, 70],
    })
    out = standardise_meter_data(raw, cfg)
    assert list(out.columns) == ['timestamp', 'meter_id', 'value', 'unit', 'commodity', 'source']
    assert len(out) == 4
