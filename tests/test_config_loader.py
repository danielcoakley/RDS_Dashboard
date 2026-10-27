from src.config_loader import load_client_config, validate_client_config


def test_load_client_config():
    cfg = load_client_config('config/clients/example_client.yaml')
    assert cfg['client']['id'] == 'example_client'


def test_validate_missing_required_field():
    bad = {
        'client': {'id': 'x'},
        'site': {'name': 's'},
        'meters': [],
        'analysis': {},
        'iso50001': {},
    }
    try:
        validate_client_config(bad)
        assert False, 'Expected ValueError'
    except ValueError as exc:
        assert 'Missing required client fields' in str(exc)
