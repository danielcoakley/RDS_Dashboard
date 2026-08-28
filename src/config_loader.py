from __future__ import annotations

from pathlib import Path
import json

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

_REQUIRED_TOP_LEVEL = ["client", "site", "meters", "analysis", "iso50001"]
_REQUIRED_CLIENT = ["id", "name", "timezone", "reporting_currency"]
_REQUIRED_SITE = ["name"]
_REQUIRED_METER = ["meter_id", "display_name", "commodity", "unit", "source_column", "is_seu"]


def _coerce(value: str):
    v = value.strip()
    if v in {"null", "~", ""}:
        return None
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    if v.startswith('"') and v.endswith('"'):
        return v[1:-1]
    try:
        return int(v) if "." not in v else float(v)
    except ValueError:
        return v


def _parse_simple_yaml(text: str) -> dict:
    # Minimal parser for current config shape.
    data = {}
    section = None
    current_list_item = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" ") and line.endswith(":"):
            section = line[:-1]
            data[section] = {} if section != "meters" else []
            current_list_item = None
            continue
        if section == "meters" and line.strip().startswith("-"):
            current_list_item = {}
            data[section].append(current_list_item)
            line = line.strip()[1:].strip()
            if line and ":" in line:
                k, v = line.split(":", 1)
                current_list_item[k.strip()] = _coerce(v)
            continue
        if ":" in line:
            k, v = line.strip().split(":", 1)
            if section == "meters" and current_list_item is not None:
                current_list_item[k.strip()] = _coerce(v)
            else:
                data[section][k.strip()] = _coerce(v)
    return data


def load_client_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Client config file not found: {path}")

    text = config_path.read_text(encoding="utf-8")
    if yaml is not None:
        config = yaml.safe_load(text) or {}
    else:
        try:
            config = json.loads(text)
        except Exception:
            config = _parse_simple_yaml(text)

    if not isinstance(config, dict):
        raise ValueError("Client config must be a YAML object at the top level.")
    validate_client_config(config)
    return config


def validate_client_config(config: dict) -> None:
    missing = [field for field in _REQUIRED_TOP_LEVEL if field not in config]
    if missing:
        raise ValueError(f"Missing top-level config sections: {', '.join(missing)}")
    client = config.get("client", {})
    site = config.get("site", {})
    meters = config.get("meters", [])
    missing_client = [field for field in _REQUIRED_CLIENT if field not in client]
    if missing_client:
        raise ValueError(f"Missing required client fields: {', '.join(missing_client)}")
    missing_site = [field for field in _REQUIRED_SITE if field not in site]
    if missing_site:
        raise ValueError(f"Missing required site fields: {', '.join(missing_site)}")
    if not isinstance(meters, list) or not meters:
        raise ValueError("'meters' must be a non-empty list of meter definitions.")
    meter_errors = []
    for idx, meter in enumerate(meters):
        missing_meter = [field for field in _REQUIRED_METER if field not in meter]
        if missing_meter:
            meter_errors.append(f"meters[{idx}] missing: {', '.join(missing_meter)}")
    if meter_errors:
        raise ValueError("; ".join(meter_errors))
