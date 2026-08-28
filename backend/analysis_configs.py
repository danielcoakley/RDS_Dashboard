from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any
from uuid import uuid4

from analytics.config_loader import validate_client_config

from .access_control import require_permission, require_resource_scope
from .domain import AnalysisConfig, AuditAction, AuditEvent, Site
from .rbac import Action
from .store import SaaSStore


@dataclass(frozen=True)
class TenantAnalysisConfig:
    organization_id: str
    config: dict[str, Any]
    source: str
    updated_by_user_id: str | None = None
    updated_at: str | None = None


@dataclass(frozen=True)
class WeatherModelSettings:
    enabled: bool
    min_baseline_records: int
    min_comparison_records: int


DEFAULT_WEATHER_NORMALISATION_ENABLED = True
DEFAULT_WEATHER_MODEL_MIN_BASELINE_RECORDS = 10
DEFAULT_WEATHER_MODEL_MIN_COMPARISON_RECORDS = 1
MAX_WEATHER_MODEL_RECORD_THRESHOLD = 3650


def _bool_setting(value: Any, default: bool, field_name: str) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise ValueError(f"{field_name} must be a boolean")


def _int_setting(value: Any, default: int, field_name: str, minimum: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    if value < minimum or value > MAX_WEATHER_MODEL_RECORD_THRESHOLD:
        raise ValueError(
            f"{field_name} must be between {minimum} and "
            f"{MAX_WEATHER_MODEL_RECORD_THRESHOLD}"
        )
    return value


def weather_model_settings_from_config(config: dict[str, Any]) -> WeatherModelSettings:
    analysis = config.get("analysis", {})
    if not isinstance(analysis, dict):
        raise ValueError("'analysis' must be a JSON object")
    return WeatherModelSettings(
        enabled=_bool_setting(
            analysis.get("weather_normalisation"),
            DEFAULT_WEATHER_NORMALISATION_ENABLED,
            "analysis.weather_normalisation",
        ),
        min_baseline_records=_int_setting(
            analysis.get("weather_model_min_baseline_records"),
            DEFAULT_WEATHER_MODEL_MIN_BASELINE_RECORDS,
            "analysis.weather_model_min_baseline_records",
            2,
        ),
        min_comparison_records=_int_setting(
            analysis.get("weather_model_min_comparison_records"),
            DEFAULT_WEATHER_MODEL_MIN_COMPARISON_RECORDS,
            "analysis.weather_model_min_comparison_records",
            1,
        ),
    )


def normalize_analysis_config(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("Analysis config must be a JSON object")
    validate_client_config(config)
    normalized = json.loads(json.dumps(config, sort_keys=True))
    analysis = normalized.get("analysis")
    if not isinstance(analysis, dict):
        raise ValueError("'analysis' must be a JSON object")
    weather_settings = weather_model_settings_from_config(normalized)
    analysis["weather_normalisation"] = weather_settings.enabled
    analysis["weather_model_min_baseline_records"] = weather_settings.min_baseline_records
    analysis["weather_model_min_comparison_records"] = weather_settings.min_comparison_records
    return normalized


def _site_for_default(
    store: SaaSStore,
    organization_id: str,
    site_id: str | None = None,
) -> Site:
    if site_id is not None:
        site = store.get_site(site_id)
        if site is None:
            raise ValueError(f"Site {site_id} was not found")
        if site.organization_id != organization_id:
            raise ValueError(f"Site {site_id} does not belong to organization {organization_id}")
        return site

    sites = store.list_sites(organization_id)
    if not sites:
        raise ValueError("Create a tenant site before generating analysis settings")
    return sites[0]


def build_analysis_config_from_tenant_assets(
    store: SaaSStore,
    organization_id: str,
    site_id: str | None = None,
) -> dict[str, Any]:
    organization = store.get_organization(organization_id)
    if organization is None:
        raise ValueError(f"Organization {organization_id} was not found")
    site = _site_for_default(store, organization_id, site_id)
    meters = store.list_meters(organization_id, site_id=site.id)
    if not meters:
        raise ValueError("Create at least one tenant meter before generating analysis settings")

    config: dict[str, Any] = {
        "client": {
            "id": organization["id"],
            "name": organization["name"],
            "timezone": site.timezone,
            "reporting_currency": "GBP",
        },
        "site": {
            "name": site.name,
            "floor_area_m2": None,
            "activity_metric": None,
        },
        "meters": [
            {
                "meter_id": meter.id,
                "display_name": meter.display_name,
                "commodity": meter.commodity,
                "unit": meter.unit,
                "source_column": meter.source_column,
                "is_seu": meter.is_seu,
            }
            for meter in meters
        ],
        "analysis": {
            "baseline_start": None,
            "baseline_end": None,
            "reporting_start": None,
            "reporting_end": None,
            "weather_normalisation": True,
            "hdd_base_temperature": 15.5,
            "cdd_base_temperature": 22.0,
            "weather_model_min_baseline_records": DEFAULT_WEATHER_MODEL_MIN_BASELINE_RECORDS,
            "weather_model_min_comparison_records": DEFAULT_WEATHER_MODEL_MIN_COMPARISON_RECORDS,
            "outlier_removal": True,
        },
        "iso50001": {
            "organisation_context": "",
            "energy_policy_reference": "",
            "audit_notes": "",
        },
    }
    return normalize_analysis_config(config)


def get_tenant_analysis_config(
    store: SaaSStore,
    user_id: str,
    organization_id: str,
    site_id: str | None = None,
) -> TenantAnalysisConfig:
    context = require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.READ,
    )
    if site_id is not None:
        site = store.get_site(site_id)
        if site is None:
            raise ValueError(f"Site {site_id} was not found")
        require_resource_scope(context, site.organization_id, "site")

    row = store.get_analysis_config(organization_id)
    if row is not None:
        return TenantAnalysisConfig(
            organization_id=organization_id,
            config=normalize_analysis_config(json.loads(row["config_json"])),
            source="saved",
            updated_by_user_id=row["updated_by_user_id"],
            updated_at=row["updated_at"],
        )

    return TenantAnalysisConfig(
        organization_id=organization_id,
        config=build_analysis_config_from_tenant_assets(store, organization_id, site_id),
        source="tenant_assets",
    )


def update_tenant_analysis_config(
    store: SaaSStore,
    user_id: str,
    organization_id: str,
    config: dict[str, Any],
) -> TenantAnalysisConfig:
    require_permission(
        store.list_memberships(user_id=user_id, organization_id=organization_id),
        user_id=user_id,
        organization_id=organization_id,
        action=Action.MANAGE_ASSETS,
    )
    normalized = normalize_analysis_config(config)
    meter_count = len(normalized["meters"])
    event = AuditEvent(
        id=f"audit_analysis_config_{uuid4().hex}",
        organization_id=organization_id,
        actor_user_id=user_id,
        action=AuditAction.ANALYSIS_CONFIG_UPDATED,
        resource_type="analysis_config",
        resource_id=organization_id,
        metadata={
            "meter_count": str(meter_count),
            "client_id": str(normalized["client"]["id"]),
            "weather_model_min_baseline_records": str(
                normalized["analysis"]["weather_model_min_baseline_records"]
            ),
            "weather_model_min_comparison_records": str(
                normalized["analysis"]["weather_model_min_comparison_records"]
            ),
            "weather_normalisation_enabled": normalized["analysis"]["weather_normalisation"],
        },
    )
    with store.conn:
        store.upsert_analysis_config(
            AnalysisConfig(
                organization_id=organization_id,
                config=normalized,
                updated_by_user_id=user_id,
            )
        )
        store.create_audit_event(event)

    row = store.get_analysis_config(organization_id)
    return TenantAnalysisConfig(
        organization_id=organization_id,
        config=normalized,
        source="saved",
        updated_by_user_id=row["updated_by_user_id"] if row is not None else user_id,
        updated_at=row["updated_at"] if row is not None else None,
    )


def resolve_analysis_config_for_run(
    store: SaaSStore,
    organization_id: str,
    site_id: str,
    submitted_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if submitted_config is not None:
        return normalize_analysis_config(submitted_config)

    row = store.get_analysis_config(organization_id)
    if row is not None:
        return normalize_analysis_config(json.loads(row["config_json"]))
    return build_analysis_config_from_tenant_assets(store, organization_id, site_id)
