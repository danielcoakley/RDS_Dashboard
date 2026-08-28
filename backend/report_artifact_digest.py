from __future__ import annotations

from typing import Any


def _digest_value(value: Any) -> str | None:
    if value is None or isinstance(value, (dict, list)):
        return None
    text = str(value).strip()
    return text or None


def report_artifact_digest_lines(content: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    total_records = _digest_value(content.get("total_records"))
    if total_records is not None:
        lines.append(f"Total records: {total_records}")

    source_package = content.get("source_package")
    if isinstance(source_package, dict):
        upload_count = _digest_value(source_package.get("upload_count"))
        if upload_count is not None:
            lines.append(f"Source files: {upload_count}")
        categories = source_package.get("categories")
        if isinstance(categories, list):
            safe_categories = [
                str(category).strip()
                for category in categories
                if isinstance(category, str) and category.strip()
            ]
            if safe_categories:
                lines.append(f"Source categories: {', '.join(safe_categories[:8])}")

    legacy_baseline = content.get("legacy_baseline")
    weather_normalization = (
        legacy_baseline.get("weather_normalization")
        if isinstance(legacy_baseline, dict)
        else None
    )
    if isinstance(weather_normalization, dict):
        status = _digest_value(weather_normalization.get("status"))
        if status is not None:
            lines.append(f"Weather normalization: {status}")
        model_count = _digest_value(weather_normalization.get("model_count"))
        if model_count is not None:
            lines.append(f"Weather models: {model_count}")
        skipped_count = _digest_value(weather_normalization.get("skipped_meter_count"))
        if skipped_count is not None:
            lines.append(f"Skipped weather models: {skipped_count}")
        estimated_savings = _digest_value(weather_normalization.get("estimated_savings"))
        if estimated_savings is not None:
            lines.append(f"Estimated normalized savings: {estimated_savings}")

    return lines
