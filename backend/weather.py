from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from urllib.parse import urlencode
from urllib.request import Request as HttpRequest, urlopen

import pandas as pd


NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
USER_AGENT = "RDS-Energy-Analytics/1.0 (energy analytics dashboard; automated degree-day lookup)"
DEFAULT_HDD_BASE_TEMPERATURE_C = 15.5
DEFAULT_CDD_BASE_TEMPERATURE_C = 22.0


@dataclass(frozen=True)
class DegreeDaySummary:
    start_date: str
    end_date: str
    days: int
    total_hdd: float
    total_cdd: float
    hdd_base_temperature_c: float
    cdd_base_temperature_c: float


def geocode_location(query: str, *, timeout: float = 10.0) -> tuple[float, float]:
    """Resolve a free-text address/postcode to (latitude, longitude) via Nominatim."""
    text = query.strip()
    if not text:
        raise ValueError("A site location is required to resolve coordinates")

    params = urlencode({"q": text, "format": "json", "limit": "1"})
    request = HttpRequest(f"{NOMINATIM_URL}?{params}", headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=timeout) as response:
            results = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # network/parse failures should surface as a usable error
        raise ValueError(f"Unable to resolve location '{text}': {exc}") from exc

    if not results:
        raise ValueError(f"No location found for '{text}'")
    try:
        return float(results[0]["lat"]), float(results[0]["lon"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Unexpected geocoding response for '{text}'") from exc


def fetch_daily_degree_days(
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    *,
    hdd_base_temperature_c: float = DEFAULT_HDD_BASE_TEMPERATURE_C,
    cdd_base_temperature_c: float = DEFAULT_CDD_BASE_TEMPERATURE_C,
    timeout: float = 15.0,
) -> pd.DataFrame:
    """Fetch daily mean temperature for a location/date range and derive HDD/CDD.

    HDD = max(0, hdd_base - mean_temp); CDD = max(0, mean_temp - cdd_base).
    Uses Open-Meteo's free historical weather archive (no API key required).
    """
    if start_date > end_date:
        raise ValueError("start_date must not be after end_date")

    params = urlencode(
        {
            "latitude": f"{latitude:.4f}",
            "longitude": f"{longitude:.4f}",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "daily": "temperature_2m_max,temperature_2m_min",
            "timezone": "UTC",
        }
    )
    request = HttpRequest(f"{OPEN_METEO_ARCHIVE_URL}?{params}", headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"Unable to fetch historical weather data: {exc}") from exc

    daily = payload.get("daily") or {}
    dates = daily.get("time") or []
    highs = daily.get("temperature_2m_max") or []
    lows = daily.get("temperature_2m_min") or []
    records = [
        (day, (high + low) / 2)
        for day, high, low in zip(dates, highs, lows)
        if high is not None and low is not None
    ]
    if not records:
        raise ValueError("No historical weather data was returned for this location and date range")

    frame = pd.DataFrame(records, columns=["date", "mean_temperature_c"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame["hdd"] = (hdd_base_temperature_c - frame["mean_temperature_c"]).clip(lower=0).round(2)
    frame["cdd"] = (frame["mean_temperature_c"] - cdd_base_temperature_c).clip(lower=0).round(2)
    return frame


def summarise_degree_days(
    frame: pd.DataFrame,
    *,
    hdd_base_temperature_c: float,
    cdd_base_temperature_c: float,
) -> DegreeDaySummary:
    return DegreeDaySummary(
        start_date=frame["date"].min().date().isoformat(),
        end_date=frame["date"].max().date().isoformat(),
        days=int(len(frame)),
        total_hdd=round(float(frame["hdd"].sum()), 2),
        total_cdd=round(float(frame["cdd"].sum()), 2),
        hdd_base_temperature_c=hdd_base_temperature_c,
        cdd_base_temperature_c=cdd_base_temperature_c,
    )
