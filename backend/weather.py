import httpx
import numpy as np
from datetime import date, timedelta
from sqlalchemy.orm import Session
from models import Site, WeatherData

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

HDD_BASE_TEMP = 15.5  # °C
CDD_BASE_TEMP = 18.0   # °C


def calculate_hdd_cdd(temp_mean: float) -> tuple:
    """Calculate HDD and CDD from mean temperature."""
    if temp_mean is None:
        return None, None
    hdd = max(0, HDD_BASE_TEMP - temp_mean)
    cdd = max(0, temp_mean - CDD_BASE_TEMP)
    return round(hdd, 1), round(cdd, 1)


async def fetch_weather_for_site(db: Session, site: Site, start_date: date, end_date: date):
    """Fetch daily weather data from Open-Meteo and store HDD/CDD."""
    if site.latitude is None or site.longitude is None:
        return {"error": "Site has no coordinates"}

    params = {
        "latitude": site.latitude,
        "longitude": site.longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": "temperature_2m_mean,temperature_2m_max,temperature_2m_min",
        "timezone": site.timezone or "UTC",
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.get(OPEN_METEO_ARCHIVE_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    temps_mean = daily.get("temperature_2m_mean", [])
    temps_max = daily.get("temperature_2m_max", [])
    temps_min = daily.get("temperature_2m_min", [])

    upserted = 0
    for i, d in enumerate(dates):
        dt = date.fromisoformat(d)
        t_mean = temps_mean[i] if i < len(temps_mean) else None
        t_max = temps_max[i] if i < len(temps_max) else None
        t_min = temps_min[i] if i < len(temps_min) else None
        hdd, cdd = calculate_hdd_cdd(t_mean)

        existing = db.query(WeatherData).filter(
            WeatherData.site_id == site.id,
            WeatherData.date == dt,
        ).first()

        if existing:
            existing.temp_mean = t_mean
            existing.temp_max = t_max
            existing.temp_min = t_min
            existing.hdd = hdd
            existing.cdd = cdd
        else:
            wd = WeatherData(
                site_id=site.id, date=dt, temp_mean=t_mean,
                temp_max=t_max, temp_min=t_min, hdd=hdd, cdd=cdd,
            )
            db.add(wd)
        upserted += 1

    db.commit()
    return {"fetched_days": upserted, "date_range": f"{start_date} to {end_date}"}


async def geocode_address(address: str) -> dict:
    """Geocode an address using Open-Meteo's geocoding API."""
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": address, "count": 1, "language": "en", "format": "json"},
        )
        resp.raise_for_status()
        data = resp.json()

    results = data.get("results", [])
    if not results:
        return {}
    r = results[0]
    return {
        "latitude": r.get("latitude"),
        "longitude": r.get("longitude"),
        "timezone": r.get("timezone", "UTC"),
        "name": r.get("name"),
        "country": r.get("country"),
    }
