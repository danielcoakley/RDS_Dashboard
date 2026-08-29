import io
import pandas as pd
from datetime import date, timedelta
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from sqlalchemy import func
from models import User, Site, Meter, EnergyReading, WeatherData
from schemas import WeatherStatus
from auth import get_current_user
from database import get_db
from weather import fetch_weather_for_site

router = APIRouter(prefix="/api/data", tags=["data"])


@router.post("/upload/{site_id}")
async def upload_energy_data(
    site_id: int,
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    site = db.query(Site).filter(Site.id == site_id, Site.org_id == user.org_id).first()
    if not site:
        raise HTTPException(status_code=404, detail="Site not found")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content), encoding="latin1")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    # Normalize column names
    df.columns = [str(c).strip().strip("'").strip('"') for c in df.columns]

    # Try to detect the format — two supported formats:
    # 1. Simple: date, meter, consumption columns
    # 2. Original format: Metered Sector, Utility, Units, Period, then date columns

    meters_created = 0
    readings_added = 0

    if "Metered Sector" in df.columns and "Utility" in df.columns:
        # Original RDS-style format
        meters_map = {}  # meter_name → meter obj

        for idx, row in df.iterrows():
            utility = str(row.get("Utility", "")).strip()
            if not utility or str(row.get("Period", "")).strip() != "Day":
                # Only process daily period rows; the meter name is in the previous row
                continue

            meter_name = str(row.get("Metered Sector", "")).strip()
            if not meter_name or meter_name == "nan":
                continue

            utility_type = "gas" if "gas" in utility.lower() else "electricity"

            # Get or create meter
            if meter_name not in meters_map:
                meter = db.query(Meter).filter(
                    Meter.site_id == site_id, Meter.name == meter_name
                ).first()
                if not meter:
                    meter = Meter(
                        site_id=site_id, name=meter_name,
                        utility_type=utility_type, units="kWh",
                        seu_category="Unknown",
                    )
                    db.add(meter)
                    db.flush()
                    meters_created += 1
                meters_map[meter_name] = meter

            meter = meters_map[meter_name]
            # Date columns are everything after the first 4 metadata columns
            date_cols = [c for c in df.columns if c not in ("Metered Sector", "Utility", "Units", "Period")]
            for col in date_cols:
                try:
                    dt = pd.to_datetime(col, dayfirst=True, errors="coerce")
                    if pd.isna(dt):
                        continue
                    val = row[col]
                    if pd.isna(val) or str(val).strip() == "":
                        continue
                    consumption = float(val)

                    existing = db.query(EnergyReading).filter(
                        EnergyReading.meter_id == meter.id,
                        EnergyReading.date == dt.date(),
                    ).first()
                    if existing:
                        existing.consumption = consumption
                    else:
                        db.add(EnergyReading(
                            meter_id=meter.id, date=dt.date(),
                            consumption=consumption, is_operational=consumption > 0,
                        ))
                    readings_added += 1
                except (ValueError, TypeError):
                    continue
    else:
        # Simple format: date, meter, consumption
        date_col = next((c for c in df.columns if "date" in c.lower()), None)
        meter_col = next((c for c in df.columns if "meter" in c.lower()), None)
        consumption_col = next((c for c in df.columns if "consumption" in c.lower() or "kwh" in c.lower() or "usage" in c.lower()), None)

        if not all([date_col, meter_col, consumption_col]):
            raise HTTPException(status_code=400, detail="CSV must have date, meter, and consumption columns (or the original Metered Sector format)")

        meters_map = {}
        for _, row in df.iterrows():
            meter_name = str(row[meter_col]).strip()
            dt = pd.to_datetime(row[date_col], dayfirst=True, errors="coerce")
            consumption = pd.to_numeric(row[consumption_col], errors="coerce")
            if pd.isna(dt) or pd.isna(consumption):
                continue

            if meter_name not in meters_map:
                meter = db.query(Meter).filter(
                    Meter.site_id == site_id, Meter.name == meter_name
                ).first()
                if not meter:
                    meter = Meter(
                        site_id=site_id, name=meter_name,
                        utility_type="electricity", units="kWh",
                        seu_category="Unknown",
                    )
                    db.add(meter)
                    db.flush()
                    meters_created += 1
                meters_map[meter_name] = meter

            meter = meters_map[meter_name]
            existing = db.query(EnergyReading).filter(
                EnergyReading.meter_id == meter.id,
                EnergyReading.date == dt.date(),
            ).first()
            if existing:
                existing.consumption = float(consumption)
            else:
                db.add(EnergyReading(
                    meter_id=meter.id, date=dt.date(),
                    consumption=float(consumption), is_operational=float(consumption) > 0,
                ))
            readings_added += 1

    db.commit()

    # Auto-fetch weather data for the date range of uploaded readings
    if site.latitude is not None and site.longitude is not None:
        date_range = db.query(
            func.min(EnergyReading.date), func.max(EnergyReading.date)
        ).join(Meter).filter(Meter.site_id == site_id).first()
        if date_range and date_range[0] and date_range[1]:
            try:
                await fetch_weather_for_site(db, site, date_range[0], date_range[1])
            except Exception:
                pass  # Weather fetch is best-effort

    return {
        "meters_created": meters_created,
        "readings_added": readings_added,
        "message": f"Uploaded {readings_added} readings across {len(meters_map)} meters",
    }


@router.post("/weather/{site_id}/fetch")
async def refresh_weather(site_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    site = db.query(Site).filter(Site.id == site_id, Site.org_id == user.org_id).first()
    if not site:
        raise HTTPException(status_code=404, detail="Site not found")
    if site.latitude is None or site.longitude is None:
        raise HTTPException(status_code=400, detail="Site has no coordinates. Set an address with geocoding.")

    # Fetch for the last 3 years by default
    end = date.today()
    start = end - timedelta(days=365 * 3)
    result = await fetch_weather_for_site(db, site, start, end)
    return result


@router.get("/weather/{site_id}/status", response_model=WeatherStatus)
def weather_status(site_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    site = db.query(Site).filter(Site.id == site_id, Site.org_id == user.org_id).first()
    if not site:
        raise HTTPException(status_code=404, detail="Site not found")
    latest = db.query(WeatherData).filter(WeatherData.site_id == site_id).order_by(WeatherData.date.desc()).first()
    count = db.query(func.count(WeatherData.id)).filter(WeatherData.site_id == site_id).scalar() or 0
    return WeatherStatus(
        site_id=site_id,
        latest_date=latest.date.isoformat() if latest else None,
        total_days=count,
        status="available" if count > 0 else "not_fetched",
    )
