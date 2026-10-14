import io
import pandas as pd
from datetime import date, timedelta
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import Response
from sqlalchemy.orm import Session
from sqlalchemy import func
from models import User, Site, Meter, EnergyReading, WeatherData, DataUpload
from schemas import WeatherStatus, DataUploadOut, DataSummary
from auth import get_current_user, require_writer
from audit import log_event
from database import get_db
from weather import fetch_weather_for_site
from routers import data_smart

router = APIRouter(prefix="/api/data", tags=["data"])


@router.post("/upload/{site_id}")
async def upload_energy_data(
    site_id: int,
    file: UploadFile = File(...),
    user: User = Depends(require_writer),
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
        # Original RDS-style format:
        # Each meter has TWO rows — a "Meter" row with the name/utility, then a
        # "Day" row with the actual consumption values in date columns.
        # We find the "Meter" rows (where Utility is set) and read data from idx+1.
        meters_map = {}  # meter_name → meter obj

        for idx in range(len(df) - 1):
            row = df.iloc[idx]
            utility = str(row.get("Utility", "")).strip()
            meter_name = str(row.get("Metered Sector", "")).strip()

            # Skip rows without a utility type or meter name
            if not utility or utility == "nan" or not meter_name or meter_name == "nan":
                continue

            # The next row should be the "Day" data row
            data_row = df.iloc[idx + 1]
            if str(data_row.get("Period", "")).strip() != "Day":
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
            # Date columns are everything after the metadata columns
            date_cols = [c for c in df.columns if c not in ("Metered Sector", "Utility", "Units", "Period")]
            for col in date_cols:
                try:
                    dt = pd.to_datetime(col, dayfirst=True, errors="coerce")
                    if pd.isna(dt):
                        continue
                    val = data_row[col]
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

    # Record the uploaded file
    upload = DataUpload(
        org_id=user.org_id, site_id=site_id,
        filename=file.filename or "upload.csv", kind="energy",
        records=readings_added,
        detail=f"{meters_created} meters, {len(meters_map)} total",
    )
    db.add(upload)
    db.commit()
    log_event(db, user, "data.uploaded", "data_upload", upload.id,
              {"site_id": site_id, "filename": upload.filename, "kind": "energy", "records": readings_added})

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
async def refresh_weather(site_id: int, user: User = Depends(require_writer), db: Session = Depends(get_db)):
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


@router.post("/seu-mapping/{site_id}")
async def upload_seu_mapping(
    site_id: int,
    file: UploadFile = File(...),
    user: User = Depends(require_writer),
    db: Session = Depends(get_db),
):
    """Upload a SEU mapping CSV (Meter, SEU_Category) and apply it to existing meters."""
    site = db.query(Site).filter(Site.id == site_id, Site.org_id == user.org_id).first()
    if not site:
        raise HTTPException(status_code=404, detail="Site not found")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    required = ["Meter", "SEU_Category"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"SEU mapping must have columns: {', '.join(required)}")

    df["Meter"] = df["Meter"].astype(str).str.strip()

    # Remove meters marked "Ignore" from the site
    ignored = df[df["SEU_Category"].astype(str).str.strip() == "Ignore"]["Meter"].tolist()
    if ignored:
        for meter_name in ignored:
            meter = db.query(Meter).filter(
                Meter.site_id == site_id, Meter.name == meter_name
            ).first()
            if meter:
                db.delete(meter)

    # Update SEU categories for remaining meters
    updated = 0
    for _, row in df.iterrows():
        meter_name = str(row["Meter"]).strip()
        seu_category = str(row["SEU_Category"]).strip()
        if seu_category == "Ignore":
            continue
        meter = db.query(Meter).filter(
            Meter.site_id == site_id, Meter.name == meter_name
        ).first()
        if meter:
            meter.seu_category = seu_category
            updated += 1

    db.commit()

    # Record the uploaded SEU mapping file
    upload = DataUpload(
        org_id=user.org_id, site_id=site_id,
        filename=file.filename or "seu_mapping.csv", kind="seu_mapping",
        records=updated,
        detail=f"{updated} meters mapped",
    )
    db.add(upload)
    db.commit()
    log_event(db, user, "data.uploaded", "data_upload", upload.id,
              {"site_id": site_id, "filename": upload.filename, "kind": "seu_mapping", "records": updated})

    return {
        "updated": updated,
        "ignored": len(ignored),
        "message": f"Updated SEU categories for {updated} meters" + (f", removed {len(ignored)} ignored meters" if ignored else ""),
    }


@router.get("/uploads/{site_id}", response_model=list[DataUploadOut])
def list_uploads(site_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """List files uploaded for a site (energy data + SEU mappings)."""
    site = db.query(Site).filter(Site.id == site_id, Site.org_id == user.org_id).first()
    if not site:
        raise HTTPException(status_code=404, detail="Site not found")
    uploads = db.query(DataUpload).filter(DataUpload.site_id == site_id).order_by(DataUpload.uploaded_at.desc()).all()
    result = []
    for u in uploads:
        out = DataUploadOut(
            id=u.id, filename=u.filename, kind=u.kind, records=u.records,
            detail=u.detail, uploaded_at=u.uploaded_at.isoformat(),
        )
        result.append(out)
    return result


@router.get("/summary/{site_id}", response_model=DataSummary)
def data_summary(site_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Data coverage summary for a site: meters, readings, date range, weather, uploads."""
    site = db.query(Site).filter(Site.id == site_id, Site.org_id == user.org_id).first()
    if not site:
        raise HTTPException(status_code=404, detail="Site not found")

    meter_count = db.query(func.count(Meter.id)).filter(Meter.site_id == site_id).scalar() or 0
    total_readings = db.query(func.count(EnergyReading.id)).join(Meter).filter(Meter.site_id == site_id).scalar() or 0
    date_range = db.query(
        func.min(EnergyReading.date), func.max(EnergyReading.date)
    ).join(Meter).filter(Meter.site_id == site_id).first()
    weather_days = db.query(func.count(WeatherData.id)).filter(WeatherData.site_id == site_id).scalar() or 0
    uploads = db.query(DataUpload).filter(DataUpload.site_id == site_id).order_by(DataUpload.uploaded_at.desc()).all()

    return DataSummary(
        meters=meter_count,
        total_readings=total_readings,
        date_start=date_range[0].isoformat() if date_range and date_range[0] else None,
        date_end=date_range[1].isoformat() if date_range and date_range[1] else None,
        weather_days=weather_days,
        weather_status="available" if weather_days > 0 else "not_fetched",
        uploads=[
            DataUploadOut(
                id=u.id, filename=u.filename, kind=u.kind, records=u.records,
                detail=u.detail, uploaded_at=u.uploaded_at.isoformat(),
            ) for u in uploads
        ],
    )


# --- Smart upload: template, preview (mapping + quality), mapped import, sample data ---

def _org_site(db: Session, user: User, site_id: int) -> Site:
    site = db.query(Site).filter(Site.id == site_id, Site.org_id == user.org_id).first()
    if not site:
        raise HTTPException(status_code=404, detail="Site not found")
    return site


async def _fetch_site_weather(db: Session, site: Site):
    if site.latitude is None or site.longitude is None:
        return
    rng = db.query(func.min(EnergyReading.date), func.max(EnergyReading.date)).join(Meter).filter(Meter.site_id == site.id).first()
    if rng and rng[0] and rng[1]:
        try:
            await fetch_weather_for_site(db, site, rng[0], rng[1])
        except Exception:
            pass  # best-effort


def _upsert_readings(db: Session, site: Site, norm, default_utility: str = "electricity") -> tuple[int, int]:
    """Insert/update readings per meter (last value wins for duplicate dates). Returns (added, meters_created)."""
    added = created = 0
    for name, g in norm.groupby("meter"):
        meter = db.query(Meter).filter(Meter.site_id == site.id, Meter.name == name).first()
        if not meter:
            meter = Meter(site_id=site.id, name=name, units="kWh", seu_category="Unknown",
                          utility_type="gas" if "gas" in name.lower() else default_utility)
            db.add(meter)
            db.flush()
            created += 1
        existing = {r.date: r for r in db.query(EnergyReading).filter(EnergyReading.meter_id == meter.id)}
        for d, v in g.drop_duplicates(subset=["date"], keep="last")[["date", "value"]].itertuples(index=False):
            v = float(v)
            if d in existing:
                existing[d].consumption = v
            else:
                db.add(EnergyReading(meter_id=meter.id, date=d, consumption=v, is_operational=v > 0))
            added += 1
    return added, created


@router.get("/template")
def download_template():
    return Response(content=data_smart.TEMPLATE_CSV, media_type="text/csv",
                    headers={"Content-Disposition": "attachment; filename=energy_data_template.csv"})


@router.post("/preview/{site_id}")
async def preview_upload(
    site_id: int,
    file: UploadFile = File(...),
    date_col: str | None = Form(None),
    value_col: str | None = Form(None),
    meter_col: str | None = Form(None),
    meter_name: str = Form("Main Meter"),
    user: User = Depends(require_writer),
    db: Session = Depends(get_db),
):
    _org_site(db, user, site_id)
    try:
        df = data_smart.read_table(await file.read(), file.filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse file: {e}")
    if data_smart.is_rds_format(df):
        return {"format": "rds", "columns": list(df.columns), "rows": len(df)}
    detected = data_smart.detect_columns(df)
    mapping = {
        "date_col": date_col or detected["date_col"],
        "value_col": value_col or detected["value_col"],
        "meter_col": (meter_col or None) if date_col else detected["meter_col"],
    }
    preview = df.head(10).astype(object).where(df.head(10).notna(), None)
    resp = {
        "format": "tabular", "columns": list(df.columns), "detected": detected, "mapping": mapping,
        "preview": [{k: (str(v) if v is not None else None) for k, v in r.items()} for r in preview.to_dict("records")],
        "quality": None,
    }
    if mapping["date_col"] and mapping["value_col"]:
        norm = data_smart.normalize(df, mapping["date_col"], mapping["value_col"], mapping["meter_col"], meter_name)
        if len(norm):
            resp["quality"] = data_smart.quality_report(norm, len(df))
    return resp


@router.post("/import/{site_id}")
async def import_mapped(
    site_id: int,
    file: UploadFile = File(...),
    date_col: str = Form(...),
    value_col: str = Form(...),
    meter_col: str | None = Form(None),
    meter_name: str = Form("Main Meter"),
    user: User = Depends(require_writer),
    db: Session = Depends(get_db),
):
    site = _org_site(db, user, site_id)
    try:
        df = data_smart.read_table(await file.read(), file.filename)
        norm = data_smart.normalize(df, date_col, value_col, meter_col or None, meter_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not import: {e}")
    if norm.empty:
        raise HTTPException(status_code=400, detail="No valid rows found with this column mapping")
    added, created = _upsert_readings(db, site, norm)
    db.add(DataUpload(org_id=user.org_id, site_id=site_id, filename=file.filename or "upload.csv",
                      kind="energy", records=added, detail=f"{created} meters created"))
    db.commit()
    await _fetch_site_weather(db, site)
    meters = norm["meter"].nunique()
    return {"meters_created": created, "readings_added": added,
            "message": f"Imported {added} readings across {meters} meters"}


@router.post("/sample/{site_id}")
async def load_sample_data(site_id: int, user: User = Depends(require_writer), db: Session = Depends(get_db)):
    site = _org_site(db, user, site_id)
    meters = db.query(Meter).filter(Meter.site_id == site_id).all()
    if not meters:
        meters = [Meter(site_id=site_id, name="Main Electricity", utility_type="electricity", units="kWh", seu_category="Unknown"),
                  Meter(site_id=site_id, name="Boiler Gas", utility_type="gas", units="kWh", seu_category="Space Heating")]
        db.add_all(meters)
        db.flush()
    added = 0
    for i, m in enumerate(meters):
        existing = {r.date: r for r in db.query(EnergyReading).filter(EnergyReading.meter_id == m.id)}
        for d, v in data_smart.synthetic_series(m.utility_type, seed=site_id * 100 + i):
            if d in existing:
                existing[d].consumption = v
            else:
                db.add(EnergyReading(meter_id=m.id, date=d, consumption=v, is_operational=v > 0))
            added += 1
    db.add(DataUpload(org_id=user.org_id, site_id=site_id, filename="sample-data", kind="energy",
                      records=added, detail=f"Synthetic 2 years, {len(meters)} meters"))
    db.commit()
    await _fetch_site_weather(db, site)
    return {"readings_added": added, "meters": len(meters),
            "message": f"Loaded {added} sample readings across {len(meters)} meters"}
