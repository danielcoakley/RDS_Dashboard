from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from models import User, Site, Meter, EnergyReading
from schemas import SiteCreate, SiteOut, SiteDetail, MeterCreate, MeterOut
from auth import get_current_user
from database import get_db
from weather import geocode_address

router = APIRouter(prefix="/api/sites", tags=["sites"])


@router.get("", response_model=list[SiteOut])
def list_sites(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    sites = db.query(Site).filter(Site.org_id == user.org_id).all()
    result = []
    for s in sites:
        meter_count = db.query(func.count(Meter.id)).filter(Meter.site_id == s.id).scalar() or 0
        out = SiteOut.model_validate(s)
        out.meter_count = meter_count
        out.created_at = s.created_at.isoformat()
        result.append(out)
    return result


@router.post("", response_model=SiteOut)
async def create_site(payload: SiteCreate, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    lat, lon, tz = payload.latitude, payload.longitude, payload.timezone

    # Auto-geocode if address provided but no coordinates
    if payload.address and lat is None:
        geo = await geocode_address(payload.address)
        lat = geo.get("latitude")
        lon = geo.get("longitude")
        tz = geo.get("timezone", "UTC")

    site = Site(
        org_id=user.org_id, name=payload.name, address=payload.address,
        latitude=lat, longitude=lon, timezone=tz or "UTC",
    )
    db.add(site)
    db.commit()
    db.refresh(site)

    out = SiteOut.model_validate(site)
    out.meter_count = 0
    out.created_at = site.created_at.isoformat()
    return out


@router.get("/{site_id}", response_model=SiteDetail)
def get_site(site_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    site = db.query(Site).filter(Site.id == site_id, Site.org_id == user.org_id).first()
    if not site:
        raise HTTPException(status_code=404, detail="Site not found")

    meter_count = db.query(func.count(Meter.id)).filter(Meter.site_id == site.id).scalar() or 0
    latest = db.query(EnergyReading).join(Meter).filter(Meter.site_id == site.id).order_by(EnergyReading.date.desc()).first()

    out = SiteDetail.model_validate(site)
    out.meter_count = meter_count
    out.created_at = site.created_at.isoformat()
    out.latest_reading = latest.date.isoformat() if latest else None
    return out


@router.delete("/{site_id}")
def delete_site(site_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    site = db.query(Site).filter(Site.id == site_id, Site.org_id == user.org_id).first()
    if not site:
        raise HTTPException(status_code=404, detail="Site not found")
    db.delete(site)
    db.commit()
    return {"ok": True}


# --- Meters ---

@router.get("/{site_id}/meters", response_model=list[MeterOut])
def list_meters(site_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    site = db.query(Site).filter(Site.id == site_id, Site.org_id == user.org_id).first()
    if not site:
        raise HTTPException(status_code=404, detail="Site not found")
    meters = db.query(Meter).filter(Meter.site_id == site_id).all()
    result = []
    for m in meters:
        count = db.query(func.count(EnergyReading.id)).filter(EnergyReading.meter_id == m.id).scalar() or 0
        out = MeterOut.model_validate(m)
        out.reading_count = count
        out.created_at = m.created_at.isoformat()
        result.append(out)
    return result


@router.post("/{site_id}/meters", response_model=MeterOut)
def create_meter(site_id: int, payload: MeterCreate, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    site = db.query(Site).filter(Site.id == site_id, Site.org_id == user.org_id).first()
    if not site:
        raise HTTPException(status_code=404, detail="Site not found")
    meter = Meter(
        site_id=site_id, name=payload.name, utility_type=payload.utility_type,
        units=payload.units, seu_category=payload.seu_category,
    )
    db.add(meter)
    db.commit()
    db.refresh(meter)
    out = MeterOut.model_validate(meter)
    out.reading_count = 0
    out.created_at = meter.created_at.isoformat()
    return out


@router.delete("/{site_id}/meters/{meter_id}")
def delete_meter(site_id: int, meter_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    meter = db.query(Meter).filter(Meter.id == meter_id, Meter.site_id == site_id).first()
    if not meter:
        raise HTTPException(status_code=404, detail="Meter not found")
    db.delete(meter)
    db.commit()
    return {"ok": True}
