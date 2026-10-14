from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from models import User, EnergyReview, Site, Meter, EnergyReading, WeatherData
from analytics import build_energy_review
from schemas import EnergyReviewCreate, EnergyReviewOut
from auth import get_current_user
from database import get_db

router = APIRouter(prefix="/api/energy-review", tags=["energy-review"])


def _out(r: EnergyReview) -> EnergyReviewOut:
    return EnergyReviewOut(id=r.id, site_id=r.site_id, review_data=r.review_data or {},
                           period_start=r.period_start, period_end=r.period_end,
                           created_at=r.created_at.isoformat())


def _org_site(db: Session, site_id: int, user: User) -> Site:
    site = db.query(Site).filter(Site.id == site_id, Site.org_id == user.org_id).first()
    if not site:
        raise HTTPException(status_code=404, detail="Site not found")
    return site


@router.get("/generate/{site_id}")
def generate_review(site_id: int, baseline_year: int, reporting_year: int,
                    user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Auto-generated ISO 50001 energy review (6.3 SEUs, 6.4 EnPIs, 6.5 baseline)."""
    site = _org_site(db, site_id, user)
    rows = (
        db.query(EnergyReading.date, EnergyReading.consumption, Meter.name, Meter.utility_type, Meter.seu_category)
        .join(Meter, EnergyReading.meter_id == Meter.id)
        .join(Site, Meter.site_id == Site.id)
        .filter(Site.id == site_id, Site.org_id == user.org_id)
        .all()
    )
    weather = (
        db.query(WeatherData.date, WeatherData.hdd, WeatherData.cdd)
        .join(Site, WeatherData.site_id == Site.id)
        .filter(Site.id == site_id, Site.org_id == user.org_id)
        .all()
    )
    readings = [{"date": r[0], "consumption": r[1], "meter": r[2], "utility_type": r[3], "seu_category": r[4]} for r in rows]
    wx = [{"date": w[0], "hdd": w[1], "cdd": w[2]} for w in weather]
    review = build_energy_review(readings, wx, baseline_year, reporting_year)
    review["site"] = {"id": site.id, "name": site.name}
    return review


@router.get("/{site_id}", response_model=list[EnergyReviewOut])
def list_reviews(site_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    _org_site(db, site_id, user)
    reviews = (db.query(EnergyReview)
               .filter(EnergyReview.site_id == site_id, EnergyReview.org_id == user.org_id)
               .order_by(EnergyReview.created_at.desc()).all())
    return [_out(r) for r in reviews]


@router.post("", response_model=EnergyReviewOut)
def create_review(payload: EnergyReviewCreate, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    site = db.query(Site).filter(Site.id == payload.site_id, Site.org_id == user.org_id).first()
    if not site:
        raise HTTPException(status_code=404, detail="Site not found")
    review = EnergyReview(
        org_id=user.org_id, site_id=payload.site_id,
        review_data=payload.review_data,
        period_start=payload.period_start, period_end=payload.period_end,
    )
    db.add(review)
    db.commit()
    db.refresh(review)
    return _out(review)


@router.put("/{review_id}", response_model=EnergyReviewOut)
def update_review(review_id: int, payload: EnergyReviewCreate, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    review = db.query(EnergyReview).filter(EnergyReview.id == review_id, EnergyReview.org_id == user.org_id).first()
    if not review:
        raise HTTPException(status_code=404, detail="Energy review not found")
    review.review_data = payload.review_data
    review.period_start = payload.period_start
    review.period_end = payload.period_end
    db.commit()
    db.refresh(review)
    return _out(review)


@router.delete("/{review_id}")
def delete_review(review_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    review = db.query(EnergyReview).filter(EnergyReview.id == review_id, EnergyReview.org_id == user.org_id).first()
    if not review:
        raise HTTPException(status_code=404, detail="Energy review not found")
    db.delete(review)
    db.commit()
    return {"ok": True}
