from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from models import User, EnergyReview, Site
from schemas import EnergyReviewCreate, EnergyReviewOut
from auth import get_current_user
from database import get_db

router = APIRouter(prefix="/api/energy-review", tags=["energy-review"])


@router.get("/{site_id}", response_model=list[EnergyReviewOut])
def list_reviews(site_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    site = db.query(Site).filter(Site.id == site_id, Site.org_id == user.org_id).first()
    if not site:
        raise HTTPException(status_code=404, detail="Site not found")
    reviews = db.query(EnergyReview).filter(EnergyReview.site_id == site_id).all()
    result = []
    for r in reviews:
        out = EnergyReviewOut.model_validate(r)
        out.created_at = r.created_at.isoformat()
        result.append(out)
    return result


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
    out = EnergyReviewOut.model_validate(review)
    out.created_at = review.created_at.isoformat()
    return out


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
    out = EnergyReviewOut.model_validate(review)
    out.created_at = review.created_at.isoformat()
    return out


@router.delete("/{review_id}")
def delete_review(review_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    review = db.query(EnergyReview).filter(EnergyReview.id == review_id, EnergyReview.org_id == user.org_id).first()
    if not review:
        raise HTTPException(status_code=404, detail="Energy review not found")
    db.delete(review)
    db.commit()
    return {"ok": True}
