from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from models import User, Objective, Site
from schemas import ObjectiveCreate, ObjectiveOut, ObjectiveUpdate
from auth import get_current_user, require_writer
from audit import log_event
from database import get_db

router = APIRouter(prefix="/api/objectives", tags=["objectives"])


def _check_site(db: Session, site_id, org_id: int):
    """Reject site ids that do not belong to the caller's organization."""
    if site_id is None:
        return
    if not db.query(Site).filter(Site.id == site_id, Site.org_id == org_id).first():
        raise HTTPException(status_code=404, detail="Site not found")


@router.get("", response_model=list[ObjectiveOut])
def list_objectives(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    objs = db.query(Objective).filter(Objective.org_id == user.org_id).all()
    result = []
    for o in objs:
        out = ObjectiveOut.model_validate(o)
        result.append(out)
    return result


@router.post("", response_model=ObjectiveOut)
def create_objective(payload: ObjectiveCreate, user: User = Depends(require_writer), db: Session = Depends(get_db)):
    _check_site(db, payload.site_id, user.org_id)
    obj = Objective(
        org_id=user.org_id,
        site_id=payload.site_id,
        title=payload.title,
        description=payload.description,
        target_pct=payload.target_pct,
        baseline_value=payload.baseline_value,
        deadline=payload.deadline,
        status="active",
    )
    db.add(obj)
    db.commit()
    db.refresh(obj)
    log_event(db, user, "objective.created", "objective", obj.id, {"title": obj.title})
    out = ObjectiveOut.model_validate(obj)

    return out


@router.put("/{obj_id}", response_model=ObjectiveOut)
def update_objective(obj_id: int, payload: ObjectiveUpdate, user: User = Depends(require_writer), db: Session = Depends(get_db)):
    obj = db.query(Objective).filter(Objective.id == obj_id, Objective.org_id == user.org_id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Objective not found")
    updates = payload.model_dump(exclude_unset=True)
    if "site_id" in updates:
        _check_site(db, updates["site_id"], user.org_id)
    changes = {}
    for field, value in updates.items():
        old = getattr(obj, field)
        if old != value:
            changes[field] = {"from": str(old) if old is not None else None, "to": str(value) if value is not None else None}
        setattr(obj, field, value)
    db.commit()
    db.refresh(obj)
    if changes:
        log_event(db, user, "objective.updated", "objective", obj.id, {"title": obj.title, "changes": changes})
    out = ObjectiveOut.model_validate(obj)

    return out


@router.delete("/{obj_id}")
def delete_objective(obj_id: int, user: User = Depends(require_writer), db: Session = Depends(get_db)):
    obj = db.query(Objective).filter(Objective.id == obj_id, Objective.org_id == user.org_id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Objective not found")
    title = obj.title
    db.delete(obj)
    log_event(db, user, "objective.deleted", "objective", obj_id, {"title": title}, commit=False)
    db.commit()
    return {"ok": True}
