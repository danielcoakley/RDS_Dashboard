from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from models import User, Objective
from schemas import ObjectiveCreate, ObjectiveOut, ObjectiveUpdate
from auth import get_current_user
from database import get_db

router = APIRouter(prefix="/api/objectives", tags=["objectives"])


@router.get("", response_model=list[ObjectiveOut])
def list_objectives(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    objs = db.query(Objective).filter(Objective.org_id == user.org_id).all()
    result = []
    for o in objs:
        out = ObjectiveOut.model_validate(o)
        result.append(out)
    return result


@router.post("", response_model=ObjectiveOut)
def create_objective(payload: ObjectiveCreate, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
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
    out = ObjectiveOut.model_validate(obj)

    return out


@router.put("/{obj_id}", response_model=ObjectiveOut)
def update_objective(obj_id: int, payload: ObjectiveUpdate, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    obj = db.query(Objective).filter(Objective.id == obj_id, Objective.org_id == user.org_id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Objective not found")
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(obj, field, value)
    db.commit()
    db.refresh(obj)
    out = ObjectiveOut.model_validate(obj)

    return out


@router.delete("/{obj_id}")
def delete_objective(obj_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    obj = db.query(Objective).filter(Objective.id == obj_id, Objective.org_id == user.org_id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Objective not found")
    db.delete(obj)
    db.commit()
    return {"ok": True}
