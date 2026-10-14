from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from models import User, ComplianceItem
from schemas import ComplianceItemOut, ComplianceUpdate
from auth import get_current_user
from database import get_db

router = APIRouter(prefix="/api/compliance", tags=["compliance"])

STATUSES = {"not_started", "in_progress", "complete"}
WEIGHT = {"complete": 1.0, "in_progress": 0.5}

from iso_clauses import ISO_50001_CLAUSES, ISO_SECTIONS, CLAUSE_TITLES


@router.get("", response_model=list[ComplianceItemOut])
def list_compliance(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    items = db.query(ComplianceItem).filter(ComplianceItem.org_id == user.org_id).all()
    existing_refs = {item.clause_ref: item for item in items}

    result = []
    for clause in ISO_50001_CLAUSES:
        item = existing_refs.get(clause["clause_ref"])
        if item:
            result.append(ComplianceItemOut(
                id=item.id, clause_ref=item.clause_ref,
                clause_title=item.clause_title or clause["clause_title"],
                status=item.status, evidence=item.evidence,
            ))
        else:
            result.append(ComplianceItemOut(
                clause_ref=clause["clause_ref"],
                clause_title=clause["clause_title"],
                status="not_started", evidence=None,
            ))
    return result


@router.put("/{clause_ref}")
def update_compliance(clause_ref: str, payload: ComplianceUpdate, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    item = db.query(ComplianceItem).filter(
        ComplianceItem.org_id == user.org_id,
        ComplianceItem.clause_ref == clause_ref,
    ).first()

    if clause_ref not in CLAUSE_TITLES:
        raise HTTPException(status_code=404, detail="Unknown clause")
    if payload.status not in STATUSES:
        raise HTTPException(status_code=400, detail="Invalid status")
    title = CLAUSE_TITLES[clause_ref]

    if item:
        item.status = payload.status
        item.evidence = payload.evidence
    else:
        item = ComplianceItem(
            org_id=user.org_id, clause_ref=clause_ref,
            clause_title=title, status=payload.status,
            evidence=payload.evidence,
        )
        db.add(item)
    db.commit()
    db.refresh(item)
    return {"ok": True, "id": item.id}


@router.get("/score")
def compliance_score(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    items = db.query(ComplianceItem).filter(ComplianceItem.org_id == user.org_id).all()
    total = len(ISO_50001_CLAUSES)
    complete = sum(1 for i in items if i.status == "complete")
    in_progress = sum(1 for i in items if i.status == "in_progress")
    return {
        "total_clauses": total,
        "complete": complete,
        "in_progress": in_progress,
        "not_started": total - complete - in_progress,
        "score": round(100 * complete / total, 0) if total else 0,
    }


@router.get("/readiness")
def compliance_readiness(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    items = db.query(ComplianceItem).filter(ComplianceItem.org_id == user.org_id).all()
    status_by_ref = {i.clause_ref: i.status for i in items}
    clauses = [{**c, "status": status_by_ref.get(c["clause_ref"], "not_started")} for c in ISO_50001_CLAUSES]

    def score(cs):
        return round(100 * sum(WEIGHT.get(c["status"], 0) for c in cs) / len(cs)) if cs else 0

    by_section = []
    for sec, title in ISO_SECTIONS.items():
        cs = [c for c in clauses if c["clause_ref"].split(".")[0] == sec]
        by_section.append({
            "section": sec, "title": title, "score": score(cs), "total": len(cs),
            "complete": sum(c["status"] == "complete" for c in cs),
        })
    counts = {s: sum(c["status"] == s for c in clauses) for s in ("complete", "in_progress", "not_started")}
    # In-progress first (closest to done), then not started, in clause order
    incomplete = [c for c in clauses if c["status"] == "in_progress"] + [c for c in clauses if c["status"] == "not_started"]
    return {
        "score": score(clauses),
        "total": len(clauses),
        "counts": counts,
        "by_section": by_section,
        "next_actions": incomplete[:5],
    }
