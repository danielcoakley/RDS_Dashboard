from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from models import User, ComplianceItem
from schemas import ComplianceItemOut, ComplianceUpdate
from auth import get_current_user
from database import get_db

router = APIRouter(prefix="/api/compliance", tags=["compliance"])

ISO_50001_CLAUSES = [
    {"clause_ref": "4.1", "clause_title": "Understanding the organization and its context"},
    {"clause_ref": "4.2", "clause_title": "Understanding needs of interested parties"},
    {"clause_ref": "4.3", "clause_title": "Determining scope of the EnMS"},
    {"clause_ref": "4.4", "clause_title": "Energy management system"},
    {"clause_ref": "5.1", "clause_title": "Leadership and commitment"},
    {"clause_ref": "5.2", "clause_title": "Energy policy"},
    {"clause_ref": "5.3", "clause_title": "Organization roles, responsibilities and authorities"},
    {"clause_ref": "6.1", "clause_title": "Actions to address risks and opportunities"},
    {"clause_ref": "6.2", "clause_title": "Objectives, energy targets and planning"},
    {"clause_ref": "6.3", "clause_title": "Energy review"},
    {"clause_ref": "6.4", "clause_title": "Energy performance indicators (EnPI)"},
    {"clause_ref": "6.5", "clause_title": "Energy baseline"},
    {"clause_ref": "6.6", "clause_title": "Planning for collection of energy data"},
    {"clause_ref": "7.1", "clause_title": "Resources"},
    {"clause_ref": "7.2", "clause_title": "Competence"},
    {"clause_ref": "7.3", "clause_title": "Awareness"},
    {"clause_ref": "7.4", "clause_title": "Communication"},
    {"clause_ref": "7.5", "clause_title": "Documented information"},
    {"clause_ref": "8.1", "clause_title": "Operational planning and control"},
    {"clause_ref": "8.2", "clause_title": "Design"},
    {"clause_ref": "8.3", "clause_title": "Procurement"},
    {"clause_ref": "9.1", "clause_title": "Monitoring, measurement, analysis and evaluation"},
    {"clause_ref": "9.2", "clause_title": "Internal audit"},
    {"clause_ref": "9.3", "clause_title": "Management review"},
    {"clause_ref": "10.1", "clause_title": "Nonconformity and corrective action"},
    {"clause_ref": "10.2", "clause_title": "Continual improvement"},
]


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

    title = next((c["clause_title"] for c in ISO_50001_CLAUSES if c["clause_ref"] == clause_ref), clause_ref)

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
