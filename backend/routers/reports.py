"""ISO 50001 management review (9.3) report data and internal audit findings (9.2)."""
from datetime import date, datetime
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth import get_current_user
from database import get_db
from models import AuditFinding, ComplianceItem, Objective, Organization, User
from routers.analytics import _compute_all, _load_site_data
from routers.compliance import ISO_50001_CLAUSES

router = APIRouter(prefix="/api/reports", tags=["reports"])

EF_KG_PER_KWH = {"electricity": 0.30, "gas": 0.20}

Severity = Literal["major", "minor", "observation"]
Status = Literal["open", "in_progress", "closed"]


class FindingIn(BaseModel):
    clause_ref: str
    finding: str
    severity: Severity = "observation"
    corrective_action: Optional[str] = None
    owner: Optional[str] = None
    due_date: Optional[date] = None
    status: Status = "open"


class FindingOut(FindingIn):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


def _get_finding(db: Session, fid: int, org_id: int) -> AuditFinding:
    f = db.query(AuditFinding).filter(AuditFinding.id == fid, AuditFinding.org_id == org_id).first()
    if not f:
        raise HTTPException(status_code=404, detail="Finding not found")
    return f


@router.get("/audit-findings", response_model=list[FindingOut])
def list_findings(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return (db.query(AuditFinding).filter(AuditFinding.org_id == user.org_id)
            .order_by(AuditFinding.created_at.desc()).all())


@router.post("/audit-findings", response_model=FindingOut)
def create_finding(payload: FindingIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    f = AuditFinding(org_id=user.org_id, **payload.model_dump())
    db.add(f)
    db.commit()
    db.refresh(f)
    return f


@router.put("/audit-findings/{fid}", response_model=FindingOut)
def update_finding(fid: int, payload: FindingIn, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    f = _get_finding(db, fid, user.org_id)
    for k, v in payload.model_dump().items():
        setattr(f, k, v)
    db.commit()
    db.refresh(f)
    return f


@router.delete("/audit-findings/{fid}")
def delete_finding(fid: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    db.delete(_get_finding(db, fid, user.org_id))
    db.commit()
    return {"ok": True}


def _sum_months(charts, key):
    out = [0.0] * 12
    for c in charts:
        for i, v in enumerate(c.get(key, [])):
            out[i] += float(v or 0)
    return out


@router.get("/management-review/{site_id}")
def management_review(
    site_id: int,
    baseline_year: int = Query(...),
    comparison_year: int = Query(...),
    elec_price: float = Query(0.25, ge=0, description="EUR per kWh electricity"),
    gas_price: float = Query(0.08, ge=0, description="EUR per kWh gas"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    site, meters, gas_df, elec_df = _load_site_data(db, site_id, user.org_id)
    org = db.query(Organization).filter(Organization.id == user.org_id).first()
    res = _compute_all(gas_df, elec_df, baseline_year, comparison_year)
    t = res["analysis"]["totals"]

    def util(name, pred, actual, price):
        saved = pred - actual
        return {
            "predicted_kwh": pred, "actual_kwh": actual, "savings_kwh": round(saved, 1),
            "savings_pct": round(100 * saved / pred, 1) if pred else 0,
            "savings_eur": round(saved * price, 2), "price_eur_per_kwh": price,
            "savings_tco2e": round(saved * EF_KG_PER_KWH[name] / 1000, 3),
            "emission_factor_kg_per_kwh": EF_KG_PER_KWH[name],
        }

    savings = {
        "electricity": util("electricity", t["elec_predicted"], t["elec_actual"], elec_price),
        "gas": util("gas", t["gas_predicted"], t["gas_actual"], gas_price),
    }
    savings["total"] = {k: round(savings["electricity"][k] + savings["gas"][k], 3)
                        for k in ("predicted_kwh", "actual_kwh", "savings_kwh", "savings_eur", "savings_tco2e")}
    tp = savings["total"]["predicted_kwh"]
    savings["total"]["savings_pct"] = round(100 * savings["total"]["savings_kwh"] / tp, 1) if tp else 0

    monthly = res["monthly"]
    trend = {"months": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]}
    for u in ("electricity", "gas"):
        trend[u] = {k: _sum_months(monthly[u], k) for k in ("baseline", "actual", "predicted")}

    seu = []
    for u in ("electricity", "gas"):
        for r in res["seuSummary"][u]:
            seu.append({
                "utility": u, "seu": r["seu_category"],
                "actual_kwh": round(float(r["actual"]), 1),
                "predicted_kwh": round(float(r["predicted"]), 1),
                "savings_kwh": round(float(r["predicted"]) - float(r["actual"]), 1),
            })
    seu.sort(key=lambda r: r["actual_kwh"], reverse=True)

    objectives = [{
        "title": o.title, "target_pct": o.target_pct, "baseline_value": o.baseline_value,
        "current_value": o.current_value, "deadline": o.deadline, "status": o.status,
    } for o in db.query(Objective).filter(Objective.org_id == user.org_id,
                                          (Objective.site_id == site_id) | (Objective.site_id.is_(None))).all()]

    stored = {c.clause_ref: c for c in db.query(ComplianceItem).filter(ComplianceItem.org_id == user.org_id).all()}
    compliance = [{
        "clause_ref": c["clause_ref"], "clause_title": c["clause_title"],
        "status": stored[c["clause_ref"]].status if c["clause_ref"] in stored else "not_started",
    } for c in ISO_50001_CLAUSES]

    findings = db.query(AuditFinding).filter(AuditFinding.org_id == user.org_id,
                                              AuditFinding.status != "closed").all()
    open_ncs = [FindingOut.model_validate(f).model_dump() for f in findings]

    return {
        "organization": org.name if org else "",
        "site": {"id": site.id, "name": site.name, "address": site.address},
        "baseline_year": baseline_year, "comparison_year": comparison_year,
        "generated_at": datetime.utcnow().isoformat(),
        "meter_count": len(meters),
        "savings": savings, "trend": trend, "seu": seu,
        "objectives": objectives, "compliance": compliance, "open_findings": open_ncs,
    }
