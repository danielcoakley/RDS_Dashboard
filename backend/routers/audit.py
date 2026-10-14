import csv
import io
import json
from datetime import date, datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from models import User, AuditEvent
from auth import get_current_user
from database import get_db

router = APIRouter(prefix="/api/audit", tags=["audit"])


def _filtered(db: Session, user: User, action, entity_type, user_id, start, end):
    q = db.query(AuditEvent).filter(AuditEvent.org_id == user.org_id)
    if action:
        q = q.filter(AuditEvent.action == action)
    if entity_type:
        q = q.filter(AuditEvent.entity_type == entity_type)
    if user_id:
        q = q.filter(AuditEvent.user_id == user_id)
    if start:
        q = q.filter(AuditEvent.created_at >= datetime.combine(start, datetime.min.time()))
    if end:
        q = q.filter(AuditEvent.created_at < datetime.combine(end + timedelta(days=1), datetime.min.time()))
    return q.order_by(AuditEvent.created_at.desc())


def _out(e: AuditEvent) -> dict:
    return {
        "id": e.id, "user_id": e.user_id, "user_email": e.user_email, "action": e.action,
        "entity_type": e.entity_type, "entity_id": e.entity_id, "details": e.details or {},
        "created_at": e.created_at.isoformat(),
    }


@router.get("")
def list_events(
    action: Optional[str] = None, entity_type: Optional[str] = None, user_id: Optional[int] = None,
    start: Optional[date] = None, end: Optional[date] = None,
    limit: int = Query(200, le=1000), offset: int = 0,
    user: User = Depends(get_current_user), db: Session = Depends(get_db),
):
    q = _filtered(db, user, action, entity_type, user_id, start, end)
    return {"total": q.count(), "events": [_out(e) for e in q.offset(offset).limit(limit).all()]}


@router.get("/actions")
def list_actions(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.query(AuditEvent.action).filter(AuditEvent.org_id == user.org_id).distinct().all()
    return sorted(r[0] for r in rows)


@router.get("/export")
def export_csv(
    action: Optional[str] = None, entity_type: Optional[str] = None, user_id: Optional[int] = None,
    start: Optional[date] = None, end: Optional[date] = None,
    user: User = Depends(get_current_user), db: Session = Depends(get_db),
):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["timestamp_utc", "user", "action", "entity_type", "entity_id", "details"])
    for e in _filtered(db, user, action, entity_type, user_id, start, end).all():
        w.writerow([e.created_at.isoformat(), e.user_email, e.action, e.entity_type or "", e.entity_id or "",
                    json.dumps(e.details or {})])
    return StreamingResponse(
        iter([buf.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=audit-log.csv"},
    )
