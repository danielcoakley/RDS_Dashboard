from typing import Any, Optional
from sqlalchemy.orm import Session
from models import AuditEvent, User


def log_event(
    db: Session, user: User, action: str,
    entity_type: Optional[str] = None, entity_id: Any = None,
    details: Optional[dict] = None, commit: bool = True,
) -> None:
    db.add(AuditEvent(
        org_id=user.org_id, user_id=user.id, user_email=user.email,
        action=action, entity_type=entity_type,
        entity_id=str(entity_id) if entity_id is not None else None,
        details=details or {},
    ))
    if commit:
        db.commit()
