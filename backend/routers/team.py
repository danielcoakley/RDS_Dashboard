import secrets
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from models import User, Invite, Organization, ComplianceItem, AuditEvent
from schemas import TokenResponse, UserOut
from auth import (
    get_current_user, require_owner, hash_password, create_access_token, ROLES, TEAM_ROLES,
)
from audit import log_event
from database import get_db

router = APIRouter(prefix="/api/team", tags=["team"])

INVITE_TTL_DAYS = 7


class InviteCreate(BaseModel):
    email: EmailStr
    role: str = "viewer"
    team_role: Optional[str] = None


class MemberUpdate(BaseModel):
    role: Optional[str] = None
    team_role: Optional[str] = None  # "" clears the label


class InviteAccept(BaseModel):
    name: str
    password: str


class OrgUpdate(BaseModel):
    name: Optional[str] = None
    sector: Optional[str] = None
    country: Optional[str] = None


def _check_role(role: Optional[str]):
    if role is not None and role not in ROLES:
        raise HTTPException(status_code=400, detail=f"Invalid role. Use one of: {', '.join(ROLES)}")


def _check_team_role(team_role: Optional[str]):
    if team_role and team_role not in TEAM_ROLES:
        raise HTTPException(status_code=400, detail=f"Invalid team role. Use one of: {', '.join(TEAM_ROLES)}")


def _member_out(u: User) -> dict:
    return {
        "id": u.id, "name": u.name, "email": u.email, "role": u.role,
        "team_role": u.team_role, "created_at": u.created_at.isoformat() if u.created_at else None,
    }


def _invite_out(i: Invite) -> dict:
    return {
        "id": i.id, "email": i.email, "role": i.role, "team_role": i.team_role, "token": i.token,
        "expires_at": i.expires_at.isoformat(), "created_at": i.created_at.isoformat() if i.created_at else None,
        "expired": i.expires_at < datetime.utcnow(),
    }


@router.get("")
def get_team(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    members = db.query(User).filter(User.org_id == user.org_id).order_by(User.created_at).all()
    result = {"members": [_member_out(m) for m in members], "invites": []}
    if user.role == "owner":
        invites = db.query(Invite).filter(
            Invite.org_id == user.org_id, Invite.accepted_at.is_(None),
        ).order_by(Invite.created_at.desc()).all()
        result["invites"] = [_invite_out(i) for i in invites]
    return result


@router.post("/invites")
def create_invite(payload: InviteCreate, user: User = Depends(require_owner), db: Session = Depends(get_db)):
    _check_role(payload.role)
    _check_team_role(payload.team_role)
    email = payload.email.lower()
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=400, detail="A user with this email already exists")
    # Replace any pending invite for the same email in this org
    db.query(Invite).filter(
        Invite.org_id == user.org_id, Invite.email == email, Invite.accepted_at.is_(None),
    ).delete()
    invite = Invite(
        org_id=user.org_id, email=email, role=payload.role, team_role=payload.team_role or None,
        token=secrets.token_urlsafe(32), invited_by_id=user.id,
        expires_at=datetime.utcnow() + timedelta(days=INVITE_TTL_DAYS),
    )
    db.add(invite)
    db.commit()
    db.refresh(invite)
    log_event(db, user, "team.invite_created", "invite", invite.id, {"email": email, "role": payload.role})
    return _invite_out(invite)


@router.delete("/invites/{invite_id}")
def revoke_invite(invite_id: int, user: User = Depends(require_owner), db: Session = Depends(get_db)):
    invite = db.query(Invite).filter(Invite.id == invite_id, Invite.org_id == user.org_id).first()
    if not invite:
        raise HTTPException(status_code=404, detail="Invite not found")
    email = invite.email
    db.delete(invite)
    log_event(db, user, "team.invite_revoked", "invite", invite_id, {"email": email}, commit=False)
    db.commit()
    return {"ok": True}


@router.patch("/members/{member_id}")
def update_member(member_id: int, payload: MemberUpdate, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    member = db.query(User).filter(User.id == member_id, User.org_id == user.org_id).first()
    if not member:
        raise HTTPException(status_code=404, detail="Member not found")

    if payload.role is not None and payload.role != member.role:
        if user.role != "owner":
            raise HTTPException(status_code=403, detail="Only owners can change roles")
        _check_role(payload.role)
        if member.role == "owner":
            owners = db.query(User).filter(User.org_id == user.org_id, User.role == "owner").count()
            if owners <= 1:
                raise HTTPException(status_code=400, detail="The organization must keep at least one owner")
        old = member.role
        member.role = payload.role
        log_event(db, user, "team.role_changed", "user", member.id,
                  {"email": member.email, "from": old, "to": payload.role}, commit=False)

    if payload.team_role is not None:
        # Owners label anyone; other users may only set their own label
        if user.role != "owner" and member.id != user.id:
            raise HTTPException(status_code=403, detail="Only owners can change other members' team roles")
        _check_team_role(payload.team_role)
        new = payload.team_role or None
        if new != member.team_role:
            log_event(db, user, "team.team_role_changed", "user", member.id,
                      {"email": member.email, "from": member.team_role, "to": new}, commit=False)
            member.team_role = new

    db.commit()
    db.refresh(member)
    return _member_out(member)


@router.delete("/members/{member_id}")
def remove_member(member_id: int, user: User = Depends(require_owner), db: Session = Depends(get_db)):
    if member_id == user.id:
        raise HTTPException(status_code=400, detail="You cannot remove yourself")
    member = db.query(User).filter(User.id == member_id, User.org_id == user.org_id).first()
    if not member:
        raise HTTPException(status_code=404, detail="Member not found")
    email = member.email
    # Detach references so history is kept but the user row can go
    db.query(ComplianceItem).filter(ComplianceItem.responsible_user_id == member_id).update({"responsible_user_id": None})
    db.query(Invite).filter(Invite.invited_by_id == member_id).update({"invited_by_id": None})
    db.query(AuditEvent).filter(AuditEvent.user_id == member_id).update({"user_id": None})
    db.delete(member)
    log_event(db, user, "team.member_removed", "user", member_id, {"email": email}, commit=False)
    db.commit()
    return {"ok": True}


# --- Organization ---
@router.get("/organization")
def get_org(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    org = db.query(Organization).filter(Organization.id == user.org_id).first()
    return {"id": org.id, "name": org.name, "sector": org.sector, "country": org.country}


@router.put("/organization")
def update_org(payload: OrgUpdate, user: User = Depends(require_owner), db: Session = Depends(get_db)):
    org = db.query(Organization).filter(Organization.id == user.org_id).first()
    changes = {}
    for field, value in payload.model_dump(exclude_unset=True).items():
        if getattr(org, field) != value:
            changes[field] = {"from": getattr(org, field), "to": value}
            setattr(org, field, value)
    db.commit()
    if changes:
        log_event(db, user, "organization.updated", "organization", org.id, {"changes": changes})
    return {"id": org.id, "name": org.name, "sector": org.sector, "country": org.country}


# --- Public invite acceptance (no auth) ---
def _valid_invite(db: Session, token: str) -> Invite:
    invite = db.query(Invite).filter(Invite.token == token).first()
    if not invite or invite.accepted_at is not None:
        raise HTTPException(status_code=404, detail="Invite not found or already used")
    if invite.expires_at < datetime.utcnow():
        raise HTTPException(status_code=410, detail="This invite has expired")
    return invite


@router.get("/invites/accept/{token}")
def get_invite(token: str, db: Session = Depends(get_db)):
    invite = _valid_invite(db, token)
    org = db.query(Organization).filter(Organization.id == invite.org_id).first()
    return {"email": invite.email, "role": invite.role, "organization": org.name if org else None}


@router.post("/invites/accept/{token}", response_model=TokenResponse)
def accept_invite(token: str, payload: InviteAccept, db: Session = Depends(get_db)):
    invite = _valid_invite(db, token)
    if not payload.name.strip() or len(payload.password) < 8:
        raise HTTPException(status_code=400, detail="Name is required and password must be at least 8 characters")
    if db.query(User).filter(User.email == invite.email).first():
        raise HTTPException(status_code=400, detail="A user with this email already exists")
    user = User(
        org_id=invite.org_id, name=payload.name.strip(), email=invite.email,
        password_hash=hash_password(payload.password), role=invite.role, team_role=invite.team_role,
    )
    db.add(user)
    invite.accepted_at = datetime.utcnow()
    db.commit()
    db.refresh(user)
    log_event(db, user, "team.invite_accepted", "user", user.id, {"email": user.email, "role": user.role})
    return TokenResponse(access_token=create_access_token(user.id, user.org_id), user=UserOut.model_validate(user))
