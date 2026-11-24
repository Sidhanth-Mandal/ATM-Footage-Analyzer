from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, Depends, Request
from sqlmodel import Session, select

from ... import audit
from ...db.models import User, now
from ...db.session import write_session
from .. import schemas as S
from .. import serial
from ..auth import LOCKOUT_FAILURES, LOCKOUT_MINUTES, create_token, hash_password, verify_password
from ..deps import APIError, admin, current_user, get_db, not_found

router = APIRouter()


@router.post("/auth/login", response_model=S.TokenOut, tags=["auth"])
def login(body: S.LoginIn, request: Request):
    ip = request.client.host if request.client else None
    with write_session() as s:
        u = s.exec(select(User).where(User.username == body.username)).first()
        if u and u.locked_until and u.locked_until > now():
            audit.append(s, "login_locked", user_id=u.id, role=u.role, object_type="user", object_id=u.id,
                         details={"ip": ip})
            raise_after = APIError(423, "LOCKED", f"too many failed attempts; try again after {LOCKOUT_MINUTES} minutes")
        elif not u or not u.is_active or not verify_password(body.password, u.password_hash):
            if u:
                u.failed_logins += 1
                if u.failed_logins >= LOCKOUT_FAILURES:
                    u.locked_until = now() + timedelta(minutes=LOCKOUT_MINUTES)
                    u.failed_logins = 0
            audit.append(s, "login_failed", user_id=u.id if u else None, role=u.role if u else None,
                         object_type="user", object_id=u.id if u else None, details={"username": body.username, "ip": ip})
            raise_after = APIError(401, "BAD_CREDENTIALS", "invalid username or password")
        else:
            u.failed_logins, u.locked_until = 0, None
            token, exp = create_token(u.id, u.role)
            audit.append(s, "login", user_id=u.id, role=u.role, object_type="user", object_id=u.id, details={"ip": ip})
            return S.TokenOut(access_token=token, role=u.role, expires_at=exp, username=u.username, user_id=u.id)
    raise raise_after


@router.get("/auth/me", response_model=S.UserOut, tags=["auth"])
def me(u: User = Depends(current_user)):
    return serial.user(u)


@router.post("/auth/media-token", response_model=S.MediaTokenOut, tags=["auth"])
def media_token(u: User = Depends(current_user)):
    """Short-lived token for <img>/<video> URLs (?t=...), which cannot carry an Authorization header."""
    t, exp = create_token(u.id, u.role, scope="media")
    return S.MediaTokenOut(token=t, expires_at=exp)


@router.get("/users", response_model=list[S.UserOut], tags=["users"])
def list_users(u: User = Depends(admin), s: Session = Depends(get_db)):
    return [serial.user(x) for x in s.exec(select(User).order_by(User.id)).all()]


@router.get("/users/directory", response_model=list[S.UserOut], tags=["users"])
def directory(u: User = Depends(current_user), s: Session = Depends(get_db)):
    """Names and roles only, so case pages can show who did what."""
    return [serial.user(x) for x in s.exec(select(User).order_by(User.id)).all()]


@router.post("/users", response_model=S.UserOut, tags=["users"])
def create_user(body: S.UserIn, u: User = Depends(admin)):
    with write_session() as s:
        if s.exec(select(User).where(User.username == body.username)).first():
            raise APIError(409, "EXISTS", "username already exists")
        nu = User(username=body.username, password_hash=hash_password(body.password), role=body.role)
        s.add(nu)
        s.flush()
        audit.append(s, "user_created", user_id=u.id, role=u.role, object_type="user", object_id=nu.id,
                     details={"username": nu.username, "role": nu.role})
        return serial.user(nu)


@router.patch("/users/{user_id}", response_model=S.UserOut, tags=["users"])
def patch_user(user_id: int, body: S.UserPatch, u: User = Depends(admin)):
    with write_session() as s:
        t = s.get(User, user_id)
        if not t:
            raise not_found("user")
        changes = {}
        if body.role is not None and body.role != t.role:
            if t.id == u.id and body.role != "admin":
                raise APIError(400, "SELF_DEMOTE", "admins cannot remove their own admin role")
            changes["role"] = [t.role, body.role]
            t.role = body.role
        if body.is_active is not None and body.is_active != t.is_active:
            if t.id == u.id and not body.is_active:
                raise APIError(400, "SELF_DISABLE", "admins cannot deactivate themselves")
            changes["is_active"] = [t.is_active, body.is_active]
            t.is_active = body.is_active
        if body.password:
            t.password_hash = hash_password(body.password)
            t.failed_logins, t.locked_until = 0, None
            changes["password"] = "reset"
        audit.append(s, "user_updated", user_id=u.id, role=u.role, object_type="user", object_id=t.id, details=changes)
        return serial.user(t)
