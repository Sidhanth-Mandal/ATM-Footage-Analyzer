"""Request dependencies: DB sessions, current user, RBAC, case membership, errors."""
from __future__ import annotations

from typing import Iterator

import jwt
from fastapi import Depends, Header, Query, Request
from sqlmodel import Session, select

from ..db.models import ROLES, Case, CaseMember, Clip, FaceRow, Track, Transaction, User
from ..db.session import read_session
from .auth import decode_token

RANK = {r: i for i, r in enumerate(ROLES)}


class APIError(Exception):
    def __init__(self, status: int, code: str, message: str, details: dict | None = None):
        self.status, self.code, self.message, self.details = status, code, message, details or {}


def not_found(what: str = "object") -> APIError:
    return APIError(404, "NOT_FOUND", f"{what} not found")


def get_db() -> Iterator[Session]:
    s = read_session()
    try:
        yield s
    finally:
        s.close()


def _user_from_token(token: str, s: Session, allow_media: bool) -> User:
    try:
        claims = decode_token(token)
    except jwt.ExpiredSignatureError:
        raise APIError(401, "TOKEN_EXPIRED", "session expired, log in again")
    except jwt.PyJWTError:
        raise APIError(401, "INVALID_TOKEN", "invalid token")
    if claims.get("scope", "api") == "media" and not allow_media:
        raise APIError(401, "INVALID_TOKEN", "media token cannot be used for the API")
    u = s.get(User, int(claims["sub"]))
    if not u or not u.is_active:
        raise APIError(401, "INVALID_TOKEN", "user inactive")
    return u


def current_user(authorization: str | None = Header(default=None), s: Session = Depends(get_db)) -> User:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise APIError(401, "NOT_AUTHENTICATED", "missing bearer token")
    return _user_from_token(authorization.split(" ", 1)[1], s, allow_media=False)


def media_user(request: Request, t: str | None = Query(default=None), s: Session = Depends(get_db)) -> User:
    """Media URLs are used in <img>/<video>, which cannot send headers: accept a short-lived media token."""
    auth = request.headers.get("authorization")
    if auth and auth.lower().startswith("bearer "):
        return _user_from_token(auth.split(" ", 1)[1], s, allow_media=True)
    if t:
        return _user_from_token(t, s, allow_media=True)
    raise APIError(401, "NOT_AUTHENTICATED", "missing token")


def require_role(min_role: str):
    def dep(u: User = Depends(current_user)) -> User:
        if RANK.get(u.role, -1) < RANK[min_role]:
            raise APIError(403, "FORBIDDEN", f"requires role {min_role}")
        return u
    return dep


viewer = require_role("viewer")
investigator = require_role("investigator")
admin = require_role("admin")


def can_access_case(s: Session, u: User, case_id: str) -> bool:
    if u.role == "admin":
        return True
    return s.get(CaseMember, (case_id, u.id)) is not None


def accessible_case_ids(s: Session, u: User) -> set[str] | None:
    """None means all (admin)."""
    if u.role == "admin":
        return None
    return {m.case_id for m in s.exec(select(CaseMember).where(CaseMember.user_id == u.id)).all()}


def case_or_404(s: Session, u: User, case_id: str) -> Case:
    c = s.get(Case, case_id)
    if not c or not can_access_case(s, u, case_id):
        raise not_found("case")
    return c


def clip_or_404(s: Session, u: User, clip_id: str) -> Clip:
    c = s.get(Clip, clip_id)
    if not c or not can_access_case(s, u, c.case_id):
        raise not_found("clip")
    return c


def track_or_404(s: Session, u: User, track_id: int) -> tuple[Track, Clip]:
    t = s.get(Track, track_id)
    if not t:
        raise not_found("track")
    return t, clip_or_404(s, u, t.clip_id)


def face_or_404(s: Session, u: User, face_id: int) -> tuple[FaceRow, Clip]:
    f = s.get(FaceRow, face_id)
    if not f:
        raise not_found("face")
    return f, clip_or_404(s, u, f.clip_id)


def txn_or_404(s: Session, u: User, txn_id: int) -> tuple[Transaction, Clip]:
    t = s.get(Transaction, txn_id)
    if not t:
        raise not_found("transaction")
    return t, clip_or_404(s, u, t.clip_id)
