"""Passwords (argon2id), JWT (HS256), lockout."""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError, InvalidHashError

from ..settings import get_config, secret_key

_ph = PasswordHasher()
LOCKOUT_FAILURES = 5
LOCKOUT_MINUTES = 5
MEDIA_TOKEN_MINUTES = 30


def hash_password(pw: str) -> str:
    return _ph.hash(pw)


def verify_password(pw: str, h: str) -> bool:
    try:
        return _ph.verify(h, pw)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def create_token(user_id: int, role: str, scope: str = "api") -> tuple[str, datetime]:
    hours = float(get_config().app.token_hours)
    exp = datetime.now(timezone.utc) + (timedelta(minutes=MEDIA_TOKEN_MINUTES) if scope == "media" else timedelta(hours=hours))
    payload = {"sub": str(user_id), "role": role, "jti": uuid.uuid4().hex, "exp": exp, "scope": scope,
               "iat": datetime.now(timezone.utc)}
    return jwt.encode(payload, secret_key(), algorithm="HS256"), exp


def decode_token(token: str) -> dict:
    return jwt.decode(token, secret_key(), algorithms=["HS256"], options={"require": ["exp", "sub", "jti"]})
