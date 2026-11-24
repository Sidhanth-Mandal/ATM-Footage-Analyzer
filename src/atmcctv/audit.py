"""Hash-chained, append-only audit log.

payload = canonical_json({ts, user_id, role, action, object_type, object_id, details})
hash    = sha256(prev_hash_hex + payload)

Appends must run inside a write session (BEGIN IMMEDIATE) so two writers can
never read the same prev_hash.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from sqlmodel import Session, select

from .db.models import AuditEvent

GENESIS = "0" * 64


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _payload(ts: str, user_id, role, action, object_type, object_id, details) -> str:
    return canonical_json({"ts": ts, "user_id": user_id, "role": role, "action": action,
                           "object_type": object_type, "object_id": object_id, "details": details})


def compute_hash(prev_hash: str, payload: str) -> str:
    return hashlib.sha256((prev_hash + payload).encode("utf-8")).hexdigest()


def append(session: Session, action: str, *, user_id: int | None = None, role: str | None = None,
           object_type: str | None = None, object_id: Any = None, details: Any = None) -> AuditEvent:
    last = session.exec(select(AuditEvent).order_by(AuditEvent.id.desc()).limit(1)).first()
    prev = last.hash if last else GENESIS
    ts = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    oid = None if object_id is None else str(object_id)
    details = json.loads(canonical_json(details)) if details is not None else None
    h = compute_hash(prev, _payload(ts, user_id, role, action, object_type, oid, details))
    ev = AuditEvent(ts=ts, user_id=user_id, role=role, action=action, object_type=object_type, object_id=oid,
                    details_json=details, prev_hash=prev, hash=h)
    session.add(ev)
    session.flush()
    return ev


def verify(session: Session) -> dict:
    prev = GENESIS
    n = 0
    for ev in session.exec(select(AuditEvent).order_by(AuditEvent.id)):
        n += 1
        if ev.prev_hash != prev:
            return {"ok": False, "checked": n, "first_bad_id": ev.id, "reason": "prev_hash does not match previous row"}
        h = compute_hash(prev, _payload(ev.ts, ev.user_id, ev.role, ev.action, ev.object_type, ev.object_id,
                                        ev.details_json))
        if h != ev.hash:
            return {"ok": False, "checked": n, "first_bad_id": ev.id, "reason": "row content does not match its hash"}
        prev = ev.hash
    return {"ok": True, "checked": n, "first_bad_id": None, "head": prev}
