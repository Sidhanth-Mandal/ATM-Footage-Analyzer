"""Job queue helpers shared by the API and the worker (no ML imports)."""
from __future__ import annotations

from sqlmodel import Session

from .db.models import Job


def enqueue(s: Session, type_: str, payload: dict, *, case_id: str | None = None, user_id: int | None = None,
            depends_on: int | None = None) -> Job:
    j = Job(type=type_, payload_json=payload, case_id=case_id, created_by=user_id, depends_on=depends_on)
    s.add(j)
    s.flush()
    return j
