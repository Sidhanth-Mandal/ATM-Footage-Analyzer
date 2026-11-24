from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from ... import audit
from ...db.models import Atm, User
from ...db.session import write_session
from .. import schemas as S
from .. import serial
from ..deps import APIError, get_db, investigator, not_found, viewer

router = APIRouter(tags=["atms"])


@router.get("/atms", response_model=list[S.AtmOut])
def list_atms(u: User = Depends(viewer), s: Session = Depends(get_db)):
    return [serial.atm(a) for a in s.exec(select(Atm).order_by(Atm.atm_code)).all()]


@router.get("/atms/{atm_id}", response_model=S.AtmOut)
def get_atm(atm_id: int, u: User = Depends(viewer), s: Session = Depends(get_db)):
    a = s.get(Atm, atm_id)
    if not a:
        raise not_found("atm")
    return serial.atm(a)


def _apply(a: Atm, b: S.AtmIn) -> None:
    a.atm_code, a.location, a.camera_type = b.atm_code, b.location, b.camera_type
    a.rois_json = {k: v for k, v in b.rois.items() if v}
    a.ocr_roi_json, a.date_order, a.tz, a.metadata_is_end = b.ocr_roi, b.date_order, b.tz, b.metadata_is_end


@router.post("/atms", response_model=S.AtmOut)
def create_atm(body: S.AtmIn, u: User = Depends(investigator)):
    with write_session() as s:
        if s.exec(select(Atm).where(Atm.atm_code == body.atm_code)).first():
            raise APIError(409, "EXISTS", "ATM code already exists")
        a = Atm(atm_code=body.atm_code)
        _apply(a, body)
        s.add(a)
        s.flush()
        audit.append(s, "atm_created", user_id=u.id, role=u.role, object_type="atm", object_id=a.id,
                     details=body.model_dump())
        return serial.atm(a)


@router.put("/atms/{atm_id}", response_model=S.AtmOut)
def update_atm(atm_id: int, body: S.AtmIn, u: User = Depends(investigator)):
    with write_session() as s:
        a = s.get(Atm, atm_id)
        if not a:
            raise not_found("atm")
        other = s.exec(select(Atm).where(Atm.atm_code == body.atm_code, Atm.id != atm_id)).first()
        if other:
            raise APIError(409, "EXISTS", "ATM code already exists")
        before = serial.atm(a).model_dump()
        _apply(a, body)
        audit.append(s, "atm_updated", user_id=u.id, role=u.role, object_type="atm", object_id=a.id,
                     details={"before": before, "after": body.model_dump()})
        return serial.atm(a)
