"""Transactions, candidates, confirmation, tracks, faces, enhancement."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlmodel import Session, select

from ... import audit, paths
from ...core.linking import near
from ...core.persons import load_tracks
from ...db.models import Atm, CandidateRow, Clip, Enhancement, FaceRow, Track, Transaction, User, now
from ...db.session import write_session
from ...queue import enqueue
from ...settings import get_config
from .. import schemas as S
from .. import serial
from ..deps import APIError, face_or_404, get_db, investigator, track_or_404, txn_or_404, viewer

router = APIRouter(tags=["analysis"])


@router.get("/transactions/{txn_id}", response_model=S.TxnOut)
def get_txn(txn_id: int, u: User = Depends(viewer), s: Session = Depends(get_db)):
    t, c = txn_or_404(s, u, txn_id)
    return serial.txn(t, c.orig_name)


@router.get("/transactions/{txn_id}/candidates", response_model=list[S.CandidateOut])
def candidates(txn_id: int, u: User = Depends(viewer), s: Session = Depends(get_db)):
    t, _ = txn_or_404(s, u, txn_id)
    rows = s.exec(select(CandidateRow).where(CandidateRow.transaction_id == t.id).order_by(CandidateRow.rank)).all()
    return [serial.candidate(s, c) for c in rows]


class ConfirmIn(BaseModel):
    track_id: int
    note: str = ""


@router.post("/transactions/{txn_id}/confirm", response_model=S.JobOut)
def confirm(txn_id: int, body: ConfirmIn, u: User = Depends(investigator)):
    """Human-in-the-loop: the investigator confirms which track performed the transaction."""
    with write_session() as s:
        t, c = txn_or_404(s, u, txn_id)
        tr = s.get(Track, body.track_id)
        if not tr or tr.clip_id != c.id:
            raise APIError(400, "BAD_TRACK", "track is not in this clip")
        cand = s.exec(select(CandidateRow).where(CandidateRow.transaction_id == t.id,
                                                 CandidateRow.track_id == tr.id)).first()
        prev = t.chosen_track_id
        t.chosen_track_id, t.confirmed_by, t.confirmed_at, t.status = tr.id, u.id, now(), "confirmed"
        j = enqueue(s, "dense", {"transaction_id": t.id, "track_id": tr.id}, case_id=c.case_id, user_id=u.id)
        audit.append(s, "candidate_confirmed", user_id=u.id, role=u.role, object_type="transaction", object_id=t.id,
                     details={"track_id": tr.id, "track_no": tr.track_no, "previous_track_id": prev,
                              "candidate_rank": cand.rank if cand else None,
                              "candidate_score": cand.score if cand else None, "note": body.note, "job_id": j.id})
        return serial.job(j)


@router.get("/tracks/{track_id}", response_model=S.TrackOut)
def get_track(track_id: int, u: User = Depends(viewer), s: Session = Depends(get_db)):
    t, _ = track_or_404(s, u, track_id)
    return serial.track(t)


@router.get("/tracks/{track_id}/samples", response_model=S.TrackSamplesOut)
def track_samples(track_id: int, u: User = Depends(viewer), s: Session = Depends(get_db)):
    t, c = track_or_404(s, u, track_id)
    tracks, meta = load_tracks(paths.clip_dir(c.case_id, c.id))
    pt = next((x for x in tracks if x.track_id == t.track_no), None)
    if pt is None:
        raise APIError(404, "NO_SAMPLES", "track samples not found")
    atm = s.get(Atm, c.atm_id) if c.atm_id else None
    cfg = get_config()
    cam = atm.camera_type if atm else "fascia"
    rois = (atm.rois_json if atm else {}) or {}
    return S.TrackSamplesOut(track_id=t.id, track_no=t.track_no, pts=[x.pts for x in pt.samples],
                             bbox=[[float(v) for v in x.bbox] for x in pt.samples],
                             near=[near(x, cam, int(meta["width"]), int(meta["height"]), rois, cfg) for x in pt.samples])


@router.get("/tracks/{track_id}/faces", response_model=list[S.FaceOut])
def track_faces(track_id: int, selected: bool | None = Query(default=None), purpose: str | None = None,
                u: User = Depends(viewer), s: Session = Depends(get_db)):
    t, _ = track_or_404(s, u, track_id)
    q = select(FaceRow).where(FaceRow.track_id == t.id)
    if selected is not None:
        q = q.where(FaceRow.selected == selected)  # noqa: E712
    if purpose:
        q = q.where(FaceRow.purpose == purpose)
    return [serial.face(s, f, with_enh=True) for f in s.exec(q.order_by(FaceRow.total.desc())).all()]


@router.get("/faces/{face_id}", response_model=S.FaceOut)
def get_face(face_id: int, u: User = Depends(viewer), s: Session = Depends(get_db)):
    f, _ = face_or_404(s, u, face_id)
    return serial.face(s, f, with_enh=True)


@router.post("/faces/{face_id}/enhance", response_model=S.EnhancementOut)
def enhance(face_id: int, body: S.EnhanceIn, u: User = Depends(investigator)):
    with write_session() as s:
        f, c = face_or_404(s, u, face_id)
        if not f.crop_path:
            raise APIError(409, "NO_CROP", "face has no stored crop")
        params = {k: float(v) for k, v in body.params.items() if k in ("gamma", "clip", "blend")}
        e = Enhancement(face_id=f.id, method=body.method, params_json=params, created_by=u.id)
        s.add(e)
        s.flush()
        j = enqueue(s, "enhance", {"enhancement_id": e.id}, case_id=c.case_id, user_id=u.id)
        audit.append(s, "enhancement_requested", user_id=u.id, role=u.role, object_type="face", object_id=f.id,
                     details={"enhancement_id": e.id, "method": body.method, "params": params, "job_id": j.id})
        return serial.enhancement(e, j.id)


@router.get("/faces/{face_id}/enhancements", response_model=list[S.EnhancementOut])
def list_enhancements(face_id: int, u: User = Depends(viewer), s: Session = Depends(get_db)):
    f, _ = face_or_404(s, u, face_id)
    return [serial.enhancement(e) for e in s.exec(select(Enhancement).where(Enhancement.face_id == f.id)).all()]


@router.get("/clips/{clip_id}/faces", response_model=list[S.FaceOut])
def clip_faces(clip_id: str, u: User = Depends(viewer), s: Session = Depends(get_db)):
    from ..deps import clip_or_404
    c = clip_or_404(s, u, clip_id)
    return [serial.face(s, f) for f in s.exec(select(FaceRow).where(FaceRow.clip_id == c.id, FaceRow.selected == True)  # noqa: E712
                                               .order_by(FaceRow.total.desc())).all()]


__all__ = ["router", "Clip"]
