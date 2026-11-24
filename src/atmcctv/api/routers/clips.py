from __future__ import annotations

import json

import cv2
from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response
from sqlmodel import Session, select

from ... import audit, paths
from ...core import timemap as tmap
from ...core.linking import near
from ...core.persons import load_tracks
from ...core.types import TimeMap
from ...core.video import FrameSource
from ...db.models import Atm, BehaviourEvent, Clip, Track, Transaction, User
from ...db.session import write_session
from ...queue import enqueue
from ...settings import get_config
from .. import schemas as S
from .. import serial
from ..deps import APIError, clip_or_404, get_db, investigator, media_user, viewer

router = APIRouter(tags=["clips"])


@router.get("/clips/{clip_id}", response_model=S.ClipOut)
def get_clip(clip_id: str, u: User = Depends(viewer), s: Session = Depends(get_db)):
    return serial.clip(s, clip_or_404(s, u, clip_id), with_reads=True)


@router.get("/clips/{clip_id}/frame", responses={200: {"content": {"image/jpeg": {}}}})
def get_frame(clip_id: str, pts: float = Query(0.0, ge=0), max_w: int = Query(1280, ge=64, le=4096),
              u: User = Depends(media_user), s: Session = Depends(get_db)):
    c = clip_or_404(s, u, clip_id)
    if not c.analysis_path:
        raise APIError(409, "NOT_INGESTED", "clip is not ingested yet")
    rot = int((c.probe_json or {}).get("analysis_rotation", 0))
    with FrameSource(paths.absp(c.analysis_path), rot) as fs:
        img = fs.get(pts)
    if img is None:
        raise APIError(404, "NO_FRAME", "no frame at that time")
    if img.shape[1] > max_w:
        sc = max_w / img.shape[1]
        img = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return Response(buf.tobytes(), media_type="image/jpeg", headers={"Cache-Control": "private, max-age=300"})


def _refresh(s: Session, c: Clip, u: User) -> list[int]:
    jobs = []
    for t in s.exec(select(Transaction).where(Transaction.clip_id == c.id)).all():
        tm = TimeMap.from_json(c.timemap_json)
        t.t_txn_pts = tmap.txn_to_pts(t.txn_time_journal, c.clock_offset_s, tm) if tm else None
        if c.status in ("tracked", "indexed") and t.status in ("pending", "linked") and tm:
            jobs.append(enqueue(s, "link", {"transaction_id": t.id}, case_id=c.case_id, user_id=u.id).id)
    return jobs


@router.put("/clips/{clip_id}/timemap", response_model=S.ClipOut)
def put_timemap(clip_id: str, body: S.TimemapIn, u: User = Depends(investigator)):
    with write_session() as s:
        c = clip_or_404(s, u, clip_id)
        atm = s.get(Atm, c.atm_id) if c.atm_id else None
        tz = atm.tz if atm else "UTC"
        details: dict = {}
        if body.clock_offset_s is not None:
            details["clock_offset_s"] = [c.clock_offset_s, body.clock_offset_s]
            c.clock_offset_s = body.clock_offset_s
        if body.method == "manual":
            if body.start is None:
                raise APIError(400, "MISSING_START", "manual time mapping needs 'start'")
            st = float((c.probe_json or {}).get("analysis_start_time") or 0.0)
            start = body.start.replace(tzinfo=None) if body.start.tzinfo is None else body.start
            tm = tmap.manual(start, tz, st)
            c.timemap_json = tm.to_json()
            details["timemap"] = {"method": "manual", "start": body.start.isoformat(), "tz": tz}
        elif body.method in ("ocr", "metadata"):
            if not c.analysis_path:
                raise APIError(409, "NOT_INGESTED", "clip is not ingested yet")
            if body.method == "ocr":
                if not body.roi or len(body.roi) != 4:
                    raise APIError(400, "MISSING_ROI", "OCR needs roi [x0,y0,x1,y1] (normalised)")
                if atm:
                    atm.ocr_roi_json = body.roi
            j = enqueue(s, "timemap", {"clip_id": c.id, "roi": body.roi if body.method == "ocr" else None,
                                       "only": True, "force": True}, case_id=c.case_id, user_id=u.id)
            details["timemap_job"] = j.id
            details["method"] = body.method
        details["relink_jobs"] = _refresh(s, c, u)
        audit.append(s, "timemap_updated", user_id=u.id, role=u.role, object_type="clip", object_id=c.id, details=details)
        return serial.clip(s, c, with_reads=True)


@router.post("/clips/{clip_id}/transactions", response_model=list[S.TxnOut])
def add_transactions(clip_id: str, body: list[S.TxnIn], u: User = Depends(investigator)):
    if not body:
        raise APIError(400, "EMPTY", "no transactions")
    with write_session() as s:
        c = clip_or_404(s, u, clip_id)
        if c.kind != "transaction":
            raise APIError(400, "GALLERY_CLIP", "transactions can only be added to transaction clips")
        tm = TimeMap.from_json(c.timemap_json)
        out = []
        for b in body:
            journal = b.txn_time.replace(tzinfo=None)
            t = Transaction(case_id=c.case_id, clip_id=c.id, txn_ref=b.txn_ref, txn_time_journal=journal,
                            amount=b.amount, window_before_s=b.window_before_s, window_after_s=b.window_after_s)
            t.t_txn_pts = tmap.txn_to_pts(journal, c.clock_offset_s, tm) if tm else None
            s.add(t)
            s.flush()
            jid = None
            if c.status in ("tracked", "indexed") and tm:
                jid = enqueue(s, "link", {"transaction_id": t.id}, case_id=c.case_id, user_id=u.id).id
            audit.append(s, "transaction_added", user_id=u.id, role=u.role, object_type="transaction", object_id=t.id,
                         details={"clip_id": c.id, "txn_ref": b.txn_ref, "txn_time": journal.isoformat(),
                                  "amount": b.amount, "window": [b.window_before_s, b.window_after_s], "link_job": jid})
            out.append(serial.txn(t, c.orig_name))
        return out


@router.get("/clips/{clip_id}/transactions", response_model=list[S.TxnOut])
def clip_transactions(clip_id: str, u: User = Depends(viewer), s: Session = Depends(get_db)):
    c = clip_or_404(s, u, clip_id)
    return [serial.txn(t, c.orig_name) for t in s.exec(select(Transaction).where(Transaction.clip_id == c.id)).all()]


@router.get("/clips/{clip_id}/tracks", response_model=list[S.TrackOut])
def clip_tracks(clip_id: str, u: User = Depends(viewer), s: Session = Depends(get_db)):
    c = clip_or_404(s, u, clip_id)
    return [serial.track(t) for t in s.exec(select(Track).where(Track.clip_id == c.id).order_by(Track.t_start)).all()]


@router.get("/clips/{clip_id}/timeline")
def clip_timeline(clip_id: str, u: User = Depends(viewer), s: Session = Depends(get_db)) -> dict:
    """Per-track spans with near-field segments for the transaction timeline."""
    c = clip_or_404(s, u, clip_id)
    rows = {t.track_no: t for t in s.exec(select(Track).where(Track.clip_id == c.id)).all()}
    if not rows:
        return {"duration": (c.probe_json or {}).get("analysis_duration"), "tracks": []}
    atm = s.get(Atm, c.atm_id) if c.atm_id else None
    cam = atm.camera_type if atm else "fascia"
    rois = (atm.rois_json if atm else {}) or {}
    tracks, meta = load_tracks(paths.clip_dir(c.case_id, c.id))
    cfg = get_config()
    gap = 2.5 / float(meta.get("sample_fps", 5))
    out = []
    for t in tracks:
        if t.track_id not in rows:
            continue
        segs: list[list[float]] = []
        for smp in t.samples:
            if near(smp, cam, int(meta["width"]), int(meta["height"]), rois, cfg):
                if segs and smp.pts - segs[-1][1] <= gap:
                    segs[-1][1] = smp.pts
                else:
                    segs.append([smp.pts, smp.pts])
        r = rows[t.track_id]
        out.append({"track_id": r.id, "track_no": r.track_no, "source": r.source, "t_start": t.t_start,
                    "t_end": t.t_end, "near": segs})
    return {"duration": (c.probe_json or {}).get("analysis_duration"), "width": meta["width"], "height": meta["height"],
            "tracks": out}


@router.get("/clips/{clip_id}/behaviour", response_model=list[S.BehaviourOut])
def clip_behaviour(clip_id: str, u: User = Depends(viewer), s: Session = Depends(get_db)):
    c = clip_or_404(s, u, clip_id)
    return [S.BehaviourOut(id=e.id, clip_id=e.clip_id, type=e.type, t_start=e.t_start, t_end=e.t_end,
                           track_ids=e.track_ids_json, details=e.details_json)
            for e in s.exec(select(BehaviourEvent).where(BehaviourEvent.clip_id == c.id).order_by(BehaviourEvent.t_start)).all()]


@router.post("/clips/{clip_id}/behaviour", response_model=S.JobOut)
def run_behaviour(clip_id: str, u: User = Depends(investigator)):
    with write_session() as s:
        c = clip_or_404(s, u, clip_id)
        if c.status not in ("tracked", "indexed"):
            raise APIError(409, "NOT_TRACKED", "clip has no tracks yet")
        j = enqueue(s, "behaviour", {"clip_id": c.id}, case_id=c.case_id, user_id=u.id)
        audit.append(s, "behaviour_requested", user_id=u.id, role=u.role, object_type="clip", object_id=c.id,
                     details={"job_id": j.id})
        return serial.job(j)


@router.get("/clips/{clip_id}/report")
def clip_report(clip_id: str, u: User = Depends(viewer), s: Session = Depends(get_db)) -> dict:
    c = clip_or_404(s, u, clip_id)
    p = paths.clip_dir(c.case_id, c.id) / "report.json"
    if not p.exists():
        raise APIError(404, "NO_REPORT", "no report yet")
    return json.loads(p.read_text(encoding="utf-8"))
