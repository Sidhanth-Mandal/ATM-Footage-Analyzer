"""Media (RBAC-checked, never a static mount), exports, jobs, audit."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from sqlmodel import Session, select

from ... import audit, paths
from ...db.models import AuditEvent, CandidateRow, Clip, Enhancement, Export, FaceRow, Job, Transaction, User
from ...db.session import write_session
from .. import schemas as S
from .. import serial
from ..deps import (
    APIError, accessible_case_ids, admin, can_access_case, case_or_404, clip_or_404, face_or_404, get_db,
    investigator, media_user, not_found, viewer,
)

router = APIRouter()
NOCACHE = {"Cache-Control": "private, no-store"}


def _file(p, media_type: str) -> FileResponse:
    if p is None or not p.exists():
        raise not_found("file")
    return FileResponse(p, media_type=media_type, headers=NOCACHE)


@router.get("/media/proxy/{clip_id}", tags=["media"])
def media_proxy(clip_id: str, u: User = Depends(media_user), s: Session = Depends(get_db)):
    c = clip_or_404(s, u, clip_id)
    return _file(paths.absp(c.proxy_path), "video/mp4")


@router.get("/media/face/{face_id}/{kind}", tags=["media"])
def media_face(face_id: int, kind: Literal["frame", "crop", "aligned"], u: User = Depends(media_user),
               s: Session = Depends(get_db)):
    f, _ = face_or_404(s, u, face_id)
    p = {"frame": f.frame_path, "crop": f.crop_path, "aligned": f.aligned_path}[kind]
    return _file(paths.absp(p), "image/jpeg" if kind == "frame" else "image/png")


@router.get("/media/enhanced/{enh_id}", tags=["media"])
def media_enhanced(enh_id: int, u: User = Depends(media_user), s: Session = Depends(get_db)):
    e = s.get(Enhancement, enh_id)
    if not e:
        raise not_found("enhancement")
    face_or_404(s, u, e.face_id)
    return _file(paths.absp(e.path), "image/png")


@router.get("/media/candidate/{cand_id}", tags=["media"])
def media_candidate(cand_id: int, u: User = Depends(media_user), s: Session = Depends(get_db)):
    c = s.get(CandidateRow, cand_id)
    if not c:
        raise not_found("candidate")
    t = s.get(Transaction, c.transaction_id)
    clip_or_404(s, u, t.clip_id)
    return _file(paths.absp(c.thumb_path), "image/jpeg")


@router.get("/exports/{export_id}", response_model=S.ExportOut, tags=["exports"])
def get_export(export_id: int, u: User = Depends(viewer), s: Session = Depends(get_db)):
    e = s.get(Export, export_id)
    if not e:
        raise not_found("export")
    case_or_404(s, u, e.case_id)
    return serial.export(e)


@router.get("/exports/{export_id}/download", tags=["exports"])
def download_export(export_id: int, u: User = Depends(media_user)):
    """PDF download (audited). Viewers may read PDFs of cases they belong to (role matrix)."""
    with write_session() as s:
        e = s.get(Export, export_id)
        if not e or not can_access_case(s, u, e.case_id):
            raise not_found("export")
        if e.status != "done" or not e.path:
            raise APIError(409, "NOT_READY", f"export is {e.status}")
        audit.append(s, "export_downloaded", user_id=u.id, role=u.role, object_type="export", object_id=e.id,
                     details={"sha256": e.sha256})
        p = paths.absp(e.path)
        sha = e.sha256
    if not p.exists():
        raise not_found("file")
    return FileResponse(p, media_type="application/pdf", filename=f"evidence_{export_id}.pdf",
                        headers={**NOCACHE, "X-Content-SHA256": sha or ""})


@router.get("/jobs/{job_id}", response_model=S.JobOut, tags=["jobs"])
def get_job(job_id: int, u: User = Depends(viewer), s: Session = Depends(get_db)):
    j = s.get(Job, job_id)
    if not j or (j.case_id and not can_access_case(s, u, j.case_id)) or (not j.case_id and u.role != "admin"):
        raise not_found("job")
    return serial.job(j)


@router.get("/jobs", response_model=list[S.JobOut], tags=["jobs"])
def list_jobs(case_id: str | None = None, status: str | None = None, u: User = Depends(viewer),
              s: Session = Depends(get_db)):
    q = select(Job).order_by(Job.id.desc()).limit(300)
    if case_id:
        case_or_404(s, u, case_id)
        q = q.where(Job.case_id == case_id)
    if status:
        q = q.where(Job.status == status)
    ids = accessible_case_ids(s, u)
    return [serial.job(j) for j in s.exec(q).all() if ids is None or j.case_id in ids]


@router.post("/jobs/{job_id}/cancel", response_model=S.JobOut, tags=["jobs"])
def cancel_job(job_id: int, u: User = Depends(investigator)):
    with write_session() as s:
        j = s.get(Job, job_id)
        if not j or (j.case_id and not can_access_case(s, u, j.case_id)):
            raise not_found("job")
        if j.status == "queued":
            j.status, j.message = "cancelled", "cancelled before start"
        elif j.status == "running":
            j.status = "cancel_requested"
        else:
            raise APIError(409, "NOT_CANCELLABLE", f"job is {j.status}")
        audit.append(s, "job_cancel_requested", user_id=u.id, role=u.role, object_type="job", object_id=j.id)
        return serial.job(j)


@router.get("/audit", response_model=list[S.AuditOut], tags=["audit"])
def list_audit(from_: datetime | None = Query(default=None, alias="from"), to: datetime | None = None,
               user: int | None = None, action: str | None = None, limit: int = Query(500, le=5000),
               u: User = Depends(admin), s: Session = Depends(get_db)):
    q = select(AuditEvent).order_by(AuditEvent.id.desc())
    if from_:
        q = q.where(AuditEvent.ts >= from_.isoformat())
    if to:
        q = q.where(AuditEvent.ts <= to.isoformat())
    if user is not None:
        q = q.where(AuditEvent.user_id == user)
    if action:
        q = q.where(AuditEvent.action == action)
    return [S.AuditOut(id=e.id, ts=e.ts, user_id=e.user_id, role=e.role, action=e.action, object_type=e.object_type,
                       object_id=e.object_id, details=e.details_json, prev_hash=e.prev_hash, hash=e.hash)
            for e in s.exec(q.limit(limit)).all()]


@router.get("/audit/verify", response_model=S.VerifyOut, tags=["audit"])
def verify_audit(u: User = Depends(admin)):
    with write_session() as s:
        r = audit.verify(s)
        audit.append(s, "audit_verified", user_id=u.id, role=u.role, details={"ok": r["ok"], "checked": r["checked"],
                                                                              "first_bad_id": r["first_bad_id"]})
    return S.VerifyOut(**r)


__all__ = ["router", "Clip", "FaceRow"]
