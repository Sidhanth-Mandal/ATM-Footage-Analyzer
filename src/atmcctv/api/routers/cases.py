from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import uuid

from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlmodel import Session, select

from ... import audit, paths
from ...db.models import Atm, Case, CaseMember, Clip, Export, Job, Track, Transaction, User
from ...db.session import write_session
from ...queue import enqueue
from ...settings import get_config
from .. import schemas as S
from .. import serial
from ..deps import APIError, accessible_case_ids, admin, case_or_404, get_db, investigator, viewer

router = APIRouter(tags=["cases"])
EXT_RE = re.compile(r"^\.[A-Za-z0-9]{1,8}$")


@router.get("/cases", response_model=list[S.CaseOut])
def list_cases(u: User = Depends(viewer), s: Session = Depends(get_db)):
    ids = accessible_case_ids(s, u)
    q = select(Case).order_by(Case.created_at.desc())
    rows = s.exec(q).all()
    return [serial.case(s, c) for c in rows if ids is None or c.id in ids]


@router.post("/cases", response_model=S.CaseOut)
def create_case(body: S.CaseIn, u: User = Depends(investigator)):
    with write_session() as s:
        c = Case(ref_no=body.ref_no, title=body.title, created_by=u.id)
        s.add(c)
        s.flush()
        s.add(CaseMember(case_id=c.id, user_id=u.id))
        audit.append(s, "case_created", user_id=u.id, role=u.role, object_type="case", object_id=c.id,
                     details={"ref_no": c.ref_no, "title": c.title})
        s.flush()
        return serial.case(s, c)


@router.get("/cases/{case_id}", response_model=S.CaseOut)
def get_case(case_id: str, u: User = Depends(viewer), s: Session = Depends(get_db)):
    return serial.case(s, case_or_404(s, u, case_id))


@router.put("/cases/{case_id}/members", response_model=S.CaseOut)
def set_members(case_id: str, body: S.MembersIn, u: User = Depends(admin)):
    with write_session() as s:
        c = case_or_404(s, u, case_id)
        for uid in body.user_ids:
            if not s.get(User, uid):
                raise APIError(400, "BAD_USER", f"user {uid} does not exist")
        old = {m.user_id for m in s.exec(select(CaseMember).where(CaseMember.case_id == case_id)).all()}
        new = set(body.user_ids)
        for m in s.exec(select(CaseMember).where(CaseMember.case_id == case_id)).all():
            if m.user_id not in new:
                s.delete(m)
        for uid in new - old:
            s.add(CaseMember(case_id=case_id, user_id=uid))
        audit.append(s, "case_members_set", user_id=u.id, role=u.role, object_type="case", object_id=case_id,
                     details={"added": sorted(new - old), "removed": sorted(old - new)})
        s.flush()
        return serial.case(s, c)


@router.get("/cases/{case_id}/clips", response_model=list[S.ClipOut])
def case_clips(case_id: str, u: User = Depends(viewer), s: Session = Depends(get_db)):
    case_or_404(s, u, case_id)
    return [serial.clip(s, c) for c in s.exec(select(Clip).where(Clip.case_id == case_id).order_by(Clip.created_at)).all()]


@router.get("/cases/{case_id}/transactions", response_model=list[S.TxnOut])
def case_txns(case_id: str, u: User = Depends(viewer), s: Session = Depends(get_db)):
    case_or_404(s, u, case_id)
    out = []
    for t in s.exec(select(Transaction).where(Transaction.case_id == case_id).order_by(Transaction.id)).all():
        out.append(serial.txn(t, s.get(Clip, t.clip_id).orig_name))
    return out


@router.post("/cases/{case_id}/clips", response_model=S.ClipOut)
async def upload_clip(case_id: str, file: UploadFile = File(...), atm_id: int | None = Form(default=None),
                      kind: str = Form(default="transaction"), u: User = Depends(investigator),
                      s: Session = Depends(get_db)):
    case_or_404(s, u, case_id)
    if kind not in ("transaction", "gallery"):
        raise APIError(400, "BAD_KIND", "kind must be transaction or gallery")
    if atm_id is not None and not s.get(Atm, atm_id):
        raise APIError(400, "BAD_ATM", "unknown atm_id")
    s.close()
    limit = int(float(get_config().ingest.max_upload_gb) * (1 << 30))
    orig_name = os.path.basename(file.filename or "upload.bin")[:255]
    ext = os.path.splitext(orig_name)[1].lower()
    ext = ext if EXT_RE.match(ext) else ".bin"
    tmp = paths.tmp_dir() / f"{uuid.uuid4().hex}.part"
    h = hashlib.sha256()
    size = 0
    try:
        with open(tmp, "wb") as out:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > limit:
                    raise APIError(413, "TOO_LARGE", f"file exceeds {get_config().ingest.max_upload_gb} GB")
                h.update(chunk)
                out.write(chunk)
        if size == 0:
            raise APIError(400, "EMPTY", "empty file")
        clip_id = str(uuid.uuid4())
        cdir = paths.clip_dir(case_id, clip_id)
        cdir.mkdir(parents=True, exist_ok=True)
        dest = cdir / f"original{ext}"
        shutil.move(str(tmp), dest)
        os.chmod(dest, stat.S_IREAD)   # originals are immutable after hashing
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    sha = h.hexdigest()
    with write_session() as w:
        c = Clip(id=clip_id, case_id=case_id, atm_id=atm_id, kind=kind, orig_name=orig_name, orig_path=paths.rel(dest),
                 orig_sha256=sha, size=size, created_by=u.id)
        w.add(c)
        w.flush()
        j = enqueue(w, "ingest", {"clip_id": c.id}, case_id=case_id, user_id=u.id)
        audit.append(w, "clip_uploaded", user_id=u.id, role=u.role, object_type="clip", object_id=c.id,
                     details={"orig_name": orig_name, "sha256": sha, "size": size, "kind": kind, "atm_id": atm_id,
                              "job_id": j.id})
        return serial.clip(w, c)


@router.post("/cases/{case_id}/gallery/index", response_model=list[S.JobOut])
def gallery_index(case_id: str, body: S.GalleryIn, u: User = Depends(investigator)):
    with write_session() as s:
        case_or_404(s, u, case_id)
        jobs = []
        for cid in body.clip_ids:
            c = s.get(Clip, cid)
            if not c or c.case_id != case_id:
                raise APIError(400, "BAD_CLIP", f"clip {cid} is not in this case")
            if c.status not in ("tracked", "indexed", "indexing"):
                raise APIError(409, "NOT_TRACKED", f"clip {c.orig_name} has no tracks yet (status {c.status})")
            jobs.append(enqueue(s, "index", {"clip_id": cid}, case_id=case_id, user_id=u.id))
        audit.append(s, "gallery_index_requested", user_id=u.id, role=u.role, object_type="case", object_id=case_id,
                     details={"clip_ids": body.clip_ids, "job_ids": [j.id for j in jobs]})
        return [serial.job(j) for j in jobs]


@router.post("/cases/{case_id}/exports", response_model=S.ExportOut)
def create_export(case_id: str, body: S.ExportIn, u: User = Depends(investigator)):
    from ...db.models import FaceRow, SearchHit, Search
    with write_session() as s:
        case_or_404(s, u, case_id)
        if not body.face_ids and not body.search_hit_ids:
            raise APIError(400, "EMPTY_SELECTION", "select at least one face or confirmed match")
        for fid in body.face_ids:
            f = s.get(FaceRow, fid)
            if not f or s.get(Clip, f.clip_id).case_id != case_id:
                raise APIError(400, "BAD_FACE", f"face {fid} is not in this case")
        for hid in body.search_hit_ids:
            h = s.get(SearchHit, hid)
            if not h or s.get(Search, h.search_id).case_id != case_id:
                raise APIError(400, "BAD_HIT", f"search hit {hid} is not in this case")
            if h.review != "confirmed":
                raise APIError(400, "UNREVIEWED_HIT", f"search hit {hid} has not been confirmed by a reviewer")
        e = Export(case_id=case_id, selection_json=body.model_dump(), created_by=u.id)
        s.add(e)
        s.flush()
        j = enqueue(s, "export", {"export_id": e.id}, case_id=case_id, user_id=u.id)
        audit.append(s, "export_requested", user_id=u.id, role=u.role, object_type="export", object_id=e.id,
                     details={**body.model_dump(), "job_id": j.id})
        return serial.export(e, j.id)


@router.get("/cases/{case_id}/exports", response_model=list[S.ExportOut])
def list_exports(case_id: str, u: User = Depends(viewer), s: Session = Depends(get_db)):
    case_or_404(s, u, case_id)
    return [serial.export(e) for e in s.exec(select(Export).where(Export.case_id == case_id).order_by(Export.id.desc())).all()]


@router.get("/cases/{case_id}/jobs", response_model=list[S.JobOut])
def case_jobs(case_id: str, u: User = Depends(viewer), s: Session = Depends(get_db)):
    case_or_404(s, u, case_id)
    return [serial.job(j) for j in s.exec(select(Job).where(Job.case_id == case_id).order_by(Job.id.desc()).limit(200)).all()]
