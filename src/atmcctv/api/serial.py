"""Row -> response schema converters."""
from __future__ import annotations

from sqlmodel import Session, func, select

from ..core.timemap import from_epoch
from ..db.models import (
    Atm, CandidateRow, Case, CaseMember, Clip, Enhancement, Export, FaceRow, Job, Track, Transaction, User,
)
from . import schemas as S


def user(u: User) -> S.UserOut:
    return S.UserOut(id=u.id, username=u.username, role=u.role, is_active=u.is_active, created_at=u.created_at)


def case(s: Session, c: Case) -> S.CaseOut:
    members = [m.user_id for m in s.exec(select(CaseMember).where(CaseMember.case_id == c.id)).all()]
    n = s.exec(select(func.count()).select_from(Clip).where(Clip.case_id == c.id)).one()
    return S.CaseOut(id=c.id, ref_no=c.ref_no, title=c.title, status=c.status, created_by=c.created_by,
                     created_at=c.created_at, member_ids=members, n_clips=int(n))


def atm(a: Atm) -> S.AtmOut:
    return S.AtmOut(id=a.id, atm_code=a.atm_code, location=a.location, camera_type=a.camera_type,
                    rois=a.rois_json or {}, ocr_roi=a.ocr_roi_json, date_order=a.date_order, tz=a.tz,
                    metadata_is_end=a.metadata_is_end)


def job(j: Job) -> S.JobOut:
    return S.JobOut(id=j.id, type=j.type, case_id=j.case_id, payload=j.payload_json or {}, result=j.result_json,
                    status=j.status, progress=j.progress, message=j.message, created_at=j.created_at,
                    started_at=j.started_at, finished_at=j.finished_at)


def clip(s: Session, c: Clip, with_reads: bool = False) -> S.ClipOut:
    n = s.exec(select(func.count()).select_from(Track).where(Track.clip_id == c.id)).one()
    start_wall = None
    tm = c.timemap_json
    if tm:
        st = float((c.probe_json or {}).get("analysis_start_time") or 0.0)
        start_wall = from_epoch(tm["a"] + tm["b"] * st, tm["tz"]).isoformat(timespec="seconds")
    return S.ClipOut(id=c.id, case_id=c.case_id, atm_id=c.atm_id, kind=c.kind, orig_name=c.orig_name,
                     orig_sha256=c.orig_sha256, size=c.size, analysis_sha256=c.analysis_sha256,
                     proxy_sha256=c.proxy_sha256, normalised=c.normalised, status=c.status, probe=c.probe_json,
                     ingest=c.ingest_json, error=c.error_json, timemap=c.timemap_json,
                     timemap_reads=c.timemap_reads_json if with_reads else None, clock_offset_s=c.clock_offset_s,
                     tracks_meta=c.tracks_meta_json, created_at=c.created_at, n_tracks=int(n),
                     proxy_url=f"/media/proxy/{c.id}" if c.proxy_path else None, start_wall=start_wall)


def txn(t: Transaction, clip_name: str | None = None) -> S.TxnOut:
    return S.TxnOut(id=t.id, case_id=t.case_id, clip_id=t.clip_id, txn_ref=t.txn_ref, txn_time_journal=t.txn_time_journal,
                    amount=t.amount, window_before_s=t.window_before_s, window_after_s=t.window_after_s,
                    t_txn_pts=t.t_txn_pts, status=t.status, chosen_track_id=t.chosen_track_id,
                    confirmed_by=t.confirmed_by, confirmed_at=t.confirmed_at, notes=t.notes_json or [],
                    clip_name=clip_name)


def candidate(s: Session, c: CandidateRow) -> S.CandidateOut:
    tr = s.get(Track, c.track_id)
    return S.CandidateOut(id=c.id, track_id=c.track_id, track_no=tr.track_no, source=tr.source, rank=c.rank,
                          score=c.score, parts=c.parts_json, reasons=c.reasons_json, best_view_pts=c.best_view_pts,
                          thumb_url=f"/media/candidate/{c.id}" if c.thumb_path else None, t_start=tr.t_start,
                          t_end=tr.t_end)


def track(t: Track) -> S.TrackOut:
    return S.TrackOut(id=t.id, clip_id=t.clip_id, track_no=t.track_no, source=t.source, t_start=t.t_start,
                      t_end=t.t_end, n_samples=t.n_samples, face_visible_frac=t.face_visible_frac,
                      dense_status=t.dense_status, indexed=t.indexed, stats=t.stats_json)


def enhancement(e: Enhancement, job_id: int | None = None) -> S.EnhancementOut:
    return S.EnhancementOut(id=e.id, face_id=e.face_id, method=e.method, params=e.params_json or {}, sha256=e.sha256,
                            url=f"/media/enhanced/{e.id}" if e.path else None, created_at=e.created_at, job_id=job_id)


def face(s: Session | None, f: FaceRow, with_enh: bool = False) -> S.FaceOut:
    urls = {k: f"/media/face/{f.id}/{k}" for k, p in (("frame", f.frame_path), ("crop", f.crop_path),
                                                      ("aligned", f.aligned_path)) if p}
    enh = []
    if with_enh and s is not None:
        enh = [enhancement(e).model_dump() for e in s.exec(select(Enhancement).where(Enhancement.face_id == f.id)).all()]
    return S.FaceOut(id=f.id, track_id=f.track_id, clip_id=f.clip_id, frame_idx=f.frame_idx, pts=f.pts,
                     wall_time=f.wall_time, bbox=f.bbox_json, total=f.total, grade=f.grade, sub=f.sub_json,
                     raw=f.raw_json, flags=f.flags_json, caps=f.caps_json, attrib=f.attrib_json or {},
                     searchable=f.searchable, selected=f.selected, purpose=f.purpose,
                     detview_corrected=f.detview_corrected, has_embedding=f.embedding is not None,
                     frame_sha256=f.frame_sha256, crop_sha256=f.crop_sha256, aligned_sha256=f.aligned_sha256,
                     urls=urls, enhancements=enh)


def export(e: Export, job_id: int | None = None) -> S.ExportOut:
    return S.ExportOut(id=e.id, case_id=e.case_id, sha256=e.sha256, selection=e.selection_json or {}, status=e.status,
                       created_by=e.created_by, created_at=e.created_at, job_id=job_id)
