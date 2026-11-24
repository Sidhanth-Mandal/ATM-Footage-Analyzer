"""Job handlers: `handler(payload, ctx) -> result`. Each finished job may enqueue successors.

Graph
  transaction clip: ingest -> timemap -> coarse -> link (per transaction) -> [human confirms] -> dense -> behaviour
  gallery clip:     ingest -> timemap -> coarse -> index
  on demand:        enhance, export, timemap (OCR re-run), behaviour
"""
from __future__ import annotations

import logging
import os
import shutil
import stat
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from sqlmodel import Session, select

from . import audit, paths
from .core import behaviour as beh
from .core import embed as emb
from .core import timemap as tmap
from .core.faces import dense_pass
from .core.index import FaceIndex
from .core.linking import score_tracks
from .core.persons import coarse_pass, load_tracks, save_tracks
from .core.quality import score_faces
from .core.report import to_jsonable, write_report
from .core.select import select_best
from .core.types import PersonTrack, ScoredFace, TimeMap
from .core.video import FrameSource, ingest, sha256_file
from .db.models import (
    Atm, BehaviourEvent, CandidateRow, Case, Clip, Enhancement, Export, FaceRow, Job, Template, Track, Transaction,
)
from .settings import Cfg, config_hash

log = logging.getLogger(__name__)


class Cancelled(Exception):
    pass


class JobError(Exception):
    """Expected failure with a user-facing message."""


class Ctx:
    def __init__(self, job: Job, cfg: Cfg, registry_factory: Callable, session_factory: Callable,
                 progress: Callable[[float, str], None], enqueue: Callable[..., int]):
        self.job = job
        self.cfg = cfg
        self._registry_factory = registry_factory
        self.write = session_factory           # context manager -> Session (BEGIN IMMEDIATE)
        self.progress = progress
        self.enqueue = enqueue
        self._registry = None

    @property
    def registry(self):
        if self._registry is None:
            self._registry = self._registry_factory()
        return self._registry

    def audit(self, s: Session, action: str, object_type: str, object_id: Any, details: Any = None):
        audit.append(s, action, user_id=self.job.created_by, role="system/worker", object_type=object_type,
                     object_id=object_id, details={"job_id": self.job.id, **(details or {})})


# ------------------------------------------------------------------ helpers

def _clip_src(clip: Clip) -> FrameSource:
    rot = int((clip.probe_json or {}).get("analysis_rotation", 0))
    return FrameSource(paths.absp(clip.analysis_path), rot)


def _atm(s: Session, clip: Clip) -> Atm | None:
    return s.get(Atm, clip.atm_id) if clip.atm_id else None


def _camera(atm: Atm | None) -> str:
    return atm.camera_type if atm and atm.camera_type in ("fascia", "lobby") else "fascia"


def _timemap(clip: Clip) -> TimeMap | None:
    return TimeMap.from_json(clip.timemap_json)


def _write_bytes(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return sha256_file(path)


def _imwrite(path: Path, img: np.ndarray, jpg_q: int = 95) -> str:
    ext = path.suffix.lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, jpg_q] if ext in (".jpg", ".jpeg") else [cv2.IMWRITE_PNG_COMPRESSION, 3]
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise JobError(f"could not encode {path.name}")
    return _write_bytes(path, buf.tobytes())


def get_index() -> FaceIndex:
    return FaceIndex(paths.index_dir())


def _txn_pts(txn: Transaction, clip: Clip) -> float | None:
    tm = _timemap(clip)
    if tm is None:
        return None
    return tmap.txn_to_pts(txn.txn_time_journal, clip.clock_offset_s, tm)


def _track_objs(clip: Clip) -> tuple[list[PersonTrack], dict]:
    return load_tracks(paths.clip_dir(clip.case_id, clip.id))


def _prefetch_frames(src: FrameSource, faces: list[ScoredFace]) -> dict[int, bytes]:
    """Decode and JPEG-encode the full frames for faces that will be stored (outside any DB transaction)."""
    out: dict[int, bytes] = {}
    for sf in faces:
        frame = src.get(sf.obs.pts)
        if frame is None:
            continue
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if ok:
            out[id(sf)] = buf.tobytes()
    return out


def _store_face(s: Session, clip: Clip, track: Track, sf: ScoredFace, frames: dict[int, bytes], tm: TimeMap | None,
                purpose: str, selected: bool) -> FaceRow:
    o, q = sf.obs, sf.q
    ox, oy = _crop_origin(o)
    row = FaceRow(track_id=track.id, clip_id=clip.id, frame_idx=int(o.frame_idx), pts=float(o.pts),
                  wall_time=tmap.wall_iso(tm, o.pts), bbox_json=[float(v) for v in o.bbox],
                  lmk_json=o.lmk5.tolist(), crop_origin_json=[ox, oy], total=q.total, grade=q.grade, sub_json=q.sub,
                  raw_json=q.raw, flags_json=q.flags, caps_json=q.caps, attrib_json=to_jsonable(sf.attrib),
                  searchable=q.searchable, selected=selected, purpose=purpose, detview_corrected=o.detview_corrected)
    s.add(row)
    s.flush()
    fdir = paths.face_dir(clip.case_id, clip.id, row.id)
    row.crop_path = paths.rel(fdir / "crop.png")
    row.crop_sha256 = _imwrite(fdir / "crop.png", o.raw_crop)
    row.aligned_path = paths.rel(fdir / "aligned.png")
    row.aligned_sha256 = _imwrite(fdir / "aligned.png", o.aligned)
    jpg = frames.get(id(sf))
    if jpg:
        row.frame_path = paths.rel(fdir / "frame.jpg")
        row.frame_sha256 = _write_bytes(fdir / "frame.jpg", jpg)
    return row


def _crop_origin(o) -> tuple[int, int]:
    """Origin of raw_crop (bbox + 25% margin, clipped), recomputed exactly as faces.crop_with_margin does."""
    H, W = o.frame_shape
    x0, y0, x1, y1 = [float(v) for v in o.bbox]
    w, h = x1 - x0, y1 - y0
    return max(0, int(x0 - 0.25 * w)), max(0, int(y0 - 0.25 * h))


def _prefilter(obs: list, n: int) -> list:
    if len(obs) <= n:
        return obs
    key = [o.det_conf * min(60.0, float(np.linalg.norm(o.lmk5[1] - o.lmk5[0]))) for o in obs]
    keep = sorted(np.argsort(key)[::-1][:n])
    return [obs[i] for i in keep]


def _delete_faces(s: Session, track: Track, purpose: str | None = None) -> list[int]:
    q = select(FaceRow).where(FaceRow.track_id == track.id)
    if purpose:
        q = q.where(FaceRow.purpose == purpose)
    ids = []
    for f in s.exec(q).all():
        for e in s.exec(select(Enhancement).where(Enhancement.face_id == f.id)).all():
            s.delete(e)
        d = paths.face_dir(track_clip_case(s, track), track.clip_id, f.id)
        shutil.rmtree(d, ignore_errors=True)
        ids.append(f.id)
        s.delete(f)
    return ids


def track_clip_case(s: Session, track: Track) -> str:
    return s.get(Clip, track.clip_id).case_id


def _embed_rows(ctx: Ctx, rows: list[tuple[FaceRow, ScoredFace]]) -> None:
    """Store embeddings for every searchable stored face (so any of them can be used as a query)."""
    todo = [(r, sf) for r, sf in rows if sf.q.searchable and r.embedding is None]
    if not todo:
        return
    vecs = emb.embed(ctx.registry, [sf.obs.aligned for _, sf in todo], enhanced=[False] * len(todo),
                     searchable=[True] * len(todo), batch=int(ctx.cfg.dense.batch))
    for (r, _), v in zip(todo, vecs):
        r.embedding = v.astype(np.float32).tobytes()


def _embed_and_index(s: Session, ctx: Ctx, track: Track, rows: list[tuple[FaceRow, ScoredFace]]) -> dict:
    """Embed searchable faces, build the template and add both to the FAISS indexes."""
    use = [(r, sf) for r, sf in rows if sf.q.searchable]
    idx = get_index()
    old_t = s.exec(select(Template).where(Template.track_id == track.id)).first()
    if old_t:
        idx.remove(old_t.face_ids_json, [track.id])
        s.delete(old_t)
        s.flush()
    if not use:
        track.indexed = False
        return {"embedded": 0, "template": False}
    _embed_rows(ctx, use)
    vecs = np.stack([np.frombuffer(r.embedding, np.float32) for r, _ in use])
    t, w = emb.template(vecs, [sf.q.total for _, sf in use])
    s.add(Template(track_id=track.id, n_faces=len(use), weights_json=w, face_ids_json=[r.id for r, _ in use],
                   vector=t.astype(np.float32).tobytes()))
    idx.add({r.id: v for (r, _), v in zip(use, vecs)}, {track.id: t})
    track.indexed = True
    return {"embedded": len(use), "template": True}


def write_clip_report(s: Session, clip: Clip, cfg: Cfg, registry=None) -> None:
    tracks = s.exec(select(Track).where(Track.clip_id == clip.id)).all()
    txns = s.exec(select(Transaction).where(Transaction.clip_id == clip.id)).all()
    rep = {
        "clip_id": clip.id, "case_id": clip.case_id, "orig_name": clip.orig_name, "orig_sha256": clip.orig_sha256,
        "analysis_sha256": clip.analysis_sha256, "proxy_sha256": clip.proxy_sha256, "normalised": clip.normalised,
        "probe": clip.probe_json, "ingest": clip.ingest_json, "timemap": clip.timemap_json,
        "clock_offset_s": clip.clock_offset_s, "tracks_meta": clip.tracks_meta_json,
        "transactions": [], "behaviour": [], "config_sha256": config_hash(cfg),
        "models": registry.model_info() if registry else None, "generated_at": datetime.utcnow().isoformat() + "Z",
    }
    for t in txns:
        cands = s.exec(select(CandidateRow).where(CandidateRow.transaction_id == t.id).order_by(CandidateRow.rank)).all()
        chosen = s.get(Track, t.chosen_track_id) if t.chosen_track_id else None
        faces = []
        if chosen:
            faces = [{"id": f.id, "pts": f.pts, "frame_idx": f.frame_idx, "wall_time": f.wall_time, "total": f.total,
                      "grade": f.grade, "sub": f.sub_json, "flags": f.flags_json, "searchable": f.searchable,
                      "frame_sha256": f.frame_sha256, "crop_sha256": f.crop_sha256}
                     for f in s.exec(select(FaceRow).where(FaceRow.track_id == chosen.id, FaceRow.selected == True)  # noqa: E712
                                     .order_by(FaceRow.total.desc())).all()]
        rep["transactions"].append({
            "id": t.id, "txn_ref": t.txn_ref, "txn_time_journal": t.txn_time_journal, "t_txn_pts": t.t_txn_pts,
            "window": [t.window_before_s, t.window_after_s],
            "candidates": [{"track_no": s.get(Track, c.track_id).track_no, "rank": c.rank, "score": c.score,
                            "parts": c.parts_json, "reasons": c.reasons_json} for c in cands],
            "chosen_track_no": chosen.track_no if chosen else None, "confirmed_by": t.confirmed_by,
            "confirmed_at": t.confirmed_at, "selected_faces": faces,
            "face_visible_frac": chosen.face_visible_frac if chosen else None,
        })
    rep["behaviour"] = [{"type": e.type, "t_start": e.t_start, "t_end": e.t_end, "track_ids": e.track_ids_json,
                         "details": e.details_json}
                        for e in s.exec(select(BehaviourEvent).where(BehaviourEvent.clip_id == clip.id)).all()]
    rep["n_tracks"] = len(tracks)
    write_report(paths.clip_dir(clip.case_id, clip.id) / "report.json", rep)


# ------------------------------------------------------------------ handlers

def h_ingest(payload: dict, ctx: Ctx) -> dict:
    cfg = ctx.cfg
    with ctx.write() as s:
        clip = s.get(Clip, payload["clip_id"])
        clip.status = "ingesting"
        case_id, clip_id, orig, sha = clip.case_id, clip.id, paths.absp(clip.orig_path), clip.orig_sha256
    ctx.progress(0.05, "probing")
    res = ingest(orig, paths.clip_dir(case_id, clip_id), cfg, sha=sha)
    with ctx.write() as s:
        clip = s.get(Clip, clip_id)
        clip.probe_json = res.probe.to_json() if res.probe else None
        clip.ingest_json = {"decisions": res.decisions}
        if res.error:
            clip.status = "failed"
            clip.error_json = res.error.to_json()
            ctx.audit(s, "ingest_failed", "clip", clip_id, {"code": res.error.code, "message": res.error.message})
            return {"ok": False, "error": res.error.to_json()}
        clip.analysis_path = paths.rel(res.analysis_path)
        clip.analysis_sha256 = res.analysis_sha256
        clip.proxy_path = paths.rel(res.proxy_path)
        clip.proxy_sha256 = res.proxy_sha256
        clip.normalised = res.normalised
        pj = dict(clip.probe_json)
        pj["analysis_rotation"] = 0 if res.normalised else int(res.probe.rotation_deg)
        with FrameSource(res.analysis_path, pj["analysis_rotation"]) as fs:
            pj["analysis_start_time"] = fs.start_time
            pj["analysis_fps"] = fs.fps
            pj["analysis_width"], pj["analysis_height"] = fs.width, fs.height
            pj["analysis_duration"] = fs.duration or res.probe.duration_s
        clip.probe_json = pj
        clip.status = "ingested"
        clip.error_json = None
        ctx.audit(s, "ingested", "clip", clip_id, {"normalised": res.normalised, "analysis_sha256": res.analysis_sha256,
                                                    "proxy_sha256": res.proxy_sha256, "decisions": res.decisions})
        ctx.enqueue("timemap", {"clip_id": clip_id}, case_id=case_id)
    return {"ok": True, "normalised": res.normalised, "decisions": res.decisions}


def h_timemap(payload: dict, ctx: Ctx) -> dict:
    cfg = ctx.cfg
    with ctx.write() as s:
        clip = s.get(Clip, payload["clip_id"])
        atm = _atm(s, clip)
        tz = atm.tz if atm else "UTC"
        date_order = atm.date_order if atm else "DMY"
        roi = payload.get("roi") or (atm.ocr_roi_json if atm else None)
        meta_end = bool(atm.metadata_is_end) if atm else False
        probe = clip.probe_json or {}
        existing = clip.timemap_json
    tm, reads, notes = None, [], []
    if roi:
        ctx.progress(0.1, "OCR of the timestamp overlay")
        with _clip_src(clip) as src:
            frames = src.iter_sampled(1.0 / float(cfg.timemap.ocr_every_s))
            tm, reads = tmap.fit_ocr(frames, roi, ctx.registry.ocr(), cfg, date_order, tz)
        if tm is None:
            notes.append(f"OCR fit failed ({sum(1 for r in reads if r['wall'])} of {len(reads)} reads parsed); "
                         "falling back")
    if tm is None and existing and existing.get("method") == "manual" and not payload.get("force"):
        tm = TimeMap.from_json(existing)
    if tm is None:
        ct = probe.get("creation_time")
        if ct:
            tm = tmap.from_metadata(datetime.fromisoformat(ct), float(probe.get("analysis_duration") or probe.get("duration_s") or 0),
                                    meta_end, tz, float(probe.get("analysis_start_time") or 0.0))
    if tm is not None:
        tm.notes = notes + tm.notes
    with ctx.write() as s:
        clip = s.get(Clip, payload["clip_id"])
        clip.timemap_json = tm.to_json() if tm else None
        clip.timemap_reads_json = reads[:200] if reads else clip.timemap_reads_json
        ctx.audit(s, "timemap_set", "clip", clip.id, {"method": tm.method if tm else None, "notes": notes,
                                                      "inliers": tm.inliers if tm else 0})
        relink = _refresh_txn_pts(s, clip, ctx)
        if not payload.get("only"):
            ctx.enqueue("coarse", {"clip_id": clip.id}, case_id=clip.case_id)
    return {"method": tm.method if tm else None, "reads": len(reads), "relinked": relink}


def _refresh_txn_pts(s: Session, clip: Clip, ctx: Ctx) -> int:
    n = 0
    for t in s.exec(select(Transaction).where(Transaction.clip_id == clip.id)).all():
        t.t_txn_pts = _txn_pts(t, clip)
        if clip.status in ("tracked",) and t.status in ("pending", "linked"):
            ctx.enqueue("link", {"transaction_id": t.id}, case_id=clip.case_id)
            n += 1
    return n


def h_coarse(payload: dict, ctx: Ctx) -> dict:
    cfg = ctx.cfg
    with ctx.write() as s:
        clip = s.get(Clip, payload["clip_id"])
        clip.status = "tracking"
        atm = _atm(s, clip)
        cam = _camera(atm)
    with _clip_src(clip) as src:
        tracks, meta = coarse_pass(src, ctx.registry, cfg, cam, progress=ctx.progress)
    cdir = paths.clip_dir(clip.case_id, clip.id)
    npz, _ = save_tracks(tracks, meta, cdir)
    with ctx.write() as s:
        clip = s.get(Clip, payload["clip_id"])
        for old in s.exec(select(Track).where(Track.clip_id == clip.id)).all():
            for c in s.exec(select(CandidateRow).where(CandidateRow.track_id == old.id)).all():
                s.delete(c)
            _delete_faces(s, old)
            t_old = s.exec(select(Template).where(Template.track_id == old.id)).first()
            if t_old:
                get_index().remove(t_old.face_ids_json, [old.id])
                s.delete(t_old)
            for t in s.exec(select(Transaction).where(Transaction.chosen_track_id == old.id)).all():
                t.chosen_track_id, t.status = None, "pending"
            s.delete(old)
        s.flush()
        for t in tracks:
            s.add(Track(clip_id=clip.id, track_no=t.track_id, source=t.source, t_start=t.t_start, t_end=t.t_end,
                        n_samples=len(t.samples), samples_path=paths.rel(npz)))
        clip.tracks_meta_json = meta
        clip.status = "tracked"
        ctx.audit(s, "tracked", "clip", clip.id, {"n_tracks": len(tracks), "camera_type": cam})
        if clip.kind == "gallery":
            ctx.enqueue("index", {"clip_id": clip.id}, case_id=clip.case_id)
        else:
            for t in s.exec(select(Transaction).where(Transaction.clip_id == clip.id)).all():
                ctx.enqueue("link", {"transaction_id": t.id}, case_id=clip.case_id)
    return {"n_tracks": len(tracks), "meta": meta}


def _thumb(src: FrameSource, pts: float, bbox: np.ndarray, out: Path) -> str | None:
    frame = src.get(pts)
    if frame is None:
        return None
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = [float(v) for v in bbox]
    w, h = x1 - x0, y1 - y0
    X0, Y0 = int(max(0, x0 - 0.2 * w)), int(max(0, y0 - 0.1 * h))
    X1, Y1 = int(min(W, x1 + 0.2 * w)), int(min(H, y1 + 0.1 * h))
    img = frame.copy()
    cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (0, 200, 255), max(2, W // 400))
    crop = img[Y0:Y1, X0:X1]
    if crop.size == 0:
        crop = img
    scale = 360 / max(crop.shape[0], 1)
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    _imwrite(out, crop, 88)
    return paths.rel(out)


def h_link(payload: dict, ctx: Ctx) -> dict:
    cfg = ctx.cfg
    with ctx.write() as s:
        txn = s.get(Transaction, payload["transaction_id"])
        clip = s.get(Clip, txn.clip_id)
        atm = _atm(s, clip)
        rois = (atm.rois_json if atm else {}) or {}
        cam = _camera(atm)
        t_pts = _txn_pts(txn, clip)
        if t_pts is None:
            raise JobError("No time mapping for this clip. Set a manual start time or OCR region first.")
        txn.t_txn_pts = t_pts
        rows = {t.track_no: t for t in s.exec(select(Track).where(Track.clip_id == clip.id)).all()}
    tracks, meta = _track_objs(clip)
    W, H = int(meta["width"]), int(meta["height"])
    notes = []
    dur = float((clip.probe_json or {}).get("analysis_duration") or 0)
    w0, w1 = t_pts - txn.window_before_s, t_pts + txn.window_after_s
    if dur and (w1 < 0 or w0 > dur):
        notes.append(f"transaction window [{w0:.1f}, {w1:.1f}] s lies outside the clip (0-{dur:.1f} s): "
                     "check the time mapping and clock offset")
    cands = score_tracks(tracks, t_pts, txn.window_before_s, txn.window_after_s, rois, cam, W, H, cfg)
    if not cands:
        notes.append("no track was in the near field during the window")
    tdir = paths.clip_dir(clip.case_id, clip.id) / "thumbs"
    thumbs = {}
    with _clip_src(clip) as src:
        for c in cands:
            tr = next(t for t in tracks if t.track_id == c.track_id)
            smp = min(tr.samples, key=lambda x: abs(x.pts - c.best_view_pts))
            thumbs[c.track_id] = _thumb(src, c.best_view_pts, smp.bbox, tdir / f"cand_{txn.id}_{c.track_id}.jpg")
    with ctx.write() as s:
        txn = s.get(Transaction, payload["transaction_id"])
        for old in s.exec(select(CandidateRow).where(CandidateRow.transaction_id == txn.id)).all():
            s.delete(old)
        for rank, c in enumerate(cands, 1):
            s.add(CandidateRow(transaction_id=txn.id, track_id=rows[c.track_id].id, rank=rank, score=c.score,
                               parts_json=c.parts, reasons_json=c.reasons, best_view_pts=c.best_view_pts,
                               thumb_path=thumbs.get(c.track_id)))
        txn.notes_json = notes
        if txn.status in ("pending", "linked"):
            txn.status = "linked"
        ctx.audit(s, "candidates_ranked", "transaction", txn.id,
                  {"t_txn_pts": t_pts, "candidates": [[c.track_id, round(c.score, 4)] for c in cands]})
    return {"t_txn_pts": t_pts, "n_candidates": len(cands), "notes": notes}


def _select_store(scored: list[ScoredFace], cfg) -> tuple[list[ScoredFace], list[ScoredFace]]:
    sel = select_best(scored, int(cfg.select.top_k), float(cfg.select.min_gap_s))
    extra = select_best([f for f in scored if f not in sel], int(cfg.select.get("store_n", 24)), 0.2)
    return sel, extra


def h_dense(payload: dict, ctx: Ctx) -> dict:
    """Dense pass + scoring + selection + query embedding on the confirmed track."""
    cfg = ctx.cfg
    with ctx.write() as s:
        txn = s.get(Transaction, payload["transaction_id"]) if payload.get("transaction_id") else None
        track = s.get(Track, payload["track_id"])
        clip = s.get(Clip, track.clip_id)
    tracks, meta = _track_objs(clip)
    pt = next((t for t in tracks if t.track_id == track.track_no), None)
    if pt is None:
        raise JobError("track samples not found; re-run tracking")

    def prog(stage, lo, hi):
        return lambda f, m: ctx.progress(lo + (hi - lo) * f, f"{stage}: {m}")

    with _clip_src(clip) as src:
        res = dense_pass(src, [pt], ctx.registry, cfg, progress=prog("dense", 0.0, 0.55))
        obs, stats = res[pt.track_id]
        obs = _prefilter(obs, int(cfg.dense.get("max_faces_per_track", 600)))
        scored = _score_with_oom_retry(obs, ctx, prog("score", 0.55, 0.85))
        sel, extra = _select_store(scored, cfg)
        tm = _timemap(clip)
        ctx.progress(0.86, "extracting full frames for selected faces")
        frames = _prefetch_frames(src, sel + extra)
    ctx.progress(0.9, "writing selected faces")
    with ctx.write() as s:
        track = s.get(Track, payload["track_id"])
        old_ids = _delete_faces(s, track, "best")
        stored = []
        for sf in sel:
            stored.append((_store_face(s, clip, track, sf, frames, tm, "best", True), sf))
        for sf in extra:
            stored.append((_store_face(s, clip, track, sf, frames, tm, "best", False), sf))
        track.face_visible_frac = stats.face_visible_frac
        track.dense_status = "done"
        track.stats_json = {"frames": stats.frames, "frames_with_face": stats.frames_with_face,
                            "dropped_inconsistent": stats.dropped_inconsistent, "faces_scored": len(scored),
                            "grades": {g: sum(1 for f in scored if f.q.grade == g) for g in "ABC"},
                            "config_sha256": config_hash(cfg)}
        _embed_rows(ctx, stored)
        emb_res = _embed_and_index(s, ctx, track, [(r, sf) for r, sf in stored if r.selected])
        if txn is not None:
            txn = s.get(Transaction, txn.id)
            txn.status = "analysed"
        ctx.audit(s, "faces_extracted", "track", track.id,
                  {"selected": [r.id for r, _ in stored if r.selected], "removed_face_ids": old_ids,
                   "face_visible_frac": round(stats.face_visible_frac, 3), **emb_res})
        write_clip_report(s, clip, cfg, ctx.registry)
        ctx.enqueue("behaviour", {"clip_id": clip.id}, case_id=clip.case_id)
    return {"faces": len(scored), "selected": len(sel), "face_visible_frac": stats.face_visible_frac, **emb_res}


def _score_with_oom_retry(obs, ctx: Ctx, progress):
    try:
        return score_faces(obs, ctx.registry, ctx.cfg, progress)
    except Exception as e:
        if "out of memory" not in str(e).lower() and type(e).__name__ != "OutOfMemoryError":
            raise
        log.warning("CUDA OOM while scoring; halving batch and retrying once")
        try:
            ctx.registry.torch.cuda.empty_cache()
        except Exception:
            pass
        ctx.cfg.dense["batch"] = max(1, int(ctx.cfg.dense.batch) // 2)
        return score_faces(obs, ctx.registry, ctx.cfg, progress)


def h_index(payload: dict, ctx: Ctx) -> dict:
    """Gallery indexing: dense pass at gallery_frame_step on every track, embed best searchable faces."""
    cfg = ctx.cfg
    with ctx.write() as s:
        clip = s.get(Clip, payload["clip_id"])
        clip.status = "indexing"
    tracks, meta = _track_objs(clip)
    if not tracks:
        with ctx.write() as s:
            s.get(Clip, clip.id).status = "indexed"
        return {"tracks": 0}
    summary = {"tracks": len(tracks), "indexed": 0, "not_comparable": 0, "too_small": 0}
    # A person box shorter than ~12 x the minimum IOD cannot contain a face with IOD >= iod_min_px.
    min_h = 12.0 * float(cfg.quality.iod_min_px)
    big = [t for t in tracks if t.source == "face" or max(s.bbox[3] - s.bbox[1] for s in t.samples) >= min_h]
    small = [t for t in tracks if t not in big]
    with ctx.write() as s:
        for pt in small:
            tr = s.exec(select(Track).where(Track.clip_id == clip.id, Track.track_no == pt.track_id)).first()
            tr.stats_json = {"skipped": f"person box never taller than {min_h:.0f} px (face too small to compare)"}
            tr.indexed = False
    summary["too_small"] = len(small)
    tracks = big
    if not tracks:
        with ctx.write() as s:
            c2 = s.get(Clip, clip.id)
            c2.status = "indexed"
            ctx.audit(s, "gallery_indexed", "clip", clip.id, summary)
        return summary
    ctx.registry.hold_open_vocab = True
    with _clip_src(clip) as src:
        res = dense_pass(src, tracks, ctx.registry, cfg, step=int(cfg.search.gallery_frame_step),
                         progress=lambda f, m: ctx.progress(0.6 * f, f"dense: {m}"))
        tm = _timemap(clip)
        for i, pt in enumerate(tracks):
            obs, stats = res[pt.track_id]
            # Only faces_per_track faces are kept per gallery track, so score a pool of the most promising ones
            # (detection confidence x inter-ocular distance), spread over time by _prefilter's ordering.
            obs = _prefilter(obs, max(60, 12 * int(cfg.search.faces_per_track)))
            ctx.progress(0.6 + 0.38 * i / len(tracks), f"scoring track {i + 1}/{len(tracks)}")
            scored = _score_with_oom_retry(obs, ctx, None) if obs else []
            picks = select_best(scored, int(cfg.search.faces_per_track), float(cfg.select.min_gap_s), only_searchable=True)
            show = picks or select_best(scored, 1, 0.0)
            frames = _prefetch_frames(src, show)
            with ctx.write() as s:
                track = s.exec(select(Track).where(Track.clip_id == clip.id, Track.track_no == pt.track_id)).first()
                _delete_faces(s, track, "gallery")
                stored = [(_store_face(s, clip, track, sf, frames, tm, "gallery", True), sf) for sf in show]
                track.face_visible_frac = stats.face_visible_frac
                track.stats_json = {"frames": stats.frames, "frames_with_face": stats.frames_with_face,
                                    "faces_scored": len(scored), "searchable": sum(1 for f in scored if f.q.searchable)}
                r = _embed_and_index(s, ctx, track, stored)
                summary["indexed" if r["template"] else "not_comparable"] += 1
    ctx.registry.hold_open_vocab = False
    ctx.registry.release("open_vocab")
    with ctx.write() as s:
        clip = s.get(Clip, payload["clip_id"])
        clip.status = "indexed"
        ctx.audit(s, "gallery_indexed", "clip", clip.id, summary)
        write_clip_report(s, clip, cfg, ctx.registry)
    return summary


def h_behaviour(payload: dict, ctx: Ctx) -> dict:
    cfg = ctx.cfg
    with ctx.write() as s:
        clip = s.get(Clip, payload["clip_id"])
        atm = _atm(s, clip)
        rois = (atm.rois_json if atm else {}) or {}
        cam = _camera(atm)
        txns = s.exec(select(Transaction).where(Transaction.clip_id == clip.id)).all()
        chosen_nos = [s.get(Track, t.chosen_track_id).track_no for t in txns if t.chosen_track_id]
        txn_pts = [t.t_txn_pts for t in txns if t.t_txn_pts is not None]
    tracks, meta = _track_objs(clip)
    W, H, fps = int(meta["width"]), int(meta["height"]), float(meta["sample_fps"])
    events = []
    for no in set(chosen_nos):
        cust = next((t for t in tracks if t.track_id == no), None)
        if cust:
            events += beh.shoulder_surfing(cust, tracks, rois, cam, W, H, cfg, fps)
    events += beh.loitering(tracks, txn_pts, rois, cam, W, H, cfg, fps)
    if rois.get("card_slot") or rois.get("keypad"):
        with _clip_src(clip) as src:
            events += beh.tampering(src, tracks, rois, cam, cfg)
    with ctx.write() as s:
        rows = {t.track_no: t.id for t in s.exec(select(Track).where(Track.clip_id == clip.id)).all()}
        for old in s.exec(select(BehaviourEvent).where(BehaviourEvent.clip_id == clip.id)).all():
            s.delete(old)
        for e in events:
            d = dict(e["details"])
            d["track_nos"] = e["track_ids"]
            s.add(BehaviourEvent(clip_id=clip.id, type=e["type"], t_start=float(e["t_start"]), t_end=float(e["t_end"]),
                                 track_ids_json=[rows.get(n) for n in e["track_ids"] if n in rows],
                                 details_json=to_jsonable(d)))
        ctx.audit(s, "behaviour_checked", "clip", clip.id, {"events": len(events)})
        write_clip_report(s, s.get(Clip, clip.id), cfg, None)
    return {"events": len(events)}


def h_enhance(payload: dict, ctx: Ctx) -> dict:
    from .core import enhance as enh
    with ctx.write() as s:
        e = s.get(Enhancement, payload["enhancement_id"])
        face = s.get(FaceRow, e.face_id)
        clip = s.get(Clip, face.clip_id)
        method, params = e.method, dict(e.params_json or {})
    crop = cv2.imread(str(paths.absp(face.crop_path)))
    if crop is None:
        raise JobError("source crop not found")
    lmk = np.asarray(face.lmk_json, np.float32) - np.asarray(face.crop_origin_json or [0, 0], np.float32)
    sess = ctx.registry.gfpgan() if method == "gfpgan_v1.4_onnx" else None
    out = enh.enhance(method, crop, lmk, params, sess)
    meta = {"source_face_id": str(face.id), "method": method, "params": params, "source_sha256": face.crop_sha256,
            "banner": f"ENHANCED: {method} - NOT ORIGINAL"}
    p = paths.clip_dir(clip.case_id, clip.id) / "enhanced" / f"{e.id}.png"
    sha = _write_bytes(p, enh.png_with_metadata(out, meta))
    with ctx.write() as s:
        e = s.get(Enhancement, payload["enhancement_id"])
        e.path, e.sha256 = paths.rel(p), sha
        ctx.audit(s, "enhancement_applied", "face", face.id, {"enhancement_id": e.id, "method": method,
                                                              "params": params, "sha256": sha})
    return {"sha256": sha}


def h_export(payload: dict, ctx: Ctx) -> dict:
    from .evidence import build_pdf
    with ctx.write() as s:
        ex = s.get(Export, payload["export_id"])
        ex.status = "running"
    out = paths.exports_dir(ex.case_id) / f"{ex.id}.pdf"
    with ctx.write() as s:
        ex = s.get(Export, payload["export_id"])
        build_pdf(s, ex, out, ctx.cfg, ctx.registry.model_info() if ctx._registry else _static_model_info(ctx.cfg))
        ex.path, ex.sha256, ex.status = paths.rel(out), sha256_file(out), "done"
        try:
            os.chmod(out, stat.S_IREAD)
        except OSError:
            pass
        ctx.audit(s, "export_created", "export", ex.id, {"sha256": ex.sha256, "selection": ex.selection_json})
    return {"sha256": ex.sha256}


def _static_model_info(cfg: Cfg) -> dict:
    from .settings import models_dir
    md = models_dir(cfg)
    files = {"person_pose": md / cfg.models.person_pose, "face_det": md / "uniface" / "scrfd_10g.onnx",
             "recognizer": md / "uniface" / "adaface_ir_101.onnx", "fiqa": md / "uniface" / "ediffiqa_m.onnx",
             "head_pose": md / "uniface" / "headpose_resnet18.onnx", "parser": md / "uniface" / "parsing_resnet18.onnx",
             "attrib": md / "uniface" / "face_attrib_net.onnx", "open_vocab": md / cfg.models.open_vocab,
             "gfpgan": md / cfg.models.gfpgan}
    return {k: {"file": p.name, "sha256": sha256_file(p)[:16] if p.exists() else None} for k, p in files.items()}


HANDLERS: dict[str, Callable[[dict, Ctx], dict]] = {
    "ingest": h_ingest, "timemap": h_timemap, "coarse": h_coarse, "link": h_link, "dense": h_dense,
    "index": h_index, "behaviour": h_behaviour, "enhance": h_enhance, "export": h_export,
}

GPU_JOBS = {"coarse", "dense", "index", "enhance", "timemap"}

__all__ = ["HANDLERS", "Ctx", "Cancelled", "JobError", "write_clip_report", "get_index", "Case"]
