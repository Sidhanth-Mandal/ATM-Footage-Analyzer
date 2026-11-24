"""Face search over the FAISS index (no torch in the API: query embeddings come from the DB)."""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from ... import audit, paths
from ...core import embed as emb
from ...core.index import FaceIndex, band_of, effective_bands
from ...db.models import Clip, FaceRow, Search, SearchHit, Template, Track, User, now
from ...db.session import write_session
from ...settings import get_config
from .. import schemas as S
from ..deps import APIError, accessible_case_ids, case_or_404, get_db, investigator, not_found, viewer

router = APIRouter(tags=["search"])


def _face_brief(f: FaceRow) -> dict:
    return {"id": f.id, "pts": f.pts, "wall_time": f.wall_time, "total": f.total, "grade": f.grade,
            "flags": f.flags_json, "searchable": f.searchable, "crop_url": f"/media/face/{f.id}/crop",
            "aligned_url": f"/media/face/{f.id}/aligned", "track_id": f.track_id, "clip_id": f.clip_id}


def _hit_out(s: Session, h: SearchHit) -> S.HitOut:
    tr = s.get(Track, h.track_id)
    c = s.get(Clip, tr.clip_id)
    faces = [_face_brief(f) for f in (s.get(FaceRow, i) for i in h.top_face_ids_json) if f]
    return S.HitOut(id=h.id, track_id=tr.id, track_no=tr.track_no, clip_id=c.id, clip_name=c.orig_name,
                    case_id=c.case_id, score=h.score, band=h.band, parts=h.parts_json, faces=faces, review=h.review,
                    reviewed_by=h.reviewed_by, reviewed_at=h.reviewed_at, note=h.note)


def _search_out(s: Session, sr: Search) -> S.SearchOut:
    hits = s.exec(select(SearchHit).where(SearchHit.search_id == sr.id).order_by(SearchHit.score.desc())).all()
    q = [_face_brief(f) for f in (s.get(FaceRow, i) for i in sr.query_face_ids_json) if f]
    bands, calibrated = effective_bands(sr.bands_json)
    return S.SearchOut(search_id=sr.id, case_id=sr.case_id, query_faces=q, bands=bands, calibrated=calibrated,
                       hits=[_hit_out(s, h) for h in hits], not_comparable=sr.not_comparable_json or [],
                       created_at=sr.created_at)


@router.post("/search", response_model=S.SearchOut)
def search(body: S.SearchIn, u: User = Depends(investigator)):
    cfg = get_config()
    with write_session() as s:
        case_or_404(s, u, body.case_id)
        faces = []
        for fid in body.face_ids:
            f = s.get(FaceRow, fid)
            if not f:
                raise not_found(f"face {fid}")
            if s.get(Clip, f.clip_id).case_id != body.case_id:
                raise APIError(400, "BAD_FACE", f"face {fid} is not in this case")
            if not f.searchable or f.embedding is None:
                raise APIError(400, "NOT_SEARCHABLE", f"face {fid} is not searchable (grade C, covered face or "
                                                      "low visibility); choose a searchable face")
            faces.append(f)
        vecs = np.stack([np.frombuffer(f.embedding, np.float32) for f in faces])
        q, _ = emb.template(vecs, [f.total for f in faces])
        query_clips = {f.clip_id for f in faces}
        allowed = accessible_case_ids(s, u)

        def visible(c: Clip) -> bool:
            return allowed is None or c.case_id in allowed

        f2t = {f.id: f.track_id for f in s.exec(select(FaceRow).where(FaceRow.embedding.is_not(None))).all()}
        idx = FaceIndex(paths.index_dir())
        res = idx.search(q, f2t)
        bands_cfg = dict(cfg.search.bands)
        bands, calibrated = effective_bands(bands_cfg)
        sr = Search(case_id=body.case_id, query_face_ids_json=body.face_ids, bands_json=bands_cfg, created_by=u.id)
        s.add(sr)
        s.flush()
        n_hits = 0
        for tid, r in sorted(res.items(), key=lambda kv: -kv[1]["score"]):
            tr = s.get(Track, tid)
            if not tr:
                continue
            c = s.get(Clip, tr.clip_id)
            if c.id in query_clips or not visible(c):
                continue
            band = band_of(r["score"], bands)
            if band is None:
                continue
            s.add(SearchHit(search_id=sr.id, track_id=tid, score=float(r["score"]), band=band,
                            parts_json={"template_sim": r.get("template_sim"), "top3_mean": r.get("top3_mean")},
                            top_face_ids_json=r.get("top_face_ids", [])))
            n_hits += 1
        nc = []
        templ_tracks = {t.track_id for t in s.exec(select(Template)).all()}
        min_vis = float(cfg.search.min_visible_frac)
        for tr in s.exec(select(Track).where((Track.dense_status == "done") | Track.stats_json.is_not(None))).all():
            if not tr.stats_json:  # never processed by a dense/index pass
                continue
            c = s.get(Clip, tr.clip_id)
            if tr.id in templ_tracks or c.id in query_clips or not visible(c):
                continue
            best = s.exec(select(FaceRow).where(FaceRow.track_id == tr.id).order_by(FaceRow.total.desc())).first()
            if tr.stats_json.get("skipped"):
                reason = "person too far away (face too small)"
            elif not best:
                reason = "no face detected"
            elif "FULL_COVER" in best.flags_json:
                reason = "face covered except the eyes"
            elif best.grade == "C":
                reason = "face quality too poor (grade C)"
            elif float(best.raw_json.get("visible_frac", 1.0)) < min_vis:
                reason = "face partly covered"
            else:
                reason = "no searchable face"
            nc.append({"track_id": tr.id, "track_no": tr.track_no, "clip_id": c.id, "clip_name": c.orig_name,
                       "reason": f"not comparable: {reason}", "face": _face_brief(best) if best else None})
        sr.not_comparable_json = nc
        audit.append(s, "search_executed", user_id=u.id, role=u.role, object_type="search", object_id=sr.id,
                     details={"case_id": body.case_id, "query_face_ids": body.face_ids, "hits": n_hits,
                              "not_comparable": len(nc), "bands": bands, "calibrated": calibrated})
        s.flush()
        return _search_out(s, sr)


@router.get("/search/{search_id}", response_model=S.SearchOut)
def get_search(search_id: int, u: User = Depends(viewer), s: Session = Depends(get_db)):
    sr = s.get(Search, search_id)
    if not sr:
        raise not_found("search")
    case_or_404(s, u, sr.case_id)
    return _search_out(s, sr)


@router.get("/cases/{case_id}/searches", response_model=list[S.SearchOut])
def case_searches(case_id: str, u: User = Depends(viewer), s: Session = Depends(get_db)):
    case_or_404(s, u, case_id)
    return [_search_out(s, sr) for sr in s.exec(select(Search).where(Search.case_id == case_id).order_by(Search.id.desc())).all()]


@router.post("/search_hits/{hit_id}/review", response_model=S.HitOut)
def review(hit_id: int, body: S.ReviewIn, u: User = Depends(investigator)):
    with write_session() as s:
        h = s.get(SearchHit, hit_id)
        if not h:
            raise not_found("search hit")
        sr = s.get(Search, h.search_id)
        case_or_404(s, u, sr.case_id)
        prev = h.review
        h.review, h.reviewed_by, h.reviewed_at, h.note = body.decision, u.id, now(), body.note
        audit.append(s, f"match_{body.decision}", user_id=u.id, role=u.role, object_type="search_hit", object_id=h.id,
                     details={"search_id": sr.id, "track_id": h.track_id, "score": h.score, "band": h.band,
                              "previous": prev, "note": body.note})
        return _hit_out(s, h)
