"""Coarse pass: person detection + pose + BoT-SORT at `coarse.sample_fps`.

Fascia mode additionally runs SCRFD on every sampled frame and builds face
tracks (UniFace BYTETracker), so a close face with no person box can still be
a candidate.
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from ..settings import ROOT
from .detview import detview
from .types import PersonTrack, TrackSample

log = logging.getLogger(__name__)

FACE_TRACK_OFFSET = 10000
HEAD_KPTS = (0, 1, 2, 3, 4)  # nose, eyes, ears (COCO)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def _tracker_yaml(cfg) -> str:
    with open(ROOT / "config" / "botsort.yaml", encoding="utf-8") as f:
        tc = yaml.safe_load(f)
    tc["track_buffer"] = max(1, round(float(cfg.coarse.lost_s) * float(cfg.coarse.sample_fps)))
    tmp = Path(tempfile.gettempdir()) / f"atmcctv_botsort_{tc['track_buffer']}.yaml"
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(tc, f)
    return str(tmp)


def iod_from_lmk(lmk: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(lmk[1]) - np.asarray(lmk[0])))


def coarse_pass(src, registry, cfg, camera_type: str = "fascia",
                progress: Callable[[float, str], None] | None = None) -> tuple[list[PersonTrack], dict]:
    model = registry.person_pose()
    tracker = _tracker_yaml(cfg)
    fps = float(cfg.coarse.sample_fps)
    total = max(1, int(src.duration * fps))
    tracks: dict[int, PersonTrack] = {}
    H, W = src.height, src.width
    fascia = camera_type == "fascia"
    face_dets_per_sample: list[tuple[float, np.ndarray, np.ndarray]] = []  # (pts, boxes(N,5), lmks(N,5,2))
    face_tracker = None
    face_tracks: dict[int, PersonTrack] = {}
    if fascia:
        from uniface import BYTETracker
        face_tracker = BYTETracker(track_thresh=float(cfg.coarse.face_track_conf),
                                   track_buffer=max(1, round(float(cfg.coarse.lost_s) * fps)), match_thresh=0.8)
        face_tracker.reset()
        fdet = registry.face_det()

    n = 0
    first = True
    for pts, _idx, frame in src.iter_sampled(fps):
        H, W = frame.shape[:2]
        view, _ = detview(frame, cfg)
        res = model.track(view, persist=not first, tracker=tracker, classes=[0], conf=float(cfg.coarse.person_conf),
                          quantize=registry.precision, verbose=False, imgsz=int(cfg.coarse.imgsz), device=registry.device)
        first = False
        r = res[0]
        boxes = r.boxes
        if boxes is not None and boxes.id is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int)
            kp = r.keypoints.data.cpu().numpy() if r.keypoints is not None else np.zeros((len(ids), 17, 3))
            for b, c, tid, k in zip(xyxy, confs, ids, kp):
                t = tracks.setdefault(int(tid), PersonTrack(int(tid), "person"))
                t.samples.append(TrackSample(pts, b.astype(np.float32), k.astype(np.float32), float(c),
                                             bool(b[1] <= 2)))
        if fascia:
            faces = fdet.detect(view)
            fb = np.array([[*f.bbox[:4], f.confidence] for f in faces], np.float32).reshape(-1, 5)
            fl = np.array([f.landmarks for f in faces], np.float32).reshape(-1, 5, 2)
            face_dets_per_sample.append((pts, fb, fl))
            out = face_tracker.update(fb.astype(np.float64)) if len(fb) else face_tracker.update(np.empty((0, 5)))
            for row in out:
                box, ftid = row[:4], int(row[4])
                j = int(np.argmax([iou(box, d[:4]) for d in fb])) if len(fb) else -1
                if j < 0 or iou(box, fb[j, :4]) < 0.3:
                    continue
                d = fb[j]
                ft = face_tracks.setdefault(FACE_TRACK_OFFSET + ftid, PersonTrack(FACE_TRACK_OFFSET + ftid, "face"))
                ft.samples.append(TrackSample(pts, d[:4].copy(), np.zeros((17, 3), np.float32), float(d[4]),
                                              bool(d[1] <= 2), face_bbox=d[:4].copy(), face_iod=iod_from_lmk(fl[j])))
        n += 1
        if progress and n % 5 == 0:
            progress(min(0.99, n / total), f"coarse pass {n}/{total} sampled frames")

    person_tracks = list(tracks.values())
    if fascia:
        person_tracks = _link_faces(person_tracks, list(face_tracks.values()))
    person_tracks = merge_fragments(person_tracks, float(cfg.coarse.merge_gap_s))
    person_tracks = [t for t in person_tracks if t.duration >= float(cfg.coarse.min_track_s) or
                     (len(t.samples) >= 3 and t.source == "face")]
    person_tracks.sort(key=lambda t: t.t_start)
    meta = {"width": W, "height": H, "n_sampled": n, "sample_fps": fps, "camera_type": camera_type}
    return person_tracks, meta


def _link_faces(persons: list[PersonTrack], faces: list[PersonTrack]) -> list[PersonTrack]:
    """Attach face boxes to person samples; unlinked face tracks become their own candidates."""
    by_pts = {}
    for p in persons:
        for s in p.samples:
            by_pts.setdefault(round(s.pts, 3), []).append((p, s))
    out = list(persons)
    for ft in faces:
        votes: dict[int, int] = {}
        shared = 0
        for fs in ft.samples:
            cands = by_pts.get(round(fs.pts, 3), [])
            if not cands:
                continue
            shared += 1
            cx, cy = (fs.bbox[0] + fs.bbox[2]) / 2, (fs.bbox[1] + fs.bbox[3]) / 2
            for p, s in cands:
                b = s.bbox
                if b[0] <= cx <= b[2] and b[1] <= cy <= b[3]:
                    votes[p.track_id] = votes.get(p.track_id, 0) + 1
        best = max(votes.items(), key=lambda kv: kv[1]) if votes else None
        if best and shared and best[1] / shared >= 0.5:
            p = next(pp for pp in persons if pp.track_id == best[0])
            fmap = {round(fs.pts, 3): fs for fs in ft.samples}
            for s in p.samples:
                fs = fmap.get(round(s.pts, 3))
                if fs is not None:
                    s.face_bbox, s.face_iod = fs.face_bbox, fs.face_iod
        else:
            out.append(ft)
    return out


def merge_fragments(tracks: list[PersonTrack], gap_s: float) -> list[PersonTrack]:
    tracks = sorted(tracks, key=lambda t: t.t_start)
    merged: list[PersonTrack] = []
    used = set()
    for i, t in enumerate(tracks):
        if i in used:
            continue
        cur = PersonTrack(t.track_id, t.source, list(t.samples))
        changed = True
        while changed:
            changed = False
            for j in range(len(tracks)):
                if j in used or j == i:
                    continue
                o = tracks[j]
                if o.source != cur.source:
                    continue
                gap = o.t_start - cur.t_end
                if not (0 <= gap <= gap_s):
                    continue
                a, b = cur.samples[-1].bbox, o.samples[0].bbox
                ha, hb = a[3] - a[1], b[3] - b[1]
                if iou(a, b) >= 0.3 and abs(ha - hb) / max(ha, hb, 1e-6) < 0.3:
                    cur.samples.extend(o.samples)
                    used.add(j)
                    changed = True
        used.add(i)
        merged.append(cur)
    return merged


# ------------------------------------------------------------------ persistence

def save_tracks(tracks: list[PersonTrack], meta: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    index = {"meta": meta, "tracks": []}
    for t in tracks:
        k = f"t{t.track_id}"
        arrays[f"{k}_pts"] = np.array([s.pts for s in t.samples], np.float64)
        arrays[f"{k}_bbox"] = np.stack([s.bbox for s in t.samples]).astype(np.float32)
        arrays[f"{k}_kpts"] = np.stack([s.kpts for s in t.samples]).astype(np.float32)
        arrays[f"{k}_conf"] = np.array([s.conf for s in t.samples], np.float32)
        arrays[f"{k}_face"] = np.stack([s.face_bbox if s.face_bbox is not None else np.full(4, np.nan, np.float32)
                                        for s in t.samples]).astype(np.float32)
        arrays[f"{k}_iod"] = np.array([s.face_iod for s in t.samples], np.float32)
        index["tracks"].append({"track_id": t.track_id, "source": t.source, "t_start": t.t_start, "t_end": t.t_end,
                                "n": len(t.samples)})
    npz = out_dir / "tracks.npz"
    js = out_dir / "tracks.json"
    np.savez_compressed(npz, **arrays)
    js.write_text(json.dumps(index, indent=1), encoding="utf-8")
    return npz, js


def load_tracks(out_dir: Path) -> tuple[list[PersonTrack], dict]:
    index = json.loads((out_dir / "tracks.json").read_text(encoding="utf-8"))
    data = np.load(out_dir / "tracks.npz")
    tracks = []
    for ti in index["tracks"]:
        k = f"t{ti['track_id']}"
        pts, bbox, kpts, conf = data[f"{k}_pts"], data[f"{k}_bbox"], data[f"{k}_kpts"], data[f"{k}_conf"]
        face, iodv = data[f"{k}_face"], data[f"{k}_iod"]
        samples = [TrackSample(float(pts[i]), bbox[i], kpts[i], float(conf[i]), bool(bbox[i][1] <= 2),
                               None if np.isnan(face[i]).any() else face[i], float(iodv[i])) for i in range(len(pts))]
        tracks.append(PersonTrack(int(ti["track_id"]), ti["source"], samples))
    return tracks, index["meta"]
