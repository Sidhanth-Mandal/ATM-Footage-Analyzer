"""Dense pass: face detection, assignment and alignment on confirmed (or gallery) tracks.

Detection runs on the detection view of an upper-body crop. Alignment, crops
and all scoring use ORIGINAL pixels.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import cv2
import numpy as np

from .detview import detview, frame_p95_luma
from .persons import iou
from .types import FaceObs, PersonTrack

log = logging.getLogger(__name__)
HEAD_IDX = (0, 1, 2, 3, 4)


@dataclass
class DenseStats:
    frames: int = 0
    frames_with_face: int = 0
    dropped_inconsistent: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def face_visible_frac(self) -> float:
        return self.frames_with_face / self.frames if self.frames else 0.0


class _Interp:
    """Linear interpolation of xyxy boxes and keypoints from coarse samples."""

    def __init__(self, track: PersonTrack, hold_s: float):
        self.t = np.array([s.pts for s in track.samples], np.float64)
        self.b = np.stack([s.bbox for s in track.samples]).astype(np.float64)
        self.k = np.stack([s.kpts for s in track.samples]).astype(np.float64)
        fb = [s.face_bbox if s.face_bbox is not None else s.bbox for s in track.samples]
        self.f = np.stack(fb).astype(np.float64)
        self.hold = hold_s
        self.source = track.source

    def at(self, pts: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, str] | None:
        t = self.t
        if pts < t[0]:
            mode = "hold" if t[0] - pts <= self.hold else "extrapolated"
            return self.b[0], self.k[0], self.f[0], mode
        if pts > t[-1]:
            mode = "hold" if pts - t[-1] <= self.hold else "extrapolated"
            return self.b[-1], self.k[-1], self.f[-1], mode
        i = int(np.searchsorted(t, pts))
        if i < len(t) and abs(t[i] - pts) < 1e-6:
            return self.b[i], self.k[i], self.f[i], "sample"
        i0, i1 = max(0, i - 1), min(len(t) - 1, i)
        if t[i1] - t[i0] > 2.5:  # long gap in the track: treat as hold of the nearer sample
            j = i0 if pts - t[i0] < t[i1] - pts else i1
            return self.b[j], self.k[j], self.f[j], "gap"
        a = (pts - t[i0]) / max(1e-9, t[i1] - t[i0])
        b = (1 - a) * self.b[i0] + a * self.b[i1]
        k = self.k[i0].copy() if a < 0.5 else self.k[i1].copy()
        k[:, :2] = (1 - a) * self.k[i0][:, :2] + a * self.k[i1][:, :2]
        k[:, 2] = np.minimum(self.k[i0][:, 2], self.k[i1][:, 2])
        f = (1 - a) * self.f[i0] + a * self.f[i1]
        return b, k, f, "interp"


def _clip_box(b, W, H):
    x0, y0, x1, y1 = b
    return int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1))


def _expand(b, f):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    w, h = (b[2] - b[0]) * f, (b[3] - b[1]) * f
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


def crop_with_margin(frame: np.ndarray, bbox, margin: float = 0.25, square: bool = False) -> np.ndarray:
    return crop_with_origin(frame, bbox, margin, square)[0]


def crop_with_origin(frame: np.ndarray, bbox, margin: float = 0.25, square: bool = False
                     ) -> tuple[np.ndarray, tuple[int, int]]:
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
    w, h = x1 - x0, y1 - y0
    if square:
        s = max(w, h)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        x0, x1, y0, y1 = cx - s / 2, cx + s / 2, cy - s / 2, cy + s / 2
        w = h = s
    x0, y0 = max(0, int(x0 - margin * w)), max(0, int(y0 - margin * h))
    x1, y1 = min(W, int(x1 + margin * w)), min(H, int(y1 + margin * h))
    return frame[y0:y1, x0:x1].copy(), (x0, y0)


def align(frame: np.ndarray, lmk5: np.ndarray) -> np.ndarray:
    from uniface.face_utils import face_alignment
    aligned, _ = face_alignment(frame, lmk5.astype(np.float32), image_size=112)
    return aligned


def _region_and_anchor(box, kpts, fbox, source: str, mode: str, W: int, H: int, cfg):
    bh = max(1.0, float(box[3] - box[1]))
    if source == "face":
        region = _expand(fbox, 2.0 if mode in ("sample", "interp") else 2.6)
        anchor = ((fbox[0] + fbox[2]) / 2, (fbox[1] + fbox[3]) / 2)
        norm_h = max(1.0, float(fbox[3] - fbox[1])) * 2.5
        head_visible = True
    else:
        top = box.copy()
        top[3] = box[1] + float(cfg.dense.upper_body_frac) * bh
        m = float(cfg.dense.margin)
        w = top[2] - top[0]
        h = top[3] - top[1]
        region = np.array([top[0] - m * w, top[1] - m * h, top[2] + m * w, top[3] + m * h])
        if mode == "extrapolated":
            region = _expand(region, 1.5)
        vis = [(kpts[i, 0], kpts[i, 1]) for i in HEAD_IDX if kpts[i, 2] >= 0.3]
        head_visible = bool(vis)
        if vis:
            a = np.asarray(vis)
            anchor = (float(a[:, 0].mean()), float(a[:, 1].mean()))
        else:
            anchor = ((box[0] + box[2]) / 2, box[1] + 0.12 * bh)
        norm_h = bh
    return region, anchor, norm_h, head_visible


def dense_pass(src, tracks: list[PersonTrack], registry, cfg, t0: float | None = None, t1: float | None = None,
               step: int = 1, progress: Callable[[float, str], None] | None = None
               ) -> dict[int, tuple[list[FaceObs], DenseStats]]:
    """Decode the union span once and process every track whose (padded) span covers the frame."""
    fdet = registry.face_det()
    pad = float(cfg.dense.pad_s)
    spans = {t.track_id: (t.t_start - pad, t.t_end + pad) for t in tracks}
    interps = {t.track_id: _Interp(t, float(cfg.dense.hold_s)) for t in tracks}
    sources = {t.track_id: t.source for t in tracks}
    lo = max(0.0, min(s[0] for s in spans.values())) if t0 is None else t0
    hi = max(s[1] for s in spans.values()) if t1 is None else t1
    out: dict[int, tuple[list[FaceObs], DenseStats]] = {tid: ([], DenseStats()) for tid in spans}
    face_ids: dict[int, list[int]] = {tid: [] for tid in spans}
    last_box: dict[int, np.ndarray | None] = {tid: None for tid in spans}
    next_fid = {tid: 0 for tid in spans}
    max_d = float(cfg.dense.max_anchor_dist)
    total = max(1, int((hi - lo) * src.fps / step))
    n = 0
    for pts, idx, frame in src.iter_all(lo, hi, step=step):
        n += 1
        H, W = frame.shape[:2]
        p95 = None
        for tid, (s0, s1) in spans.items():
            if not (s0 <= pts <= s1):
                continue
            r = interps[tid].at(pts)
            if r is None:
                continue
            box, kpts, fbox, mode = r
            stats = out[tid][1]
            stats.frames += 1
            region, anchor, norm_h, head_visible = _region_and_anchor(box, kpts, fbox, sources[tid], mode, W, H, cfg)
            x0, y0, x1, y1 = _clip_box(region, W, H)
            if x1 - x0 < 16 or y1 - y0 < 16:
                continue
            crop = frame[y0:y1, x0:x1]
            view, corrected = detview(crop, cfg)
            dets = fdet.detect(view)
            if not dets:
                continue
            ebox = _expand(box, 1.2) if sources[tid] == "person" else _expand(fbox, 3.0)
            best, best_d = None, 1e9
            for d in dets:
                b = d.bbox[:4] + np.array([x0, y0, x0, y0])
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                if not (ebox[0] <= cx <= ebox[2] and ebox[1] <= cy <= ebox[3]):
                    continue
                dist = float(np.hypot(cx - anchor[0], cy - anchor[1]) / norm_h)
                if dist < best_d:
                    best, best_d = (d, b), dist
            if best is None or best_d > max_d:
                continue
            d, b = best
            lmk = d.landmarks + np.array([x0, y0])
            # face-level IoU identity for temporal consistency
            lb = last_box[tid]
            if lb is None or iou(lb, b) < 0.2:
                next_fid[tid] += 1
            last_box[tid] = b
            face_ids[tid].append(next_fid[tid])
            if p95 is None:
                p95 = frame_p95_luma(frame)
            touches_top = bool(box[1] <= 2)
            edge = bool(b[0] <= 2 or b[1] <= 2 or b[2] >= W - 2 or b[3] >= H - 2)
            fcrop, forigin = crop_with_origin(frame, b, 0.15, square=True)
            obs = FaceObs(
                track_id=tid, frame_idx=idx, pts=float(pts), bbox=b.astype(np.float32), lmk5=lmk.astype(np.float32),
                det_conf=float(d.confidence), aligned=align(frame, lmk), raw_crop=crop_with_margin(frame, b, 0.25),
                head_out_of_frame=bool(touches_top and not head_visible and sources[tid] == "person"),
                frame_shape=(H, W), touches_edge=edge, detview_corrected=corrected, frame_p95_luma=float(p95),
                face_crop=fcrop, face_crop_origin=forigin, head_crop=crop_with_margin(frame, b, 0.6),
            )
            out[tid][0].append(obs)
            stats.frames_with_face += 1
        if progress and n % 10 == 0:
            progress(min(0.99, n / total), f"dense pass {n}/{total} frames")

    for tid, (obs, stats) in out.items():
        ids = face_ids[tid]
        keep = []
        for i, o in enumerate(obs):
            if 0 < i < len(obs) - 1 and ids[i - 1] == ids[i + 1] != ids[i]:
                stats.dropped_inconsistent += 1
                stats.frames_with_face -= 1
                continue
            keep.append(o)
        out[tid] = (keep, stats)
    return out


def encode_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else b""
