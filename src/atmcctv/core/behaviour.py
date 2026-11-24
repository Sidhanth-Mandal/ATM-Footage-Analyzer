"""Experimental, rule-based behaviour checks: shoulder surfing, card-slot/keypad tampering, loitering."""
from __future__ import annotations

import numpy as np

from .linking import near, point_in_polygon
from .types import PersonTrack


def _intervals(pts: list[float], max_gap: float) -> list[tuple[float, float]]:
    if not pts:
        return []
    pts = sorted(pts)
    out = [[pts[0], pts[0]]]
    for p in pts[1:]:
        if p - out[-1][1] <= max_gap:
            out[-1][1] = p
        else:
            out.append([p, p])
    return [(a, b) for a, b in out]


def _attention_towards(s, target_cx: float) -> bool:
    k = s.kpts
    if k[0, 2] < 0.3 or k[5, 2] < 0.3 or k[6, 2] < 0.3:
        return False
    mid = (k[5, 0] + k[6, 0]) / 2
    sw = abs(k[5, 0] - k[6, 0]) + 1e-6
    off = (k[0, 0] - mid) / sw
    return bool(np.sign(off) == np.sign(target_cx - mid) and abs(off) > 0.15)


def shoulder_surfing(customer: PersonTrack, others: list[PersonTrack], rois: dict, camera_type: str,
                     W: int, H: int, cfg, sample_fps: float) -> list[dict]:
    gap = 2.5 / sample_fps
    cust_near = [s.pts for s in customer.samples if near(s, camera_type, W, H, rois, cfg)]
    cust_iv = _intervals(cust_near, gap)
    by_pts = {round(s.pts, 3): s for s in customer.samples}
    events = []
    for o in others:
        if o.track_id == customer.track_id:
            continue
        hits, attn = [], 0
        for s in o.samples:
            if not any(a <= s.pts <= b for a, b in cust_iv):
                continue
            close = near(s, camera_type, W, H, rois, cfg) if camera_type == "fascia" else point_in_polygon(
                float((s.bbox[0] + s.bbox[2]) / 2 / W), float((s.bbox[1] + s.bbox[3]) / 2 / H), rois.get("atm_zone"))
            if not close:
                continue
            hits.append(s.pts)
            cs = by_pts.get(round(s.pts, 3))
            if cs is not None and _attention_towards(s, float((cs.bbox[0] + cs.bbox[2]) / 2)):
                attn += 1
        for a, b in _intervals(hits, gap):
            dur = b - a + 1.0 / sample_fps
            if dur < float(cfg.behaviour.ss_min_s):
                continue
            inside = [s for s in o.samples if a <= s.pts <= b]
            best = max(inside, key=lambda s: float((s.bbox[2] - s.bbox[0]) * (s.bbox[3] - s.bbox[1])))
            events.append({"type": "shoulder_surfing", "t_start": a, "t_end": b, "track_ids": [customer.track_id, o.track_id],
                           "details": {"duration_s": round(dur, 2), "other_track": o.track_id,
                                       "attention_frac": round(attn / max(1, len(hits)), 2),
                                       "best_view_pts": best.pts, "experimental": True}})
    return events


def _roi_rect(poly, W, H):
    a = np.asarray(poly, np.float32)
    x0, y0 = int(max(0, a[:, 0].min() * W)), int(max(0, a[:, 1].min() * H))
    x1, y1 = int(min(W, a[:, 0].max() * W)), int(min(H, a[:, 1].max() * H))
    return x0, y0, x1, y1


def tampering(src, tracks: list[PersonTrack], rois: dict, camera_type: str, cfg, progress=None) -> list[dict]:
    """SSIM of card-slot/keypad ROIs in empty-scene frames against a median reference."""
    from skimage.metrics import structural_similarity as ssim
    import cv2

    names = [n for n in ("card_slot", "keypad") if rois.get(n)]
    if not names:
        return []
    fps = float(cfg.behaviour.sample_fps)
    W, H = src.width, src.height
    occupied = []
    for t in tracks:
        for s in t.samples:
            if near(s, camera_type, W, H, rois, cfg) or (camera_type == "fascia" and
                                                         (s.bbox[3] - s.bbox[1]) > 0.25 * H):
                occupied.append(s.pts)
    occupied = np.array(sorted(occupied))
    tol = 1.0 / float(cfg.coarse.sample_fps) + 0.3

    def is_empty(p):
        if not len(occupied):
            return True
        i = np.searchsorted(occupied, p)
        near_d = min(abs(occupied[min(i, len(occupied) - 1)] - p), abs(occupied[max(i - 1, 0)] - p))
        return near_d > tol

    ref_n = int(cfg.behaviour.tamper_ref_n)
    refs: dict[str, list[np.ndarray]] = {n: [] for n in names}
    reference: dict[str, np.ndarray] = {}
    series: dict[str, list[tuple[float, float]]] = {n: [] for n in names}
    for pts, _i, frame in src.iter_sampled(fps):
        if not is_empty(pts):
            continue
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for n in names:
            x0, y0, x1, y1 = _roi_rect(rois[n], frame.shape[1], frame.shape[0])
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            roi = g[y0:y1, x0:x1]
            if n not in reference:
                refs[n].append(roi)
                if len(refs[n]) >= ref_n:
                    reference[n] = np.median(np.stack(refs[n]), axis=0).astype(np.uint8)
                continue
            series[n].append((pts, float(ssim(reference[n], roi, data_range=255))))
    for n in names:  # short clips: build the reference from whatever empty frames exist
        if n not in reference and len(refs[n]) >= 3:
            reference[n] = np.median(np.stack(refs[n]), axis=0).astype(np.uint8)
    events = []
    thr = float(cfg.behaviour.tamper_ssim)
    persist = float(cfg.behaviour.tamper_persist_s)
    for n in names:
        low = [p for p, v in series[n] if v < thr]
        for a, b in _intervals(low, 3.0 / fps):
            if b - a < persist:
                continue
            before = [t for t in tracks if t.t_end <= a + 1.0]
            last = max(before, key=lambda t: t.t_end).track_id if before else None
            vals = [v for p, v in series[n] if a <= p <= b]
            events.append({"type": "tampering", "t_start": a, "t_end": b, "track_ids": [last] if last is not None else [],
                           "details": {"roi": n, "min_ssim": round(min(vals), 3), "threshold": thr,
                                       "experimental": True}})
    return events


def loitering(tracks: list[PersonTrack], txn_pts: list[float], rois: dict, camera_type: str, W: int, H: int,
              cfg, sample_fps: float) -> list[dict]:
    events = []
    gap = 2.5 / sample_fps
    for t in tracks:
        np_ = [s.pts for s in t.samples if near(s, camera_type, W, H, rois, cfg)]
        for a, b in _intervals(np_, gap):
            if b - a < float(cfg.behaviour.tamper_loiter_s):
                continue
            if any(a - 30 <= p <= b + 30 for p in txn_pts):
                continue
            events.append({"type": "loitering", "t_start": a, "t_end": b, "track_ids": [t.track_id],
                           "details": {"duration_s": round(b - a, 1), "experimental": True,
                                       "note": "near the machine with no journal transaction"}})
    return events
