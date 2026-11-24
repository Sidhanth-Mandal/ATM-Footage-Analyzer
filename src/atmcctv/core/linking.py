"""Transaction linking: rank person tracks by how likely they performed the transaction."""
from __future__ import annotations

import math

import numpy as np

from .types import Candidate, PersonTrack, TrackSample

HEAD_IDX = (0, 1, 2, 3, 4)
WRISTS = (9, 10)
TORSO = (5, 6, 11, 12)
ROI_HANDS = ("keypad", "cash_slot", "card_slot")


def point_in_polygon(x: float, y: float, poly: list[list[float]] | None) -> bool:
    """Ray casting; poly and point in the same (normalised) coordinates."""
    if not poly or len(poly) < 3:
        return False
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            xin = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x < xin:
                inside = not inside
        j = i
    return inside


def _torso_centre(s: TrackSample) -> tuple[float, float] | None:
    k = s.kpts
    pts = [(k[i, 0], k[i, 1]) for i in TORSO if k[i, 2] >= 0.3]
    if not pts:
        return None
    a = np.asarray(pts)
    return float(a[:, 0].mean()), float(a[:, 1].mean())


def near(s: TrackSample, camera_type: str, W: int, H: int, rois: dict, cfg) -> bool:
    bh = float(s.bbox[3] - s.bbox[1])
    if camera_type == "fascia":
        if bh >= float(cfg.linking.near_h_frac) * H:
            return True
        if s.face_iod >= float(cfg.linking.near_iod_px):
            return True
        if s.touches_top and bh >= 0.5 * H:
            return True
        return False
    c = _torso_centre(s)
    if c is None:
        c = ((s.bbox[0] + s.bbox[2]) / 2, s.bbox[1] + 0.45 * bh)
    return point_in_polygon(c[0] / W, c[1] / H, rois.get("atm_zone"))


def hand(s: TrackSample, W: int, H: int, rois: dict) -> bool:
    for wi in WRISTS:
        x, y, c = s.kpts[wi]
        if c < 0.3:
            continue
        for name in ROI_HANDS:
            if point_in_polygon(float(x) / W, float(y) / H, rois.get(name)):
                return True
    return False


def score_tracks(tracks: list[PersonTrack], t_txn: float, before_s: float, after_s: float, rois: dict,
                 camera_type: str, W: int, H: int, cfg) -> list[Candidate]:
    w0, w1 = t_txn - before_s, t_txn + after_s
    wlen = max(1e-6, w1 - w0)
    weights = dict(cfg.linking.weights[camera_type])
    tau = float(cfg.linking.tau_s)
    frame_area = float(W * H)
    cands: list[Candidate] = []
    for t in tracks:
        ov = max(0.0, min(t.t_end, w1) - max(t.t_start, w0)) / wlen
        ws = [s for s in t.samples if w0 <= s.pts <= w1]
        if not ws:
            continue
        nf = [near(s, camera_type, W, H, rois, cfg) for s in ws]
        dwell = sum(nf) / len(ws)
        if dwell == 0:
            continue  # gate: never near the machine (street pedestrians, guard through the doorway)
        hd = (sum(hand(s, W, H, rois) for s in ws) / len(ws)) if camera_type == "lobby" else 0.0
        areas = [float((s.bbox[2] - s.bbox[0]) * (s.bbox[3] - s.bbox[1])) / frame_area for s in ws]
        prox = min(1.0, max(areas) / float(cfg.linking.near_area_ref))
        near_s = [s for s, f in zip(ws, nf) if f]
        dmin = min(abs(s.pts - t_txn) for s in near_s)
        clos = math.exp(-dmin / tau)
        parts = {"overlap": ov, "dwell": dwell, "proximity": prox, "closeness": clos}
        if camera_type == "lobby":
            parts["hand"] = hd
        score = sum(weights.get(k, 0.0) * v for k, v in parts.items())
        best = max(near_s, key=lambda s: float((s.bbox[2] - s.bbox[0]) * (s.bbox[3] - s.bbox[1])))
        reasons = [f"near the machine in {dwell:.0%} of window samples",
                   f"closest near-field moment {dmin:.1f} s from the transaction"]
        if camera_type == "lobby":
            reasons.append(f"wrist in keypad/cash/card slot in {hd:.0%} of window samples")
        if t.source == "face":
            reasons.append("face-only track (no person box)")
        cands.append(Candidate(t.track_id, float(score), {k: float(v) for k, v in parts.items()}, float(best.pts), reasons))
    cands.sort(key=lambda c: -c.score)
    top_k = int(cfg.linking.top_k)
    eps = float(cfg.linking.tie_eps)
    if len(cands) >= 2 and cands[0].score - cands[1].score <= eps:
        for c in cands:
            if cands[0].score - c.score <= eps:
                c.reasons.insert(0, "ambiguous: score within %.2f of the top candidate" % eps)
    return cands[:top_k]
