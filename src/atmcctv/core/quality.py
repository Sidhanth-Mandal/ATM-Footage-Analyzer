"""Per-face quality: learned FIQA + interpretable sub-scores, caps and grade.

All measures use original pixels (aligned crop / face crop). Normalisations use
fixed bounds so scores are comparable across clips.
"""
from __future__ import annotations

import logging
import math
from typing import Callable

import cv2
import numpy as np

from . import flags as flagmod
from .types import FaceObs, QualityResult, ScoredFace

log = logging.getLogger(__name__)

# CelebAMask-HQ 19-class order (verified against uniface.draw.FACE_PARSING_LABELS)
SKIN, L_BROW, R_BROW, L_EYE, R_EYE, EYE_G, L_EAR, R_EAR, EAR_R, NOSE, MOUTH, U_LIP, L_LIP, NECK, NECK_L, CLOTH, HAIR, HAT = range(1, 19)
VISIBLE = (SKIN, L_BROW, R_BROW, L_EYE, R_EYE, NOSE, MOUTH, U_LIP, L_LIP)

DEFAULT_FIQA_BOUNDS = (0.25, 0.80)   # replaced by calibrate.py output when present
PARSE_UNRELIABLE_HAT = 0.6
UNKNOWN_VISIBLE = 0.5                # neutral occlusion value when parsing is unreliable


def clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


# ------------------------------------------------------------------ pure maths

def sub_scores(raw: dict, cfg) -> dict[str, float]:
    lo, hi = cfg.quality.fiqa_bounds if cfg.quality.fiqa_bounds and cfg.quality.fiqa_bounds[0] is not None \
        else DEFAULT_FIQA_BOUNDS
    s = {}
    s["fiqa"] = clip01((raw["fiqa_raw"] - lo) / max(1e-9, hi - lo))
    v = max(raw["lap_var"], 1e-6)
    s["sharpness"] = clip01((math.log10(v) - math.log10(20)) / (math.log10(500) - math.log10(20)))
    s["pose"] = clip01(1 - math.sqrt((raw["yaw"] / 60) ** 2 + (raw["pitch"] / 40) ** 2))
    mu, c, r = raw["mean_luma"], raw["clip_frac"], raw["lr_ratio"]
    e = 1 - clip01((abs(mu - 0.5) - 0.25) / 0.25)
    k = 1 - clip01(c / 0.2)
    y = 1 - clip01((r - 1.2) / 0.8)
    s["illumination"] = float((e * k * y) ** (1 / 3)) if e * k * y > 0 else 0.0
    s["occlusion"] = clip01((raw["visible_frac"] - 0.3) / 0.6)
    s["resolution"] = clip01((raw["iod_px"] - 15) / 45)
    return s


def grade_of(total: float, cfg) -> str:
    if total >= float(cfg.quality.grade_a):
        return "A"
    if total >= float(cfg.quality.grade_b):
        return "B"
    return "C"


def finalize(raw: dict, flags: list[str], cfg, *, touches_edge: bool, head_out_of_frame: bool) -> QualityResult:
    sub = sub_scores(raw, cfg)
    w = dict(cfg.quality.weights)
    total = 100.0 * sum(float(w[k]) * sub[k] for k in w)
    caps = []
    cap_val = float(cfg.quality.grade_b) - 1.0  # 54 with default grade_b=55
    if raw["iod_px"] < float(cfg.quality.iod_min_px):
        caps.append(f"IOD < {cfg.quality.iod_min_px} px")
    if touches_edge:
        caps.append("face box touches frame edge")
    if head_out_of_frame:
        caps.append("head out of frame")
    if raw["visible_frac"] < 0.35:
        caps.append("visible face fraction < 0.35")
    if caps:
        total = min(total, cap_val)
    total = round(total, 2)
    grade = grade_of(total, cfg)
    searchable = (grade != "C" and raw["visible_frac"] >= float(cfg.search.min_visible_frac)
                  and "FULL_COVER" not in flags)
    return QualityResult(total=total, grade=grade, sub={k: round(v, 4) for k, v in sub.items()},
                         raw={k: (round(float(v), 4) if isinstance(v, (int, float)) else v) for k, v in raw.items()},
                         flags=list(flags), caps=caps, searchable=bool(searchable))


# ------------------------------------------------------------------ measurements

def laplacian_var(aligned: np.ndarray) -> float:
    g = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    m = int(0.1 * h)
    inner = g[m:h - m, m:w - m]
    return float(cv2.Laplacian(inner, cv2.CV_64F).var())


def face_region_polygon(lmk: np.ndarray) -> np.ndarray:
    """Approximate brow-to-chin face region from 5 landmarks (same coordinates as lmk)."""
    le, re, nose, lm, rm = [np.asarray(p, np.float64) for p in lmk]
    eye_c = (le + re) / 2
    mouth_c = (lm + rm) / 2
    v = mouth_c - eye_c
    d = max(1.0, float(np.linalg.norm(v)))
    up = -v / d
    iod_vec = re - le
    iod = max(1.0, float(np.linalg.norm(iod_vec)))
    side = iod_vec / iod
    pts = [
        le + up * 0.55 * d - side * 0.35 * iod,
        re + up * 0.55 * d + side * 0.35 * iod,
        re + side * 0.45 * iod,
        rm + side * 0.25 * iod,
        mouth_c - up * 0.55 * d + side * 0.15 * iod,
        mouth_c - up * 0.55 * d - side * 0.15 * iod,
        lm - side * 0.25 * iod,
        le - side * 0.45 * iod,
    ]
    return np.asarray(pts, np.float32)


def parse_measures(mask: np.ndarray, crop: np.ndarray, lmk_crop: np.ndarray) -> dict:
    """mask: class ids for crop; lmk_crop: landmarks in crop coordinates."""
    poly = face_region_polygon(lmk_crop)
    hull = cv2.convexHull(poly.astype(np.int32))
    region = np.zeros(mask.shape, np.uint8)
    cv2.fillConvexPoly(region, hull, 1)
    region = region.astype(bool)
    area = max(1, int(region.sum()))
    m = mask[region]

    def frac(classes):
        return float(np.isin(m, classes).sum()) / area

    eye_y = float((lmk_crop[0][1] + lmk_crop[1][1]) / 2)
    mouth_y = float((lmk_crop[3][1] + lmk_crop[4][1]) / 2)
    brow_y = int(max(0, eye_y - 0.25 * max(1.0, mouth_y - eye_y)))
    above = mask[:brow_y] if brow_y > 0 else mask[:1]
    hat_above = float((above == HAT).sum()) / max(1, above.size)

    luma = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    skin = (mask == SKIN) & region
    if skin.sum() < 30:
        skin = region
    vals = luma[skin]
    mu = float(vals.mean()) if vals.size else 0.0
    clip_frac = float(((vals < 0.03) | (vals > 0.97)).mean()) if vals.size else 1.0
    cx = float((lmk_crop[0][0] + lmk_crop[1][0]) / 2)
    xs = np.arange(mask.shape[1])[None, :].repeat(mask.shape[0], 0)
    left, right = luma[skin & (xs < cx)], luma[skin & (xs >= cx)]
    if left.size > 10 and right.size > 10:
        ml, mr = float(left.mean()), float(right.mean())
        lr = max(ml, mr) / max(1e-3, min(ml, mr))
    else:
        lr = 2.0
    hat_in_region = frac((HAT,))
    return {
        # BiSeNet's failure mode on blurred / silhouetted faces is to label the whole face "hat".
        # A hat covering the eyes-nose-mouth hull is implausible, so the parse is treated as unreliable.
        "parse_unreliable": hat_in_region > PARSE_UNRELIABLE_HAT,
        "hat_in_region": hat_in_region,
        "visible_frac": frac(VISIBLE),
        "lower_face_frac": frac((NOSE, MOUTH, U_LIP, L_LIP)),
        "eyes_frac": frac((L_EYE, R_EYE)),
        "cover_frac": frac((HAT, HAIR, CLOTH, NECK, NECK_L)),
        "eyeglass_frac": frac((EYE_G,)),
        "hat_above_brow": hat_above,
        "mean_luma": mu,
        "clip_frac": clip_frac,
        "lr_ratio": lr,
    }


def _fiqa_batch(registry, aligned: list[np.ndarray], batch: int) -> list[float]:
    m = registry.fiqa()
    blobs = [m.preprocess(a) for a in aligned]
    outs = registry.run_batched(m.session, m.input_name, [m.output_name], blobs, batch)
    return [float(np.squeeze(o)) for o in outs]


def _parse_batch(registry, crops: list[np.ndarray], batch: int) -> list[np.ndarray]:
    m = registry.parser()
    blobs = [m.preprocess(c) for c in crops]
    outs = registry.run_batched(m.session, m.input_name, m.output_names, blobs, min(batch, 8))
    masks = []
    for o, c in zip(outs, crops):
        pm = o.argmax(0).astype(np.uint8)
        masks.append(cv2.resize(pm, (c.shape[1], c.shape[0]), interpolation=cv2.INTER_NEAREST))
    return masks


def _attrib_batch(registry, crops: list[np.ndarray], batch: int) -> list[dict]:
    m = registry.attrib()
    blobs = [m.preprocess(c) for c in crops]
    outs = registry.run_batched(m.session, m.input_name, m.output_names, blobs, batch)
    res = []
    for o in outs:
        le, ri, eg, mask, sg = np.asarray(o, float).ravel().tolist()[:5]
        res.append({"left_eye_open": le, "right_eye_open": ri, "eyeglasses": eg, "mask": mask, "sunglasses": sg})
    return res


def _pose(registry, crops: list[np.ndarray]) -> list[tuple[float, float, float]]:
    m = registry.head_pose()
    out = []
    for c in crops:
        r = m.estimate(c)
        out.append((float(r.yaw), float(r.pitch), float(r.roll)))
    return out


def score_faces(obs: list[FaceObs], registry, cfg, progress: Callable[[float, str], None] | None = None
                ) -> list[ScoredFace]:
    """Measure, flag and grade every observation. YOLOE headwear checks run on the top-N afterwards."""
    if not obs:
        return []
    batch = int(cfg.dense.batch)
    out: list[ScoredFace] = []
    for i0 in range(0, len(obs), 256):
        chunk = obs[i0:i0 + 256]
        crops = [o.face_crop for o in chunk]
        fq = _fiqa_batch(registry, [o.aligned for o in chunk], batch)
        masks = _parse_batch(registry, crops, batch)
        attrs = _attrib_batch(registry, crops, batch)
        poses = _pose(registry, crops)
        for o, q, mk, at, (yaw, pitch, roll) in zip(chunk, fq, masks, attrs, poses):
            lmk_crop = o.lmk5 - np.array(o.face_crop_origin, np.float32)
            pm = parse_measures(mk, o.face_crop, lmk_crop)
            iod = float(np.linalg.norm(o.lmk5[1] - o.lmk5[0]))
            vis = UNKNOWN_VISIBLE if pm["parse_unreliable"] else pm["visible_frac"]
            raw = {"yaw": yaw, "pitch": pitch, "roll": roll, "iod_px": iod, "lap_var": laplacian_var(o.aligned),
                   "mean_luma": pm["mean_luma"], "clip_frac": pm["clip_frac"], "lr_ratio": pm["lr_ratio"],
                   "visible_frac": vis, "fiqa_raw": q, "parse_unreliable": float(pm["parse_unreliable"])}
            fl = flagmod.basic_flags(raw, pm, at, o, cfg)
            qr = finalize(raw, fl, cfg, touches_edge=o.touches_edge, head_out_of_frame=o.head_out_of_frame)
            out.append(ScoredFace(o, qr, at, pm))
        if progress:
            progress(min(0.99, (i0 + len(chunk)) / len(obs)), f"scored {i0 + len(chunk)}/{len(obs)} faces")
    if bool(cfg.flags.use_yoloe):
        try:
            flagmod.yoloe_flags(out, registry, cfg)
        except Exception as e:  # open-vocab checks are advisory; never fail the job for them
            log.warning("YOLOE headwear check skipped: %s", e)
    # re-finalise after YOLOE flags (searchable depends on flags)
    for sf in out:
        sf.q = finalize(sf.q.raw, sf.q.flags, cfg, touches_edge=sf.obs.touches_edge,
                        head_out_of_frame=sf.obs.head_out_of_frame)
    return out
