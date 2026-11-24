"""Reliability flags shown on every face (UI and evidence sheet)."""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

ALL_FLAGS = ["MASK", "FULL_COVER", "SUNGLASSES", "CAP_HAT", "HELMET", "HOOD_SCARF", "PARTIAL_FACE",
             "EXTREME_POSE", "LOW_RES", "MOTION_BLUR", "LOW_LIGHT", "BACKLIT", "ENHANCED"]

YOLOE_CLASSES = ["cap", "hat", "motorcycle helmet", "hood", "scarf", "hijab"]
_YOLOE_MAP = {"cap": "CAP_HAT", "hat": "CAP_HAT", "motorcycle helmet": "HELMET", "hood": "HOOD_SCARF",
              "scarf": "HOOD_SCARF", "hijab": "HOOD_SCARF"}

FLAG_HELP = {
    "MASK": "Face mask or lower face covered",
    "FULL_COVER": "Face covered except the eyes (e.g. niqab); not used for search",
    "SUNGLASSES": "Sunglasses",
    "CAP_HAT": "Cap or hat",
    "HELMET": "Helmet",
    "HOOD_SCARF": "Hood, scarf or hijab (hair covering; informational)",
    "PARTIAL_FACE": "Face cut by the frame edge or head out of frame",
    "EXTREME_POSE": "Head turned or tilted strongly",
    "LOW_RES": "Low resolution (inter-ocular distance < 25 px)",
    "MOTION_BLUR": "Blurred",
    "LOW_LIGHT": "Dark face",
    "BACKLIT": "Backlit: dark face against a bright background",
    "ENHANCED": "Enhanced image, not original pixels",
}


def basic_flags(raw: dict, pm: dict, attrib: dict, obs, cfg) -> list[str]:
    from .quality import sub_scores
    f: list[str] = []
    # Parsed eyes are required for the parsing-based rules: in a backlit silhouette nothing is parsed at all,
    # and "no lips/nose visible" must not be read as "mask".
    reliable = not pm.get("parse_unreliable", False)
    eyes_parsed = reliable and pm.get("eyes_frac", 0.0) > 0.003
    lower_hidden = reliable and pm.get("lower_face_frac", 1.0) < 0.02
    # FaceAttribNet saturates on dark, low-resolution crops. When a reliable parse clearly shows the
    # nose/mouth/lips (>= 5 % of the face region), that evidence overrides the classifier.
    lower_visible = reliable and pm.get("lower_face_frac", 0.0) >= 0.05
    mask = (attrib.get("mask", 0.0) >= 0.5 and not lower_visible) or (lower_hidden and eyes_parsed)
    if mask:
        f.append("MASK")
        if eyes_parsed and lower_hidden and pm.get("cover_frac", 0.0) > 0.6:
            f.append("FULL_COVER")
    if attrib.get("sunglasses", 0.0) >= 0.5:
        f.append("SUNGLASSES")
    if reliable and pm.get("hat_above_brow", 0.0) >= 0.08:
        f.append("CAP_HAT")
    if obs.touches_edge or obs.head_out_of_frame:
        f.append("PARTIAL_FACE")
    if abs(raw["yaw"]) > float(cfg.flags.extreme_yaw) or abs(raw["pitch"]) > float(cfg.flags.extreme_pitch):
        f.append("EXTREME_POSE")
    if raw["iod_px"] < float(cfg.flags.low_res_iod):
        f.append("LOW_RES")
    sub = sub_scores(raw, cfg)
    if sub["sharpness"] < 0.25:
        f.append("MOTION_BLUR")
    if raw["mean_luma"] < 0.2:
        f.append("LOW_LIGHT")
    if raw["mean_luma"] < 0.3 and obs.frame_p95_luma > 0.85:
        f.append("BACKLIT")
    return f


def yoloe_flags(scored: list, registry, cfg) -> None:
    """Open-vocabulary headwear checks on the head crops of the top-N faces; unloads YOLOE afterwards."""
    n = int(cfg.flags.yoloe_top_n)
    conf = float(cfg.flags.yoloe_conf)
    top = sorted(scored, key=lambda s: -s.q.total)[:n]
    top = [s for s in top if s.obs.head_crop is not None and s.obs.head_crop.size > 0]
    if not top:
        return
    model = registry.open_vocab(YOLOE_CLASSES)
    try:
        crops = [s.obs.head_crop for s in top]
        results = model.predict(crops, conf=conf, verbose=False, device=registry.device, imgsz=320)
        for s, r in zip(top, results):
            if r.boxes is None or not len(r.boxes):
                continue
            cls = r.boxes.cls.cpu().numpy().astype(int)
            cf = r.boxes.conf.cpu().numpy()
            names = r.names
            hits = {}
            for c, p in zip(cls, cf):
                label = names[int(c)] if isinstance(names, (dict, list)) else str(c)
                flag = _YOLOE_MAP.get(label)
                if flag and p >= conf:
                    hits[flag] = max(hits.get(flag, 0.0), float(p))
            for flag in hits:
                if flag not in s.q.flags:
                    s.q.flags.append(flag)
            if hits:
                s.attrib["yoloe"] = {k: round(v, 3) for k, v in hits.items()}
    finally:
        if not getattr(registry, "hold_open_vocab", False):  # batch jobs keep it loaded and release at the end
            registry.release("open_vocab")


def summarise(flags_lists: list[list[str]]) -> dict[str, float]:
    n = max(1, len(flags_lists))
    out = {}
    for fl in ALL_FLAGS:
        c = sum(1 for f in flags_lists if fl in f)
        if c:
            out[fl] = c / n
    return out


__all__ = ["ALL_FLAGS", "FLAG_HELP", "basic_flags", "yoloe_flags", "summarise", "np"]
