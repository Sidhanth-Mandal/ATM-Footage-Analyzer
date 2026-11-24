"""Detection view: luminance-corrected copy for detectors under bad lighting.

Only detectors see this view. Scoring, crops and evidence always use original pixels.
"""
from __future__ import annotations

import cv2
import numpy as np

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def needs_correction(img: np.ndarray, dark_mean: float, backlight_spread: float) -> bool:
    small = cv2.resize(img, (160, max(1, int(160 * img.shape[0] / max(img.shape[1], 1)))), interpolation=cv2.INTER_AREA)
    L = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    p50, p95 = np.percentile(L, [50, 95])
    return float(L.mean()) < dark_mean or float(p95 - p50) > backlight_spread


def correct(img: np.ndarray, gamma: float = 0.6) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = _CLAHE.apply(lab[..., 0])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    lut = (np.power(np.arange(256) / 255.0, gamma) * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.LUT(out, lut)


def detview(img: np.ndarray, cfg) -> tuple[np.ndarray, bool]:
    """Return (view, corrected)."""
    if img.size == 0:
        return img, False
    if needs_correction(img, float(cfg.det.dark_mean), float(cfg.det.backlight_spread)):
        return correct(img, float(cfg.det.gamma)), True
    return img, False


def frame_p95_luma(img: np.ndarray) -> float:
    small = cv2.resize(img, (160, max(1, int(160 * img.shape[0] / max(img.shape[1], 1)))), interpolation=cv2.INTER_AREA)
    return float(np.percentile(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 95) / 255.0)
