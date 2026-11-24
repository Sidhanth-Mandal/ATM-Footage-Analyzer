"""Frame PTS -> wall-clock mapping: OCR of burned-in overlay, metadata, or manual.

Convention: wall_epoch_s = a + b * pts_s.
clock_offset_s = DVR clock - bank journal clock, so t_dvr = t_journal + offset.
"""
from __future__ import annotations

import logging
import random
import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from .types import TimeMap

log = logging.getLogger(__name__)

_CONFUSIONS = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "I": "1", "l": "1", "|": "1", "i": "1",
                             "S": "5", "s": "5", "B": "8", "Z": "2", "z": "2", "G": "6", "g": "9"})

# Each pattern yields named groups; date order is applied for the ambiguous numeric form.
_PATTERNS = [
    re.compile(r"(?P<y>\d{4})[-/.](?P<m>\d{1,2})[-/.](?P<d>\d{1,2})\D{0,12}?(?P<H>\d{1,2})[:.](?P<M>\d{2})[:.](?P<S>\d{2})"
               r"(?:\s*(?P<ap>[AaPp][Mm]))?"),
    re.compile(r"(?P<a>\d{1,2})[-/.](?P<b>\d{1,2})[-/.](?P<y>\d{4})\D{0,12}?(?P<H>\d{1,2})[:.](?P<M>\d{2})[:.](?P<S>\d{2})"
               r"(?:\s*(?P<ap>[AaPp][Mm]))?"),
]


def normalise_ocr_text(text: str) -> str:
    """Fix common OCR confusions only in positions that should be digits.

    A character is treated as a digit position when it sits between digits or
    separators (-/.:) or directly next to a digit.
    """
    chars = list(text)
    n = len(chars)
    seps = set("-/.: ")
    for i, c in enumerate(chars):
        if c.isdigit() or c not in _CONFUSIONS_KEYS:
            continue
        left = chars[i - 1] if i > 0 else ""
        right = chars[i + 1] if i + 1 < n else ""
        if (left.isdigit() or left in seps and left != " ") or (right.isdigit() or right in seps and right != " "):
            # do not touch AM/PM markers
            pair = (c + right).upper() if right else ""
            if pair in ("AM", "PM") or (left + c).upper() in ("AM", "PM"):
                continue
            chars[i] = c.translate(_CONFUSIONS)
    return "".join(chars)


_CONFUSIONS_KEYS = set("OoDQIl|iSsBZzGg")


def parse_timestamp(text: str, date_order: str = "DMY") -> datetime | None:
    """Parse an overlay string to a naive datetime (local DVR clock)."""
    t = normalise_ocr_text(text)
    for pat in _PATTERNS:
        m = pat.search(t)
        if not m:
            continue
        g = m.groupdict()
        try:
            y = int(g["y"])
            if "m" in g and g.get("m") is not None:
                mo, d = int(g["m"]), int(g["d"])
            else:
                a, b = int(g["a"]), int(g["b"])
                if date_order.upper().startswith("M"):
                    mo, d = a, b
                else:
                    d, mo = a, b
            H, M, S = int(g["H"]), int(g["M"]), int(g["S"])
            ap = (g.get("ap") or "").upper()
            if ap == "PM" and H < 12:
                H += 12
            if ap == "AM" and H == 12:
                H = 0
            return datetime(y, mo, d, H, M, S)
        except (ValueError, TypeError):
            continue
    return None


def to_epoch(dt: datetime, tz: str) -> float:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo(tz))
    return dt.timestamp()


def from_epoch(epoch: float, tz: str) -> datetime:
    return datetime.fromtimestamp(epoch, tz=ZoneInfo(tz))


def ransac_fit(pts: list[float], walls: list[float], tol: float = 1.5, iters: int = 200,
               min_inliers: int = 5, min_ratio: float = 0.6, seed: int = 0) -> tuple[float, float, int, float] | None:
    """Fit wall = a + b*pts with RANSAC. Returns (a, b, inliers, residual_s) or None."""
    x = np.asarray(pts, dtype=np.float64)
    y = np.asarray(walls, dtype=np.float64)
    n = len(x)
    if n < 2:
        return None
    rng = random.Random(seed)
    best_mask = None
    best_count = 0
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(pairs) > iters:
        pairs = rng.sample(pairs, iters)
    for i, j in pairs:
        if abs(x[j] - x[i]) < 1e-6:
            continue
        b = (y[j] - y[i]) / (x[j] - x[i])
        if not (0.9 <= b <= 1.1):
            continue
        a = y[i] - b * x[i]
        mask = np.abs(y - (a + b * x)) <= tol
        c = int(mask.sum())
        if c > best_count:
            best_count, best_mask = c, mask
    if best_mask is None or best_count < min_inliers or best_count / n < min_ratio:
        return None
    xi, yi = x[best_mask], y[best_mask]
    if np.ptp(xi) < 1e-6:
        b = 1.0
        a = float(np.mean(yi - xi))
    else:
        b, a = np.polyfit(xi, yi, 1)
        if not (0.9 <= b <= 1.1):
            b = 1.0
            a = float(np.mean(yi - xi))
    res = float(np.sqrt(np.mean((yi - (a + b * xi)) ** 2)))
    return float(a), float(b), best_count, res


def _ocr_variants(crop: np.ndarray) -> list[np.ndarray]:
    up = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    grey = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thi = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return [cv2.cvtColor(th, cv2.COLOR_GRAY2BGR), cv2.cvtColor(thi, cv2.COLOR_GRAY2BGR)]


def ocr_read(ocr_engine, crop: np.ndarray) -> tuple[str, float]:
    best_text, best_conf = "", 0.0
    for img in _ocr_variants(crop):
        try:
            res = ocr_engine(img)
        except Exception as e:  # OCR failures are non-fatal
            log.debug("OCR failed: %s", e)
            continue
        txts = list(getattr(res, "txts", None) or [])
        scores = list(getattr(res, "scores", None) or [])
        if not txts:
            continue
        text = " ".join(txts)
        conf = float(np.mean(scores)) if scores else 0.0
        if conf > best_conf:
            best_text, best_conf = text, conf
    return best_text, best_conf


def fit_ocr(frames, roi_norm: list[float], ocr_engine, cfg, date_order: str, tz: str) -> tuple[TimeMap | None, list[dict]]:
    """frames: iterable of (pts, idx, bgr). roi_norm: [x0, y0, x1, y1] normalised."""
    reads = []
    for pts, _idx, img in frames:
        h, w = img.shape[:2]
        x0, y0, x1, y1 = roi_norm
        crop = img[max(0, int(y0 * h)):min(h, int(y1 * h)), max(0, int(x0 * w)):min(w, int(x1 * w))]
        if crop.size == 0:
            continue
        text, conf = ocr_read(ocr_engine, crop)
        dt = parse_timestamp(text, date_order) if text else None
        reads.append({"pts": pts, "text": text, "conf": conf, "parsed": dt.isoformat() if dt else None,
                      "wall": to_epoch(dt, tz) if dt else None})
    good = [r for r in reads if r["wall"] is not None]
    if not good:
        return None, reads
    fit = ransac_fit([r["pts"] for r in good], [r["wall"] for r in good], tol=float(cfg.timemap.ransac_tol_s),
                     iters=int(cfg.timemap.ransac_iters), min_inliers=int(cfg.timemap.min_inliers),
                     min_ratio=float(cfg.timemap.min_inlier_ratio))
    if not fit:
        return None, reads
    a, b, inl, res = fit
    return TimeMap("ocr", a, b, tz, inl, res, [f"{len(good)}/{len(reads)} OCR reads parsed",
                                                "OCR reads have 1 s resolution"]), reads


def from_metadata(creation_time: datetime | None, duration_s: float, metadata_is_end: bool, tz: str,
                  start_pts: float = 0.0) -> TimeMap | None:
    if creation_time is None:
        return None
    ct = creation_time if creation_time.tzinfo else creation_time.replace(tzinfo=timezone.utc)
    epoch = ct.timestamp()
    if metadata_is_end:
        a = epoch - duration_s - start_pts
        note = "creation_time treated as END of recording"
    else:
        a = epoch - start_pts
        note = "creation_time treated as START of recording"
    return TimeMap("metadata", a, 1.0, tz, 0, 0.0, [note, "verify once with a clip of known start time"])


def manual(start: datetime, tz: str, start_pts: float = 0.0) -> TimeMap:
    return TimeMap("manual", to_epoch(start, tz) - start_pts, 1.0, tz, 0, 0.0, ["investigator-entered start time"])


def txn_to_pts(txn_journal: datetime, clock_offset_s: float, tm: TimeMap) -> float:
    """Journal time -> video PTS: t_dvr = t_journal + clock_offset_s, pts = (epoch(t_dvr) - a) / b."""
    epoch = to_epoch(txn_journal, tm.tz) + clock_offset_s
    return tm.pts(epoch)


def wall_iso(tm: TimeMap | None, pts: float) -> str | None:
    if tm is None:
        return None
    return from_epoch(tm.wall(pts), tm.tz).isoformat(timespec="milliseconds")


__all__ = ["parse_timestamp", "ransac_fit", "fit_ocr", "from_metadata", "manual", "txn_to_pts", "wall_iso",
           "to_epoch", "from_epoch", "timedelta"]
