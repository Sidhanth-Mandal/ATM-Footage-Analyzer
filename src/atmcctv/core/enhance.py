"""Optional, clearly labelled enhancement. Output is never embedded or used for matching."""
from __future__ import annotations

import io
import json

import cv2
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

METHODS = ("clahe", "gamma", "gfpgan_v1.4_onnx")

# FFHQ 512 five-point template (eyes, nose, mouth corners), normalised
FFHQ_512 = np.array([[0.37691676, 0.46864664], [0.62285697, 0.46912813], [0.50123859, 0.61331904],
                     [0.39308822, 0.72541100], [0.61150205, 0.72490465]], np.float32) * 512


def clahe(img: np.ndarray, clip: float = 2.0) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def gamma(img: np.ndarray, g: float = 0.7) -> np.ndarray:
    g = float(min(3.0, max(0.2, g)))
    lut = (np.power(np.arange(256) / 255.0, g) * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def gfpgan(session, crop: np.ndarray, lmk_crop: np.ndarray, blend: float = 1.0) -> np.ndarray:
    iod = float(np.linalg.norm(lmk_crop[1] - lmk_crop[0]))
    tmpl_iod = float(np.linalg.norm(FFHQ_512[1] - FFHQ_512[0]))
    s = float(min(4.0, max(1.0, tmpl_iod / max(iod, 1.0))))
    big = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC) if s > 1 else crop.copy()
    lmk = lmk_crop.astype(np.float32) * s
    M, _ = cv2.estimateAffinePartial2D(lmk, FFHQ_512, method=cv2.LMEDS)
    face = cv2.warpAffine(big, M, (512, 512), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    x = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = ((x - 0.5) / 0.5).transpose(2, 0, 1)[None]
    inp = session.get_inputs()[0].name
    y = session.run(None, {inp: x.astype(np.float32)})[0][0]
    y = ((np.clip(y, -1, 1) + 1) / 2 * 255).transpose(1, 2, 0).round().astype(np.uint8)
    restored = cv2.cvtColor(y, cv2.COLOR_RGB2BGR)
    Minv = cv2.invertAffineTransform(M)
    h, w = big.shape[:2]
    back = cv2.warpAffine(restored, Minv, (w, h), flags=cv2.INTER_LINEAR)
    mask = np.zeros((512, 512), np.float32)
    cv2.rectangle(mask, (24, 24), (488, 488), 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 16)
    mback = cv2.warpAffine(mask, Minv, (w, h))[..., None] * float(blend)
    out = (back.astype(np.float32) * mback + big.astype(np.float32) * (1 - mback)).clip(0, 255).astype(np.uint8)
    return out


def add_banner(img: np.ndarray, method: str) -> np.ndarray:
    text = f"ENHANCED: {method} - NOT ORIGINAL"
    w = max(img.shape[1], 360)
    if img.shape[1] < w:
        pad = np.zeros((img.shape[0], w - img.shape[1], 3), np.uint8)
        img = np.concatenate([img, pad], axis=1)
    scale = max(0.45, min(1.2, w / 700))
    th = int(34 * scale) + 10
    bar = np.zeros((th, w, 3), np.uint8)
    bar[:] = (0, 0, 200)
    cv2.putText(bar, text, (8, th - 12), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), max(1, int(2 * scale)),
                cv2.LINE_AA)
    out = np.concatenate([bar, img, bar], axis=0)
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), (0, 0, 200), 3)
    return out


def enhance(method: str, crop: np.ndarray, lmk_crop: np.ndarray | None, params: dict, gfpgan_session=None) -> np.ndarray:
    if method == "clahe":
        out = clahe(crop, params.get("clip", 2.0))
    elif method == "gamma":
        out = gamma(crop, params.get("gamma", 0.7))
    elif method == "gfpgan_v1.4_onnx":
        if gfpgan_session is None or lmk_crop is None:
            raise ValueError("GFPGAN needs the ONNX session and landmarks")
        out = gfpgan(gfpgan_session, crop, lmk_crop, float(params.get("blend", 1.0)))
    else:
        raise ValueError(f"unknown method {method!r}; choose one of {METHODS}")
    return add_banner(out, method)


def png_with_metadata(img: np.ndarray, meta: dict) -> bytes:
    info = PngInfo()
    for k, v in meta.items():
        info.add_text(k, v if isinstance(v, str) else json.dumps(v))
    info.add_text("ENHANCED", "true - NOT ORIGINAL PIXELS")
    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(buf, format="PNG", pnginfo=info)
    return buf.getvalue()
