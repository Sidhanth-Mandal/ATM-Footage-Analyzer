"""Create short synthetic clips with a clearly visible face (public-domain NASA portrait shipped with
scikit-image) to exercise search, review and export when the real samples have no searchable face.

  python scripts/make_synthetic_face_clips.py [--out samples/synthetic]
Writes: face_a.mp4 (slow zoom-in, 6 s) and face_b.mp4 (different crop/brightness, 6 s) of the same person.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
from skimage import data

ROOT = Path(__file__).resolve().parents[1]


def render(img: np.ndarray, out: Path, zoom: tuple[float, float], shift: tuple[int, int], gain: float, secs=6, fps=25):
    H, W = 1080, 1920
    tmp = out.with_suffix(".y4m.tmp.avi")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"MJPG"), fps, (W, H))
    n = secs * fps
    for i in range(n):
        z = zoom[0] + (zoom[1] - zoom[0]) * i / max(1, n - 1)
        canvas = np.full((H, W, 3), 40, np.uint8)
        s = int(H * z / img.shape[0] * img.shape[0])
        im = cv2.resize(img, (s, s))
        im = np.clip(im.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        x0, y0 = (W - s) // 2 + shift[0], H - s + shift[1]
        xa, ya = max(0, x0), max(0, y0)
        xb, yb = min(W, x0 + s), min(H, y0 + s)
        canvas[ya:yb, xa:xb] = im[ya - y0:yb - y0, xa - x0:xb - x0]
        vw.write(canvas)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(tmp), "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p",
                    "-metadata", "creation_time=2026-03-03T03:12:17Z", str(out)], check=True)
    tmp.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "samples" / "synthetic"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    img = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    render(img, out / "face_a.mp4", (0.85, 1.0), (0, 0), 1.0)
    render(cv2.flip(img, 1), out / "face_b.mp4", (0.9, 0.95), (-250, 30), 0.85)
    print("wrote", sorted(p.name for p in out.glob("*.mp4")))


if __name__ == "__main__":
    main()
