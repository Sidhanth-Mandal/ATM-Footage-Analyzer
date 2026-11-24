"""Download and cache every model weight in models/ so the app runs offline.

Usage:  python scripts/download_models.py [--skip-yoloe] [--skip-gfpgan]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atmcctv.settings import get_config, models_dir  # noqa: E402

GFPGAN_SOURCES = [
    "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gfpgan_1.4.onnx",
    "https://huggingface.co/HowToSD/GFPGAN-ONNX/resolve/main/GFPGANv1.4.onnx",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(urls: list[str], dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached {dest.name}")
        return
    for url in urls:
        try:
            print(f"  downloading {url}")
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.replace(dest)
            return
        except Exception as e:  # try next source
            print(f"  failed: {e}")
    raise SystemExit(f"could not download {dest.name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-yoloe", action="store_true")
    ap.add_argument("--skip-gfpgan", action="store_true")
    args = ap.parse_args()

    cfg = get_config()
    mdir = models_dir(cfg)
    mdir.mkdir(parents=True, exist_ok=True)
    os.environ["UNIFACE_CACHE_DIR"] = str(mdir / "uniface")

    import torch  # noqa: F401  (load CUDA DLLs before onnxruntime)
    from uniface.constants import (
        AdaFaceWeights, EDifFIQAWeights, FaceAttribNetWeights, HeadPoseWeights,
        ParsingWeights, SCRFDWeights,
    )
    from uniface.model_store import verify_model_weights

    print("UniFace weights ->", mdir / "uniface")
    for w in (SCRFDWeights.SCRFD_10G_KPS, AdaFaceWeights.IR_101, EDifFIQAWeights.M,
              HeadPoseWeights.RESNET18, ParsingWeights.RESNET18, FaceAttribNetWeights.DEFAULT):
        p = verify_model_weights(w)
        print(f"  {w.value}: {p}")

    print("Ultralytics weights ->", mdir)
    from ultralytics import YOLO
    from ultralytics.utils.downloads import attempt_download_asset

    for name in [cfg.models.person_pose] + ([] if args.skip_yoloe else [cfg.models.open_vocab]):
        dest = mdir / name
        if not dest.exists():
            attempt_download_asset(str(dest))
        print(f"  {name}: {dest.exists()}")
    if not args.skip_yoloe:
        # YOLOE needs its MobileCLIP text encoder for set_classes(); trigger the download once
        # with cwd=models so the encoder file is cached next to the other weights.
        cwd = os.getcwd()
        os.chdir(mdir)
        try:
            m = YOLO(str(mdir / cfg.models.open_vocab))
            m.set_classes(["cap", "hat"])
            print("  YOLOE text encoder ready")
        finally:
            os.chdir(cwd)

    if not args.skip_gfpgan:
        print("GFPGAN v1.4 ONNX ->", mdir)
        fetch(GFPGAN_SOURCES, mdir / cfg.models.gfpgan)

    print("RapidOCR models (downloaded by the package on first use)")
    try:
        from rapidocr import RapidOCR
        RapidOCR()
        print("  RapidOCR ready")
    except Exception as e:
        print(f"  RapidOCR init failed: {e}")

    print("\nModel files:")
    for p in sorted(mdir.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(mdir)}  {p.stat().st_size/1e6:.1f} MB  {sha256(p)[:16]}")


if __name__ == "__main__":
    main()
