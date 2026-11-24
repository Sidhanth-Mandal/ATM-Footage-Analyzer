"""Download the Kaggle "ATM Video" dataset (rohulaminlabid/atm-video, CC0) file by file.

  python scripts/fetch_kaggle_atm.py [--limit N] [--folder Normal|Hand_Sign]

Files go to samples/kaggle_atm_video/<Normal|Hand_Sign>/<n>.mp4 and a manifest.json with size and SHA-256.
Fallback if unauthenticated downloads stop working: `kaggle datasets download rohulaminlabid/atm-video`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "samples" / "kaggle_atm_video"
LIST = "https://www.kaggle.com/api/v1/datasets/list/rohulaminlabid/atm-video?pageSize=200"
DL = "https://www.kaggle.com/api/v1/datasets/download/rohulaminlabid/atm-video/"


def list_files() -> list[dict]:
    files, token = [], None
    while True:
        url = LIST + (f"&pageToken={urllib.parse.quote(token)}" if token else "")
        with urllib.request.urlopen(url, timeout=60) as r:
            d = json.loads(r.read())
        files += d.get("datasetFiles", [])
        token = d.get("nextPageToken") if d.get("hasNextPageToken") else None
        if not token:
            return files


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--folder", choices=["Normal", "Hand_Sign"])
    a = ap.parse_args()
    files = [f for f in list_files() if f["name"].lower().endswith(".mp4")]
    if a.folder:
        files = [f for f in files if f"/{a.folder}(" in f["name"]]
    files.sort(key=lambda f: (f["name"].split("/")[1], int(Path(f["name"]).stem) if Path(f["name"]).stem.isdigit() else 0))
    if a.limit:
        files = files[:a.limit]
    manifest_p = OUT / "manifest.json"
    manifest = json.loads(manifest_p.read_text()) if manifest_p.exists() else {}
    for f in files:
        name = f["name"]                                    # ATM_Video/Normal(Resize)/1.mp4
        folder = name.split("/")[1].split("(")[0]
        dest = OUT / folder / Path(name).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size == f["totalBytes"]:
            print("cached", dest.relative_to(ROOT))
        else:
            url = DL + urllib.parse.quote(name, safe="()")
            print("downloading", name, f"{f['totalBytes'] / 1e6:.1f} MB")
            tmp = dest.with_suffix(".part")
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as out:
                while chunk := r.read(1 << 20):
                    out.write(chunk)
            tmp.replace(dest)
        manifest[str(dest.relative_to(OUT).as_posix())] = {"source": name, "size": dest.stat().st_size,
                                                           "sha256": sha256(dest), "label": folder}
        manifest_p.write_text(json.dumps(manifest, indent=1))
    print(f"{len(files)} files, manifest at {manifest_p}")


if __name__ == "__main__":
    main()
