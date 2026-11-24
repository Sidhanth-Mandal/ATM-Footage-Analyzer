"""Ground-truth labelling for the Kaggle clips (§14.2). OpenCV window, keyboard driven.

Prerequisite: the clips were uploaded as gallery clips and indexed (or run `--index-all` from the CLI with the
API's data dir), so each track has a template.

  python scripts/label_tool.py clusters  [--thr 0.45]     # cluster templates, review contact sheets, assign ids
  python scripts/label_tool.py customers                  # per clip: mark the customer track, interval, best frames
  python scripts/label_tool.py transactions [--seed 0]    # write synthetic transactions.csv from the labels

labels.json (samples/kaggle_atm_video/labels.json):
  {"track_identity": {track_db_id: "P001" | "unknown_covered"},
   "sessions": {clip_name: "S01"},
   "clips": {clip_name: {"customer_track_no": n, "at_machine": [t0, t1], "best_pts": [...]}},
   "flags": {face_id: ["MASK", ...]}}

Keys in the cluster view: [n]ext cluster, [m]erge with previous, [s]plit (each track its own id), [u]nknown_covered,
[q]uit and save. Customer view: [1-9] pick candidate rank, [b] mark current frame as a best face, [a]/[d] seek -/+ 0.5 s,
[[]/[]] set interval start/end, [enter] next clip.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
LABELS = ROOT / "samples" / "kaggle_atm_video" / "labels.json"


def load() -> dict:
    return json.loads(LABELS.read_text()) if LABELS.exists() else {"track_identity": {}, "sessions": {}, "clips": {}, "flags": {}}


def save(d: dict) -> None:
    LABELS.write_text(json.dumps(d, indent=1))
    print("saved", LABELS)


def sheet(paths: list[Path], title: str) -> np.ndarray:
    tiles = []
    for p in paths[:24]:
        im = cv2.imread(str(p))
        tiles.append(cv2.resize(im, (112, 112)) if im is not None else np.zeros((112, 112, 3), np.uint8))
    while len(tiles) % 8:
        tiles.append(np.zeros((112, 112, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)] or [np.zeros((112, 896, 3), np.uint8)]
    img = np.vstack(rows)
    cv2.putText(img, title, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    return img


def cmd_clusters(a):
    from sqlmodel import select
    from atmcctv import paths
    from atmcctv.db.models import Clip, FaceRow, Template, Track
    from atmcctv.db.session import read_session

    s = read_session()
    tm = {t.track_id: (np.frombuffer(t.vector, np.float32), t.face_ids_json) for t in s.exec(select(Template)).all()}
    ids = sorted(tm)
    if not ids:
        sys.exit("no templates: index the clips first")
    X = np.stack([tm[i][0] for i in ids])
    # agglomerative, cosine, average linkage
    clusters = [[i] for i in range(len(ids))]
    sim = X @ X.T
    while True:
        best, pair = -1.0, None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                v = float(np.mean(sim[np.ix_(clusters[i], clusters[j])]))
                if v > best:
                    best, pair = v, (i, j)
        if pair is None or best < a.thr:
            break
        i, j = pair
        clusters[i] += clusters.pop(j)
    d = load()
    for c in s.exec(select(Clip)).all():
        d["sessions"].setdefault(c.orig_name, c.orig_name.split("_")[0])
    pid = 1 + max([int(v[1:]) for v in d["track_identity"].values() if str(v).startswith("P") and v[1:].isdigit()] or [0])
    prev = None
    for k, cl in enumerate(clusters):
        tids = [ids[i] for i in cl]
        crops = [paths.absp(s.get(FaceRow, f).crop_path) for t in tids for f in tm[t][1][:3] if s.get(FaceRow, f)]
        names = {s.get(Clip, s.get(Track, t).clip_id).orig_name for t in tids}
        cv2.imshow("clusters", sheet(crops, f"cluster {k + 1}/{len(clusters)}: {len(tids)} tracks in {sorted(names)[:4]}"))
        key = chr(cv2.waitKey(0) & 0xFF)
        if key == "q":
            break
        if key == "m" and prev:
            label = prev
        elif key == "u":
            label = "unknown_covered"
        elif key == "s":
            for t in tids:
                d["track_identity"][str(t)] = f"P{pid:03d}"
                pid += 1
            prev = None
            continue
        else:
            label = f"P{pid:03d}"
            pid += 1
        for t in tids:
            d["track_identity"][str(t)] = label
        prev = label
    cv2.destroyAllWindows()
    save(d)


def cmd_customers(a):
    from sqlmodel import select
    from atmcctv import paths
    from atmcctv.core.persons import load_tracks
    from atmcctv.core.video import FrameSource
    from atmcctv.db.models import Clip
    from atmcctv.db.session import read_session

    s = read_session()
    d = load()
    for c in s.exec(select(Clip).where(Clip.status.in_(["tracked", "indexed"]))).all():
        if c.orig_name in d["clips"] and not a.redo:
            continue
        tracks, meta = load_tracks(paths.clip_dir(c.case_id, c.id))
        tracks.sort(key=lambda t: -max(float((x.bbox[3] - x.bbox[1]) * (x.bbox[2] - x.bbox[0])) for x in t.samples))
        rot = int((c.probe_json or {}).get("analysis_rotation", 0))
        lab = {"customer_track_no": None, "at_machine": [0.0, 0.0], "best_pts": []}
        with FrameSource(paths.absp(c.analysis_path), rot) as fs:
            t = 0.0
            while True:
                img = fs.get(t)
                if img is None:
                    break
                for rank, tr in enumerate(tracks[:9], 1):
                    smp = min(tr.samples, key=lambda x: abs(x.pts - t))
                    if abs(smp.pts - t) < 0.5:
                        x0, y0, x1, y1 = map(int, smp.bbox)
                        col = (0, 255, 0) if tr.track_id == lab["customer_track_no"] else (0, 200, 255)
                        cv2.rectangle(img, (x0, y0), (x1, y1), col, 3)
                        cv2.putText(img, f"[{rank}] track {tr.track_id}", (x0 + 4, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)
                cv2.putText(img, f"{c.orig_name} t={t:.1f}s at_machine={lab['at_machine']} best={len(lab['best_pts'])}",
                            (10, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.imshow("customer", cv2.resize(img, None, fx=0.6, fy=0.6))
                k = cv2.waitKey(0) & 0xFF
                ch = chr(k)
                if ch.isdigit() and 1 <= int(ch) <= min(9, len(tracks)):
                    lab["customer_track_no"] = tracks[int(ch) - 1].track_id
                elif ch == "a":
                    t = max(0.0, t - 0.5)
                elif ch == "d":
                    t = min(fs.duration - 0.05, t + 0.5)
                elif ch == "[":
                    lab["at_machine"][0] = t
                elif ch == "]":
                    lab["at_machine"][1] = t
                elif ch == "b":
                    lab["best_pts"].append(round(t, 3))
                elif k in (13, 10):
                    break
                elif ch == "q":
                    cv2.destroyAllWindows()
                    save(d)
                    return
        d["clips"][c.orig_name] = lab
        save(d)
    cv2.destroyAllWindows()


def cmd_transactions(a):
    from sqlmodel import select
    from atmcctv.core.timemap import from_epoch
    from atmcctv.db.models import Clip
    from atmcctv.db.session import read_session

    rng = random.Random(a.seed)
    d = load()
    s = read_session()
    out = ROOT / "samples" / "kaggle_atm_video" / "transactions.csv"
    rows = []
    for name, lab in d["clips"].items():
        c = s.exec(select(Clip).where(Clip.orig_name == name)).first()
        if not c or not c.timemap_json or lab.get("customer_track_no") is None:
            continue
        tm = c.timemap_json
        mid = (lab["at_machine"][0] + lab["at_machine"][1]) / 2
        wall = from_epoch(tm["a"] + tm["b"] * mid, tm["tz"]).replace(tzinfo=None) + timedelta(seconds=rng.uniform(-20, 20))
        offset = round(rng.uniform(-180, 180), 1) if rng.random() < 0.5 else 0.0
        journal = wall - timedelta(seconds=offset)      # t_dvr = t_journal + offset
        rows.append({"clip": name, "atm_id": "KAGGLE-ATM-01", "card_last4": "XXXX", "txn_time": journal.isoformat(timespec="seconds"),
                     "amount": rng.choice([500, 1000, 2000, 5000, 10000]), "clock_offset_s": offset})
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "atm_id", "card_last4", "txn_time", "amount", "clock_offset_s"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} synthetic transactions to {out}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("clusters")
    p.add_argument("--thr", type=float, default=0.45)
    p.set_defaults(fn=cmd_clusters)
    p = sp.add_parser("customers")
    p.add_argument("--redo", action="store_true")
    p.set_defaults(fn=cmd_customers)
    p = sp.add_parser("transactions")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_transactions)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
