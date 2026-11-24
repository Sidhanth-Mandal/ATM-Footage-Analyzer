"""Evaluation (§14.4).

  python scripts/evaluate.py --ingest [--formats samples/formats]
      accept/reject and error code per generated file (acceptance: 100 %)
  python scripts/evaluate.py --labels samples/kaggle_atm_video/labels.json --db
      linking top-1, best-frame overlap, flag precision/recall, search TAR@FMR from the labelled DB state
  python scripts/evaluate.py --timemap-synthetic
      OCR time-map error on a clip with a burned-in timestamp (acceptance: <= 1 s)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def eval_ingest(formats: Path) -> bool:
    from atmcctv.core.video import ingest
    from atmcctv.settings import get_config

    cfg = get_config()
    exp = json.loads((formats / "expected.json").read_text())
    ok_all = True
    rows = []
    with tempfile.TemporaryDirectory() as d:
        for name, e in exp.items():
            r = ingest(formats / name, Path(d) / name.replace(".", "_"), cfg)
            accepted = r.error is None
            ok = accepted == e["accept"]
            if ok and not accepted and e.get("code"):
                ok = r.error.code in e["code"]
            if ok and accepted and e.get("normalised") is not None:
                ok = r.normalised == e["normalised"]
            ok_all &= ok
            rows.append((name, "PASS" if ok else "FAIL", "accepted" if accepted else f"rejected {r.error.code}",
                         f"normalised={r.normalised}" if accepted else r.error.message[:70],
                         "; ".join(r.decisions)[:90]))
    for row in rows:
        print(f"{row[0]:22} {row[1]:5} {row[2]:32} {row[3]:30} {row[4]}")
    print(f"\ningest: {sum(r[1] == 'PASS' for r in rows)}/{len(rows)} correct (acceptance 100%)")
    return ok_all


def eval_timemap_synthetic() -> bool:
    """Burn a known timestamp into a clip, run OCR fitting, compare to the truth."""
    from atmcctv.core import timemap as tmap
    from atmcctv.core.registry import ModelRegistry
    from atmcctv.core.video import FrameSource
    from atmcctv.settings import get_config

    cfg = get_config()
    src = ROOT / "samples" / "kaggle_atm_video" / "normal_1.mp4"
    start = datetime(2026, 3, 3, 8, 42, 10, tzinfo=timezone.utc)
    epoch = int(start.timestamp())
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "ts.mp4"
        # gmtime's default format is "YYYY-MM-DD HH:MM:SS"; an explicit font avoids fontconfig on Windows
        font = "fontfile='C\\:/Windows/Fonts/arial.ttf':" if sys.platform == "win32" else ""
        draw = (f"drawtext={font}text='%{{pts\\:gmtime\\:{epoch}}}':x=20:y=20:fontsize=48:"
                "fontcolor=white:box=1:boxcolor=black@0.6")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-t", "20", "-vf", f"loop=loop=2:size=210,{draw}",
                        "-c:v", "libx264", "-an", str(out)], check=True)
        reg = ModelRegistry(cfg)
        with FrameSource(out) as fs:
            tm, reads = tmap.fit_ocr(fs.iter_sampled(1.0 / float(cfg.timemap.ocr_every_s)), [0.0, 0.0, 0.35, 0.12],
                                     reg.ocr(), cfg, "YMD", "UTC")
    if tm is None:
        print("OCR fit failed; reads:", reads[:5])
        return False
    errs = [abs(tm.wall(p) - (epoch + p)) for p in (0.0, 5.0, 10.0, 15.0)]
    print(f"timemap: method {tm.method}, inliers {tm.inliers}/{len(reads)}, b={tm.b:.5f}, max abs error {max(errs):.2f} s")
    return max(errs) <= 1.0


def eval_labels(labels_p: Path) -> None:
    """Metrics against labels.json written by scripts/label_tool.py."""
    import numpy as np
    from sqlmodel import select

    from atmcctv.db.models import CandidateRow, Clip, FaceRow, Template, Track, Transaction
    from atmcctv.db.session import read_session

    labels = json.loads(labels_p.read_text())
    s = read_session()
    # linking top-1
    hit = tot = 0
    overlaps = []
    for clip_name, lab in labels.get("clips", {}).items():
        clip = s.exec(select(Clip).where(Clip.orig_name == clip_name)).first()
        if not clip or lab.get("customer_track_no") is None:
            continue
        for t in s.exec(select(Transaction).where(Transaction.clip_id == clip.id)).all():
            top = s.exec(select(CandidateRow).where(CandidateRow.transaction_id == t.id).order_by(CandidateRow.rank)).first()
            if not top:
                continue
            tot += 1
            hit += int(s.get(Track, top.track_id).track_no == lab["customer_track_no"])
            if t.chosen_track_id and lab.get("best_pts"):
                sel = [f.pts for f in s.exec(select(FaceRow).where(FaceRow.track_id == t.chosen_track_id,
                                                                   FaceRow.selected == True)).all()]  # noqa: E712
                overlaps.append(sum(any(abs(p - q) <= 0.5 for q in sel) for p in lab["best_pts"]))
    if tot:
        print(f"linking top-1: {hit}/{tot} = {hit / tot:.0%} (acceptance >= 90%)")
    if overlaps:
        print(f"best frames: mean overlap with human top-6 {np.mean(overlaps):.2f} (acceptance >= 4)")
    # flags precision / recall
    fl = labels.get("flags", {})  # face_id -> [flags]
    for flag in ("MASK", "FULL_COVER", "CAP_HAT", "PARTIAL_FACE"):
        tp = fp = fn = 0
        for fid, truth in fl.items():
            f = s.get(FaceRow, int(fid))
            if not f:
                continue
            pred, true = flag in f.flags_json, flag in truth
            tp += pred and true
            fp += pred and not true
            fn += true and not pred
        if tp + fp + fn:
            print(f"{flag:13} precision {tp / max(1, tp + fp):.2f} recall {tp / max(1, tp + fn):.2f} "
                  "(acceptance P>=0.9, R>=0.8)")
    # search: genuine/impostor over labelled identities
    ident = {int(k): v for k, v in labels.get("track_identity", {}).items()}
    tm = {t.track_id: np.frombuffer(t.vector, np.float32) for t in s.exec(select(Template)).all() if t.track_id in ident}
    ids = sorted(tm)
    gen, imp = [], []
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if "unknown" in str(ident[a]) or "unknown" in str(ident[b]):
                continue
            sim = float(tm[a] @ tm[b])
            (gen if ident[a] == ident[b] else imp).append(sim)
    if gen and imp:
        imp_s = np.sort(imp)[::-1]
        thr = imp_s[max(0, int(0.01 * len(imp_s)) - 1)]
        print(f"search: {len(gen)} genuine / {len(imp)} impostor pairs; TAR@FMR1% = {np.mean(np.array(gen) > thr):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ingest", action="store_true")
    ap.add_argument("--formats", default=str(ROOT / "samples" / "formats"))
    ap.add_argument("--timemap-synthetic", action="store_true")
    ap.add_argument("--labels")
    a = ap.parse_args()
    ok = True
    if a.ingest:
        ok &= eval_ingest(Path(a.formats))
    if a.timemap_synthetic:
        ok &= eval_timemap_synthetic()
    if a.labels:
        eval_labels(Path(a.labels))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
