"""Calibrate search bands and FIQA bounds from labelled identities (§7.12).

  python scripts/calibrate.py --labels samples/kaggle_atm_video/labels.json [--pairs extra_pairs.json] [--dry-run]

Inputs
  labels.json        written by scripts/label_tool.py: track_identity {track_db_id: identity}, session per clip
  --pairs (optional) external genuine/impostor score lists, e.g. ChokePoint: {"genuine": [...], "impostor": [...]}
Outputs
  config/calibration.yaml (merged over config/default.yaml at load time):
    search.bands {strong: T@FMR0.1%, possible: T@FMR1%, weak: T@FMR5%}, quality.fiqa_bounds [p5, p95],
    calibration metadata (date, dataset hash, EER, pair counts).
Also reports clothing/background leakage: genuine scores for same-session vs cross-session pairs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def track_score(q_t, q_faces, g_t, g_faces) -> float:
    """Same rule as search: max(template cosine, mean of top-3 face cosines)."""
    face_sims = sorted((float(g @ q_t) for g in g_faces), reverse=True)[:3]
    return max(float(q_t @ g_t), float(np.mean(face_sims)) if face_sims else -1.0)


def thr_at_fmr(imp: np.ndarray, fmr: float) -> float:
    s = np.sort(imp)[::-1]
    k = max(0, int(np.ceil(fmr * len(s))) - 1)
    return float(s[k]) + 1e-6


def eer(gen: np.ndarray, imp: np.ndarray) -> tuple[float, float]:
    """Equal error rate and its threshold."""
    best_gap, best_rate, best_t = 2.0, 1.0, 0.0
    for t in np.unique(np.concatenate([gen, imp])):
        frr = float(np.mean(gen < t))
        far = float(np.mean(imp >= t))
        if abs(frr - far) < best_gap:
            best_gap, best_rate, best_t = abs(frr - far), (frr + far) / 2, float(t)
    return best_rate, best_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--pairs")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    from sqlmodel import select
    from atmcctv.db.models import Clip, FaceRow, Template, Track
    from atmcctv.db.session import read_session
    from atmcctv.settings import CALIBRATION

    labels = json.loads(Path(a.labels).read_text())
    ident = {int(k): str(v) for k, v in labels.get("track_identity", {}).items()}
    sessions = labels.get("sessions", {})  # clip_name -> session id
    s = read_session()
    T, F, sess = {}, {}, {}
    for t in s.exec(select(Template)).all():
        if t.track_id not in ident or ident[t.track_id].startswith("unknown"):
            continue
        T[t.track_id] = np.frombuffer(t.vector, np.float32)
        F[t.track_id] = [np.frombuffer(f.embedding, np.float32) for f in (s.get(FaceRow, i) for i in t.face_ids_json)
                         if f is not None and f.embedding is not None]
        tr = s.get(Track, t.track_id)
        sess[t.track_id] = sessions.get(s.get(Clip, tr.clip_id).orig_name, tr.clip_id)
    ids = sorted(T)
    gen, imp, gen_same, gen_cross = [], [], [], []
    for i, x in enumerate(ids):
        for y in ids[i + 1:]:
            sc = track_score(T[x], F[x], T[y], F[y])
            if ident[x] == ident[y]:
                gen.append(sc)
                (gen_same if sess[x] == sess[y] else gen_cross).append(sc)
            else:
                imp.append(sc)
    if a.pairs:
        p = json.loads(Path(a.pairs).read_text())
        gen += list(map(float, p.get("genuine", [])))
        imp += list(map(float, p.get("impostor", [])))
    if len(gen) < 5 or len(imp) < 50:
        sys.exit(f"not enough labelled pairs (genuine {len(gen)}, impostor {len(imp)}); label more tracks first")
    gen_a, imp_a = np.array(gen), np.array(imp)
    e, et = eer(gen_a, imp_a)
    bands = {"strong": round(thr_at_fmr(imp_a, 0.001), 4), "possible": round(thr_at_fmr(imp_a, 0.01), 4),
             "weak": round(thr_at_fmr(imp_a, 0.05), 4)}
    fiqa = np.array([f.raw_json.get("fiqa_raw") for f in s.exec(select(FaceRow)).all()
                     if f.raw_json.get("fiqa_raw") is not None], np.float64)
    fiqa_bounds = [round(float(np.percentile(fiqa, 5)), 4), round(float(np.percentile(fiqa, 95)), 4)] if len(fiqa) >= 50 else None
    digest = hashlib.sha256(json.dumps(labels, sort_keys=True).encode()).hexdigest()
    print(f"pairs: {len(gen)} genuine, {len(imp)} impostor; EER {e:.3f} at {et:.3f}")
    print(f"TAR at bands: strong {np.mean(gen_a >= bands['strong']):.2f}, possible {np.mean(gen_a >= bands['possible']):.2f}, "
          f"weak {np.mean(gen_a >= bands['weak']):.2f}")
    print(f"bands: {bands}; fiqa bounds: {fiqa_bounds}")
    if gen_same or gen_cross:
        print(f"leakage check: genuine same-session mean {np.mean(gen_same) if gen_same else float('nan'):.3f} (n={len(gen_same)}), "
              f"cross-session mean {np.mean(gen_cross) if gen_cross else float('nan'):.3f} (n={len(gen_cross)}). "
              "A large gap means scores also reflect clothing/background, not only faces.")
    out = {"search": {"bands": bands},
           "calibration": {"date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "labels_sha256": digest,
                           "eer": round(e, 4), "n_genuine": len(gen), "n_impostor": len(imp),
                           "n_genuine_same_session": len(gen_same), "n_genuine_cross_session": len(gen_cross)}}
    if fiqa_bounds:
        out["quality"] = {"fiqa_bounds": fiqa_bounds}
    if a.dry_run:
        print(yaml.safe_dump(out, sort_keys=False))
        return
    CALIBRATION.write_text("# written by scripts/calibrate.py; merged over default.yaml\n" + yaml.safe_dump(out, sort_keys=False),
                           encoding="utf-8")
    print("wrote", CALIBRATION)


if __name__ == "__main__":
    main()
