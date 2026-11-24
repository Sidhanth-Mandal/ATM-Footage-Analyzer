"""Command line: run the core pipeline without the server, admin tasks, benchmarks.

  atmcctv analyse <video> [--txn-time ISO] [--offset S] [--camera fascia|lobby] [--out DIR]
  atmcctv init-db
  atmcctv create-user <username> --role admin|investigator|viewer
  atmcctv atms-sync
  atmcctv verify-audit
  atmcctv bench <video>
"""
from __future__ import annotations

import argparse
import getpass
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2


def _pipeline(video: Path, out: Path, txn_time: str | None, offset: float, camera: str, tz: str,
              manual_start: str | None, top_k: int) -> dict:
    from .core import timemap as tmap
    from .core.faces import dense_pass
    from .core.linking import score_tracks
    from .core.persons import coarse_pass, save_tracks
    from .core.quality import score_faces
    from .core.registry import get_registry
    from .core.report import write_report
    from .core.select import select_best
    from .core.video import FrameSource, ingest
    from .settings import get_config

    cfg = get_config()
    reg = get_registry(cfg)
    out.mkdir(parents=True, exist_ok=True)
    timings = {}
    t = time.perf_counter()
    res = ingest(video, out, cfg)
    timings["ingest_s"] = time.perf_counter() - t
    if res.error:
        print(json.dumps(res.error.to_json(), indent=2))
        return {"error": res.error.to_json()}
    rot = 0 if res.normalised else res.probe.rotation_deg
    src = FrameSource(res.analysis_path, rot)
    if manual_start:
        tm = tmap.manual(datetime.fromisoformat(manual_start), tz, src.start_time)
    else:
        tm = tmap.from_metadata(res.probe.creation_time, src.duration, False, tz, src.start_time)
    t = time.perf_counter()
    tracks, meta = coarse_pass(src, reg, cfg, camera, progress=lambda f, m: print(f"\r{m}", end=""))
    timings["coarse_s"] = time.perf_counter() - t
    print()
    save_tracks(tracks, meta, out)
    if txn_time and tm:
        t_pts = tmap.txn_to_pts(datetime.fromisoformat(txn_time), offset, tm)
    else:
        t_pts = src.duration / 2
    cands = score_tracks(tracks, t_pts, float(cfg.linking.window_before_s), float(cfg.linking.window_after_s), {},
                         camera, meta["width"], meta["height"], cfg)
    report = {"probe": res.probe.to_json(), "decisions": res.decisions, "timemap": tm.to_json() if tm else None,
              "t_txn_pts": t_pts, "tracks": [{"id": tr.track_id, "source": tr.source, "t": [tr.t_start, tr.t_end],
                                              "n": len(tr.samples)} for tr in tracks],
              "candidates": [c.__dict__ for c in cands], "faces": [], "timings": timings}
    if cands:
        chosen = next(tr for tr in tracks if tr.track_id == cands[0].track_id)
        t = time.perf_counter()
        obs, stats = dense_pass(src, [chosen], reg, cfg, progress=lambda f, m: print(f"\r{m}", end=""))[chosen.track_id]
        timings["dense_s"] = time.perf_counter() - t
        print()
        t = time.perf_counter()
        scored = score_faces(obs, reg, cfg)
        timings["score_s"] = time.perf_counter() - t
        best = select_best(scored, top_k, float(cfg.select.min_gap_s))
        report["face_visible_frac"] = stats.face_visible_frac
        report["n_faces"] = len(scored)
        for i, sf in enumerate(best):
            cv2.imwrite(str(out / f"best_{i}_crop.png"), sf.obs.raw_crop)
            cv2.imwrite(str(out / f"best_{i}_aligned.png"), sf.obs.aligned)
            report["faces"].append({"pts": sf.obs.pts, "frame_idx": sf.obs.frame_idx, "total": sf.q.total,
                                    "grade": sf.q.grade, "sub": sf.q.sub, "raw": sf.q.raw, "flags": sf.q.flags,
                                    "caps": sf.q.caps, "searchable": sf.q.searchable,
                                    "wall": tmap.wall_iso(tm, sf.obs.pts)})
    src.close()
    report["models"] = reg.model_info()
    report["vram"] = reg.vram()
    write_report(out / "report.json", report)
    return report


def cmd_analyse(a):
    rep = _pipeline(Path(a.video), Path(a.out), a.txn_time, a.offset, a.camera, a.tz, a.start, a.top_k)
    if "error" in rep:
        sys.exit(2)
    print(json.dumps({"candidates": [(c["track_id"], round(c["score"], 3)) for c in rep["candidates"]],
                      "faces": [(round(f["total"], 1), f["grade"], f["flags"]) for f in rep["faces"]],
                      "face_visible_frac": rep.get("face_visible_frac"), "timings": rep["timings"],
                      "vram": rep.get("vram")}, indent=1))


def cmd_bench(a):
    rep = _pipeline(Path(a.video), Path(a.out), None, 0.0, a.camera, "UTC", None, 6)
    dur = rep.get("probe", {}).get("duration_s", 0)
    t = rep.get("timings", {})
    fps = float(rep.get("probe", {}).get("avg_fps") or 25)
    print(json.dumps({
        "clip_s": dur, "coarse_x_realtime": dur / t["coarse_s"] if t.get("coarse_s") else None,
        "coarse_fps_sampled": dur * 5 / t["coarse_s"] if t.get("coarse_s") else None,
        "dense_fps": (rep.get("n_faces", 0) and (dur * fps) / t["dense_s"]) if t.get("dense_s") else None,
        "timings": t, "vram": rep.get("vram")}, indent=1))
    # search latency over a synthetic 1,000-track index
    import numpy as np
    from .core.index import FaceIndex
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        idx = FaceIndex(Path(d))
        rng = np.random.default_rng(0)
        faces = {i: v for i, v in enumerate(rng.normal(size=(5000, 512)).astype("float32"))}
        faces = {k: v / np.linalg.norm(v) for k, v in faces.items()}
        tmpls = {i: faces[i * 5] for i in range(1000)}
        idx.add(faces, tmpls)
        f2t = {k: k // 5 for k in faces}
        t0 = time.perf_counter()
        idx.search(faces[0], f2t)
        print(json.dumps({"search_1000_tracks_ms": round((time.perf_counter() - t0) * 1000, 2)}))


def cmd_init_db(a):
    from .db.session import init_db
    init_db()
    print("database ready")


def cmd_create_user(a):
    from sqlmodel import select
    from . import audit
    from .api.auth import hash_password
    from .db.models import ROLES, User
    from .db.session import init_db, write_session

    if a.role not in ROLES:
        sys.exit(f"role must be one of {ROLES}")
    init_db()
    pw = a.password or getpass.getpass("password: ")
    if len(pw) < 10:
        sys.exit("password must be at least 10 characters")
    with write_session() as s:
        if s.exec(select(User).where(User.username == a.username)).first():
            sys.exit("user exists")
        u = User(username=a.username, password_hash=hash_password(pw), role=a.role)
        s.add(u)
        s.flush()
        audit.append(s, "user_created", user_id=None, role="cli", object_type="user", object_id=u.id,
                     details={"username": a.username, "role": a.role})
    print(f"created {a.username} ({a.role})")


def cmd_atms_sync(a):
    from .api.atm_files import sync_atm_files
    print(sync_atm_files())


def cmd_verify_audit(a):
    from . import audit
    from .db.session import init_db, read_session
    init_db()
    with read_session() as s:
        print(json.dumps(audit.verify(s), indent=1))


def main(argv=None):
    logging.basicConfig(level=logging.WARNING)
    ap = argparse.ArgumentParser(prog="atmcctv")
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("analyse", help="run ingest -> coarse -> link -> dense -> score on one clip")
    p.add_argument("video")
    p.add_argument("--out", default="data/cli_out")
    p.add_argument("--txn-time", help="journal time ISO, e.g. 2026-03-03T08:42:10")
    p.add_argument("--offset", type=float, default=0.0, help="clock offset DVR - journal (s)")
    p.add_argument("--camera", default="fascia", choices=["fascia", "lobby"])
    p.add_argument("--tz", default="UTC")
    p.add_argument("--start", help="manual clip start time ISO (overrides metadata)")
    p.add_argument("--top-k", type=int, default=6)
    p.set_defaults(fn=cmd_analyse)
    p = sp.add_parser("bench")
    p.add_argument("video")
    p.add_argument("--out", default="data/bench_out")
    p.add_argument("--camera", default="fascia")
    p.set_defaults(fn=cmd_bench)
    sp.add_parser("init-db").set_defaults(fn=cmd_init_db)
    p = sp.add_parser("create-user")
    p.add_argument("username")
    p.add_argument("--role", default="investigator")
    p.add_argument("--password")
    p.set_defaults(fn=cmd_create_user)
    sp.add_parser("atms-sync").set_defaults(fn=cmd_atms_sync)
    sp.add_parser("verify-audit").set_defaults(fn=cmd_verify_audit)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
