"""Turn a recording from record_demo.py into docs/demo/demo.mp4 and docs/demo/voiceover.md.

Real time is kept 1:1 except the GPU waits, which are compressed to a few seconds each and
marked "FAST-FORWARD" on screen.

    .venv\\Scripts\\python.exe scripts\\demo\\build_demo.py [--rec DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "demo"
FPS = 30
FONT = "C\\:/Windows/Fonts/segoeuib.ttf"
WPS = 2.6  # speaking rate used to flag voice-over lines that will not fit


def build_map(t0: float, t1: float, fast: list[dict]):
    """Piecewise-linear map from wall time to output seconds."""
    pts = [(t0, 0.0)]
    out = 0.0
    cur = t0
    for f in sorted(fast, key=lambda f: f["start"]):
        a, b = max(f["start"], cur), min(f["end"], t1)
        if b <= a:
            continue
        out += a - cur
        pts.append((a, out))
        out += f["out"]
        pts.append((b, out))
        cur = b
    out += t1 - cur
    pts.append((t1, out))

    def m(t: float) -> float:
        t = min(max(t, t0), t1)
        for (xa, ya), (xb, yb) in zip(pts, pts[1:]):
            if t <= xb:
                return ya if xb == xa else ya + (t - xa) * (yb - ya) / (xb - xa)
        return pts[-1][1]

    return m, out


def fmt(s: float) -> str:
    return f"{int(s // 60)}:{s % 60:04.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", default=str(Path(os.environ.get("TEMP", "/tmp")) / "atmdemo" / "rec"))
    a = ap.parse_args()
    rec = Path(a.rec)
    tl = json.loads((rec / "timeline.json").read_text())
    t0, t1 = tl["marks"]["start"], tl["marks"]["end"]
    m, total = build_map(t0, t1, tl["fast"])
    frames = sorted(tl["frames"])

    # concat list: each frame lasts until the next one (in output time)
    lines, used = [], 0
    for i, (ts, name) in enumerate(frames):
        nxt = frames[i + 1][0] if i + 1 < len(frames) else t1
        if nxt <= t0 or ts >= t1:
            continue
        dur = m(nxt) - m(max(ts, t0))
        if dur <= 0:
            continue
        p = (rec / "frames" / name).as_posix()
        lines += [f"file '{p}'", f"duration {dur:.5f}"]
        used += 1
        last = p
    lines.append(f"file '{last}'")
    (rec / "concat.txt").write_text("\n".join(lines) + "\n")

    # FAST-FORWARD badge during compressed waits
    draws = []
    for f in tl["fast"]:
        a_, b_ = m(f["start"]), m(f["end"])
        if b_ - a_ < 0.5:
            continue
        x = (f["end"] - f["start"]) / f["out"]
        en = f"enable='between(t,{a_:.3f},{b_:.3f})'"
        draws.append(f"drawbox=x=(iw-300)/2:y=18:w=300:h=46:color=0x0B1220@0.88:t=fill:{en}")
        draws.append(f"drawtext=fontfile='{FONT}':text='FAST-FORWARD  {x:.0f}x':fontcolor=0x38BDF8:fontsize=24:"
                     f"x=(w-tw)/2:y=29:{en}")
    vf = ",".join(["scale=1920:1080:force_original_aspect_ratio=decrease", "pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
                   f"fps={FPS}", *draws, f"fade=t=in:st=0:d=0.6", f"fade=t=out:st={max(0, total - 0.8):.3f}:d=0.8",
                   "format=yuv420p"])
    OUT.mkdir(parents=True, exist_ok=True)
    mp4 = OUT / "demo.mp4"
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(rec / "concat.txt"),
           "-vf", vf, "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-tune", "stillimage",
           "-movflags", "+faststart", str(mp4)]
    subprocess.run(cmd, check=True)

    # voice-over script
    caps = tl["captions"]
    rows = []
    for i, c in enumerate(caps):
        start = m(c["t"])
        end = m(caps[i + 1]["t"]) if i + 1 < len(caps) else total
        words = len(c["vo"].split())
        need = words / WPS
        rows.append((start, end, c, need))
    md = ["# Demo voice-over", "",
          f"Video: `docs/demo/demo.mp4` · {fmt(total)} · 1920×1080 · {FPS} fps, no audio.", "",
          "Start each line at its timestamp. \"On screen\" is the caption label visible at that moment.",
          "Sped-up GPU waits are marked FAST-FORWARD in the video.", "",
          "| Time | On screen | Say |", "|---|---|---|"]
    for start, end, c, need in rows:
        if not c["vo"]:
            continue
        screen = re.sub("<[^>]+>", "", c["kicker"] or c["text"])
        md.append(f"| {fmt(start)} | {screen} | {c['vo']} |")
        if need > end - start + 0.3:
            print(f"  VO may not fit at {fmt(start)}: needs {need:.1f}s, has {end - start:.1f}s")
    (OUT / "voiceover.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    for name in ("evidence_pages",):
        if (rec / name).exists():
            shutil.copytree(rec / name, OUT / name, dirs_exist_ok=True)
    print(f"frames used {used}/{len(frames)}  duration {fmt(total)}  -> {mp4}")


if __name__ == "__main__":
    main()
