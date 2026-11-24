"""Generate the format-coverage set (§14.3) from one source clip.

  python scripts/make_formats.py [--src samples/kaggle_atm_video/normal_1.mp4] [--out samples/formats]

Writes MP4, AVI (MPEG-4 Part 2 and MJPEG), MOV, MKV, WMV (WMV2), FLV, MPG (MPEG-2), raw .264, a VFR copy,
an interlaced copy, a truncated file, a renamed non-video file and an HLS playlist, plus expected.json
with the expected ingest outcome for each (used by scripts/evaluate.py --ingest).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", *args], check=True, timeout=600)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "samples" / "kaggle_atm_video" / "normal_1.mp4"))
    ap.add_argument("--out", default=str(ROOT / "samples" / "formats"))
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    small = ["-vf", "scale=960:-2", "-t", "6"]
    exp: dict[str, dict] = {}

    def make(name: str, args: list[str], accept=True, normalised=None, code=None):
        ff("-i", str(src), *small, *args, str(out / name))
        exp[name] = {"accept": accept, "normalised": normalised, "code": code}

    make("mp4_h264.mp4", ["-c:v", "libx264", "-an"], normalised=False)
    make("avi_mpeg4.avi", ["-c:v", "mpeg4", "-q:v", "4", "-an"], normalised=False)
    make("avi_mjpeg.avi", ["-c:v", "mjpeg", "-q:v", "5", "-an"], normalised=False)
    make("mov_h264.mov", ["-c:v", "libx264", "-an"], normalised=False)
    make("mkv_h264.mkv", ["-c:v", "libx264", "-an"], normalised=False)
    make("wmv_wmv2.wmv", ["-c:v", "wmv2", "-b:v", "2M", "-an"], normalised=False)
    make("flv_flv1.flv", ["-c:v", "flv", "-b:v", "2M", "-an"], normalised=False)
    make("mpg_mpeg2.mpg", ["-c:v", "mpeg2video", "-b:v", "4M", "-an"], normalised=None)
    make("raw_h264.264", ["-c:v", "libx264", "-bsf:v", "h264_mp4toannexb", "-an", "-f", "h264"], normalised=True)
    # VFR: drop frames irregularly, keep timestamps
    ff("-i", str(src), "-t", "6", "-vf", "scale=960:-2,select='not(mod(n\\,3))+not(mod(n\\,7))',setpts=N/FRAME_RATE/TB",
       "-fps_mode", "vfr", "-c:v", "libx264", "-an", str(out / "vfr.mkv"))
    ff("-i", str(src), "-t", "6", "-vf", "select='if(lt(mod(n\\,10)\\,5)\\,1\\,not(mod(n\\,3)))'",
       "-fps_mode", "vfr", "-c:v", "libx264", "-an", str(out / "vfr_select.mp4"))
    exp["vfr.mkv"] = {"accept": True, "normalised": None, "code": None}
    exp["vfr_select.mp4"] = {"accept": True, "normalised": True, "code": None}
    ff("-i", str(src), "-t", "6", "-vf", "scale=720:576,interlace", "-c:v", "libx264", "-flags", "+ilme+ildct",
       "-x264opts", "tff=1", "-an", str(out / "interlaced.mkv"))
    exp["interlaced.mkv"] = {"accept": True, "normalised": True, "code": None}
    data = (out / "mp4_h264.mp4").read_bytes()
    (out / "truncated.mp4").write_bytes(data[: len(data) // 50])
    exp["truncated.mp4"] = {"accept": False, "normalised": None, "code": ["CORRUPT", "UNSUPPORTED_CONTAINER", "NO_VIDEO_STREAM"]}
    (out / "not_a_video.mp4").write_bytes(b"This is a text file renamed to .mp4\n" * 200)
    exp["not_a_video.mp4"] = {"accept": False, "normalised": None, "code": ["UNSUPPORTED_CONTAINER", "CORRUPT"]}
    (out / "playlist.m3u8").write_text("#EXTM3U\n#EXT-X-VERSION:3\n#EXTINF:6.0,\nmp4_h264.mp4\n#EXT-X-ENDLIST\n")
    exp["playlist.m3u8"] = {"accept": False, "normalised": None, "code": ["PLAYLIST_REJECTED"]}
    (out / "sneaky_playlist.mp4").write_text("#EXTM3U\n#EXTINF:6.0,\nfile:///C:/Windows/win.ini\n")
    exp["sneaky_playlist.mp4"] = {"accept": False, "normalised": None, "code": ["PLAYLIST_REJECTED"]}
    (out / "expected.json").write_text(json.dumps(exp, indent=1))
    print("wrote", len(exp), "files to", out)


if __name__ == "__main__":
    main()
