"""Ingest (probe, validate, normalise, proxy) and frame access.

Every FFmpeg/ffprobe call uses an argument list (no shell), a timeout and
`-protocol_whitelist file,pipe` so a crafted file cannot make FFmpeg open URLs
or other local files.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from . import dvr
from .types import IngestError, IngestResult, ProbeResult

log = logging.getLogger(__name__)

WHITELIST = ["-protocol_whitelist", "file,pipe"]
REJECT_FORMATS = {"hls", "applehttp", "concat", "ffconcat", "image2", "image2pipe", "tee", "sdp", "rtp", "rtsp"}


class FFmpegMissing(RuntimeError):
    pass


def _bin(name: str) -> str:
    p = shutil.which(name)
    if not p:
        raise FFmpegMissing(f"{name} not found on PATH")
    return p


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _run(args: list[str], timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(args, shell=False, capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL)


def _fraction(s: str | None) -> float:
    if not s or s in ("0/0", "N/A"):
        return 0.0
    if "/" in s:
        n, d = s.split("/", 1)
        try:
            return float(n) / float(d) if float(d) else 0.0
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _rotation(stream: dict) -> int:
    rot = 0.0
    for sd in stream.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = -float(sd["rotation"])  # display matrix is counter-clockwise; store clockwise
            break
    else:
        tag = (stream.get("tags") or {}).get("rotate")
        if tag:
            rot = float(tag)
    return int(round(rot)) % 360 // 90 * 90


def ffprobe_json(path: Path, timeout: float, input_args: list[str] | None = None) -> dict | None:
    args = [_bin("ffprobe"), "-v", "error", *WHITELIST, *(input_args or []), "-show_format", "-show_streams",
            "-of", "json", str(path)]
    try:
        r = _run(args, timeout)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return None


def trial_decode(path: Path, n: int = 50, timeout: float = 60, input_args: list[str] | None = None) -> int:
    """Decode up to n video frames in a subprocess; return how many decoded."""
    args = [_bin("ffmpeg"), "-nostdin", "-v", "error", *WHITELIST, *(input_args or []), "-i", str(path),
            "-map", "0:v:0", "-frames:v", str(n), "-f", "framemd5", "-"]
    try:
        r = _run(args, timeout)
    except subprocess.TimeoutExpired:
        return 0
    lines = [ln for ln in r.stdout.decode("utf-8", "replace").splitlines() if ln and not ln.startswith("#")]
    return len(lines)


def packet_pts(path: Path, n: int = 300, timeout: float = 30, input_args: list[str] | None = None) -> list[float]:
    args = [_bin("ffprobe"), "-v", "error", *WHITELIST, *(input_args or []), "-select_streams", "v:0",
            "-read_intervals", f"%+#{n}", "-show_entries", "packet=pts_time", "-of", "csv=p=0", str(path)]
    try:
        r = _run(args, timeout)
    except subprocess.TimeoutExpired:
        return []
    out = []
    for ln in r.stdout.decode("utf-8", "replace").splitlines():
        ln = ln.strip().strip(",")
        try:
            out.append(float(ln))
        except ValueError:
            pass
    return sorted(out)


def _is_vfr(r_fps: float, avg_fps: float, pts: list[float]) -> bool:
    if r_fps > 0 and avg_fps > 0 and abs(r_fps - avg_fps) / r_fps > 0.01:
        return True
    if len(pts) > 10:
        d = np.diff(np.asarray(pts))
        d = d[d > 0]
        if len(d) > 5:
            med = float(np.median(d))
            if med > 0 and float(np.percentile(np.abs(d - med), 95)) / med > 0.25:
                return True
    return False


def probe(path: Path, cfg, sha: str | None = None) -> tuple[ProbeResult | None, IngestError | None, list[str]]:
    """Inspect a file. Returns (probe, error, input_args used for decoding)."""
    path = Path(path)
    size = path.stat().st_size
    sha = sha or sha256_file(path)
    vendor, head = dvr.sniff_vendor(path)
    timeout = float(cfg.ingest.ffprobe_timeout_s)

    if dvr.is_playlist(head, path):
        return None, IngestError("PLAYLIST_REJECTED", "Playlists and concat lists are not accepted; upload the video file itself.",
                                 None, ["Upload a single MP4/AVI/MKV/MOV file exported from the recorder."]), []
    if size > float(cfg.ingest.max_upload_gb) * (1 << 30):
        return None, IngestError("TOO_LARGE", f"File exceeds {cfg.ingest.max_upload_gb} GB.", vendor,
                                 ["Split the export into shorter segments around the transaction time."]), []

    input_args: list[str] = []
    info = ffprobe_json(path, timeout)
    raw_stream = None
    if info and info.get("format", {}).get("format_name") in ("h264", "hevc"):
        raw_stream = info["format"]["format_name"]
        vendor = vendor or "raw"
    if (not info or not any(s.get("codec_type") == "video" for s in info.get("streams", []))) and vendor == "raw":
        raw_stream = dvr.raw_codec_guess(head)
        input_args = ["-f", raw_stream, "-framerate", str(cfg.ingest.raw_fps)]
        info = ffprobe_json(path, timeout, input_args)
    if raw_stream and not input_args:
        input_args = ["-f", raw_stream, "-framerate", str(cfg.ingest.raw_fps)]

    if not info:
        if vendor in ("dahua", "hikvision", "self_player", "generic_dvr"):
            return None, IngestError("ENCRYPTED_OR_PROPRIETARY", f"FFmpeg cannot read this {vendor} file.", vendor,
                                     dvr.instructions_for(vendor)), []
        return None, IngestError("UNSUPPORTED_CONTAINER", "The file is not a recognised video container.", vendor,
                                 dvr.instructions_for(vendor)), []

    fmt = info.get("format", {})
    format_name = fmt.get("format_name", "")
    if any(f in REJECT_FORMATS for f in format_name.split(",")):
        return None, IngestError("PLAYLIST_REJECTED", f"Format '{format_name}' is not a single-file container.", vendor,
                                 ["Upload a single MP4/AVI/MKV/MOV file."]), []
    if format_name == "dhav":
        vendor = "dahua"

    vstreams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"
                and not (s.get("disposition") or {}).get("attached_pic")]
    if not vstreams:
        return None, IngestError("NO_VIDEO_STREAM", "The file has no video stream.", vendor,
                                 dvr.instructions_for(vendor)), []
    vs = vstreams[0]
    problems: list[str] = []
    r_fps = _fraction(vs.get("r_frame_rate"))
    avg_fps = _fraction(vs.get("avg_frame_rate"))
    pts = packet_pts(path, 300, timeout, input_args)
    vfr = _is_vfr(r_fps, avg_fps, pts) if not raw_stream else False
    field_order = vs.get("field_order")
    interlaced = field_order not in (None, "progressive", "unknown")
    duration = _fraction(vs.get("duration")) or _fraction(fmt.get("duration"))
    n_frames = int(vs["nb_frames"]) if str(vs.get("nb_frames", "")).isdigit() else None
    if not duration and n_frames and (avg_fps or r_fps):
        duration = n_frames / (avg_fps or r_fps)
    ct = (fmt.get("tags") or {}).get("creation_time") or (vs.get("tags") or {}).get("creation_time")
    creation = None
    if ct:
        try:
            creation = datetime.fromisoformat(ct.replace("Z", "+00:00"))
        except ValueError:
            problems.append(f"unparseable creation_time {ct!r}")
    decoded = trial_decode(path, 50, max(60.0, timeout), input_args)
    if not duration:
        problems.append("duration unknown (broken index)")
    if vfr:
        problems.append("variable frame rate")
    if interlaced:
        problems.append(f"interlaced ({field_order})")

    p = ProbeResult(
        path=path, sha256=sha, size=size, format_name=format_name, video_codec=vs.get("codec_name", "?"),
        profile=vs.get("profile"), width=int(vs.get("width") or 0), height=int(vs.get("height") or 0),
        rotation_deg=_rotation(vs), r_fps=r_fps, avg_fps=avg_fps, is_vfr=vfr, is_interlaced=interlaced,
        duration_s=float(duration or 0.0), n_frames=n_frames,
        has_audio=any(s.get("codec_type") == "audio" for s in info.get("streams", [])),
        creation_time=creation, decodable=decoded >= 10, decoded_test_frames=decoded, vendor=vendor,
        problems=problems, raw_stream=raw_stream,
    )
    return p, None, input_args


def _strip_to_annexb(path: Path, out: Path) -> bool:
    off = dvr.first_annexb_offset(path)
    if off <= 0:
        return False
    with open(path, "rb") as src, open(out, "wb") as dst:
        src.seek(off)
        shutil.copyfileobj(src, dst, 1 << 20)
    return True


def _ffmpeg(args: list[str], timeout: float) -> tuple[bool, str]:
    try:
        r = _run([_bin("ffmpeg"), "-nostdin", "-y", "-v", "error", *args], timeout)
    except subprocess.TimeoutExpired:
        return False, "ffmpeg timed out"
    return r.returncode == 0, r.stderr.decode("utf-8", "replace")[-2000:]


def normalise(src: Path, dst: Path, p: ProbeResult, input_args: list[str], cfg) -> tuple[bool, str]:
    fps = p.r_fps if 1 <= p.r_fps <= 120 else (p.avg_fps if 1 <= p.avg_fps <= 120 else float(cfg.ingest.raw_fps))
    vf = ["yadif"] if p.is_interlaced else []
    args = [*WHITELIST, *input_args, "-i", str(src)]
    if vf:
        args += ["-vf", ",".join(vf)]
    args += ["-map", "0:v:0", "-fps_mode", "cfr", "-r", f"{fps:.6f}", "-c:v", "libx264", "-crf", str(cfg.ingest.normalise_crf),
             "-preset", "fast", "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", str(dst)]
    return _ffmpeg(args, float(cfg.ingest.ffmpeg_timeout_s))


def make_proxy(src: Path, dst: Path, cfg) -> tuple[bool, str]:
    h = int(cfg.ingest.proxy_max_height)
    common = [*WHITELIST, "-i", str(src), "-map", "0:v:0", "-vf", f"scale=-2:'min({h},ih)'", "-an",
              "-movflags", "+faststart"]
    ok, err = _ffmpeg([*common, "-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23", "-pix_fmt", "yuv420p", str(dst)],
                      float(cfg.ingest.ffmpeg_timeout_s))
    if ok:
        return ok, err
    log.warning("h264_nvenc proxy failed (%s); falling back to libx264", err.strip()[:200])
    return _ffmpeg([*common, "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p", str(dst)],
                   float(cfg.ingest.ffmpeg_timeout_s))


def ingest(original: Path, workdir: Path, cfg, sha: str | None = None) -> IngestResult:
    """Probe, decide, normalise if needed, and always build a browser proxy."""
    workdir.mkdir(parents=True, exist_ok=True)
    decisions: list[str] = []
    p, err, input_args = probe(original, cfg, sha)
    if err:
        return IngestResult(None, None, None, False, None, None, err, decisions)
    assert p is not None
    src = original

    if not p.decodable:
        retried = False
        if p.vendor in ("dahua", "hikvision", "generic_dvr") or p.vendor is None:
            stripped = workdir / "stripped.bin"
            if _strip_to_annexb(original, stripped):
                head = stripped.open("rb").read(4096)
                codec = dvr.raw_codec_guess(head)
                ia = ["-f", codec, "-framerate", str(cfg.ingest.raw_fps)]
                n = trial_decode(stripped, 50, 60, ia)
                decisions.append(f"header strip at first Annex-B start code, {n} frames decodable as {codec}")
                if n >= 10:
                    src, input_args, retried = stripped, ia, True
                    p.decodable, p.decoded_test_frames, p.raw_stream = True, n, codec
        if not retried:
            code = "ENCRYPTED_OR_PROPRIETARY" if p.vendor in ("dahua", "hikvision", "self_player", "generic_dvr") else "CORRUPT"
            msg = (f"No decodable video frames ({p.decoded_test_frames} decoded in trial)."
                   + (f" Detected vendor: {p.vendor}." if p.vendor else ""))
            return IngestResult(p, None, None, False, None, None,
                                IngestError(code, msg, p.vendor, dvr.instructions_for(p.vendor)), decisions)

    reasons = []
    if p.is_vfr:
        reasons.append("VFR")
    if p.is_interlaced:
        reasons.append("interlaced")
    if p.raw_stream:
        reasons.append("raw elementary stream")
    if p.vendor in ("dahua", "hikvision") or p.format_name == "dhav":
        reasons.append("vendor container")
    if not p.duration_s:
        reasons.append("broken index")
    if p.video_codec not in ("h264", "hevc", "mpeg4", "mjpeg", "wmv2", "wmv3", "mpeg2video", "vp9", "av1", "flv1",
                             "msmpeg4v2", "msmpeg4v3", "vc1", "h263"):
        reasons.append(f"uncommon codec {p.video_codec}")

    if reasons:
        analysis = workdir / "analysis.mp4"
        ok, msg = normalise(src, analysis, p, input_args, cfg)
        decisions.append(f"normalised to H.264 CFR MP4 because: {', '.join(reasons)}")
        if not ok:
            return IngestResult(p, None, None, False, None, None,
                                IngestError("CORRUPT", f"Normalisation failed: {msg.strip()[:300]}", p.vendor,
                                            dvr.instructions_for(p.vendor)), decisions)
        normalised = True
        if src != original and src.exists():
            src.unlink(missing_ok=True)
    else:
        analysis = original
        normalised = False
        decisions.append("decoded as is (no normalisation needed)")

    proxy = workdir / "proxy.mp4"
    ok, msg = make_proxy(analysis, proxy, cfg)
    if not ok:
        return IngestResult(p, analysis, sha256_file(analysis) if normalised else p.sha256, normalised, None, None,
                            IngestError("CORRUPT", f"Proxy build failed: {msg.strip()[:300]}", p.vendor,
                                        dvr.instructions_for(p.vendor)), decisions)
    decisions.append("browser proxy built (audio removed)")
    return IngestResult(p, analysis, sha256_file(analysis) if normalised else p.sha256, normalised, proxy,
                        sha256_file(proxy), None, decisions)


# --------------------------------------------------------------------------- frames

def _rotate(img: np.ndarray, deg: int) -> np.ndarray:
    if deg == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


class FrameSource:
    """PTS-based frame access with PyAV. All timing uses frame.pts * time_base."""

    def __init__(self, path: str | Path, rotation_deg: int = 0):
        import av

        self.path = str(path)
        self.rotation = rotation_deg
        self._av = av
        self.container = av.open(self.path)
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"
        self.tb = float(self.stream.time_base)
        self.fps = float(self.stream.average_rate or self.stream.guessed_rate or 25)
        self.duration = float(self.stream.duration * self.stream.time_base) if self.stream.duration else (
            float(self.container.duration) / 1e6 if self.container.duration else 0.0)
        self.start_time = float(self.stream.start_time * self.stream.time_base) if self.stream.start_time else 0.0
        w, h = self.stream.codec_context.width, self.stream.codec_context.height
        self.width, self.height = (h, w) if rotation_deg in (90, 270) else (w, h)

    def close(self) -> None:
        try:
            self.container.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def _reopen(self):
        self.close()
        self.container = self._av.open(self.path)
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"

    def _pts(self, frame) -> float:
        if frame.pts is not None:
            return frame.pts * self.tb
        return float(frame.time or 0.0)

    def _decode_from(self, t0: float):
        self._reopen()
        if t0 > 0:
            self.container.seek(int(max(0.0, t0) / self.tb), stream=self.stream, backward=True, any_frame=False)
        return self.container.decode(self.stream)

    def iter_sampled(self, fps: float, t0: float = 0.0, t1: float | None = None) -> Iterator[tuple[float, int, np.ndarray]]:
        step = 1.0 / fps
        next_t = t0
        idx = -1
        for frame in self._decode_from(max(0.0, t0 - 2.0) if t0 > 0 else 0.0):
            idx += 1
            pts = self._pts(frame)
            if pts < t0 - 1e-6:
                continue
            if t1 is not None and pts > t1:
                break
            if pts + 1e-6 >= next_t:
                next_t += step
                while next_t <= pts:
                    next_t += step
                yield pts, idx, _rotate(frame.to_ndarray(format="bgr24"), self.rotation)

    def iter_all(self, t0: float = 0.0, t1: float | None = None, step: int = 1) -> Iterator[tuple[float, int, np.ndarray]]:
        idx = -1
        kept = 0
        for frame in self._decode_from(max(0.0, t0 - 2.0) if t0 > 0 else 0.0):
            idx += 1
            pts = self._pts(frame)
            if pts < t0 - 1e-6:
                continue
            if t1 is not None and pts > t1:
                break
            kept += 1
            if (kept - 1) % step:
                continue
            yield pts, idx, _rotate(frame.to_ndarray(format="bgr24"), self.rotation)

    def frame_index_of(self, pts: float) -> int:
        return int(round((pts - self.start_time) * self.fps))

    def get(self, pts: float) -> np.ndarray | None:
        best = None
        half = 0.5 / max(self.fps, 1.0)
        for frame in self._decode_from(pts):
            fp = self._pts(frame)
            best = frame
            if fp + half >= pts:
                break
        if best is None:
            return None
        return _rotate(best.to_ndarray(format="bgr24"), self.rotation)


def luma(img: np.ndarray) -> np.ndarray:
    """Luma in [0,1] (Rec.601 via OpenCV)."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
