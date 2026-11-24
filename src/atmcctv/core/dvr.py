"""Vendor (DVR/NVR) detection and export instructions for proprietary files."""
from __future__ import annotations

from pathlib import Path

ANNEXB = b"\x00\x00\x00\x01"
ANNEXB3 = b"\x00\x00\x01"

# Extensions that are always vendor/self-player formats FFmpeg cannot read reliably.
VENDOR_EXT = {
    ".dav": "dahua",
    ".grd": "generic_dvr",
    ".ssf": "generic_dvr",
    ".exe": "self_player",
    ".h264": "raw",
    ".264": "raw",
    ".h265": "raw",
    ".265": "raw",
    ".hevc": "raw",
    ".mfs": "hikvision",
    ".hik": "hikvision",
}

INSTRUCTIONS: dict[str, list[str]] = {
    "dahua": [
        "Open the file in Dahua SmartPlayer or SmartPSS / DSS Client.",
        "Use 'Export' / 'Convert' and choose MP4 or AVI (H.264).",
        "If the file is encrypted, enter the device password in SmartPlayer before exporting.",
        "Upload the exported MP4/AVI here.",
    ],
    "hikvision": [
        "Open the file in Hikvision VSPlayer or iVMS-4200.",
        "Use 'Convert' / 'Video Clip' and choose MP4 output.",
        "Alternatively, use the NVR's own backup menu and export as MP4.",
        "Upload the converted MP4 here.",
    ],
    "self_player": [
        "This is a self-contained player executable, not a video file. Do not run it on an analysis machine.",
        "Ask the DVR operator to re-export the footage using the recorder's backup menu as AVI or MP4.",
    ],
    "raw": [
        "This looks like a raw H.264/H.265 elementary stream without a container.",
        "If decoding fails, check the frame rate setting for the ATM (ingest.raw_fps) or re-export as MP4.",
    ],
    "generic_dvr": [
        "Use the recorder's own backup/export function and choose AVI or MP4.",
        "If only a vendor player is available, open the file in it and use 'Export' or 'Convert' to MP4/AVI.",
    ],
    "unknown": [
        "The file could not be decoded. Re-export the footage from the recorder's backup menu as AVI or MP4.",
        "If the recorder offers a 'standard format' or 'AVI' option, choose it instead of the native format.",
    ],
}


def sniff_vendor(path: Path) -> tuple[str | None, bytes]:
    """Return (vendor, head bytes). Reads only the first 64 KiB."""
    with open(path, "rb") as f:
        head = f.read(64 * 1024)
    if b"DHAV" in head[:4096] or head.startswith(b"DHAV"):
        return "dahua", head
    if head.startswith(b"IMKH") or b"IMKH" in head[:64]:
        return "hikvision", head
    if head.startswith(ANNEXB) or head.startswith(ANNEXB3):
        return "raw", head
    ext_vendor = VENDOR_EXT.get(path.suffix.lower())
    if ext_vendor:
        return ext_vendor, head
    if head[:2] == b"MZ":
        return "self_player", head
    return None, head


def is_playlist(head: bytes, path: Path) -> bool:
    text = head[:256].lstrip().lower()
    if text.startswith(b"#extm3u") or text.startswith(b"ffconcat") or text.startswith(b"file '"):
        return True
    if path.suffix.lower() in {".m3u8", ".m3u", ".ffconcat", ".concat", ".txt", ".sdp"}:
        return True
    return False


def raw_codec_guess(head: bytes) -> str:
    """Guess h264 vs hevc from the first NAL unit header after an Annex-B start code."""
    i = head.find(ANNEXB3)
    if i < 0 or i + 3 >= len(head):
        return "h264"
    nal = head[i + 3]
    h264_type = nal & 0x1F
    hevc_type = (nal >> 1) & 0x3F
    if h264_type in (7, 8, 5, 1, 6, 9):
        return "h264"
    if hevc_type in (32, 33, 34, 35, 39, 19, 20):
        return "hevc"
    return "h264"


def first_annexb_offset(path: Path, limit: int = 4 * 1024 * 1024) -> int:
    with open(path, "rb") as f:
        buf = f.read(limit)
    i = buf.find(ANNEXB)
    return i


def instructions_for(vendor: str | None) -> list[str]:
    return INSTRUCTIONS.get(vendor or "unknown", INSTRUCTIONS["unknown"])
