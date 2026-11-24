from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np


@dataclass
class ProbeResult:
    path: Path
    sha256: str
    size: int
    format_name: str
    video_codec: str
    profile: str | None
    width: int
    height: int
    rotation_deg: int
    r_fps: float
    avg_fps: float
    is_vfr: bool
    is_interlaced: bool
    duration_s: float
    n_frames: int | None
    has_audio: bool
    creation_time: datetime | None
    decodable: bool
    decoded_test_frames: int
    vendor: str | None
    problems: list[str] = field(default_factory=list)
    raw_stream: str | None = None   # "h264" | "hevc" when the file is an elementary stream

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        d["path"] = str(self.path)
        d["creation_time"] = self.creation_time.isoformat() if self.creation_time else None
        return d


IngestErrorCode = Literal[
    "UNSUPPORTED_CONTAINER", "ENCRYPTED_OR_PROPRIETARY", "NO_VIDEO_STREAM",
    "CORRUPT", "PLAYLIST_REJECTED", "TOO_LARGE",
]


@dataclass
class IngestError:
    code: IngestErrorCode
    message: str
    vendor: str | None
    instructions: list[str]

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class IngestResult:
    probe: ProbeResult | None
    analysis_path: Path | None
    analysis_sha256: str | None
    normalised: bool
    proxy_path: Path | None
    proxy_sha256: str | None
    error: IngestError | None = None
    decisions: list[str] = field(default_factory=list)


@dataclass
class TimeMap:
    method: Literal["ocr", "metadata", "manual"]
    a: float            # wall_epoch_s = a + b * pts_s
    b: float
    tz: str
    inliers: int
    residual_s: float
    notes: list[str] = field(default_factory=list)

    def wall(self, pts: float) -> float:
        return self.a + self.b * pts

    def pts(self, wall_epoch: float) -> float:
        return (wall_epoch - self.a) / self.b

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_json(d: dict | None) -> "TimeMap | None":
        return TimeMap(**d) if d else None


@dataclass
class TrackSample:
    pts: float
    bbox: np.ndarray          # xyxy pixels
    kpts: np.ndarray          # (17, 3) x, y, conf (zeros for face tracks)
    conf: float
    touches_top: bool
    face_bbox: np.ndarray | None = None   # linked face box (fascia mode), xyxy
    face_iod: float = 0.0


@dataclass
class PersonTrack:
    track_id: int
    source: Literal["person", "face"]
    samples: list[TrackSample] = field(default_factory=list)

    @property
    def t_start(self) -> float:
        return self.samples[0].pts if self.samples else 0.0

    @property
    def t_end(self) -> float:
        return self.samples[-1].pts if self.samples else 0.0

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


@dataclass
class Candidate:
    track_id: int
    score: float
    parts: dict[str, float]
    best_view_pts: float
    reasons: list[str] = field(default_factory=list)


@dataclass
class FaceObs:
    track_id: int
    frame_idx: int
    pts: float
    bbox: np.ndarray
    lmk5: np.ndarray
    det_conf: float
    aligned: np.ndarray                 # 112x112 BGR from ORIGINAL pixels
    raw_crop: np.ndarray                # unresized crop + 25% margin, original pixels
    head_out_of_frame: bool
    frame_shape: tuple[int, int] = (0, 0)
    touches_edge: bool = False
    detview_corrected: bool = False
    frame_p95_luma: float = 0.0
    face_crop: np.ndarray | None = None  # square crop (+15%) for pose/parsing/attributes (original pixels)
    head_crop: np.ndarray | None = None  # face box x2.2 for open-vocabulary headwear checks
    face_crop_origin: tuple[int, int] = (0, 0)  # (x0, y0) of face_crop in frame coordinates
    frame: np.ndarray | None = None      # full original frame; kept only for faces that may be selected


@dataclass
class QualityResult:
    total: float
    grade: Literal["A", "B", "C"]
    sub: dict[str, float]
    raw: dict[str, float]
    flags: list[str]
    caps: list[str]
    searchable: bool


@dataclass
class ScoredFace:
    obs: FaceObs
    q: QualityResult
    attrib: dict[str, float] = field(default_factory=dict)
    parse_frac: dict[str, float] = field(default_factory=dict)
