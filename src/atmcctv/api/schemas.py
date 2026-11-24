"""Pydantic request/response schemas (also drive the generated TypeScript client types)."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

Role = Literal["viewer", "investigator", "admin"]


class ErrorOut(BaseModel):
    code: str
    message: str
    details: dict = {}


class LoginIn(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class TokenOut(BaseModel):
    access_token: str
    role: Role
    expires_at: datetime
    username: str
    user_id: int


class MediaTokenOut(BaseModel):
    token: str
    expires_at: datetime


class UserOut(BaseModel):
    id: int
    username: str
    role: Role
    is_active: bool
    created_at: datetime


class UserIn(BaseModel):
    username: str = Field(min_length=3, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(min_length=10, max_length=256)
    role: Role = "investigator"


class UserPatch(BaseModel):
    role: Optional[Role] = None
    is_active: Optional[bool] = None
    password: Optional[str] = Field(default=None, min_length=10, max_length=256)


class CaseIn(BaseModel):
    ref_no: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=200)


class CaseOut(BaseModel):
    id: str
    ref_no: str
    title: str
    status: str
    created_by: Optional[int]
    created_at: datetime
    member_ids: list[int] = []
    n_clips: int = 0


class MembersIn(BaseModel):
    user_ids: list[int]


Point = list[float]


class AtmIn(BaseModel):
    atm_code: str = Field(min_length=1, max_length=64)
    location: str = ""
    camera_type: Literal["fascia", "lobby"] = "fascia"
    rois: dict[str, list[Point]] = {}
    ocr_roi: Optional[list[float]] = None
    date_order: Literal["DMY", "MDY", "YMD"] = "DMY"
    tz: str = "Asia/Kolkata"
    metadata_is_end: bool = False

    @field_validator("rois")
    @classmethod
    def _rois(cls, v):
        allowed = {"keypad", "cash_slot", "card_slot", "atm_zone"}
        for k, poly in v.items():
            if k not in allowed:
                raise ValueError(f"unknown ROI {k}; allowed {sorted(allowed)}")
            if poly and len(poly) < 3:
                raise ValueError(f"ROI {k} needs at least 3 points")
            for p in poly:
                if len(p) != 2 or not all(0.0 <= c <= 1.0 for c in p):
                    raise ValueError("ROI points must be normalised [x, y] in 0..1")
        return v

    @field_validator("ocr_roi")
    @classmethod
    def _ocr(cls, v):
        if v is None:
            return v
        if len(v) != 4 or not all(0.0 <= c <= 1.0 for c in v) or v[2] <= v[0] or v[3] <= v[1]:
            raise ValueError("ocr_roi must be [x0, y0, x1, y1] normalised with x1>x0, y1>y0")
        return v

    @field_validator("tz")
    @classmethod
    def _tz(cls, v):
        from zoneinfo import ZoneInfo
        ZoneInfo(v)
        return v


class AtmOut(BaseModel):
    id: int
    atm_code: str
    location: str
    camera_type: str
    rois: dict
    ocr_roi: Optional[list[float]]
    date_order: str
    tz: str
    metadata_is_end: bool


class JobOut(BaseModel):
    id: int
    type: str
    case_id: Optional[str]
    payload: dict
    result: Optional[dict]
    status: str
    progress: float
    message: str
    created_at: datetime
    started_at: Optional[datetime]
    finished_at: Optional[datetime]


class ClipOut(BaseModel):
    id: str
    case_id: str
    atm_id: Optional[int]
    kind: str
    orig_name: str
    orig_sha256: str
    size: int
    analysis_sha256: Optional[str]
    proxy_sha256: Optional[str]
    normalised: bool
    status: str
    probe: Optional[dict]
    ingest: Optional[dict]
    error: Optional[dict]
    timemap: Optional[dict]
    timemap_reads: Optional[list] = None
    clock_offset_s: float
    tracks_meta: Optional[dict]
    created_at: datetime
    n_tracks: int = 0
    proxy_url: Optional[str] = None
    start_wall: Optional[str] = None


class TimemapIn(BaseModel):
    method: Optional[Literal["manual", "ocr", "metadata"]] = None
    start: Optional[datetime] = None          # manual: wall time (DVR clock) at the first frame
    roi: Optional[list[float]] = None         # ocr: [x0,y0,x1,y1] normalised
    clock_offset_s: Optional[float] = Field(default=None, ge=-86400, le=86400)


class TxnIn(BaseModel):
    txn_ref: str = Field(default="", max_length=64)
    txn_time: datetime                         # bank journal time (local, ATM tz)
    amount: Optional[float] = None
    window_before_s: float = Field(default=90, ge=0, le=3600)
    window_after_s: float = Field(default=30, ge=0, le=3600)


class TxnOut(BaseModel):
    id: int
    case_id: str
    clip_id: str
    txn_ref: str
    txn_time_journal: datetime
    amount: Optional[float]
    window_before_s: float
    window_after_s: float
    t_txn_pts: Optional[float]
    status: str
    chosen_track_id: Optional[int]
    confirmed_by: Optional[int]
    confirmed_at: Optional[datetime]
    notes: list = []
    clip_name: Optional[str] = None


class CandidateOut(BaseModel):
    id: int
    track_id: int
    track_no: int
    source: str
    rank: int
    score: float
    parts: dict
    reasons: list
    best_view_pts: float
    thumb_url: Optional[str]
    t_start: float
    t_end: float


class TrackOut(BaseModel):
    id: int
    clip_id: str
    track_no: int
    source: str
    t_start: float
    t_end: float
    n_samples: int
    face_visible_frac: Optional[float]
    dense_status: str
    indexed: bool
    stats: Optional[dict]


class TrackSamplesOut(BaseModel):
    track_id: int
    track_no: int
    pts: list[float]
    bbox: list[list[float]]
    near: list[bool]


class FaceOut(BaseModel):
    id: int
    track_id: int
    clip_id: str
    frame_idx: int
    pts: float
    wall_time: Optional[str]
    bbox: list
    total: float
    grade: str
    sub: dict
    raw: dict
    flags: list
    caps: list
    attrib: dict
    searchable: bool
    selected: bool
    purpose: str
    detview_corrected: bool
    has_embedding: bool
    frame_sha256: Optional[str]
    crop_sha256: Optional[str]
    aligned_sha256: Optional[str]
    urls: dict[str, str]
    enhancements: list = []


class FaceBriefOut(BaseModel):
    id: int
    pts: float
    wall_time: Optional[str]
    total: float
    grade: str
    flags: list[str]
    searchable: bool
    crop_url: str
    aligned_url: str
    track_id: int
    clip_id: str


class EnhanceIn(BaseModel):
    method: Literal["clahe", "gamma", "gfpgan_v1.4_onnx"]
    params: dict[str, float] = {}


class EnhancementOut(BaseModel):
    id: int
    face_id: int
    method: str
    params: dict
    sha256: Optional[str]
    url: Optional[str]
    created_at: datetime
    job_id: Optional[int] = None


class GalleryIn(BaseModel):
    clip_ids: list[str]


class SearchIn(BaseModel):
    case_id: str
    face_ids: list[int] = Field(min_length=1, max_length=20)


class HitOut(BaseModel):
    id: int
    track_id: int
    track_no: int
    clip_id: str
    clip_name: str
    case_id: str
    score: float
    band: str
    parts: dict[str, Optional[float]]
    faces: list[FaceBriefOut]
    review: Optional[str]
    reviewed_by: Optional[int]
    reviewed_at: Optional[datetime]
    note: Optional[str]


class SearchOut(BaseModel):
    search_id: int
    case_id: str
    query_faces: list[FaceBriefOut]
    bands: dict[str, float]
    calibrated: bool
    hits: list[HitOut]
    not_comparable: list[dict]
    created_at: datetime


class ReviewIn(BaseModel):
    decision: Literal["confirmed", "rejected"]
    note: str = Field(default="", max_length=2000)


class BehaviourOut(BaseModel):
    id: int
    clip_id: str
    type: str
    t_start: float
    t_end: float
    track_ids: list
    details: dict


class ExportIn(BaseModel):
    face_ids: list[int] = []
    search_hit_ids: list[int] = []
    include_enhanced: bool = False


class ExportOut(BaseModel):
    id: int
    case_id: str
    sha256: Optional[str]
    selection: dict
    status: str
    created_by: Optional[int]
    created_at: datetime
    job_id: Optional[int] = None


class AuditOut(BaseModel):
    id: int
    ts: str
    user_id: Optional[int]
    role: Optional[str]
    action: str
    object_type: Optional[str]
    object_id: Optional[str]
    details: Any
    prev_hash: str
    hash: str


class VerifyOut(BaseModel):
    ok: bool
    checked: int
    first_bad_id: Optional[int]
    reason: Optional[str] = None
    head: Optional[str] = None
