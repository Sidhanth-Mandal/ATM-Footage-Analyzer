"""SQLModel tables (SQLite, WAL)."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import NaiveDatetime
from sqlalchemy import JSON, Column, LargeBinary, UniqueConstraint
from sqlmodel import Field, SQLModel

JSONN = JSON(none_as_null=True)  # store Python None as SQL NULL, not the JSON text 'null'


def now() -> datetime:
    return datetime.now(timezone.utc)


def uid() -> str:
    return str(uuid.uuid4())


ROLES = ("viewer", "investigator", "admin")


class User(SQLModel, table=True):
    __tablename__ = "user"
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    password_hash: str
    role: str = "viewer"
    is_active: bool = True
    failed_logins: int = 0
    locked_until: Optional[datetime] = None
    created_at: datetime = Field(default_factory=now)


class Case(SQLModel, table=True):
    __tablename__ = "case"
    id: str = Field(default_factory=uid, primary_key=True)
    ref_no: str = Field(index=True)
    title: str
    status: str = "open"
    created_by: Optional[int] = Field(default=None, foreign_key="user.id")
    created_at: datetime = Field(default_factory=now)


class CaseMember(SQLModel, table=True):
    __tablename__ = "case_member"
    case_id: str = Field(foreign_key="case.id", primary_key=True)
    user_id: int = Field(foreign_key="user.id", primary_key=True)


class Atm(SQLModel, table=True):
    __tablename__ = "atm"
    id: Optional[int] = Field(default=None, primary_key=True)
    atm_code: str = Field(index=True, unique=True)
    location: str = ""
    camera_type: str = "fascia"          # fascia | lobby
    rois_json: dict = Field(default_factory=dict, sa_column=Column(JSONN))       # name -> [[x,y],...] normalised
    ocr_roi_json: Optional[list] = Field(default=None, sa_column=Column(JSONN))  # [x0,y0,x1,y1] normalised
    date_order: str = "DMY"
    tz: str = "Asia/Kolkata"
    metadata_is_end: bool = False
    reference_frame_path: Optional[str] = None


class Clip(SQLModel, table=True):
    __tablename__ = "clip"
    id: str = Field(default_factory=uid, primary_key=True)
    case_id: str = Field(foreign_key="case.id", index=True)
    atm_id: Optional[int] = Field(default=None, foreign_key="atm.id")
    kind: str = "transaction"            # transaction | gallery
    orig_name: str
    orig_path: str
    orig_sha256: str
    size: int
    analysis_path: Optional[str] = None
    analysis_sha256: Optional[str] = None
    proxy_path: Optional[str] = None
    proxy_sha256: Optional[str] = None
    normalised: bool = False
    probe_json: Optional[dict] = Field(default=None, sa_column=Column(JSONN))
    ingest_json: Optional[dict] = Field(default=None, sa_column=Column(JSONN))
    timemap_json: Optional[dict] = Field(default=None, sa_column=Column(JSONN))
    timemap_reads_json: Optional[list] = Field(default=None, sa_column=Column(JSONN))
    clock_offset_s: float = 0.0
    tracks_meta_json: Optional[dict] = Field(default=None, sa_column=Column(JSONN))
    status: str = "uploaded"             # uploaded|ingesting|ingested|tracking|tracked|indexing|indexed|failed
    error_json: Optional[dict] = Field(default=None, sa_column=Column(JSONN))
    created_by: Optional[int] = Field(default=None, foreign_key="user.id")
    created_at: datetime = Field(default_factory=now)


class Transaction(SQLModel, table=True):
    __tablename__ = "transaction"
    id: Optional[int] = Field(default=None, primary_key=True)
    case_id: str = Field(foreign_key="case.id", index=True)
    clip_id: str = Field(foreign_key="clip.id", index=True)
    txn_ref: str = ""
    txn_time_journal: NaiveDatetime          # bank journal local time (ATM time zone), stored as entered
    amount: Optional[float] = None
    window_before_s: float = 90
    window_after_s: float = 30
    t_txn_pts: Optional[float] = None
    status: str = "pending"              # pending|linked|confirmed|analysed
    chosen_track_id: Optional[int] = Field(default=None, foreign_key="track.id")
    confirmed_by: Optional[int] = Field(default=None, foreign_key="user.id")
    confirmed_at: Optional[datetime] = None
    notes_json: Optional[list] = Field(default=None, sa_column=Column(JSONN))


class Track(SQLModel, table=True):
    __tablename__ = "track"
    __table_args__ = (UniqueConstraint("clip_id", "track_no"),)
    id: Optional[int] = Field(default=None, primary_key=True)
    clip_id: str = Field(foreign_key="clip.id", index=True)
    track_no: int
    source: str = "person"
    t_start: float
    t_end: float
    n_samples: int = 0
    samples_path: str = ""
    face_visible_frac: Optional[float] = None
    dense_status: str = "none"           # none|done
    indexed: bool = False
    stats_json: Optional[dict] = Field(default=None, sa_column=Column(JSONN))


class CandidateRow(SQLModel, table=True):
    __tablename__ = "candidate"
    id: Optional[int] = Field(default=None, primary_key=True)
    transaction_id: int = Field(foreign_key="transaction.id", index=True)
    track_id: int = Field(foreign_key="track.id")
    rank: int
    score: float
    parts_json: dict = Field(default_factory=dict, sa_column=Column(JSONN))
    reasons_json: list = Field(default_factory=list, sa_column=Column(JSONN))
    best_view_pts: float = 0.0
    thumb_path: Optional[str] = None


class FaceRow(SQLModel, table=True):
    __tablename__ = "face"
    id: Optional[int] = Field(default=None, primary_key=True)
    track_id: int = Field(foreign_key="track.id", index=True)
    clip_id: str = Field(foreign_key="clip.id", index=True)
    frame_idx: int
    pts: float
    wall_time: Optional[str] = None
    bbox_json: list = Field(default_factory=list, sa_column=Column(JSONN))
    lmk_json: list = Field(default_factory=list, sa_column=Column(JSONN))
    crop_origin_json: list = Field(default_factory=list, sa_column=Column(JSONN))
    frame_path: Optional[str] = None
    crop_path: Optional[str] = None
    aligned_path: Optional[str] = None
    frame_sha256: Optional[str] = None
    crop_sha256: Optional[str] = None
    aligned_sha256: Optional[str] = None
    total: float = 0.0
    grade: str = "C"
    sub_json: dict = Field(default_factory=dict, sa_column=Column(JSONN))
    raw_json: dict = Field(default_factory=dict, sa_column=Column(JSONN))
    flags_json: list = Field(default_factory=list, sa_column=Column(JSONN))
    caps_json: list = Field(default_factory=list, sa_column=Column(JSONN))
    attrib_json: dict = Field(default_factory=dict, sa_column=Column(JSONN))
    searchable: bool = False
    selected: bool = False
    purpose: str = "best"                # best (dense on confirmed track) | gallery
    detview_corrected: bool = False
    embedding: Optional[bytes] = Field(default=None, sa_column=Column(LargeBinary))


class Template(SQLModel, table=True):
    __tablename__ = "template"
    id: Optional[int] = Field(default=None, primary_key=True)
    track_id: int = Field(foreign_key="track.id", unique=True)
    n_faces: int
    weights_json: list = Field(default_factory=list, sa_column=Column(JSONN))
    face_ids_json: list = Field(default_factory=list, sa_column=Column(JSONN))
    vector: bytes = Field(sa_column=Column(LargeBinary))


class Enhancement(SQLModel, table=True):
    __tablename__ = "enhancement"
    id: Optional[int] = Field(default=None, primary_key=True)
    face_id: int = Field(foreign_key="face.id", index=True)
    method: str
    params_json: dict = Field(default_factory=dict, sa_column=Column(JSONN))
    path: Optional[str] = None
    sha256: Optional[str] = None
    created_by: Optional[int] = Field(default=None, foreign_key="user.id")
    created_at: datetime = Field(default_factory=now)


class Search(SQLModel, table=True):
    __tablename__ = "search"
    id: Optional[int] = Field(default=None, primary_key=True)
    case_id: str = Field(foreign_key="case.id", index=True)
    query_face_ids_json: list = Field(default_factory=list, sa_column=Column(JSONN))
    bands_json: dict = Field(default_factory=dict, sa_column=Column(JSONN))
    not_comparable_json: list = Field(default_factory=list, sa_column=Column(JSONN))
    created_by: Optional[int] = Field(default=None, foreign_key="user.id")
    created_at: datetime = Field(default_factory=now)


class SearchHit(SQLModel, table=True):
    __tablename__ = "search_hit"
    id: Optional[int] = Field(default=None, primary_key=True)
    search_id: int = Field(foreign_key="search.id", index=True)
    track_id: int = Field(foreign_key="track.id")
    score: float
    band: str
    parts_json: dict = Field(default_factory=dict, sa_column=Column(JSONN))
    top_face_ids_json: list = Field(default_factory=list, sa_column=Column(JSONN))
    review: Optional[str] = None          # confirmed | rejected
    reviewed_by: Optional[int] = Field(default=None, foreign_key="user.id")
    reviewed_at: Optional[datetime] = None
    note: Optional[str] = None


class BehaviourEvent(SQLModel, table=True):
    __tablename__ = "behaviour_event"
    id: Optional[int] = Field(default=None, primary_key=True)
    clip_id: str = Field(foreign_key="clip.id", index=True)
    type: str
    t_start: float
    t_end: float
    track_ids_json: list = Field(default_factory=list, sa_column=Column(JSONN))
    details_json: dict = Field(default_factory=dict, sa_column=Column(JSONN))


class Export(SQLModel, table=True):
    __tablename__ = "export"
    id: Optional[int] = Field(default=None, primary_key=True)
    case_id: str = Field(foreign_key="case.id", index=True)
    path: Optional[str] = None
    sha256: Optional[str] = None
    selection_json: dict = Field(default_factory=dict, sa_column=Column(JSONN))
    status: str = "queued"
    created_by: Optional[int] = Field(default=None, foreign_key="user.id")
    created_at: datetime = Field(default_factory=now)


class Job(SQLModel, table=True):
    __tablename__ = "job"
    id: Optional[int] = Field(default=None, primary_key=True)
    type: str = Field(index=True)
    case_id: Optional[str] = Field(default=None, foreign_key="case.id", index=True)
    payload_json: dict = Field(default_factory=dict, sa_column=Column(JSONN))
    result_json: Optional[dict] = Field(default=None, sa_column=Column(JSONN))
    status: str = Field(default="queued", index=True)  # queued|running|done|failed|cancel_requested|cancelled|blocked
    progress: float = 0.0
    message: str = ""
    depends_on: Optional[int] = Field(default=None, foreign_key="job.id")
    created_by: Optional[int] = Field(default=None, foreign_key="user.id")
    created_at: datetime = Field(default_factory=now)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    heartbeat_at: Optional[datetime] = None


class AuditEvent(SQLModel, table=True):
    __tablename__ = "audit_event"
    id: Optional[int] = Field(default=None, primary_key=True)
    ts: str
    user_id: Optional[int] = None
    role: Optional[str] = None
    action: str = Field(index=True)
    object_type: Optional[str] = None
    object_id: Optional[str] = None
    details_json: Any = Field(default=None, sa_column=Column(JSONN))
    prev_hash: str
    hash: str
