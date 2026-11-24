"""File layout under data/ (paths are stored relative to data/ in the DB)."""
from __future__ import annotations

from pathlib import Path

from .settings import data_dir


def root() -> Path:
    return data_dir()


def rel(p: Path | str | None) -> str | None:
    if p is None:
        return None
    p = Path(p).resolve()
    try:
        return p.relative_to(root().resolve()).as_posix()
    except ValueError:
        return str(p)


def absp(s: str | None) -> Path | None:
    if not s:
        return None
    p = Path(s)
    return p if p.is_absolute() else root() / p


def case_dir(case_id: str) -> Path:
    return root() / "cases" / case_id


def clip_dir(case_id: str, clip_id: str) -> Path:
    return case_dir(case_id) / "clips" / clip_id


def face_dir(case_id: str, clip_id: str, face_id: int) -> Path:
    return clip_dir(case_id, clip_id) / "faces" / str(face_id)


def exports_dir(case_id: str) -> Path:
    return case_dir(case_id) / "exports"


def index_dir() -> Path:
    return root() / "index"


def tmp_dir() -> Path:
    p = root() / "tmp"
    p.mkdir(parents=True, exist_ok=True)
    return p
