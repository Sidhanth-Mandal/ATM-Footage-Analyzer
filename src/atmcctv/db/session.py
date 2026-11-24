"""SQLite engine with WAL, busy timeout, foreign keys and explicit BEGIN control.

pysqlite's implicit transaction handling is disabled so we can issue
`BEGIN IMMEDIATE` for write sessions (audit chain appends, job claims) and a
plain deferred `BEGIN` for read sessions.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Iterator

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

from ..settings import data_dir, db_url
from . import models  # noqa: F401  (register tables)


def _make_engine(url: str) -> Engine:
    if url.startswith("sqlite:///"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    eng = create_engine(url, connect_args={"check_same_thread": False, "timeout": 30, "isolation_level": None},
                        pool_pre_ping=True)

    @event.listens_for(eng, "connect")
    def _pragmas(dbapi_conn, _rec):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA busy_timeout=5000")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.close()

    @event.listens_for(eng, "begin")
    def _begin(conn):
        mode = conn.get_execution_options().get("sqlite_begin", "DEFERRED")
        conn.exec_driver_sql(f"BEGIN {mode}")

    return eng


@lru_cache(maxsize=4)
def get_engine(url: str | None = None) -> Engine:
    return _make_engine(url or db_url())


def init_db(url: str | None = None) -> Engine:
    eng = get_engine(url)
    data_dir().mkdir(parents=True, exist_ok=True)
    SQLModel.metadata.create_all(eng)
    return eng


def read_session(url: str | None = None) -> Session:
    return Session(get_engine(url), expire_on_commit=False)


@contextmanager
def write_session(url: str | None = None) -> Iterator[Session]:
    """Session whose transaction starts with BEGIN IMMEDIATE (single writer, serialised)."""
    s = Session(get_engine(url), expire_on_commit=False)
    try:
        s.connection(execution_options={"sqlite_begin": "IMMEDIATE"})
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
