"""Back up data/, then delete every case and everything that belongs to it.

Run only while the API and worker are stopped:
    .venv\\Scripts\\python.exe scripts\\demo\\reset_cases.py

Users, ATMs and the audit log are kept (the audit log is a hash chain).
The FAISS index is rebuilt from the remaining embeddings by the worker at start-up.
The backup goes to data/_backup_<timestamp>/ (database + cases + index).
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import stat
from datetime import datetime
from pathlib import Path

DATA = Path("data")
DB = DATA / "atmcctv.db"


def backup() -> Path:
    dst = DATA / f"_backup_{datetime.now():%Y%m%d_%H%M%S}"
    dst.mkdir()
    src = sqlite3.connect(DB)
    out = sqlite3.connect(dst / "atmcctv.db")
    src.backup(out)  # consistent copy including the WAL
    out.close()
    src.close()
    for name in ("cases", "index"):
        if (DATA / name).exists():
            shutil.copytree(DATA / name, dst / name)
    return dst


def _writable(func, path, _exc):
    os.chmod(path, stat.S_IWRITE)  # originals and PDFs are stored read-only
    func(path)


def delete_cases() -> list[str]:
    c = sqlite3.connect(DB)
    ids = [r[0] for r in c.execute('SELECT id FROM "case"')]
    clips = "SELECT id FROM clip WHERE case_id IN (SELECT id FROM \"case\")"
    tracks = f"SELECT id FROM track WHERE clip_id IN ({clips})"
    steps = [
        "DELETE FROM search_hit WHERE search_id IN (SELECT id FROM search)",
        "DELETE FROM search",
        f"DELETE FROM enhancement WHERE face_id IN (SELECT id FROM face WHERE clip_id IN ({clips}))",
        "DELETE FROM candidate WHERE transaction_id IN (SELECT id FROM \"transaction\")",
        f"DELETE FROM template WHERE track_id IN ({tracks})",
        f"DELETE FROM face WHERE clip_id IN ({clips})",
        f"DELETE FROM behaviour_event WHERE clip_id IN ({clips})",
        'DELETE FROM "transaction"',
        f"DELETE FROM track WHERE clip_id IN ({clips})",
        "DELETE FROM clip",
        "DELETE FROM export",
        "DELETE FROM job WHERE case_id IS NOT NULL",
        "DELETE FROM case_member",
        'DELETE FROM "case"',
    ]
    with c:
        for sql in steps:
            n = c.execute(sql).rowcount
            print(f"{n:5d}  {sql[:70]}")
    c.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    c.close()
    cases_dir = DATA / "cases"
    if cases_dir.exists():
        for d in cases_dir.iterdir():
            shutil.rmtree(d, onerror=_writable)
    return ids


if __name__ == "__main__":
    import sys

    if "--no-backup" not in sys.argv:
        print("backup:", backup())
    print("deleted cases:", len(delete_cases()))
