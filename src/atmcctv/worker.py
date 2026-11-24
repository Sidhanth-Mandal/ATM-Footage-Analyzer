"""Single GPU worker process: `python -m atmcctv.worker`.

Owns every model (loaded once), runs one job at a time, claims jobs with
BEGIN IMMEDIATE, writes heartbeat/progress at most every 0.5 s and honours
cancellation between frames.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
import traceback
from datetime import timedelta

from sqlalchemy import text
from sqlmodel import Session, select

from . import audit
from .db.models import Job, now
from .db.session import init_db, write_session
from .jobs import HANDLERS, Cancelled, Ctx, JobError, get_index
from .queue import enqueue
from .settings import config_hash, data_dir, get_config, load_config

log = logging.getLogger("atmcctv.worker")
STALE_S = 60
_stop = False


def claim() -> Job | None:
    with write_session() as s:
        row = s.exec(text(
            "SELECT id FROM job WHERE status='queued' AND (depends_on IS NULL OR depends_on IN "
            "(SELECT id FROM job WHERE status='done')) "
            # interactive steps (an investigator is waiting) before batch gallery indexing
            "ORDER BY CASE type WHEN 'index' THEN 2 WHEN 'behaviour' THEN 1 ELSE 0 END, created_at, id LIMIT 1")).first()
        if not row:
            return None
        j = s.get(Job, row[0])
        j.status, j.started_at, j.heartbeat_at, j.progress, j.message = "running", now(), now(), 0.0, "started"
        s.add(j)
        s.flush()
        return j


def requeue_stale() -> int:
    with write_session() as s:
        cutoff = now() - timedelta(seconds=STALE_S)
        n = 0
        for j in s.exec(select(Job).where(Job.status.in_(["running", "cancel_requested"]))).all():
            if j.heartbeat_at is None or j.heartbeat_at < cutoff:
                if j.status == "cancel_requested":
                    j.status = "cancelled"
                else:
                    j.status, j.message = "queued", "re-queued after worker restart"
                n += 1
        # jobs whose dependency failed can never run
        for j in s.exec(select(Job).where(Job.status == "queued", Job.depends_on.is_not(None))).all():
            dep = s.get(Job, j.depends_on)
            if dep and dep.status in ("failed", "cancelled"):
                j.status, j.message = "blocked", f"dependency job {dep.id} {dep.status}"
        return n


def check_index() -> None:
    """Rebuild the FAISS indexes from DB embeddings if missing or inconsistent."""
    import numpy as np
    from .db.models import FaceRow, Template
    from .db.session import read_session

    idx = get_index()
    with read_session() as s:
        faces = {f.id: np.frombuffer(f.embedding, np.float32) for f in s.exec(
            select(FaceRow).where(FaceRow.embedding.is_not(None))).all()}
        tmpls = {t.track_id: np.frombuffer(t.vector, np.float32) for t in s.exec(select(Template)).all()}
    if not idx.consistent(len(faces), len(tmpls)):
        log.warning("FAISS index inconsistent with DB (%d/%d vs %d/%d); rebuilding",
                    idx.faces.ntotal, idx.templates.ntotal, len(faces), len(tmpls))
        idx.rebuild(faces, tmpls)


class Runner:
    def __init__(self, cfg):
        self.cfg = cfg
        self._registry = None

    def registry(self):
        if self._registry is None:
            from .core.registry import ModelRegistry
            self._registry = ModelRegistry(self.cfg)
        return self._registry

    def run(self, job: Job) -> None:
        last = [0.0]
        job_id = job.id
        followups: list[tuple[str, dict, str | None]] = []

        def progress(frac: float, msg: str = "") -> None:
            t = time.monotonic()
            if t - last[0] < 0.5 and frac < 1.0:
                return
            last[0] = t
            with write_session() as s:
                j = s.get(Job, job_id)
                if j.status == "cancel_requested":
                    raise Cancelled()
                j.progress, j.message, j.heartbeat_at = float(max(0.0, min(1.0, frac))), msg[:300], now()

        def enq(type_: str, payload: dict, case_id: str | None = None) -> int:
            followups.append((type_, payload, case_id))
            return len(followups)

        ctx = Ctx(job, load_config(), self.registry, write_session, progress, enq)
        handler = HANDLERS.get(job.type)
        t0 = time.monotonic()
        try:
            if handler is None:
                raise JobError(f"unknown job type {job.type}")
            result = handler(job.payload_json or {}, ctx)
            result = dict(result or {})
            result["config_sha256"] = config_hash(ctx.cfg)
            result["elapsed_s"] = round(time.monotonic() - t0, 2)
            with write_session() as s:
                j = s.get(Job, job_id)
                j.status, j.progress, j.finished_at, j.result_json = "done", 1.0, now(), result
                j.message = "done"
                for type_, payload, case_id in followups:
                    enqueue(s, type_, payload, case_id=case_id, user_id=j.created_by, depends_on=None)
            log.info("job %s %s done in %.1fs", job_id, job.type, time.monotonic() - t0)
        except Cancelled:
            with write_session() as s:
                j = s.get(Job, job_id)
                j.status, j.finished_at, j.message = "cancelled", now(), "cancelled by user"
                audit.append(s, "job_cancelled", user_id=j.created_by, role="system/worker", object_type="job",
                             object_id=job_id)
        except Exception as e:  # noqa: BLE001
            log.error("job %s %s failed:\n%s", job_id, job.type, traceback.format_exc())
            if self._registry is not None and self._registry.hold_open_vocab:
                self._registry.hold_open_vocab = False
                self._registry.release("open_vocab")
            msg = str(e) if isinstance(e, JobError) else f"{type(e).__name__}: {e}"
            with write_session() as s:
                j = s.get(Job, job_id)
                j.status, j.finished_at, j.message = "failed", now(), msg[:500]
                _mark_object_failed(s, j, msg)
                audit.append(s, "job_failed", user_id=j.created_by, role="system/worker", object_type="job",
                             object_id=job_id, details={"type": j.type, "message": msg[:300]})


def _mark_object_failed(s: Session, j: Job, msg: str) -> None:
    from .db.models import Clip, Export
    p = j.payload_json or {}
    if j.type in ("ingest", "coarse", "index") and p.get("clip_id"):
        c = s.get(Clip, p["clip_id"])
        if c:
            c.status = "failed"
            c.error_json = c.error_json or {"code": "JOB_FAILED", "message": msg[:300], "vendor": None,
                                            "instructions": [f"The {j.type} step failed. Check the worker log."]}
    if j.type == "export" and p.get("export_id"):
        e = s.get(Export, p["export_id"])
        if e:
            e.status = "failed"


def _sigterm(*_):
    global _stop
    _stop = True


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="atmcctv.worker")
    ap.add_argument("--once", action="store_true", help="run queued jobs until the queue is empty, then exit")
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(data_dir() / "worker.log", encoding="utf-8")]
                        if data_dir().exists() else [logging.StreamHandler(sys.stdout)])
    cfg = get_config()
    init_db()
    n = requeue_stale()
    if n:
        log.info("re-queued %d stale jobs", n)
    check_index()
    runner = Runner(cfg)
    if not args.no_warmup:
        try:
            runner.registry().warmup()
            log.info("models warmed up: %s", runner.registry().providers)
        except Exception as e:
            log.error("warm-up failed: %s", e)
            raise
    signal.signal(signal.SIGINT, _sigterm)
    signal.signal(signal.SIGTERM, _sigterm)
    log.info("worker ready (pid %d)", os.getpid())
    idle = 0.0
    while not _stop:
        job = claim()
        if job is None:
            if args.once:
                break
            time.sleep(min(1.0, 0.2 + idle))
            idle = min(1.0, idle + 0.1)
            continue
        idle = 0.0
        log.info("job %s %s claimed", job.id, job.type)
        runner.run(job)


if __name__ == "__main__":
    main()
