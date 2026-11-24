"""FastAPI app. Run: uvicorn atmcctv.api.main:app --host 127.0.0.1 --port 8000

The API process never imports torch or onnxruntime; all GPU work goes through the job queue.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sqlmodel import select
from starlette.exceptions import HTTPException as StarletteHTTPException

from .. import __version__, audit
from ..db.models import User
from ..db.session import init_db, write_session
from ..settings import ROOT, get_config, secret_key
from .atm_files import sync_atm_files
from .auth import hash_password
from .deps import APIError
from .routers import analysis, atms, auth_users, cases, clips, misc, search

log = logging.getLogger("atmcctv.api")


def _bootstrap_admin() -> None:
    """Create the first admin from ATMCCTV_ADMIN_PASSWORD if there are no users yet."""
    pw = os.environ.get("ATMCCTV_ADMIN_PASSWORD")
    with write_session() as s:
        if s.exec(select(User)).first():
            return
        if not pw:
            log.warning("no users exist: create one with `atmcctv create-user admin --role admin` "
                        "or set ATMCCTV_ADMIN_PASSWORD and restart")
            return
        u = User(username=os.environ.get("ATMCCTV_ADMIN_USER", "admin"), password_hash=hash_password(pw), role="admin")
        s.add(u)
        s.flush()
        audit.append(s, "user_created", role="bootstrap", object_type="user", object_id=u.id,
                     details={"username": u.username, "role": "admin"})


@asynccontextmanager
async def lifespan(app: FastAPI):
    secret_key()  # fail fast when ATMCCTV_SECRET is missing
    init_db()
    _bootstrap_admin()
    r = sync_atm_files()
    if r["errors"]:
        log.warning("ATM file errors: %s", r["errors"])
    yield


def create_app() -> FastAPI:
    cfg = get_config()
    app = FastAPI(title="ATM CCTV suspect face extraction (PS04)", version=__version__, lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=list(cfg.app.cors_origins), allow_credentials=False,
                       allow_methods=["GET", "POST", "PUT", "PATCH"], allow_headers=["Authorization", "Content-Type", "Range"],
                       expose_headers=["Content-Range", "Accept-Ranges", "X-Content-SHA256", "Content-Disposition"])

    @app.exception_handler(APIError)
    async def _api_error(_req: Request, e: APIError):
        return JSONResponse({"code": e.code, "message": e.message, "details": e.details}, status_code=e.status)

    @app.exception_handler(RequestValidationError)
    async def _validation(_req: Request, e: RequestValidationError):
        return JSONResponse({"code": "VALIDATION_ERROR", "message": "invalid request",
                             "details": {"errors": [{"loc": list(x.get("loc", [])), "msg": x.get("msg")} for x in e.errors()]}},
                            status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_req: Request, e: StarletteHTTPException):
        return JSONResponse({"code": f"HTTP_{e.status_code}", "message": str(e.detail), "details": {}},
                            status_code=e.status_code)

    @app.middleware("http")
    async def _security_headers(request: Request, call_next):
        resp = await call_next(request)
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        return resp

    for r in (auth_users.router, cases.router, atms.router, clips.router, analysis.router, search.router, misc.router):
        app.include_router(r, prefix="/api")

    @app.get("/api/health", tags=["meta"])
    def health() -> dict:
        return {"ok": True, "version": __version__}

    # Serve the built SPA (frontend/dist) if present; API and media stay under /api.
    dist = ROOT / "frontend" / "dist"
    if dist.exists():
        @app.get("/{full_path:path}", include_in_schema=False)
        def spa(full_path: str):
            p = (dist / full_path).resolve()
            if full_path and p.is_file() and dist.resolve() in p.parents:
                return FileResponse(p)
            return FileResponse(dist / "index.html")

    return app


app = create_app()


def run() -> None:
    import uvicorn
    cfg = get_config()
    uvicorn.run("atmcctv.api.main:app", host=str(cfg.app.host), port=int(cfg.app.port))


__all__ = ["app", "Path"]
