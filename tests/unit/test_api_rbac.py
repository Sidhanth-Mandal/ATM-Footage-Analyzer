"""Role matrix: every state-changing route returns 403 for viewer; membership hides other cases."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from atmcctv.api.main import app
    with TestClient(app) as c:
        yield c


def _mkuser(name, role):
    from sqlmodel import select
    from atmcctv.api.auth import hash_password
    from atmcctv.db.models import User
    from atmcctv.db.session import init_db, write_session
    init_db()
    with write_session() as s:
        if not s.exec(select(User).where(User.username == name)).first():
            s.add(User(username=name, password_hash=hash_password("password-123456"), role=role))


def _login(client, name):
    r = client.post("/api/auth/login", json={"username": name, "password": "password-123456"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture(scope="module")
def tokens(client):
    for n, r in (("adm", "admin"), ("inv", "investigator"), ("view", "viewer"), ("inv2", "investigator")):
        _mkuser(n, r)
    return {n: _login(client, n) for n in ("adm", "inv", "view", "inv2")}


WRITE_ROUTES = [
    ("post", "/api/cases", {"ref_no": "x", "title": "y"}),
    ("post", "/api/atms", {"atm_code": "A1"}),
    ("put", "/api/atms/1", {"atm_code": "A1"}),
    ("post", "/api/cases/{case}/gallery/index", {"clip_ids": []}),
    ("post", "/api/cases/{case}/exports", {"face_ids": [1]}),
    ("put", "/api/clips/c1/timemap", {"clock_offset_s": 1}),
    ("post", "/api/clips/c1/transactions", [{"txn_time": "2026-01-01T10:00:00"}]),
    ("post", "/api/clips/c1/behaviour", None),
    ("post", "/api/transactions/1/confirm", {"track_id": 1}),
    ("post", "/api/faces/1/enhance", {"method": "clahe"}),
    ("post", "/api/search", {"case_id": "x", "face_ids": [1]}),
    ("post", "/api/search_hits/1/review", {"decision": "confirmed"}),
    ("post", "/api/jobs/1/cancel", None),
    ("post", "/api/users", {"username": "zzz", "password": "0123456789ab"}),
    ("patch", "/api/users/1", {"role": "viewer"}),
    ("put", "/api/cases/{case}/members", {"user_ids": []}),
]


def test_viewer_gets_403_on_every_write_route(client, tokens):
    case = client.post("/api/cases", json={"ref_no": "R1", "title": "t"}, headers=tokens["inv"]).json()["id"]
    for method, path, body in WRITE_ROUTES:
        p = path.format(case=case)
        r = getattr(client, method)(p, json=body, headers=tokens["view"]) if body is not None else \
            getattr(client, method)(p, headers=tokens["view"])
        assert r.status_code == 403, (method, p, r.status_code, r.text)
    files = {"file": ("a.mp4", b"\x00" * 10, "video/mp4")}
    r = client.post(f"/api/cases/{case}/clips", files=files, headers=tokens["view"])
    assert r.status_code == 403


def test_admin_only_routes(client, tokens):
    for path in ("/api/users", "/api/audit", "/api/audit/verify"):
        assert client.get(path, headers=tokens["inv"]).status_code == 403
        assert client.get(path, headers=tokens["adm"]).status_code == 200
    v = client.get("/api/audit/verify", headers=tokens["adm"]).json()
    assert v["ok"] is True


def test_case_membership(client, tokens):
    c = client.post("/api/cases", json={"ref_no": "R2", "title": "private"}, headers=tokens["inv"]).json()
    assert client.get(f"/api/cases/{c['id']}", headers=tokens["inv2"]).status_code == 404
    assert c["id"] not in [x["id"] for x in client.get("/api/cases", headers=tokens["inv2"]).json()]
    assert client.get(f"/api/cases/{c['id']}", headers=tokens["adm"]).status_code == 200


def test_unauthenticated_and_lockout(client, tokens):
    assert client.get("/api/cases").status_code == 401
    _mkuser("lock", "viewer")
    for _ in range(5):
        assert client.post("/api/auth/login", json={"username": "lock", "password": "wrong"}).status_code == 401
    r = client.post("/api/auth/login", json={"username": "lock", "password": "password-123456"})
    assert r.status_code == 423


def test_media_token_cannot_call_api(client, tokens):
    mt = client.post("/api/auth/media-token", headers=tokens["inv"]).json()["token"]
    assert client.get("/api/cases", headers={"Authorization": f"Bearer {mt}"}).status_code == 401
