"""End-to-end smoke test against a running API + worker.

  python scripts/smoke_e2e.py --admin-password <pw> [--base http://127.0.0.1:8000/api]

Flow: login -> case -> upload transaction clip + gallery clips -> wait for ingest/tracking ->
add transaction -> ranked candidates -> confirm -> best frames -> enhance -> search -> review ->
export PDF -> download -> verify audit chain.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples" / "kaggle_atm_video"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000/api")
    ap.add_argument("--admin", default="admin")
    ap.add_argument("--admin-password", required=True)
    ap.add_argument("--txn-clip", default=str(SAMPLES / "handsign_1.mp4"))
    ap.add_argument("--gallery", nargs="*", default=[str(SAMPLES / "handsign_14.mp4"), str(SAMPLES / "normal_2.mp4")])
    ap.add_argument("--timeout", type=float, default=900)
    a = ap.parse_args()
    c = httpx.Client(base_url=a.base, timeout=120)

    def ok(r: httpx.Response) -> dict | list:
        if r.status_code >= 400:
            sys.exit(f"{r.request.method} {r.request.url} -> {r.status_code} {r.text[:500]}")
        return r.json()

    tok = ok(c.post("/auth/login", json={"username": a.admin, "password": a.admin_password}))
    c.headers["Authorization"] = f"Bearer {tok['access_token']}"
    print("logged in as", tok["username"], tok["role"])

    atms = ok(c.get("/atms"))
    atm = next((x for x in atms if x["atm_code"] == "KAGGLE-ATM-01"), None)
    case = ok(c.post("/cases", json={"ref_no": f"SMOKE-{int(time.time())}", "title": "Smoke test"}))
    print("case", case["id"])

    def upload(path: str, kind: str) -> dict:
        with open(path, "rb") as f:
            r = c.post(f"/cases/{case['id']}/clips", files={"file": (Path(path).name, f, "video/mp4")},
                       data={"kind": kind, **({"atm_id": str(atm["id"])} if atm else {})})
        clip = ok(r)
        print(f"uploaded {Path(path).name} as {kind}: {clip['id']} sha256 {clip['orig_sha256'][:12]}")
        return clip

    txn_clip = upload(a.txn_clip, "transaction")
    gallery = [upload(p, "gallery") for p in a.gallery]

    def wait_clip(cid: str, states: set[str]) -> dict:
        t0 = time.time()
        while time.time() - t0 < a.timeout:
            cl = ok(c.get(f"/clips/{cid}"))
            if cl["status"] in states:
                return cl
            if cl["status"] == "failed":
                sys.exit(f"clip failed: {cl['error']}")
            time.sleep(2)
        sys.exit(f"timeout waiting for clip {cid}")

    def wait_job(jid: int) -> dict:
        t0 = time.time()
        while time.time() - t0 < a.timeout:
            j = ok(c.get(f"/jobs/{jid}"))
            if j["status"] == "done":
                return j
            if j["status"] in ("failed", "cancelled", "blocked"):
                sys.exit(f"job {jid} {j['type']} {j['status']}: {j['message']}")
            time.sleep(1.5)
        sys.exit(f"timeout waiting for job {jid}")

    tc = wait_clip(txn_clip["id"], {"tracked"})
    print("transaction clip tracked:", tc["n_tracks"], "tracks; time map", tc["timemap"]["method"], "start", tc["start_wall"])
    start = datetime.fromisoformat(tc["start_wall"]).replace(tzinfo=None)
    txns = ok(c.post(f"/clips/{tc['id']}/transactions", json=[{
        "txn_ref": "TXN-0001", "txn_time": (start + timedelta(seconds=4)).isoformat(), "amount": 5000,
        "window_before_s": 90, "window_after_s": 30}]))
    txn = txns[0]
    print("transaction", txn["id"], "t_txn_pts", txn["t_txn_pts"])
    t0 = time.time()
    cands = []
    while time.time() - t0 < a.timeout:
        t = ok(c.get(f"/transactions/{txn['id']}"))
        if t["status"] in ("linked", "confirmed", "analysed"):
            cands = ok(c.get(f"/transactions/{txn['id']}/candidates"))
            break
        time.sleep(1.5)
    if not cands:
        sys.exit("no candidates")
    for cd in cands:
        print(f"  candidate rank {cd['rank']} track {cd['track_no']} score {cd['score']:.3f} {cd['reasons'][:2]}")
    job = ok(c.post(f"/transactions/{txn['id']}/confirm", json={"track_id": cands[0]["track_id"], "note": "smoke"}))
    wait_job(job["id"])
    faces = ok(c.get(f"/tracks/{cands[0]['track_id']}/faces", params={"selected": True}))
    print(f"best frames: {len(faces)}")
    for f in faces:
        print(f"  face {f['id']} total {f['total']:.1f} grade {f['grade']} flags {f['flags']} searchable {f['searchable']}")
    assert faces, "no faces selected"

    enh = ok(c.post(f"/faces/{faces[0]['id']}/enhance", json={"method": "clahe", "params": {"clip": 2.0}}))
    wait_job(enh["job_id"])
    print("enhancement ok")

    for g in gallery:
        wait_clip(g["id"], {"indexed"})
    print("gallery indexed")

    searchable = [f["id"] for f in faces if f["searchable"]]
    hits = []
    if searchable:
        sr = ok(c.post("/search", json={"case_id": case["id"], "face_ids": searchable[:3]}))
        hits = sr["hits"]
        print(f"search {sr['search_id']}: {len(hits)} hits, {len(sr['not_comparable'])} not comparable, "
              f"bands {sr['bands']} calibrated={sr['calibrated']}")
        for h in hits[:5]:
            print(f"  hit {h['id']} {h['clip_name']} track {h['track_no']} score {h['score']:.3f} {h['band']}")
        if hits:
            ok(c.post(f"/search_hits/{hits[0]['id']}/review", json={"decision": "confirmed", "note": "smoke review"}))
    else:
        print("no searchable faces in the transaction clip (covered/poor faces): search skipped")

    ex = ok(c.post(f"/cases/{case['id']}/exports", json={"face_ids": [f["id"] for f in faces],
                                                         "search_hit_ids": [hits[0]["id"]] if hits else [],
                                                         "include_enhanced": True}))
    wait_job(ex["job_id"])
    r = c.get(f"/exports/{ex['id']}/download")
    if r.status_code != 200 or not r.content.startswith(b"%PDF"):
        sys.exit(f"download failed {r.status_code}")
    out = ROOT / "data" / f"smoke_export_{ex['id']}.pdf"
    out.write_bytes(r.content)
    sha = hashlib.sha256(r.content).hexdigest()
    rec = ok(c.get(f"/exports/{ex['id']}"))
    print(f"PDF {out} {len(r.content)} bytes sha256 matches record: {sha == rec['sha256']}")

    rng = c.get(f"/media/proxy/{tc['id']}", headers={"Range": "bytes=0-1023"})
    print("proxy range request:", rng.status_code, rng.headers.get("content-range"))
    v = ok(c.get("/audit/verify"))
    print("audit chain:", v)
    print("SMOKE OK")


if __name__ == "__main__":
    main()
