# PS04: ATM CCTV suspect face extraction

Local, offline investigator tool. It ingests ATM CCTV footage, finds the person who made a given transaction,
extracts and grades their best face frames, flags unreliable faces, searches other clips for possible matches and
exports a hash-stamped evidence sheet. Every step that names a person needs a human decision; results are
"candidate leads", never identifications.

Design docs and demo screenshots are kept local for now (the `docs/` folder is git-ignored).

## Components

| Part | Where | Notes |
|---|---|---|
| Core pipeline (no DB/HTTP) | `src/atmcctv/core/` | ingest, time map, detection view, coarse pass, linking, dense pass, quality, flags, selection, embeddings, FAISS index, enhancement, behaviour |
| Job handlers + GPU worker | `src/atmcctv/jobs.py`, `worker.py` | one process owns all models, runs one job at a time |
| API | `src/atmcctv/api/` | FastAPI on `127.0.0.1:8000`, all routes under `/api`; never imports torch/onnxruntime |
| DB + audit | `src/atmcctv/db/`, `audit.py` | SQLite WAL, hash-chained audit log |
| Evidence PDF | `src/atmcctv/evidence.py` | ReportLab |
| Frontend | `frontend/` | React 18 + TS + Vite + TanStack Query + Tailwind; types generated from OpenAPI |
| Scripts | `scripts/` | model download, Kaggle fetch, format set, labelling, calibration, evaluation, smoke test |

## Quick start (Windows)

Prerequisites: Windows 10/11, an NVIDIA RTX GPU with a current driver, [uv](https://docs.astral.sh/uv/),
Python 3.12, Node.js 18+, and FFmpeg on `PATH`.

1. Clone the repo and open a PowerShell in it:
   ```powershell
   git clone https://github.com/Sidhanth-Mandal/ATM-Footage-Analyzer.git
   cd ATM-Footage-Analyzer
   ```
2. Run the one-time setup below (virtualenv, PyTorch, package, model weights, frontend deps).
3. Create a `.env` in the project root (git-ignored):
   ```
   ATMCCTV_SECRET=<random string, 32+ chars>
   ATMCCTV_ADMIN_PASSWORD=<first admin password>
   ```
   `start.bat` generates `ATMCCTV_SECRET` for you if it is missing. The admin password is only used while the
   user table is empty; the first admin user is `admin`.
4. Double-click `start.bat` (or run it from PowerShell). It opens three windows: API, GPU worker and the Vite dev
   server.
5. Open http://127.0.0.1:5173 and sign in as `admin`.

`data/`, `models/` and `samples/` are git-ignored and are created or downloaded locally.

## Setup (Windows, RTX GPU)

```powershell
uv venv --python 3.12
$env:VIRTUAL_ENV = "$PWD\.venv"
# onnxruntime-gpu 1.32 is built for CUDA 13, so torch must be the cu130 build (cu128 makes ORT fall back to CPU)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install -e ".[dev]"
.venv\Scripts\python scripts\download_models.py      # caches every weight in models/ (about 1.5 GB)
cd frontend; npm install; npm run build; cd ..
```

FFmpeg (with `h264_nvenc`) must be on `PATH`. The proxy encoder falls back to libx264 when NVENC is unavailable.

## Run

```powershell
$env:ATMCCTV_SECRET = "<random string, 32+ chars>"      # required; the API refuses to start without it
$env:ATMCCTV_ADMIN_PASSWORD = "<first admin password>"   # only used when the user table is empty
.venv\Scripts\python -m uvicorn atmcctv.api.main:app --host 127.0.0.1 --port 8000   # terminal 1
.venv\Scripts\python -m atmcctv.worker                                              # terminal 2
```

Open http://127.0.0.1:8000 (the API serves `frontend/dist`). For frontend development run `npm run dev` in
`frontend/` and open http://127.0.0.1:5173 (Vite proxies `/api`). After changing API schemas, run
`npm run gen:api` with the API running.

More users: Admin page, or `atmcctv create-user <name> --role investigator`.

## Workflow

1. **Cases → New case**, then upload footage. Choose *transaction clip* for the footage around the withdrawal and
   *gallery clip* for other footage to search. Pick the ATM (camera type, ROIs, time zone).
2. Ingest runs automatically: SHA-256, ffprobe, trial decode, normalisation (VFR, interlaced, raw streams,
   vendor containers), browser proxy. Proprietary or encrypted files get an error with vendor export steps.
3. **Setup**: check the time mapping (OCR of the timestamp overlay, `creation_time` metadata, or a manual start
   time) and set the clock offset (DVR clock − journal clock). For lobby cameras draw the ATM zone, keypad, cash
   and card slot polygons on a frame.
4. **Add transaction** (journal time, window −90/+30 s by default). The ranked candidates show per-part scores
   and reasons. The investigator confirms one track.
5. **Best frames**: the confirmed track is processed at full frame rate. Each face shows grade, total, six
   sub-scores and reliability flags. Optional enhancement creates a separate, bannered copy.
6. **Search**: select searchable faces → strong / possible / weak possible matches in indexed clips, plus a
   "not comparable" list. Each match must be confirmed or rejected by a reviewer.
7. **Export**: choose faces and confirmed matches → PDF evidence sheet with hashes, time mapping, candidate table,
   sub-scores, flags, reviewers and disclaimer. The PDF hash goes into the audit log.

## Verification

```powershell
.venv\Scripts\python -m pytest tests/unit              # quality maths, time map, linking, audit chain, RBAC, embedding guard
.venv\Scripts\python scripts\make_formats.py           # 16 format/edge-case files in samples/formats
.venv\Scripts\python scripts\evaluate.py --ingest      # accept/reject + error code per file
.venv\Scripts\python scripts\evaluate.py --timemap-synthetic
.venv\Scripts\python -m pytest tests/golden            # full GPU pipeline on MP4 / AVI / raw .264
.venv\Scripts\python scripts\smoke_e2e.py --admin-password <pw>   # end-to-end through the running API + worker
.venv\Scripts\python -m atmcctv.cli bench samples\kaggle_atm_video\handsign_14.mp4
```

Search thresholds and FIQA bounds ship as uncalibrated defaults (the UI and PDF say so). To calibrate: index the
Kaggle clips (`scripts/fetch_kaggle_atm.py` downloads all 92), label identities with `scripts/label_tool.py
clusters` and `customers`, then run `scripts/calibrate.py --labels samples/kaggle_atm_video/labels.json`, which
writes `config/calibration.yaml`. `scripts/evaluate.py --labels ...` reports linking top-1, best-frame overlap,
flag precision/recall and TAR@FMR.

`scripts/make_synthetic_face_clips.py` builds two clips of a clearly visible public-domain face (the Kaggle
footage has no searchable faces: backlight, niqab, caps), which exercises search, review and export.

## Security notes

- Binds to `127.0.0.1` only. CORS allows only the Vite dev origin.
- argon2id passwords, lockout after 5 failures for 5 minutes, JWT HS256 (8 h), key from `ATMCCTV_SECRET`.
- RBAC on every route plus case membership. Media is served by id after the same checks, never from a static
  mount. `<img>`/`<video>` use a separate 30-minute media token that the API rejects for normal calls.
- FFmpeg/ffprobe run with argument lists, timeouts and `-protocol_whitelist file,pipe`; playlists are rejected.
- Originals are set read-only after hashing. `data/`, `samples/`, `models/` and `docs/` are git-ignored.
