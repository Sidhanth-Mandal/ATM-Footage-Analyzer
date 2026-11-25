# Features: ATM CCTV suspect face extraction

This walkthrough covers every screen of the app, using the two clips in `samples/Video Demo/`. All screenshots come from a real run on 24 Nov 2026 (RTX 4060 Laptop GPU, CUDA for every model). Nothing in them was staged or edited.

| Demo clip | Used as | Overlay | Length |
|---|---|---|---|
| `gemini_generated_video_112a7b7f.mp4` | Transaction clip (the withdrawal) | `CAM-02 ATM 01`, starts 21:19:13 | 10 s, 1280×720, 24 fps |
| `gemini_generated_video_4972d49f.mp4` | Gallery clip (other footage to search) | `CAM-01 ATM-02`, starts 21:30:00 | 10 s, 1280×720, 24 fps |

Both clips show the same man, so the face search has a real match to find.

**How the run was set up.** Ports 8000 and 5173 were already in use by your own instance, so this run used:

- API: `http://127.0.0.1:8010` (uvicorn), plus its own GPU worker
- Frontend: `http://127.0.0.1:5180` (Vite dev server, proxies `/api` to 8010)
- Data: `data/_featuredoc/` (separate database, index and case files; your `data/atmcctv.db` was not touched)

The steps to start it again are at the end of this file.

---

## Contents

1. [Sign-in and security](#1-sign-in-and-security)
2. [Admin: users](#2-admin-users)
3. [ATMs](#3-atms)
4. [Cases and the guided workflow](#4-cases-and-the-guided-workflow)
5. [Upload and ingest](#5-upload-and-ingest)
6. [Format handling and rejection messages](#6-format-handling-and-rejection-messages)
7. [Clip setup: regions of interest](#7-clip-setup-regions-of-interest)
8. [Clip setup: time mapping (OCR and manual)](#8-clip-setup-time-mapping-ocr-and-manual)
9. [Add a transaction](#9-add-a-transaction)
10. [Pick the customer](#10-pick-the-customer)
11. [Review faces (best frames)](#11-review-faces-best-frames)
12. [Face detail drawer](#12-face-detail-drawer)
13. [Enhancement (labelled copies)](#13-enhancement-labelled-copies)
14. [Face search across clips](#14-face-search-across-clips)
15. [Match review (confirm / reject)](#15-match-review-confirm--reject)
16. [Evidence export (PDF)](#16-evidence-export-pdf)
17. [Case access and the viewer role](#17-case-access-and-the-viewer-role)
18. [Audit log](#18-audit-log)
19. [Background jobs](#19-background-jobs)
20. [Features without a UI screen](#20-features-without-a-ui-screen)
21. [Issues seen during the run](#21-issues-seen-during-the-run)
22. [How to re-run this setup](#22-how-to-re-run-this-setup)

---

## 1. Sign-in and security

The app is a local tool with username and password sign-in. The login page reminds users that every action is audited and that results are leads, not identifications.

![Login](docs/feature-screenshots/01-login.png)

A wrong password shows an error. The failed attempt is also written to the audit log as `login_failed`. After 5 failures the account is locked for 5 minutes.

![Login error](docs/feature-screenshots/02-login-error.png)

Security details:

- Passwords are hashed with argon2id. Sessions use a JWT (HS256, 8 h) signed with `ATMCCTV_SECRET`.
- The token lives in `sessionStorage`, so closing the tab signs the user out.
- Images and videos use a separate 30-minute media token. The API rejects that token for normal calls.
- The API only binds to `127.0.0.1`.

## 2. Admin: users

**Admin → Users** lists accounts with their role (`viewer`, `investigator`, `admin`) and an enable/disable toggle. You can change roles inline. The form checks the input before sending it: usernames must be 3 to 64 characters from letters, digits and `_ . -`, and passwords need at least 10 characters. The screenshot shows the username rule rejecting `x`.

![Admin users](docs/feature-screenshots/04-admin-users.png)

On a fresh database the first admin is created from `ATMCCTV_ADMIN_PASSWORD`. You can also create users from the CLI with `atmcctv create-user <name> --role ...`.

## 3. ATMs

Each ATM stores its camera type (fascia or lobby), regions of interest, timestamp box, date order (DMY, MDY or YMD), time zone, and whether the file's `creation_time` marks the start or the end of the recording. ATMs defined in `config/atms/*.yaml` are synced at start-up, which is where `KAGGLE-ATM-01` comes from.

![ATMs list](docs/feature-screenshots/05-atms-list.png)

For the demo I added `ATM-01` (fascia, YMD, Asia/Kolkata) and assigned it to both clips at upload.

![ATM added](docs/feature-screenshots/06-atms-added.png)

## 4. Cases and the guided workflow

A case holds the footage, transactions, searches and exports for one incident. Investigators and admins can create cases.

![Cases empty](docs/feature-screenshots/03-cases-empty.png)
![New case dialog](docs/feature-screenshots/07-new-case-dialog.png)

Every page inside a case shows a **7-step stepper**: Upload footage → Set up clip → Add transaction → Pick the customer → Review faces → Search other clips → Export. Below it, a **Next** hint says what to do and links to the right page. The step states come from the case data, and locked steps cannot be clicked.

![Empty case](docs/feature-screenshots/08-case-empty.png)

The case list shows reference, title, member count, clip count, status and creation time.

![Cases list](docs/feature-screenshots/11-cases-list.png)

## 5. Upload and ingest

The upload card asks for the clip type (**transaction** or **gallery**) and an optional ATM. Files can be dropped or picked, and several can go at once, each with its own progress bar. Supported formats: MP4, AVI, MOV, MKV, WMV, FLV, MPG, raw `.264/.265` and Dahua `.dav`.

Ingest runs by itself: SHA-256 hash, ffprobe, a trial decode, normalisation when needed, and a browser proxy. The original file is then set read-only. Person tracking (`coarse`) follows, and gallery clips are indexed for face search. The floating job cards show progress, each with a **Cancel** button.

![Upload and ingest running](docs/feature-screenshots/09-upload-and-ingest-running.png)

Once ingest finishes, each clip shows a status chip ("People found", "Searchable") and a time-mapping chip. Its details list the SHA-256, size, ATM, codec and resolution, frame rate and duration, normalisation decisions, time map, clock offset and the number of people tracked. Gallery clips also get an **Index for search** button to re-index them. The screenshot below was taken after the whole workflow was done.

![Case with clips](docs/feature-screenshots/39-case-complete.png)

Measured job times for the two demo clips: ingest 0.9 to 1.2 s, tracking 1.7 to 2.6 s, gallery indexing 17 s.

## 6. Format handling and rejection messages

To show the error paths, I created a second case (`FIR-2026-1125`) and uploaded six files from `samples/formats/`:

| File | Result |
|---|---|
| `raw_h264.264` | Accepted. Normalised to H.264 CFR MP4 (raw elementary stream, broken index) |
| `interlaced.mkv` | Accepted. Normalised (interlaced). The clip details show "interlaced" |
| `vfr.mkv` | Accepted as is |
| `not_a_video.mp4` | Rejected: `UNSUPPORTED_CONTAINER`, with steps to re-export from the recorder |
| `sneaky_playlist.mp4` | Rejected: `PLAYLIST_REJECTED` (a playlist renamed to `.mp4`) |
| `truncated.mp4` | Rejected: `UNSUPPORTED_CONTAINER` |

Each rejection shows the error code, a message and numbered fix-it steps.

![Format normalisation and rejections](docs/feature-screenshots/41-format-normalisation-and-rejections.png)

## 7. Clip setup: regions of interest

**Set up** opens the clip setup page. It has a frame viewer with a position slider, the ATM panel and the time-mapping panel.

![Clip setup](docs/feature-screenshots/12-clip-setup-initial.png)

On a frame you can draw polygons for the **keypad**, **cash slot**, **card slot** and **ATM zone**, and drag a box over the **timestamp overlay**. Each tool has Undo point and Clear. Coordinates are stored normalised (0 to 1) on the ATM, so every clip from that ATM reuses them. Lobby cameras rank people by torso-in-zone and wrist-in-slot. Fascia cameras rank by how close and how present a person is.

![ROIs drawn](docs/feature-screenshots/13-roi-editor-drawn.png)

**Save ATM setup** stores the regions, and **Run OCR on timestamp box** queues an OCR time-mapping job.

![ROI saved, OCR queued](docs/feature-screenshots/14-roi-saved-ocr-queued.png)

## 8. Clip setup: time mapping (OCR and manual)

The time mapping converts video time into bank journal time. A RANSAC line is fitted through the OCR reads (`wall = a + b × pts`).

On the transaction clip, OCR read the burned-in `2026-11-24 21:19:14…21:19:22` overlay. All 5 reads parsed, with a 0.40 s residual, and the clip start came out as **21:19:13 +05:30**. The OCR preview table lists each read with its PTS, text, confidence and parsed value. **Clock offset** (DVR clock minus journal clock) can be set for drift.

![Time map by OCR](docs/feature-screenshots/15-timemap-ocr-result.png)

On the gallery clip I used **Manual start time** (21:30:00) to show the manual method.

![Manual time map](docs/feature-screenshots/16-gallery-clip-manual-time.png)

Other supported methods: the `creation_time` metadata, where the ATM setting decides whether it marks the start or the end of the recording.

## 9. Add a transaction

Section 3 of the transaction clip's setup page takes the journal reference, the journal time (in the ATM's time zone), the amount, and the search window, which defaults to −90 s / +30 s.

![Add transaction form](docs/feature-screenshots/17-add-transaction-form.png)

Submitting the form queues a `link` job that ranks the people near the machine. The details show the video time of the transaction (6.2 s here) and the window.

![Transaction added](docs/feature-screenshots/18-transaction-added.png)

## 10. Pick the customer

This page has the proxy video, a **"Who was on screen"** timeline and the ranked **Candidates**.

- The timeline draws one row per person. A shaded band marks the window, darker bars show when the person was near the machine, and a vertical line marks the transaction. Clicking a bar seeks the video, and clicking a label selects that person.
- Selecting a candidate jumps to their best view and draws their bounding box over the video.

![Pick the customer](docs/feature-screenshots/19-pick-customer.png)

**Show score** opens the per-part scores (overlap, dwell, proximity, closeness for fascia cameras) and plain-language reasons. Person 1 scored 0.725: "near the machine in 100% of window samples; closest near-field moment 0.0 s from the transaction". A candidate gets an "Ambiguous" badge when it ties closely with another.

![Candidate score and overlay](docs/feature-screenshots/20-candidate-score-and-overlay.png)

Confirming requires an explicit dialog with an optional reason. The decision is audited under your name.

![Confirm dialog](docs/feature-screenshots/21-confirm-customer-dialog.png)

After confirming, the full-frame-rate face pass (`dense`) runs: 284 frames in 41 s. When it finishes, the app opens Review faces.

![Dense extraction running](docs/feature-screenshots/22-dense-extraction-running.png)

## 11. Review faces (best frames)

This page shows the confirmed person's best faces, selected with a minimum time gap between them. Each card shows a grade (A ≥ 75, B ≥ 55, C below), the total score, video time and wall-clock time, and **Searchable** or **Not searchable**. The scoring summary gives 235 faces scored (A 0, B 35, C 200), with the face visible in 98% of the track's frames.

![Review faces](docs/feature-screenshots/23-review-faces.png)

**Show alternates** lists every scored face, not just the selected set.

![With alternates](docs/feature-screenshots/24-review-faces-with-alternates.png)

The search guard: grade-C or covered faces cannot be used as a search query. If you select one, the Search button is disabled and a warning names the faces to remove.

![Non-searchable in selection](docs/feature-screenshots/30-selection-with-non-searchable.png)

With only searchable faces selected you can **Search other clips** or **Export selected**.

![Ready to search](docs/feature-screenshots/31-selection-searchable-ready.png)

## 12. Face detail drawer

**Details** opens a drawer with:

- the full frame with the face box, the crop, and the aligned 112 px face
- the grade, Searchable status, flag chips with explanations, and any grade-C cap reasons
- six sub-score bars: FIQA (eDifFIQA), sharpness, pose, illumination, occlusion, resolution
- measurements and hashes: PTS and frame number, wall time, inter-ocular distance, yaw/pitch/roll, Laplacian variance, luma and clipping, visible fraction, raw FIQA, attributes, and the SHA-256 of the frame, crop and aligned image

Possible flags: `MASK`, `FULL_COVER`, `SUNGLASSES`, `CAP_HAT`, `HELMET`, `HOOD_SCARF`, `PARTIAL_FACE`, `EXTREME_POSE`, `LOW_RES`, `MOTION_BLUR`, `LOW_LIGHT`, `BACKLIT`, `ENHANCED`. All demo faces carry `EXTREME_POSE`, because the camera sees the man in profile.

![Face detail](docs/feature-screenshots/25-face-detail-drawer.png)

**Jump to video** plays the proxy from that face's moment.

![Jump to video](docs/feature-screenshots/26-face-detail-jump-to-video.png)

## 13. Enhancement (labelled copies)

**Enhance…** creates a separate copy and never changes the evidence pixels. Each copy is stamped `ENHANCED … NOT ORIGINAL`, and enhanced copies are never used for matching. There are three methods:

CLAHE (local contrast):

![CLAHE](docs/feature-screenshots/27-enhance-clahe.png)

Gamma, with an adjustable value:

![Gamma](docs/feature-screenshots/28-enhance-gamma.png)

GFPGAN v1.4, which is generative. The dialog shows a red warning that it can invent detail and alter identity.

![GFPGAN](docs/feature-screenshots/29-enhance-gfpgan.png)

Earlier enhancements are listed in the dialog, and the face card shows an "N enhanced" badge.

## 14. Face search across clips

The search compares the selected faces' AdaFace embeddings against the FAISS index of the case's gallery clips. Results come in **Strong**, **Possible** and **Weak** bands, plus a **Not comparable** list, so that no indexed person is left out without notice. Band thresholds are shown on the page. The page also states that they are uncalibrated defaults (strong ≥ 0.45, possible ≥ 0.35, weak ≥ 0.25) until `scripts/calibrate.py` is run.

The 4 grade-B query faces found the same man in the gallery clip as a **strong** match (score 0.672).

![Search results](docs/feature-screenshots/32-search-results.png)

Hit details show the score, template similarity, top-3 face mean, wall time and flags.

![Search hit details](docs/feature-screenshots/33-search-hit-details.png)

**Play at moment** opens the gallery clip's proxy one second before the matched face.

![Play at moment](docs/feature-screenshots/34-search-play-at-moment.png)

## 15. Match review (confirm / reject)

Every match must be reviewed by a person. The reviewer adds a note and clicks **Possible match: confirm** or **Reject**. Only confirmed matches can be exported. When every non-weak match has been reviewed, the hint moves on to Export.

![Confirmed](docs/feature-screenshots/35-search-hit-confirmed.png)

A rejected match, on an earlier search of the same faces:

![Rejected](docs/feature-screenshots/36-search-hit-rejected.png)

## 16. Evidence export (PDF)

The export builder lists the best frames of confirmed customers and the reviewer-confirmed matches. Everything is pre-ticked. You can also include the enhanced images, which go in a separate labelled section.

![Export builder](docs/feature-screenshots/37-export-builder.png)

**Generate PDF** runs an `export` job (1.5 s here). The exports list shows each PDF's SHA-256 and a download link. The hash is also written to the audit log.

![Export done](docs/feature-screenshots/38-export-done.png)

The 6-page PDF that was generated (SHA-256 `9ecca637…5214133f`):

| Page | Content |
|---|---|
| 1 | Case header, who prepared it, config hash, model files with hashes, thresholds (marked uncalibrated), source clips with original, analysis and proxy hashes and their time mapping |
| 2–3 | Transaction, candidate table with parts and reasons, ROI overlay at the transaction time, best frames (full frame, crop, sub-score bars, flags, hashes) |
| 4 | Confirmed possible matches: query and match crops, band, score, reviewer, time, note |
| 5–6 | Enhanced images (bannered, with method, parameters and hash) and the disclaimer |

![PDF page 1](docs/feature-screenshots/40-evidence-pdf-page1.png)
![PDF page 2](docs/feature-screenshots/40-evidence-pdf-page2.png)
![PDF page 3](docs/feature-screenshots/40-evidence-pdf-page3.png)
![PDF page 4](docs/feature-screenshots/40-evidence-pdf-page4.png)
![PDF page 5](docs/feature-screenshots/40-evidence-pdf-page5.png)
![PDF page 6](docs/feature-screenshots/40-evidence-pdf-page6.png)

## 17. Case access and the viewer role

Under **Admin → Case access** you pick a case and tick the users who may open it. Each change is audited as `case_members_set`.

![Case access and audit](docs/feature-screenshots/42-admin-case-access-and-audit.png)

I made `viewer01` (viewer role) a member of `FIR-2026-1124`. Viewers only see cases they belong to. They have no Admin link and no New case button.

![Viewer cases](docs/feature-screenshots/44-viewer-cases.png)

Inside the case the viewer can look but cannot change anything: there is no upload card, no Index button, and no Confirm, Enhance or Generate buttons.

![Viewer case](docs/feature-screenshots/45-viewer-case-readonly.png)
![Viewer search](docs/feature-screenshots/46-viewer-search-readonly.png)

Access rules are checked on every API route and also against case membership. Media is served by id after the same checks, never from a static folder.

## 18. Audit log

Every action is appended to a hash-chained log: logins, failed logins, uploads, ingest results, time maps, confirmations, reviews, exports, membership changes and job failures. **Verify chain** recomputes the chain. Here it reported "Chain intact: 93 events verified".

You can filter the log by action. The screenshot shows the filter set to `search_hit_reviewed`, listing the review decisions.

![Audit filter](docs/feature-screenshots/43-audit-filter.png)

## 19. Background jobs

All GPU work goes through a queue served by one worker process, which loads every model once. Job types seen in this run:

| Job | What it does | Time on the demo |
|---|---|---|
| `ingest` | hash, probe, trial decode, normalise, proxy | 0.9–1.6 s |
| `timemap` | OCR / metadata time mapping | 0.0–18.7 s (OCR) |
| `coarse` | person detection and tracking at 5 fps | 0.8–2.6 s |
| `index` | face embeddings for gallery search | 3–17 s |
| `link` | rank candidates for a transaction | 0.09 s |
| `dense` | full-frame-rate face pass, scoring, flags, selection | 40.7 s |
| `behaviour` | rule-based behaviour events (queued after dense) | 0.28 s |
| `enhance` | CLAHE / gamma / GFPGAN copy | 0.02–1.6 s |
| `export` | PDF evidence sheet | 1.5 s |

Job cards float at the bottom right of every case page and show progress, a message and a **Cancel** button. Failed jobs stay visible in red. Interactive jobs run before batch indexing. If a job's heartbeat goes stale it is re-queued when the worker restarts.

## 20. Features without a UI screen

These exist in the code but have no page, so I could not take screenshots of them:

- **Behaviour events** (experimental shoulder-surfing and tamper rules). They are available at `GET/POST /api/clips/{id}/behaviour` and appear in the PDF when there are events. This run produced none.
- **CLI**: `atmcctv create-user`, and `python -m atmcctv.cli bench <clip>`.
- **Scripts**: model download, Kaggle fetch, building the format test set, labelling, calibration (`config/calibration.yaml`), evaluation, and the end-to-end smoke test.

## 21. Issues seen during the run

These are left as found and were not fixed:

1. **CLAHE and gamma copies of small crops are padded with black.** `add_banner` in `src/atmcctv/core/enhance.py` widens the image to at least 360 px so the banner text fits, and the extra width is filled with black (screenshots 27 and 28, PDF page 5). The UI also draws its own `ENHANCED · NOT ORIGINAL` label over the baked-in banner, so the two texts overlap.
2. **Wall time is missing on gallery matches** when the clip's time map is set after indexing. The gallery clip was indexed before I set its manual start time, so the hit details and the PDF show `wall -`. Re-indexing would probably fix it, but I did not test that.
3. **Overlapping text in the PDF header**: a long case title ("…ATM-01 (demo videos)") runs into the "Case ID" label on page 1.
4. **`raw_h264.264` shows a duration of 0.0 s** in its clip details, even though it was normalised and tracked.
5. The worker I started for this run exited once without any error output, shortly after the jobs were queued. A restart fixed it. Your own worker was restarted (through `scripts/demo/run_with_env.ps1`) around the same time, so the two may be connected.
6. Face 3 shows "6 enhanced", and there are two searches, because one capture script ran twice. This only affects the counts in screenshots 23 to 31 and 36.

## 22. How to re-run this setup

The demo data is still in `data/_featuredoc/`, which is git-ignored. The config override (`featuredoc.yaml`) and the capture scripts were written to `%TEMP%\fdoc\`. To start the same isolated instance, run these in three terminals:

```powershell
# featuredoc.yaml contents:
# app: { data_dir: "D:/CODING/Projects/ATM CCTV Detection/data/_featuredoc", port: 8010,
#        cors_origins: ["http://127.0.0.1:5180", "http://localhost:5180"] }
$env:ATMCCTV_SECRET = "<from .env>"; $env:ATMCCTV_CONFIG = "$env:TEMP\fdoc\featuredoc.yaml"
.venv\Scripts\python.exe -m uvicorn atmcctv.api.main:app --host 127.0.0.1 --port 8010   # terminal 1
.venv\Scripts\python.exe -m atmcctv.worker                                              # terminal 2 (same env vars)
cd frontend; npx vite --config vite.featuredoc.config.ts                                # terminal 3, see note
```

The Vite proxy target is hard-coded to port 8000 in `frontend/vite.config.ts`. For this run I used a temporary `frontend/vite.featuredoc.config.ts` and deleted it afterwards. To recreate it, copy `vite.config.ts` and change `port` to `5180` and the proxy `target` to `http://127.0.0.1:8010`.

Demo accounts in that database (local only): `admin / FeatureDoc-Admin-2026`, `investigator / Investigator-2026`, `viewer01 / Viewer-2026-x`.
