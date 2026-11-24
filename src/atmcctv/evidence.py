"""Evidence sheet PDF (ReportLab platypus)."""
from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)
from sqlmodel import Session, select

from . import __version__, paths
from .core.flags import FLAG_HELP
from .core.index import effective_bands
from .core.timemap import from_epoch
from .core.video import FrameSource
from .db.models import (
    Atm, BehaviourEvent, CandidateRow, Case, Clip, Enhancement, Export, FaceRow, Search, SearchHit, Track,
    Transaction, User,
)
from .settings import config_hash

DISCLAIMER = (
    "This sheet presents investigative leads produced by automated video analysis and reviewed by the named "
    "investigator. It is not an identification. Every candidate and possible match requires independent human "
    "verification before any action is taken. Quality scores describe image suitability, not identity. Enhanced "
    "images (if any) are clearly labelled, are not original pixels and were never used for matching."
)

SS = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=SS["Heading1"], fontSize=15, spaceAfter=4)
H2 = ParagraphStyle("h2", parent=SS["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=4,
                    textColor=colors.HexColor("#1f3a5f"))
BODY = ParagraphStyle("b", parent=SS["BodyText"], fontSize=8.5, leading=11)
SMALL = ParagraphStyle("s", parent=BODY, fontSize=7.2, leading=9)
WARN = ParagraphStyle("w", parent=BODY, textColor=colors.HexColor("#9a1c1c"))
CENTER = ParagraphStyle("c", parent=SMALL, alignment=TA_CENTER)


def _img_from_array(img: np.ndarray, max_w: float, max_h: float) -> Image:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h)
    return Image(io.BytesIO(buf.tobytes()), width=w * s, height=h * s)


def _img_from_path(p: Path | None, max_w: float, max_h: float) -> Image | Paragraph:
    if p is None or not p.exists():
        return Paragraph("(image missing)", SMALL)
    img = cv2.imread(str(p))
    if img is None:
        return Paragraph("(unreadable image)", SMALL)
    return _img_from_array(img, max_w, max_h)


def _frame_with_box(face: FaceRow, max_w: float, max_h: float):
    p = paths.absp(face.frame_path)
    if not p or not p.exists():
        return Paragraph("(frame missing)", SMALL)
    img = cv2.imread(str(p))
    x0, y0, x1, y1 = [int(v) for v in face.bbox_json]
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 255), max(2, img.shape[1] // 300))
    return _img_from_array(img, max_w, max_h)


def _bars(sub: dict) -> Drawing:
    keys = ["fiqa", "sharpness", "pose", "illumination", "occlusion", "resolution"]
    d = Drawing(62 * mm, 6 * 4.2 * mm)
    for i, k in enumerate(keys):
        y = (5 - i) * 4.2 * mm
        v = float(sub.get(k, 0.0))
        d.add(String(0, y + 1, k, fontSize=6.5))
        d.add(Rect(20 * mm, y, 36 * mm, 3 * mm, fillColor=colors.HexColor("#e5e7eb"), strokeColor=None))
        col = "#16a34a" if v >= 0.66 else "#d97706" if v >= 0.33 else "#dc2626"
        d.add(Rect(20 * mm, y, 36 * mm * v, 3 * mm, fillColor=colors.HexColor(col), strokeColor=None))
        d.add(String(57 * mm, y + 1, f"{v:.2f}", fontSize=6.5))
    return d


def _tbl(data, widths, header=True, font=7.2):
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    st = [("FONTSIZE", (0, 0), (-1, -1), font), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#9ca3af")),
          ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(st))
    return t


def _roi_snapshot(clip: Clip, atm: Atm | None, pts: float | None, max_w: float, max_h: float):
    try:
        rot = int((clip.probe_json or {}).get("analysis_rotation", 0))
        with FrameSource(paths.absp(clip.analysis_path), rot) as fs:
            t = pts if pts is not None and 0 <= pts <= (fs.duration or 1e9) else 0.0
            img = fs.get(t)
    except Exception:
        img = None
    if img is None:
        return Paragraph("(no frame)", SMALL)
    H, W = img.shape[:2]
    cols = {"keypad": (0, 200, 0), "cash_slot": (0, 140, 255), "card_slot": (255, 0, 200), "atm_zone": (255, 200, 0)}
    for name, poly in ((atm.rois_json if atm else {}) or {}).items():
        if not poly:
            continue
        p = (np.asarray(poly, np.float32) * [W, H]).astype(np.int32)
        cv2.polylines(img, [p], True, cols.get(name, (255, 255, 255)), max(2, W // 400))
        cv2.putText(img, name, tuple(p[0]), cv2.FONT_HERSHEY_SIMPLEX, max(0.6, W / 1600), cols.get(name, (255, 255, 255)), 2)
    return _img_from_array(img, max_w, max_h)


def _fmt_s(v) -> str:
    return "-" if v is None else f"{float(v):.2f} s"


def build_pdf(s: Session, ex: Export, out: Path, cfg, model_info: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    case = s.get(Case, ex.case_id)
    creator = s.get(User, ex.created_by) if ex.created_by else None
    sel = ex.selection_json or {}
    face_ids = [int(x) for x in sel.get("face_ids", [])]
    hit_ids = [int(x) for x in sel.get("search_hit_ids", [])]
    include_enh = bool(sel.get("include_enhanced", False))
    faces = [f for f in (s.get(FaceRow, i) for i in face_ids) if f]
    hits = [h for h in (s.get(SearchHit, i) for i in hit_ids) if h and h.review == "confirmed"]
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    W = A4[0] - 30 * mm
    story = []

    story.append(Paragraph("ATM CCTV evidence sheet: investigative lead (not an identification)", H1))
    story.append(_tbl([
        ["Case", f"{case.ref_no}: {case.title}", "Case ID", case.id],
        ["Prepared by", creator.username if creator else "-", "Generated (UTC)", generated],
        ["App version", __version__, "Config SHA-256", config_hash(cfg)[:32]],
        ["Export ID", str(ex.id), "Selection", f"{len(faces)} faces, {len(hits)} confirmed matches"],
    ], [28 * mm, W / 2 - 28 * mm, 30 * mm, W / 2 - 30 * mm], header=False))
    story.append(Spacer(1, 3 * mm))
    bands, calibrated = effective_bands(dict(cfg.search.bands))
    mi = [["Model", "File", "SHA-256 (16)"]] + [[k, v.get("file", "-"), v.get("sha256") or "-"]
                                                 for k, v in sorted(model_info.items())]
    story.append(Paragraph("Models and thresholds", H2))
    story.append(_tbl(mi, [35 * mm, 70 * mm, W - 105 * mm]))
    story.append(Paragraph(
        f"Grades: A &ge; {cfg.quality.grade_a}, B &ge; {cfg.quality.grade_b}, C below. Search bands (cosine): "
        f"strong &ge; {bands['strong']:.3f}, possible &ge; {bands['possible']:.3f}, weak &ge; {bands['weak']:.3f}"
        + ("" if calibrated else " <b>(uncalibrated defaults)</b>") + ".", BODY))

    # ---- sources
    clip_ids = []
    for f in faces:
        if f.clip_id not in clip_ids:
            clip_ids.append(f.clip_id)
    for h in hits:
        tr = s.get(Track, h.track_id)
        if tr and tr.clip_id not in clip_ids:
            clip_ids.append(tr.clip_id)
    clips = {cid: s.get(Clip, cid) for cid in clip_ids}
    story.append(Paragraph("Sources", H2))
    rows = [["Clip", "Original SHA-256", "Analysis / proxy SHA-256", "Video", "Time mapping"]]
    for c in clips.values():
        p = c.probe_json or {}
        tm = c.timemap_json or {}
        rows.append([
            Paragraph(f"{c.orig_name}<br/>{c.kind}, {c.id[:8]}", SMALL),
            Paragraph(c.orig_sha256, SMALL),
            Paragraph(f"{c.analysis_sha256 or '-'}<br/>{c.proxy_sha256 or '-'}<br/>normalised: {c.normalised}", SMALL),
            Paragraph(f"{p.get('format_name', '?')} / {p.get('video_codec', '?')} {p.get('width')}x{p.get('height')}"
                      f"<br/>{p.get('avg_fps', 0):.3f} fps, {p.get('duration_s', 0):.1f} s", SMALL),
            Paragraph(f"{tm.get('method', 'none')}; inliers {tm.get('inliers', 0)}; residual "
                      f"{tm.get('residual_s', 0):.2f} s<br/>clock offset {c.clock_offset_s:+.1f} s (DVR - journal)"
                      f"<br/>tz {tm.get('tz', '-')}", SMALL),
        ])
    story.append(_tbl(rows, [32 * mm, 40 * mm, 45 * mm, 30 * mm, W - 147 * mm]))

    # ---- transactions + best frames grouped by track
    by_track: dict[int, list[FaceRow]] = {}
    for f in faces:
        by_track.setdefault(f.track_id, []).append(f)
    for tid, fl in by_track.items():
        tr = s.get(Track, tid)
        clip = clips[tr.clip_id]
        atm = s.get(Atm, clip.atm_id) if clip.atm_id else None
        txns = s.exec(select(Transaction).where(Transaction.chosen_track_id == tid)).all()
        story.append(PageBreak())
        story.append(Paragraph(f"Track {tr.track_no} in {clip.orig_name}", H2))
        for t in txns:
            confirmer = s.get(User, t.confirmed_by) if t.confirmed_by else None
            story.append(Paragraph(
                f"Transaction <b>{t.txn_ref or t.id}</b>: journal time {t.txn_time_journal.isoformat(sep=' ')}"
                f"{'' if t.amount is None else f', amount {t.amount:,.2f}'}; video PTS {_fmt_s(t.t_txn_pts)}; window "
                f"-{t.window_before_s:.0f} s / +{t.window_after_s:.0f} s. Candidate confirmed by "
                f"<b>{confirmer.username if confirmer else '-'}</b> at {t.confirmed_at.isoformat(sep=' ', timespec='seconds') if t.confirmed_at else '-'} (UTC).",
                BODY))
            cands = s.exec(select(CandidateRow).where(CandidateRow.transaction_id == t.id).order_by(CandidateRow.rank)).all()
            crow = [["Rank", "Track", "Score", "Parts", "Reasons"]]
            for c in cands:
                ct = s.get(Track, c.track_id)
                crow.append([str(c.rank), f"{ct.track_no}{' (chosen)' if ct.id == tid else ''}", f"{c.score:.3f}",
                             Paragraph(", ".join(f"{k} {v:.2f}" for k, v in c.parts_json.items()), SMALL),
                             Paragraph("; ".join(c.reasons_json), SMALL)])
            story.append(_tbl(crow, [11 * mm, 20 * mm, 14 * mm, 55 * mm, W - 100 * mm]))
            story.append(Spacer(1, 2 * mm))
            story.append(KeepTogether([Paragraph("ATM regions of interest at the transaction time", SMALL),
                                       _roi_snapshot(clip, atm, t.t_txn_pts, W * 0.6, 60 * mm)]))
        if tr.face_visible_frac is not None:
            story.append(Paragraph(f"Face visible in {tr.face_visible_frac:.0%} of the frames of this track.", BODY))
        story.append(Paragraph("Best frames (original pixels)", H2))
        for f in sorted(fl, key=lambda x: -x.total):
            flags = ", ".join(f.flags_json) or "none"
            meta = Paragraph(
                f"<b>Face {f.id}</b> | grade <b>{f.grade}</b> | total <b>{f.total:.1f}</b>/100<br/>"
                f"PTS {f.pts:.3f} s | frame {f.frame_idx}<br/>wall time {f.wall_time or '-'}<br/>"
                f"IOD {f.raw_json.get('iod_px', 0):.0f} px | yaw {f.raw_json.get('yaw', 0):+.0f} | pitch "
                f"{f.raw_json.get('pitch', 0):+.0f}<br/>flags: <b>{flags}</b>"
                + (f"<br/>caps: {'; '.join(f.caps_json)}" if f.caps_json else "")
                + f"<br/>searchable: {'yes' if f.searchable else 'no'}"
                + f"<br/>frame SHA-256 {(f.frame_sha256 or '-')[:24]}<br/>crop SHA-256 {(f.crop_sha256 or '-')[:24]}",
                SMALL)
            row = [[_frame_with_box(f, 62 * mm, 40 * mm), _img_from_path(paths.absp(f.crop_path), 30 * mm, 40 * mm),
                    [meta, _bars(f.sub_json)]]]
            story.append(KeepTogether([_tbl(row, [64 * mm, 32 * mm, W - 96 * mm], header=False)]))
            story.append(Spacer(1, 2 * mm))
        used = sorted({fl_ for f in fl for fl_ in f.flags_json})
        if used:
            story.append(Paragraph("Flag meanings: " + "; ".join(f"<b>{k}</b> {FLAG_HELP.get(k, '')}" for k in used), SMALL))

    # ---- confirmed matches
    if hits:
        story.append(PageBreak())
        story.append(Paragraph("Possible matches confirmed by a reviewer (require independent verification)", H2))
        for h in hits:
            srch = s.get(Search, h.search_id)
            q = s.get(FaceRow, (srch.query_face_ids_json or [None])[0]) if srch.query_face_ids_json else None
            m = s.get(FaceRow, (h.top_face_ids_json or [None])[0]) if h.top_face_ids_json else None
            tr = s.get(Track, h.track_id)
            mc = s.get(Clip, tr.clip_id)
            rv = s.get(User, h.reviewed_by) if h.reviewed_by else None
            info = Paragraph(
                f"Band <b>{h.band}</b>, score {h.score:.3f}<br/>match: {mc.orig_name}, track {tr.track_no}"
                f"{'' if not m else f', PTS {m.pts:.2f} s, wall {m.wall_time or "-"}'}<br/>reviewed by "
                f"<b>{rv.username if rv else '-'}</b> at {h.reviewed_at.isoformat(sep=' ', timespec='seconds') if h.reviewed_at else '-'}"
                f"<br/>note: {h.note or '-'}", SMALL)
            story.append(KeepTogether([_tbl([[
                [Paragraph("query", CENTER), _img_from_path(paths.absp(q.crop_path) if q else None, 35 * mm, 40 * mm)],
                [Paragraph("match", CENTER), _img_from_path(paths.absp(m.crop_path) if m else None, 35 * mm, 40 * mm)],
                info]], [40 * mm, 40 * mm, W - 80 * mm], header=False), Spacer(1, 2 * mm)]))

    # ---- enhanced (separate, labelled)
    if include_enh and faces:
        enh = s.exec(select(Enhancement).where(Enhancement.face_id.in_([f.id for f in faces]))).all()
        enh = [e for e in enh if e.path]
        if enh:
            story.append(PageBreak())
            story.append(Paragraph("ENHANCED IMAGES: NOT ORIGINAL PIXELS, NOT USED FOR MATCHING", H2))
            story.append(Paragraph("Shown for orientation only. Generative restoration can alter identity.", WARN))
            for e in enh:
                story.append(KeepTogether([_tbl([[
                    _img_from_path(paths.absp(e.path), 60 * mm, 60 * mm),
                    Paragraph(f"Enhancement {e.id} of face {e.face_id}<br/>method <b>{e.method}</b><br/>params "
                              f"{e.params_json}<br/>SHA-256 {e.sha256[:32] if e.sha256 else '-'}", SMALL)]],
                    [64 * mm, W - 64 * mm], header=False), Spacer(1, 2 * mm)]))

    # ---- behaviour (experimental)
    events = s.exec(select(BehaviourEvent).where(BehaviourEvent.clip_id.in_(list(clips.keys())))).all() if clips else []
    if events:
        story.append(Paragraph("Behaviour events (experimental, rule-based)", H2))
        rows = [["Clip", "Type", "Interval (PTS)", "Tracks", "Details"]]
        for e in events:
            c = clips[e.clip_id]
            tm = c.timemap_json
            wall = ""
            if tm:
                wall = " | " + from_epoch(tm["a"] + tm["b"] * e.t_start, tm["tz"]).strftime("%Y-%m-%d %H:%M:%S")
            rows.append([Paragraph(c.orig_name, SMALL), e.type, f"{e.t_start:.1f}-{e.t_end:.1f} s{wall}",
                         ", ".join(str(x) for x in (e.details_json or {}).get("track_nos", [])),
                         Paragraph(", ".join(f"{k}: {v}" for k, v in (e.details_json or {}).items() if k != "track_nos"), SMALL)])
        story.append(_tbl(rows, [35 * mm, 25 * mm, 40 * mm, 15 * mm, W - 115 * mm]))

    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph("Disclaimer", H2))
    story.append(Paragraph(DISCLAIMER, BODY))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.drawString(15 * mm, 8 * mm, f"Case {case.ref_no} | export {ex.id} | generated {generated} | "
                                           "investigative lead, requires human verification")
        canvas.drawRightString(A4[0] - 15 * mm, 8 * mm, f"page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, topMargin=14 * mm,
                            bottomMargin=14 * mm, title=f"Evidence sheet {case.ref_no}", author="atmcctv")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
