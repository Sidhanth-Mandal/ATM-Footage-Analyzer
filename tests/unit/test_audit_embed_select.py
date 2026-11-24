import numpy as np
import pytest
from sqlalchemy import text

from atmcctv import audit
from atmcctv.core.embed import NotEmbeddable, embed, template
from atmcctv.core.select import select_best
from atmcctv.core.types import FaceObs, QualityResult, ScoredFace
from atmcctv.db.session import init_db, read_session, write_session


def test_audit_append_and_verify_detects_tamper():
    init_db()
    with write_session() as s:
        for i in range(5):
            audit.append(s, f"act{i}", user_id=1, role="admin", object_type="x", object_id=i, details={"i": i, "f": 0.5})
    with read_session() as s:
        r = audit.verify(s)
    assert r["ok"] and r["checked"] >= 5
    with write_session() as s:
        last = s.exec(text("SELECT id FROM audit_event ORDER BY id DESC LIMIT 1 OFFSET 2")).first()[0]
        s.exec(text(f"UPDATE audit_event SET details_json='{{\"i\": 99, \"f\": 0.5}}' WHERE id={last}"))
    with read_session() as s:
        r = audit.verify(s)
    assert not r["ok"] and r["first_bad_id"] == last
    with write_session() as s:  # restore so later tests see a valid chain
        s.exec(text(f"DELETE FROM audit_event WHERE id>={last - 2}"))


def test_embed_rejects_enhanced_and_unsearchable():
    face = np.zeros((112, 112, 3), np.uint8)
    with pytest.raises(NotEmbeddable):
        embed(None, [face], enhanced=[True], searchable=[True])
    with pytest.raises(NotEmbeddable):
        embed(None, [face, face], enhanced=[False, False], searchable=[True, False])


def test_template_weighting():
    e = np.eye(3, dtype=np.float32)
    t, w = template(e[:2], [100, 50])
    assert w == pytest.approx([1.0, 0.25])
    assert np.linalg.norm(t) == pytest.approx(1.0)
    assert t[0] > t[1] > 0


def _sf(pts, total, yaw=0.0, grade=None):
    g = grade or ("A" if total >= 75 else "B" if total >= 55 else "C")
    obs = FaceObs(1, int(pts * 30), pts, np.zeros(4), np.zeros((5, 2)), 0.9, None, None, False)
    return ScoredFace(obs, QualityResult(total, g, {}, {"yaw": yaw}, [], [], g != "C"))


def test_select_temporal_nms_and_pose_diversity():
    faces = [_sf(1.0, 90, 5), _sf(1.1, 89, 5), _sf(1.2, 88, 5), _sf(2.0, 80, 5), _sf(3.0, 78, 5),
             _sf(4.0, 77, 5), _sf(5.0, 76, 5), _sf(6.0, 75, 5), _sf(7.0, 60, 35)]
    sel = select_best(faces, top_k=6, min_gap_s=0.5)
    assert len(sel) == 6
    pts = sorted(f.obs.pts for f in sel)
    assert all(b - a >= 0.5 for a, b in zip(pts, pts[1:]))
    assert any(25 <= abs(f.q.raw["yaw"]) <= 50 for f in sel), "three-quarter view must be included"
    assert any(abs(f.q.raw["yaw"]) <= 15 for f in sel)
