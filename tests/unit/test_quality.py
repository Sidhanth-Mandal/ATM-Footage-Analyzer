import math

import pytest

from atmcctv.core.quality import finalize, grade_of, sub_scores


def raw(**kw):
    base = dict(yaw=0.0, pitch=0.0, roll=0.0, iod_px=60.0, lap_var=500.0, mean_luma=0.5, clip_frac=0.0,
                lr_ratio=1.0, visible_frac=0.9, fiqa_raw=0.80)
    base.update(kw)
    return base


def test_perfect_face_scores_100(cfg):
    s = sub_scores(raw(), cfg)
    assert all(v == pytest.approx(1.0) for v in s.values())
    q = finalize(raw(), [], cfg, touches_edge=False, head_out_of_frame=False)
    assert q.total == pytest.approx(100.0)
    assert q.grade == "A" and q.searchable and not q.caps


def test_sub_score_formulas(cfg):
    s = sub_scores(raw(lap_var=20.0, yaw=60.0, pitch=0.0, iod_px=37.5, visible_frac=0.6, fiqa_raw=0.525), cfg)
    assert s["sharpness"] == pytest.approx(0.0)
    assert s["pose"] == pytest.approx(0.0)
    assert s["resolution"] == pytest.approx(0.5)
    assert s["occlusion"] == pytest.approx(0.5)
    assert s["fiqa"] == pytest.approx(0.5)  # default bounds (0.25, 0.80)
    s = sub_scores(raw(yaw=30.0, pitch=20.0), cfg)
    assert s["pose"] == pytest.approx(1 - math.sqrt(0.25 + 0.25))


def test_illumination(cfg):
    assert sub_scores(raw(mean_luma=0.75), cfg)["illumination"] == pytest.approx(1.0)
    assert sub_scores(raw(mean_luma=1.0), cfg)["illumination"] == pytest.approx(0.0)
    assert sub_scores(raw(clip_frac=0.2), cfg)["illumination"] == pytest.approx(0.0)
    assert sub_scores(raw(lr_ratio=2.0), cfg)["illumination"] == pytest.approx(0.0)
    e, k, y = 1.0, 0.5, 1.0
    assert sub_scores(raw(clip_frac=0.1), cfg)["illumination"] == pytest.approx((e * k * y) ** (1 / 3))


@pytest.mark.parametrize("kw,edge,hoof,cap", [
    (dict(iod_px=14.0), False, False, "IOD"),
    ({}, True, False, "edge"),
    ({}, False, True, "head out of frame"),
    (dict(visible_frac=0.30), False, False, "visible"),
])
def test_caps_force_grade_c(cfg, kw, edge, hoof, cap):
    q = finalize(raw(**kw), [], cfg, touches_edge=edge, head_out_of_frame=hoof)
    assert q.total <= 54
    assert q.grade == "C"
    assert not q.searchable
    assert any(cap in c for c in q.caps)


def test_grade_boundaries(cfg):
    assert grade_of(75.0, cfg) == "A"
    assert grade_of(74.99, cfg) == "B"
    assert grade_of(55.0, cfg) == "B"
    assert grade_of(54.99, cfg) == "C"


def test_full_cover_not_searchable(cfg):
    q = finalize(raw(), ["MASK", "FULL_COVER"], cfg, touches_edge=False, head_out_of_frame=False)
    assert q.grade == "A" and not q.searchable


def test_low_visible_not_searchable(cfg):
    q = finalize(raw(visible_frac=0.45), [], cfg, touches_edge=False, head_out_of_frame=False)
    assert q.grade in ("A", "B") and not q.searchable
