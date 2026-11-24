import random
from datetime import datetime, timezone

import pytest

from atmcctv.core import timemap as tm


def test_ransac_with_20pct_corrupted_reads():
    rng = random.Random(1)
    a_true, b_true = 1_772_500_000.0, 1.0004
    pts = [i * 2.0 for i in range(40)]
    walls = [a_true + b_true * p + rng.uniform(-0.5, 0.5) for p in pts]   # 1 s resolution reads
    bad = rng.sample(range(40), 8)
    for i in bad:
        walls[i] += rng.choice([-1, 1]) * rng.uniform(30, 3600)           # misreads
    a, b, inl, res = tm.ransac_fit(pts, walls, tol=1.5, iters=300, min_inliers=5, min_ratio=0.6)
    assert inl >= 30
    assert abs(b - b_true) < 0.002
    for p in (0.0, 40.0, 78.0):
        assert abs((a + b * p) - (a_true + b_true * p)) < 1.0


def test_ransac_rejects_garbage():
    pts = [0, 2, 4, 6, 8, 10]
    walls = [5, 900, 33, 1e6, 12, 77]
    assert tm.ransac_fit(pts, walls) is None


def test_parse_formats_and_confusions():
    assert tm.parse_timestamp("2026-03-03 08:42:10") == datetime(2026, 3, 3, 8, 42, 10)
    assert tm.parse_timestamp("03-04-2026 Tue 08:42:10", "DMY") == datetime(2026, 4, 3, 8, 42, 10)
    assert tm.parse_timestamp("03/04/2026 08:42:10 PM", "MDY") == datetime(2026, 3, 4, 20, 42, 10)
    assert tm.parse_timestamp("2O26-O3-O3 O8:4Z:1O") == datetime(2026, 3, 3, 8, 42, 10)
    assert tm.parse_timestamp("12/01/2026 12:05:00 AM", "DMY") == datetime(2026, 1, 12, 0, 5, 0)
    assert tm.parse_timestamp("no time here") is None


def test_metadata_start_vs_end():
    ct = datetime(2026, 3, 3, 3, 12, 17, tzinfo=timezone.utc)
    start = tm.from_metadata(ct, 7.0, False, "UTC")
    end = tm.from_metadata(ct, 7.0, True, "UTC")
    assert start.wall(0.0) == pytest.approx(ct.timestamp())
    assert end.wall(7.0) == pytest.approx(ct.timestamp())
    assert start.b == end.b == 1.0


def test_offset_sign_convention():
    # DVR clock runs 120 s fast vs the bank journal: clock_offset_s = DVR - journal = +120
    m = tm.manual(datetime(2026, 3, 3, 10, 0, 0), "Asia/Kolkata")       # DVR wall time at pts 0
    journal = datetime(2026, 3, 3, 10, 1, 0)                             # journal says 10:01:00
    pts = tm.txn_to_pts(journal, 120.0, m)                               # on DVR clock this is 10:03:00
    assert pts == pytest.approx(180.0)
    assert tm.txn_to_pts(journal, -30.0, m) == pytest.approx(30.0)
