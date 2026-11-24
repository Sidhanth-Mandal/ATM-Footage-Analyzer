import numpy as np

from atmcctv.core.linking import point_in_polygon, score_tracks
from atmcctv.core.types import PersonTrack, TrackSample

W, H = 1920, 1080


def track(tid, t0, t1, box, kpts=None, fps=5):
    samples = []
    for i in range(int((t1 - t0) * fps) + 1):
        k = np.zeros((17, 3), np.float32) if kpts is None else kpts.copy()
        b = np.asarray(box, np.float32)
        samples.append(TrackSample(t0 + i / fps, b, k, 0.9, bool(b[1] <= 2)))
    return PersonTrack(tid, "person", samples)


def test_point_in_polygon_normalised():
    sq = [[0.2, 0.2], [0.6, 0.2], [0.6, 0.6], [0.2, 0.6]]
    assert point_in_polygon(0.4, 0.4, sq)
    assert not point_in_polygon(0.7, 0.4, sq)
    assert not point_in_polygon(0.4, 0.1, sq)
    tri = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    assert point_in_polygon(0.2, 0.2, tri) and not point_in_polygon(0.8, 0.8, tri)
    assert not point_in_polygon(0.5, 0.5, None)


def test_fascia_far_pedestrian_never_ranks(cfg):
    far = track(1, 0, 60, [900, 400, 960, 560])          # small person seen through the doorway
    near = track(2, 20, 50, [500, 0, 1400, 1080])         # fills the frame at the machine
    c = score_tracks([far, near], 40.0, 90, 30, {}, "fascia", W, H, cfg)
    assert [x.track_id for x in c] == [2]


def test_fascia_track_inside_window_ranks_first(cfg):
    early = track(1, 0, 10, [500, 0, 1400, 1080])         # customer long before the transaction
    cust = track(2, 100, 140, [500, 0, 1400, 1080])
    c = score_tracks([early, cust], 130.0, 30, 30, {}, "fascia", W, H, cfg)
    assert c[0].track_id == 2


def test_lobby_hand_in_keypad(cfg):
    rois = {"atm_zone": [[0.3, 0.3], [0.7, 0.3], [0.7, 0.9], [0.3, 0.9]],
            "keypad": [[0.45, 0.5], [0.55, 0.5], [0.55, 0.6], [0.45, 0.6]]}
    k_use = np.zeros((17, 3), np.float32)
    k_use[[5, 6, 11, 12]] = [[0.45 * W, 0.4 * H, 1], [0.55 * W, 0.4 * H, 1], [0.46 * W, 0.7 * H, 1], [0.54 * W, 0.7 * H, 1]]
    k_use[9] = [0.5 * W, 0.55 * H, 1]
    k_stand = k_use.copy()
    k_stand[9] = [0.35 * W, 0.8 * H, 1]
    k_stand[10] = [0.65 * W, 0.8 * H, 1]
    user = track(1, 0, 30, [0.4 * W, 0.2 * H, 0.6 * W, 0.95 * H], k_use)
    waiter = track(2, 0, 30, [0.4 * W, 0.2 * H, 0.6 * W, 0.95 * H], k_stand)
    outside = track(3, 0, 30, [0.05 * W, 0.2 * H, 0.2 * W, 0.95 * H], None)
    c = score_tracks([waiter, user, outside], 20.0, 90, 30, rois, "lobby", W, H, cfg)
    assert c[0].track_id == 1
    assert c[0].parts["hand"] == 1.0
    assert 3 not in [x.track_id for x in c]


def test_ambiguous_reason_on_tie(cfg):
    a = track(1, 20, 50, [500, 0, 1400, 1080])
    b = track(2, 20, 50, [500, 0, 1400, 1080])
    c = score_tracks([a, b], 40.0, 90, 30, {}, "fascia", W, H, cfg)
    assert all(x.reasons[0].startswith("ambiguous") for x in c)
