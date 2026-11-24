"""Golden tests (§14.5): full core pipeline on three short clips compared with stored reports.

Requires the GPU models (scripts/download_models.py) and the sample/format files
(scripts/make_formats.py). Regenerate expectations after an intended change with:
    $env:ATMCCTV_UPDATE_GOLDEN=1; python -m pytest tests/golden
"""
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).parent / "expected"
CLIPS = {
    "kaggle_mp4": ROOT / "samples" / "kaggle_atm_video" / "handsign_1.mp4",
    "converted_avi": ROOT / "samples" / "formats" / "avi_mpeg4.avi",
    "raw_264": ROOT / "samples" / "formats" / "raw_h264.264",
}
UPDATE = os.environ.get("ATMCCTV_UPDATE_GOLDEN") == "1"


def _have_models() -> bool:
    return (ROOT / "models" / "yolo26s-pose.pt").exists() and (ROOT / "models" / "uniface" / "scrfd_10g.onnx").exists()


def _summary(rep: dict) -> dict:
    fps = float(rep["probe"]["avg_fps"] or rep["probe"]["r_fps"] or 25)
    return {
        "fps": fps,
        "normalised": any("normalised" in d for d in rep["decisions"]),
        "candidates": [c["track_id"] for c in rep["candidates"]],
        "faces": [{"pts": f["pts"], "grade": f["grade"], "total": f["total"], "flags": sorted(f["flags"])} for f in rep["faces"]],
    }


@pytest.mark.skipif(not _have_models(), reason="models not downloaded")
@pytest.mark.parametrize("name", list(CLIPS))
def test_golden(name, tmp_path):
    clip = CLIPS[name]
    if not clip.exists():
        pytest.skip(f"{clip} missing (run scripts/make_formats.py)")
    from atmcctv.cli import _pipeline
    rep = _pipeline(clip, tmp_path, None, 0.0, "fascia", "UTC", None, 6)
    got = _summary(rep)
    exp_p = EXP / f"{name}.json"
    if UPDATE or not exp_p.exists():
        EXP.mkdir(parents=True, exist_ok=True)
        exp_p.write_text(json.dumps(got, indent=1))
        if not UPDATE:
            pytest.skip("golden file created; re-run to compare")
        return
    exp = json.loads(exp_p.read_text())
    assert got["normalised"] == exp["normalised"]
    assert got["candidates"][:1] == exp["candidates"][:1], "top candidate changed"
    tol = 1.0 / exp["fps"] + 1e-3
    assert len(got["faces"]) == len(exp["faces"])
    for e in exp["faces"]:
        m = [g for g in got["faces"] if abs(g["pts"] - e["pts"]) <= tol]
        assert m, f"selected frame at {e['pts']:.3f}s missing"
        g = m[0]
        assert g["grade"] == e["grade"]
        assert g["flags"] == e["flags"]
        assert abs(g["total"] - e["total"]) <= 3.0
