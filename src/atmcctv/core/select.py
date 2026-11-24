"""Best-frame selection: score order, temporal NMS, pose diversity."""
from __future__ import annotations

from .types import ScoredFace


def _yaw(f: ScoredFace) -> float:
    return abs(float(f.q.raw.get("yaw", 0.0)))


def _frontal(f: ScoredFace) -> bool:
    return _yaw(f) <= 15


def _three_quarter(f: ScoredFace) -> bool:
    return 25 <= _yaw(f) <= 50


def select_best(faces: list[ScoredFace], top_k: int = 6, min_gap_s: float = 0.5,
                only_searchable: bool = False) -> list[ScoredFace]:
    pool = [f for f in faces if f.q.searchable] if only_searchable else list(faces)
    order = sorted(pool, key=lambda f: -f.q.total)
    chosen: list[ScoredFace] = []
    for f in order:
        if any(abs(f.obs.pts - c.obs.pts) < min_gap_s for c in chosen):
            continue
        chosen.append(f)
        if len(chosen) == top_k:
            break

    for cond in (_frontal, _three_quarter):
        if any(cond(c) for c in chosen):
            continue
        protected = [c for c in chosen if _frontal(c)][:1] + [c for c in chosen if _three_quarter(c)][:1]
        victim = None
        if len(chosen) >= top_k:
            removable = sorted((c for c in chosen if c not in protected), key=lambda c: c.q.total)
            if not removable:
                continue
            victim = removable[0]
        rest = [c for c in chosen if c is not victim]
        cand = next((f for f in order if cond(f) and f.q.grade in ("A", "B") and f not in chosen
                     and not any(abs(f.obs.pts - c.obs.pts) < min_gap_s for c in rest)), None)
        if cand is None:
            continue
        chosen = rest + [cand]
    chosen.sort(key=lambda f: -f.q.total)
    return chosen
