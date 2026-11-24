"""AdaFace embeddings and quality-weighted track templates.

Enhanced or non-searchable faces are never embedded.
"""
from __future__ import annotations

import numpy as np


class NotEmbeddable(ValueError):
    pass


def embed(registry, aligned: list[np.ndarray], *, enhanced: list[bool], searchable: list[bool],
          batch: int = 32) -> np.ndarray:
    if len(aligned) != len(enhanced) or len(aligned) != len(searchable):
        raise ValueError("aligned/enhanced/searchable length mismatch")
    if any(enhanced):
        raise NotEmbeddable("enhanced images must never be embedded or used for matching")
    if not all(searchable):
        raise NotEmbeddable("non-searchable faces (grade C, covered or low visibility) cannot be embedded")
    if not aligned:
        return np.zeros((0, 512), np.float32)
    m = registry.recognizer()
    blobs = [m.preprocess(a) for a in aligned]
    outs = registry.run_batched(m.session, m.input_name, m.output_names, blobs, batch)
    e = np.stack([np.asarray(o, np.float32).ravel() for o in outs])
    return l2n(e)


def l2n(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    if x.ndim == 1:
        n = np.linalg.norm(x)
        return x / n if n > 0 else x
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1
    return x / n


def template(embs: np.ndarray, totals: list[float]) -> tuple[np.ndarray, list[float]]:
    """t = normalise(sum w_i e_i), w_i = (total_i/100)^2."""
    w = np.array([(t / 100.0) ** 2 for t in totals], np.float32)
    if len(embs) == 0:
        raise ValueError("no embeddings for template")
    t = (w[:, None] * embs).sum(0)
    return l2n(t), w.tolist()
