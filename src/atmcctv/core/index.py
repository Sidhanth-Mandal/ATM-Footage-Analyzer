"""FAISS indexes: per-face vectors keyed by face.id and track templates keyed by track.id."""
from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

DIM = 512
DEFAULT_BANDS = {"strong": 0.45, "possible": 0.35, "weak": 0.25}  # uncalibrated defaults (AdaFace cosine)


def _new():
    import faiss
    return faiss.IndexIDMap2(faiss.IndexFlatIP(DIM))


class FaceIndex:
    def __init__(self, index_dir: Path):
        import faiss

        self.faiss = faiss
        self.dir = Path(index_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.faces_path = self.dir / "faces.faiss"
        self.tmpl_path = self.dir / "templates.faiss"
        self.lock = threading.RLock()
        self.faces = faiss.read_index(str(self.faces_path)) if self.faces_path.exists() else _new()
        self.templates = faiss.read_index(str(self.tmpl_path)) if self.tmpl_path.exists() else _new()

    def save(self) -> None:
        with self.lock:
            self.faiss.write_index(self.faces, str(self.faces_path))
            self.faiss.write_index(self.templates, str(self.tmpl_path))

    def rebuild(self, face_vecs: dict[int, np.ndarray], tmpl_vecs: dict[int, np.ndarray]) -> None:
        with self.lock:
            self.faces, self.templates = _new(), _new()
            self._add(self.faces, face_vecs)
            self._add(self.templates, tmpl_vecs)
            self.save()

    def consistent(self, n_faces: int, n_templates: int) -> bool:
        return self.faces.ntotal == n_faces and self.templates.ntotal == n_templates

    @staticmethod
    def _add(idx, vecs: dict[int, np.ndarray]) -> None:
        if not vecs:
            return
        ids = np.array(list(vecs.keys()), np.int64)
        x = np.stack([np.asarray(v, np.float32).ravel() for v in vecs.values()])
        idx.remove_ids(ids)
        idx.add_with_ids(x, ids)

    def add(self, face_vecs: dict[int, np.ndarray], tmpl_vecs: dict[int, np.ndarray]) -> None:
        with self.lock:
            self._add(self.faces, face_vecs)
            self._add(self.templates, tmpl_vecs)
            self.save()

    def remove(self, face_ids: list[int], track_ids: list[int]) -> None:
        with self.lock:
            if face_ids:
                self.faces.remove_ids(np.array(face_ids, np.int64))
            if track_ids:
                self.templates.remove_ids(np.array(track_ids, np.int64))
            self.save()

    def search(self, q: np.ndarray, face_to_track: dict[int, int], k_faces: int = 2000
               ) -> dict[int, dict]:
        """Return {track_id: {score, template_sim, top3_mean, top_face_ids}} for every indexed track."""
        q = np.asarray(q, np.float32).reshape(1, -1)
        out: dict[int, dict] = {}
        with self.lock:
            if self.templates.ntotal:
                d, i = self.templates.search(q, self.templates.ntotal)
                for s, tid in zip(d[0], i[0]):
                    if tid >= 0:
                        out.setdefault(int(tid), {})["template_sim"] = float(s)
            if self.faces.ntotal:
                k = min(k_faces, self.faces.ntotal)
                d, i = self.faces.search(q, k)
                per: dict[int, list[tuple[float, int]]] = {}
                for s, fid in zip(d[0], i[0]):
                    if fid < 0:
                        continue
                    tid = face_to_track.get(int(fid))
                    if tid is None:
                        continue
                    per.setdefault(tid, []).append((float(s), int(fid)))
                for tid, lst in per.items():
                    lst.sort(reverse=True)
                    top = lst[:3]
                    o = out.setdefault(tid, {})
                    o["top3_mean"] = float(np.mean([s for s, _ in top]))
                    o["top_face_ids"] = [f for _, f in top]
        for tid, o in out.items():
            o["score"] = max(o.get("template_sim", -1.0), o.get("top3_mean", -1.0))
            o.setdefault("top_face_ids", [])
        return out


def band_of(score: float, bands: dict) -> str | None:
    b = {k: (v if v is not None else DEFAULT_BANDS[k]) for k, v in (bands or {}).items()} or DEFAULT_BANDS
    for name in ("strong", "possible", "weak"):
        if score >= float(b.get(name, DEFAULT_BANDS[name])):
            return name
    return None


def effective_bands(bands: dict | None) -> tuple[dict, bool]:
    """Return (bands, calibrated?)."""
    bands = bands or {}
    calibrated = all(bands.get(k) is not None for k in ("strong", "possible", "weak"))
    return {k: float(bands.get(k) if bands.get(k) is not None else DEFAULT_BANDS[k]) for k in DEFAULT_BANDS}, calibrated
