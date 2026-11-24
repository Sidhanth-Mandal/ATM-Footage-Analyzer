"""Model runtime: each model is loaded once (lazily) in the GPU worker process.

`torch` is imported before any ONNX Runtime session so the CUDA/cuDNN DLLs
shipped with the torch wheels are on the loader path.
"""
from __future__ import annotations

import gc
import hashlib
import logging
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

from ..settings import Cfg, get_config, models_dir

log = logging.getLogger(__name__)


class GPUFallbackError(RuntimeError):
    pass


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class ModelRegistry:
    def __init__(self, cfg: Cfg | None = None):
        self.cfg = cfg or get_config()
        self.mdir = models_dir(self.cfg)
        os.environ.setdefault("UNIFACE_CACHE_DIR", str(self.mdir / "uniface"))
        os.environ.setdefault("YOLO_OFFLINE", "1")
        self._lock = threading.RLock()
        self._models: dict[str, Any] = {}
        self.providers: dict[str, str] = {}
        self.files: dict[str, str] = {}
        self._torch = None
        self._yoloe_classes: list[str] | None = None
        self.hold_open_vocab = False

    # ------------------------------------------------------------------ helpers
    @property
    def torch(self):
        if self._torch is None:
            import torch
            self._torch = torch
        return self._torch

    @property
    def precision(self) -> int:
        """Ultralytics `quantize` value: 16 (FP16) on GPU when models.half, else 32."""
        return 16 if bool(self.cfg.models.half) and self.device != "cpu" else 32

    @property
    def device(self) -> str:
        return "cuda:0" if self.torch.cuda.is_available() else "cpu"

    def _check_session(self, name: str, session) -> None:
        prov = session.get_providers()[0]
        self.providers[name] = prov
        log.info("model %s uses %s", name, prov)
        if bool(self.cfg.models.require_gpu) and prov == "CPUExecutionProvider":
            raise GPUFallbackError(
                f"{name} fell back to CPU. Install onnxruntime-gpu only (no CPU onnxruntime) and the CUDA/cuDNN 12 "
                f"runtime (pip install nvidia-cudnn-cu12), or set models.require_gpu=false.")

    def _ort_providers(self) -> list[str]:
        self.torch  # noqa: B018  (DLL preload)
        import onnxruntime as ort
        avail = ort.get_available_providers()
        if "CUDAExecutionProvider" in avail:
            try:
                ort.preload_dlls()
            except Exception:
                pass
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _get(self, name: str, factory):
        with self._lock:
            m = self._models.get(name)
            if m is None:
                m = factory()
                self._models[name] = m
            return m

    def release(self, name: str) -> None:
        with self._lock:
            if self._models.pop(name, None) is not None:
                gc.collect()
                if self.torch.cuda.is_available():
                    self.torch.cuda.empty_cache()

    # ------------------------------------------------------------------ models
    def person_pose(self):
        def f():
            from ultralytics import YOLO
            p = self.mdir / self.cfg.models.person_pose
            m = YOLO(str(p))
            self.files["person_pose"] = p.name
            self.providers["person_pose"] = self.device
            if bool(self.cfg.models.require_gpu) and self.device == "cpu":
                raise GPUFallbackError("CUDA not available for torch; person_pose would run on CPU")
            return m
        return self._get("person_pose", f)

    def new_person_pose(self):
        """Fresh YOLO instance (tracker state is attached to the model object)."""
        from ultralytics import YOLO
        return YOLO(str(self.mdir / self.cfg.models.person_pose))

    def open_vocab(self, classes: list[str]):
        def f():
            from ultralytics import YOLO
            p = self.mdir / self.cfg.models.open_vocab
            cwd = os.getcwd()
            os.chdir(self.mdir)  # MobileCLIP text encoder is cached in models/
            try:
                m = YOLO(str(p))
                m.set_classes(classes)
            finally:
                os.chdir(cwd)
            self._yoloe_classes = list(classes)
            self.files["open_vocab"] = p.name
            self.providers["open_vocab"] = self.device
            return m
        m = self._get("open_vocab", f)
        if self._yoloe_classes != list(classes):
            m.set_classes(classes)
            self._yoloe_classes = list(classes)
        return m

    def face_det(self):
        def f():
            from uniface import SCRFD
            from uniface.constants import SCRFDWeights
            m = SCRFD(model_name=SCRFDWeights.SCRFD_10G_KPS, confidence_threshold=float(self.cfg.dense.face_conf),
                      input_size=(640, 640), providers=self._ort_providers())
            self._check_session("face_det", m.session)
            self.files["face_det"] = "scrfd_10g.onnx"
            return m
        return self._get("face_det", f)

    def recognizer(self):
        def f():
            from uniface import AdaFace
            from uniface.constants import AdaFaceWeights
            m = AdaFace(model_name=AdaFaceWeights.IR_101, providers=self._ort_providers())
            self._check_session("recognizer", m.session)
            self.files["recognizer"] = "adaface_ir_101.onnx"
            return m
        return self._get("recognizer", f)

    def fiqa(self):
        def f():
            from uniface import EDifFIQA
            from uniface.constants import EDifFIQAWeights
            m = EDifFIQA(model_name=EDifFIQAWeights.M, providers=self._ort_providers())
            self._check_session("fiqa", m.session)
            self.files["fiqa"] = "ediffiqa_m.onnx"
            return m
        return self._get("fiqa", f)

    def head_pose(self):
        def f():
            from uniface import HeadPose
            from uniface.constants import HeadPoseWeights
            m = HeadPose(model_name=HeadPoseWeights.RESNET18, providers=self._ort_providers())
            self._check_session("head_pose", m.session)
            self.files["head_pose"] = "headpose_resnet18.onnx"
            return m
        return self._get("head_pose", f)

    def parser(self):
        def f():
            from uniface import BiSeNet
            from uniface.constants import ParsingWeights
            m = BiSeNet(model_name=ParsingWeights.RESNET18, providers=self._ort_providers())
            self._check_session("parser", m.session)
            self.files["parser"] = "parsing_resnet18.onnx"
            return m
        return self._get("parser", f)

    def attrib(self):
        def f():
            from uniface import FaceAttribNet
            m = FaceAttribNet(providers=self._ort_providers())
            self._check_session("attrib", m.session)
            self.files["attrib"] = "face_attrib_net.onnx"
            return m
        return self._get("attrib", f)

    def ocr(self):
        def f():
            from rapidocr import RapidOCR
            m = RapidOCR()
            self.providers["ocr"] = "onnxruntime (RapidOCR default)"
            self.files["ocr"] = "rapidocr"
            return m
        return self._get("ocr", f)

    def gfpgan(self):
        def f():
            import onnxruntime as ort
            p = self.mdir / self.cfg.models.gfpgan
            if not p.exists():
                raise FileNotFoundError(f"{p} missing: run scripts/download_models.py")
            s = ort.InferenceSession(str(p), providers=self._ort_providers())
            self.providers["gfpgan"] = s.get_providers()[0]
            self.files["gfpgan"] = p.name
            return s
        return self._get("gfpgan", f)

    # ------------------------------------------------------------------ batching
    @staticmethod
    def run_batched(session, input_name: str, output_names: list[str], blobs: list[np.ndarray], batch: int) -> list[np.ndarray]:
        """Run an ONNX session over (1,C,H,W) blobs in batches; returns first output per item."""
        out: list[np.ndarray] = []
        dyn = not isinstance(session.get_inputs()[0].shape[0], int)
        if not dyn:
            batch = 1
        for i in range(0, len(blobs), batch):
            x = np.concatenate(blobs[i:i + batch], axis=0).astype(np.float32)
            y = session.run(output_names[:1], {input_name: x})[0]
            out.extend(list(y))
        return out

    # ------------------------------------------------------------------ reporting
    def model_info(self) -> dict:
        info = {}
        for name, fname in self.files.items():
            p = self.mdir / fname if (self.mdir / fname).exists() else self.mdir / "uniface" / fname
            info[name] = {"file": fname, "provider": self.providers.get(name),
                          "sha256": _file_sha(p)[:16] if p.exists() else None}
        return info

    def vram(self) -> dict:
        t = self.torch
        d = {"torch_max_allocated_mb": round(t.cuda.max_memory_allocated() / 1e6, 1) if t.cuda.is_available() else 0}
        try:
            import subprocess
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, timeout=10)
            d["device_used_mb"] = float(r.stdout.decode().strip().splitlines()[0])
        except Exception:
            pass
        return d

    def warmup(self) -> None:
        dummy = np.zeros((640, 640, 3), np.uint8)
        self.person_pose().predict(dummy, quantize=self.precision, verbose=False, device=self.device)
        self.face_det().detect(dummy)
        face = np.zeros((112, 112, 3), np.uint8)
        self.recognizer().get_embedding(face)
        self.fiqa().score_aligned(face)
        self.head_pose().estimate(face)
        self.parser().parse(face)
        from uniface.types import Face
        self.attrib().predict(face, Face(bbox=np.array([0, 0, 112, 112], np.float32), confidence=1.0,
                                         landmarks=np.zeros((5, 2), np.float32)))


_registry: ModelRegistry | None = None


def get_registry(cfg: Cfg | None = None) -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry(cfg)
    return _registry
