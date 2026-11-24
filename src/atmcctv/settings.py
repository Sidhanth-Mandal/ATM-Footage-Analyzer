"""Configuration loading.

`config/default.yaml` holds every tunable. `ATMCCTV_CONFIG` may point to an
override YAML that is deep-merged on top. Secrets come only from the
environment (`ATMCCTV_SECRET`).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "default.yaml"
CALIBRATION = ROOT / "config" / "calibration.yaml"


class Cfg(dict):
    """dict with attribute access for nested config sections."""

    def __getattr__(self, name: str) -> Any:
        try:
            v = self[name]
        except KeyError as e:
            raise AttributeError(name) from e
        return Cfg(v) if isinstance(v, dict) and not isinstance(v, Cfg) else v

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __deepcopy__(self, memo):
        return Cfg(copy.deepcopy(dict(self), memo))


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _to_cfg(d: Any) -> Any:
    if isinstance(d, dict):
        return Cfg({k: _to_cfg(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_to_cfg(v) for v in d]
    return d


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> Cfg:
    with open(DEFAULT_CONFIG, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # thresholds written by scripts/calibrate.py (search bands, FIQA bounds)
    if CALIBRATION.exists():
        with open(CALIBRATION, encoding="utf-8") as f:
            data = _deep_merge(data, yaml.safe_load(f) or {})
    extra = path or os.environ.get("ATMCCTV_CONFIG")
    if extra and Path(extra).resolve() != DEFAULT_CONFIG.resolve():
        with open(extra, encoding="utf-8") as f:
            data = _deep_merge(data, yaml.safe_load(f) or {})
    if overrides:
        data = _deep_merge(data, overrides)
    return _to_cfg(data)


@lru_cache(maxsize=1)
def get_config() -> Cfg:
    return load_config()


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()


def data_dir(cfg: Cfg | None = None) -> Path:
    cfg = cfg or get_config()
    p = Path(cfg.app.data_dir)
    return p if p.is_absolute() else ROOT / p


def models_dir(cfg: Cfg | None = None) -> Path:
    cfg = cfg or get_config()
    p = Path(cfg.app.models_dir)
    return p if p.is_absolute() else ROOT / p


def secret_key() -> str:
    key = os.environ.get("ATMCCTV_SECRET")
    if not key or len(key) < 16:
        raise RuntimeError(
            "ATMCCTV_SECRET is not set (or shorter than 16 characters). "
            "Set it before starting the API, e.g. $env:ATMCCTV_SECRET = '<random 32+ chars>'"
        )
    return key


def db_url(cfg: Cfg | None = None) -> str:
    override = os.environ.get("ATMCCTV_DB")
    if override:
        return override
    return f"sqlite:///{(data_dir(cfg) / 'atmcctv.db').as_posix()}"
