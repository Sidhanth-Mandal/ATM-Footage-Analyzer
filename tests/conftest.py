import os
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="atmcctv_test_"))
_cfg = _TMP / "test.yaml"
_cfg.write_text(f"app:\n  data_dir: '{(_TMP / 'data').as_posix()}'\n", encoding="utf-8")
os.environ["ATMCCTV_CONFIG"] = str(_cfg)
os.environ["ATMCCTV_DB"] = f"sqlite:///{(_TMP / 'data' / 'test.db').as_posix()}"
os.environ["ATMCCTV_SECRET"] = "test-secret-0123456789-abcdefghijkl"
os.environ.pop("ATMCCTV_ADMIN_PASSWORD", None)


@pytest.fixture(scope="session")
def tmp_root() -> Path:
    return _TMP


@pytest.fixture()
def cfg():
    from atmcctv.settings import load_config
    return load_config()
