"""Optional file-based ATM setup: config/atms/<atm_code>.yaml is upserted into the atm table."""
from __future__ import annotations

import yaml
from sqlmodel import select

from .. import audit
from ..db.models import Atm
from ..db.session import init_db, write_session
from ..settings import ROOT
from .schemas import AtmIn


def sync_atm_files() -> dict:
    init_db()
    d = ROOT / "config" / "atms"
    done, errors = [], []
    if not d.exists():
        return {"synced": done, "errors": errors}
    for p in sorted(d.glob("*.yaml")):
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            data.setdefault("atm_code", p.stem)
            b = AtmIn(**data)
        except Exception as e:  # report and continue
            errors.append(f"{p.name}: {e}")
            continue
        with write_session() as s:
            a = s.exec(select(Atm).where(Atm.atm_code == b.atm_code)).first() or Atm(atm_code=b.atm_code)
            a.location, a.camera_type = b.location, b.camera_type
            a.rois_json = {k: v for k, v in b.rois.items() if v}
            a.ocr_roi_json, a.date_order, a.tz, a.metadata_is_end = b.ocr_roi, b.date_order, b.tz, b.metadata_is_end
            s.add(a)
            s.flush()
            audit.append(s, "atm_synced_from_file", role="system", object_type="atm", object_id=a.id,
                         details={"file": p.name})
        done.append(b.atm_code)
    return {"synced": done, "errors": errors}
