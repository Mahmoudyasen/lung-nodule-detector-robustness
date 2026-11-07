"""TCIA LDCT chest (Siemens) full- and 10 %-dose images -> the shared S2 module, LUNA16 detector spacing.

    python scripts/16_tcia_ldct_preprocess.py [--limit N]

Per case: both series are loaded, their geometry is checked to be identical, the lung mask is computed ONCE on the
full-dose scan (lungmask R231 on the CPU, classical fallback) and BOTH doses are cropped with it, so the two stored
volumes are voxel-for-voxel aligned and the full-vs-low comparison is exactly paired. Output:
<data>/tcia_ldct/preprocessed/<case>_{full,low}.{lz4,json}. Resumable (existing pairs are skipped).
"""
from __future__ import annotations
import os
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lungcad import config as C
from lungcad.preprocess import (load_dicom_series, resample, lung_mask_lungmask, lung_mask_classical, mask_is_plausible,
                                finalize_volume, save_volume, exists as stem_exists)

RAW = Path(os.environ.get("LUNGCAD_NLST_HDD", "data/raw"), "tcia_ldct")
PRE = C.WORK.parent / "tcia_ldct" / "preprocessed"


def same_geometry(a, b) -> bool:
    return (a.shape == b.shape and np.allclose(a.spacing, b.spacing, atol=1e-4) and np.allclose(a.origin, b.origin, atol=1e-3)
            and np.allclose(a.direction, b.direction, atol=1e-6))


def process_case(case: str, inferer) -> str:
    stems = {d: PRE / f"{case}_{d}" for d in ("full", "low")}
    if all(stem_exists(s) for s in stems.values()):
        return "skip"
    vols = {d: load_dicom_series(RAW / case / d) for d in ("full", "low")}
    if not same_geometry(vols["full"], vols["low"]):
        raise RuntimeError(f"{case}: full/low geometry differs")
    t0 = time.time()
    res = {d: resample(v, C.LUNA_SPACING) for d, v in vols.items()}
    mask, src = None, "lungmask"
    try:
        mask = lung_mask_lungmask(res["full"], inferer)
        if not mask_is_plausible(mask, C.LUNA_SPACING):
            mask, src = None, "lungmask_implausible"
    except Exception as e:  # noqa
        mask, src = None, f"lungmask_failed({type(e).__name__})"
    if mask is None:
        m = lung_mask_classical(res["full"].hu, C.LUNA_SPACING)
        if mask_is_plausible(m, C.LUNA_SPACING):
            mask, src = m, src + "+classical"
    for d in ("full", "low"):
        arr, meta = finalize_volume(vols[d], res[d], mask, src + "(full-dose mask)", C.LUNA_SPACING, t0=t0)
        meta.update({"domain": "tcia_ldct", "case": case, "dose": d})
        save_volume(stems[d], arr, meta)
    return f"ok {arr.shape} {src} {time.time() - t0:.0f}s"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    from lungmask import LMInferer
    inferer = LMInferer(modelname="R231", force_cpu=False, batch_size=16, tqdm_disable=True)   # GPU
    PRE.mkdir(parents=True, exist_ok=True)
    cases = sorted(p.parent.parent.name for p in RAW.glob("C*/low/.done") if (p.parent.parent / "full" / ".done").exists())
    if a.limit:
        cases = cases[:a.limit]
    log = open(PRE / "preprocess.log", "a")
    for i, case in enumerate(cases):
        try:
            msg = process_case(case, inferer)
        except Exception as e:
            msg = f"FAIL {e}"
        line = f"{time.strftime('%F %T')} {i+1}/{len(cases)} {case} {msg}"; print(line, flush=True); log.write(line + "\n"); log.flush()
