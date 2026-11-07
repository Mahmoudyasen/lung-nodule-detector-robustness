"""P2 task 4: convert the fixed NLST sample (analysis/p2_nlst/sample.csv) from raw DICOM at detector spacing.

    python scripts/20_nlst_native_sample.py [--procs 4]

Why: the 1.5 mm NLST archive is coarser in-plane than the detector (0.70 mm), and blur is exactly what E1 shows the
detector is sensitive to, so the archive must not be used for the P2 robustness check. lungcad.data.nlst.convert_native
runs the LUNA16 path (B-spline straight to detector spacing, same window) and reuses each series' validated archive
lung box as the crop. Resumable; log in nlst/native_detector/convert.log.
"""
from __future__ import annotations
import argparse, sys, time
from multiprocessing import Pool
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lungcad import config as C
from lungcad.data.nlst import usable_series, convert_native, NATIVE


def job(a):
    stem, lp = a
    try:
        return stem, convert_native(stem, lp)
    except Exception as e:
        return stem, f"FAIL {type(e).__name__}: {e}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=4); a = ap.parse_args()
    s = pd.read_csv(C.WORK / "wp0" / "analysis" / "p2_nlst" / "sample.csv", dtype={"uid": str})
    lp = usable_series().set_index("uid").local_path
    jobs = [(r.stem, lp[r.uid]) for r in s.itertuples()]
    NATIVE.mkdir(parents=True, exist_ok=True); log = open(NATIVE / "convert.log", "a"); t0 = time.time(); nf = 0
    with Pool(a.procs) as pool:
        for n, (stem, msg) in enumerate(pool.imap_unordered(job, jobs), 1):
            nf += msg.startswith("FAIL")
            line = f"{time.strftime('%F %T')} {n}/{len(jobs)} {Path(stem).name} {msg}"; log.write(line + "\n"); log.flush()
            if n % 100 == 0 or n == len(jobs):
                print(f"{line}  fails {nf}  {(time.time() - t0) / 60:.0f} min", flush=True)
