"""Visual review of the TCIA chest lesion positions. `astra` env. Nothing is decided here: it renders.

    python scripts/19b_tcia_lesion_review.py

1. review_all.png: every lesion, axial 60 px crop at its given (or override) slice with the marked circle, next to the
   radiologist's snapshot JPG (shown as-is: the JPGs are crops of varying size, so they are compared by eye, not by code).
2. review_broken.png: for each lesion whose slice field is broken (not a positive integer; it equals x − y), the top
   candidate slices at the marked (x, y), ranked by a blob score = mean HU in the marked disc − mean HU in the ring
   R..2R (a compact nodule is brighter than its surroundings), restricted to local maxima along z.
The chosen slices go to eval/slice_overrides.csv (case, lesion, instance, decided_by, note), read by 19_tcia_annotations.
"""
from __future__ import annotations
import glob, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib; A = importlib.import_module("19_tcia_annotations")
OUT = A.OUT


def blob_scores(ds, col, row, R):
    yy, xx = np.mgrid[-int(3 * R):int(3 * R) + 1, -int(3 * R):int(3 * R) + 1]; r = np.hypot(yy, xx)
    disc, ring = r <= R, (r > R) & (r <= 2 * R); r0, c0 = int(round(row)), int(round(col)); h = int(3 * R)
    sc = {}
    for i, d in ds.items():
        p = A.hu(d)[r0 - h:r0 + h + 1, c0 - h:c0 + h + 1]
        if p.shape == disc.shape:
            sc[i] = float(p[disc].mean() - p[ring].mean())
    ks = sorted(sc); v = np.array([sc[k] for k in ks])
    peaks = [ks[j] for j in range(len(ks)) if v[j] == v[max(j - 3, 0):j + 4].max()]
    return sorted(peaks, key=lambda k: -sc[k]), sc


def snap(case, name):
    f = next(iter(glob.glob(str(A.CLIN / f"{name}.[jJ][pP][gG]"))), None)
    return np.asarray(Image.open(f).convert("L")) if f else None


def crop(d, col, row, h=30):
    r, c = int(round(row)), int(round(col))
    return np.clip(A.hu(d)[max(r - h, 0):r + h, max(c - h, 0):c + h], -1000, 200)


def main():
    x = pd.read_excel(A.CLIN / "Chest_v11 2026.xlsx", sheet_name="Chest", header=1)
    x = x[x.Manufacturer.astype(str).str.lower().str.startswith("siemens")].reset_index(drop=True)
    ov = OUT / "slice_overrides.csv"; ovd = pd.read_csv(ov) if ov.exists() else pd.DataFrame(columns=["case", "lesion", "instance"])
    rows = []
    for r in x.itertuples():
        R, col, row, sl = (float(v) for v in A.PAT.search(r._8).groups())
        broken = sl < 0 or not float(sl).is_integer()
        o = ovd[(ovd.case == r.Case) & (ovd.lesion == int(r.Lesion_ID))]
        rows.append((r, R, col, row, int(sl) if not broken else (int(o.instance.iloc[0]) if len(o) else None), broken))
    # 1. all lesions at their given / override slice
    n = len(rows); f, ax = plt.subplots(n, 2, figsize=(5.2, 2.5 * n))
    for i, (r, R, col, row, inst, broken) in enumerate(rows):
        ds = A.series(r.Case)
        if inst is not None:
            ax[i, 0].imshow(crop(ds[inst], col, row), cmap="gray", vmin=-1000, vmax=200)
            ax[i, 0].add_patch(plt.Circle((30, 30), R, fill=False, color="r", lw=0.8))
        ax[i, 0].set_title(f"{r.Case} L{int(r.Lesion_ID)} inst {inst}{' (override)' if broken else ''} | {str(r.Diagnosis).split(',')[0]}", fontsize=6.5)
        s = snap(r.Case, r.Snapshot_Filename)
        if s is not None:
            ax[i, 1].imshow(s, cmap="gray")
        ax[i, 1].set_title("radiologist snapshot", fontsize=6.5)
        for a in ax[i]: a.axis("off")
    plt.tight_layout(); plt.savefig(OUT / "review_all.png", dpi=80); plt.close()
    # 2. candidates for broken rows
    br = [t for t in rows if t[5]]
    f, ax = plt.subplots(len(br), 6, figsize=(13, 2.6 * len(br)))
    for i, (r, R, col, row, _, _) in enumerate(br):
        ds = A.series(r.Case); peaks, sc = blob_scores(ds, col, row, R)
        s = snap(r.Case, r.Snapshot_Filename); ax[i, 0].imshow(s, cmap="gray"); ax[i, 0].set_title(f"{r.Case} L{int(r.Lesion_ID)} snapshot", fontsize=7)
        for j, k in enumerate(peaks[:5]):
            ax[i, j + 1].imshow(crop(ds[k], col, row), cmap="gray", vmin=-1000, vmax=200)
            ax[i, j + 1].add_patch(plt.Circle((30, 30), R, fill=False, color="r", lw=0.8))
            ax[i, j + 1].set_title(f"inst {k}  score {sc[k]:.0f}", fontsize=7)
        for a in ax[i]: a.axis("off")
        print(r.Case, int(r.Lesion_ID), str(r.Diagnosis).split(",")[0], "top candidates:", [(k, round(sc[k])) for k in peaks[:5]])
    plt.tight_layout(); plt.savefig(OUT / "review_broken.png", dpi=90); plt.close()


if __name__ == "__main__":
    main()
