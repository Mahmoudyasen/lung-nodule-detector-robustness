"""Noise-texture test (lung-nodule-detector-analysis-plans/plans_original/texture_test_prespec.md): measure the real noise power spectrum of the TCIA
pairs in PHYSICAL frequency (cycles/mm), so it can be imposed on LUNA16 scans of any pixel spacing. ml env.

    python scripts/25_tcia_nps.py [--every 12] [--procs 8]

D = low - full is the providers' inserted noise alone (same reasoning as calibrate_tcia.py). Per sampled slice:
the 2D NPS of D in 48 x 48 soft-tissue ROIs (same masks and ROI rule as calibrate_tcia.py), binned radially in
cycles/mm with the case's pixel spacing; and the Pearson correlation of D between the slice and the next one in the
same soft-tissue mask (descriptive check of the z-independence assumption). Chest C* = B50f (sharp kernels),
abdomen L* = B30f (soft kernels). Output: analysis/p2_texture/tcia_nps.json.
"""
from __future__ import annotations
import os
import argparse, glob, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pydicom
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lungcad import config as C

RAW = Path(os.environ.get("LUNGCAD_NLST_HDD", "data/raw"), "tcia_ldct")
OUT = C.WORK / "wp0" / "analysis" / "p2_texture"
ROI = 48
FBIN = 0.02                                  # cycles/mm bin width
NBIN = 40                                    # up to 0.8 cycles/mm (TCIA Nyquist is ~0.7 for ~0.7 mm pixels)


def series(case, dose):
    fs = glob.glob(str(RAW / case / dose / "*.dcm"))
    hd = sorted(((int(pydicom.dcmread(f, stop_before_pixels=True).InstanceNumber), f) for f in fs))
    return [f for _, f in hd]


def hu(f):
    d = pydicom.dcmread(f); return (d.pixel_array * float(d.RescaleSlope) + float(d.RescaleIntercept)).astype(np.float32), [float(v) for v in d.PixelSpacing], str(d.get("ConvolutionKernel", ""))


def soft_mask(full):                         # = calibrate_tcia.masks()[1]
    body = ndimage.binary_fill_holes(ndimage.binary_opening(full > -500, iterations=3))
    return ndimage.binary_erosion(body & (full > -200) & (full < 200), iterations=6)


def radial_nps(img, soft, sp):
    """Sum and count of NPS values per physical radial frequency bin over all valid ROIs (NPS in HU^2 mm^2)."""
    h = ROI // 2; ok = ndimage.binary_erosion(soft, np.ones((ROI, ROI)))
    ys, xs = np.nonzero(ok[::h, ::h]); s_sum = np.zeros(NBIN); n_sum = np.zeros(NBIN); nroi = 0
    fy = np.fft.fftshift(np.fft.fftfreq(ROI, sp[0])); fx = np.fft.fftshift(np.fft.fftfreq(ROI, sp[1]))
    fr = np.hypot(*np.meshgrid(fy, fx, indexing="ij")); b = (fr / FBIN).astype(int)
    for y, x in zip(ys * h, xs * h):
        p = img[y - h:y + h, x - h:x + h]; p = p - p.mean()
        s = np.abs(np.fft.fftshift(np.fft.fft2(p))) ** 2 * sp[0] * sp[1] / ROI ** 2
        m = b < NBIN; s_sum += np.bincount(b[m], s[m], NBIN); n_sum += np.bincount(b[m], minlength=NBIN); nroi += 1
    return s_sum, n_sum, nroi


def work(args):
    case, k, ff, fl = args
    f0, sp, kern = hu(ff[k]); l0, _, _ = hu(fl[k]); f1, _, _ = hu(ff[k + 1]); l1, _, _ = hu(fl[k + 1])
    d0, d1 = l0 - f0, l1 - f1; soft = soft_mask(f0)
    s, n, nroi = radial_nps(d0, soft, sp)
    m = soft & soft_mask(f1)
    rz = float(np.corrcoef(d0[m], d1[m])[0, 1]) if m.sum() > 1000 else None
    return case, kern, sp, s, n, nroi, rz, float(d0[soft].std()) if soft.sum() else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--every", type=int, default=12); ap.add_argument("--procs", type=int, default=8); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); res = {}
    for region, pat in (("chest_B50f", "C*"), ("abdomen_B30f", "L*")):
        cases = sorted(p.parent.parent.name for p in RAW.glob(f"{pat}/low/.done") if (p.parent.parent / "full" / ".done").exists())
        jobs = []
        for c in cases:
            ff, fl = series(c, "full"), series(c, "low"); assert len(ff) == len(fl), c
            jobs += [(c, k, ff, fl) for k in range(len(ff) // 6, len(ff) - len(ff) // 6 - 1, a.every)]
        with Pool(a.procs) as pool:
            rows = pool.map(work, jobs, chunksize=1)
        S = sum(r[3] for r in rows); N = sum(r[4] for r in rows); nps = np.where(N > 0, S / np.maximum(N, 1), np.nan)
        kerns = sorted({r[1] for r in rows}); sps = np.array([r[2][0] for r in rows]); rz = [r[6] for r in rows if r[6] is not None]
        fmax = float(0.5 / sps.max())        # highest frequency present in every case (coarsest pixel's Nyquist)
        res[region] = {"cases": cases, "kernels": kerns, "n_slices": len(rows), "n_roi": int(sum(r[5] for r in rows)),
                       "pixel_mm_min": float(sps.min()), "pixel_mm_max": float(sps.max()), "f_bin_cycles_per_mm": FBIN,
                       "f_center": ((np.arange(NBIN) + 0.5) * FBIN).tolist(), "nps": [None if v != v else float(v) for v in nps],
                       "f_max_common": fmax, "z_corr_adjacent_median": float(np.median(rz)), "z_corr_adjacent_iqr": [float(np.percentile(rz, 25)), float(np.percentile(rz, 75))],
                       "sd_soft_median_hu": float(np.median([r[7] for r in rows if r[7] is not None]))}
        print(region, kerns, f"{len(rows)} slices, {res[region]['n_roi']} ROIs, pixel {sps.min():.3f}-{sps.max():.3f} mm, f_max_common {fmax:.3f}, "
              f"z-corr median {res[region]['z_corr_adjacent_median']:.3f}, peak at {res[region]['f_center'][int(np.nanargmax(nps))]:.2f} cyc/mm", flush=True)
    json.dump(res, open(OUT / "tcia_nps.json", "w"), indent=1)
