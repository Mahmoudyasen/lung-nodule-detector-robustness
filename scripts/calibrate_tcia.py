"""Calibrate and validate proj_noise.py on the TCIA full / 10 % pairs. Runs in the `astra` env.

    python calibrate_tcia.py --cases C002 C004 C012 [--every 12] [--procs 6]

TCIA's 10 % images were made by the data providers' validated projection-noise insertion on the same raw data, so
D = low - full is the inserted noise alone: the reference for our simulation. Per slice and per FBP filter we
compare our inserted noise N_sim (a = 0.10, trial I0 = I0_REF) with D: noise SD in the body (lungs excluded) and the
2D noise power spectrum (NPS) in soft-tissue ROIs. Real scans use tube-current modulation, so the real SD varies
much less between slices than at a constant I0; per filter we fit the power law sd_real = c * sd_sim^beta (log-log
least squares over slices, proj_noise.tcm_factor) and report its error on held-out cases (fit on half the cases,
test on the other half). The NPS shape picks the filter. Output: calib_<cases>.{csv,json}.
"""
from __future__ import annotations
import os
import argparse, glob, json
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd, pydicom
from scipy import ndimage
import proj_noise as P

RAW = Path(os.environ.get("LUNGCAD_NLST_HDD", "data/raw"), "tcia_ldct")
OUT = Path(os.environ.get("LUNGCAD_DATA", "data"), "tcia_ldct", "calibration")
I0_REF, SIGMA_E, A = 1.0e5, 10.0, 0.10
FILTERS = ("ram-lak", "shepp-logan", "cosine", "hamming", "hann")
ROI = 48


@lru_cache(maxsize=2)                  # one case (both doses) per worker at a time
def load_case(case: str, dose: str):
    ds = sorted((pydicom.dcmread(f) for f in glob.glob(str(RAW / case / dose / "*.dcm"))), key=lambda d: int(d.InstanceNumber))
    hu = np.stack([d.pixel_array * float(d.RescaleSlope) + float(d.RescaleIntercept) for d in ds]).astype(np.float32)
    return hu, [float(v) for v in ds[0].PixelSpacing], str(ds[0].get("ConvolutionKernel", ""))


def masks(full: np.ndarray):
    body = ndimage.binary_fill_holes(ndimage.binary_opening(full > -500, iterations=3))
    soft = ndimage.binary_erosion(body & (full > -200) & (full < 200), iterations=6)
    return body & (full > -400), soft


def nps(img: np.ndarray, soft: np.ndarray):
    """Mean 2D NPS over ROI x ROI soft-tissue patches fully inside `soft`; radial profile (16 bins)."""
    h = ROI // 2; ok = ndimage.binary_erosion(soft, np.ones((ROI, ROI)))
    ys, xs = np.nonzero(ok[::ROI // 2, ::ROI // 2]); spec = []
    for y, x in zip(ys * (ROI // 2), xs * (ROI // 2)):
        p = img[y - h:y + h, x - h:x + h]; p = p - p.mean()
        spec.append(np.abs(np.fft.fftshift(np.fft.fft2(p))) ** 2 / ROI ** 2)
    if not spec:
        return None
    s = np.mean(spec, 0); yy, xx = np.mgrid[:ROI, :ROI] - h; r = np.hypot(yy, xx)
    return np.array([s[(r >= i) & (r < i + 1.5)].mean() for i in np.linspace(1, h - 1.5, 16)])


def work(args):
    case, k, flt, cuda = args
    full, sp, kern = load_case(case, "full"); low, _, _ = load_case(case, "low")
    body = P.body_mask(full[k]); soft = masks(full[k])[1]; d = low[k] - full[k]
    geo = P.SliceGeom(full.shape[1], full.shape[2], sp[0], sp[1], cuda=cuda)   # projector must match the one used to generate the LUNA16 scans
    P_FILTER[0] = flt
    n = P.insert_noise_slice(full[k], geo, A, I0_REF, SIGMA_E, np.random.default_rng(k)); geo.close()
    return {"case": case, "slice": k, "filter": flt, "kernel": kern, "sd_real": float(d[body].std()), "sd_sim": float(n[body].std()),
            "nps_real": nps(d, soft), "nps_sim": nps(n, soft)}


P_FILTER = ["ram-lak"]
_fbp = P.SliceGeom.fbp
def _fbp_with_filter(self, sino):                 # same as SliceGeom.fbp, filter chosen per run
    import astra
    sid = astra.data2d.create("-sino", self.pg, sino.astype(np.float32)); rid = astra.data2d.create("-vol", self.vg, 0)
    cfg = astra.astra_dict("FBP_CUDA" if self.cuda else "FBP")
    name, _, dd = P_FILTER[0].partition(":")
    cfg.update(ReconstructionDataId=rid, ProjectionDataId=sid, ProjectorId=self.pid, FilterType=name)
    if dd:
        cfg["option"] = {"FilterD": float(dd)}
    alg = astra.algorithm.create(cfg); astra.algorithm.run(alg); out = astra.data2d.get(rid)
    astra.algorithm.delete(alg); astra.data2d.delete([sid, rid]); return out
P.SliceGeom.fbp = _fbp_with_filter


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--cases", nargs="+", required=True)
    ap.add_argument("--every", type=int, default=12); ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--filters", nargs="+", default=list(FILTERS))
    ap.add_argument("--cuda", action="store_true", help="ASTRA GPU projector/FBP (use the same setting in 17_luna_projnoise.py)"); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.cases == ["abdomen"]:                    # B30f pairs: used for the NPS shape of soft kernels only
        a.cases = sorted(p.parent.parent.name for p in RAW.glob("L*/low/.done") if (p.parent.parent / "full" / ".done").exists())
    if a.cases == ["all"]:
        a.cases = sorted(p.parent.parent.name for p in RAW.glob("C*/low/.done") if (p.parent.parent / "full" / ".done").exists())
    jobs = []
    for c in a.cases:
        nz = len(glob.glob(str(RAW / c / "full" / "*.dcm")))
        jobs += [(c, k, f, a.cuda) for k in range(nz // 6, nz - nz // 6, a.every) for f in a.filters]   # skip the extreme ends
    with Pool(a.procs) as pool:
        rows = pool.map(work, jobs, chunksize=1)
    df = pd.DataFrame(rows); summ = {}
    cs = sorted(df.case.unique()); half = (set(cs[0::2]), set(cs[1::2]))
    def fit(g):
        b, lc = np.polyfit(np.log(g.sd_sim), np.log(g.sd_real), 1); return float(np.exp(lc)), float(b)
    def err(g, cb):
        pred = cb[0] * g.sd_sim ** cb[1]; return float(np.median(np.abs(pred / g.sd_real - 1)))
    for f, g in df.groupby("filter"):
        c, beta = fit(g)
        held = [err(g[g.case.isin(te)], fit(g[g.case.isin(tr)])) for tr, te in (half, half[::-1])] if len(cs) > 1 else [np.nan]
        nr = np.mean([x for x in g.nps_real if x is not None], 0); ns = np.mean([x for x in g.nps_sim if x is not None], 0)
        shape_err = float(np.abs(ns / ns.sum() - nr / nr.sum()).sum())       # L1 distance of normalised radial NPS
        summ[f] = {"tcm_c": c, "tcm_beta": beta, "median_rel_err_in_sample": err(g, (c, beta)), "median_rel_err_held_out": float(np.nanmean(held)),
                   "median_rel_err_const_i0": float(np.median(np.abs(g.sd_sim * (g.sd_real / g.sd_sim).median() / g.sd_real - 1))),
                   "nps_shape_L1": shape_err, "n_slices": len(g), "nps_real": nr.tolist(), "nps_sim": ns.tolist()}
        print(f"{f:12s} TCM c {c:.3g} beta {beta:.2f}  median |rel err| in-sample {summ[f]['median_rel_err_in_sample']:.3f} "
              f"held-out {summ[f]['median_rel_err_held_out']:.3f} (constant I0 {summ[f]['median_rel_err_const_i0']:.3f})  NPS L1 {shape_err:.3f}  ({len(g)} slices)", flush=True)
    tag = (("abdomen" if a.cases[0].startswith("L") else "all") if len(a.cases) >= 10 else "_".join(a.cases)) + ("_cuda" if a.cuda else "")
    df.drop(columns=["nps_real", "nps_sim"]).to_csv(OUT / f"calib_{tag}.csv", index=False)
    json.dump({"cases": a.cases, "kernels": sorted(df.kernel.unique().tolist()), "I0_ref": I0_REF, "tcm_model": "sd_real = c * sd_sim^beta", "sigma_e": SIGMA_E, "a": A, "projector": "cuda" if a.cuda else "cpu-linear", "filters": summ},
              open(OUT / f"calib_{tag}.json", "w"), indent=1)
