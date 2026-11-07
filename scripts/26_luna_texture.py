"""P2 task 5 (noise-texture test, docs/texture_test_prespec.md): LUNA16 test scans of folds 0-3 with stationary
image-domain noise of three textures, at the per-scan level of the projection-domain noise at 10 % dose. ml env.

    python scripts/26_luna_texture.py --selftest            # known-answer checks (NPS shape, identity), no output volumes
    python scripts/26_luna_texture.py [--conds tex_nps tex_gauss tex_white] [--procs 8]

Same path as scripts/17_luna_projnoise.py: noise is added to the ORIGINAL full-field image (HU, native grid), then the
unchanged S2 preprocessing with the same LUNA16 lung mask; the crop is checked against the clean archive.
  tex_nps   : per slice, in-plane NPS = measured real TCIA noise NPS (scripts/25_tcia_nps.py): chest B50f for
              sharp-kernel scans, abdomen B30f for soft-kernel scans; zero above the measured band; slices independent.
  tex_gauss : 3D Gaussian low-pass of white noise, sigma 1.41 mm in-plane, 2.5 mm in z (the draft's 1.4 mm grain).
  tex_white : white noise per native voxel.
Level: added SD over the preprocessed crop (noisy - clean archive) matched to the scan's projnoise a0.1
noise_sd_added_hu, by two passes (pass 2 rescales by target / pass-1 SD); |error| <= 3 % expected, recorded per scan.
Output: <thesis-data>/luna16/texture/<cond>/subset<k>/<uid>.{lz4,json}. Resumable.
P2 review R2 (docs/review_prespec.md): --factor F scales the per-scan target (same noise fields and seeds) and writes
<root>/<cond>_x<F>/...; --root sets the output root (default: the factor-1 location above).
"""
from __future__ import annotations
import argparse, json, sys, time, zlib
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy import ndimage, fft as sfft

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from lungcad import config as C
from lungcad.preprocess import Volume, load_mhd, preprocess_volume, save_volume, load_volume, exists as stem_exists
from lungcad.data.luna16_index import load_index, get_split
from lungcad.data.nlst_inventory import kernel_class

NPSF = C.WORK / "wp0" / "analysis" / "p2_texture" / "tcia_nps.json"
PN = C.THESIS_DATA / "luna16" / "projnoise" / "a0.1"
OUT = C.THESIS_DATA / "luna16" / "texture"
HDR = C.THESIS_DATA / "luna16" / "index" / "lidc_headers.csv"
CONDS = ("tex_nps", "tex_gauss", "tex_white")
GAUSS_MM = (2.5, 1.41, 1.41)                     # z, y, x: 2 voxels of the detector grid 1.25 x 0.703 x 0.703 mm
_N = json.load(open(NPSF)) if NPSF.exists() else None
REGION = {"sharp": "chest_B50f", "soft": "abdomen_B30f"}


def nps_curve(kclass):
    r = _N[REGION[kclass]]; f = np.array(r["f_center"]); v = np.array([0.0 if x is None else x for x in r["nps"]])
    return f, v


def nps_filter(shape_yx, sp_yx, kclass):
    """sqrt(NPS) on the rfft2 grid of a slice with pixel spacing sp_yx (radial, cycles/mm); zero beyond the measured band."""
    f, v = nps_curve(kclass)
    fy = sfft.fftfreq(shape_yx[0], sp_yx[0]); fx = sfft.rfftfreq(shape_yx[1], sp_yx[1])
    fr = np.hypot(fy[:, None], fx[None, :])
    return np.sqrt(np.interp(fr, f, v, left=v[0], right=0.0)).astype(np.float32)


def unit_noise(shape, sp_zyx, cond, kclass, seed):
    """Zero-mean noise field of the given texture, normalised to unit SD over the whole field (float32)."""
    rng = np.random.default_rng(seed); w = rng.standard_normal(shape, dtype=np.float32)
    if cond == "tex_nps":
        H = nps_filter(shape[1:], sp_zyx[1:], kclass)
        n = sfft.irfft2(sfft.rfft2(w, axes=(1, 2), workers=4) * H[None], s=shape[1:], axes=(1, 2), workers=4).astype(np.float32)
    elif cond == "tex_gauss":
        n = ndimage.gaussian_filter(w, [g / s for g, s in zip(GAUSS_MM, sp_zyx)])
    elif cond == "tex_white":
        n = w
    else:
        raise ValueError(cond)
    n -= n.mean(); n /= n.std(); return n


def measure_radial(n2d, sp_yx, roi=48):
    """Radial NPS of a 2D noise image, same ROI rule / binning as 25_tcia_nps.radial_nps (whole image as 'soft')."""
    import importlib; T = importlib.import_module("25_tcia_nps")
    s, c, _ = T.radial_nps(n2d, np.ones_like(n2d, bool), sp_yx); return np.where(c > 0, s / np.maximum(c, 1), np.nan)


def cond_dir(cond, factor):
    return cond if factor == 1.0 else f"{cond}_x{factor:g}"


def clean_stem(uid, subset):
    return C.LUNA_PRE / f"subset{subset}" / uid


def process(args):
    uid, subset, mhd, cond, kclass, target, factor = args
    stem = OUT / cond_dir(cond, factor) / f"subset{subset}" / uid
    if stem_exists(stem):
        return uid, cond, "skip"
    t0 = time.time()
    vol = load_mhd(mhd); seg = load_mhd(C.LUNA_SEG / f"{uid}.mhd"); ref_arr, ref = load_volume(clean_stem(uid, subset))
    n = unit_noise(vol.hu.shape, vol.spacing, cond, kclass, zlib.crc32(f"{uid}|{cond}".encode()))

    def run(scale):
        v = Volume(hu=vol.hu + scale * n, spacing=vol.spacing, origin=vol.origin, direction=vol.direction, uid=vol.uid, source=vol.source)
        arr, meta = preprocess_volume(v, C.LUNA_SPACING, mask_vol=seg, mask_labels=C.LUNA_SEG_LUNG_LABELS)
        assert meta["crop_offset_zyx"] == ref["crop_offset_zyx"] and arr.shape == ref_arr.shape, "crop differs from the clean archive"
        return arr, meta, float((arr.astype(np.float32) - ref_arr).std())
    s1 = target; _, _, m1 = run(s1)
    s2 = s1 * target / m1; arr, meta, m2 = run(s2)
    meta.update({"subset": int(subset), "domain": "luna16_texture", "condition": cond, "kernel_class": kclass,
                 "nps_region": REGION[kclass] if cond == "tex_nps" else None, "gauss_sigma_mm_zyx": list(GAUSS_MM) if cond == "tex_gauss" else None,
                 "target_sd_hu": target, "scale_pass1_hu": s1, "sd_pass1_hu": m1, "scale_hu": s2, "noise_sd_added_hu": m2,
                 "rel_err": m2 / target - 1, "seed": zlib.crc32(f"{uid}|{cond}".encode()), "level_factor": factor})
    save_volume(stem, arr, meta)
    return uid, cond, f"ok {time.time() - t0:.0f}s target {target:.1f} achieved {m2:.1f} HU ({100 * (m2 / target - 1):+.1f} %)"


def selftest(uids, idx, KCL):
    """Known-answer checks 1-2 of the pre-spec: NPS shape of the generated noise; zero-amplitude identity."""
    ok = True
    for kc in ("sharp", "soft"):
        for sp in (0.55, 0.703, 0.9):
            n = unit_noise((4, 512, 512), (1.25, sp, sp), "tex_nps", kc, 1)
            m = np.nanmean([measure_radial(n[i], (sp, sp)) for i in range(4)], 0)
            f, v = nps_curve(kc); fmax = 0.5 / sp                                  # compare where both exist
            sel = (~np.isnan(m)) & (f < min(fmax, f.max())) & (f > 0.03)
            vv = v[sel] / v[sel].sum(); mm = m[sel] / m[sel].sum(); L1 = float(np.abs(vv - mm).sum())
            print(f"selftest tex_nps {kc:5s} pixel {sp} mm: radial NPS shape L1 vs target {L1:.3f}  ({'OK' if L1 < 0.05 else 'FAIL'})"); ok &= L1 < 0.05
    n = unit_noise((4, 256, 256), (1.25, 0.7, 0.7), "tex_white", "sharp", 1); m = np.nanmean([measure_radial(n[i], (0.7, 0.7)) for i in range(4)], 0)
    sel = ~np.isnan(m) & (np.arange(len(m)) > 1) & (np.arange(len(m)) < 30); cv = float(np.nanstd(m[sel]) / np.nanmean(m[sel]))
    print(f"selftest tex_white: radial NPS flat (CV {cv:.3f}, expect < 0.15)"); ok &= cv < 0.15
    for uid in uids[:3]:                                                          # zero amplitude -> byte-identical to the archive
        sub = int(idx.loc[uid, "subset"]); vol = load_mhd(idx.loc[uid, "mhd_path"]); seg = load_mhd(C.LUNA_SEG / f"{uid}.mhd")
        n = unit_noise(vol.hu.shape, vol.spacing, "tex_nps", KCL[uid], 0)
        v = Volume(hu=vol.hu + 0.0 * n, spacing=vol.spacing, origin=vol.origin, direction=vol.direction, uid=vol.uid, source=vol.source)
        arr, _ = preprocess_volume(v, C.LUNA_SPACING, mask_vol=seg, mask_labels=C.LUNA_SEG_LUNG_LABELS); ref_arr, _ = load_volume(clean_stem(uid, sub))
        same = arr.shape == ref_arr.shape and np.array_equal(arr, ref_arr); print(f"selftest identity {uid[-12:]}: {'byte-identical' if same else 'DIFFERS'}"); ok &= same
    print("SELFTEST", "PASSED" if ok else "FAILED"); return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3]); ap.add_argument("--conds", nargs="+", default=list(CONDS))
    ap.add_argument("--procs", type=int, default=8); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--limit", type=int)
    ap.add_argument("--factor", type=float, default=1.0); ap.add_argument("--root")
    a = ap.parse_args()
    if a.root:
        OUT = Path(a.root)                                   # module global, inherited by the forked workers
    import pandas as pd
    h = pd.read_csv(HDR, dtype={"seriesuid": str}).drop_duplicates("seriesuid")
    KCL = {u: kernel_class(str(k)) for u, k in zip(h.seriesuid, h.ConvolutionKernel)}
    idx = load_index().set_index("seriesuid")
    uids = [u for k in a.folds for u in get_split(load_index(), k)["test"]]
    if a.selftest:
        sys.exit(0 if selftest(uids, idx, KCL) else 1)
    tgt = {u: json.load(open(PN / f"subset{int(idx.loc[u, 'subset'])}" / f"{u}.json"))["noise_sd_added_hu"] for u in uids}
    assert all(KCL.get(u) in REGION for u in uids)
    jobs = [(u, int(idx.loc[u, "subset"]), idx.loc[u, "mhd_path"], c, KCL[u], tgt[u] * a.factor, a.factor) for c in a.conds for u in uids][: a.limit]
    print(f"{len(jobs)} jobs", flush=True); t0 = time.time()
    with Pool(a.procs) as pool:
        for n, (uid, c, msg) in enumerate(pool.imap_unordered(process, jobs), 1):
            print(f"{time.strftime('%F %T')} {n}/{len(jobs)} {c} {uid[-12:]} {msg}  ({(time.time() - t0) / 60:.0f} min)", flush=True)
