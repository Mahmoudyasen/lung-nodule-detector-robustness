"""P2 task 2: projection-domain low-dose versions of the LUNA16 test scans of folds 0-3. Runs in the `astra` env.

    python scripts/17_luna_projnoise.py --check                 # no-noise identity check on ALL selected scans
    python scripts/17_luna_projnoise.py [--folds 0 1 2 3] [--doses 0.5 0.25 0.1] [--procs 8] [--cuda]

Each ORIGINAL full-field .mhd gets noise-only insertion (scripts/proj_noise.py; reference I0, Ram-Lak and the
tube-current-modulation fit from calibrate_tcia.py --cases all),
then goes through the unchanged S2 module with the same LUNA16 lung mask, so the stored crop is voxel-aligned with the
clean archive (checked per scan). Seeds are fixed per (scan, dose). Output, same layout as the S2 archive:
<thesis-data>/luna16/projnoise/a<dose>/subset<k>/<uid>.{lz4,json}. Resumable.
"""
from __future__ import annotations
import argparse, json, sys, time, zlib
from multiprocessing import Pool
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from lungcad import config as C
from lungcad.preprocess import Volume, load_mhd, preprocess_volume, save_volume, load_volume, exists as stem_exists
from lungcad.data.luna16_index import load_index, get_split
from lungcad.data.nlst_inventory import kernel_class
import proj_noise as P

_SUF = "_cuda" if "--cuda" in sys.argv else ""    # the calibration must come from the same ASTRA projector (GPU or CPU)
CALIB = C.THESIS_DATA / "tcia_ldct" / "calibration" / f"calib_all{_SUF}.json"   # calibrate_tcia.py --cases all [--cuda]
_cal = json.load(open(CALIB)); I0, SIGMA_E = _cal["I0_ref"], _cal["sigma_e"]
assert _cal.get("projector", "cpu-linear") == ("cuda" if _SUF else "cpu-linear"), f"{CALIB.name} was made with another projector"
TCM = (_cal["filters"]["ram-lak"]["tcm_c"], _cal["filters"]["ram-lak"]["tcm_beta"])   # tube-current modulation, Ram-Lak FBP
# noise texture per kernel class: sharp = Ram-Lak (matches TCIA chest B50f); soft = the filter whose NPS best matches the
# TCIA abdomen B30f pairs (calibrate_tcia.py --cases abdomen). Class from the scan's DICOM kernel (same mapping as NLST).
_soft_cal = json.load(open(CALIB.with_name(f"calib_abdomen{_SUF}.json"))); _soft = _soft_cal["filters"]
assert _soft_cal.get("projector", "cpu-linear") == _cal.get("projector", "cpu-linear"), "abdomen calibration made with another projector"
FILT = {"sharp": "ram-lak", "soft": min(_soft, key=lambda f: _soft[f]["nps_shape_L1"])}
REF_THK = 1.5                       # TCIA calibration slices are 1.5 mm thick
HDR = C.THESIS_DATA / "luna16" / "index" / "lidc_headers.csv"   # scripts/21_lidc_headers.py: true SliceThickness per scan
OUT = C.THESIS_DATA / "luna16" / "projnoise"
CUDA = False


def clean_stem(uid, subset):
    return C.LUNA_PRE / f"subset{subset}" / uid


def process(args):
    uid, subset, mhd, a, thk_hdr, kclass = args
    stem = OUT / f"a{a}" / f"subset{subset}" / uid
    if a is not None and stem_exists(stem):
        return uid, a, "skip"
    t0 = time.time()
    vol = load_mhd(mhd); seg = load_mhd(C.LUNA_SEG / f"{uid}.mhd")
    if a is not None:
        sp_yx = (vol.spacing[1], vol.spacing[2])
        thk = thk_hdr or vol.spacing[0]                      # header thickness; z-spacing only if the header lacks it
        hu = P.insert_noise_volume(vol.hu, sp_yx, a, I0, SIGMA_E, seed=zlib.crc32(f"{uid}|{a}".encode()), cuda=CUDA, tcm=TCM,
                                   scale=float(np.sqrt(REF_THK / thk)), filt=FILT[kclass])
        vol = Volume(hu=hu, spacing=vol.spacing, origin=vol.origin, direction=vol.direction, uid=vol.uid, source=vol.source)
    arr, meta = preprocess_volume(vol, C.LUNA_SPACING, mask_vol=seg, mask_labels=C.LUNA_SEG_LUNG_LABELS)
    ref_arr, ref = load_volume(clean_stem(uid, subset))
    if meta["crop_offset_zyx"] != ref["crop_offset_zyx"] or arr.shape != ref_arr.shape:
        return uid, a, f"FAIL crop differs from the clean archive {meta['crop_offset_zyx']} vs {ref['crop_offset_zyx']}"
    if a is None:
        return uid, a, f"identity max|diff| {int(np.abs(arr.astype(np.int32) - ref_arr).max())}"
    meta.update({"subset": int(subset), "domain": "luna16_projnoise", "dose_fraction": a, "i0": I0, "sigma_e": SIGMA_E, "tcm_c_beta": list(TCM), "projector": _cal.get("projector", "cpu-linear"), "calibration_file": CALIB.name, "slice_thickness_mm": float(thk_hdr or vol.spacing[0]), "thickness_from_header": bool(thk_hdr), "kernel_class": kclass, "noise_filter": FILT[kclass],
                 "noise_sd_added_hu": float((arr.astype(np.float32) - ref_arr).std())})
    save_volume(stem, arr, meta)
    return uid, a, f"ok {time.time() - t0:.0f}s added SD {meta['noise_sd_added_hu']:.1f} HU"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3]); ap.add_argument("--doses", type=float, nargs="+", default=[0.5, 0.25, 0.1])
    ap.add_argument("--procs", type=int, default=8); ap.add_argument("--cuda", action="store_true"); ap.add_argument("--check", action="store_true")
    a = ap.parse_args(); CUDA = a.cuda
    import pandas as pd
    h = pd.read_csv(HDR, dtype={"seriesuid": str}).drop_duplicates("seriesuid")
    THK = {u: float(t) for u, t in zip(h.seriesuid, pd.to_numeric(h.SliceThickness, errors="coerce")) if t == t and t > 0}
    KCL = {u: kernel_class(str(k)) for u, k in zip(h.seriesuid, h.ConvolutionKernel)}
    idx = load_index().set_index("seriesuid")
    uids = [u for k in a.folds for u in get_split(load_index(), k)["test"]]
    if a.check:
        jobs = [(u, int(idx.loc[u, "subset"]), idx.loc[u, "mhd_path"], None, None, None) for u in uids]
    else:
        missing = [u for u in uids if u not in THK]
        assert not missing, f"{len(missing)} scans lack a header thickness: run scripts/21_lidc_headers.py first"
        unk = sorted({str(h.set_index("seriesuid").loc[u, "ConvolutionKernel"]) for u in uids if KCL.get(u) not in FILT})
        assert not unk, f"kernels without a soft/sharp class (add them to nlst_inventory, do not guess): {unk}"
        print("noise filter per kernel class:", FILT, flush=True)
        jobs = [(u, int(idx.loc[u, "subset"]), idx.loc[u, "mhd_path"], d, THK[u], KCL[u]) for d in a.doses for u in uids]
    print(f"{len(jobs)} jobs", flush=True); t0 = time.time()
    with Pool(a.procs) as pool:                       # with --cuda each worker uses its own ASTRA GPU context
        for n, (uid, d, msg) in enumerate(pool.imap_unordered(process, jobs), 1):
            print(f"{time.strftime('%F %T')} {n}/{len(jobs)} a={d} {uid[-12:]} {msg}  ({(time.time() - t0) / 60:.0f} min)", flush=True)
