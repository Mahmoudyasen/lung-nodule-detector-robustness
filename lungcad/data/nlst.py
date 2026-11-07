"""NLST series table and loader for the detector.

usable_series(): one row per converted primary series, MINUS nlst/meta/nlst_exclusions.csv (316 series: 313 failed lung
masks stored uncropped + 3 bad source DICOMs). Every NLST consumer must go through it, never glob the archive.
load_for_detector(): the 1.5 mm archive volume resampled to the detector spacing (C.LUNA_SPACING) with the same
geometry, so stage1 inference and voxel_to_world work exactly as on LUNA16. NOTE: the 1.5 mm archive is coarser in-plane
than the detector (0.70 mm), so this path blurs every scan (down- then up-sampling); not used for
the robustness measurements.
convert_native(): the LUNA16 path for NLST: raw DICOM -> B-spline straight to detector spacing -> crop -> window, with
the crop taken from the validated archive lung box (no new lung mask), stored in NATIVE. Used for the NLST analyses.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, SimpleITK as sitk, torch
from .. import config as C
from ..preprocess import (load_meta, load_volume, normalize, load_dicom_series, resample, finalize_volume, save_volume,
                          exists as stem_exists)
from .nlst_convert import stem_for

META = C.NLST_PRE.parent / "meta"
NATIVE = C.NLST_PRE.parent / "native_detector"
COLS = ["PatientID", "round", "SeriesInstanceUID", "Manufacturer", "ManufacturerModelName", "kernel", "kernel_class", "thk_mm",
        "local_path"]


def exclusions() -> pd.DataFrame:
    return pd.read_csv(META / "nlst_exclusions.csv", dtype={"series_uid": str})


def usable_series(check_files: bool = False) -> pd.DataFrame:
    d = pd.read_csv(META / "nlst_primary_series.csv", usecols=COLS, low_memory=False, dtype={"SeriesInstanceUID": str})
    d = d.rename(columns={"PatientID": "pid", "SeriesInstanceUID": "uid"})
    d = d[~d.uid.isin(set(exclusions().series_uid))].copy()
    d["vendor"] = d.Manufacturer.str.split().str[0].str.upper()
    d["stem"] = [str(stem_for(p, r, u)) for p, r, u in zip(d.pid, d["round"], d.uid)]
    if check_files:
        d = d[[stem_exists(s) for s in d.stem]]
    return d.reset_index(drop=True)


def load_for_detector(stem, spacing=C.LUNA_SPACING):
    """-> ((1,1,D,H,W) float tensor in [0,1], meta) at detector spacing; meta keeps origin/direction, spacing/shape updated."""
    arr, meta = load_volume(stem)
    img = sitk.GetImageFromArray(arr.astype(np.float32))
    img.SetSpacing(tuple(float(s) for s in meta["spacing_zyx"][::-1])); img.SetOrigin(tuple(meta["origin_xyz"]))
    img.SetDirection(tuple(meta["direction"]))
    size = [int(round(n * s / t)) for n, s, t in zip(img.GetSize(), img.GetSpacing(), spacing[::-1])]
    out = sitk.Resample(img, size, sitk.Transform(), sitk.sitkBSpline, img.GetOrigin(), tuple(float(s) for s in spacing[::-1]),
                        img.GetDirection(), -1000.0, sitk.sitkFloat32)
    res = sitk.GetArrayFromImage(out)
    meta = {**meta, "archive_spacing_zyx": meta["spacing_zyx"], "spacing_zyx": [float(s) for s in spacing], "shape_zyx": list(res.shape)}
    x = np.clip(normalize(res), 0.0, 1.0)            # B-spline overshoot past the HU window (the LUNA16 archive is windowed after resampling)
    return torch.from_numpy(x.astype(np.float32))[None, None], meta


def native_stem(stem) -> Path:
    stem = Path(stem); return NATIVE / stem.parent.name / stem.name


def convert_native(stem, dicom_dir, spacing=C.LUNA_SPACING) -> str:
    """Raw NLST DICOM -> detector spacing, cropped to the archive's lung box (same world region), windowed, saved."""
    out = native_stem(stem)
    if stem_exists(out):
        return "skip"
    ref = load_meta(stem)
    vol = load_dicom_series(dicom_dir, ref["uid"])
    if vol.uid != ref["uid"]:
        raise RuntimeError(f"series UID mismatch {vol.uid} vs {ref['uid']}")
    res = resample(vol, spacing)
    # archive crop box in world mm -> index box on the new grid, shrunk by the margin so finalize_volume adds it back
    D = np.array(res.direction).reshape(3, 3); sp_xyz = np.array(spacing, float)[::-1]
    a_lo = np.array(ref["origin_xyz"], float)
    a_hi = a_lo + D @ ((np.array(ref["crop_stop_zyx"]) - np.array(ref["crop_offset_zyx"]))[::-1] * np.array(ref["spacing_zyx"])[::-1])
    lo = D.T @ (a_lo - np.array(res.origin)) / sp_xyz; hi = D.T @ (a_hi - np.array(res.origin)) / sp_xyz   # (x, y, z) index
    m = ref["margin_mm"] / sp_xyz
    lo, hi = np.floor(np.minimum(lo, hi) + m).astype(int)[::-1], np.ceil(np.maximum(lo, hi) - m).astype(int)[::-1]   # -> (z, y, x)
    shp = np.array(res.hu.shape); lo = np.clip(lo, 0, shp - 1); hi = np.clip(hi, lo + 1, shp)
    mask = np.zeros(res.hu.shape, np.uint8); mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = 1
    arr, meta = finalize_volume(vol, res, mask, "archive_lung_box", spacing)
    meta.update({"domain": "nlst_native", "archive_stem": str(stem)})
    save_volume(out, arr, meta)
    return f"ok {arr.shape}"


def load_native(stem):
    """-> ((1,1,D,H,W) float tensor in [0,1], meta), exactly like stage1.dataset.load_full_volume on LUNA16."""
    arr, meta = load_volume(native_stem(stem))
    return torch.from_numpy(normalize(arr.astype(np.float32)))[None, None], meta
