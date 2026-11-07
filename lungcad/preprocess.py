"""S2 — the single shared preprocessing module.

One HU window. One resampling function. One normalisation. One storage format.
Used by every stage and BOTH domains (LUNA16 .mhd and NLST DICOM), imported, never copied.

Pipeline (preprocess_volume):
    load  ->  HU float32 (z,y,x)  ->  resample to target spacing (B-spline image / nearest mask)
          ->  lung mask (LUNA16 seg-lungs or classical threshold)  ->  crop to lung bbox + margin
          ->  clip to [HU_MIN, HU_MAX] as int16  ->  LZ4 frame + JSON sidecar

The sidecar stores the exact ITK world<->voxel transform of the *stored crop*, so any world
coordinate (LUNA16 annotation, NLST nodule location, detector output) round-trips losslessly.

Intensity normalisation to [0,1] happens at load time (normalize / normalize_torch) and is the
same function for both domains — so both domains are preprocessed identically.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import lz4.frame
from scipy import ndimage

from . import config as C

PREPROCESS_VERSION = "1.0.0"
_INTERP = {"bspline": sitk.sitkBSpline, "linear": sitk.sitkLinear, "nearest": sitk.sitkNearestNeighbor}


@dataclass
class Volume:
    """A CT volume in HU with full ITK geometry. Array axis order is (z, y, x)."""
    hu: np.ndarray
    spacing: tuple      # (z, y, x) mm
    origin: tuple       # (x, y, z) mm, ITK world position of voxel (0,0,0)
    direction: tuple    # 9 floats, ITK row-major direction cosines
    uid: str = ""
    source: str = ""

    @property
    def shape(self):
        return tuple(int(s) for s in self.hu.shape)


# ----------------------------------------------------------------------------- I/O
def _from_sitk(img: sitk.Image, uid: str, source: str) -> Volume:
    arr = sitk.GetArrayFromImage(img)          # (z, y, x)
    sx, sy, sz = img.GetSpacing()
    return Volume(hu=arr, spacing=(float(sz), float(sy), float(sx)), origin=tuple(float(o) for o in img.GetOrigin()),
                  direction=tuple(float(d) for d in img.GetDirection()), uid=uid, source=source)


def _to_sitk(vol: Volume) -> sitk.Image:
    img = sitk.GetImageFromArray(vol.hu)
    img.SetSpacing((vol.spacing[2], vol.spacing[1], vol.spacing[0]))
    img.SetOrigin(vol.origin)
    img.SetDirection(vol.direction)
    return img


def load_mhd(path) -> Volume:
    """LUNA16 MetaImage (.mhd + .raw/.zraw). Values are already HU."""
    path = Path(path)
    img = sitk.ReadImage(str(path))
    if img.GetPixelID() != sitk.sitkFloat32 and "seg-lungs" not in str(path):
        img = sitk.Cast(img, sitk.sitkFloat32)
    return _from_sitk(img, path.stem, str(path))


def load_dicom_series(dirpath, series_uid: str | None = None) -> Volume:
    """NLST DICOM series directory -> HU volume. SimpleITK applies RescaleSlope/Intercept, sorts slices by
    position and derives z-spacing from the slice positions (SliceThickness is NOT trusted)."""
    dirpath = str(dirpath)
    reader = sitk.ImageSeriesReader()
    ids = reader.GetGDCMSeriesIDs(dirpath)
    if not ids:
        raise ValueError(f"no DICOM series in {dirpath}")
    sid = series_uid if series_uid in ids else ids[0]
    files = reader.GetGDCMSeriesFileNames(dirpath, sid)
    reader.SetFileNames(files)
    img = reader.Execute()
    img = sitk.Cast(img, sitk.sitkFloat32)
    return _from_sitk(img, sid, dirpath)


def dicom_z_uniformity(dirpath) -> dict:
    """Check slice-position uniformity independently of SimpleITK (flags missing/duplicate slices)."""
    import pydicom
    zs = []
    for f in Path(dirpath).glob("*.dcm"):
        ds = pydicom.dcmread(str(f), stop_before_pixels=True, specific_tags=["ImagePositionPatient"])
        zs.append(float(ds.ImagePositionPatient[2]))
    zs = np.sort(np.array(zs))
    dz = np.diff(zs)
    return {"n_slices": int(len(zs)), "dz_median": float(np.median(dz)) if len(dz) else float("nan"),
            "dz_min": float(dz.min()) if len(dz) else float("nan"), "dz_max": float(dz.max()) if len(dz) else float("nan"),
            "uniform": bool(len(dz) and np.allclose(dz, np.median(dz), atol=0.05))}


# ----------------------------------------------------------------------------- resampling
def resample(vol: Volume, new_spacing, interp: str = "bspline", is_mask: bool = False) -> Volume:
    """Resample to new (z,y,x) spacing keeping origin/direction. Masks use nearest neighbour."""
    img = _to_sitk(vol)
    old_sp = np.array(img.GetSpacing(), dtype=np.float64)
    old_sz = np.array(img.GetSize(), dtype=np.float64)
    new_sp = np.array([new_spacing[2], new_spacing[1], new_spacing[0]], dtype=np.float64)
    new_sz = np.maximum(1, np.round(old_sz * old_sp / new_sp)).astype(int)
    interpolator = sitk.sitkNearestNeighbor if is_mask else _INTERP[interp]
    default = 0.0 if is_mask else float(C.HU_MIN)
    out = sitk.Resample(img, [int(s) for s in new_sz], sitk.Transform(), interpolator,
                        img.GetOrigin(), [float(s) for s in new_sp], img.GetDirection(), default, img.GetPixelID())
    return _from_sitk(out, vol.uid, vol.source)


# ----------------------------------------------------------------------------- lung mask + crop
def lung_mask_classical(hu: np.ndarray, spacing, thresh: float = -320.0) -> np.ndarray:
    """Threshold + connected components. Good enough for a bounding-box crop, deterministic, ~1-2 s.
    Removes air connected to the x/y image border (outside the body), keeps components >= 10% of the largest."""
    air = hu < thresh
    lab, n = ndimage.label(air)
    if n == 0:
        return np.zeros(hu.shape, np.uint8)
    border = np.unique(np.concatenate([lab[:, 0, :].ravel(), lab[:, -1, :].ravel(), lab[:, :, 0].ravel(), lab[:, :, -1].ravel()]))
    sizes = ndimage.sum(air, lab, index=np.arange(1, n + 1))
    sizes[border[border > 0] - 1] = 0
    if sizes.max() == 0:
        return np.zeros(hu.shape, np.uint8)
    keep = np.where(sizes >= 0.10 * sizes.max())[0] + 1
    mask = np.isin(lab, keep)
    # drop tiny gut-air etc. by requiring a minimum physical volume of 50 ml
    vox_ml = float(np.prod(spacing)) / 1000.0
    if mask.sum() * vox_ml < 50:
        return np.zeros(hu.shape, np.uint8)
    return mask.astype(np.uint8)


def lung_bbox(mask: np.ndarray, spacing, margin_mm: float = C.LUNG_MARGIN_MM):
    """(start_zyx, stop_zyx) of the mask bounding box expanded by margin_mm, clipped to the volume."""
    idx = np.where(mask > 0)
    if len(idx[0]) == 0:
        return np.zeros(3, int), np.array(mask.shape, int)
    lo = np.array([a.min() for a in idx]); hi = np.array([a.max() + 1 for a in idx])
    m = np.ceil(margin_mm / np.array(spacing)).astype(int)
    return np.maximum(lo - m, 0), np.minimum(hi + m, np.array(mask.shape))


# ----------------------------------------------------------------------------- intensity
def window_hu(hu: np.ndarray) -> np.ndarray:
    """Clip to the ONE HU window and store as int16."""
    return np.clip(np.rint(hu), C.HU_MIN, C.HU_MAX).astype(np.int16)


def normalize(x):
    """[HU_MIN, HU_MAX] -> [0, 1]. Works for numpy arrays and torch tensors (cast to float first)."""
    return (x - C.HU_MIN) / (C.HU_MAX - C.HU_MIN)


def normalize_torch(t):
    import torch
    return (t.to(torch.float32) - C.HU_MIN) / (C.HU_MAX - C.HU_MIN)


def intensity_stats(arr: np.ndarray) -> dict:
    a = arr.astype(np.float32)
    return {"mean_hu": float(a.mean()), "std_hu": float(a.std()), "p01": float(np.percentile(a, 1)),
            "p50": float(np.percentile(a, 50)), "p99": float(np.percentile(a, 99)),
            "frac_air(<-900)": float((a < -900).mean()), "frac_soft(-100..100)": float(((a > -100) & (a < 100)).mean()),
            "min": float(a.min()), "max": float(a.max())}


# ----------------------------------------------------------------------------- transforms
def world_to_voxel(meta: dict, xyz) -> np.ndarray:
    """World (x,y,z) mm -> continuous voxel index (z,y,x) of the stored crop."""
    xyz = np.atleast_2d(np.asarray(xyz, dtype=np.float64))
    D = np.array(meta["direction"], dtype=np.float64).reshape(3, 3)
    o = np.array(meta["origin_xyz"], dtype=np.float64)
    sp = np.array(meta["spacing_zyx"], dtype=np.float64)[::-1]
    idx_xyz = (np.linalg.inv(D) @ (xyz - o).T).T / sp
    return idx_xyz[:, ::-1]


def voxel_to_world(meta: dict, zyx) -> np.ndarray:
    """Continuous voxel index (z,y,x) of the stored crop -> world (x,y,z) mm."""
    zyx = np.atleast_2d(np.asarray(zyx, dtype=np.float64))
    D = np.array(meta["direction"], dtype=np.float64).reshape(3, 3)
    o = np.array(meta["origin_xyz"], dtype=np.float64)
    sp = np.array(meta["spacing_zyx"], dtype=np.float64)[::-1]
    idx_xyz = zyx[:, ::-1] * sp
    return (D @ idx_xyz.T).T + o


# ----------------------------------------------------------------------------- storage
# NOTE: SeriesInstanceUIDs contain dots, so Path.with_suffix() would truncate them — always append by string.
def lz4_path(stem) -> Path: return Path(f"{stem}.lz4")
def json_path(stem) -> Path: return Path(f"{stem}.json")
def exists(stem) -> bool: return lz4_path(stem).exists() and json_path(stem).exists()


def save_volume(stem, arr: np.ndarray, meta: dict) -> Path:
    stem = Path(stem); stem.parent.mkdir(parents=True, exist_ok=True)
    assert arr.dtype == np.int16 and arr.ndim == 3
    arr = np.ascontiguousarray(arr)
    meta = dict(meta, shape_zyx=[int(s) for s in arr.shape], dtype="int16", preprocess_version=PREPROCESS_VERSION,
                hu_window=[C.HU_MIN, C.HU_MAX], created=time.strftime("%Y-%m-%dT%H:%M:%S"))
    tmp = Path(f"{stem}.lz4.tmp")
    with open(tmp, "wb") as f:
        f.write(lz4.frame.compress(arr.tobytes(), compression_level=1, store_size=True))
    tmp.rename(lz4_path(stem))
    with open(json_path(stem), "w") as f:
        json.dump(meta, f, indent=1)
    return stem


def load_meta(stem) -> dict:
    with open(json_path(stem)) as f:
        return json.load(f)


def load_volume(stem):
    """-> (int16 array (z,y,x), meta dict). Raw DICOM/mhd is never touched by a training loop."""
    meta = load_meta(stem)
    with open(lz4_path(stem), "rb") as f:
        buf = lz4.frame.decompress(f.read())
    arr = np.frombuffer(buf, dtype=np.int16).reshape(meta["shape_zyx"])
    return arr, meta


# ----------------------------------------------------------------------------- the pipeline
def lung_mask_lungmask(res: Volume, inferer=None, batch_size: int = 16) -> np.ndarray:
    """Deep-learning lung mask (Hofmanninger et al. 2020, model R231) on the resampled volume; labels 1/2 -> lung.
    Pass a shared LMInferer to avoid reloading the model; runs on the GPU if available."""
    if inferer is None:
        from lungmask import LMInferer
        inferer = LMInferer(modelname="R231", force_cpu=False, batch_size=batch_size, tqdm_disable=True)
    seg = inferer.apply(_to_sitk(res))
    return (seg > 0).astype(np.uint8)


def mask_is_plausible(mask: np.ndarray, spacing, lo_l: float = 1.0, hi_l: float = 12.0) -> bool:
    return lo_l <= mask.sum() * float(np.prod(spacing)) / 1e6 <= hi_l


def finalize_volume(vol: Volume, res: Volume, mask: np.ndarray | None, mask_source: str, target_spacing,
                    margin_mm: float = C.LUNG_MARGIN_MM, interp: str = "bspline", t0: float | None = None):
    """Second half of the pipeline: crop the resampled volume to the mask bbox, window, build the sidecar."""
    t0 = t0 or time.time(); hu = res.hu
    if mask is None or mask.sum() < 1000:
        mask = np.ones(hu.shape, np.uint8); mask_source += "+fallback_nocrop"
    lo, hi = lung_bbox(mask, target_spacing, margin_mm)
    crop = hu[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    arr = window_hu(crop)
    D = np.array(res.direction).reshape(3, 3)
    sp_xyz = np.array(target_spacing, dtype=np.float64)[::-1]
    origin_crop = np.array(res.origin) + D @ (lo[::-1].astype(np.float64) * sp_xyz)
    meta = {
        "uid": vol.uid, "source": vol.source,
        "spacing_zyx": [float(s) for s in target_spacing], "origin_xyz": [float(o) for o in origin_crop],
        "direction": [float(d) for d in res.direction],
        "crop_offset_zyx": [int(v) for v in lo], "crop_stop_zyx": [int(v) for v in hi],
        "resampled_shape_zyx": [int(s) for s in hu.shape],
        "original_spacing_zyx": [float(s) for s in vol.spacing], "original_shape_zyx": list(vol.shape),
        "original_origin_xyz": [float(o) for o in vol.origin],
        "mask_source": mask_source, "mask_voxels": int(mask.sum()), "mask_litres": round(float(mask.sum()) * float(np.prod(target_spacing)) / 1e6, 3),
        "margin_mm": margin_mm, "interp": interp,
        "lung_bbox_mm_zyx": [float((hi[i] - lo[i]) * target_spacing[i]) for i in range(3)],
        "preprocess_seconds": round(time.time() - t0, 2),
    }
    return arr, meta


def preprocess_volume(vol: Volume, target_spacing, mask_vol: Volume | None = None, mask_labels=C.LUNA_SEG_LUNG_LABELS,
                      mask_mode: str = "classical", margin_mm: float = C.LUNG_MARGIN_MM, interp: str = "bspline", lungmask_inferer=None):
    """Full S2 pipeline. Returns (int16 array (z,y,x), sidecar meta).

    mask_vol:  an external lung mask in the ORIGINAL geometry (LUNA16 seg-lungs); labels in mask_labels are lung.
    mask_mode: 'lungmask' (R231 model, classical fallback), 'classical' (threshold + components) or 'none'.
    """
    t0 = time.time()
    res = resample(vol, target_spacing, interp=interp)
    mask = None; mask_source = "none"
    if mask_vol is not None:
        m = resample(mask_vol, target_spacing, is_mask=True).hu
        mask = np.isin(m, mask_labels).astype(np.uint8); mask_source = "external"
        if mask.sum() < 1000: mask = None
    if mask is None and mask_mode == "lungmask":
        try:
            mask = lung_mask_lungmask(res, lungmask_inferer); mask_source = "lungmask"
            if not mask_is_plausible(mask, target_spacing): mask = None; mask_source = "lungmask_implausible"
        except Exception as e:  # noqa
            mask = None; mask_source = f"lungmask_failed({type(e).__name__})"
    if mask is None and mask_mode in ("classical", "lungmask"):
        m = lung_mask_classical(res.hu, target_spacing)
        if mask_is_plausible(m, target_spacing): mask = m; mask_source = (mask_source + "+" if mask_source not in ("none",) else "") + "classical"
    return finalize_volume(vol, res, mask, mask_source, target_spacing, margin_mm, interp, t0)


# ----------------------------------------------------------------------------- CLI: the S2 "done when" check
def _cli():
    import argparse
    ap = argparse.ArgumentParser(description="Run one LUNA16 .mhd and/or one NLST DICOM dir through the shared module and print intensity stats.")
    ap.add_argument("--mhd"); ap.add_argument("--dicom"); ap.add_argument("--out", help="output stem prefix (optional)")
    a = ap.parse_args()
    for kind, src in (("luna16", a.mhd), ("nlst", a.dicom)):
        if not src:
            continue
        vol = load_mhd(src) if kind == "luna16" else load_dicom_series(src)
        sp = C.LUNA_SPACING if kind == "luna16" else C.NLST_SPACING
        mask_vol = None
        if kind == "luna16":
            seg = C.LUNA_SEG / f"{vol.uid}.mhd"
            if seg.exists():
                mask_vol = load_mhd(seg)
        arr, meta = preprocess_volume(vol, sp, mask_vol=mask_vol)
        print(f"[{kind}] {vol.uid}\n  original {vol.shape} @ {vol.spacing}\n  stored   {arr.shape} @ {meta['spacing_zyx']}  mask={meta['mask_source']}  {meta['preprocess_seconds']}s")
        for k, v in intensity_stats(arr).items():
            print(f"  {k:22s} {v: .4f}")
        n = normalize(arr.astype(np.float32)); print(f"  normalized range [{n.min():.3f}, {n.max():.3f}] mean {n.mean():.4f}")
        if a.out:
            save_volume(f"{a.out}_{kind}", arr, meta)


if __name__ == "__main__":
    _cli()
