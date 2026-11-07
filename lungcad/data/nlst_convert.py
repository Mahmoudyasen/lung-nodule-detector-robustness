"""NLST DICOM -> 1.5 mm isotropic, lung-cropped, int16, LZ4 (Execution Guide v5 §2 "preprocess NLST early").

Pipeline per series:  worker (CPU): read DICOM, slice-uniformity check, header, B-spline resample to 1.5 mm
                      main (GPU):   lungmask R231 lung mask (plausibility 1-12 L), classical threshold fallback,
                                    crop + window + LZ4 + sidecar.
The pilot showed the classical mask alone fails on ~30 % of NLST series (tight FOV cuts the lungs at the image edge,
connecting lung air to the outside), hence the model. ~5 s/series on the 3090 while training runs alongside.

Resumable: a series is done when its .json sidecar exists; progress is appended to done.log / failed.log. Run the
50-scan pilot (--pilot 50 --montage) and inspect the montage before the full queue. The full run refuses to start if
the projected output (pilot mean size x queue length) exceeds 90 % of free space.

Output layout:  nlst/preprocessed/<pid>/<pid>_T<round>_<uid-tail>.{lz4,json}
"""
from __future__ import annotations
import argparse, json, shutil, threading, time, traceback
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from .. import config as C
from ..preprocess import (load_dicom_series, resample, finalize_volume, lung_mask_lungmask, lung_mask_classical, mask_is_plausible,
                          save_volume, dicom_z_uniformity, load_volume, exists as stem_exists, lz4_path, Volume)

HDR_KEYS = ["SliceThickness", "ConvolutionKernel", "PixelSpacing", "Manufacturer", "ManufacturerModelName", "KVP", "XRayTubeCurrent",
            "Exposure", "ReconstructionDiameter", "ClinicalTrialTimePointID", "StudyDate", "PatientID"]


def stem_for(pid, rnd, uid) -> Path:
    return C.NLST_PRE / str(pid) / f"{pid}_T{rnd}_{str(uid)[-10:]}"


def load_stage(job):
    """Worker: everything that needs no GPU. Returns (job, resampled Volume, original Volume geometry, info) or an error."""
    pid, rnd, uid, path = job
    try:
        import pydicom
        t0 = time.time()
        zinfo = dicom_z_uniformity(path)
        vol = load_dicom_series(path, uid)
        ds = pydicom.dcmread(str(next(Path(path).glob("*.dcm"))), stop_before_pixels=True)
        hdr = {k: (str(getattr(ds, k)) if hasattr(ds, k) else None) for k in HDR_KEYS}
        res = resample(vol, C.NLST_SPACING)
        geom = Volume(hu=np.zeros((1, 1, 1), np.int16), spacing=vol.spacing, origin=vol.origin, direction=vol.direction, uid=vol.uid, source=vol.source)
        geom_shape = vol.shape
        return job, res, (geom, geom_shape), {"hdr": hdr, "z": zinfo, "load_s": round(time.time() - t0, 1)}, None
    except Exception as e:
        return job, None, None, None, f"{type(e).__name__}: {e} | {traceback.format_exc().splitlines()[-1]}"


def finish_stage(job, res, geom_pack, info, inferer, mask_mode):
    """Main process: mask (GPU) -> crop -> save."""
    pid, rnd, uid, path = job
    geom, geom_shape = geom_pack
    t0 = time.time(); mask = None; src = "none"
    if mask_mode == "lungmask":
        try:
            mask = lung_mask_lungmask(res, inferer); src = "lungmask"
            if not mask_is_plausible(mask, C.NLST_SPACING): mask, src = None, "lungmask_implausible"
        except Exception as e:  # noqa
            mask, src = None, f"lungmask_failed({type(e).__name__})"
    if mask is None:
        m = lung_mask_classical(res.hu, C.NLST_SPACING)
        if mask_is_plausible(m, C.NLST_SPACING): mask = m; src = src + "+classical" if src != "none" else "classical"
    # a lightweight Volume standing in for the original (only its geometry is written to the sidecar)
    class _G: pass
    g = _G(); g.uid, g.source, g.spacing, g.origin, g.shape = geom.uid, geom.source, geom.spacing, geom.origin, geom_shape
    arr, meta = finalize_volume(g, res, mask, src, C.NLST_SPACING, t0=t0)
    meta.update({"domain": "nlst", "pid": int(pid), "round": int(rnd), "series_uid": str(uid), "dicom_dir": str(path), "dicom": info["hdr"],
                 "z_uniformity": info["z"], "timepoint_consistent": (info["hdr"].get("ClinicalTrialTimePointID") in (None, "", f"T{rnd}")),
                 "load_seconds": info["load_s"]})
    stem = stem_for(pid, rnd, uid); save_volume(stem, arr, meta)
    return json.dumps({"shape": list(arr.shape), "mask": src, "litres": meta["mask_litres"], "bytes": lz4_path(stem).stat().st_size,
                       "s": round(time.time() - t0 + info["load_s"], 1), "tp": info["hdr"].get("ClinicalTrialTimePointID"), "uniform_z": info["z"]["uniform"]})


def montage(stems, out_png, n=50):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    stems = stems[:n]; cols = 10; rows = int(np.ceil(len(stems) / cols)) * 2
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    for i, st in enumerate(stems):
        arr, meta = load_volume(st); r, c = (i // cols) * 2, i % cols
        axes[r, c].imshow(arr[arr.shape[0] // 2], cmap="gray", vmin=-1000, vmax=400)
        axes[r, c].set_title(f"{Path(st).name[:13]} {meta['mask_source'][:9]} {meta.get('mask_litres', 0):.1f}L", fontsize=5)
        axes[r + 1, c].imshow(arr[:, arr.shape[1] // 2, :], cmap="gray", vmin=-1000, vmax=400, aspect="auto")
    for ax in axes.ravel(): ax.axis("off")
    plt.tight_layout(); plt.savefig(out_png, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pilot", type=int, default=0, help="convert only N randomly chosen series of the queue")
    ap.add_argument("--montage", action="store_true")
    ap.add_argument("--rounds", default="0,1,2")
    ap.add_argument("--mask", default="lungmask", choices=["lungmask", "classical"])
    ap.add_argument("--force", action="store_true", help="skip the free-space check")
    a = ap.parse_args()
    q = pd.read_csv(C.NLST_META / "nlst_primary_series.csv", low_memory=False)
    q = q[q["round"].isin([int(r) for r in a.rounds.split(",")])].sort_values(["PatientID", "round"])
    if a.pilot:
        rng = np.random.RandomState(0); q = q.iloc[rng.choice(len(q), size=min(a.pilot, len(q)), replace=False)]
    jobs = [(int(r.PatientID), int(r["round"]), r.SeriesInstanceUID, r.local_path) for _, r in q.iterrows()]
    C.NLST_PRE.mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not stem_exists(stem_for(j[0], j[1], j[2]))]
    print(f"queue {len(jobs)} series, {len(jobs) - len(todo)} already done, {len(todo)} to convert", flush=True)
    done_log = open(C.NLST_PRE / "done.log", "a"); fail_log = open(C.NLST_PRE / "failed.log", "a")
    if not a.pilot and not a.force and todo:
        done = [l.split("\t") for l in (C.NLST_PRE / "done.log").read_text().splitlines() if "\t" in l]
        sizes = [json.loads(d[2]).get("bytes", 0) for d in done if d[1] == "ok" and d[2].startswith("{")]
        if not sizes: raise SystemExit("run a pilot first (--pilot 50) so the output size can be projected")
        need = np.mean(sizes) * len(todo); free = shutil.disk_usage(C.THESIS_DATA).free
        print(f"projected output {need/1e9:.0f} GB for {len(todo)} series; free {free/1e9:.0f} GB", flush=True)
        if need > 0.9 * free: raise SystemExit("insufficient free space — reduce the queue (--rounds) or free space, or --force")
    inferer = None
    if a.mask == "lungmask":
        from lungmask import LMInferer
        inferer = LMInferer(modelname="R231", force_cpu=False, batch_size=16, tqdm_disable=True)
    t0 = time.time(); counts = {"ok": 0, "fail": 0}
    # Back-pressure: workers load ~2x faster than the GPU stage consumes, and imap_unordered buffers finished results
    # without limit in this process (grew to ~48 GB in 90 min on 2026-09-22). Cap loaded-but-unfinished series.
    slots = threading.BoundedSemaphore(a.workers * 2)
    def feed():
        for j in todo:
            slots.acquire(); yield j
    with Pool(a.workers) as pool:
        for i, (job, res, geom, info, err) in enumerate(pool.imap_unordered(load_stage, feed(), chunksize=1), 1):
            slots.release()
            uid = job[2]
            if err is None:
                try:
                    msg = finish_stage(job, res, geom, info, inferer, a.mask); status = "ok"
                except Exception as e:  # noqa
                    msg = f"{type(e).__name__}: {e} | {traceback.format_exc().splitlines()[-1]}"; status = "fail"
            else:
                msg, status = err, "fail"
            counts[status] += 1
            (fail_log if status == "fail" else done_log).write(f"{time.strftime('%F %T')}\t{status}\t{msg}\t{uid}\n")
            (fail_log if status == "fail" else done_log).flush()
            if i % 25 == 0 or status == "fail":
                el = time.time() - t0
                print(f"{i}/{len(todo)} {counts} {el:.0f}s  {el/i:.1f}s/series  ETA {(len(todo)-i)*el/i/3600:.1f} h", flush=True)
    print("finished", counts, f"{time.time()-t0:.0f}s")
    if a.montage:
        stems = [stem_for(p, r, u) for p, r, u, _ in jobs if stem_exists(stem_for(p, r, u))]
        out = C.NLST_PRE / f"pilot_montage_{time.strftime('%Y%m%d_%H%M')}.png"; montage(stems, out); print("montage ->", out)


if __name__ == "__main__":
    main()
