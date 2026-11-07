"""Run all LUNA16 scans through the shared module -> preprocessed/subsetK/<uid>.{lz4,json}
and emit voxel-space annotation / candidate tables. Resumable: existing sidecars are skipped."""
from __future__ import annotations
import argparse, json, time, traceback
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from .. import config as C
from ..preprocess import load_mhd, preprocess_volume, save_volume, load_meta, world_to_voxel, exists as stem_exists
from .luna16_index import load_index


def stem_for(uid: str, subset: int) -> Path:
    return C.LUNA_PRE / f"subset{subset}" / uid


def process_one(args):
    uid, subset, mhd_path = args
    stem = stem_for(uid, subset)
    if stem_exists(stem):
        return uid, "skip", ""
    try:
        vol = load_mhd(mhd_path)
        seg = C.LUNA_SEG / f"{uid}.mhd"
        mask_vol = load_mhd(seg) if seg.exists() else None
        arr, meta = preprocess_volume(vol, C.LUNA_SPACING, mask_vol=mask_vol, mask_labels=C.LUNA_SEG_LUNG_LABELS)
        meta["subset"] = int(subset); meta["domain"] = "luna16"
        save_volume(stem, arr, meta)
        return uid, "ok", f"{arr.shape} {meta['mask_source']} {meta['preprocess_seconds']}s"
    except Exception as e:
        return uid, "fail", f"{e}\n{traceback.format_exc()}"


def to_voxel_table(df_world: pd.DataFrame, index: pd.DataFrame, radius_col: str | None) -> pd.DataFrame:
    """Add vz,vy,vx (continuous voxel coords in the stored crop), in_crop, and per-axis radius in voxels."""
    sub = index.set_index("seriesuid").subset
    out = []
    for uid, g in df_world.groupby("seriesuid", sort=False):
        if uid not in sub.index:
            continue
        meta = load_meta(stem_for(uid, int(sub[uid])))
        v = world_to_voxel(meta, g[["coordX", "coordY", "coordZ"]].values)
        shp = np.array(meta["shape_zyx"])
        gg = g.copy()
        gg["vz"], gg["vy"], gg["vx"] = v[:, 0], v[:, 1], v[:, 2]
        gg["in_crop"] = np.all((v >= 0) & (v < shp), axis=1)
        if radius_col:
            sp = np.array(meta["spacing_zyx"])
            r = gg[radius_col].values / 2.0
            gg["rz_vox"], gg["ry_vox"], gg["rx_vox"] = r / sp[0], r / sp[1], r / sp[2]
        out.append(gg)
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    index = load_index()
    jobs = [(r.seriesuid, int(r.subset), r.mhd_path) for r in index.itertuples()]
    if a.limit:
        jobs = jobs[:a.limit]
    C.LUNA_PRE.mkdir(parents=True, exist_ok=True)
    log = open(C.LUNA_PRE / "preprocess.log", "a")
    t0 = time.time(); counts = {"ok": 0, "skip": 0, "fail": 0}
    with Pool(a.workers) as pool:
        for i, (uid, status, msg) in enumerate(pool.imap_unordered(process_one, jobs), 1):
            counts[status] += 1
            line = f"[{time.strftime('%F %T')}] {status:4s} {uid} {msg.splitlines()[0] if msg else ''}"
            log.write(line + ("\n" + msg if status == "fail" else "") + "\n"); log.flush()
            if i % 25 == 0 or status == "fail":
                print(f"{i}/{len(jobs)} {counts} {time.time()-t0:.0f}s", flush=True)
    print("done", counts, f"{time.time()-t0:.0f}s")
    if a.limit:
        return
    ann = pd.read_csv(C.LUNA_RAW / "annotations.csv")
    excl = pd.read_csv(C.LUNA_EVAL / "annotations" / "annotations_excluded.csv")
    cand = pd.read_csv(C.LUNA_RAW / "candidates_V2.csv")
    annv = to_voxel_table(ann, index, "diameter_mm"); annv.to_csv(C.LUNA_PRE / "annotations_voxel.csv", index=False)
    exclv = to_voxel_table(excl, index, None); exclv.to_csv(C.LUNA_PRE / "annotations_excluded_voxel.csv", index=False)
    candv = to_voxel_table(cand, index, None); candv["cand_id"] = np.arange(len(candv)); candv.to_csv(C.LUNA_PRE / "candidates_v2_voxel.csv", index=False)
    summ = {"annotations": int(len(annv)), "annotations_in_crop": int(annv.in_crop.sum()),
            "candidates_v2": int(len(candv)), "candidates_in_crop": int(candv.in_crop.sum()),
            "candidates_pos_in_crop": int(candv[candv["class"] == 1].in_crop.sum()), "candidates_pos": int((candv["class"] == 1).sum())}
    with open(C.LUNA_PRE / "voxel_tables_summary.json", "w") as f:
        json.dump(summ, f, indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
