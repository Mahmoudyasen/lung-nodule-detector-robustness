"""E5 — review of the nodules the S4 detector missed (all 10 folds, own_candidates_all.csv).

    python scripts/e5_miss_review.py

For every reference nodule that no candidate hits (official rule: centre distance <= radius) it records the
nearest candidate (distance, distance/radius, probability), whether it lies in the stored lung crop, nearby
excluded findings, local intensity (mean HU in the nodule sphere, fraction of the 2-radius shell above -400 HU
as a juxta-pleural/vascular-contact proxy), and renders axial/coronal/sagittal 60 mm crops with the reference
circle and the nearest candidates. Analysis only: frozen files are read, never written.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lungcad import config as C
from lungcad.data.luna16_index import load_index
from lungcad.data.luna16_preprocess import stem_for
from lungcad.preprocess import load_volume, world_to_voxel
from lungcad.eval.froc import load_reference, cap_candidates

OUT = C.WORK / "wp0" / "analysis" / "e5_misses"
CANDS = C.STAGE1 / "own_candidates" / "own_candidates_all.csv"
HALF_MM = 30.0


def find_misses():
    ann, excl, _ = load_reference(); index = load_index(); sub = index.set_index("seriesuid")
    cands = pd.read_csv(CANDS, dtype={"seriesuid": str}); cg = {k: cap_candidates(g, 100) for k, g in cands.groupby("seriesuid")}
    rows = []
    for _, n in ann.iterrows():
        uid = str(n.seriesuid); r = n.diameter_mm / 2; xyz = n[["coordX", "coordY", "coordZ"]].values.astype(float)
        c = cg.get(uid)
        if c is None or not len(c):
            d = np.array([np.inf]); p = np.array([np.nan]); i = 0
        else:
            d = np.sqrt(((c[["coordX", "coordY", "coordZ"]].values - xyz) ** 2).sum(1)); p = c.probability.values; i = int(np.argmin(d))
        hit = bool((d <= r).any()) if np.isfinite(d).any() else False
        near2 = (d <= 2 * r) if np.isfinite(d).any() else np.array([False])
        e = excl[excl.seriesuid == uid]
        de = np.sqrt(((e[["coordX", "coordY", "coordZ"]].values - xyz) ** 2).sum(1)) if len(e) else np.array([np.inf])
        rows.append({"seriesuid": uid, "fold": int(sub.loc[uid, "fold"]), "subset": int(sub.loc[uid, "subset"]),
                     "coordX": xyz[0], "coordY": xyz[1], "coordZ": xyz[2], "diameter_mm": n.diameter_mm, "hit": hit,
                     "nearest_mm": float(d[i]), "nearest_over_r": float(d[i] / r), "nearest_prob": float(p[i]) if np.isfinite(d[i]) else np.nan,
                     "best_prob_within_2r": float(p[near2].max()) if near2.any() else np.nan,
                     "n_cands_scan": 0 if c is None else len(c), "nearest_excluded_mm": float(de.min()),
                     "n_nodules_scan": int((ann.seriesuid == uid).sum())})
    df = pd.DataFrame(rows); return df, cands, excl


def local_stats(arr, meta, xyz, diam):
    v = world_to_voxel(meta, xyz)[0]; sp = np.array(meta["spacing_zyx"]); shp = np.array(arr.shape)
    in_crop = bool(np.all((v >= 0) & (v < shp)))
    r = diam / 2; R = 2 * r; lo = np.maximum(0, np.floor(v - R / sp).astype(int)); hi = np.minimum(shp, np.ceil(v + R / sp).astype(int) + 1)
    if not in_crop or np.any(hi <= lo):
        return in_crop, np.nan, np.nan, np.nan
    blk = arr[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32)
    zz, yy, xx = np.meshgrid(*[(np.arange(lo[k], hi[k]) - v[k]) * sp[k] for k in range(3)], indexing="ij"); dist = np.sqrt(zz**2 + yy**2 + xx**2)
    inside = dist <= r; shell = (dist > r) & (dist <= R)
    # crop-boundary distance in mm: a nodule near the crop edge suggests the lung mask cut it
    edge_mm = float(np.min(np.r_[v * sp, (shp - 1 - v) * sp]))
    return in_crop, float(blk[inside].mean()) if inside.any() else np.nan, float((blk[shell] > -400).mean()) if shell.any() else np.nan, edge_mm


def render(arr, meta, m, cands_scan, path):
    v = world_to_voxel(meta, [[m.coordX, m.coordY, m.coordZ]])[0]; sp = np.array(meta["spacing_zyx"]); shp = np.array(arr.shape)
    vi = np.clip(np.round(v).astype(int), 0, shp - 1); h = np.ceil(HALF_MM / sp).astype(int)
    cv = world_to_voxel(meta, cands_scan[["coordX", "coordY", "coordZ"]].values) if len(cands_scan) else np.zeros((0, 3))
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.6))
    views = [("axial", 0, (1, 2)), ("coronal", 1, (0, 2)), ("sagittal", 2, (0, 1))]
    for ax, (name, ax_fix, (a0, a1)) in zip(axs[:3], views):
        sl = [slice(None)] * 3; sl[ax_fix] = vi[ax_fix]
        lo0, hi0 = max(0, vi[a0] - h[a0]), min(shp[a0], vi[a0] + h[a0]); lo1, hi1 = max(0, vi[a1] - h[a1]), min(shp[a1], vi[a1] + h[a1])
        sl[a0] = slice(lo0, hi0); sl[a1] = slice(lo1, hi1); img = arr[tuple(sl)]
        ext = [lo1 * sp[a1], hi1 * sp[a1], hi0 * sp[a0], lo0 * sp[a0]]
        ax.imshow(img, cmap="gray", vmin=-1000, vmax=200, extent=ext, interpolation="bilinear")
        ax.add_patch(Circle((v[a1] * sp[a1], v[a0] * sp[a0]), m.diameter_mm / 2, fill=False, ec="#2bd96b", lw=1.4))
        for k in range(len(cv)):
            if abs(cv[k, ax_fix] - v[ax_fix]) * sp[ax_fix] < 8 and lo0 <= cv[k, a0] < hi0 and lo1 <= cv[k, a1] < hi1:
                ax.plot(cv[k, a1] * sp[a1], cv[k, a0] * sp[a0], "x", color="#ff4d4d", ms=7, mew=1.6)
                ax.text(cv[k, a1] * sp[a1] + 1.5, cv[k, a0] * sp[a0] - 1.5, f"{cands_scan.probability.values[k]:.2f}", color="#ff4d4d", fontsize=7)
        ax.set_title(name, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    # context: whole axial slice of the crop
    axs[3].imshow(arr[vi[0]], cmap="gray", vmin=-1000, vmax=200, extent=[0, shp[2] * sp[2], shp[1] * sp[1], 0])
    axs[3].add_patch(Circle((v[2] * sp[2], v[1] * sp[1]), max(m.diameter_mm, 6), fill=False, ec="#2bd96b", lw=1.2))
    axs[3].set_title("context (axial, lung crop)", fontsize=9); axs[3].set_xticks([]); axs[3].set_yticks([])
    fig.suptitle(f"miss {m.miss_id}: fold {m.fold}, {m.diameter_mm:.1f} mm, nearest cand {m.nearest_mm:.1f} mm "
                 f"(p={m.nearest_prob:.2f}), mean HU {m.mean_hu:.0f}, shell>-400HU {m.shell_dense:.0%}", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True); (OUT / "views").mkdir(exist_ok=True)
    df, cands, excl = find_misses(); index = load_index().set_index("seriesuid")
    print(f"{len(df)} nodules, {df.hit.sum()} hit, {(~df.hit).sum()} missed")
    stats = []
    for i, n in df.iterrows():
        arr, meta = load_volume(stem_for(n.seriesuid, int(n.subset)))
        stats.append(local_stats(arr, meta, [[n.coordX, n.coordY, n.coordZ]], n.diameter_mm))
    df["in_crop"], df["mean_hu"], df["shell_dense"], df["crop_edge_mm"] = zip(*stats)
    df.to_csv(OUT / "all_nodules_hit_status.csv", index=False)
    mi = df[~df.hit].sort_values("diameter_mm", ascending=False).reset_index(drop=True); mi["miss_id"] = np.arange(1, len(mi) + 1)
    for _, m in mi.iterrows():
        arr, meta = load_volume(stem_for(m.seriesuid, int(m.subset)))
        render(arr, meta, m, cands[cands.seriesuid == m.seriesuid], OUT / "views" / f"miss_{m.miss_id:02d}.png")
    mi.to_csv(OUT / "misses.csv", index=False)
    print(mi[["miss_id", "fold", "diameter_mm", "nearest_mm", "nearest_over_r", "nearest_prob", "best_prob_within_2r", "in_crop", "mean_hu",
              "shell_dense", "crop_edge_mm", "nearest_excluded_mm", "n_nodules_scan"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
