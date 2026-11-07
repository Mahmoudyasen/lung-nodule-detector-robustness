"""S3 vs. S4 detectors on real NLST screening scans (descriptive; NLST has no nodule coordinates).

    python scripts/18_nlst_descriptive.py --sample-only          # write the stratified sample, no GPU
    python scripts/18_nlst_descriptive.py [--folds 0] [--per-cell 100]

Sample: fixed (seed 0), stratified by vendor x kernel class (soft / sharp) x screening round, up to --per-cell series
per cell, at most one series per patient, from usable_series() (exclusions applied). Scans are the native-resolution
conversions of scripts/20_nlst_native_sample.py (NOT the blurring 1.5 mm archive). Kernel and vendor are confounded
in NLST (sharp kernels are mostly Toshiba), hence the stratification. Inference reuses e1_dose_robustness.load_model
and the same sliding-window call; candidates are saved with world coordinates. Resumable.
Output: <data>/work/wp0/analysis/p2_nlst/{sample.csv, nlst_candidates.csv, setup.json}.
"""
from __future__ import annotations
import argparse, json, pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from lungcad import config as C
from lungcad.data.nlst import usable_series, load_native
from lungcad.preprocess import voxel_to_world
from e1_multifold import fold_models
from e1_dose_robustness import load_model

OUT = C.WORK / "wp0" / "analysis" / "p2_nlst"


def make_sample(per_cell: int, seed: int = 0) -> pd.DataFrame:
    d = usable_series()
    d = d[d.kernel_class.isin(["soft", "sharp"])].sample(frac=1.0, random_state=seed)    # shuffle once, fixed
    used, rows = set(), []
    # smallest cells first, so rare cells get their patients before the big cells use them
    for key, g in sorted(d.groupby(["vendor", "kernel_class", "round"]), key=lambda kv: len(kv[1])):
        g = g[~g.pid.isin(used)].drop_duplicates("pid").head(per_cell)
        used |= set(g.pid); rows.append(g)
    return pd.concat(rows).sort_values(["vendor", "kernel_class", "round", "pid"]).reset_index(drop=True)


@torch.no_grad()
def run(sample: pd.DataFrame, folds):
    part = OUT / "rows_partial.pkl"
    st = pickle.load(open(part, "rb")) if part.exists() else {"done": [], "rows": [], "shapes": {}}
    models = {f"{m}_f{k}": load_model(p) for k in folds for m, p in fold_models(k).items()}
    json.dump({n: {"run_tag": v[2], "best_epoch": v[3]} for n, v in models.items()} | {"n_series": len(sample), "folds": folds},
              open(OUT / "setup.json", "w"), indent=1)
    done = set(st["done"]); todo = sample[~sample.uid.isin(done)]; t0 = time.time()
    print(f"{len(done)} done, {len(todo)} to go, models {list(models)}", flush=True)
    for n, r in enumerate(todo.itertuples(), 1):
        x, meta = load_native(r.stem); sp = np.array(meta["spacing_zyx"]); st["shapes"][r.uid] = list(x.shape[2:])
        for name, (det, cfg, _, _) in models.items():
            with torch.autocast("cuda", dtype=torch.float16, enabled=cfg["train"]["amp"]):
                out = det(x, use_inferer=True)[0]
            boxes = out[det.target_box_key].float().cpu().numpy(); scores = out[det.pred_score_key].float().cpu().numpy()
            if len(boxes):
                ctr = (boxes[:, :3] + boxes[:, 3:]) / 2; size = (boxes[:, 3:] - boxes[:, :3]) * sp; w = voxel_to_world(meta, ctr)
                st["rows"] += [(name, r.uid, *w[i], float(scores[i]), *size[i]) for i in range(len(boxes))]
        st["done"].append(r.uid)
        if n % 10 == 0 or n == len(todo):
            pickle.dump(st, open(str(part) + ".tmp", "wb")); Path(str(part) + ".tmp").replace(part)
            el = time.time() - t0; print(f"  {n}/{len(todo)}  {el/60:.0f} min  ETA {el/n*(len(todo)-n)/60:.0f} min", flush=True)
    cols = ["model", "uid", "coordX", "coordY", "coordZ", "probability", "dz_mm", "dy_mm", "dx_mm"]
    pd.DataFrame(st["rows"], columns=cols).to_csv(OUT / "nlst_candidates.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--folds", type=int, nargs="+", default=[0])
    ap.add_argument("--per-cell", type=int, default=100); ap.add_argument("--sample-only", action="store_true"); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); sp = OUT / "sample.csv"
    if sp.exists():
        sample = pd.read_csv(sp, dtype={"uid": str})
    else:
        sample = make_sample(a.per_cell); sample.to_csv(sp, index=False)
    print(f"sample: {len(sample)} series, {sample.pid.nunique()} patients")
    print(sample.groupby(["vendor", "kernel_class", "round"]).size().unstack("round").to_string())
    if not a.sample_only:
        run(sample, a.folds)
