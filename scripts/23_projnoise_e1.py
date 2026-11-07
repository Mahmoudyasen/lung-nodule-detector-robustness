"""P2 task 2: S3 vs. S4 on the projection-domain low-dose test scans (scripts/17_luna_projnoise.py), folds 0-3.

    python scripts/23_projnoise_e1.py [--folds 0 1 2 3] [--doses 0.5 0.25 0.1]

For each fold k and dose a, every fold-k test scan's noisy volume (projnoise/a<a>/subset<s>/<uid>) is scored by that
fold's S3 and S4 (paired: identical input). Candidates are written per fold in the E1 format with condition
'proj<a>', next to the image-domain E1 candidates, so the pooled analysis treats them like any other condition.
Resumable per (fold, dose).
"""
from __future__ import annotations
import argparse, pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from lungcad import config as C
from lungcad.preprocess import load_volume, normalize, voxel_to_world
from lungcad.data.luna16_index import load_index, get_split
from e1_multifold import fold_models, fold_out
from e1_dose_robustness import load_model

PN = C.THESIS_DATA / "luna16" / "projnoise"


@torch.no_grad()
def run(k: int, doses, index):
    models = {m: load_model(p) for m, p in fold_models(k).items()}
    sub = index.set_index("seriesuid").subset; uids = get_split(index, k)["test"]
    for a in doses:
        out = fold_out(k) / f"e1_candidates_proj{a}.csv"
        if out.exists():
            print(f"fold {k} a={a}: done"); continue
        part = fold_out(k) / f"rows_partial_proj{a}.pkl"
        st = pickle.load(open(part, "rb")) if part.exists() else {"done": [], "rows": []}
        t0 = time.time()
        for n, uid in enumerate([u for u in uids if u not in set(st["done"])], 1):
            arr, meta = load_volume(PN / f"a{a}" / f"subset{int(sub[uid])}" / uid)
            assert meta["dose_fraction"] == a
            x = torch.from_numpy(normalize(arr.astype(np.float32)))[None, None]; sp = np.array(meta["spacing_zyx"])
            for m, (det, cfg, _, _) in models.items():
                with torch.autocast("cuda", dtype=torch.float16, enabled=cfg["train"]["amp"]):
                    o = det(x, use_inferer=True)[0]
                b = o[det.target_box_key].float().cpu().numpy(); sc = o[det.pred_score_key].float().cpu().numpy()
                if len(b):
                    ctr = (b[:, :3] + b[:, 3:]) / 2; size = b[:, 3:] - b[:, :3]; w = voxel_to_world(meta, ctr)
                    st["rows"] += [(m, f"proj{a}", uid, *w[i], float(sc[i]), *(size[i] * sp)) for i in range(len(b))]
            st["done"].append(uid)
            if n % 5 == 0:
                pickle.dump(st, open(str(part) + ".tmp", "wb")); Path(str(part) + ".tmp").replace(part)
                print(f"  fold {k} a={a} {n}/{len(uids)}  {(time.time() - t0) / 60:.0f} min", flush=True)
        cols = ["model", "condition", "seriesuid", "coordX", "coordY", "coordZ", "probability", "dz_mm", "dy_mm", "dx_mm"]
        pd.DataFrame(st["rows"], columns=cols).to_csv(out, index=False); part.unlink(missing_ok=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--doses", type=float, nargs="+", default=[0.5, 0.25, 0.1]); a = ap.parse_args()
    index = load_index()
    for k in a.folds:
        run(k, a.doses, index)
