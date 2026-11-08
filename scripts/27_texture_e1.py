"""Noise-texture test (lung-nodule-detector-analysis-plans/plans_original/texture_test_prespec.md): S3 vs S4 on the texture test scans
(scripts/26_luna_texture.py), folds 0-3. Same inference code as scripts/23_projnoise_e1.py.

    python scripts/27_texture_e1.py --check          # known-answer: re-score one proj0.1 scan, compare with the stored candidates
    python scripts/27_texture_e1.py [--folds 0 1 2 3] [--conds tex_nps tex_gauss tex_white]

Output: analysis/p2_texture/fold<k>/e1_candidates_<cond>.csv (E1 format). Separate from the E1 folders on purpose, so
that the existing final_* results are not touched. Resumable per (fold, condition).
Review experiments: --factor F --root R reads <R>/<cond>_x<F>/ (R2 level response) and writes analysis/p2_review/levels/;
--seed1 scores fold 0 with the seed-1 models (R4) into analysis/p2_review/seed1_texture/; --proj also scores proj0.5/0.25/0.1.
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

TX = C.THESIS_DATA / "luna16" / "texture"; PN = C.THESIS_DATA / "luna16" / "projnoise"
OUT = C.WORK / "wp0" / "analysis" / "p2_texture"
COLS = ["model", "condition", "seriesuid", "coordX", "coordY", "coordZ", "probability", "dz_mm", "dy_mm", "dx_mm"]


@torch.no_grad()
def score_volume(models, stem, cond, uid):
    arr, meta = load_volume(stem)
    x = torch.from_numpy(normalize(arr.astype(np.float32)))[None, None]; sp = np.array(meta["spacing_zyx"]); rows = []
    for m, (det, cfg, _, _) in models.items():
        with torch.autocast("cuda", dtype=torch.float16, enabled=cfg["train"]["amp"]):
            o = det(x, use_inferer=True)[0]
        b = o[det.target_box_key].float().cpu().numpy(); sc = o[det.pred_score_key].float().cpu().numpy()
        if len(b):
            ctr = (b[:, :3] + b[:, 3:]) / 2; size = b[:, 3:] - b[:, :3]; w = voxel_to_world(meta, ctr)
            rows += [(m, cond, uid, *w[i], float(sc[i]), *(size[i] * sp)) for i in range(len(b))]
    return rows, meta


def src_stem(c, factor, sub, uid):
    if c.startswith("proj"):
        return PN / f"a{c[4:]}" / f"subset{sub}" / uid
    return TX / (c if factor == 1.0 else f"{c}_x{factor:g}") / f"subset{sub}" / uid


def run(k, conds, index, models_paths=None, factor=1.0):
    models = {m: load_model(p) for m, p in (models_paths or fold_models(k)).items()}
    sub = index.set_index("seriesuid").subset; uids = get_split(index, k)["test"]; od = OUT / f"fold{k}"; od.mkdir(parents=True, exist_ok=True)
    for c0 in conds:
        c = c0 if factor == 1.0 else f"{c0}_x{factor:g}"
        out = od / f"e1_candidates_{c}.csv"
        if out.exists():
            print(f"fold {k} {c}: done"); continue
        part = od / f"rows_partial_{c}.pkl"; st = pickle.load(open(part, "rb")) if part.exists() else {"done": [], "rows": []}; t0 = time.time()
        for n, uid in enumerate([u for u in uids if u not in set(st["done"])], 1):
            rows, meta = score_volume(models, src_stem(c0, factor, int(sub[uid]), uid), c, uid)
            assert c0.startswith("proj") or (meta["condition"] == c0 and float(meta.get("level_factor", 1.0)) == factor)
            st["rows"] += rows; st["done"].append(uid)
            if n % 5 == 0:
                pickle.dump(st, open(str(part) + ".tmp", "wb")); Path(str(part) + ".tmp").replace(part)
                print(f"  fold {k} {c} {n}/{len(uids)}  {(time.time() - t0) / 60:.0f} min", flush=True)
        assert len(st["done"]) == len(uids)
        pd.DataFrame(st["rows"], columns=COLS).to_csv(out, index=False); part.unlink(missing_ok=True)


def check(index):
    k = 0; uid = get_split(index, k)["test"][0]; sub = int(index.set_index("seriesuid").subset[uid])
    models = {m: load_model(p) for m, p in fold_models(k).items()}
    rows, _ = score_volume(models, PN / "a0.1" / f"subset{sub}" / uid, "proj0.1", uid)
    new = pd.DataFrame(rows, columns=COLS).sort_values(["model", "probability"]).reset_index(drop=True)
    old = pd.read_csv(fold_out(k) / "e1_candidates_proj0.1.csv", dtype={"seriesuid": str})
    old = old[old.seriesuid == uid].sort_values(["model", "probability"]).reset_index(drop=True)
    same = len(new) == len(old) and (new.model == old.model).all() and np.allclose(new[COLS[3:]].values, old[COLS[3:]].values, atol=1e-5)
    print(f"check: {len(new)} new vs {len(old)} stored candidates for {uid[-12:]}: {'IDENTICAL' if same else 'DIFFERENT'}"
          + ("" if same else f" max diff {np.abs(new[COLS[3:]].values - old[COLS[3:]].values).max() if len(new) == len(old) else 'n/a'}"))
    return same


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--conds", nargs="+", default=["tex_nps", "tex_gauss", "tex_white"]); ap.add_argument("--check", action="store_true")
    ap.add_argument("--factor", type=float, default=1.0); ap.add_argument("--root"); ap.add_argument("--seed1", action="store_true")
    a = ap.parse_args()
    index = load_index()
    if a.check:
        sys.exit(0 if check(index) else 1)
    if a.root:
        TX = Path(a.root)
    if a.factor != 1.0:
        OUT = C.WORK / "wp0" / "analysis" / "p2_review" / "levels"
    if a.seed1:
        OUT = C.WORK / "wp0" / "analysis" / "p2_review" / "seed1_texture"
        sm = {"S3": C.STAGE1 / "fold0" / "seed1_s3_noaug" / "best.pt", "S4": C.STAGE1 / "fold0" / "seed1_s4_aug" / "best.pt"}
        run(0, a.conds, index, sm, a.factor); sys.exit(0)
    for k in a.folds:
        run(k, a.conds, index, None, a.factor)
