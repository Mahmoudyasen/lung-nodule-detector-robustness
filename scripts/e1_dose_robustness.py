"""E1 — simulated low-dose / kernel robustness of the fold-0 detectors: S3 (no dose aug) vs. S4 (dose+kernel aug).

    python scripts/e1_dose_robustness.py            # inference (resumable), then scoring
    python scripts/e1_dose_robustness.py --score    # scoring only, from the saved candidates

Each fold-0 test scan is loaded once; every perturbation is applied with a fixed per-(scan, condition) seed, so
both models see byte-identical perturbed volumes and the S4-vs-S3 comparison is paired. The perturbations use
the same image-domain model as lungcad.augment (Poisson-Gaussian noise with a_frac 0.5; Gaussian blur = softer
kernel; unsharp mask = sharper kernel), applied deterministically at fixed strengths. Training ranges were
noise 10-60 HU, blur 0.3-1.0 vox, sharpen 0.3-1.0; points beyond them test extrapolation.
Analysis only: the frozen stage-1 files are read, never written.
"""
from __future__ import annotations
import argparse, json, pickle, sys, time, zlib
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lungcad import config as C
from lungcad.augment import HU_RANGE, separable_blur
from lungcad.data.luna16_index import load_index, get_split
from lungcad.preprocess import voxel_to_world
from lungcad.stage1.dataset import load_full_volume
from lungcad.stage1.model import build_detector, set_inferer
from lungcad.eval.froc import evaluate
from lungcad.eval.bootstrap import paired_delta_cpm

MODELS = {"S3": C.STAGE1 / "fold0" / "seed0_s3_gate_noaug" / "best.pt",
          "S4": C.STAGE1 / "fold0" / "seed0" / "best.pt"}
# name -> (kernel, kernel strength, noise sigma in HU); kernel applied first, then noise (same order as training)
CONDITIONS = {
    "clean":       (None, 0.0, 0),
    "noise20":     (None, 0.0, 20), "noise40": (None, 0.0, 40), "noise60": (None, 0.0, 60),
    "noise80":     (None, 0.0, 80), "noise100": (None, 0.0, 100),
    "soft0.5":     ("blur", 0.5, 0), "soft1.0": ("blur", 1.0, 0), "soft1.5": ("blur", 1.5, 0),
    "sharp0.5":    ("sharp", 0.5, 0), "sharp1.0": ("sharp", 1.0, 0), "sharp1.5": ("sharp", 1.5, 0),
    "sharp1.0+noise60": ("sharp", 1.0, 60), "soft1.0+noise60": ("blur", 1.0, 60),
}
# added after the first run showed both models ignore white (voxel-independent) noise: spatially correlated noise,
# white noise low-passed with a Gaussian of `corr` voxels and rescaled to the same per-voxel SD (closer to real CT noise texture)
CONDITIONS_EXTRA = {"cnoise40_c1": (None, 0.0, 40, 1.0), "cnoise80_c1": (None, 0.0, 80, 1.0),
                    "cnoise40_c2": (None, 0.0, 40, 2.0), "cnoise80_c2": (None, 0.0, 80, 2.0)}
OUT = C.WORK / "wp0" / "analysis" / "e1_dose_fold0"


@torch.no_grad()
def perturb(x: torch.Tensor, kernel, strength: float, sigma_hu: float, seed: int, corr: float = 0.0) -> torch.Tensor:
    """Deterministic version of lungcad.augment.dose_kernel_augment for one volume x (1,1,D,H,W) in [0,1]."""
    if kernel == "blur":
        x = separable_blur(x, strength)
    elif kernel == "sharp":
        x = x + strength * (x - separable_blur(x, 0.8))
    if sigma_hu > 0:
        s = sigma_hu / HU_RANGE; var = s * s; a = 0.5 * var; b = 0.5 * var
        g = torch.Generator(device=x.device).manual_seed(seed)
        n = torch.randn(x.shape, generator=g, device=x.device, dtype=x.dtype)
        if corr > 0:
            n = separable_blur(n, corr); n = n / n.std()
        x = x + n * torch.sqrt(a * x.clamp(0, 1) + b)
    return x.clamp(0, 1)


def load_model(path: Path):
    ck = torch.load(path, map_location="cuda", weights_only=False); cfg = ck["cfg"]
    det = build_detector(cfg["model"]); det.network.load_state_dict(ck["model"])
    set_inferer(det, cfg["infer"]["roi_size"], cfg["infer"]["overlap"], cfg["infer"]["sw_batch_size"])
    det.eval()
    return det, cfg, ck.get("run_tag"), ck.get("state", {}).get("best_epoch")


@torch.no_grad()
def run_inference(uids, index, conditions=CONDITIONS, suffix=""):
    OUT.mkdir(parents=True, exist_ok=True); part = OUT / f"rows_partial{suffix}.pkl"
    rows = pickle.load(open(part, "rb")) if part.exists() else {"done": [], "rows": []}
    models = {k: load_model(p) for k, p in MODELS.items()}
    json.dump({k: {"checkpoint": str(MODELS[k]), "run_tag": v[2], "best_epoch": v[3]} for k, v in models.items()} |
              {"conditions": {k: list(v) for k, v in conditions.items()}, "n_scans": len(uids)},
              open(OUT / f"e1_setup{suffix}.json", "w"), indent=1)
    idx = index.set_index("seriesuid"); t0 = time.time(); todo = [u for u in uids if u not in set(rows["done"])]
    print(f"{len(uids) - len(todo)} scans already done, {len(todo)} to go", flush=True)
    for n, uid in enumerate(todo):
        img, meta = load_full_volume(uid, int(idx.loc[uid, "subset"])); img = img.cuda(); sp = np.array(meta["spacing_zyx"])
        for ci, (cname, (kern, st, sig, *cr)) in enumerate(conditions.items()):
            x = perturb(img, kern, st, sig, seed=zlib.crc32(uid.encode()) * 100 + ci + (50 if suffix else 0), corr=cr[0] if cr else 0.0).cpu()   # inferer stitches on CPU, like infer.py
            for mname, (det, cfg, _, _) in models.items():
                with torch.autocast("cuda", dtype=torch.float16, enabled=cfg["train"]["amp"]):
                    out = det(x, use_inferer=True)[0]
                boxes = out[det.target_box_key].float().cpu().numpy(); scores = out[det.pred_score_key].float().cpu().numpy()
                if len(boxes):
                    ctr = (boxes[:, :3] + boxes[:, 3:]) / 2; size = boxes[:, 3:] - boxes[:, :3]; w = voxel_to_world(meta, ctr)
                    for i in range(len(boxes)):
                        rows["rows"].append((mname, cname, uid, w[i, 0], w[i, 1], w[i, 2], float(scores[i]), *(size[i] * sp)))
            del x
        rows["done"].append(uid)
        if (n + 1) % 5 == 0 or n + 1 == len(todo):
            pickle.dump(rows, open(str(part) + ".tmp", "wb")); Path(str(part) + ".tmp").replace(part)
            el = time.time() - t0
            print(f"  {n+1}/{len(todo)} scans  {el/60:.1f} min  ETA {el/(n+1)*(len(todo)-n-1)/60:.0f} min", flush=True)
    df = pd.DataFrame(rows["rows"], columns=["model", "condition", "seriesuid", "coordX", "coordY", "coordZ", "probability", "dz_mm", "dy_mm", "dx_mm"])
    df.to_csv(OUT / f"e1_candidates_all{suffix}.csv", index=False)
    return df


def score(uids):
    df = pd.read_csv(OUT / "e1_candidates_all.csv", dtype={"seriesuid": str}); res = []
    if (OUT / "e1_candidates_all_extra.csv").exists():
        df = pd.concat([df, pd.read_csv(OUT / "e1_candidates_all_extra.csv", dtype={"seriesuid": str})], ignore_index=True)
    conds = [c for c in list(CONDITIONS) + list(CONDITIONS_EXTRA) if (df.condition == c).any()]
    for cname in conds:
        per = {}
        for m in MODELS:
            c = df[(df.model == m) & (df.condition == cname)]
            r = evaluate(c, uids, n_bootstrap=1000, tag=f"{m}_{cname}")
            per[m] = c
            res.append({"model": m, "condition": cname, "cpm": r["cpm"], "ci_lo": r["cpm_ci95"][0], "ci_hi": r["cpm_ci95"][1],
                        **{f"s@{p}": s for p, s in zip(r["fp_points"], r["sens_at_fp"])}, "detected": r["n_detected"],
                        "n_nodules": r["n_nodules"], "cands_per_scan": len(c) / len(uids), "n_fp": r["n_fp"]})
        d = paired_delta_cpm(per["S4"], per["S3"], uids, n_boot=2000)
        res.append({"model": "S4-S3", "condition": cname, "cpm": d["delta"], "ci_lo": d["delta_ci95"][0], "ci_hi": d["delta_ci95"][1],
                    "p_delta_le_0": d["p_delta_le_0"]})
        print(f"{cname:18s} S3 {res[-3]['cpm']:.3f}  S4 {res[-2]['cpm']:.3f}  Δ {d['delta']:+.3f} [{d['delta_ci95'][0]:+.3f}, {d['delta_ci95'][1]:+.3f}]", flush=True)
    # paired drop from clean, per model (does each model lose significantly under a condition?)
    out = pd.DataFrame(res); drops = []
    for m in MODELS:
        clean = df[(df.model == m) & (df.condition == "clean")]
        for cname in conds:
            if cname == "clean":
                continue
            d = paired_delta_cpm(df[(df.model == m) & (df.condition == cname)], clean, uids, n_boot=2000)
            drops.append({"model": m, "condition": cname, "delta_vs_clean": d["delta"], "ci_lo": d["delta_ci95"][0], "ci_hi": d["delta_ci95"][1]})
    out.to_csv(OUT / "e1_results.csv", index=False); pd.DataFrame(drops).to_csv(OUT / "e1_drop_vs_clean.csv", index=False)
    print(pd.DataFrame(drops).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--score", action="store_true"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--extra", action="store_true", help="run only the correlated-noise conditions (CONDITIONS_EXTRA)")
    a = ap.parse_args()
    index = load_index(); uids = get_split(index, 0)["test"]
    if a.limit:
        uids = uids[:a.limit]; OUT = OUT.parent / "e1_smoke"
    if not a.score:
        run_inference(uids, index, *((CONDITIONS_EXTRA, "_extra") if a.extra else (CONDITIONS, "")))
    score(uids)
