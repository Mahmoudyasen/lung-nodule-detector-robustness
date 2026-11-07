"""E1 on folds 0-3 (P2 strengthening, task 1): S3 (no dose aug) vs. S4 (dose+kernel aug), pooled over the four test splits.

    python scripts/e1_multifold.py --folds 1 2 3     # inference per fold (resumable; white + correlated noise), then pooled scoring
    python scripts/e1_multifold.py --score           # pooled scoring only, from the saved per-fold candidates

Inference reuses scripts/e1_dose_robustness.py unchanged (same perturbations, same per-scan seeds), pointed at each fold's
models: S3 = fold{k}/seed0_s3_noaug (fold 0: the gate, seed0_s3_gate_noaug), S4 = fold{k}/seed0 (the frozen S5 detector).
Fold 0's candidates already exist (analysis/e1_dose_fold0) and are not rerun. Pooled scoring concatenates the four folds'
candidates and evaluates on the union of their test scans, so the paired bootstrap resamples over all ~355 scans.
Analysis only: the frozen stage-1 files are read, never written.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1_dose_robustness as E
from e1_dose_robustness import C, CONDITIONS, CONDITIONS_EXTRA, load_index, get_split, evaluate, paired_delta_cpm

POOLED = C.WORK / "wp0" / "analysis" / "e1_dose_pooled"


def fold_models(k: int) -> dict:
    s3 = "seed0_s3_gate_noaug" if k == 0 else "seed0_s3_noaug"
    return {"S3": C.STAGE1 / f"fold{k}" / s3 / "best.pt", "S4": C.STAGE1 / f"fold{k}" / "seed0" / "best.pt"}


def fold_out(k: int) -> Path:
    return C.WORK / "wp0" / "analysis" / f"e1_dose_fold{k}"


def infer_fold(k: int, index):
    E.MODELS, E.OUT = fold_models(k), fold_out(k)          # e1_dose_robustness reads both as module globals
    for p in E.MODELS.values():
        assert p.exists(), f"missing checkpoint {p}"
    uids = get_split(index, k)["test"]
    print(f"== fold {k}: {len(uids)} test scans", flush=True)
    E.run_inference(uids, index, CONDITIONS, "")
    E.run_inference(uids, index, CONDITIONS_EXTRA, "_extra")


def load_fold(k: int) -> pd.DataFrame:
    parts = [fold_out(k) / "e1_candidates_all.csv", fold_out(k) / "e1_candidates_all_extra.csv"]
    df = pd.concat([pd.read_csv(p, dtype={"seriesuid": str}) for p in parts], ignore_index=True)
    return df.assign(fold=k)


def score_pooled(folds, index):
    POOLED.mkdir(parents=True, exist_ok=True)
    df = pd.concat([load_fold(k) for k in folds], ignore_index=True)
    uids = [u for k in folds for u in get_split(index, k)["test"]]
    conds = [c for c in list(CONDITIONS) + list(CONDITIONS_EXTRA) if (df.condition == c).any()]
    res, drops = [], []
    for cname in conds:
        per = {}
        for m in ("S3", "S4"):
            c = per[m] = df[(df.model == m) & (df.condition == cname)]
            r = evaluate(c, uids, n_bootstrap=1000, tag=f"pooled_{m}_{cname}")
            res.append({"model": m, "condition": cname, "cpm": r["cpm"], "ci_lo": r["cpm_ci95"][0], "ci_hi": r["cpm_ci95"][1],
                        **{f"s@{p}": s for p, s in zip(r["fp_points"], r["sens_at_fp"])}, "detected": r["n_detected"],
                        "n_nodules": r["n_nodules"], "cands_per_scan": len(c) / len(uids), "n_fp": r["n_fp"]})
        d = paired_delta_cpm(per["S4"], per["S3"], uids, n_boot=2000)
        res.append({"model": "S4-S3", "condition": cname, "cpm": d["delta"], "ci_lo": d["delta_ci95"][0], "ci_hi": d["delta_ci95"][1],
                    "p_delta_le_0": d["p_delta_le_0"]})
        print(f"{cname:18s} S3 {res[-3]['cpm']:.3f}  S4 {res[-2]['cpm']:.3f}  Δ {d['delta']:+.3f} [{d['delta_ci95'][0]:+.3f}, {d['delta_ci95'][1]:+.3f}]", flush=True)
    for m in ("S3", "S4"):
        clean = df[(df.model == m) & (df.condition == "clean")]
        for cname in conds[1:]:
            d = paired_delta_cpm(df[(df.model == m) & (df.condition == cname)], clean, uids, n_boot=2000)
            drops.append({"model": m, "condition": cname, "delta_vs_clean": d["delta"], "ci_lo": d["delta_ci95"][0], "ci_hi": d["delta_ci95"][1]})
    tag = "".join(map(str, folds))
    pd.DataFrame(res).to_csv(POOLED / f"e1_results_folds{tag}.csv", index=False)
    pd.DataFrame(drops).to_csv(POOLED / f"e1_drop_vs_clean_folds{tag}.csv", index=False)
    print(f"{len(uids)} scans, folds {folds} -> {POOLED}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3], help="folds to run inference on (0 already exists)")
    ap.add_argument("--score", action="store_true", help="pooled scoring only")
    ap.add_argument("--pool", type=int, nargs="+", default=[0, 1, 2, 3], help="folds to pool in the scoring")
    a = ap.parse_args()
    index = load_index()
    if not a.score:
        for k in a.folds:
            infer_fold(k, index)
    score_pooled(a.pool, index)
