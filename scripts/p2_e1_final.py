"""Final E1 analysis over folds 0-3 (families F1 and F2), exactly as pre-specified on 30 Oct 2026 (analysis plan for F1-F3;
revised the same day, before any new result, and on 1 Nov, when the projection-domain conditions were added to F2).

    python scripts/p2_e1_final.py

Inputs per fold k (analysis/e1_dose_fold<k>/): e1_candidates_all.csv, e1_candidates_all_extra.csv (image domain). Each fold's models score only that fold's test scans, so S3 / S4
candidates of the four folds concatenate into one pooled system per condition (356 scans).
Bootstrap over scans, 10,000 replicates, all systems on common resamples (scripts/p2_stats.py).
Output (analysis/e1_dose_pooled/): final_delta.csv (ΔCPM S4−S3 per condition, family, raw and Holm P),
final_drop.csv (each model's drop from clean), final_per_fold.csv (ΔCPM per fold, sign consistency), final_cpm.csv.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2_stats as S
from e1_multifold import fold_out, POOLED
from e1_dose_robustness import load_index, get_split

FOLDS = [0, 1, 2, 3]
F1 = ["soft1.0", "soft1.5", "soft1.0+noise60"]
CNOISE = ["cnoise40_c1", "cnoise80_c1", "cnoise40_c2", "cnoise80_c2"]
PROJ = ["proj0.5", "proj0.25", "proj0.1"]           # projection-domain noise (scripts/23_projnoise_e1.py); in F2 since 1 Nov
F2 = CNOISE + PROJ                                  # 7 comparisons, Holm within the family


def main():
    index = load_index(); uids = {k: get_split(index, k)["test"] for k in FOLDS}; allu = [u for k in FOLDS for u in uids[k]]
    df = pd.concat([pd.read_csv(p, dtype={"seriesuid": str}) for k in FOLDS for p in sorted(fold_out(k).glob("e1_candidates_*.csv"))],
                   ignore_index=True)
    conds = list(dict.fromkeys(df.condition))
    for c in F1 + CNOISE + ["clean"]:
        assert c in conds, f"condition {c} missing: run its inference first"
    f2_missing = [c for c in F2 if c not in conds]
    tag = "_provisional" if f2_missing else ""          # F2 Holm is final only with all 7 members
    if f2_missing:
        print(f"F2 incomplete (missing {f2_missing}): Holm over {len(F2) - len(f2_missing)} of {len(F2)}; outputs marked {tag}", flush=True)
    systems = {f"{m}|{c}": S.scan_results(df[(df.model == m) & (df.condition == c)], allu) for c in conds for m in ("S3", "S4")}
    point, boot = S.bootstrap(systems, [[u] for u in allu])
    boot.to_parquet(POOLED / f"final_boot{tag}.parquet")
    d = pd.DataFrame([{"condition": c, "family": "F1" if c in F1 else "F2" if c in F2 else "secondary",
                       "cpm_S3": point[f"S3|{c}"], "cpm_S4": point[f"S4|{c}"], **S.contrast(point, boot, {f"S4|{c}": 1, f"S3|{c}": -1})}
                      for c in conds])
    d["p_holm"] = float("nan")
    for fam in ("F1", "F2"):
        m = d.family == fam; d.loc[m, "p_holm"] = S.holm(d.loc[m, "p_two_sided"]).values; d.loc[m, "holm_m"] = int(m.sum())
    d.to_csv(POOLED / f"final_delta{tag}.csv", index=False)
    drops = pd.DataFrame([{"model": m, "condition": c, **S.contrast(point, boot, {f"{m}|{c}": 1, f"{m}|clean": -1})}
                          for m in ("S3", "S4") for c in conds if c != "clean"])
    drops.to_csv(POOLED / f"final_drop{tag}.csv", index=False)
    pf = []
    for k in FOLDS:                                      # per fold, point estimates only (secondary: sign consistency)
        for c in conds:
            pf.append({"fold": k, "condition": c, "delta": S.cpm(systems[f"S4|{c}"], uids[k]) - S.cpm(systems[f"S3|{c}"], uids[k])})
    pf = pd.DataFrame(pf); pf.to_csv(POOLED / f"final_per_fold{tag}.csv", index=False)
    q = boot.quantile([0.025, 0.975]).T; q.columns = ["ci_lo", "ci_hi"]; pd.concat([point.rename("cpm"), q], axis=1).to_csv(POOLED / f"final_cpm{tag}.csv")
    cons = pf.groupby("condition").delta.apply(lambda x: int((x > 0).sum()))
    print(d.assign(folds_S4_better=d.condition.map(cons)).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
