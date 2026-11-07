"""P2 secondary analyses added at review (2 Nov 2026, after the main results; labelled as such in the paper unless pre-specified).

    python scripts/29_p2_secondary.py [--skip-levels]

1. By native kernel class and vendor of the LUNA16 scan (PRE-SPECIFIED secondary, plan of 30 Oct: "results by vendor
   and kernel of the original LUNA16 scan"; omitted from draft v1): CPM S3/S4, Delta S4-S3 and each model's drop from
   clean, paired bootstrap 10,000 (seed 0) within the subset.
2. Difference in degradation from clean, DD = [CPM_S4(c) - CPM_S4(clean)] - [CPM_S3(c) - CPM_S3(clean)], from the
   stored common-resample bootstrap of the main analysis (final_boot.parquet), with per-fold values and Holm within
   F1/F2 as a SENSITIVITY analysis (not pre-specified).
3. Actual added-noise SD over the preprocessed lung crop for the image-domain noise conditions (nominal SD refers to
   the top of the HU window), all 356 scans, same measure as the projection noise (SD of perturbed - clean).
Output: analysis/p2_secondary/{by_subset.csv, dd_sensitivity.csv, noise_levels.csv}.
"""
from __future__ import annotations
import argparse, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2_stats as S
from lungcad import config as C
from lungcad.data.nlst_inventory import kernel_class
from e1_multifold import fold_out, POOLED
from e1_dose_robustness import load_index, get_split

OUT = C.WORK / "wp0" / "analysis" / "p2_secondary"; FOLDS = [0, 1, 2, 3]
SUBCONDS = ["clean", "soft1.0", "soft1.5", "soft1.0+noise60", "cnoise80_c2", "proj0.1"]
F1 = ["soft1.0", "soft1.5", "soft1.0+noise60"]; F2 = ["cnoise40_c1", "cnoise80_c1", "cnoise40_c2", "cnoise80_c2", "proj0.5", "proj0.25", "proj0.1"]


def by_subset(index, allu):
    df = pd.concat([pd.read_csv(p, dtype={"seriesuid": str}) for k in FOLDS for p in sorted(fold_out(k).glob("e1_candidates_*.csv"))], ignore_index=True)
    df = df[df.condition.isin(SUBCONDS)]
    systems = {f"{m}|{c}": S.scan_results(df[(df.model == m) & (df.condition == c)], allu) for c in SUBCONDS for m in ("S3", "S4")}
    hdr = pd.read_csv(C.LUNA_INDEX / "lidc_headers.csv", dtype={"seriesuid": str}).drop_duplicates("seriesuid").set_index("seriesuid")
    kc = {u: kernel_class(str(hdr.loc[u, "ConvolutionKernel"])) for u in allu}
    vd = {u: str(hdr.loc[u, "Manufacturer"]).split()[0].upper() for u in allu}
    subsets = {f"kernel {c}": [u for u in allu if kc[u] == c] for c in ("soft", "sharp")}
    subsets |= {f"vendor {v}": [u for u in allu if vd[u] == v] for v in sorted(set(vd.values()))}
    R = []
    for name, us in subsets.items():
        point, boot = S.bootstrap(systems, [[u] for u in us])
        for c in SUBCONDS:
            d = S.contrast(point, boot, {f"S4|{c}": 1, f"S3|{c}": -1})
            row = {"subset": name, "n_scans": len(us), "condition": c, "cpm_S3": point[f"S3|{c}"], "cpm_S4": point[f"S4|{c}"],
                   "delta": d["estimate"], "delta_lo": d["ci_lo"], "delta_hi": d["ci_hi"], "delta_p": d["p_two_sided"]}
            if c != "clean":
                for m in ("S3", "S4"):
                    q = S.contrast(point, boot, {f"{m}|{c}": 1, f"{m}|clean": -1}); row |= {f"drop_{m}": q["estimate"], f"drop_{m}_lo": q["ci_lo"], f"drop_{m}_hi": q["ci_hi"]}
            R.append(row)
        print(f"{name}: {len(us)} scans done", flush=True)
    return pd.DataFrame(R)


def dd_sensitivity():
    b = pd.read_parquet(POOLED / "final_boot.parquet"); d = pd.read_csv(POOLED / "final_delta.csv")
    pf = pd.read_csv(POOLED / "final_per_fold.csv").pivot(index="condition", columns="fold", values="delta")
    pt = {f"{m}|{r.condition}": getattr(r, f"cpm_{m}") for r in d.itertuples() for m in ("S3", "S4")}
    R = []
    for c in d.condition:
        if c == "clean":
            continue
        est = (pt[f"S4|{c}"] - pt["S4|clean"]) - (pt[f"S3|{c}"] - pt["S3|clean"])
        x = ((b[f"S4|{c}"] - b["S4|clean"]) - (b[f"S3|{c}"] - b["S3|clean"])).values
        p = max(2 / len(x), min(1.0, 2 * min((x <= 0).mean(), (x >= 0).mean())))
        R.append({"condition": c, "family": "F1" if c in F1 else "F2" if c in F2 else "secondary", "dd": est, "lo": float(np.percentile(x, 2.5)),
                  "hi": float(np.percentile(x, 97.5)), "p": p, **{f"fold{k}": float(pf.loc[c, k] - pf.loc["clean", k]) for k in FOLDS}})
    r = pd.DataFrame(R); r["p_holm"] = np.nan
    for fam in ("F1", "F2"):
        m = r.family == fam; r.loc[m, "p_holm"] = S.holm(r.loc[m, "p"]).values
    return r


def noise_levels(index, allu):
    import zlib
    from e1_dose_robustness import perturb, CONDITIONS, CONDITIONS_EXTRA
    from lungcad.stage1.dataset import load_full_volume
    sub = index.set_index("seriesuid").subset
    # exactly the E1 seeds: crc32(uid) * 100 + condition index (+ 50 for the extra conditions), e1_dose_robustness.run_inference
    conds = {c: (v[2], 0.0, ci) for ci, (c, v) in enumerate(CONDITIONS.items()) if v[0] is None and v[2] > 0}
    conds |= {c: (v[2], v[3], ci + 50) for ci, (c, v) in enumerate(CONDITIONS_EXTRA.items())}
    R = []
    for u in allu:
        x, _ = load_full_volume(u, int(sub[u])); x = x.cuda()
        for c, (s, corr, off) in conds.items():
            y = perturb(x, None, 0.0, s, seed=zlib.crc32(u.encode()) * 100 + off, corr=corr)
            R.append({"uid": u, "condition": c, "nominal_hu": s, "added_sd_hu": float((y - x).std()) * 1400})
    return pd.DataFrame(R)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--skip-levels", action="store_true"); ap.add_argument("--only", choices=["subset", "dd", "levels"]); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); index = load_index(); allu = [u for k in FOLDS for u in get_split(index, k)["test"]]
    if a.only in (None, "dd"):
        r = dd_sensitivity(); r.to_csv(OUT / "dd_sensitivity.csv", index=False); print(r.round(4).to_string(index=False))
    if a.only in (None, "levels") and not a.skip_levels:
        lv = noise_levels(index, allu); lv.to_csv(OUT / "noise_levels.csv", index=False)
        print(lv.groupby("condition").added_sd_hu.median().round(1).to_string())
    if a.only in (None, "subset"):
        bs = by_subset(index, allu); bs.to_csv(OUT / "by_subset.csv", index=False)
        print(bs[["subset", "n_scans", "condition", "cpm_S3", "cpm_S4", "delta", "delta_lo", "delta_hi"]].round(3).to_string(index=False))
