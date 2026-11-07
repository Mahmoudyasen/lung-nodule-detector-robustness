"""NLST descriptive analysis, as pre-specified (analysis plan, revision of 1 Nov 2026):
S3 and S4 of fold 0 on the fixed stratified NLST sample (1,955 series, one per patient); candidates per scan and score
distributions at the frozen threshold (0.02, max 100 per scan), by vendor x kernel class x round; S4 - S3 differences
with paired bootstrap 95 % CIs over patients (10,000 replicates, seed 0, percentile). No P values. NLST has no nodule
coordinates, so nothing here is a sensitivity.

Pre-specified statistics per stratum: mean candidates per scan, pooled candidate-score quartiles (median with CI).
Also reported, NOT pre-specified (labelled as such in the outputs): share of scans without any candidate and mean
candidates per scan with score >= 0.5. Reference row: the same detectors on the 89 clean LUNA16 fold-0 test scans.

    python scripts/p2_nlst_analysis.py            # self-test, then analysis
Output: <data>/work/wp0/analysis/p2_nlst/{per_series.csv, strata.csv, analysis.json}
"""
from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from lungcad import config as C
from p2_stats import N_BOOT

AN = C.WORK / "wp0" / "analysis"; OUT = AN / "p2_nlst"
THR, MAXC, HIGH = 0.02, 100, 0.5
MODELS = {"S3": "S3_f0", "S4": "S4_f0"}


def wmedian(x_sorted: np.ndarray, w: np.ndarray) -> float:
    """Weighted median (lower median, as np.percentile 'lower' on the expanded sample) of sorted values."""
    c = np.cumsum(w); return float(x_sorted[np.searchsorted(c, 0.5 * c[-1])])


def boot_stratum(per: pd.DataFrame, cand: dict[str, pd.DataFrame], n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """per: one row per series (uid, n_S3, n_S4, ...); cand: model -> candidates of these series.
    Common resample of series (= patients) for both models; returns point values and percentile CIs."""
    n = len(per); pos = {u: i for i, u in enumerate(per.uid)}
    rng = np.random.RandomState(seed); W = np.stack([np.bincount(rng.randint(0, n, n), minlength=n) for _ in range(n_boot)])
    out = {"n_series": n}
    def ci(v): return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
    for stat, col in (("cands_per_scan", "n"), ("high_per_scan", "nhigh"), ("share_zero", "zero")):
        vals = {m: per[f"{col}_{m}"].to_numpy(float) for m in MODELS}
        bm = {m: W @ vals[m] / n for m in MODELS}
        for m in MODELS: out[f"{stat}_{m}"] = float(vals[m].mean()); out[f"{stat}_{m}_lo"], out[f"{stat}_{m}_hi"] = ci(bm[m])
        out[f"{stat}_diff"] = out[f"{stat}_S4"] - out[f"{stat}_S3"]; out[f"{stat}_diff_lo"], out[f"{stat}_diff_hi"] = ci(bm["S4"] - bm["S3"])
    med = {}
    for m in MODELS:
        c = cand[m].sort_values("probability"); x = c.probability.to_numpy(); s = c.uid.map(pos).to_numpy()
        q = np.percentile(x, [25, 50, 75]) if len(x) else [np.nan] * 3
        out[f"score_q25_{m}"], out[f"score_q50_{m}"], out[f"score_q75_{m}"] = map(float, q)
        med[m] = np.array([wmedian(x, w[s]) if w[s].sum() else np.nan for w in W]) if len(x) else np.full(n_boot, np.nan)
        out[f"score_q50_{m}_lo"], out[f"score_q50_{m}_hi"] = ci(med[m])
    out["score_q50_diff"] = out["score_q50_S4"] - out["score_q50_S3"]; out["score_q50_diff_lo"], out["score_q50_diff_hi"] = ci(med["S4"] - med["S3"])
    return out


def per_series(sample: pd.DataFrame, cands: pd.DataFrame) -> pd.DataFrame:
    p = sample[["pid", "uid", "vendor", "kernel", "kernel_class", "round", "thk_mm"]].copy()
    for m, name in MODELS.items():
        c = cands[cands.model == name]
        p[f"n_{m}"] = p.uid.map(c.groupby("uid").size()).fillna(0).astype(int)
        p[f"nhigh_{m}"] = p.uid.map(c[c.probability >= HIGH].groupby("uid").size()).fillna(0).astype(int)
        p[f"zero_{m}"] = (p[f"n_{m}"] == 0).astype(int)
    return p


def selftest():
    rng = np.random.RandomState(1)
    # 1) known answer: S4 has exactly one more candidate on every scan, same scores shifted by +0.1 -> diff CIs collapse
    u = [f"u{i}" for i in range(50)]; per = pd.DataFrame({"uid": u, "n_S3": 3, "n_S4": 4, "nhigh_S3": 1, "nhigh_S4": 1, "zero_S3": 0, "zero_S4": 0})
    c3 = pd.DataFrame({"uid": np.repeat(u, 3), "probability": np.tile([0.1, 0.3, 0.6], 50)})
    c4 = pd.DataFrame({"uid": np.repeat(u, 4), "probability": np.tile([0.2, 0.4, 0.4, 0.7], 50)})
    r = boot_stratum(per, {"S3": c3, "S4": c4}, n_boot=500)
    assert r["cands_per_scan_diff"] == 1 and r["cands_per_scan_diff_lo"] == 1 and r["cands_per_scan_diff_hi"] == 1, r
    assert abs(r["score_q50_S3"] - 0.3) < 1e-12 and abs(r["score_q50_S4"] - 0.4) < 1e-12 and r["score_q50_diff_lo"] == r["score_q50_diff_hi"], r
    # 2) weighted median == median of the expanded sample (lower median)
    for _ in range(200):
        x = np.sort(rng.rand(rng.randint(1, 30))); w = rng.randint(0, 4, len(x))
        if w.sum() == 0: continue
        e = np.sort(np.repeat(x, w)); assert wmedian(x, w) == e[(len(e) - 1) // 2]
    # 3) coverage of the mean-difference CI on paired Poisson counts with true difference 0.5 (should be ~95 %)
    hit = 0
    for k in range(200):
        lam = rng.gamma(2, 3, 200); a = rng.poisson(lam); b = rng.poisson(lam + 0.5)
        p = pd.DataFrame({"uid": range(200), "n_S3": a, "n_S4": b, "nhigh_S3": a, "nhigh_S4": b, "zero_S3": 0, "zero_S4": 0})
        e = pd.DataFrame({"uid": [0], "probability": [0.5]})
        r = boot_stratum(p, {"S3": e, "S4": e}, n_boot=1000, seed=k); hit += r["cands_per_scan_diff_lo"] <= 0.5 <= r["cands_per_scan_diff_hi"]
    assert 0.90 <= hit / 200 <= 0.99, hit
    print(f"self-test OK (known-answer, weighted median, CI coverage {hit / 200:.3f})")


def main():
    selftest()
    sample = pd.read_csv(OUT / "sample.csv"); cands = pd.read_csv(OUT / "nlst_candidates.csv")
    raw = pickle.load(open(OUT / "rows_partial.pkl", "rb"))
    # input checks
    assert sample.uid.is_unique and sample.pid.is_unique and len(sample) == 1955
    assert set(raw["done"]) == set(sample.uid) and set(cands.uid) <= set(sample.uid)
    assert len(raw["rows"]) == len(cands) and cands.probability.min() >= THR and cands.groupby(["model", "uid"]).size().max() <= MAXC
    assert set(cands.model) == set(MODELS.values())
    per = per_series(sample, cands)
    rc = pd.DataFrame(raw["rows"]).iloc[:, :2].value_counts()           # independent count from the raw rows
    for m, name in MODELS.items():
        assert (per.uid.map(lambda u: rc.get((name, u), 0)) == per[f"n_{m}"]).all()
    per.to_csv(OUT / "per_series.csv", index=False)

    strata = [("all", "all", "all")]
    strata += [(v, "all", "all") for v in sorted(per.vendor.unique())] + [("all", k, "all") for k in ("soft", "sharp")]
    strata += [("all", "all", r) for r in sorted(per["round"].unique())]
    strata += [(v, k, r) for (v, k, r), _ in per.groupby(["vendor", "kernel_class", "round"])]
    strata += [(v, k, "all") for (v, k), _ in per.groupby(["vendor", "kernel_class"])]
    rows = []
    for v, k, r in strata:
        sel = per[((per.vendor == v) | (v == "all")) & ((per.kernel_class == k) | (k == "all")) & ((per["round"] == r) | (r == "all"))]
        cs = {m: cands[(cands.model == name) & cands.uid.isin(sel.uid)][["uid", "probability"]] for m, name in MODELS.items()}
        rows.append({"dataset": "NLST", "vendor": v, "kernel_class": k, "round": r} | boot_stratum(sel.reset_index(drop=True), cs))
        print(f"{v:8s} {k:5s} {str(r):3s} n={len(sel):4d}  cands/scan S3 {rows[-1]['cands_per_scan_S3']:.2f} S4 {rows[-1]['cands_per_scan_S4']:.2f}"
              f"  diff {rows[-1]['cands_per_scan_diff']:+.2f} [{rows[-1]['cands_per_scan_diff_lo']:+.2f}, {rows[-1]['cands_per_scan_diff_hi']:+.2f}]"
              f"  median score S3 {rows[-1]['score_q50_S3']:.3f} S4 {rows[-1]['score_q50_S4']:.3f}", flush=True)

    # reference: the same two detectors on the clean LUNA16 fold-0 test scans
    luna = pd.read_csv(AN / "e1_dose_fold0" / "e1_candidates_all.csv"); luna = luna[luna.condition == "clean"]
    uids = pickle.load(open(AN / "e1_dose_fold0" / "rows_partial.pkl", "rb"))["done"]; assert len(uids) == 89
    lp = pd.DataFrame({"uid": uids})
    for m in MODELS:
        c = luna[luna.model == m]; assert c.probability.min() >= THR and set(c.seriesuid) <= set(uids)
        lp[f"n_{m}"] = lp.uid.map(c.groupby("seriesuid").size()).fillna(0).astype(int)
        lp[f"nhigh_{m}"] = lp.uid.map(c[c.probability >= HIGH].groupby("seriesuid").size()).fillna(0).astype(int)
        lp[f"zero_{m}"] = (lp[f"n_{m}"] == 0).astype(int)
    lc = {m: luna[luna.model == m].rename(columns={"seriesuid": "uid"})[["uid", "probability"]] for m in MODELS}
    rows.append({"dataset": "LUNA16 fold 0 clean (reference)", "vendor": "all", "kernel_class": "all", "round": "all"} | boot_stratum(lp, lc))
    d = pd.DataFrame(rows); d.to_csv(OUT / "strata.csv", index=False)
    json.dump({"threshold": THR, "max_cands": MAXC, "high_score": HIGH, "n_boot": N_BOOT, "seed": 0, "n_series": len(per),
               "n_cands": cands.model.value_counts().to_dict(),
               "pre_specified": ["cands_per_scan", "score_q25/q50/q75"], "not_pre_specified": ["share_zero", "high_per_scan"]},
              open(OUT / "analysis.json", "w"), indent=1)
    r = d.iloc[-1]; print(f"LUNA16 ref n={int(r.n_series)} cands/scan S3 {r.cands_per_scan_S3:.2f} S4 {r.cands_per_scan_S4:.2f} "
                          f"median score S3 {r.score_q50_S3:.3f} S4 {r.score_q50_S4:.3f}")
    print("wrote", OUT / "strata.csv")


if __name__ == "__main__":
    main()
