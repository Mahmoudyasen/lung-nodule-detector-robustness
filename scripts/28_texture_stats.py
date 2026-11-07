"""Noise-texture test statistics, exactly as pre-specified in analysis_plans/texture_test_prespec.md (hash in analysis/p2_texture/PRESPEC_SHA256.txt).

    python scripts/28_texture_stats.py

Systems: S3, S4 x {clean, proj0.1 (E1 folders), tex_nps, tex_gauss, tex_white (analysis/p2_texture/fold<k>)}, pooled
over folds 0-3 (356 scans). Paired bootstrap over scans, 10,000 common resamples, seed 0 (scripts/p2_stats.py), so the
clean and proj0.1 CPMs equal those of final_cpm.csv.
F4 (primary, Holm m = 2): CPM_M(tex_nps) - CPM_M(tex_gauss), M in {S3, S4}.
Equivalence (secondary): CPM_M(tex_nps) - CPM_M(proj0.1), 95 % CI inside +-0.02.
Secondary: drops from clean, tex_white - tex_nps, Delta S4 - S3 per condition, all of these by native kernel class,
candidates per scan, achieved noise levels, TCIA adjacent-slice noise correlation.
Output: analysis/p2_texture/texture_{contrasts.csv,results.json}.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2_stats as S
from lungcad import config as C
from lungcad.data.nlst_inventory import kernel_class
from e1_multifold import fold_out
from e1_dose_robustness import load_index, get_split

OUT = C.WORK / "wp0" / "analysis" / "p2_texture"; TX = C.THESIS_DATA / "luna16" / "texture"
FOLDS = [0, 1, 2, 3]; TEX = ["tex_nps", "tex_gauss", "tex_white"]; CONDS = ["clean", "proj0.1"] + TEX; MARGIN = 0.02


def contrasts(systems, uids, n_boot):
    point, boot = S.bootstrap(systems, [[u] for u in uids], n_boot=n_boot)
    R = []
    def add(kind, model, w):
        R.append({"kind": kind, "model": model, **S.contrast(point, boot, w)})
    for m in ("S3", "S4"):
        add("F4 nps-gauss", m, {f"{m}|tex_nps": 1, f"{m}|tex_gauss": -1})
        add("equiv nps-proj0.1", m, {f"{m}|tex_nps": 1, f"{m}|proj0.1": -1})
        add("white-nps", m, {f"{m}|tex_white": 1, f"{m}|tex_nps": -1})
        for c in CONDS[1:]:
            add(f"drop {c}", m, {f"{m}|{c}": 1, f"{m}|clean": -1})
    for c in CONDS:
        add(f"delta S4-S3 {c}", "S4-S3", {f"S4|{c}": 1, f"S3|{c}": -1})
    return point, pd.DataFrame(R)


def main():
    index = load_index(); uids = {k: get_split(index, k)["test"] for k in FOLDS}; allu = [u for k in FOLDS for u in uids[k]]
    e1 = pd.concat([pd.read_csv(fold_out(k) / f, dtype={"seriesuid": str}) for k in FOLDS for f in ("e1_candidates_all.csv", "e1_candidates_proj0.1.csv")])
    e1 = e1[e1.condition.isin(["clean", "proj0.1"])]
    tx = pd.concat([pd.read_csv(OUT / f"fold{k}" / f"e1_candidates_{c}.csv", dtype={"seriesuid": str}) for k in FOLDS for c in TEX])
    df = pd.concat([e1, tx], ignore_index=True)
    systems = {f"{m}|{c}": S.scan_results(df[(df.model == m) & (df.condition == c)], allu) for c in CONDS for m in ("S3", "S4")}

    point, r = contrasts(systems, allu, S.N_BOOT); r["subset"] = "all"
    ref = pd.read_csv(C.WORK / "wp0" / "analysis" / "e1_dose_pooled" / "final_cpm.csv", index_col=0)
    for k in ("S3|clean", "S4|clean", "S3|proj0.1", "S4|proj0.1"):            # known answer: same CPMs as the main analysis
        assert abs(point[k] - ref.loc[k, "cpm"]) < 1e-12, k
    f4 = r.kind == "F4 nps-gauss"; r.loc[f4, "p_holm"] = S.holm(r.loc[f4, "p_two_sided"]).values

    hdr = pd.read_csv(C.LUNA_INDEX / "lidc_headers.csv", dtype={"seriesuid": str}).drop_duplicates("seriesuid")
    kc = dict(zip(hdr.seriesuid, hdr.ConvolutionKernel.map(lambda k: kernel_class(str(k)))))
    parts = [r]
    for cl in ("soft", "sharp"):
        _, rc = contrasts(systems, [u for u in allu if kc[u] == cl], S.N_BOOT); rc["subset"] = f"native {cl}"; parts.append(rc)
    res = pd.concat(parts, ignore_index=True); res.to_csv(OUT / "texture_contrasts.csv", index=False)

    a = r.set_index(["kind", "model"])
    f4_sig = {m: bool(a.loc[("F4 nps-gauss", m), "p_holm"] < 0.05 and a.loc[("F4 nps-gauss", m), "estimate"] > 0) for m in ("S3", "S4")}
    eq = {m: bool(a.loc[("equiv nps-proj0.1", m), "ci_lo"] > -MARGIN and a.loc[("equiv nps-proj0.1", m), "ci_hi"] < MARGIN) for m in ("S3", "S4")}
    verdict = "supported" if all(f4_sig.values()) and all(eq.values()) else "partly supported" if any(f4_sig.values()) else "not supported"
    if any(a.loc[("F4 nps-gauss", m), "p_holm"] < 0.05 and a.loc[("F4 nps-gauss", m), "estimate"] < 0 for m in ("S3", "S4")):
        verdict += " (note: a significant F4 contrast in the OPPOSITE direction)"

    cps = df.groupby(["model", "condition"]).size() / len(allu)
    lv = []
    for c in TEX:
        for k in FOLDS:
            sub = index.set_index("seriesuid").subset
            for u in uids[k]:
                j = json.load(open(TX / c / f"subset{int(sub[u])}" / f"{u}.json")); lv.append({"cond": c, "uid": u, "target": j["target_sd_hu"], "achieved": j["noise_sd_added_hu"], "rel_err": j["rel_err"]})
    lv = pd.DataFrame(lv)
    nps = json.load(open(OUT / "tcia_nps.json"))
    out = {"cpm": point.round(6).to_dict(), "verdict": verdict, "f4_significant_positive": f4_sig, "equivalence_holds": eq,
           "cands_per_scan": {f"{m}|{c}": float(v) for (m, c), v in cps.items()},
           "level": {c: {"target_median": float(g.target.median()), "achieved_median": float(g.achieved.median()),
                         "max_abs_rel_err": float(g.rel_err.abs().max()), "n_outside_3pct": int((g.rel_err.abs() > 0.03).sum()), "n": len(g)} for c, g in lv.groupby("cond")},
           "tcia_z_corr": {k: {"median": v["z_corr_adjacent_median"], "iqr": v["z_corr_adjacent_iqr"]} for k, v in nps.items()}}
    json.dump(out, open(OUT / "texture_results.json", "w"), indent=1)
    pd.set_option("display.width", 200)
    print(r[["kind", "model", "estimate", "ci_lo", "ci_hi", "p_two_sided", "p_holm"]].round(4).to_string(index=False))
    print(json.dumps({k: out[k] for k in ("verdict", "f4_significant_positive", "equivalence_holds", "level", "tcia_z_corr")}, indent=1))


if __name__ == "__main__":
    main()
