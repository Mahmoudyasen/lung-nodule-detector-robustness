"""P2 review statistics, exactly as pre-specified in docs/review_prespec.md (hash in analysis/p2_review/PRESPEC_SHA256.txt).

    python scripts/32_review_stats.py inplane     # R1, family F5
    python scripts/32_review_stats.py levels      # R2, family F6
    python scripts/32_review_stats.py nlst        # R3, family F7 + kernel calibration summary
    python scripts/32_review_stats.py seed        # R4, seed-1 pair on fold 0 + sensitivity of F1-F3 (fold-0 S3 -> seed 1)

Same helpers as the main analysis (scripts/p2_stats.py: paired bootstrap, 10,000 common resamples, seed 0, two-sided P,
Holm). Known-answer checks: CPMs of conditions that also exist in the main analysis must equal final_cpm.csv.
Output: analysis/p2_review/<step>_*.csv / .json.
"""
from __future__ import annotations
import argparse, os, json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import p2_stats as S
from lungcad import config as C
from e1_multifold import fold_out
from e1_dose_robustness import load_index, get_split

AN = C.WORK / "wp0" / "analysis"; OUT = AN / "p2_review"; FOLDS = [0, 1, 2, 3]
REF = pd.read_csv(AN / "e1_dose_pooled" / "final_cpm.csv", index_col=0)
TEX = ["tex_nps", "tex_gauss", "tex_white"]


def read(p):
    return pd.read_csv(p, dtype={"seriesuid": str})


def check_ref(point, keys):
    for k in keys:
        assert abs(point[k] - REF.loc[k, "cpm"]) < 1e-12, f"{k}: {point[k]} vs final_cpm {REF.loc[k, 'cpm']}"
    print(f"known answer: {len(keys)} CPMs equal final_cpm.csv")


def boot_contrasts(systems, uids, spec):
    point, boot = S.bootstrap(systems, [[u] for u in uids])
    rows = [{"kind": kind, "model": model, **S.contrast(point, boot, w)} for kind, model, w in spec]
    return point, boot, pd.DataFrame(rows)


# ------------------------------------------------------------------------------------------------------------ R1
def inplane():
    index = load_index(); allu = [u for k in FOLDS for u in get_split(index, k)["test"]]
    e1 = pd.concat([read(fold_out(k) / "e1_candidates_all.csv") for k in FOLDS]); e1 = e1[e1.condition.isin(["clean", "soft1.0", "soft1.5"])]
    xy = pd.concat([read(p) for k in FOLDS for p in sorted((OUT / "inplane" / f"fold{k}").glob("e1_candidates_all_xy*.csv"))])
    df = pd.concat([e1, xy]); conds = ["clean", "soft1.0", "soft1.5"] + [c for c in ("xy1.0", "xy1.5", "xycal") if (df.condition == c).any()]
    for c in conds[3:]:
        assert df[df.condition == c].seriesuid.nunique() <= len(allu)
    systems = {f"{m}|{c}": S.scan_results(df[(df.model == m) & (df.condition == c)], allu) for c in conds for m in ("S3", "S4")}
    spec = [(f"delta {c}", "S4-S3", {f"S4|{c}": 1, f"S3|{c}": -1}) for c in conds]
    spec += [(f"drop {c}", m, {f"{m}|{c}": 1, f"{m}|clean": -1}) for m in ("S3", "S4") for c in conds[1:]]
    point, boot, r = boot_contrasts(systems, allu, spec)
    check_ref(point, [f"{m}|{c}" for m in ("S3", "S4") for c in ("clean", "soft1.0", "soft1.5")])
    f5 = r.kind.isin(["delta xy1.0", "delta xy1.5"]); r["family"] = np.where(f5, "F5", "secondary")
    r.loc[f5, "p_holm"] = S.holm(r.loc[f5, "p_two_sided"]).values
    ratio = []                                       # loss under in-plane blur / loss under 3D blur of the same sigma (voxels)
    for m in ("S3", "S4"):
        for xyc, c3 in (("xy1.0", "soft1.0"), ("xy1.5", "soft1.5")):
            num = boot[f"{m}|clean"] - boot[f"{m}|{xyc}"]; den = boot[f"{m}|clean"] - boot[f"{m}|{c3}"]; q = (num / den).values
            ratio.append({"model": m, "inplane": xyc, "blur3d": c3, "ratio": float((point[f"{m}|clean"] - point[f"{m}|{xyc}"]) / (point[f"{m}|clean"] - point[f"{m}|{c3}"])),
                          "ci_lo": float(np.percentile(q, 2.5)), "ci_hi": float(np.percentile(q, 97.5))})
    cal = OUT / "nlst_pairs" / "kernel_calibration.json"
    out = {"cpm": point.to_dict(), "loss_ratio": ratio, "xycal_mm": json.load(open(cal))["xycal_mm"] if cal.exists() else None,
           "cands_per_scan": {f"{m}|{c}": float(((df.model == m) & (df.condition == c)).sum() / len(allu)) for m in ("S3", "S4") for c in conds}}
    r.to_csv(OUT / "inplane_contrasts.csv", index=False); json.dump(out, open(OUT / "inplane_results.json", "w"), indent=1)
    print(r.round(4).to_string(index=False)); print(pd.DataFrame(ratio).round(3).to_string(index=False))


# ------------------------------------------------------------------------------------------------------------ R2
def levels():
    index = load_index(); allu = [u for k in FOLDS for u in get_split(index, k)["test"]]
    e1 = pd.concat([read(fold_out(k) / "e1_candidates_all.csv") for k in FOLDS]); e1 = e1[e1.condition == "clean"]
    t1 = pd.concat([read(AN / "p2_texture" / f"fold{k}" / f"e1_candidates_{c}.csv") for k in FOLDS for c in TEX])
    t1 = t1.assign(condition=t1.condition + "_x1")
    tl = pd.concat([read(OUT / "levels" / f"fold{k}" / f"e1_candidates_{c}_x{f:g}.csv") for k in FOLDS for c in TEX for f in (0.5, 2.0)])
    df = pd.concat([e1, t1, tl]); conds = ["clean"] + [f"{c}_x{f:g}" for c in TEX for f in (0.5, 1, 2)]
    systems = {f"{m}|{c}": S.scan_results(df[(df.model == m) & (df.condition == c)], allu) for c in conds for m in ("S3", "S4")}
    spec = [("F6 nps_x2 - gauss_x0.5", m, {f"{m}|tex_nps_x2": 1, f"{m}|tex_gauss_x0.5": -1}) for m in ("S3", "S4")]
    spec += [("white_x2 - gauss_x0.5", m, {f"{m}|tex_white_x2": 1, f"{m}|tex_gauss_x0.5": -1}) for m in ("S3", "S4")]
    spec += [(f"drop {c}", m, {f"{m}|{c}": 1, f"{m}|clean": -1}) for m in ("S3", "S4") for c in conds[1:]]
    spec += [(f"delta {c}", "S4-S3", {f"S4|{c}": 1, f"S3|{c}": -1}) for c in conds]
    point, boot, r = boot_contrasts(systems, allu, spec)
    check_ref(point, ["S3|clean", "S4|clean"])
    tref = pd.read_csv(AN / "p2_texture" / "texture_contrasts.csv"); tref = tref[(tref.subset == "all") & tref.kind.str.startswith("drop tex")]
    for t in tref.itertuples():                       # known answer: factor-1 drops = the texture test's
        assert abs(point[f"{t.model}|{t.kind[5:]}_x1"] - point[f"{t.model}|clean"] - t.estimate) < 1e-12
    print("known answer: factor-1 texture drops equal texture_contrasts.csv")
    f6 = r.kind == "F6 nps_x2 - gauss_x0.5"; r["family"] = np.where(f6, "F6", "secondary"); r.loc[f6, "p_holm"] = S.holm(r.loc[f6, "p_two_sided"]).values
    sig = {m: bool(r[f6 & (r.model == m)].p_holm.iloc[0] < 0.05 and r[f6 & (r.model == m)].estimate.iloc[0] > 0) for m in ("S3", "S4")}
    wording = ("spectrum mattered more than a four-fold change in level" if all(sig.values()) else
               "keep 'at equal level'; a four-fold level change partly offset the spectrum" if any(sig.values()) else
               "keep 'at equal level'; a four-fold level change offset the spectrum")
    lv = []
    sub = index.set_index("seriesuid").subset; LEV = Path(os.environ.get("LUNGCAD_NLST_HDD", "data/raw"), "p2_review", "texture_levels")
    for c in TEX:
        for f in (0.5, 2.0):
            for u in allu:
                j = json.load(open(LEV / f"{c}_x{f:g}" / f"subset{int(sub[u])}" / f"{u}.json"))
                lv.append({"cond": f"{c}_x{f:g}", "target": j["target_sd_hu"], "achieved": j["noise_sd_added_hu"], "rel_err": j["rel_err"]})
    lv = pd.DataFrame(lv)
    out = {"cpm": point.to_dict(), "f6_significant_positive": sig, "wording": wording,
           "level": {c: {"target_median": float(g.target.median()), "achieved_median": float(g.achieved.median()), "max_abs_rel_err": float(g.rel_err.abs().max()),
                         "n_outside_3pct": int((g.rel_err.abs() > 0.03).sum()), "n": len(g)} for c, g in lv.groupby("cond")},
           "cands_per_scan": {f"{m}|{c}": float(((df.model == m) & (df.condition == c)).sum() / len(allu)) for m in ("S3", "S4") for c in conds}}
    r.to_csv(OUT / "levels_contrasts.csv", index=False); json.dump(out, open(OUT / "levels_results.json", "w"), indent=1)
    print(r.round(4).to_string(index=False)); print(json.dumps({k: out[k] for k in ("f6_significant_positive", "wording", "level")}, indent=1))
    # NOT pre-specified (added 3 Nov after seeing the x2 level shortfall, caused by the HU-window clip in preprocessing):
    # F6 restricted to scans whose tex_nps_x2 AND tex_gauss_x0.5 achieved level is within 3 % of target.
    lv["uid"] = [u for c in TEX for f in (0.5, 2.0) for u in allu]
    bad = set(lv[lv.cond.isin(["tex_nps_x2", "tex_gauss_x0.5"]) & (lv.rel_err.abs() > 0.03)].uid)
    keep = [u for u in allu if u not in bad]
    sysk = {k: v for k, v in systems.items() if k.split("|")[1] in ("tex_nps_x2", "tex_gauss_x0.5")}
    sysk = {k: S.scan_results(df[(df.model == k.split("|")[0]) & (df.condition == k.split("|")[1])], keep) for k in sysk}
    _, _, rs = boot_contrasts(sysk, keep, spec[:2])
    rs["p_holm"] = S.holm(rs.p_two_sided).values; rs["n_scans"] = len(keep); rs["n_excluded"] = len(bad)
    rs.to_csv(OUT / "levels_sens_within3pct.csv", index=False)
    print("SENSITIVITY (not pre-specified), scans within 3 % for nps_x2 and gauss_x0.5:"); print(rs.round(4).to_string(index=False))


# ------------------------------------------------------------------------------------------------------------ R3
def match_exam(soft: pd.DataFrame, sharp: pd.DataFrame, thr=0.5, dist=5.0):
    """Confident detections (>= thr) matched one-to-one, greedy by score (soft side in descending score, nearest unmatched
    confident sharp detection within dist mm). Returns n_soft, n_sharp, n_matched, matched score pairs, persistence counts."""
    cs = soft[soft.probability >= thr].sort_values("probability", ascending=False); ch = sharp[sharp.probability >= thr]
    P = cs[["coordX", "coordY", "coordZ"]].values; Q = ch[["coordX", "coordY", "coordZ"]].values; used = np.zeros(len(ch), bool); pairs = []
    for i in range(len(P)):
        if len(Q) == 0:
            break
        d = np.linalg.norm(Q - P[i], axis=1); d[used] = np.inf; j = int(np.argmin(d))
        if d[j] <= dist:
            used[j] = True; pairs.append((cs.probability.iloc[i], ch.probability.iloc[j]))
    def persist(a, b):                                # confident on a with any candidate on b within dist
        A = a[a.probability >= thr][["coordX", "coordY", "coordZ"]].values; B = b[["coordX", "coordY", "coordZ"]].values
        if len(A) == 0:
            return 0, 0
        if len(B) == 0:
            return 0, len(A)
        return int((np.linalg.norm(A[:, None] - B[None], axis=2).min(1) <= dist).sum()), len(A)
    ps, ns = persist(soft, sharp); ph, nh = persist(sharp, soft)
    return {"n_soft": len(cs), "n_sharp": len(ch), "n_match": len(pairs), "pairs": pairs, "pers_soft": ps, "pers_sharp": ph,
            "cands_soft": len(soft), "cands_sharp": len(sharp)}


def nlst():
    D = OUT / "nlst_pairs"; s = pd.read_csv(D / "sample.csv", dtype={"pid": str, "uid_primary": str, "uid_partner": str})
    ok = pd.read_csv(D / "geometry_check.csv", dtype={"pid": str}); s = s[s.pid.isin(set(ok[ok.geometry_ok].pid))].reset_index(drop=True)
    cand = pd.read_csv(D / "pair_candidates.csv", dtype={"pid": str, "uid": str})
    models = sorted(cand.model.unique()); assert len(models) == 8, models
    soft_uid = np.where(s.kclass_primary == "soft", s.uid_primary, s.uid_partner); sharp_uid = np.where(s.kclass_primary == "soft", s.uid_partner, s.uid_primary)
    g = {k: v for k, v in cand.groupby(["model", "uid"])}; empty = cand.iloc[:0]
    per = []
    for mo in models:
        for i, r in s.iterrows():
            e = match_exam(g.get((mo, soft_uid[i]), empty), g.get((mo, sharp_uid[i]), empty))
            per.append({"model": mo, "exam": i, "pid": r.pid, "vendor": r.vendor, **{k: v for k, v in e.items() if k != "pairs"},
                        "score_diff_sum": float(sum(a - b for a, b in e["pairs"]))})
    per = pd.DataFrame(per); per.to_csv(D / "per_exam.csv", index=False)
    n = len(s); M = {mo: per[per.model == mo].sort_values("exam") for mo in models}
    rng = np.random.RandomState(0); idx = rng.randint(0, n, (S.N_BOOT, n))

    def agree(mo, ix=None):
        d = M[mo]; mt, ns, nh = d.n_match.values, d.n_soft.values, d.n_sharp.values
        if ix is None:
            return 2 * mt.sum() / (ns.sum() + nh.sum())
        return 2 * mt[ix].sum(1) / (ns[ix].sum(1) + nh[ix].sum(1))

    def summarize(name, pt, bt, p=True):
        o = {"quantity": name, "estimate": float(pt), "ci_lo": float(np.percentile(bt, 2.5)), "ci_hi": float(np.percentile(bt, 97.5))}
        if p:
            o["p_two_sided"] = float(max(min(1.0, 2 * min((bt <= 0).mean(), (bt >= 0).mean())), 2.0 / len(bt)))
        return o
    A = {mo: (agree(mo), agree(mo, idx)) for mo in models}
    fk = sorted({mo.split("_")[1] for mo in models})
    a3 = (np.mean([A[f"S3_{f}"][0] for f in fk]), np.mean([A[f"S3_{f}"][1] for f in fk], 0))
    a4 = (np.mean([A[f"S4_{f}"][0] for f in fk]), np.mean([A[f"S4_{f}"][1] for f in fk], 0))
    rows = [{**summarize("F7 delta agreement S4-S3 (mean of fold models)", a4[0] - a3[0], a4[1] - a3[1]), "family": "F7"},
            {**summarize("agreement S3 (mean)", a3[0], a3[1], False), "family": "secondary"},
            {**summarize("agreement S4 (mean)", a4[0], a4[1], False), "family": "secondary"},
            {**summarize("fold 0 only: delta agreement S4-S3", A["S4_f0"][0] - A["S3_f0"][0], A["S4_f0"][1] - A["S3_f0"][1], False), "family": "secondary"}]
    for mo in models:
        rows.append({**summarize(f"agreement {mo}", *A[mo], p=False), "family": "secondary"})
    for typ in ("S3", "S4"):                          # persistence, score shift, candidates per scan: mean over fold models
        d = per[per.model.str.startswith(typ)]
        for col, den, lab in (("pers_soft", "n_soft", "persistence soft->sharp"), ("pers_sharp", "n_sharp", "persistence sharp->soft")):
            pt = d[col].sum() / d[den].sum(); ex = d.groupby("exam")[[col, den]].sum().reindex(range(n)).values
            bt = ex[idx, 0].sum(1) / ex[idx, 1].sum(1); rows.append({**summarize(f"{lab} {typ}", pt, bt, False), "family": "secondary"})
        ex = d.groupby("exam")[["score_diff_sum", "n_match"]].sum().reindex(range(n)).values
        rows.append({**summarize(f"mean score soft-sharp of matched {typ}", ex[:, 0].sum() / ex[:, 1].sum(), ex[idx, 0].sum(1) / ex[idx, 1].sum(1), False), "family": "secondary"})
        for k in ("cands_soft", "cands_sharp", "n_soft", "n_sharp"):
            ex = d.groupby("exam")[k].mean().reindex(range(n)).values
            rows.append({**summarize(f"per scan {k} {typ}", ex.mean(), ex[idx].mean(1), False), "family": "secondary"})
    res = pd.DataFrame(rows)
    vend = []
    for v, gv in per.groupby("vendor"):
        for typ in ("S3", "S4"):
            d = gv[gv.model.str.startswith(typ)]; vend.append({"vendor": v, "model": typ, "exams": d.exam.nunique(),
                                                                "agreement": 2 * d.n_match.sum() / (d.n_soft.sum() + d.n_sharp.sum())})
    vend = pd.DataFrame(vend)
    f7 = res.iloc[0]; sig = f7.p_two_sided < 0.05 and f7.estimate > 0
    out = {"n_exams": n, "n_sample": int(len(pd.read_csv(D / "sample.csv"))), "n_geometry_dropped": int((~ok.geometry_ok).sum()),
           "vendors": s.vendor.value_counts().to_dict(),
           "wording": "augmentation made detections more consistent across real kernels" if sig else
                      "augmentation did not measurably change the consistency across real kernels",
           "calibration": json.load(open(D / "kernel_calibration.json")), "by_vendor": vend.to_dict("records")}
    res.to_csv(OUT / "nlst_pairs_results.csv", index=False); json.dump(out, open(OUT / "nlst_pairs_results.json", "w"), indent=1, default=str)
    print(res.round(4).to_string(index=False)); print(vend.round(3).to_string(index=False)); print(json.dumps({k: out[k] for k in ("n_exams", "n_geometry_dropped", "wording")}, indent=1))


# ------------------------------------------------------------------------------------------------------------ R4
def seed():
    index = load_index(); u0 = get_split(index, 0)["test"]
    from e1_dose_robustness import CONDITIONS, CONDITIONS_EXTRA
    s0 = pd.concat([read(fold_out(0) / f) for f in ("e1_candidates_all.csv", "e1_candidates_all_extra.csv")] +
                   [read(fold_out(0) / f"e1_candidates_proj{a}.csv") for a in ("0.5", "0.25", "0.1")] +
                   [read(AN / "p2_texture" / "fold0" / f"e1_candidates_{c}.csv") for c in TEX] +
                   [read(p) for p in sorted((OUT / "inplane" / "fold0").glob("e1_candidates_all_xy*.csv"))])
    s1 = pd.concat([read(OUT / "seed1_e1" / "fold0" / f) for f in ("e1_candidates_all.csv", "e1_candidates_all_extra.csv")] +
                   [read(OUT / "seed1_texture" / "fold0" / f"e1_candidates_{c}.csv") for c in ("proj0.5", "proj0.25", "proj0.1", *TEX)] +
                   [read(p) for p in sorted((OUT / "inplane_seed1" / "fold0").glob("e1_candidates_all_xy*.csv"))])
    conds = [c for c in list(CONDITIONS) + list(CONDITIONS_EXTRA) + ["proj0.5", "proj0.25", "proj0.1", *TEX, "xy1.0", "xy1.5", "xycal"]
             if (s0.condition == c).any() and (s1.condition == c).any()]
    systems = {}
    for tag, df in (("s0", s0), ("s1", s1)):
        for m in ("S3", "S4"):
            for c in conds:
                systems[f"{m}{tag}|{c}"] = S.scan_results(df[(df.model == m) & (df.condition == c)], u0)
    spec = []
    for c in conds:
        spec += [(f"seed diff {c}", m, {f"{m}s1|{c}": 1, f"{m}s0|{c}": -1}) for m in ("S3", "S4")]
        spec += [(f"delta {c}", f"S4-S3 {t}", {f"S4{t}|{c}": 1, f"S3{t}|{c}": -1}) for t in ("s0", "s1")]
    point, boot, r = boot_contrasts(systems, u0, spec)
    rep = {c: bool(r[(r.kind == f"delta {c}") & (r.model == "S4-S3 s1")].ci_lo.iloc[0] > 0) for c in ("soft1.0", "soft1.5")}
    seedvar = r[r.kind.str.startswith("seed diff")].assign(absd=lambda d: d.estimate.abs())
    out = {"cpm": point.to_dict(), "replicated_blur_benefit": rep,
           "seed_abs_diff_clean": {m: float(seedvar[(seedvar.kind == "seed diff clean") & (seedvar.model == m)].absd.iloc[0]) for m in ("S3", "S4")},
           "seed_abs_diff_median_all_conditions": {m: float(seedvar[seedvar.model == m].absd.median()) for m in ("S3", "S4")},
           "seed_abs_diff_max_all_conditions": {m: float(seedvar[seedvar.model == m].absd.max()) for m in ("S3", "S4")}}
    r.to_csv(OUT / "seed_contrasts.csv", index=False)
    # (iv) sensitivity: F1-F3 with fold 0's S3 replaced by the seed-1 S3 (post sampler fix)
    F1 = ["soft1.0", "soft1.5", "soft1.0+noise60"]; F2 = ["cnoise40_c1", "cnoise80_c1", "cnoise40_c2", "cnoise80_c2", "proj0.5", "proj0.25", "proj0.1"]
    allu = [u for k in FOLDS for u in get_split(index, k)["test"]]
    df = pd.concat([read(p) for k in FOLDS for p in sorted(fold_out(k).glob("e1_candidates_*.csv"))]); df = df[df.condition.isin(["clean"] + F1 + F2)]
    df = pd.concat([df[~((df.model == "S3") & df.seriesuid.isin(u0))], s1[(s1.model == "S3") & s1.condition.isin(["clean"] + F1 + F2)]])
    syst = {f"{m}|{c}": S.scan_results(df[(df.model == m) & (df.condition == c)], allu) for c in ["clean"] + F1 + F2 for m in ("S3", "S4")}
    pt, bt = S.bootstrap(syst, [[u] for u in allu])
    sens = pd.DataFrame([{"condition": c, "family": "F1" if c in F1 else "F2", **S.contrast(pt, bt, {f"S4|{c}": 1, f"S3|{c}": -1})} for c in ["clean"] + F1 + F2])
    for fam in ("F1", "F2"):
        m = sens.condition.isin(F1 if fam == "F1" else F2); sens.loc[m, "p_holm"] = S.holm(sens.loc[m, "p_two_sided"]).values
    import importlib; T = importlib.import_module("22_tcia_e1")
    tc = pd.read_csv(AN / "p2_tcia" / "tcia_candidates.csv"); t1 = pd.read_csv(OUT / "seed1_tcia" / "tcia_candidates.csv")
    tc = pd.concat([tc[tc.model != "S3_f0"], t1[t1.model == "S3_f0"]]); T.OUT = OUT / "sens_tcia_s3seed1"; T.OUT.mkdir(parents=True, exist_ok=True)
    tc.to_csv(T.OUT / "tcia_candidates.csv", index=False)
    cases = sorted(p.name[:-len("_full.json")] for p in T.PRE.glob("C*_full.json"))
    excl = set(pd.read_csv(T.EVAL / "excluded_patients.csv").case); T.score([c for c in cases if c not in excl])
    f3 = pd.read_csv(T.OUT / "tcia_contrasts.csv"); f3 = f3[f3.family == "F3"]
    sens = pd.concat([sens, pd.DataFrame([{"condition": r_.contrast, "family": "F3", "estimate": r_.estimate, "ci_lo": r_.ci_lo, "ci_hi": r_.ci_hi,
                                           "p_two_sided": r_.p_two_sided, "p_holm": r_.p_holm} for r_ in f3.itertuples()])])
    orig = pd.read_csv(AN / "e1_dose_pooled" / "final_delta.csv").set_index("condition")
    sens["same_significance_as_main"] = [None if f == "F3" else bool((p < 0.05) == (orig.loc[c, "p_holm"] < 0.05)) if c in orig.index and not np.isnan(p) else None
                                         for c, f, p in zip(sens.condition, sens.family, sens.p_holm)]
    sens.to_csv(OUT / "seed_sensitivity_f1f3.csv", index=False); out["sensitivity_same_significance"] = bool(sens.same_significance_as_main.dropna().all())
    json.dump(out, open(OUT / "seed_results.json", "w"), indent=1)
    print(r[r.kind.isin([f"delta {c}" for c in ("clean", "soft1.0", "soft1.5")] + ["seed diff clean"])].round(4).to_string(index=False))
    print(sens.round(4).to_string(index=False)); print(json.dumps({k: v for k, v in out.items() if k != "cpm"}, indent=1))


def levels_kernel():
    """NOT pre-specified (added 3 Nov after the within-3 % sensitivity reversed F6): F6 and the level losses by the native
    kernel class of the scan (the real-NPS texture is the B30f spectrum on soft- and the B50f spectrum on sharp-kernel
    scans; the 2x level shortfall is almost all on sharp-kernel scans, whose target level is higher)."""
    index = load_index(); allu = [u for k in FOLDS for u in get_split(index, k)["test"]]; sub = index.set_index("seriesuid").subset
    e1 = pd.concat([read(fold_out(k) / "e1_candidates_all.csv") for k in FOLDS]); e1 = e1[e1.condition == "clean"]
    tl = pd.concat([read(OUT / "levels" / f"fold{k}" / f"e1_candidates_{c}_x{f:g}.csv") for k in FOLDS for c in TEX for f in (0.5, 2.0)])
    t1 = pd.concat([read(AN / "p2_texture" / f"fold{k}" / f"e1_candidates_{c}.csv") for k in FOLDS for c in TEX]); t1 = t1.assign(condition=t1.condition + "_x1")
    df = pd.concat([e1, t1, tl]); conds = ["clean"] + [f"{c}_x{f:g}" for c in TEX for f in (0.5, 1, 2)]
    LEV = Path(os.environ.get("LUNGCAD_NLST_HDD", "data/raw"), "p2_review", "texture_levels")
    kc, tgt, err = {}, {}, {}
    for u in allu:
        j = json.load(open(LEV / "tex_nps_x2" / f"subset{int(sub[u])}" / f"{u}.json")); kc[u] = j["kernel_class"]; tgt[u] = j["target_sd_hu"]; err[u] = j["rel_err"]
    rows = []
    for k_ in ("soft", "sharp"):
        us = [u for u in allu if kc[u] == k_]
        systems = {f"{m}|{c}": S.scan_results(df[(df.model == m) & (df.condition == c)], us) for c in conds for m in ("S3", "S4")}
        spec = [("F6 nps_x2 - gauss_x0.5", m, {f"{m}|tex_nps_x2": 1, f"{m}|tex_gauss_x0.5": -1}) for m in ("S3", "S4")]
        spec += [(f"drop {c}", m, {f"{m}|{c}": 1, f"{m}|clean": -1}) for m in ("S3", "S4") for c in conds[1:]]
        _, _, r = boot_contrasts(systems, us, spec)
        r["kernel"] = k_; r["n_scans"] = len(us); r["target_x2_median_hu"] = float(np.median([tgt[u] for u in us]))
        r["n_outside_3pct_nps_x2"] = int(sum(abs(err[u]) > 0.03 for u in us))
        r["shortfall_nps_x2_median"] = float(-np.median([err[u] for u in us])); r["shortfall_nps_x2_p90"] = float(-np.percentile([err[u] for u in us], 10))
        rows.append(r)
    r = pd.concat(rows); r.to_csv(OUT / "levels_by_kernel.csv", index=False)
    print(r[r.kind.str.startswith("F6") | r.kind.str.contains("nps|gauss")].round(4).to_string(index=False))


def nlst_vendor():
    """Secondary (pre-spec 'by-vendor results'): the F7 estimator (mean over fold models of S4 - S3 agreement) within each
    vendor, bootstrap over that vendor's exams (10,000, seed 0). Reads nlst_pairs/per_exam.csv written by nlst()."""
    per = pd.read_csv(OUT / "nlst_pairs" / "per_exam.csv", dtype={"pid": str}); folds = sorted({m.split("_")[1] for m in per.model})
    rows = []
    for v in ["all"] + sorted(per.vendor.unique()):
        d = per if v == "all" else per[per.vendor == v]; ex = sorted(d.exam.unique()); n = len(ex)
        idx = np.random.RandomState(0).randint(0, n, (S.N_BOOT, n))
        def A(mo):
            x = d[d.model == mo].set_index("exam").reindex(ex); mt, ns, nh = x.n_match.values, x.n_soft.values, x.n_sharp.values
            return 2 * mt.sum() / (ns.sum() + nh.sum()), 2 * mt[idx].sum(1) / (ns[idx].sum(1) + nh[idx].sum(1))
        a = {f"{t}_{f}": A(f"{t}_{f}") for t in ("S3", "S4") for f in folds}
        m3 = np.mean([a[f"S3_{f}"][0] for f in folds]); m4 = np.mean([a[f"S4_{f}"][0] for f in folds])
        b = np.mean([a[f"S4_{f}"][1] for f in folds], 0) - np.mean([a[f"S3_{f}"][1] for f in folds], 0)
        rows.append({"vendor": v, "n_exams": n, "agree_S3": m3, "agree_S4": m4, "delta": m4 - m3,
                     "ci_lo": float(np.percentile(b, 2.5)), "ci_hi": float(np.percentile(b, 97.5)),
                     **{f"delta_{f}": a[f"S4_{f}"][0] - a[f"S3_{f}"][0] for f in folds}})
    r = pd.DataFrame(rows); ref = pd.read_csv(OUT / "nlst_pairs_results.csv").set_index("quantity")
    assert abs(r.iloc[0].delta - ref.loc["F7 delta agreement S4-S3 (mean of fold models)", "estimate"]) < 1e-12
    assert abs(r.iloc[0].ci_lo - ref.loc["F7 delta agreement S4-S3 (mean of fold models)", "ci_lo"]) < 1e-12
    print("known answer: 'all' row equals F7 (estimate and CI)")
    r.to_csv(OUT / "nlst_pairs_vendor.csv", index=False); print(r.round(4).to_string(index=False))


def nlst_threshold():
    """Sensitivity (NOT pre-specified, added 3 Nov after review): F7 agreement at confidence thresholds 0.3/0.5/0.7, plus a
    count-matched variant (each S4 fold model thresholded so that its total number of confident detections over all soft
    and sharp series equals that of the S3 model of the same fold at 0.5), because S4 scores are shifted lower and a fixed
    threshold could change agreement through calibration alone. Same estimator and bootstrap (10,000, seed 0) as F7.
    Known answer: the 0.5 row reproduces F7."""
    D = OUT / "nlst_pairs"; s = pd.read_csv(D / "sample.csv", dtype={"pid": str, "uid_primary": str, "uid_partner": str})
    ok = pd.read_csv(D / "geometry_check.csv", dtype={"pid": str}); s = s[s.pid.isin(set(ok[ok.geometry_ok].pid))].reset_index(drop=True)
    cand = pd.read_csv(D / "pair_candidates.csv", dtype={"pid": str, "uid": str}); models = sorted(cand.model.unique())
    soft_uid = np.where(s.kclass_primary == "soft", s.uid_primary, s.uid_partner); sharp_uid = np.where(s.kclass_primary == "soft", s.uid_partner, s.uid_primary)
    g = {k: v for k, v in cand.groupby(["model", "uid"])}; empty = cand.iloc[:0]; n = len(s); folds = sorted({m.split("_")[1] for m in models})
    used = set(soft_uid) | set(sharp_uid); cu = cand[cand.uid.isin(used)]
    thr = {(mo, t): t for mo in models for t in (0.3, 0.5, 0.7)}
    for f in folds:                                   # count-matched threshold for S4_f: k-th largest score, k = S3_f count at 0.5
        k = int((cu[cu.model == f"S3_{f}"].probability >= 0.5).sum()); sc = np.sort(cu[cu.model == f"S4_{f}"].probability.values)[::-1]
        thr[(f"S4_{f}", "cm")] = float(sc[k - 1]); thr[(f"S3_{f}", "cm")] = 0.5
    idx = np.random.RandomState(0).randint(0, n, (S.N_BOOT, n)); rows = []
    for t in (0.3, 0.5, 0.7, "cm"):
        ag = {}
        for mo in models:
            e = [match_exam(g.get((mo, soft_uid[i]), empty), g.get((mo, sharp_uid[i]), empty), thr=thr[(mo, t)]) for i in range(n)]
            mt = np.array([x["n_match"] for x in e]); ns = np.array([x["n_soft"] for x in e]); nh = np.array([x["n_sharp"] for x in e])
            ps = np.array([x["pers_soft"] for x in e]); ph = np.array([x["pers_sharp"] for x in e])
            ag[mo] = (2 * mt.sum() / (ns.sum() + nh.sum()), 2 * mt[idx].sum(1) / (ns[idx].sum(1) + nh[idx].sum(1)),
                      (ns.sum() + nh.sum()) / (2 * n), (ps.sum() + ph.sum()) / max(1, ns.sum() + nh.sum()))
        m = {typ: np.mean([ag[f"{typ}_{f}"][0] for f in folds]) for typ in ("S3", "S4")}
        b = np.mean([ag[f"S4_{f}"][1] for f in folds], 0) - np.mean([ag[f"S3_{f}"][1] for f in folds], 0)
        d = m["S4"] - m["S3"]; p = float(max(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean())), 2.0 / len(b)))
        rows.append({"threshold": str(t), "thr_S4_folds": " ".join(f"{thr[(f'S4_{f}', t)]:.3f}" for f in folds),
                     "agree_S3": m["S3"], "agree_S4": m["S4"], "delta": d, "ci_lo": float(np.percentile(b, 2.5)), "ci_hi": float(np.percentile(b, 97.5)),
                     "p_two_sided": p, **{f"conf_per_series_{typ}": np.mean([ag[f"{typ}_{f}"][2] for f in folds]) for typ in ("S3", "S4")},
                     **{f"persistence_{typ}": np.mean([ag[f"{typ}_{f}"][3] for f in folds]) for typ in ("S3", "S4")},
                     **{f"delta_{f}": ag[f"S4_{f}"][0] - ag[f"S3_{f}"][0] for f in folds}})
    r = pd.DataFrame(rows); ref = pd.read_csv(OUT / "nlst_pairs_results.csv").set_index("quantity").loc["F7 delta agreement S4-S3 (mean of fold models)"]
    r5 = r[r.threshold == "0.5"].iloc[0]
    assert abs(r5.delta - ref.estimate) < 1e-12 and abs(r5.ci_lo - ref.ci_lo) < 1e-12 and abs(r5.ci_hi - ref.ci_hi) < 1e-12
    print("known answer: threshold 0.5 reproduces F7 (estimate and CI)")
    r.to_csv(OUT / "nlst_pairs_threshold.csv", index=False); print(r.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("step", choices=["inplane", "levels", "levelskernel", "nlst", "nlstvendor", "nlstthr", "seed"]); a = ap.parse_args()
    {"inplane": inplane, "levels": levels, "levelskernel": levels_kernel, "nlst": nlst, "nlstvendor": nlst_vendor,
     "nlstthr": nlst_threshold, "seed": seed}[a.step]()
