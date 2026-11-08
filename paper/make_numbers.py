"""Generate paper/numbers.tex: every number quoted in the manuscript, read from the result files (never typed).

    $PY paper/make_numbers.py          (ml env; needs paper/data/tcia_ctdi.json from paper/tcia_ctdi.py, astra env)

In LaTeX: \\val{key} prints a value; an unknown key prints a bold ?? and a warning, so a typo cannot pass silently.
Keys: d.<cond>.{est,lo,hi,p,ph,folds}  (Delta CPM S4-S3), c.<model>.<cond>.{cpm,lo,hi}, drop.<model>.<cond>.{est,lo,hi},
cps.<model>.<cond> (candidates per scan), and named scalars (see the code).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from lungcad import config as C

AN = C.WORK / "wp0" / "analysis"; POOL = AN / "e1_dose_pooled"; TC = AN / "p2_tcia"; NL = AN / "p2_nlst"
TCIA = C.THESIS_DATA / "tcia_ldct"; HERE = Path(__file__).resolve().parent
V: dict[str, str] = {}


def f3(x): return f"{x:.3f}".replace("-", "$-$")
def s3(x): return "0.000" if abs(x) < 0.0005 else ("+" if x >= 0 else "$-$") + f"{abs(x):.3f}"          # signed, for differences
def s2(x): return "0.00" if abs(x) < 0.005 else ("+" if x >= 0 else "$-$") + f"{abs(x):.2f}"
def pv(p, floor):                                                               # P values: floor = 2/n_boot
    if p is None or pd.isna(p): return "--"
    if p <= floor + 1e-12: return f"$\\le${floor:g}"                         # bootstrap floor 2/n_boot
    if p < 0.01: return f"{p:.1g}" if f"{p:.2g}".endswith("0") else f"{p:.2g}"
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"
def put(k, v): assert k not in V, k; V[k] = str(v)


# ---------------------------------------------------------------- LUNA16 pooled E1 (4 folds)
d = pd.read_csv(POOL / "final_delta.csv")
pf = pd.read_csv(POOL / "final_per_fold.csv"); d["folds_S4_better"] = d.condition.map(pf[pf.delta > 1e-12].groupby("condition").size()).fillna(0)
floor = 2 / d.n_boot.iloc[0]
put("nboot", f"{int(d.n_boot.iloc[0]):,}".replace(",", "{,}"))
for r in d.itertuples():
    c = r.condition
    put(f"d.{c}.est", s3(r.estimate)); put(f"d.{c}.lo", s3(r.ci_lo)); put(f"d.{c}.hi", s3(r.ci_hi))
    put(f"d.{c}.p", pv(r.p_two_sided, floor)); put(f"d.{c}.ph", pv(r.p_holm, floor) if not pd.isna(r.p_holm) else "--")
    put(f"d.{c}.folds", int(r.folds_S4_better))
    put(f"c.S3.{c}.cpm", f3(r.cpm_S3)); put(f"c.S4.{c}.cpm", f3(r.cpm_S4))
cp = pd.read_csv(POOL / "final_cpm.csv", index_col=0)
for k, r in cp.iterrows():
    m, c = k.split("|"); put(f"c.{m}.{c}.lo", f3(r.ci_lo)); put(f"c.{m}.{c}.hi", f3(r.ci_hi))
dr = pd.read_csv(POOL / "final_drop.csv")
for r in dr.itertuples():
    put(f"loss.{r.model}.{r.condition}.est", f"{-r.estimate:.3f}"); put(f"loss.{r.model}.{r.condition}.lo", f"{-r.ci_hi:.3f}"); put(f"loss.{r.model}.{r.condition}.hi", f"{-r.ci_lo:.3f}")
    put(f"drop.{r.model}.{r.condition}.est", s3(r.estimate)); put(f"drop.{r.model}.{r.condition}.lo", s3(r.ci_lo)); put(f"drop.{r.model}.{r.condition}.hi", s3(r.ci_hi))
# white-noise equivalence: every white-noise (and clean) Delta CI inside +-0.02
wn = d[d.condition.isin(["clean", "noise20", "noise40", "noise60", "noise80", "noise100"])]
put("eq.inside", int(((wn.ci_lo > -0.02) & (wn.ci_hi < 0.02)).sum())); put("eq.n", len(wn))
assert int(V["eq.inside"]) == int(V["eq.n"]) == 6, "main.tex says all six white-noise/clean intervals lie inside +-0.02: update the text"
wd = dr[dr.condition.isin(["noise20", "noise40", "noise60", "noise80", "noise100"])]
put("white.maxdrop", f"{-wd.estimate.min():.3f}")
cd = dr[dr.condition.isin(["cnoise80_c1", "cnoise80_c2"])]
put("corr80.mindrop", f"{-cd.estimate.max():.2f}"); put("corr80.maxdrop", f"{-cd.estimate.min():.2f}")

# test-set size and per-operating-point sensitivities (clean), from the pooled E1 table
res = pd.read_csv(POOL / "e1_results_folds0123.csv")
cl = res[(res.condition == "clean")].set_index("model")
put("luna.nnod", int(cl.loc["S3", "n_nodules"]))
for m in ("S3", "S4"):
    put(f"luna.found.{m}", int(cl.loc[m, "detected"])); put(f"sens8.{m}.clean", f"{cl.loc[m, 's@0.125']:.3f}")

# candidates per scan, every condition, from the candidate files of the four folds (threshold 0.02 as stored)
cands = []
for k in range(4):
    D = AN / f"e1_dose_fold{k}"
    for f in ["e1_candidates_all.csv", "e1_candidates_all_extra.csv"] + [f"e1_candidates_proj{a}.csv" for a in ("0.5", "0.25", "0.1")]:
        x = pd.read_csv(D / f, usecols=["model", "condition", "seriesuid", "probability"]); x["fold"] = k; cands.append(x)
cands = pd.concat(cands); assert cands.probability.min() >= 0.02
uids = [u for k in range(4) for u in __import__("pickle").load(open(AN / f"e1_dose_fold{k}" / "rows_partial.pkl", "rb"))["done"]]
nscan = len(set(uids)); put("luna.nscan", nscan); assert nscan == 356
cps = cands.groupby(["model", "condition"]).size() / nscan
assert abs(cps[("S3", "clean")] - cl.loc["S3", "cands_per_scan"]) < 1e-9, "candidate count does not match the E1 table"
for (m, c), v in cps.items(): put(f"cps.{m}.{c}", f"{v:.1f}")

# LUNA16 test scans: acquisition mix (LIDC DICOM headers)
hdr = pd.read_csv(C.LUNA_INDEX / "lidc_headers.csv"); hdr = hdr[hdr.seriesuid.isin(set(uids))]
assert len(hdr) == nscan
thk = pd.to_numeric(hdr.SliceThickness, errors="coerce")
put("luna.thk.min", f"{thk.min():.2f}".rstrip("0").rstrip(".")); put("luna.thk.max", f"{thk.max():.1f}")
put("luna.nvendor", hdr.Manufacturer.str.split().str[0].str.upper().nunique())
sys.path.insert(0, str(ROOT / "scripts"))
from lungcad.data.nlst_inventory import kernel_class
kc = hdr.ConvolutionKernel.map(lambda k: kernel_class(str(k))); put("luna.nsoft", int((kc == "soft").sum())); put("luna.nsharp", int((kc == "sharp").sum()))

# per-fold validation CPM (S3 vs S4), for the Methods
for k in range(4):
    for m, sub in (("S3", "seed0_s3_gate_noaug" if k == 0 else "seed0_s3_noaug"), ("S4", "seed0")):
        h = pd.read_json(C.STAGE1 / f"fold{k}" / sub / "val_history.jsonl", lines=True)
        put(f"val.{m}.f{k}", f3(h.cpm.max()))

# ---------------------------------------------------------------- projection-domain noise
cal = json.load(open(TCIA / "calibration" / "calib_all_cuda.json")); rl = cal["filters"]["ram-lak"]
put("cal.ncases", len(cal["cases"])); put("cal.nslices", rl["n_slices"])
put("cal.err.heldout", f"{100 * rl['median_rel_err_held_out']:.0f}"); put("cal.err.const", f"{100 * rl['median_rel_err_const_i0']:.0f}")
put("cal.tcm.c", f"{rl['tcm_c']:.1f}"); put("cal.tcm.beta", f"{rl['tcm_beta']:.2f}"); put("cal.i0", f"{cal['I0_ref']:.0e}".replace("e+0", "\\times10^{") + "}")
put("cal.sigmae", f"{cal['sigma_e']:.0f}")
others = {f: v["nps_shape_L1"] for f, v in cal["filters"].items() if f != "ram-lak"}
put("cal.nps.rl", f"{rl['nps_shape_L1']:.2f}"); put("cal.nps.next", f"{min(others.values()):.2f}")
ab = json.load(open(TCIA / "calibration" / "calib_abdomen_cuda.json"))["filters"]
best = min(ab, key=lambda f: ab[f]["nps_shape_L1"]); assert best == "hamming"
put("cal.abd.best", f"{ab[best]['nps_shape_L1']:.2f}"); put("cal.abd.rl", f"{ab['ram-lak']['nps_shape_L1']:.2f}")
put("cal.abd.nslices", ab[best]["n_slices"])
cc = pd.read_csv(TCIA / "calibration" / "calib_all_cuda.csv"); cc = cc[cc["filter"] == "ram-lak"]
pc = cc.groupby("case").sd_real.median()
put("tcia.sd.min", f"{pc.min():.0f}"); put("tcia.sd.max", f"{pc.max():.0f}"); put("tcia.sd.med", f"{cc.sd_real.median():.0f}")
sd = {}
for a in ("0.5", "0.25", "0.1"):
    js = [json.load(open(p)) for p in (C.THESIS_DATA / "luna16" / "projnoise" / f"a{a}").glob("subset*/*.json")]
    assert len(js) == nscan and all(j["projector"] == "cuda" for j in js)
    sd[a] = np.array([j["noise_sd_added_hu"] for j in js]); put(f"pn.sd.{a}", f"{np.median(sd[a]):.0f}")
    put(f"pn.sd.{a}.q1", f"{np.percentile(sd[a], 25):.0f}"); put(f"pn.sd.{a}.q3", f"{np.percentile(sd[a], 75):.0f}")
    if a == "0.1":
        put("pn.nsharp", sum(j["noise_filter"] == "ram-lak" for j in js)); put("pn.nsoft", sum(j["noise_filter"] == "hamming" for j in js))
        for filt, lab in (("ram-lak", "sharp"), ("hamming", "soft")):
            put(f"pn.sd.0.1.{lab}", f"{np.median([j['noise_sd_added_hu'] for j in js if j['noise_filter'] == filt]):.0f}")

# ---------------------------------------------------------------- TCIA paired full vs 10 %
ck = pd.read_csv(TCIA / "eval" / "lesions_checked.csv"); excl = set(pd.read_csv(TCIA / "eval" / "excluded_patients.csv").case)
v = ck[(ck.status == "verified") & ~ck.case.isin(excl)]; u = ck[ck.status != "verified"]
put("tcia.nles.all", len(ck)); put("tcia.npat.les", ck.case.nunique()); put("tcia.nles", len(v)); put("tcia.npat", 50 - len(excl))
put("tcia.nles.pat", v.case.nunique()); put("tcia.nsolid", int(v.diagnosis.str.startswith("Solid").sum()))
put("tcia.nsubsolid", len(v) - int(v.diagnosis.str.startswith("Solid").sum())); put("tcia.nunres", len(u))
put("tcia.nslice", int((u.status == "slice_unknown").sum()))
put("tcia.dmin", f"{v.diameter_mm.min():.1f}"); put("tcia.dmax", f"{v.diameter_mm.max():.1f}"); put("tcia.dmed", f"{v.diameter_mm.median():.1f}")
ct = json.load(open(HERE / "data" / "tcia_ctdi.json"))
put("tcia.ctdi.min", f"{ct['ctdi_min']:.1f}"); put("tcia.ctdi.max", f"{ct['ctdi_max']:.1f}"); put("tcia.ctdi.med", f"{ct['ctdi_median']:.1f}")
put("tcia.ctdi.n", ct["n_ctdi"]); put("tcia.ctdi.low", f"{ct['ctdi_median'] / 10:.1f}")
tc = pd.read_csv(TC / "tcia_cpm.csv", index_col=0).cpm
for m in ("S3", "S4"):
    for dose in ("full", "low"):
        vals = tc[[f"{m}_f{k}_{dose}" for k in range(4)]]
        put(f"tcia.{m}.{dose}", f3(vals.mean())); put(f"tcia.{m}.{dose}.min", f3(vals.min())); put(f"tcia.{m}.{dose}.max", f3(vals.max()))
put("tcia.cpm.min", f"{tc.min():.2f}"); put("tcia.cpm.max", f"{tc.max():.2f}")
tcs = pd.read_csv(TC / "tcia_cpm_solid.csv", index_col=0).cpm
for m in ("S3", "S4"):
    for dose in ("full", "low"): put(f"tciasolid.{m}.{dose}", f3(tcs[[f"{m}_f{k}_{dose}" for k in range(4)]].mean()))
keys = {"DD (S4 loses less than S3, mean of 4 pairs)": "dd", "D_low S4-S3 at 10 %": "dlow", "D_full S4-S3 at full dose": "dfull",
        "S3 drop full->10 % (mean of 4)": "dropS3", "S4 drop full->10 % (mean of 4)": "dropS4"}
for name, df in (("tcia", pd.read_csv(TC / "tcia_contrasts.csv")), ("tciasolid", pd.read_csv(TC / "tcia_contrasts_solid.csv"))):
    tfloor = 2 / df.n_boot.iloc[0]
    for r in df.itertuples():
        k = keys.get(r.contrast) or (f"dd.f{r.contrast[5]}" if r.contrast.startswith("fold") else None); assert k, r.contrast
        put(f"{name}.{k}.est", s3(r.estimate)); put(f"{name}.{k}.lo", s3(r.ci_lo)); put(f"{name}.{k}.hi", s3(r.ci_hi)); put(f"{name}.{k}.p", pv(r.p_two_sided, tfloor))
        if name == "tcia" and r.family == "F3": put(f"tcia.{k}.ph", pv(r.p_holm, tfloor))
tcand = pd.read_csv(TC / "tcia_candidates.csv")
grp = tcand.assign(dose=tcand.seriesuid.str.split("_").str[-1]).groupby(["dose"]).size() / (50 - len(excl)) / 8
put("tcia.cps.full", f"{grp['full']:.1f}"); put("tcia.cps.low", f"{grp['low']:.1f}")

# ---------------------------------------------------------------- NLST descriptive
st = pd.read_csv(NL / "strata.csv"); st["round"] = st["round"].astype(str); ps = pd.read_csv(NL / "per_series.csv")
def row(v="all", k="all", r="all", ds="NLST"):
    x = st[(st.dataset.str.startswith(ds)) & (st.vendor == v) & (st.kernel_class == k) & (st["round"] == r)]; assert len(x) == 1; return x.iloc[0]
put("nlst.n", len(ps)); put("nlst.ncells", ps.groupby(["vendor", "kernel_class", "round"]).ngroups)
a = row()
for s, lab in (("cands_per_scan", "cps"), ("high_per_scan", "high"), ("score_q50", "med")):
    for m in ("S3", "S4"): put(f"nlst.{lab}.{m}", f"{a[f'{s}_{m}']:.2f}" if lab != "med" else f3(a[f"{s}_{m}"]))
    fmt = s2 if lab != "med" else s3
    put(f"nlst.{lab}.d", fmt(a[f"{s}_diff"])); put(f"nlst.{lab}.lo", fmt(a[f"{s}_diff_lo"])); put(f"nlst.{lab}.hi", fmt(a[f"{s}_diff_hi"]))
for m in ("S3", "S4"): put(f"nlst.zero.{m}", f"{100 * a[f'share_zero_{m}']:.1f}")
for v_ in ("GE", "PHILIPS", "SIEMENS", "TOSHIBA"):
    for k in ("soft", "sharp"):
        r = row(v_, k); put(f"nlst.{v_}.{k}.n", int(r.n_series))
        put(f"nlst.{v_}.{k}.d", s2(r.cands_per_scan_diff)); put(f"nlst.{v_}.{k}.lo", s2(r.cands_per_scan_diff_lo)); put(f"nlst.{v_}.{k}.hi", s2(r.cands_per_scan_diff_hi))
others = st[(st.dataset == "NLST") & (st.vendor != "all") & (st.kernel_class != "all") & (st["round"] == "all") & (st.vendor != "TOSHIBA") & (st.n_series >= 100)]
put("nlst.other.dmin", f"{others.cands_per_scan_diff.min():.1f}"); put("nlst.other.dmax", f"{others.cands_per_scan_diff.max():.1f}")
lr = st[st.dataset != "NLST"].iloc[0]
put("lunaref.cps.d", s2(lr.cands_per_scan_diff)); put("lunaref.cps.lo", s2(lr.cands_per_scan_diff_lo)); put("lunaref.cps.hi", s2(lr.cands_per_scan_diff_hi))
put("lunaref.med.d", s3(lr.score_q50_diff)); put("lunaref.high.d", s2(lr.high_per_scan_diff)); put("lunaref.high.lo", s2(lr.high_per_scan_diff_lo)); put("lunaref.high.hi", s2(lr.high_per_scan_diff_hi))
put("nlst.thick.n", int((ps.thk_mm > 2.5).sum())); put("nlst.thk.min", f"{ps.thk_mm.min():.1f}"); put("nlst.thk.max", f"{ps.thk_mm.max():.1f}")
put("nlst.nsoft", int((ps.kernel_class == "soft").sum())); put("nlst.nsharp", int((ps.kernel_class == "sharp").sum()))
put("nlst.ntosh.sharp", int(((ps.vendor == "TOSHIBA") & (ps.kernel_class == "sharp")).sum()))
inv = pd.read_csv(NL / "sample.csv"); put("nlst.nkernels", inv.kernel.nunique())

# ---------------------------------------------------------------- missed nodules (ten-fold S4-configuration detectors)
nodf = pd.read_csv(AN / "e5_misses" / "all_nodules_hit_status.csv")      # the Figure 1 nodule (same rule as make_figures.fig1)
put("fig1.diam", f"{nodf[(nodf.fold == 0) & nodf.hit & nodf.diameter_mm.between(7, 10)].sort_values('diameter_mm').iloc[0].diameter_mm:.0f}")
mr = pd.read_csv(AN / "e5_misses" / "misses_reviewed.csv"); lo = pd.read_csv(AN / "e5_misses" / "misses_low_threshold.csv")
allnod = pd.read_csv(AN / "e5_misses" / "all_nodules_hit_status.csv")
put("miss.n", len(mr)); put("miss.nnod", len(allnod)); put("miss.recall", f"{100 * allnod.hit.mean():.1f}")
put("miss.inside", int(mr.in_crop.sum())); put("miss.seen", int(np.isfinite(lo.best_score_within_r).sum()))
for c_, n_ in mr.category.value_counts().items(): put(f"miss.cat.{c_}", int(n_))

# ---------------------------------------------------------------- review additions (2 Nov): secondary analyses (scripts/29_p2_secondary.py)
SEC = AN / "p2_secondary"
wdr = dr[dr.condition.isin(["noise20", "noise40", "noise60", "noise80", "noise100"])]
put("white.droplo", f"{-wdr.ci_lo.min():.3f}")                    # largest loss inside any white-noise paired CI (both models)
put("white.nsig", int((wdr.ci_hi < 0).sum()))                     # white-noise drops whose CI excludes zero
bs = pd.read_csv(SEC / "by_subset.csv")
for r in bs.itertuples():
    sk = r.subset.replace(" ", "_")
    put(f"sub.{sk}.n", r.n_scans) if r.condition == "clean" else None
    put(f"sub.{sk}.{r.condition}.S3", f3(r.cpm_S3)); put(f"sub.{sk}.{r.condition}.S4", f3(r.cpm_S4))
    put(f"sub.{sk}.{r.condition}.d", s3(r.delta)); put(f"sub.{sk}.{r.condition}.lo", s3(r.delta_lo)); put(f"sub.{sk}.{r.condition}.hi", s3(r.delta_hi))
    if r.condition != "clean":
        for m in ("S3", "S4"):
            put(f"sub.{sk}.{r.condition}.loss{m}", f"{-getattr(r, f'drop_{m}'):.3f}")
ddf = pd.read_csv(SEC / "dd_sensitivity.csv")
for r in ddf.itertuples():
    put(f"dd.{r.condition}.est", s3(r.dd)); put(f"dd.{r.condition}.lo", s3(r.lo)); put(f"dd.{r.condition}.hi", s3(r.hi))
    put(f"dd.{r.condition}.ph", pv(r.p_holm, floor) if not pd.isna(r.p_holm) else "--")
    fv = [getattr(r, f"fold{k}") for k in range(4)]; put(f"dd.{r.condition}.fmin", s3(min(fv))); put(f"dd.{r.condition}.fmax", s3(max(fv)))
    put(f"dd.{r.condition}.fpos", sum(v > 0 for v in fv))
sig_dd = {r.condition for r in ddf.itertuples() if r.family in ("F1", "F2") and r.p_holm < 0.05}
sig_d = set(d[(d.family.isin(["F1", "F2"])) & (d.p_holm < 0.05)].condition)
put("dd.same", "yes" if sig_dd == sig_d else "NO"); assert sig_dd == sig_d, "DD sensitivity changes which comparisons are significant: update the text"
cf = pf[pf.condition == "clean"].delta; put("clean.fmin", s3(cf.min())); put("clean.fmax", s3(cf.max())); put("clean.fspread", f"{cf.max() - cf.min():.3f}")
lv = pd.read_csv(SEC / "noise_levels.csv"); lvm = lv.groupby("condition").added_sd_hu.median()
for c_, v_ in lvm.items(): put(f"lvl.{c_}", f"{v_:.0f}")

# ---------------------------------------------------------------- noise-texture test (scripts/25-28, pre-specified: lung-nodule-detector-analysis-plans/plans_original/texture_test_prespec.md)
TXD = AN / "p2_texture"
tr = json.load(open(TXD / "texture_results.json")); tcx = pd.read_csv(TXD / "texture_contrasts.csv")
put("tex.verdict", tr["verdict"])
for k_, v_ in tr["cpm"].items():
    m, c = k_.split("|"); put(f"tex.cpm.{m}.{c}", f3(v_))
for r in tcx.itertuples():
    sk = {"all": "all", "native soft": "soft", "native sharp": "sharp"}[r.subset]
    kk = r.kind.replace(" ", "_").replace("S4-S3_", "")
    key = f"tex.{sk}.{kk}.{r.model}"
    put(f"{key}.est", s3(r.estimate)); put(f"{key}.lo", s3(r.ci_lo)); put(f"{key}.hi", s3(r.ci_hi)); put(f"{key}.p", pv(r.p_two_sided, floor))
    if r.kind.startswith("drop"):
        put(f"{key}.loss", f3(-r.estimate)); put(f"{key}.losslo", f3(-r.ci_hi)); put(f"{key}.losshi", f3(-r.ci_lo))
    if r.subset == "all" and r.kind == "F4 nps-gauss":
        put(f"{key}.ph", pv(r.p_holm, floor))
for c_, g in tr["level"].items():
    put(f"tex.lvl.{c_}.med", f"{g['achieved_median']:.0f}"); put(f"tex.lvl.{c_}.maxerr", f"{100 * g['max_abs_rel_err']:.1f}"); put(f"tex.lvl.{c_}.nout", g["n_outside_3pct"])
for k_, v_ in tr["cands_per_scan"].items():
    m, c = k_.split("|"); put(f"tex.cps.{m}.{c}", f"{v_:.1f}")
nps = json.load(open(TXD / "tcia_nps.json"))
for reg, lab in (("chest_B50f", "chest"), ("abdomen_B30f", "abd")):
    r = nps[reg]; put(f"tex.nps.{lab}.zc", f"{r['z_corr_adjacent_median']:.2f}"); put(f"tex.nps.{lab}.nsl", r["n_slices"])
    pk = r["f_center"][int(np.nanargmax([np.nan if x is None else x for x in r["nps"]]))]; put(f"tex.nps.{lab}.peak", f"{pk:.2f}")

# ---------------------------------------------------------------- review experiments R1-R4 (pre-specified: lung-nodule-detector-analysis-plans/plans_original/review_prespec.md)
RV = AN / "p2_review"
sys.path.insert(0, str(ROOT / "scripts")); from e1_dose_robustness import load_index as _li, get_split as _gs
put("luna.nscan.f0", len(_gs(_li(), 0)["test"]))
def kslug(s): return s.replace(" - ", "-").replace(" ", "_")
sp_z, sp_xy = C.LUNA_SPACING[0], C.LUNA_SPACING[1]                      # blur sigma in mm (the E1 blur is isotropic in voxels)
for sv in (0.5, 1.0, 1.5):
    put(f"blur.{sv}.z", f"{sv * sp_z:.2f}"); put(f"blur.{sv}.xy", f"{sv * sp_xy:.2f}")
for step in ("inplane", "levels"):
    f_ = RV / f"{step}_contrasts.csv"
    if not f_.exists():
        continue
    for r in pd.read_csv(f_).itertuples():
        key = f"rv.{step}.{kslug(r.kind)}.{r.model.replace(' ', '')}"
        put(f"{key}.est", s3(r.estimate)); put(f"{key}.lo", s3(r.ci_lo)); put(f"{key}.hi", s3(r.ci_hi)); put(f"{key}.p", pv(r.p_two_sided, floor))
        if not pd.isna(getattr(r, "p_holm", np.nan)):
            put(f"{key}.ph", pv(r.p_holm, floor))
        if r.kind.startswith("drop"):
            put(f"{key}.loss", f3(-r.estimate)); put(f"{key}.losslo", f3(-r.ci_hi)); put(f"{key}.losshi", f3(-r.ci_lo))
    js = json.load(open(RV / f"{step}_results.json"))
    for k_, v_ in js["cpm"].items():
        put(f"rv.{step}.cpm.{k_.replace('|', '.')}", f3(v_))
    for k_, v_ in js["cands_per_scan"].items():
        put(f"rv.{step}.cps.{k_.replace('|', '.')}", f"{v_:.1f}")
    if step == "inplane":
        for r in js["loss_ratio"]:
            put(f"rv.inplane.ratio.{r['model']}.{r['inplane']}", f"{r['ratio']:.2f}"); put(f"rv.inplane.ratio.{r['model']}.{r['inplane']}.lo", f"{r['ci_lo']:.2f}")
            put(f"rv.inplane.ratio.{r['model']}.{r['inplane']}.hi", f"{r['ci_hi']:.2f}")
        if js["xycal_mm"] is not None:
            put("rv.xycal.mm", f"{js['xycal_mm']:.2f}")
    else:
        put("rv.levels.wording", js["wording"])
        for c_, g in js["level"].items():
            put(f"rv.levels.lvl.{c_}", f"{g['achieved_median']:.0f}"); put(f"rv.levels.tgt.{c_}", f"{g['target_median']:.0f}"); put(f"rv.levels.lvl.{c_}.nout", g["n_outside_3pct"]); put(f"rv.levels.lvl.{c_}.maxerr", f"{100 * g['max_abs_rel_err']:.1f}")
if (RV / "nlst_pairs_results.csv").exists():
    nr = pd.read_csv(RV / "nlst_pairs_results.csv"); nj = json.load(open(RV / "nlst_pairs_results.json"))
    for r in nr.itertuples():
        key = "rv.nlst." + {"F7 delta agreement S4-S3 (mean of fold models)": "f7", "fold 0 only: delta agreement S4-S3": "f7f0"}.get(r.quantity, kslug(r.quantity))
        fmt = s3 if key.endswith(("f7", "f7f0")) or "score" in key else (lambda x: f"{x:.2f}") if "per_scan" in key else f3
        put(f"{key}.est", fmt(r.estimate)); put(f"{key}.lo", fmt(r.ci_lo)); put(f"{key}.hi", fmt(r.ci_hi))
        if not pd.isna(getattr(r, "p_two_sided", np.nan)):
            put(f"{key}.p", pv(r.p_two_sided, floor))
    put("rv.nlst.n", nj["n_exams"]); put("rv.nlst.nsample", nj["n_sample"]); put("rv.nlst.ndrop", nj["n_geometry_dropped"]); put("rv.nlst.wording", nj["wording"])
    for v_, n_ in nj["vendors"].items():
        put(f"rv.nlst.nv.{v_}", n_)
    for r in nj["by_vendor"]:
        put(f"rv.nlst.agree.{r['vendor']}.{r['model']}", f3(r["agreement"]))
    cal = nj["calibration"]; put("rv.cal.med", f"{cal['overall_median_mm']:.2f}"); put("rv.cal.q1", f"{cal['iqr_mm'][0]:.2f}"); put("rv.cal.q3", f"{cal['iqr_mm'][1]:.2f}")
    put("rv.cal.n", cal["n"]); put("rv.cal.nedge", cal["n_edge"])
    for v_, g in cal["by_vendor"].items():
        put(f"rv.cal.{v_}.med", f"{g['median']:.2f}"); put(f"rv.cal.{v_}.q1", f"{g['q1']:.2f}"); put(f"rv.cal.{v_}.q3", f"{g['q3']:.2f}")
if (RV / "nlst_pairs_vendor.csv").exists():
    for r in pd.read_csv(RV / "nlst_pairs_vendor.csv").itertuples():
        put(f"rv.nv.{r.vendor}.d", s3(r.delta)); put(f"rv.nv.{r.vendor}.lo", s3(r.ci_lo)); put(f"rv.nv.{r.vendor}.hi", s3(r.ci_hi))
        put(f"rv.nv.{r.vendor}.S3", f3(r.agree_S3)); put(f"rv.nv.{r.vendor}.S4", f3(r.agree_S4)); put(f"rv.nv.{r.vendor}.n", r.n_exams)
        fd = [getattr(r, f"delta_f{k}") for k in range(4)]
        put(f"rv.nv.{r.vendor}.fmin", s3(min(fd))); put(f"rv.nv.{r.vendor}.fmax", s3(max(fd))); put(f"rv.nv.{r.vendor}.fpos", sum(x > 0 for x in fd))
if (RV / "nlst_pairs_threshold.csv").exists():                # NOT pre-specified threshold sensitivity of F7
    for r in pd.read_csv(RV / "nlst_pairs_threshold.csv", dtype={"threshold": str}).itertuples():
        t = {"0.3": "t03", "0.5": "t05", "0.7": "t07", "cm": "cm"}[r.threshold]; b = f"rv.nthr.{t}"
        put(f"{b}.d", s3(r.delta)); put(f"{b}.lo", s3(r.ci_lo)); put(f"{b}.hi", s3(r.ci_hi)); put(f"{b}.p", pv(r.p_two_sided, floor))
        put(f"{b}.S3", f3(r.agree_S3)); put(f"{b}.S4", f3(r.agree_S4))
        put(f"{b}.c3", f"{r.conf_per_series_S3:.2f}"); put(f"{b}.c4", f"{r.conf_per_series_S4:.2f}")
        put(f"{b}.pers3", f3(r.persistence_S3)); put(f"{b}.pers4", f3(r.persistence_S4))
        fd = [getattr(r, f"delta_f{k}") for k in range(4)]; put(f"{b}.fmin", s3(min(fd))); put(f"{b}.fmax", s3(max(fd))); put(f"{b}.fpos", sum(x > 0 for x in fd))
        if t == "cm":
            th = [float(x) for x in r.thr_S4_folds.split()]; put("rv.nthr.cm.thrmin", f"{min(th):.2f}"); put("rv.nthr.cm.thrmax", f"{max(th):.2f}")
if (RV / "levels_sens_within3pct.csv").exists():
    for r in pd.read_csv(RV / "levels_sens_within3pct.csv").itertuples():
        put(f"rv.lsens.{r.model}.est", s3(r.estimate)); put(f"rv.lsens.{r.model}.lo", s3(r.ci_lo)); put(f"rv.lsens.{r.model}.hi", s3(r.ci_hi))
        put(f"rv.lsens.{r.model}.ph", pv(r.p_holm, floor)); V["rv.lsens.n"] = str(r.n_scans); V["rv.lsens.nex"] = str(r.n_excluded)
if (RV / "levels_by_kernel.csv").exists():
    for r in pd.read_csv(RV / "levels_by_kernel.csv").itertuples():
        key = f"rv.lk.{r.kernel}.{kslug(r.kind)}.{r.model}"
        put(f"{key}.est", s3(r.estimate)); put(f"{key}.lo", s3(r.ci_lo)); put(f"{key}.hi", s3(r.ci_hi))
        if r.kind.startswith("drop"):
            put(f"{key}.loss", f3(-r.estimate))
        V[f"rv.lk.{r.kernel}.n"] = str(r.n_scans); V[f"rv.lk.{r.kernel}.tgt"] = f"{r.target_x2_median_hu:.0f}"; V[f"rv.lk.{r.kernel}.nout"] = str(r.n_outside_3pct_nps_x2)
        if hasattr(r, "shortfall_nps_x2_median"):
            V[f"rv.lk.{r.kernel}.short"] = f"{100 * r.shortfall_nps_x2_median:.1f}"; V[f"rv.lk.{r.kernel}.short90"] = f"{100 * r.shortfall_nps_x2_p90:.1f}"
if (RV / "seed_contrasts.csv").exists():
    for r in pd.read_csv(RV / "seed_contrasts.csv").itertuples():
        key = f"rv.seed.{kslug(r.kind)}.{r.model.replace(' ', '')}"
        put(f"{key}.est", s3(r.estimate)); put(f"{key}.lo", s3(r.ci_lo)); put(f"{key}.hi", s3(r.ci_hi))
    sj = json.load(open(RV / "seed_results.json"))
    for k_, v_ in sj["cpm"].items():
        put(f"rv.seed.cpm.{k_.replace('|', '.')}", f3(v_))
    for m in ("S3", "S4"):
        put(f"rv.seed.absclean.{m}", f"{sj['seed_abs_diff_clean'][m]:.3f}"); put(f"rv.seed.absmed.{m}", f"{sj['seed_abs_diff_median_all_conditions'][m]:.3f}")
        put(f"rv.seed.absmax.{m}", f"{sj['seed_abs_diff_max_all_conditions'][m]:.3f}")
    put("rv.seed.replicated", "yes" if all(sj["replicated_blur_benefit"].values()) else "NO"); put("rv.seed.senssame", "yes" if sj["sensitivity_same_significance"] else "NO")
    for r in pd.read_csv(RV / "seed_sensitivity_f1f3.csv").itertuples():
        key = "rv.sens." + ("F3DD" if r.condition.startswith("DD") else "F3Dlow" if r.condition.startswith("D_low") else kslug(r.condition))
        put(f"{key}.est", s3(r.estimate)); put(f"{key}.lo", s3(r.ci_lo)); put(f"{key}.hi", s3(r.ci_hi))
        if not pd.isna(r.p_holm):
            put(f"{key}.ph", pv(r.p_holm, floor))

# ---------------------------------------------------------------- write
with open(HERE / "numbers.tex", "w", encoding="utf-8", newline="\n") as fh:
    fh.write("% GENERATED by paper/make_numbers.py from the result files -- do not edit by hand\n")
    fh.write("\\makeatletter\n\\newcommand{\\val}[1]{\\ifcsname pv@#1\\endcsname\\csname pv@#1\\endcsname"
             "\\else\\textbf{??\\detokenize{#1}}\\PackageWarning{numbers}{unknown value #1}\\fi}\n")
    import re
    for k in list(V):
        if re.fullmatch(r"\d{4,}", V[k]): V[k] = f"{int(V[k]):,}".replace(",", "{,}")
    for k, v_ in V.items(): fh.write(f"\\expandafter\\def\\csname pv@{k}\\endcsname{{{v_}}}\n")
    fh.write("\\makeatother\n")
print(f"wrote {len(V)} values to", HERE / "numbers.tex")
print("miss categories:", mr.category.value_counts().to_dict())


# ---------------------------------------------------------------- supplementary tables (generated LaTeX)
LAB = {"clean": "clean", "noise20": "white 20 HU", "noise40": "white 40 HU", "noise60": "white 60 HU", "noise80": "white 80 HU",
       "noise100": "white 100 HU", "soft0.5": "3D blur 0.35 mm", "soft1.0": "3D blur 0.70 mm", "soft1.5": "3D blur 1.05 mm",
       "sharp0.5": "unsharp 0.5", "sharp1.0": "unsharp 1.0", "sharp1.5": "unsharp 1.5", "sharp1.0+noise60": "unsharp 1.0 + white 60 HU",
       "soft1.0+noise60": "3D blur 0.70 mm + white 60 HU", "cnoise40_c1": "correlated 40 HU, 0.7 mm", "cnoise80_c1": "correlated 80 HU, 0.7 mm",
       "cnoise40_c2": "correlated 40 HU, 1.4 mm", "cnoise80_c2": "correlated 80 HU, 1.4 mm", "proj0.5": "projection, 50 \\% dose",
       "proj0.25": "projection, 25 \\% dose", "proj0.1": "projection, 10 \\% dose"}
FAM = {"F1": "F1", "F2": "F2", "secondary": "sec."}
L = []
L.append("\\begin{sidewaystable}\n\\caption{All 21 LUNA16 test conditions, pooled over the four test folds (356 scans, 487 nodules). "
         "CPM with 95\\% bootstrap CI; $\\Delta$CPM = S4 $-$ S3 with paired bootstrap 95\\% CI; P two-sided bootstrap (floor 0.0002); "
         "P$_\\mathrm{Holm}$ within the pre-specified family; last columns: CPM change from clean and per-fold $\\Delta$CPM (folds 0--3)."
         + (f" $^\\dagger$Not significant when the fold-0 S3 model was replaced by its seed-1 retrain ($\\Delta$CPM {V['rv.sens.cnoise80_c2.est']}, P$_\\mathrm{{Holm}}$ = {V['rv.sens.cnoise80_c2.ph']})." if "rv.sens.cnoise80_c2.ph" in V else "")
         + "}\\label{tab:s_all}\n"
         "\\tiny\\setlength{\\tabcolsep}{2pt}\n\\begin{tabular*}{\\textheight}{@{\\extracolsep\\fill}llcccccccc}\n\\toprule\n"
         "Condition & Fam. & CPM S3 & CPM S4 & $\\Delta$CPM [95\\% CI] & P & P$_\\mathrm{Holm}$ & Change S3 & Change S4 & $\\Delta$ per fold \\\\\n\\midrule\n")
for r in d.itertuples():
    c = r.condition; ch = lambda m: "--" if c == "clean" else V[f"drop.{m}.{c}.est"]
    pff = " / ".join(f"{x:+.3f}".replace("-", "$-$") for x in pf[pf.condition == c].sort_values("fold").delta)
    L.append(f"{LAB[c]}{'$^\\dagger$' if c == 'cnoise80_c2' else ''} & {FAM[r.family]} & {V[f'c.S3.{c}.cpm']} & {V[f'c.S4.{c}.cpm']} & {V[f'd.{c}.est']} [{V[f'd.{c}.lo']}, {V[f'd.{c}.hi']}] & "
             f"{V[f'd.{c}.p']} & {V[f'd.{c}.ph']} & {ch('S3')} & {ch('S4')} & {pff} \\\\\n")
L.append("\\botrule\n\\end{tabular*}\n\\end{sidewaystable}\n\n")
L.append("\\begin{sidewaystable}\n\\caption{NLST descriptive analysis by vendor, kernel class and screening round (T0--T2): candidates per scan "
         "(score $\\geq$ 0.02) for S3 and S4 and their difference with a 95\\% bootstrap CI over patients; median candidate score. No P values (pre-specified).}"
         "\\label{tab:s_nlst}\n\\scriptsize\n\\begin{tabular*}{\\textheight}{@{\\extracolsep\\fill}lllrccccc}\n\\toprule\n"
         "Vendor & Kernel & Round & n & Cand./scan S3 & Cand./scan S4 & $\\Delta$ [95\\% CI] & Median score S3 & Median score S4 \\\\\n\\midrule\n")
cells = st[(st.dataset == "NLST") & (st.vendor != "all") & (st.kernel_class != "all") & (st["round"] != "all")]
for r in cells.sort_values(["vendor", "kernel_class", "round"]).itertuples():
    L.append(f"{'GE' if r.vendor == 'GE' else r.vendor.title()} & {r.kernel_class} & T{r.round} & {int(r.n_series)} & {r.cands_per_scan_S3:.2f} & {r.cands_per_scan_S4:.2f} & "
             f"{s2(r.cands_per_scan_diff)} [{s2(r.cands_per_scan_diff_lo)}, {s2(r.cands_per_scan_diff_hi)}] & {r.score_q50_S3:.3f} & {r.score_q50_S4:.3f} \\\\\n")
L.append("\\botrule\n\\end{tabular*}\n\\end{sidewaystable}\n\n")
CAT = {"ggn": "Faint ground-glass or subsolid", "wall": "Juxta-pleural, mediastinal or diaphragmatic", "paren": "In abnormal lung (fibrosis, emphysema)",
       "vessel": "Vessel-attached or vessel mimic", "near": "Localisation near-miss", "noise": "Severe noise or streak artefact", "unexpl": "Conspicuous solid, unexplained"}
m2 = mr.merge(lo[["miss_id", "best_score_within_r"]], on="miss_id")
L.append("\\begin{table}[h]\n\\caption{Reference nodules missed by the ten LUNA16 fold detectors of the augmented (S4) configuration at the operating threshold 0.02, "
         "by primary cause from visual review; last column: nodules that receive a box within their radius when the threshold is lowered to 0.001.}"
         "\\label{tab:s_miss}\n\\footnotesize\n\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}p{0.42\\textwidth}cccc}\n\\toprule\n"
         "Primary cause & Misses & $\\geq$10 mm & Median (mm) & At 0.001 \\\\\n\\midrule\n")
for k in m2.category.value_counts().index:
    g = m2[m2.category == k]
    L.append(f"{CAT[k]} & {len(g)} & {(g.diameter_mm >= 10).sum()} & {g.diameter_mm.median():.1f} & {np.isfinite(g.best_score_within_r).sum()} \\\\\n")
L.append(f"\\midrule\nTotal & {len(m2)} & {(m2.diameter_mm >= 10).sum()} & {m2.diameter_mm.median():.1f} & {np.isfinite(m2.best_score_within_r).sum()} \\\\\n"
         "\\botrule\n\\end{tabular*}\n\\end{table}\n")
SUBLAB = {"kernel soft": "Native soft kernel", "kernel sharp": "Native sharp kernel", "vendor GE": "GE", "vendor PHILIPS": "Philips",
          "vendor SIEMENS": "Siemens", "vendor TOSHIBA": "Toshiba"}
L.append("\\begin{sidewaystable}\n\\caption{Results by the native reconstruction kernel and the manufacturer of the LUNA16 test scan (pre-specified secondary analysis; "
         "kernel class from the DICOM kernel name). CPM of S3 and S4, $\\Delta$CPM = S4 $-$ S3 with paired bootstrap 95\\% CI within the subset. "
         "Subsets differ in patients, scanners and slice thickness, so CPMs are not comparable between rows; the Toshiba subset (one scan) is too small to interpret.}"
         "\\label{tab:s_sub}\n\\tiny\\setlength{\\tabcolsep}{1.5pt}\n\\begin{tabular*}{\\textheight}{@{\\extracolsep\\fill}lrcccccc}\n\\toprule\n"
         "Subset & Scans & Clean S3 / S4 & $\\Delta$ clean & 3D blur " + V["blur.1.0.xy"] + "\\,mm $\\Delta$ & 3D blur " + V["blur.1.5.xy"] + "\\,mm $\\Delta$ & Corr. 80\\,HU, 1.4\\,mm $\\Delta$ & Proj. 10\\% $\\Delta$ \\\\\n\\midrule\n")
for sub_ in SUBLAB:
    g = bs[bs.subset == sub_].set_index("condition"); ci = lambda c: f"{s3(g.loc[c, 'delta'])} [{s3(g.loc[c, 'delta_lo'])}, {s3(g.loc[c, 'delta_hi'])}]"
    L.append(f"{SUBLAB[sub_]} & {int(g.loc['clean', 'n_scans'])} & {f3(g.loc['clean', 'cpm_S3'])} / {f3(g.loc['clean', 'cpm_S4'])} & {ci('clean')} & {ci('soft1.0')} & "
             f"{ci('soft1.5')} & {ci('cnoise80_c2')} & {ci('proj0.1')} \\\\\n")
L.append("\\botrule\n\\end{tabular*}\n\\end{sidewaystable}\n\n")
L.append("\\begin{sidewaystable}\n\\caption{Sensitivity analysis (not pre-specified): difference in degradation from clean, "
         "$\\Delta\\Delta$ = [CPM$_\\mathrm{S4}$(condition) $-$ CPM$_\\mathrm{S4}$(clean)] $-$ [CPM$_\\mathrm{S3}$(condition) $-$ CPM$_\\mathrm{S3}$(clean)], "
         "which removes each fold pair's difference on clean scans (per-fold clean $\\Delta$CPM " + V["clean.fmin"] + " to " + V["clean.fmax"] + "). "
         "Pooled over folds 0--3 with paired bootstrap 95\\% CI (same resamples as Table~S1) and Holm correction within the families of the main analysis; last column: per-fold values.}"
         "\\label{tab:s_dd}\n\\tiny\\setlength{\\tabcolsep}{2pt}\n\\begin{tabular*}{\\textheight}{@{\\extracolsep\\fill}llcccc}\n\\toprule\n"
         "Condition & Fam. & $\\Delta\\Delta$ [95\\% CI] & P & P$_\\mathrm{Holm}$ & $\\Delta\\Delta$ per fold \\\\\n\\midrule\n")
for r in ddf.itertuples():
    pff = " / ".join(s3(getattr(r, f"fold{k}")) for k in range(4))
    L.append(f"{LAB[r.condition]} & {FAM[r.family]} & {V[f'dd.{r.condition}.est']} [{V[f'dd.{r.condition}.lo']}, {V[f'dd.{r.condition}.hi']}] & {pv(r.p, floor)} & "
             f"{V[f'dd.{r.condition}.ph']} & {pff} \\\\\n")
L.append("\\botrule\n\\end{tabular*}\n\\end{sidewaystable}\n\n")
TXL = {"tex_nps": "real NPS", "tex_gauss": "Gaussian grain", "tex_white": "white", "proj0.1": "projection 10\\%"}
L.append("\\begin{sidewaystable}\n\\caption{Noise-texture test, all scans and by native kernel class. Each model's CPM loss from clean (95\\% CI) under stationary noise of three textures, "
         "all at the per-scan level of the projection-domain noise at 10\\% dose and inserted on the same path, and under the projection-domain noise itself; "
         "last rows: CPM contrasts (F4 primary, equivalence check pre-specified, white $-$ real NPS secondary). Real NPS: B30f spectrum for soft-kernel and B50f spectrum for sharp-kernel scans.}\\label{tab:s_tex}\n\\tiny\\setlength{\\tabcolsep}{1.5pt}\n"
         "\\begin{tabular*}{\\textheight}{@{\\extracolsep\\fill}lcccccc}\n\\toprule\n & \\multicolumn{2}{c}{All (" + V["luna.nscan"] + " scans)} & \\multicolumn{2}{c}{Native soft (" + V["sub.kernel_soft.n"] + ")} & \\multicolumn{2}{c}{Native sharp (" + V["sub.kernel_sharp.n"] + ")} \\\\\n"
         "Quantity & S3 & S4 & S3 & S4 & S3 & S4 \\\\\n\\midrule\n")
def txv(sub, kind, m, loss=False):
    k = f"tex.{sub}.{kind}.{m}"
    return f"{V[k + '.loss']} [{V[k + '.losslo']}, {V[k + '.losshi']}]" if loss else f"{V[k + '.est']} [{V[k + '.lo']}, {V[k + '.hi']}]"
for c in ("tex_nps", "tex_gauss", "tex_white", "proj0.1"):
    L.append(f"Loss, {TXL[c]} & " + " & ".join(txv(s, f"drop_{c}", m, True) for s in ("all", "soft", "sharp") for m in ("S3", "S4")) + " \\\\\n")
L.append("\\midrule\n")
for kind, lab in (("F4_nps-gauss", "Real NPS $-$ Gaussian grain"), ("equiv_nps-proj0.1", "Real NPS $-$ projection"), ("white-nps", "White $-$ real NPS")):
    L.append(f"{lab} & " + " & ".join(txv(s, kind, m) for s in ("all", "soft", "sharp") for m in ("S3", "S4")) + " \\\\\n")
L.append("\\botrule\n\\end{tabular*}\n\\end{sidewaystable}\n")
# ---- review experiments (lung-nodule-detector-analysis-plans/plans_original/review_prespec.md)
def vv(k, loss=False):
    return f"{V[k + '.loss']} [{V[k + '.losslo']}, {V[k + '.losshi']}]" if loss else f"{V[k + '.est']} [{V[k + '.lo']}, {V[k + '.hi']}]"
if (RV / "inplane_contrasts.csv").exists():
    L.append("\\begin{table}[h]\n\\caption{Gaussian blur in 3D and in-plane only (LUNA16, " + V["luna.nscan"] + " scans). The 3D blur is isotropic in voxels "
             "($\\sigma$ 1.0 voxel = " + V["blur.1.0.xy"] + "\\,mm in-plane and " + V["blur.1.0.z"] + "\\,mm along $z$); the in-plane blur has the same in-plane $\\sigma$ and none along $z$. "
             "0.50\\,mm in-plane: median blur that turns the sharp into the soft reconstruction of the same NLST exam. CPM loss from clean and $\\Delta$CPM = S4 $-$ S3 with paired bootstrap 95\\% CI; "
             "P$_\\mathrm{Holm}$: 3D rows family F1, in-plane rows at 0.70 and 1.05\\,mm family F5 (pre-specified, Holm, 2 comparisons).}\\label{tab:s_xy}\n\\tiny\\setlength{\\tabcolsep}{1.5pt}\n"
             "\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}lcccc}\n\\toprule\nBlur & Loss S3 & Loss S4 & $\\Delta$CPM [95\\% CI] & P$_\\mathrm{Holm}$ \\\\\n\\midrule\n")
    for c, lab in (("soft1.0", f"3D, {V['blur.1.0.xy']}\\,mm in-plane"), ("soft1.5", f"3D, {V['blur.1.5.xy']}\\,mm in-plane"), ("xycal", f"in-plane, {V['rv.xycal.mm']}\\,mm"),
                   ("xy1.0", f"in-plane, {V['blur.1.0.xy']}\\,mm (F5)"), ("xy1.5", f"in-plane, {V['blur.1.5.xy']}\\,mm (F5)")):
        L.append(f"{lab} & {vv(f'rv.inplane.drop_{c}.S3', True)} & {vv(f'rv.inplane.drop_{c}.S4', True)} & {vv(f'rv.inplane.delta_{c}.S4-S3')} & "
                 f"{V.get(f'rv.inplane.delta_{c}.S4-S3.ph', V.get(f'd.{c}.ph', '--'))} \\\\\n")
    L.append("\\botrule\n\\end{tabular*}\n\\end{table}\n\n")
if (RV / "levels_contrasts.csv").exists():
    L.append("\\begin{sidewaystable}\n\\caption{Noise texture and level. Stationary noise of three textures at 0.5, 1 and 2 times the per-scan level of projection-domain noise at 10\\% dose, "
             "inserted on the same path (" + V["luna.nscan"] + " scans; the same noise fields scaled). CPM loss from clean and $\\Delta$CPM = S4 $-$ S3, paired bootstrap 95\\% CI. "
             "Achieved level: median SD added over the lung crop and number of scans outside $\\pm$3\\% of the target (the clip to the HU window limits the level at $\\times$2). "
             "Last rows: pre-specified F6 contrast (Holm, 2 comparisons), the secondary white contrast, and, not pre-specified, the F6 contrast restricted to scans within $\\pm$3\\% and by native kernel class (real NPS: B50f spectrum on sharp-, B30f spectrum on soft-kernel scans).}\\label{tab:s_lev}\n"
             "\\tiny\\setlength{\\tabcolsep}{2pt}\n\\begin{tabular*}{\\textheight}{@{\\extracolsep\\fill}lccccc}\n\\toprule\n"
             "Texture, level & Achieved SD (HU) & Outside $\\pm$3\\% & Loss S3 & Loss S4 & $\\Delta$CPM [95\\% CI] \\\\\n\\midrule\n")
    for c in ("tex_nps", "tex_white", "tex_gauss"):
        for f in ("0.5", "1", "2"):
            k = f"{c}_x{f}"; lv_ = f"{V.get(f'rv.levels.lvl.{k}', V.get(f'tex.lvl.{c}.med', '--'))}" if f != "1" else V.get(f"tex.lvl.{c}.med", "--")
            no = V.get(f"rv.levels.lvl.{k}.nout", "--") if f != "1" else V.get(f"tex.lvl.{c}.nout", "--")
            L.append(f"{TXL[c]}, $\\times${f} & {lv_} & {no} & {vv(f'rv.levels.drop_{k}.S3', True)} & {vv(f'rv.levels.drop_{k}.S4', True)} & {vv(f'rv.levels.delta_{k}.S4-S3')} \\\\\n")
    L.append("\\midrule\n")
    for m in ("S3", "S4"):
        L.append(f"\\multicolumn{{5}}{{l}}{{F6 (pre-specified), {m}: real NPS $\\times$2 $-$ Gaussian grain $\\times$0.5; P$_\\mathrm{{Holm}}$ {V[f'rv.levels.F6_nps_x2-gauss_x0.5.{m}.ph']}}} & {vv(f'rv.levels.F6_nps_x2-gauss_x0.5.{m}')} \\\\\n")
    for m in ("S3", "S4"):
        L.append(f"\\multicolumn{{5}}{{l}}{{Secondary, {m}: white $\\times$2 $-$ Gaussian grain $\\times$0.5}} & {vv(f'rv.levels.white_x2-gauss_x0.5.{m}')} \\\\\n")
    if "rv.lsens.n" in V:
        for m in ("S3", "S4"):
            L.append(f"\\multicolumn{{5}}{{l}}{{F6 contrast, {m}, {V['rv.lsens.n']} scans within $\\pm$3\\% of the target (not pre-specified)}} & {vv(f'rv.lsens.{m}')} \\\\\n")
    if "rv.lk.soft.n" in V:
        for k_ in ("sharp", "soft"):
            for m in ("S3", "S4"):
                L.append(f"\\multicolumn{{5}}{{l}}{{F6 contrast, {m}, native {k_} kernel ({V[f'rv.lk.{k_}.n']} scans; median target at $\\times$2 {V[f'rv.lk.{k_}.tgt']}\\,HU; not pre-specified)}} & {vv(f'rv.lk.{k_}.F6_nps_x2-gauss_x0.5.{m}')} \\\\\n")
    L.append("\\botrule\n\\end{tabular*}\n\\end{sidewaystable}\n\n")
if (RV / "nlst_pairs_vendor.csv").exists():
    L.append("\\begin{table}[h]\n\\caption{NLST exams reconstructed with a soft and a sharp kernel (same acquisition, same voxel grid; " + V["rv.nlst.n"] + " exams). "
             "Agreement = 2 $\\times$ matched / (confident soft + confident sharp), confident = score $\\geq$ 0.5, match = centres within 5\\,mm; mean of the four fold models "
             "with a bootstrap 95\\% CI over exams. First row: pre-specified F7 contrast; other rows secondary. Count-matched: each S4 fold model thresholded so that it makes as many confident detections over all series as the S3 model of the same fold at 0.5. Calibrated blur: in-plane Gaussian $\\sigma$ that best turns the sharp into the soft "
             "reconstruction (median [IQR]).}\\label{tab:s_nlstk}\n\\tiny\\setlength{\\tabcolsep}{1.5pt}\n"
             "\\begin{tabular*}{\\textwidth}{@{\\extracolsep\\fill}lrccccc}\n\\toprule\n"
             "Vendor & Exams & S3 & S4 & $\\Delta$ [95\\% CI] & Fold range & Blur (mm) \\\\\n\\midrule\n")
    for v_ in ("all", "GE", "PHILIPS", "SIEMENS", "TOSHIBA"):
        cal_ = f"{V['rv.cal.med']} [{V['rv.cal.q1']}, {V['rv.cal.q3']}]" if v_ == "all" else f"{V[f'rv.cal.{v_}.med']} [{V[f'rv.cal.{v_}.q1']}, {V[f'rv.cal.{v_}.q3']}]"
        lab = {"all": "All (F7)", "GE": "GE"}.get(v_, v_.title())
        L.append(f"{lab} & {V[f'rv.nv.{v_}.n']} & {V[f'rv.nv.{v_}.S3']} & {V[f'rv.nv.{v_}.S4']} & {V[f'rv.nv.{v_}.d']} [{V[f'rv.nv.{v_}.lo']}, {V[f'rv.nv.{v_}.hi']}] & "
                 f"{V[f'rv.nv.{v_}.fmin']}, {V[f'rv.nv.{v_}.fmax']} & {cal_} \\\\\n")
    if "rv.nthr.t03.d" in V:                          # threshold sensitivity (not pre-specified)
        L.append("\\midrule\n\\multicolumn{7}{l}{All exams, other confidence thresholds (not pre-specified)} \\\\\n")
        for t, lab in (("t03", "Threshold 0.3"), ("t05", "Threshold 0.5 (F7)"), ("t07", "Threshold 0.7"), ("cm", "S4 count-matched to S3")):
            L.append(f"{lab} & & {V[f'rv.nthr.{t}.S3']} & {V[f'rv.nthr.{t}.S4']} & {V[f'rv.nthr.{t}.d']} [{V[f'rv.nthr.{t}.lo']}, {V[f'rv.nthr.{t}.hi']}] & "
                     f"{V[f'rv.nthr.{t}.fmin']}, {V[f'rv.nthr.{t}.fmax']} & \\\\\n")
        for t, lab in (("t03", "0.3"), ("t05", "0.5"), ("t07", "0.7")):
            L.append(f"Confident detections per series, {lab} & & {V[f'rv.nthr.{t}.c3']} & {V[f'rv.nthr.{t}.c4']} & & & \\\\\n")
    L.append("\\midrule\n")
    for q, lab in (("persistence_soft->sharp", "Confident on soft, any candidate on sharp"), ("persistence_sharp->soft", "Confident on sharp, any candidate on soft"),
                   ("mean_score_soft-sharp_of_matched", "Score soft $-$ sharp, matched pairs"), ("per_scan_cands_soft", "Candidates per scan, soft"),
                   ("per_scan_cands_sharp", "Candidates per scan, sharp")):
        L.append(f"{lab} & & {V[f'rv.nlst.{q}_S3.est']} & {V[f'rv.nlst.{q}_S4.est']} & & & \\\\\n")
    L.append("\\botrule\n\\end{tabular*}\n\\end{table}\n\n")
if (RV / "seed_contrasts.csv").exists():
    sc = pd.read_csv(RV / "seed_contrasts.csv"); conds_ = [k[len("seed diff "):] for k in sc.kind.unique() if k.startswith("seed diff ")]
    L.append("\\begin{sidewaystable}\n\\caption{Second training seed, fold 0 (" + V["luna.nscan.f0"] + " test scans). CPM of the seed-0 and seed-1 models of S3 and S4, the seed difference (seed 1 $-$ seed 0) and "
             "$\\Delta$CPM = S4 $-$ S3 for each seed, with paired bootstrap 95\\% CIs (pre-specified secondary analysis of the second training seed; Methods).}\\label{tab:s_seed}\n\\tiny\\setlength{\\tabcolsep}{2pt}\n"
             "\\begin{tabular*}{\\textheight}{@{\\extracolsep\\fill}lcccccccc}\n\\toprule\n"
             "Condition & S3 s0 & S3 s1 & S4 s0 & S4 s1 & Seed diff. S3 & Seed diff. S4 & $\\Delta$CPM seed 0 & $\\Delta$CPM seed 1 \\\\\n\\midrule\n")
    for c in conds_:
        XLAB = {"tex_nps": "real-NPS noise, $\\times$1", "tex_gauss": "Gaussian grain, $\\times$1", "tex_white": "white noise (projection path), $\\times$1",
                "xy1.0": f"in-plane blur {V['blur.1.0.xy']} mm", "xy1.5": f"in-plane blur {V['blur.1.5.xy']} mm", "xycal": f"in-plane blur {V['rv.xycal.mm']} mm (NLST-calibrated)"}
        kc = kslug(c); lab = LAB.get(c, XLAB.get(c, c.replace("_", " ")))
        L.append(f"{lab} & {V[f'rv.seed.cpm.S3s0.{c}']} & {V[f'rv.seed.cpm.S3s1.{c}']} & {V[f'rv.seed.cpm.S4s0.{c}']} & {V[f'rv.seed.cpm.S4s1.{c}']} & "
                 f"{vv(f'rv.seed.seed_diff_{kc}.S3')} & {vv(f'rv.seed.seed_diff_{kc}.S4')} & {vv(f'rv.seed.delta_{kc}.S4-S3s0')} & {vv(f'rv.seed.delta_{kc}.S4-S3s1')} \\\\\n")
    L.append("\\botrule\n\\end{tabular*}\n\\end{sidewaystable}\n\n")
with open(HERE / "supp_tables.tex", "w", encoding="utf-8", newline="\n") as fh:
    fh.write("% GENERATED by paper/make_numbers.py -- do not edit by hand\n" + "".join(L))
print("wrote", HERE / "supp_tables.tex")
