"""S3 vs. S4 on TCIA LDCT chest, full vs. 10 % dose, four fold-model pairs; analysis as pre-specified.

    python scripts/22_tcia_e1.py            # inference (resumable) then scoring
    python scripts/22_tcia_e1.py --score    # scoring only

Volumes: scripts/16_tcia_ldct_preprocess.py (both doses share one crop). Reference: scripts/19_tcia_annotations.py.
Pseudo-series ids <case>_full / <case>_low. Models: S3 / S4 of folds 0-3 (TCIA is outside LUNA16, so every fold's models
are valid; they are replicate models, not independent samples). Statistics (scripts/p2_stats.py): bootstrap over
PATIENTS (both doses of a patient resampled together), 10,000 replicates, all 16 systems on the same resamples.
Primary (F3): DD = mean_k[(S4_k low - S4_k full) - (S3_k low - S3_k full)] and D_low = mean_k[S4_k low - S3_k low].
Primary set (fixed 30 Oct 2026, before any TCIA result): 49 patients (C128 excluded: lesion position unreliable) and the
34 verified lesions; for the 6 lesions with an unknown slice (eval/unresolved.csv) any detection within their radius
in-plane, at any z, is ignored (neither hit nor false positive). Inference runs on all 50 patients.
Secondary: per-fold values, full-dose D, solid lesions only.
"""
from __future__ import annotations
import argparse, json, pickle, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from lungcad import config as C
from lungcad.preprocess import load_volume, load_meta, normalize, voxel_to_world, world_to_voxel
from e1_multifold import fold_models
from e1_dose_robustness import load_model
import p2_stats as S

PRE = C.THESIS_DATA / "tcia_ldct" / "preprocessed"; EVAL = C.THESIS_DATA / "tcia_ldct" / "eval"
OUT = C.WORK / "wp0" / "analysis" / "p2_tcia"; FOLDS = [0, 1, 2, 3]


@torch.no_grad()
def infer(cases):
    part = OUT / "rows_partial.pkl"; st = pickle.load(open(part, "rb")) if part.exists() else {"done": [], "rows": []}
    models = {f"{m}_f{k}": load_model(p) for k in FOLDS for m, p in fold_models(k).items()}
    json.dump({n: {"run_tag": v[2], "best_epoch": v[3]} for n, v in models.items()}, open(OUT / "setup.json", "w"), indent=1)
    todo = [f"{c}_{d}" for c in cases for d in ("full", "low") if f"{c}_{d}" not in set(st["done"])]; t0 = time.time()
    for n, sid in enumerate(todo, 1):
        arr, meta = load_volume(PRE / sid); x = torch.from_numpy(normalize(arr.astype(np.float32)))[None, None]
        sp = np.array(meta["spacing_zyx"])
        for name, (det, cfg, _, _) in models.items():
            with torch.autocast("cuda", dtype=torch.float16, enabled=cfg["train"]["amp"]):
                out = det(x, use_inferer=True)[0]
            b = out[det.target_box_key].float().cpu().numpy(); sc = out[det.pred_score_key].float().cpu().numpy()
            if len(b):
                w = voxel_to_world(meta, (b[:, :3] + b[:, 3:]) / 2)
                st["rows"] += [(name, sid, *w[i], float(sc[i])) for i in range(len(b))]
        st["done"].append(sid)
        pickle.dump(st, open(str(part) + ".tmp", "wb")); Path(str(part) + ".tmp").replace(part)
        print(f"  {n}/{len(todo)} {sid}  {(time.time() - t0) / 60:.0f} min", flush=True)
    pd.DataFrame(st["rows"], columns=["model", "seriesuid", "coordX", "coordY", "coordZ", "probability"]).to_csv(OUT / "tcia_candidates.csv", index=False)


def check_lesions_in_crop():
    """Every reference lesion must lie inside the stored crop of BOTH doses, or preprocessing would decide the result."""
    ann = pd.read_csv(EVAL / "annotations" / "annotations.csv"); bad = []
    for r in ann.itertuples():
        m = load_meta(PRE / r.seriesuid); v = world_to_voxel(m, [[r.coordX, r.coordY, r.coordZ]])[0]
        if not np.all((v >= 0) & (v < np.array(m["shape_zyx"]))):
            bad.append((r.seriesuid, np.round(v, 1).tolist(), m["shape_zyx"]))
    print(f"lesions inside the stored crop: {len(ann) - len(bad)} / {len(ann)}", *bad, sep="\n  ")
    assert not bad, "lesion outside the crop: fix before scoring"


def ignore_unresolved(df: pd.DataFrame) -> pd.DataFrame:
    """Drop candidates within the radius (in-plane, any z) of a lesion whose slice is unknown: the LUNA16 'excluded
    finding' rule applied as a column, so an unlocatable lesion is neither a hit nor a false positive."""
    u = pd.read_csv(EVAL / "unresolved.csv"); u = u[u.status == "slice_unknown"]; keep = np.ones(len(df), bool)
    for r in u.itertuples():
        m = df.seriesuid.str.startswith(r.case + "_").values
        dxy = np.hypot(df.coordX.values - r.coordX, df.coordY.values - r.coordY)
        keep &= ~(m & (dxy <= r.diameter_mm / 2))
    print(f"ignored {int((~keep).sum())} candidates near {len(u)} lesions of unknown slice")
    return df[keep]


def score(cases, solid_only=False):
    df = ignore_unresolved(pd.read_csv(OUT / "tcia_candidates.csv")); ev = EVAL
    if solid_only:                                   # reference restricted to solid lesions; subsolid marks become "excluded" (ignored)
        ck = pd.read_csv(EVAL / "lesions_checked.csv"); ann = pd.read_csv(EVAL / "annotations" / "annotations.csv")
        sub = ck[~ck.diagnosis.str.startswith("Solid")]
        key = lambda d: d.coordX.round(3).astype(str) + d.coordY.round(3).astype(str) + d.coordZ.round(3).astype(str)
        is_sub = key(ann).isin(set(key(sub)))
        ev = EVAL / "solid_only"; (ev / "annotations").mkdir(parents=True, exist_ok=True)
        ann[~is_sub].to_csv(ev / "annotations" / "annotations.csv", index=False); ann[is_sub].to_csv(ev / "annotations" / "annotations_excluded.csv", index=False)
        pd.read_csv(EVAL / "annotations" / "seriesuids.csv", header=None).to_csv(ev / "annotations" / "seriesuids.csv", index=False, header=False)
    clusters = [[f"{c}_full", f"{c}_low"] for c in cases]; allu = [u for cl in clusters for u in cl]
    systems = {}
    for k in FOLDS:
        for m in ("S3", "S4"):
            res = S.scan_results(df[df.model == f"{m}_f{k}"], allu, eval_dir=ev)
            for d in ("full", "low"):          # a dose-specific system sees only its own pseudo-series
                systems[f"{m}_f{k}_{d}"] = {u: r for u, r in res.items() if u.endswith(d)}
    # per-dose clusters for CPM: each system is evaluated on its own dose's uids, resampled by patient
    def cpm_on(sysres, us):
        return S.cpm(sysres, [u for u in us if u in sysres])
    rng = np.random.RandomState(0); B = S.N_BOOT
    point = pd.Series({k: cpm_on(v, allu) for k, v in systems.items()}); boot = np.empty((B, len(systems)))
    for i in range(B):
        idx = rng.randint(0, len(clusters), len(clusters)); us = [u for j in idx for u in clusters[j]]
        boot[i] = [cpm_on(v, us) for v in systems.values()]
    boot = pd.DataFrame(boot, columns=list(systems)); w4 = 1 / len(FOLDS)
    def dd(k):                                   # (S4 low - S4 full) - (S3 low - S3 full) for fold k
        return {f"S4_f{k}_low": 1, f"S4_f{k}_full": -1, f"S3_f{k}_low": -1, f"S3_f{k}_full": 1}
    DD = {key: w4 * wt for k in FOLDS for key, wt in dd(k).items()}
    rows = [{"contrast": "DD (S4 loses less than S3, mean of 4 pairs)", **S.contrast(point, boot, DD)},
            {"contrast": "D_low S4-S3 at 10 %", **S.contrast(point, boot, {**{f"S4_f{k}_low": w4 for k in FOLDS}, **{f"S3_f{k}_low": -w4 for k in FOLDS}})},
            {"contrast": "D_full S4-S3 at full dose", **S.contrast(point, boot, {**{f"S4_f{k}_full": w4 for k in FOLDS}, **{f"S3_f{k}_full": -w4 for k in FOLDS}})}]
    for m in ("S3", "S4"):
        rows.append({"contrast": f"{m} drop full->10 % (mean of 4)", **S.contrast(point, boot, {**{f"{m}_f{k}_low": w4 for k in FOLDS}, **{f"{m}_f{k}_full": -w4 for k in FOLDS}})})
    for k in FOLDS:
        rows.append({"contrast": f"fold {k}: DD", **S.contrast(point, boot, dd(k))})
    res = pd.DataFrame(rows); res["family"] = "secondary"
    if not solid_only:                           # F3 (primary): DD and D_low, Holm within the family (analysis plan, 30 Oct)
        f3 = res.contrast.str.startswith(("DD (", "D_low")); res.loc[f3, "family"] = "F3"
        res.loc[f3, "p_holm"] = S.holm(res.loc[f3, "p_two_sided"]).values
    tag = "_solid" if solid_only else ""
    res.to_csv(OUT / f"tcia_contrasts{tag}.csv", index=False)
    point.rename("cpm").to_csv(OUT / f"tcia_cpm{tag}.csv")
    q = boot.quantile([0.025, 0.975]).T; q.columns = ["ci_lo", "ci_hi"]; q.to_csv(OUT / f"tcia_cpm_ci{tag}.csv")
    print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--score", action="store_true"); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cases = sorted(p.name[:-len("_full.json")] for p in PRE.glob("C*_full.json"))
    assert len(cases) == 50, f"{len(cases)} preprocessed cases"
    check_lesions_in_crop()
    if not a.score:
        infer(cases)                                    # all 50 patients
    excl = set(pd.read_csv(EVAL / "excluded_patients.csv").case)
    primary = [c for c in cases if c not in excl]; assert len(primary) == 49, len(primary)
    score(primary); score(primary, solid_only=True)
