"""Review experiment R3 (lung-nodule-detector-analysis-plans/plans_original/review_prespec.md): real reconstruction kernels in NLST. Two reconstructions (soft, sharp)
of the same acquisition, converted to detector spacing with one crop, scored by all eight fold models.

    python scripts/30_nlst_kernel_pairs.py select              # fixed sample (seed 0, 125 exams per vendor)
    python scripts/30_nlst_kernel_pairs.py convert [--procs 4] # both series of each exam, LUNA16 path, primary's lung box
    python scripts/30_nlst_kernel_pairs.py calibrate [--procs 8]   # in-plane sigma (mm) mapping sharp -> soft, per exam
    python scripts/30_nlst_kernel_pairs.py infer [--folds 0 1 2 3]  # 8 models x 2 reconstructions (GPU, resumable)
    python scripts/30_nlst_kernel_pairs.py check               # known answer: our converter = convert_native on 2 sample series

Eligibility (pre-spec): usable primary archive series (exclusions applied), kernel class soft/sharp; partner = other
class in the same study, same slice thickness, same number of images; thickness <= 2.5 mm; lowest series number if
several; one exam per patient. Output: <analysis>/p2_review/nlst_pairs/; volumes in <nlst>/kernel_pairs/<pid>/<uid>.
"""
from __future__ import annotations
import argparse, json, pickle, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from lungcad import config as C
from lungcad.data.nlst import usable_series, META, NATIVE, native_stem, convert_native
from lungcad.preprocess import (load_meta, load_volume, normalize, load_dicom_series, resample, finalize_volume, save_volume,
                                voxel_to_world, exists as stem_exists)

OUT = C.WORK / "wp0" / "analysis" / "p2_review" / "nlst_pairs"
VOL = NATIVE.parent / "kernel_pairs"
PER_VENDOR, SEED, MAXTHK = 125, 0, 2.5
SIGMAS = np.round(np.arange(0.0, 3.0001, 0.05), 2)                    # mm, pre-specified grid


def select() -> pd.DataFrame:
    u = usable_series(); u = u[u.kernel_class.isin(["soft", "sharp"])]
    st = pd.read_csv(META / "nlst_primary_series.csv", usecols=["SeriesInstanceUID", "StudyInstanceUID"], dtype=str)
    u = u.merge(st.rename(columns={"SeriesInstanceUID": "uid"}), on="uid")
    m = pd.read_csv(META / "nlst_series_manifest.csv", low_memory=False, dtype={"SeriesInstanceUID": str, "StudyInstanceUID": str})
    m = m[m.is_axial_ct == True]
    nf = m.set_index("SeriesInstanceUID").n_files
    p = m[["StudyInstanceUID", "SeriesInstanceUID", "kernel", "kernel_class", "thk_mm", "n_files", "SeriesNumber", "local_path"]].rename(
        columns={"SeriesInstanceUID": "uid_partner", "kernel": "kernel_partner", "kernel_class": "kclass_partner", "thk_mm": "thk_partner",
                 "n_files": "n_partner", "SeriesNumber": "sn_partner", "local_path": "lp_partner"})
    j = u.merge(p, on="StudyInstanceUID")
    j = j[(j.uid_partner != j.uid) & j.kclass_partner.isin(["soft", "sharp"]) & (j.kclass_partner != j.kernel_class)
          & (j.thk_partner == j.thk_mm) & (j.thk_mm <= MAXTHK)]
    j = j[j.uid.map(nf).values == j.n_partner.values]
    j = j.sort_values(["uid", "sn_partner", "uid_partner"]).drop_duplicates("uid")              # lowest series number
    j = j.sample(frac=1.0, random_state=SEED).drop_duplicates("pid")                           # one exam per patient
    s = pd.concat([g.head(PER_VENDOR) for _, g in j.groupby("vendor")])
    s = s.rename(columns={"uid": "uid_primary", "kernel": "kernel_primary", "kernel_class": "kclass_primary", "local_path": "lp_primary"})
    cols = ["pid", "round", "vendor", "StudyInstanceUID", "thk_mm", "stem", "uid_primary", "kernel_primary", "kclass_primary", "lp_primary",
            "uid_partner", "kernel_partner", "kclass_partner", "lp_partner"]
    return s[cols].sort_values(["vendor", "pid"]).reset_index(drop=True)


def out_stem(pid, uid) -> Path:
    return VOL / str(pid) / uid


def convert_with_box(ref_stem, dicom_dir, uid, out, spacing=C.LUNA_SPACING) -> str:
    """lungcad.data.nlst.convert_native, but the lung box comes from ref_stem (the primary series of the same exam)."""
    if stem_exists(out):
        return "skip"
    ref = load_meta(ref_stem)
    vol = load_dicom_series(dicom_dir, uid)
    if vol.uid != uid:
        raise RuntimeError(f"series UID mismatch {vol.uid} vs {uid}")
    res = resample(vol, spacing)
    D = np.array(res.direction).reshape(3, 3); sp_xyz = np.array(spacing, float)[::-1]
    a_lo = np.array(ref["origin_xyz"], float)
    a_hi = a_lo + D @ ((np.array(ref["crop_stop_zyx"]) - np.array(ref["crop_offset_zyx"]))[::-1] * np.array(ref["spacing_zyx"])[::-1])
    lo = D.T @ (a_lo - np.array(res.origin)) / sp_xyz; hi = D.T @ (a_hi - np.array(res.origin)) / sp_xyz
    m = ref["margin_mm"] / sp_xyz
    lo, hi = np.floor(np.minimum(lo, hi) + m).astype(int)[::-1], np.ceil(np.maximum(lo, hi) - m).astype(int)[::-1]
    shp = np.array(res.hu.shape); lo = np.clip(lo, 0, shp - 1); hi = np.clip(hi, lo + 1, shp)
    mask = np.zeros(res.hu.shape, np.uint8); mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = 1
    arr, meta = finalize_volume(vol, res, mask, "archive_lung_box_of_primary", spacing)
    meta.update({"domain": "nlst_kernel_pair", "archive_stem": str(ref_stem)})
    out.parent.mkdir(parents=True, exist_ok=True); save_volume(out, arr, meta)
    return f"ok {arr.shape}"


def conv_job(r):
    msgs = []
    for uid, lp in ((r["uid_primary"], r["lp_primary"]), (r["uid_partner"], r["lp_partner"])):
        try:
            msgs.append(convert_with_box(r["stem"], lp, uid, out_stem(r["pid"], uid)))
        except Exception as e:
            msgs.append(f"FAIL {type(e).__name__}: {e}")
    return r["pid"], msgs


def geometry_check(s: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in s.itertuples():
        a, b = out_stem(r.pid, r.uid_primary), out_stem(r.pid, r.uid_partner)
        if not (stem_exists(a) and stem_exists(b)):
            rows.append((r.pid, False, "conversion failed")); continue
        ma, mb = load_meta(a), load_meta(b)
        same = (ma["resampled_shape_zyx"] == mb["resampled_shape_zyx"] and ma["crop_offset_zyx"] == mb["crop_offset_zyx"]
                and ma["crop_stop_zyx"] == mb["crop_stop_zyx"] and np.allclose(ma["origin_xyz"], mb["origin_xyz"], atol=1e-3)
                and np.allclose(ma["spacing_zyx"], mb["spacing_zyx"]))
        rows.append((r.pid, bool(same), "" if same else f"shape {ma['resampled_shape_zyx']} vs {mb['resampled_shape_zyx']}, "
                     f"origin {np.round(ma['origin_xyz'], 2)} vs {np.round(mb['origin_xyz'], 2)}"))
    return pd.DataFrame(rows, columns=["pid", "geometry_ok", "detail"])


def cal_job(r):
    from scipy.ndimage import gaussian_filter
    soft_uid, sharp_uid = (r["uid_primary"], r["uid_partner"]) if r["kclass_primary"] == "soft" else (r["uid_partner"], r["uid_primary"])
    s, ms = load_volume(out_stem(r["pid"], soft_uid)); h, _ = load_volume(out_stem(r["pid"], sharp_uid))
    s = normalize(s.astype(np.float32))[::4]; h = normalize(h.astype(np.float32))[::4]           # every 4th slice
    sp = ms["spacing_zyx"][1]
    mse = [float(((gaussian_filter(h, (0, sg / sp, sg / sp), mode="reflect") if sg > 0 else h) - s).__pow__(2).mean()) for sg in SIGMAS]
    return {"pid": r["pid"], "vendor": r["vendor"], "sigma_mm": float(SIGMAS[int(np.argmin(mse))]), "mse_min": min(mse), "mse_0": mse[0],
            "edge": bool(np.argmin(mse) in (0, len(SIGMAS) - 1)), "n_slices": int(s.shape[0])}


@torch.no_grad()
def infer(s: pd.DataFrame, folds):
    from e1_multifold import fold_models
    from e1_dose_robustness import load_model
    ok = pd.read_csv(OUT / "geometry_check.csv"); ok = set(ok[ok.geometry_ok].pid.astype(str))
    s = s[s.pid.astype(str).isin(ok)]
    series = [(str(r.pid), u) for r in s.itertuples() for u in (r.uid_primary, r.uid_partner)]
    part = OUT / "rows_partial.pkl"; st = pickle.load(open(part, "rb")) if part.exists() else {"done": [], "rows": []}
    models = {f"{m}_f{k}": load_model(p) for k in folds for m, p in fold_models(k).items()}
    json.dump({n: {"run_tag": v[2], "best_epoch": v[3]} for n, v in models.items()} | {"n_series": len(series), "folds": folds},
              open(OUT / "infer_setup.json", "w"), indent=1)
    done = set(st["done"]); todo = [x for x in series if x[1] not in done]; t0 = time.time()
    print(f"{len(done)} done, {len(todo)} to go, models {list(models)}", flush=True)
    for n, (pid, uid) in enumerate(todo, 1):
        arr, meta = load_volume(out_stem(pid, uid)); x = torch.from_numpy(normalize(arr.astype(np.float32)))[None, None]
        sp = np.array(meta["spacing_zyx"])
        for name, (det, cfg, _, _) in models.items():
            with torch.autocast("cuda", dtype=torch.float16, enabled=cfg["train"]["amp"]):
                o = det(x, use_inferer=True)[0]
            b = o[det.target_box_key].float().cpu().numpy(); sc = o[det.pred_score_key].float().cpu().numpy()
            if len(b):
                ctr = (b[:, :3] + b[:, 3:]) / 2; size = (b[:, 3:] - b[:, :3]) * sp; w = voxel_to_world(meta, ctr)
                st["rows"] += [(name, pid, uid, *w[i], float(sc[i]), *size[i]) for i in range(len(b))]
        st["done"].append(uid)
        if n % 10 == 0 or n == len(todo):
            pickle.dump(st, open(str(part) + ".tmp", "wb")); Path(str(part) + ".tmp").replace(part)
            el = time.time() - t0; print(f"  {n}/{len(todo)}  {el/60:.0f} min  ETA {el/n*(len(todo)-n)/60:.0f} min", flush=True)
    assert len(set(st["done"])) == len(series)
    pd.DataFrame(st["rows"], columns=["model", "pid", "uid", "coordX", "coordY", "coordZ", "probability", "dz_mm", "dy_mm", "dx_mm"]).to_csv(
        OUT / "pair_candidates.csv", index=False)


def check(s: pd.DataFrame):
    """Known answer: for primary series that the earlier NLST sample already converted with convert_native, our converter
    (with its own series as reference) must give the identical array."""
    prev = pd.read_csv(C.WORK / "wp0" / "analysis" / "p2_nlst" / "sample.csv", dtype={"uid": str}).head(2)
    tmp = OUT / "check_tmp"; ok = True
    for r in prev.itertuples():
        a, _ = load_volume(native_stem(r.stem))
        out = tmp / r.uid; convert_with_box(r.stem, usable_series().set_index("uid").local_path[r.uid], r.uid, out)
        b, _ = load_volume(out); same = a.shape == b.shape and np.array_equal(a, b); ok &= same
        print(f"check {r.uid[-12:]}: {'IDENTICAL' if same else 'DIFFERENT'}")
    print("CHECK", "PASSED" if ok else "FAILED"); return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("step", choices=["select", "convert", "calibrate", "infer", "check"])
    ap.add_argument("--procs", type=int, default=4); ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--limit", type=int); a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); sp = OUT / "sample.csv"
    if a.step == "select":
        assert not sp.exists(), "sample already fixed"
        s = select(); s.to_csv(sp, index=False)
        print(f"sample: {len(s)} exams"); print(s.groupby(["vendor", "kernel_primary", "kernel_partner"]).size().to_string()); sys.exit(0)
    s = pd.read_csv(sp, dtype={"uid_primary": str, "uid_partner": str, "pid": str})
    if a.limit:
        s = s.head(a.limit)
    if a.step == "check":
        sys.exit(0 if check(s) else 1)
    if a.step == "convert":
        log = open(OUT / "convert.log", "a"); t0 = time.time(); nf = 0
        with Pool(a.procs) as pool:
            for n, (pid, msgs) in enumerate(pool.imap_unordered(conv_job, [r._asdict() for r in s.itertuples(index=False)]), 1):
                nf += sum(m.startswith("FAIL") for m in msgs); line = f"{time.strftime('%F %T')} {n}/{len(s)} {pid} {msgs}"
                log.write(line + "\n"); log.flush()
                if n % 25 == 0 or n == len(s):
                    print(f"{line}  fails {nf}  {(time.time() - t0) / 60:.0f} min", flush=True)
        g = geometry_check(s); g.to_csv(OUT / "geometry_check.csv", index=False)
        print(f"geometry ok {int(g.geometry_ok.sum())}/{len(g)}"); print(g[~g.geometry_ok].head(20).to_string())
    elif a.step == "calibrate":
        ok = pd.read_csv(OUT / "geometry_check.csv", dtype={"pid": str}); s = s[s.pid.isin(set(ok[ok.geometry_ok].pid))]
        with Pool(a.procs) as pool:
            res = pd.DataFrame(list(pool.imap_unordered(cal_job, [r._asdict() for r in s.itertuples(index=False)])))
        res.to_csv(OUT / "kernel_calibration.csv", index=False)
        summ = {"overall_median_mm": float(res.sigma_mm.median()), "iqr_mm": [float(res.sigma_mm.quantile(.25)), float(res.sigma_mm.quantile(.75))],
                "n": len(res), "n_edge": int(res.edge.sum()),
                "by_vendor": {v: {"median": float(g.sigma_mm.median()), "q1": float(g.sigma_mm.quantile(.25)), "q3": float(g.sigma_mm.quantile(.75)),
                                  "n": len(g)} for v, g in res.groupby("vendor")}}
        summ["xycal_mm"] = round(round(summ["overall_median_mm"] / 0.05) * 0.05, 2)
        json.dump(summ, open(OUT / "kernel_calibration.json", "w"), indent=1); print(json.dumps(summ, indent=1))
    elif a.step == "infer":
        infer(s, a.folds)
