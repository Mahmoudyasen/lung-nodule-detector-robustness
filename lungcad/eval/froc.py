"""LUNA16 FROC / CPM scorer — a Python-3 re-implementation of the official noduleCADEvaluationLUNA16.py logic.

Rules (identical to the official script):
  * A candidate hits a nodule when its distance to the nodule centre <= nodule radius (diameter/2; diameter < 0 -> 10 mm).
  * Hits on an *included* nodule: the highest-probability candidate counts as the detection, other hits are ignored.
  * Hits on an *excluded* (irrelevant) finding are ignored — neither TP nor FP.
  * Every other candidate is a false positive.
  * Undetected nodules enter the FROC list with probability -inf (never detected).
  * As in the official main(), at most max_cands_per_scan=100 candidates per scan are kept: those with probability
    strictly greater than the 101st-highest (ties at the boundary are dropped), so submit <= 100 marks per scan.
  * Sensitivity is interpolated at 0.125, 0.25, 0.5, 1, 2, 4, 8 FP/scan; CPM = mean of the seven.
  * Bootstrap over scans (default 1000) gives the 95% CI; a paired bootstrap between two candidate sets lives in bootstrap.py.
The port is validated against the official script's own example output (see tests/test_froc_official.py).
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np, pandas as pd
from .. import config as C

FP_POINTS = np.array(C.FP_POINTS)


@dataclass
class ScanResult:
    uid: str
    n_nodules: int
    tp_probs: np.ndarray    # one entry per detected nodule
    fp_probs: np.ndarray    # one entry per false positive
    n_missed: int
    n_ignored: int = 0      # double detections + marks on excluded findings


def load_reference(eval_dir: Path = C.LUNA_EVAL):
    ann = pd.read_csv(eval_dir / "annotations" / "annotations.csv")
    excl = pd.read_csv(eval_dir / "annotations" / "annotations_excluded.csv")
    uids = pd.read_csv(eval_dir / "annotations" / "seriesuids.csv", header=None)[0].astype(str).tolist()
    return ann, excl, uids


def cap_candidates(cands: pd.DataFrame, max_n: int) -> pd.DataFrame:
    """Official maxNumberOfCADMarks rule: keep marks with probability > the (max_n+1)-th largest, in file order, up to max_n."""
    if max_n <= 0 or len(cands) <= max_n:
        return cands
    probs = np.sort(cands["probability"].values.astype(float))[::-1]
    thr = probs[max_n]
    return cands[cands["probability"].astype(float) > thr].head(max_n)


def score_scan(uid: str, cands: pd.DataFrame, nod: pd.DataFrame, exc: pd.DataFrame, max_cands_per_scan: int = 100) -> ScanResult:
    """cands: rows for this scan with coordX/Y/Z/probability; nod/exc: annotation rows for this scan."""
    cands = cap_candidates(cands, max_cands_per_scan)
    inc_xyz = nod[["coordX", "coordY", "coordZ"]].values.astype(float) if len(nod) else np.zeros((0, 3))
    inc_d = nod["diameter_mm"].values.astype(float) if len(nod) else np.zeros(0)
    inc_r2 = np.where(inc_d < 0, 10.0, inc_d) ** 2 / 4.0
    exc_xyz = exc[["coordX", "coordY", "coordZ"]].values.astype(float) if len(exc) else np.zeros((0, 3))
    exc_d = exc["diameter_mm"].values.astype(float) if len(exc) else np.zeros(0)
    exc_r2 = np.where(exc_d < 0, 10.0, exc_d) ** 2 / 4.0
    best = np.full(len(inc_xyz), -np.inf)
    fps = []
    if len(cands):
        cxyz = cands[["coordX", "coordY", "coordZ"]].values.astype(float)
        prob = cands["probability"].values.astype(float)
        d_inc = ((cxyz[:, None, :] - inc_xyz[None, :, :]) ** 2).sum(-1) if len(inc_xyz) else np.zeros((len(cxyz), 0))
        d_exc = ((cxyz[:, None, :] - exc_xyz[None, :, :]) ** 2).sum(-1) if len(exc_xyz) else np.zeros((len(cxyz), 0))
        hit_inc = d_inc < inc_r2[None, :]
        hit_exc = (d_exc < exc_r2[None, :]).any(1) if len(exc_xyz) else np.zeros(len(cxyz), bool)
        for i in range(len(cxyz)):
            h = np.where(hit_inc[i])[0]
            if len(h):
                for j in h:
                    if prob[i] > best[j]:
                        best[j] = prob[i]
            elif hit_exc[i]:
                continue
            else:
                fps.append(prob[i])
    tp = best[np.isfinite(best)]
    n_ignored = int(len(cands) - len(tp) - len(fps))
    return ScanResult(uid, len(inc_xyz), tp, np.array(fps, float), int((~np.isfinite(best)).sum()), n_ignored)


def froc_from_results(results: list[ScanResult], fp_points=FP_POINTS):
    """-> (fps_per_scan array, sensitivity array, sens_at_points, cpm)."""
    n_img = len(results); n_nod = sum(r.n_nodules for r in results)
    probs = np.concatenate([np.concatenate([r.tp_probs, r.fp_probs]) for r in results] + [np.zeros(0)])
    is_tp = np.concatenate([np.concatenate([np.ones(len(r.tp_probs)), np.zeros(len(r.fp_probs))]) for r in results] + [np.zeros(0)])
    if n_nod == 0 or len(probs) == 0:
        return np.zeros(0), np.zeros(0), np.zeros(len(fp_points)), 0.0
    order = np.argsort(-probs, kind="stable")
    probs, is_tp = probs[order], is_tp[order]
    # threshold at each distinct probability (candidates with equal prob enter together)
    distinct = np.r_[np.where(np.diff(probs) != 0)[0], len(probs) - 1]
    tps = np.cumsum(is_tp)[distinct]; fpc = np.cumsum(1 - is_tp)[distinct]
    sens = tps / n_nod; fps = fpc / n_img
    fps_i = np.r_[0.0, fps]; sens_i = np.r_[0.0, sens]     # FROC starts at (0,0) exactly as the official plot
    s_at = np.interp(fp_points, fps_i, sens_i)
    return fps, sens, s_at, float(s_at.mean())


def evaluate(candidates: pd.DataFrame | str | Path, uids: list[str] | None = None, eval_dir: Path = C.LUNA_EVAL,
             out_dir: Path | None = None, n_bootstrap: int = 1000, seed: int = 0, tag: str = "", max_cands_per_scan: int = 100) -> dict:
    """Score a candidate table (seriesuid, coordX, coordY, coordZ, probability) on the given scans (default: all 888)."""
    if not isinstance(candidates, pd.DataFrame):
        candidates = pd.read_csv(candidates)
    ann, excl, all_uids = load_reference(eval_dir)
    uids = [str(u) for u in (uids if uids is not None else all_uids)]
    candidates = candidates.copy(); candidates["seriesuid"] = candidates.seriesuid.astype(str)
    candidates = candidates[candidates.seriesuid.isin(set(uids))]
    cg = {k: g for k, g in candidates.groupby("seriesuid")}
    ag = {k: g for k, g in ann.groupby("seriesuid")}; eg = {k: g for k, g in excl.groupby("seriesuid")}
    empty = pd.DataFrame(columns=["coordX", "coordY", "coordZ", "diameter_mm", "probability"])
    results = [score_scan(u, cg.get(u, empty), ag.get(u, empty), eg.get(u, empty), max_cands_per_scan) for u in uids]
    fps, sens, s_at, cpm = froc_from_results(results)
    n_used = sum(len(r.tp_probs) + len(r.fp_probs) + r.n_ignored for r in results)
    out = {"tag": tag, "n_scans": len(uids), "n_nodules": int(sum(r.n_nodules for r in results)), "max_cands_per_scan": max_cands_per_scan,
           "n_candidates": int(len(candidates)), "n_candidates_used": int(n_used), "n_ignored": int(sum(r.n_ignored for r in results)), "n_detected": int(sum(len(r.tp_probs) for r in results)),
           "n_missed": int(sum(r.n_missed for r in results)), "n_fp": int(sum(len(r.fp_probs) for r in results)),
           "fp_points": FP_POINTS.tolist(), "sens_at_fp": s_at.tolist(), "cpm": cpm,
           "max_sensitivity": float(sens.max()) if len(sens) else 0.0, "fp_per_scan_at_max": float(fps.max()) if len(fps) else 0.0}
    if n_bootstrap:
        rng = np.random.RandomState(seed); cpms = []; sens_b = []
        for _ in range(n_bootstrap):
            idx = rng.randint(0, len(results), len(results))
            _, _, sb, cb = froc_from_results([results[i] for i in idx])
            cpms.append(cb); sens_b.append(sb)
        cpms = np.array(cpms); sens_b = np.array(sens_b)
        out.update({"cpm_ci95": [float(np.percentile(cpms, 2.5)), float(np.percentile(cpms, 97.5))], "cpm_boot_mean": float(cpms.mean()),
                    "sens_at_fp_ci95_low": np.percentile(sens_b, 2.5, axis=0).tolist(), "sens_at_fp_ci95_high": np.percentile(sens_b, 97.5, axis=0).tolist()})
    if out_dir:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / f"froc{('_' + tag) if tag else ''}.json", "w") as f:
            json.dump(out, f, indent=1)
        pd.DataFrame({"fps_per_scan": fps, "sensitivity": sens}).to_csv(out_dir / f"froc_curve{('_' + tag) if tag else ''}.csv", index=False)
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 4)); ax.semilogx(np.r_[0.001, fps], np.r_[0, sens], drawstyle="steps-post"); ax.set_xlim(0.125 / 2, 8 * 2)
            for p, s in zip(FP_POINTS, s_at): ax.plot(p, s, "ro", ms=4)
            ax.set_xlabel("false positives per scan"); ax.set_ylabel("sensitivity"); ax.set_ylim(0, 1); ax.grid(alpha=.3)
            ax.set_title(f"FROC {tag}  CPM={cpm:.4f}"); fig.tight_layout(); fig.savefig(out_dir / f"froc{('_' + tag) if tag else ''}.png", dpi=130); plt.close(fig)
        except Exception:
            pass
    return out


def per_scan_results(candidates: pd.DataFrame, uids: list[str], eval_dir: Path = C.LUNA_EVAL, max_cands_per_scan: int = 100) -> list[ScanResult]:
    ann, excl, _ = load_reference(eval_dir)
    candidates = candidates.copy(); candidates["seriesuid"] = candidates.seriesuid.astype(str)
    cg = {k: g for k, g in candidates.groupby("seriesuid")}; ag = {k: g for k, g in ann.groupby("seriesuid")}; eg = {k: g for k, g in excl.groupby("seriesuid")}
    empty = pd.DataFrame(columns=["coordX", "coordY", "coordZ", "diameter_mm", "probability"])
    return [score_scan(u, cg.get(u, empty), ag.get(u, empty), eg.get(u, empty), max_cands_per_scan) for u in uids]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("candidates"); ap.add_argument("--uids", help="file with one seriesuid per line")
    ap.add_argument("--out"); ap.add_argument("--tag", default=""); ap.add_argument("--bootstrap", type=int, default=1000)
    a = ap.parse_args()
    uids = [l.strip() for l in open(a.uids) if l.strip()] if a.uids else None
    r = evaluate(a.candidates, uids, out_dir=a.out, n_bootstrap=a.bootstrap, tag=a.tag)
    print(json.dumps({k: r[k] for k in ["n_scans", "n_nodules", "n_candidates", "n_detected", "n_fp", "sens_at_fp", "cpm"] + (["cpm_ci95"] if "cpm_ci95" in r else [])}, indent=1))
