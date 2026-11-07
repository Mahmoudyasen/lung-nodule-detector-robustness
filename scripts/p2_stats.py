"""Statistics, as pre-specified on 30 Oct 2026 (analysis plan for F1-F3):
paired bootstrap with a COMMON resample for all systems, 10,000 replicates, 95 % percentile CIs, two-sided
P = 2 * min(P*(d <= 0), P*(d >= 0)) (floor 2 / n_boot), Holm correction within each primary family.

Clusters: the unit that is resampled. LUNA16: the scan. TCIA: the patient, whose full- and 10 %-dose pseudo-series are
resampled together (cluster = list of uids). Every system is scored on the same resampled clusters in each replicate.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lungcad import config as C
from lungcad.eval.froc import per_scan_results, froc_from_results

N_BOOT = 10000


def scan_results(cands: pd.DataFrame, uids: list[str], eval_dir: Path = C.LUNA_EVAL) -> dict:
    """uid -> per-scan FROC result (the official LUNA16 hit rules; eval_dir holds the reference)."""
    return dict(zip(uids, per_scan_results(cands, uids, eval_dir=eval_dir)))


def cpm(res: dict, uids) -> float:
    return float(froc_from_results([res[u] for u in uids])[3])


def bootstrap(systems: dict[str, dict], clusters: list[list[str]], n_boot: int = N_BOOT, seed: int = 0) -> tuple[pd.Series, pd.DataFrame]:
    """systems: name -> {uid: result}; clusters: resampling units (lists of uids). Returns point CPMs and an
    (n_boot x systems) table of CPMs on common resamples."""
    allu = [u for c in clusters for u in c]
    point = pd.Series({k: cpm(v, allu) for k, v in systems.items()})
    rng = np.random.RandomState(seed); out = np.empty((n_boot, len(systems)))
    for i in range(n_boot):
        idx = rng.randint(0, len(clusters), len(clusters)); us = [u for j in idx for u in clusters[j]]
        out[i] = [cpm(v, us) for v in systems.values()]
    return point, pd.DataFrame(out, columns=list(systems))


def contrast(point: pd.Series, boot: pd.DataFrame, weights: dict[str, float]) -> dict:
    """Linear contrast sum_k w_k * CPM_k (e.g. {'S4': 1, 'S3': -1}, or the TCIA difference-in-differences)."""
    est = float(sum(w * point[k] for k, w in weights.items())); d = sum(w * boot[k] for k, w in weights.items()).values
    n = len(d); p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())); p = max(p, 2.0 / n)
    return {"estimate": est, "ci_lo": float(np.percentile(d, 2.5)), "ci_hi": float(np.percentile(d, 97.5)), "p_two_sided": float(p), "n_boot": n}


def holm(p: pd.Series) -> pd.Series:
    """Holm step-down adjusted P values (monotone, capped at 1)."""
    order = p.sort_values(); m = len(p); adj = np.maximum.accumulate([(m - i) * v for i, v in enumerate(order.values)])
    return pd.Series(np.minimum(adj, 1.0), index=order.index).reindex(p.index)


if __name__ == "__main__":                       # self-checks of the helpers
    print(holm(pd.Series({"a": 0.01, "b": 0.04, "c": 0.03})).round(3).to_dict(), "(expect a 0.03, b 0.06, c 0.06)")
    b = pd.DataFrame({"x": np.random.RandomState(1).normal(0.02, 0.01, 10000), "y": 0.0})
    print(contrast(pd.Series({"x": 0.02, "y": 0.0}), b, {"x": 1, "y": -1}))
