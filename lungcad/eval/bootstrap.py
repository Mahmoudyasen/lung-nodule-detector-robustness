"""Paired bootstrap over scans for ΔCPM between two candidate sets on the SAME scans.

Both models are scored on the same resampled scan list in every replicate, so the CI on the difference
accounts for scan-level correlation. 'C beats B' is claimed only when the 95% CI of ΔCPM excludes 0.
"""
from __future__ import annotations
import json
import numpy as np, pandas as pd
from .froc import per_scan_results, froc_from_results


def paired_delta_cpm(cands_a: pd.DataFrame, cands_b: pd.DataFrame, uids: list[str], n_boot: int = 10000, seed: int = 0) -> dict:
    ra = per_scan_results(cands_a, uids); rb = per_scan_results(cands_b, uids)
    _, _, sa, ca = froc_from_results(ra); _, _, sb, cb = froc_from_results(rb)
    rng = np.random.RandomState(seed); d = np.empty(n_boot); da = np.empty(n_boot); db = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, len(uids), len(uids))
        _, _, _, xa = froc_from_results([ra[j] for j in idx]); _, _, _, xb = froc_from_results([rb[j] for j in idx])
        da[i], db[i], d[i] = xa, xb, xa - xb
    return {"cpm_a": ca, "cpm_b": cb, "delta": ca - cb, "delta_ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_delta_le_0": float((d <= 0).mean()), "cpm_a_ci95": [float(np.percentile(da, 2.5)), float(np.percentile(da, 97.5))],
            "cpm_b_ci95": [float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))], "n_boot": n_boot, "n_scans": len(uids)}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--uids", required=True); ap.add_argument("--n", type=int, default=10000)
    x = ap.parse_args(); uids = [l.strip() for l in open(x.uids) if l.strip()]
    print(json.dumps(paired_delta_cpm(pd.read_csv(x.a), pd.read_csv(x.b), uids, x.n), indent=1))
