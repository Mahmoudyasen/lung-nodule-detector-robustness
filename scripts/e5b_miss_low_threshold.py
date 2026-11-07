"""E5b — do the 27 missed nodules appear at all if the detector's score threshold is lowered from 0.02 to 0.001?

    python scripts/e5b_miss_low_threshold.py

Re-runs each fold's frozen best.pt (analysis only, nothing frozen is written) on the scans that contain a miss,
with score_thresh 0.001 and up to 1000 detections, and reports for every miss the best score of any box whose
centre lies within the nodule radius (official hit rule) and within 2x radius.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lungcad import config as C
from lungcad.data.luna16_index import load_index
from lungcad.stage1.infer import infer_scans
from lungcad.stage1.model import build_detector, set_inferer

OUT = C.WORK / "wp0" / "analysis" / "e5_misses"


def main():
    mi = pd.read_csv(OUT / "misses.csv", dtype={"seriesuid": str}); index = load_index(); res = []
    for fold, g in mi.groupby("fold"):
        ck = torch.load(C.STAGE1 / f"fold{fold}" / "seed0" / "best.pt", map_location="cuda", weights_only=False); cfg = ck["cfg"]
        mc = dict(cfg["model"], score_thresh=0.001, detections_per_img=1000)
        det = build_detector(mc); det.network.load_state_dict(ck["model"])
        set_inferer(det, cfg["infer"]["roi_size"], cfg["infer"]["overlap"], cfg["infer"]["sw_batch_size"])
        c = infer_scans(det, index, sorted(g.seriesuid.unique()), cfg["infer"], amp=cfg["train"]["amp"])
        for _, m in g.iterrows():
            cc = c[c.seriesuid == m.seriesuid]
            d = np.sqrt(((cc[["coordX", "coordY", "coordZ"]].values - m[["coordX", "coordY", "coordZ"]].values.astype(float)) ** 2).sum(1))
            r = m.diameter_mm / 2; p = cc.probability.values
            rank = lambda s: int((p > s).sum()) + 1 if np.isfinite(s) else None
            s1 = p[d <= r].max() if (d <= r).any() else -np.inf; s2 = p[d <= 2 * r].max() if (d <= 2 * r).any() else -np.inf
            res.append({"miss_id": m.miss_id, "fold": fold, "diameter_mm": m.diameter_mm, "n_boxes_scan_0.001": len(cc),
                        "best_score_within_r": s1, "rank_in_scan_within_r": rank(s1), "best_score_within_2r": s2, "nearest_mm_0.001": float(d.min()) if len(d) else np.nan})
        print(f"fold {fold}: {len(g)} misses checked", flush=True)
    df = pd.DataFrame(res).sort_values("miss_id"); df.to_csv(OUT / "misses_low_threshold.csv", index=False)
    print(df.to_string(index=False))
    print(f"recovered within r at score >= 0.001: {np.isfinite(df.best_score_within_r).sum()} / {len(df)}")


if __name__ == "__main__":
    main()
