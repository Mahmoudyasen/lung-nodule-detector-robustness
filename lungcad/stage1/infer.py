"""Inference: sliding-window detection over whole preprocessed scans -> world-coordinate candidate CSV.

    python -m lungcad.stage1.infer --checkpoint work/wp0/stage1/fold0/seed0/best.pt --fold 0 --split test

Outputs a version-tagged file  candidates_fold{k}_{split}_{run_tag}.csv  (+ .meta.json) with columns
seriesuid, coordX, coordY, coordZ, probability, dz_mm, dy_mm, dx_mm — directly scorable by lungcad.eval.froc.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from .. import config as C
from ..preprocess import voxel_to_world
from ..data.luna16_index import load_index, get_split
from .model import build_detector, set_inferer
from .dataset import load_full_volume


@torch.no_grad()
def infer_scans(det, index: pd.DataFrame, uids: list[str], infer_cfg: dict, amp: bool = True, verbose: bool = False) -> pd.DataFrame:
    det.eval(); idx = index.set_index("seriesuid"); rows = []; t0 = time.time()
    for n, uid in enumerate(uids):
        img, meta = load_full_volume(uid, int(idx.loc[uid, "subset"]))
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            out = det(img, use_inferer=True)[0]
        boxes = out[det.target_box_key].float().cpu().numpy(); scores = out[det.pred_score_key].float().cpu().numpy()
        if len(boxes):
            ctr = (boxes[:, :3] + boxes[:, 3:]) / 2; size = boxes[:, 3:] - boxes[:, :3]
            w = voxel_to_world(meta, ctr); sp = np.array(meta["spacing_zyx"])
            for i in range(len(boxes)):
                rows.append((uid, w[i, 0], w[i, 1], w[i, 2], float(scores[i]), *(size[i] * sp)))
        if verbose and (n + 1) % 10 == 0:
            print(f"  {n+1}/{len(uids)} scans, {len(rows)} candidates, {time.time()-t0:.0f}s", flush=True)
    return pd.DataFrame(rows, columns=["seriesuid", "coordX", "coordY", "coordZ", "probability", "dz_mm", "dy_mm", "dx_mm"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--split", default="test", choices=["test", "val", "train", "all"]); ap.add_argument("--out", default=None)
    ap.add_argument("--score", action="store_true", help="also run the FROC scorer on the output")
    a = ap.parse_args()
    ck = torch.load(a.checkpoint, map_location="cuda", weights_only=False); cfg = ck["cfg"]
    det = build_detector(cfg["model"]); det.network.load_state_dict(ck["model"])
    set_inferer(det, cfg["infer"]["roi_size"], cfg["infer"]["overlap"], cfg["infer"]["sw_batch_size"])
    index = load_index(); split = get_split(index, a.fold)
    uids = index.seriesuid.tolist() if a.split == "all" else split[a.split]
    tag = ck.get("run_tag", "untagged")
    out_dir = Path(a.out or Path(a.checkpoint).parent); out_dir.mkdir(parents=True, exist_ok=True)
    df = infer_scans(det, index, uids, cfg["infer"], amp=cfg["train"]["amp"], verbose=True)
    name = f"candidates_fold{a.fold}_{a.split}_{tag}"
    df.to_csv(out_dir / f"{name}.csv", index=False)
    meta = {"checkpoint": str(a.checkpoint), "run_tag": tag, "fold": a.fold, "split": a.split, "n_scans": len(uids), "n_candidates": int(len(df)),
            "candidates_per_scan": float(len(df) / max(1, len(uids))), "epoch": ck.get("state", {}).get("best_epoch"), "infer_cfg": cfg["infer"],
            "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if a.score:
        from ..eval.froc import evaluate
        res = evaluate(df, uids, out_dir=out_dir, n_bootstrap=1000, tag=f"fold{a.fold}_{a.split}")
        meta["froc"] = {k: res[k] for k in ["cpm", "cpm_ci95", "sens_at_fp", "max_sensitivity", "n_fp", "n_nodules", "n_missed"]}
        print(json.dumps(meta["froc"], indent=1))
    with open(out_dir / f"{name}.meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print("wrote", out_dir / f"{name}.csv", meta["n_candidates"], "candidates")


if __name__ == "__main__":
    main()
