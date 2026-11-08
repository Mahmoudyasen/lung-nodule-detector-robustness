"""Review experiment R4 (lung-nodule-detector-analysis-plans/plans_original/review_prespec.md): evaluate the seed-1 fold-0 models (S3 fold0/seed1_s3_noaug, S4
fold0/seed1_s4_aug) exactly like the seed-0 pair: all E1 conditions on the 89 fold-0 test scans (same per-scan seeds, so
the perturbed inputs are identical to the seed-0 run) and the TCIA pairs (both doses). Projection noise and textures:
scripts/27_texture_e1.py --seed1; in-plane blur: scripts/31_inplane_blur_e1.py --seed1.

    python scripts/33_seed1_eval.py e1
    python scripts/33_seed1_eval.py tcia
Output: analysis/p2_review/seed1_e1/fold0/, analysis/p2_review/seed1_tcia/.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1_dose_robustness as E
from e1_dose_robustness import C, CONDITIONS, CONDITIONS_EXTRA, load_index, get_split

ROOT = C.WORK / "wp0" / "analysis" / "p2_review"
SEED1 = {"S3": C.STAGE1 / "fold0" / "seed1_s3_noaug" / "best.pt", "S4": C.STAGE1 / "fold0" / "seed1_s4_aug" / "best.pt"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("step", choices=["e1", "tcia"]); a = ap.parse_args()
    for p in SEED1.values():
        assert p.exists(), f"missing checkpoint {p}"
    if a.step == "e1":
        index = load_index(); uids = get_split(index, 0)["test"]
        E.MODELS, E.OUT = SEED1, ROOT / "seed1_e1" / "fold0"
        E.run_inference(uids, index, CONDITIONS, "")
        E.run_inference(uids, index, CONDITIONS_EXTRA, "_extra")
    else:
        import importlib; T = importlib.import_module("22_tcia_e1")
        T.OUT = ROOT / "seed1_tcia"; T.OUT.mkdir(parents=True, exist_ok=True); T.FOLDS = [0]; T.fold_models = lambda k: SEED1
        cases = sorted(p.name[:-len("_full.json")] for p in T.PRE.glob("C*_full.json")); assert len(cases) == 50
        T.infer(cases)
