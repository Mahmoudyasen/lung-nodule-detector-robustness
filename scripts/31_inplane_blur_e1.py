"""Review experiment R1 (analysis_plans/review_prespec.md): in-plane-only Gaussian blur, S3 vs S4, LUNA16 folds 0-3.

    python scripts/31_inplane_blur_e1.py --check                 # known answer: blur_xy with z added = the E1 3D blur
    python scripts/31_inplane_blur_e1.py [--folds 0 1 2 3]        # xy1.0, xy1.5 (+ xycal once the NLST calibration exists)
    python scripts/31_inplane_blur_e1.py --seed1                  # fold 0, seed-1 models (R4)

Reuses scripts/e1_dose_robustness.run_inference unchanged (same loading, inference, output format); only the perturbation
is replaced by a blur along y and x (same separable Gaussian kernel and reflect padding as lungcad.augment.separable_blur).
Output: analysis/p2_review/inplane[_seed1]/fold<k>/e1_candidates_all.csv.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch, torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
import e1_dose_robustness as E
from e1_dose_robustness import C, load_index, get_split
from e1_multifold import fold_models
from lungcad.augment import _gauss3d, separable_blur

ROOT = C.WORK / "wp0" / "analysis" / "p2_review"
VOXEL_XY = 0.703125


def blur_dims(x, sigma, dims):
    if sigma <= 0:
        return x
    k, r = _gauss3d(sigma, x.device, x.dtype); Cc = x.shape[1]; y = x
    for dim in dims:
        shape = [1, 1, 1, 1, 1]; shape[dim] = k.numel()
        w = k.view(shape).expand(Cc, 1, *shape[2:]).contiguous()
        pad = [0] * 6; pad[2 * (4 - dim)] = r; pad[2 * (4 - dim) + 1] = r
        y = F.conv3d(F.pad(y, pad, mode="reflect"), w, groups=Cc)
    return y


@torch.no_grad()
def perturb_xy(x, kernel, strength, sigma_hu, seed, corr=0.0):
    assert kernel == "blur_xy" and sigma_hu == 0
    return blur_dims(x, strength, (3, 4)).clamp(0, 1)


def conditions():
    c = {"xy1.0": ("blur_xy", 1.0, 0), "xy1.5": ("blur_xy", 1.5, 0)}
    cal = ROOT / "nlst_pairs" / "kernel_calibration.json"
    if cal.exists():
        mm = json.load(open(cal))["xycal_mm"]; c["xycal"] = ("blur_xy", mm / VOXEL_XY, 0)
        print(f"xycal = {mm} mm = {mm / VOXEL_XY:.3f} voxels in-plane")
    else:
        print("NOTE: no NLST calibration yet -> xycal not run")
    return c


def seed1_models():
    return {"S3": C.STAGE1 / "fold0" / "seed1_s3_noaug" / "best.pt", "S4": C.STAGE1 / "fold0" / "seed1_s4_aug" / "best.pt"}


def check():
    torch.manual_seed(0); x = torch.rand(1, 1, 20, 40, 40, device="cuda")
    for s in (0.5, 1.0, 1.5):
        a = separable_blur(x, s); b = blur_dims(blur_dims(x, s, (3, 4)), s, (2,))
        print(f"sigma {s}: max |3D blur - (xy then z)| = {float((a - b).abs().max()):.2e}")
        assert float((a - b).abs().max()) < 1e-5
    print(f"xy blur leaves z untouched: {float((blur_dims(x, 1.0, (3, 4)).mean((3, 4)) - x.mean((3, 4))).abs().max()):.2e} (in-plane mean preserved up to borders)")
    print("CHECK PASSED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--check", action="store_true"); ap.add_argument("--seed1", action="store_true"); ap.add_argument("--only", nargs="+")
    a = ap.parse_args()
    if a.check:
        check(); sys.exit(0)
    E.perturb = perturb_xy                                   # run_inference looks the perturbation up in its module
    index = load_index(); conds = conditions()
    if a.only:
        conds = {k: v for k, v in conds.items() if k in a.only}
    for k in ([0] if a.seed1 else a.folds):
        E.MODELS = seed1_models() if a.seed1 else fold_models(k)
        for p in E.MODELS.values():
            assert p.exists(), f"missing checkpoint {p}"
        uids = get_split(index, k)["test"]
        for cname, spec in conds.items():                  # one output per condition, so xycal can be added later
            E.OUT = ROOT / ("inplane_seed1" if a.seed1 else "inplane") / f"fold{k}"
            if (E.OUT / f"e1_candidates_all_{cname}.csv").exists():
                print(f"fold {k} {cname}: done"); continue
            print(f"== fold {k} {cname}: {len(uids)} scans", flush=True)
            E.run_inference(uids, index, {cname: spec}, f"_{cname}")
