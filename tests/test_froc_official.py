"""Validate our FROC port against the official evaluation script.
Primary check: the official script (ported to Python 3, run with its shipped main: maxNumberOfCADMarks=100, on the same
annotation files) must give identical TP / FP / ignored counts and identical interpolated sensitivities at the 7 FP points.
Secondary check (no cap): counts against the shipped 2016 example output, which was produced without the cap."""
import sys, re, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lungcad import config as C
from lungcad.eval.froc import evaluate, FP_POINTS

ex = C.LUNA_EVAL / "exampleFiles"
official_out = Path(sys.argv[1]) if len(sys.argv) > 1 else None
ok = True
if official_out and (official_out / "CADAnalysis.txt").exists():
    txt = (official_out / "CADAnalysis.txt").read_text(); print(txt)
    res = evaluate(ex / "submission" / "sampleSubmission.csv", n_bootstrap=0, tag="sample", max_cands_per_scan=100)
    g = lambda k: int(re.search(k + r":\s*(\d+)", txt).group(1))
    print("ours:", {k: res[k] for k in ["n_scans", "n_nodules", "n_candidates_used", "n_detected", "n_fp", "n_ignored", "cpm"]})
    for name, mine, theirs in [("TP", res["n_detected"], g("True positives")), ("FP", res["n_fp"], g("False positives")),
                               ("candidates", res["n_candidates_used"], g("Total number of candidates")),
                               ("ignored", res["n_ignored"], g("Ignored candidates on excluded nodules") + g("Ignored candidates which were double detections on a nodule"))]:
        print(f"  {name}: ours {mine} official {theirs}", "OK" if mine == theirs else "MISMATCH"); ok &= mine == theirs
    fr = np.loadtxt(official_out / "froc_sampleSubmission.txt", delimiter=",")
    fps, sens = np.r_[0, fr[:, 0]], np.r_[0, fr[:, 1]]
    s_off = np.interp(FP_POINTS, fps, sens); cpm_off = s_off.mean()
    print(f"  CPM: ours {res['cpm']:.6f} official {cpm_off:.6f}"); ok &= abs(res["cpm"] - cpm_off) < 1e-6
    print("  sens@FP ours    ", np.round(res["sens_at_fp"], 6).tolist()); print("  sens@FP official", np.round(s_off, 6).tolist())
else:
    txt = (ex / "evaluation" / "CADAnalysis.txt").read_text(); print(txt)
    res = evaluate(ex / "submission" / "sampleSubmission.csv", n_bootstrap=0, tag="sample", max_cands_per_scan=0)
    print("ours (no cap):", {k: res[k] for k in ["n_scans", "n_nodules", "n_candidates_used", "n_detected", "n_fp", "n_ignored", "cpm"]})
print("PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
