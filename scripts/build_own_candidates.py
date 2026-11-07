"""Freeze stage 1: concatenate the ten folds' TEST-split candidate lists (each scan scored by the model that never saw
it) into work/wp0/stage1/own_candidates/own_candidates_all.csv with a manifest of the exact files + hashes.
Refuses to freeze unless every fold finished training (done.json) and its candidates cover exactly its test split."""
import glob, hashlib, json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lungcad import config as C
from lungcad.data.luna16_index import load_index, get_split
index = load_index(); out = C.STAGE1 / "own_candidates"; parts = []; man = []
for k in range(10):
    d = C.STAGE1 / f"fold{k}" / "seed0"
    if not (d / "done.json").exists(): sys.exit(f"fold {k}: training not finished (no done.json), not freezing")
    fs = sorted(glob.glob(str(d / f"candidates_fold{k}_test_*.csv")), key=lambda f: Path(f).stat().st_mtime)
    if not fs: sys.exit(f"fold {k}: no test candidate file yet")
    if len(fs) > 1: print(f"fold {k}: {len(fs)} candidate files, using the newest: {Path(fs[-1]).name}")
    f = fs[-1]; df = pd.read_csv(f); test = set(get_split(index, k)["test"])
    extra = set(df.seriesuid) - test
    if extra: sys.exit(f"fold {k}: {len(extra)} scans in {Path(f).name} are not in fold {k}'s test split, not freezing")
    done = json.loads((d / "done.json").read_text())
    parts.append(df); man.append({"fold": k, "file": f, "md5": hashlib.md5(Path(f).read_bytes()).hexdigest(), "n": len(df),
                                  "scans_with_candidates": int(df.seriesuid.nunique()), "test_scans": len(test),
                                  "train_run_tag": done.get("run_tag"), "best_epoch": done.get("best_epoch")})
out.mkdir(parents=True, exist_ok=True)
all_ = pd.concat(parts, ignore_index=True); all_.to_csv(out / "own_candidates_all.csv", index=False)
json.dump({"files": man, "n_total": len(all_), "n_scans": int(all_.seriesuid.nunique())}, open(out / "manifest.json", "w"), indent=1)
print("frozen:", len(all_), "candidates over", all_.seriesuid.nunique(), "scans ->", out)
