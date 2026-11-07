"""S1 — one versioned index keyed on SeriesInstanceUID; official 10-fold split; LIDC exclusion list.

Done when: index CSV with one row per series, LUNA16 membership flag, official fold assignment.
LUNA16 fold i:  test = subset i,  val = subset (i+1) % 10,  train = the remaining eight.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import pandas as pd
from .. import config as C

INDEX_VERSION = "v1"


def build_index(raw: Path = C.LUNA_RAW, out_dir: Path = C.LUNA_INDEX) -> pd.DataFrame:
    rows = []
    for sub in sorted(raw.glob("subset*")):
        k = int(sub.name.replace("subset", ""))
        for mhd in sorted(sub.glob("*.mhd")):
            raw_file = mhd.with_suffix(".raw")
            rows.append(dict(seriesuid=mhd.stem, subset=k, mhd_path=str(mhd),
                             raw_bytes=raw_file.stat().st_size if raw_file.exists() else -1))
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit(f"no subsets found under {raw}")
    dup = df[df.duplicated("seriesuid", keep=False)]
    if len(dup):
        print(f"WARNING {dup.seriesuid.nunique()} SeriesInstanceUIDs appear in more than one subset; keeping first occurrence")
        print(dup.sort_values("seriesuid").to_string())
    df = df.drop_duplicates("seriesuid", keep="first").reset_index(drop=True)

    ann = pd.read_csv(raw / "annotations.csv")
    cand = pd.read_csv(raw / "candidates_V2.csv")
    official = pd.read_csv(C.LUNA_EVAL / "annotations" / "seriesuids.csv", header=None)[0]
    excl = pd.read_csv(C.LUNA_EVAL / "annotations" / "annotations_excluded.csv")
    df["n_nodules"] = df.seriesuid.map(ann.groupby("seriesuid").size()).fillna(0).astype(int)
    df["n_excluded"] = df.seriesuid.map(excl.groupby("seriesuid").size()).fillna(0).astype(int)
    df["n_candidates_v2"] = df.seriesuid.map(cand.groupby("seriesuid").size()).fillna(0).astype(int)
    df["n_candidates_v2_pos"] = df.seriesuid.map(cand[cand["class"] == 1].groupby("seriesuid").size()).fillna(0).astype(int)
    df["in_official_list"] = df.seriesuid.isin(set(official))
    df["has_seg_mask"] = [(C.LUNA_SEG / f"{u}.mhd").exists() for u in df.seriesuid]
    df["in_luna16"] = True          # authoritative membership flag
    df["lidc_exclude"] = True       # every LUNA16 series is excluded from any LIDC-IDRI auxiliary use
    df["fold"] = df.subset
    df["index_version"] = INDEX_VERSION

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"luna16_index_{INDEX_VERSION}.csv"
    df.to_csv(path, index=False)
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    (out_dir / f"luna16_index_{INDEX_VERSION}.md5").write_text(md5 + "\n")
    df[["seriesuid"]].to_csv(out_dir / "lidc_exclusion_list.csv", index=False)
    folds = {k: {"test": df[df.fold == k].seriesuid.tolist(), "val": df[df.fold == (k + 1) % 10].seriesuid.tolist(),
                 "train": df[~df.fold.isin([k, (k + 1) % 10])].seriesuid.tolist()} for k in range(10)}
    with open(out_dir / "folds.json", "w") as f:
        json.dump(folds, f)
    summary = {"n_series": int(len(df)), "n_duplicates_dropped": int(len(dup) - dup.seriesuid.nunique()) if len(dup) else 0,
               "n_official": int(df.in_official_list.sum()), "n_nodules": int(df.n_nodules.sum()),
               "n_candidates_v2": int(df.n_candidates_v2.sum()), "n_candidates_v2_pos": int(df.n_candidates_v2_pos.sum()),
               "per_subset": df.groupby("subset").size().to_dict(), "missing_seg": int((~df.has_seg_mask).sum()),
               "index_md5": md5}
    with open(out_dir / "index_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    return df


def load_index(version: str = INDEX_VERSION) -> pd.DataFrame:
    return pd.read_csv(C.LUNA_INDEX / f"luna16_index_{version}.csv")


def get_split(df: pd.DataFrame, fold: int) -> dict:
    return {"test": df[df.fold == fold].seriesuid.tolist(), "val": df[df.fold == (fold + 1) % 10].seriesuid.tolist(),
            "train": df[~df.fold.isin([fold, (fold + 1) % 10])].seriesuid.tolist()}


if __name__ == "__main__":
    build_index()
