"""Local inventory of the NLST imaging collection.

Reads the IDC download manifest (metadata.csv), verifies every series directory on disk (file count and
bytes), parses the series-description tokens
    "<round>,<type>,<vendor>,<model>,<kernel>,<fov>,<thickness>,<kVp>,<mA>,<pitch>,<interval>"
and writes:
    nlst/meta/nlst_series_manifest.csv   one row per series, with on-disk status (+ a `convert_status` column)
    nlst/meta/nlst_primary_series.csv    ONE chosen axial series per (pid, screening round) — the conversion queue
    nlst/meta/inventory_summary.json     summary counts of the inventory
    nlst/meta/inventory_tables.md        the same, as markdown tables
Screening rounds come from the first token (0/1/2) and are cross-checked against the DICOM tag
ClinicalTrialTimePointID (T0/T1/T2) during conversion.
"""
from __future__ import annotations
import argparse, json, os, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
from .. import config as C

SOFT_KERNELS = {"STANDARD", "B30f", "B31f", "B20f", "B10f", "B35f", "B40f", "FC10", "FC01", "FC02", "FC11", "FC30", "C", "B", "A", "T20s", "SOFT", "DETAIL", "T20f",
                "B20s", "FC03"}   # added for 2 LUNA16 test scans, by the numbering already used here (Siemens B/T <= 40 soft; Toshiba FC0x/FC1x/FC3x soft, FC5x sharp)
SHARP_KERNELS = {"BONE", "LUNG", "B50f", "B60f", "B70f", "B80f", "B45f", "FC51", "FC50", "FC53", "FC52", "D", "E", "EXPERIMENTAL7", "EDGE", "T80f", "T80s", "B46f", "B50s"}


def kernel_class(k: str) -> str:
    if k in SOFT_KERNELS: return "soft"
    if k in SHARP_KERNELS: return "sharp"
    return "unknown"


def parse_tokens(desc: str) -> dict:
    t = (desc or "").split(",") + [""] * 11
    return {"round": t[0], "imgtype": t[1], "vendor_tok": t[2], "model_tok": t[3], "kernel": t[4], "fov_tok": t[5],
            "thk_tok": t[6], "kvp_tok": t[7], "ma_tok": t[8], "pitch_tok": t[9], "interval_tok": t[10]}


def local_path(manifest_path: str) -> str:
    p = manifest_path.replace("\\", "/")
    p = p.split("nlst/", 1)[1] if "nlst/" in p else p
    return str(C.NLST_DICOM / p)


def scan_dir(path: str):
    try:
        n = 0; b = 0
        with os.scandir(path) as it:
            for e in it:
                if e.is_file():
                    n += 1; b += e.stat().st_size
        return n, b
    except FileNotFoundError:
        return 0, -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-disk", action="store_true", help="skip the on-disk verification (fast, manifest only)")
    a = ap.parse_args()
    t0 = time.time()
    m = pd.read_csv(C.NLST_METADATA, low_memory=False)
    if a.limit:
        m = m.head(a.limit)
    tok = pd.DataFrame([parse_tokens(s) for s in m.SeriesDescription.fillna("")])
    m = pd.concat([m.reset_index(drop=True), tok], axis=1)
    m["kernel_class"] = m.kernel.map(kernel_class)
    m["thk_mm"] = pd.to_numeric(m.thk_tok, errors="coerce")
    m.loc[(m.thk_mm > 20) | (m.thk_mm <= 0), "thk_mm"] = np.nan   # token-shift artefacts (350.2 etc.)
    m["local_path"] = m.S5cmdManifestPath.map(local_path)
    if a.no_disk:
        m["n_files"] = -1; m["bytes_on_disk"] = -1; m["disk_status"] = "unchecked"
    else:
        print(f"verifying {len(m)} series directories on disk with {a.threads} threads ...", flush=True)
        res = [None] * len(m)
        with ThreadPoolExecutor(a.threads) as ex:
            for i, r in enumerate(ex.map(scan_dir, m.local_path.tolist(), chunksize=64)):
                res[i] = r
                if i % 10000 == 0:
                    print(f"  {i}/{len(m)} {time.time()-t0:.0f}s", flush=True)
        m["n_files"] = [r[0] for r in res]; m["bytes_on_disk"] = [r[1] for r in res]
        m["disk_status"] = np.where(m.bytes_on_disk < 0, "missing",
                              np.where((m.bytes_on_disk - m.FileSize).abs() <= m.n_files.clip(lower=1), "complete", "size_mismatch"))
    m["is_axial_ct"] = (m.n_files >= 30) if not a.no_disk else (m.imgtype == "OPA")
    m["convert_status"] = "pending"
    C.NLST_META.mkdir(parents=True, exist_ok=True)
    m.to_csv(C.NLST_META / "nlst_series_manifest.csv", index=False)

    # ---- primary series per (pid, round): axial, soft kernel first, thinnest slices, most files
    ax = m[m.is_axial_ct & m["round"].isin(["0", "1", "2"])].copy()
    ax["kernel_rank"] = ax.kernel_class.map({"soft": 0, "unknown": 1, "sharp": 2})
    ax["thk_rank"] = ax.thk_mm.fillna(9.9)
    ax = ax.sort_values(["PatientID", "round", "kernel_rank", "thk_rank", "n_files"], ascending=[True, True, True, True, False])
    prim = ax.drop_duplicates(["PatientID", "round"], keep="first").copy()
    prim["round"] = prim["round"].astype(int)
    prim.to_csv(C.NLST_META / "nlst_primary_series.csv", index=False)

    # ---- summary numbers
    rounds_per_pid = ax.groupby("PatientID")["round"].nunique()
    prim_rounds = prim.groupby("PatientID")["round"].nunique()
    s = {
        "manifest_rows": int(len(m)), "patients": int(m.PatientID.nunique()), "studies": int(m.StudyInstanceUID.nunique()),
        "series_total": int(len(m)), "series_axial_ct(>=30 files)": int(m.is_axial_ct.sum()),
        "disk_status": m.disk_status.value_counts().to_dict(),
        "bytes_expected_TB": round(float(m.FileSize.sum()) / 1e12, 3),
        "bytes_on_disk_TB": round(float(m.bytes_on_disk.clip(lower=0).sum()) / 1e12, 3) if not a.no_disk else None,
        "axial_series_per_round": ax["round"].value_counts().sort_index().to_dict(),
        "patients_by_n_rounds_with_axial_ct": rounds_per_pid.value_counts().sort_index().to_dict(),
        "patients_with_3_rounds": int((rounds_per_pid == 3).sum()),
        "patients_with_>=2_rounds": int((rounds_per_pid >= 2).sum()),
        "primary_series": int(len(prim)), "primary_series_by_kernel_class": prim.kernel_class.value_counts().to_dict(),
        "primary_thickness_mm": prim.thk_mm.round(1).value_counts().sort_index().to_dict(),
        "primary_manufacturer": prim.Manufacturer.value_counts().to_dict(),
        "primary_patients_with_3_rounds": int((prim_rounds == 3).sum()),
        "kernel_distribution_axial": ax.kernel.value_counts().head(25).to_dict(),
        "kernel_class_axial": ax.kernel_class.value_counts().to_dict(),
        "thickness_distribution_axial_mm": ax.thk_mm.round(1).value_counts().sort_index().to_dict(),
        "slices_per_axial_series": {k: float(v) for k, v in ax.n_files.describe().to_dict().items()} if not a.no_disk else None,
        "study_desc": m.StudyDesc.value_counts().to_dict(),
        "elapsed_s": round(time.time() - t0),
    }
    with open(C.NLST_META / "inventory_summary.json", "w") as f:
        json.dump(s, f, indent=1, default=str)
    md = ["# NLST local inventory\n", f"Patients: **{s['patients']}**, studies: {s['studies']}, series: {s['series_total']}, axial CT series: {s['series_axial_ct(>=30 files)']}\n",
          "\n## Patients by number of screening rounds with an axial CT\n", "| rounds | patients |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in s["patients_by_n_rounds_with_axial_ct"].items()]
    md += ["\n## Primary series: slice thickness (mm)\n", "| thickness | series |", "|---|---|"] + [f"| {k} | {v} |" for k, v in s["primary_thickness_mm"].items()]
    md += ["\n## Primary series: kernel class\n", "| class | series |", "|---|---|"] + [f"| {k} | {v} |" for k, v in s["primary_series_by_kernel_class"].items()]
    md += ["\n## Axial series: reconstruction kernel (top 25)\n", "| kernel | series |", "|---|---|"] + [f"| {k} | {v} |" for k, v in s["kernel_distribution_axial"].items()]
    md += ["\n## Primary series: manufacturer\n", "| manufacturer | series |", "|---|---|"] + [f"| {k} | {v} |" for k, v in s["primary_manufacturer"].items()]
    md += ["\n## On-disk verification\n", "| status | series |", "|---|---|"] + [f"| {k} | {v} |" for k, v in s["disk_status"].items()]
    (C.NLST_META / "inventory_tables.md").write_text("\n".join(md) + "\n")
    print(json.dumps(s, indent=1, default=str))


if __name__ == "__main__":
    main()
