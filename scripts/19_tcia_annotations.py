"""TCIA LDCT chest (Siemens) lesion table -> LUNA16-style reference folder for lungcad.eval.froc. `astra` env.

    python scripts/19_tcia_annotations.py

Source: clinical/chest/Chest_v11 2026.xlsx, column "Location_(Pixel Coordinates)" = "R <radius px> c <col> <row> <slice>"
on the FULL-dose series; slice = DICOM InstanceNumber (checked: the lesion is the bright run of slices around it).
Six rows have slice = col - row (a broken spreadsheet formula: C050, C077, C179 L6/L7, C227, C258). Automatic recovery
was tried and FAILED validation (snapshot level + blob search found the known slice for only 23 of 36 verified lesions),
so their slice is NOT guessed: they are written to unresolved.csv with their world (x, y), and 22_tcia_e1.py ignores any
detection within their radius in-plane at any z (the LUNA16 'excluded finding' rule, as a column). One more lesion,
C128 L2, fails verification: its (x, y) nearly equals C128 L1's and lies in mediastinal soft tissue at the given slice,
while the snapshot shows a posterior left-lung lesion; patient C128 is therefore excluded from the primary analysis.
Verification of the other 35 (scripts/19b_tcia_lesion_review.py + visual check, 30 Oct 2026): 20 solid lesions sit on a
soft-tissue run with the blob peak at the given slice; 15 subsolid lesions confirmed visually (ground glass in the circle). World (x, y, z) mm = ImagePositionPatient of that instance + pixel
offsets along ImageOrientationPatient (= ITK/LUNA16 physical space). Diameter = 2 R x pixel spacing.
Full and 10 % images share their geometry, so each lesion is written for both pseudo-series <case>_full / <case>_low.
Output: <data>/tcia_ldct/eval/annotations/{annotations.csv, annotations_excluded.csv, seriesuids.csv} (the layout of
lungcad.eval.froc.load_reference) and eval/{lesions_checked.csv, montage_recovered.png}
"""
from __future__ import annotations
import os
import glob, re
from pathlib import Path
import numpy as np, pandas as pd, pydicom
from PIL import Image

RAW = Path(os.environ.get("LUNGCAD_NLST_HDD", "data/raw"), "tcia_ldct"); CLIN = RAW / "clinical" / "chest"
OUT = Path(os.environ.get("LUNGCAD_DATA", "data"), "tcia_ldct", "eval")
PAT = re.compile(r"R\s*([\d.]+)\s*c\s*([\d.]+)\s+([\d.]+)\s+(-?[\d.]+)")
UNVERIFIED = {("C128", 2): "(x, y) nearly equals C128 L1 and lies in mediastinal soft tissue at slice 25; the snapshot shows a "
                            "posterior left-lung lesion: position unreliable, patient excluded from the primary analysis"}


def series(case: str):
    ds = {int(d.InstanceNumber): d for d in (pydicom.dcmread(f) for f in glob.glob(str(RAW / case / "full" / "*.dcm")))}
    return ds


def hu(d) -> np.ndarray:
    return d.pixel_array * float(d.RescaleSlope) + float(d.RescaleIntercept)


def profile(ds, col, row, w=1):
    r, c = int(round(row)), int(round(col))
    return {i: float(hu(d)[r - w:r + w + 1, c - w:c + w + 1].mean()) for i, d in ds.items()}


def runs(prof, thr=-300.0):
    """Contiguous instance runs with HU > thr at the point -> list of (start, stop, centre)."""
    ks = sorted(prof); out, cur = [], []
    for k in ks:
        if prof[k] > thr:
            cur.append(k)
        elif cur:
            out.append(cur); cur = []
    if cur:
        out.append(cur)
    return [(r[0], r[-1], (r[0] + r[-1]) / 2) for r in out]


def snap_corr(ds, inst, col, row, snap, h=40):
    img = np.clip(hu(ds[inst]), -1350, 150).astype(np.float32)
    r, c = int(round(row)), int(round(col))
    a = img[max(r - h, 0):r + h, max(c - h, 0):c + h]; b = snap[max(r - h, 0):r + h, max(c - h, 0):c + h]
    a = (a - a.mean()) / (a.std() + 1e-6); b = (b - b.mean()) / (b.std() + 1e-6)
    return float((a * b).mean())


def world(d, col, row):
    ipp = np.array(d.ImagePositionPatient, float); iop = np.array(d.ImageOrientationPatient, float)
    dr, dc = (float(v) for v in d.PixelSpacing)            # row spacing, column spacing
    return ipp + col * dc * iop[:3] + row * dr * iop[3:]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    x = pd.read_excel(CLIN / "Chest_v11 2026.xlsx", sheet_name="Chest", header=1)
    x = x[x.Manufacturer.astype(str).str.lower().str.startswith("siemens")]
    cases = sorted(p.parent.parent.name for p in RAW.glob("C*/full/.done"))
    rows, checked = [], []
    for r in x.itertuples():
        R, col, row, sl = (float(v) for v in PAT.search(r._8).groups())
        ds = series(r.Case)
        broken = sl < 0 or not float(sl).is_integer()
        key = (r.Case, int(r.Lesion_ID)); d = next(iter(ds.values())) if broken else ds[int(sl)]
        w = world(d, col, row); diam = 2 * R * float(d.PixelSpacing[1])
        status = "slice_unknown" if broken else ("unverified" if key in UNVERIFIED else "verified")
        checked.append({"case": r.Case, "lesion": key[1], "status": status, "instance": None if broken else int(sl), "col": col, "row": row,
                        "diameter_mm": diam, "diagnosis": r.Diagnosis, "coordX": w[0], "coordY": w[1], "coordZ": None if broken else w[2],
                        "note": UNVERIFIED.get(key, "slice field = x - y (spreadsheet formula error)" if broken else "")})
        if status == "verified":
            for dose in ("full", "low"):
                rows.append({"seriesuid": f"{r.Case}_{dose}", "coordX": w[0], "coordY": w[1], "coordZ": w[2], "diameter_mm": diam})
    A = OUT / "annotations"; A.mkdir(parents=True, exist_ok=True)          # layout lungcad.eval.froc.load_reference expects
    pd.DataFrame(rows).to_csv(A / "annotations.csv", index=False)
    pd.DataFrame(columns=["seriesuid", "coordX", "coordY", "coordZ", "diameter_mm"]).to_csv(A / "annotations_excluded.csv", index=False)
    pd.DataFrame({"seriesuid": [f"{c}_{d}" for c in cases for d in ("full", "low")]}).to_csv(A / "seriesuids.csv", index=False, header=False)
    ck = pd.DataFrame(checked); ck.to_csv(OUT / "lesions_checked.csv", index=False)
    ck[ck.status != "verified"].to_csv(OUT / "unresolved.csv", index=False)
    excl = sorted({c for c, _ in UNVERIFIED}); pd.Series(excl, name="case").to_csv(OUT / "excluded_patients.csv", index=False)
    print(ck[["case", "lesion", "status", "instance", "diameter_mm", "diagnosis"]].to_string())
    print(ck.status.value_counts().to_dict(), f"| {len(cases)} cases; excluded from primary: {excl}")


if __name__ == "__main__":
    main()
