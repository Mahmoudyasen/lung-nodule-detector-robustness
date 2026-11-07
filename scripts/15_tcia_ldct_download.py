"""Download the TCIA LDCT-and-Projection-data chest images, Siemens only (full + 10 % dose reconstructions).

    python -I scripts/15_tcia_ldct_download.py [--out $LUNGCAD_NLST_HDD/tcia_ldct] [--limit N]
    python -I scripts/15_tcia_ldct_download.py --region L --cases 10    # abdomen (B30f) pairs, for the soft-kernel noise texture

Public NBIA REST API, no login. Collection: https://doi.org/10.7937/9npb-2637 (CC BY 4.0; cite Moen et al., Med Phys 2021).
Layout: <out>/<PatientID>/<full|low>/<SOPInstanceUID-or-name>.dcm, plus <out>/series_manifest.json.
Resumable: a series counts as done when its folder holds exactly ImageCount files and has a .done marker.
Zips are extracted member by member with path checks (flat, .dcm only); the zip itself is deleted after extraction.
Only stdlib, so it runs with any Python.
"""
from __future__ import annotations
import os
import argparse, json, shutil, sys, time, urllib.request, zipfile
from pathlib import Path

API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
COLLECTION = "LDCT-and-Projection-data"
KIND = {"Full Dose Images": "full", "Low Dose Images": "low"}


def get(url: str, dest: Path | None = None, tries: int = 5):
    for t in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                if dest is None:
                    return r.read()
                with open(dest, "wb") as f:
                    shutil.copyfileobj(r, f, 1 << 20)
                return dest
        except Exception as e:
            print(f"  retry {t+1}/{tries}: {e}", flush=True); time.sleep(20 * (t + 1))
    raise RuntimeError(f"failed: {url}")


def series_list(region: str = "C", n_cases: int = 0) -> list[dict]:
    allser = json.loads(get(f"{API}/getSeries?Collection={COLLECTION}&format=json"))
    keep = [s for s in allser if s["PatientID"].startswith(region) and "siemens" in s["Manufacturer"].lower()
            and s["SeriesDescription"] in KIND]
    if n_cases:                                    # first n cases (sorted IDs) that have BOTH doses
        both = sorted(p for p in {s["PatientID"] for s in keep} if sum(s["PatientID"] == p for s in keep) == 2)[:n_cases]
        keep = [s for s in keep if s["PatientID"] in both]
    return sorted(keep, key=lambda s: (s["PatientID"], s["SeriesDescription"]))


def fetch(s: dict, out: Path) -> int:
    d = out / s["PatientID"] / KIND[s["SeriesDescription"]]
    if (d / ".done").exists() and len(list(d.glob("*.dcm"))) == s["ImageCount"]:
        return 0
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    z = d.parent / f".{d.name}.zip.part"
    get(f"{API}/getImage?SeriesInstanceUID={s['SeriesInstanceUID']}", z)
    with zipfile.ZipFile(z) as zf:
        for m in zf.infolist():
            name = Path(m.filename).name                         # flatten: never trust member paths
            if m.is_dir() or not name.lower().endswith(".dcm") or name.startswith("."):
                continue
            with zf.open(m) as src, open(d / name, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
    z.unlink()
    n = len(list(d.glob("*.dcm")))
    if n != s["ImageCount"]:
        raise RuntimeError(f"{s['PatientID']} {d.name}: {n} files, expected {s['ImageCount']}")
    (d / ".done").write_text(s["SeriesInstanceUID"] + "\n")
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.environ.get("LUNGCAD_NLST_HDD", "data/raw"), "tcia_ldct")); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--region", default="C", help="C chest, L abdomen"); ap.add_argument("--cases", type=int, default=0)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ser = series_list(a.region, a.cases)
    json.dump(ser, open(out / f"series_manifest{'' if a.region == 'C' else '_' + a.region}.json", "w"), indent=1)
    pats = sorted({s["PatientID"] for s in ser})
    print(f"{len(ser)} series, {len(pats)} patients, {sum(s['FileSize'] for s in ser)/1e9:.1f} GB", flush=True)
    if a.limit:
        ser = ser[:a.limit]
    t0 = time.time(); got = 0
    for i, s in enumerate(ser):
        n = fetch(s, out); got += s["FileSize"] if n else 0
        el = time.time() - t0
        print(f"{i+1}/{len(ser)} {s['PatientID']} {KIND[s['SeriesDescription']]:4s} {'downloaded ' + str(n) if n else 'already done'}"
              f"  {got/1e9:.1f} GB in {el/60:.0f} min", flush=True)
    print("all done", flush=True)
