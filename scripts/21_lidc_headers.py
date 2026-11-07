"""Acquisition metadata (vendor, model, kernel, kVp, mA, exposure, thickness, CTDIvol) for all 888 LUNA16 scans,
read from ONE original LIDC-IDRI DICOM image per series via TCIA's public NBIA API (LUNA16 .mhd files drop it).

    python -I scripts/21_lidc_headers.py        # `astra` env (pydicom); resumable; ~450 MB downloaded, nothing kept but the table

Output: <data>/luna16/index/lidc_headers.csv. Used to stratify E1 and to choose the texture of inserted noise.
"""
from __future__ import annotations
import os
import csv, io, json, sys, time, urllib.request
from pathlib import Path
import pydicom

API = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
IDX = Path(os.environ.get("LUNGCAD_DATA", "data"), "luna16", "index")
OUT = IDX / "lidc_headers.csv"
F = ["seriesuid", "Manufacturer", "ManufacturerModelName", "ConvolutionKernel", "KVP", "XRayTubeCurrent", "Exposure",
     "SliceThickness", "CTDIvol", "PixelSpacing"]


def get(url, tries=5):
    for t in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except Exception as e:
            print("  retry", t + 1, e, flush=True); time.sleep(10 * (t + 1))
    raise RuntimeError(url)


uids = [r["seriesuid"] for r in csv.DictReader(open(IDX / "luna16_index_v1.csv"))]
assert len(uids) == 888, len(uids)
done = set()
if OUT.exists():
    done = {r["seriesuid"] for r in csv.DictReader(open(OUT))}
new = not OUT.exists(); f = open(OUT, "a", newline=""); w = csv.DictWriter(f, F)
if new:
    w.writeheader()
def one(u):
    sop = json.loads(get(f"{API}/getSOPInstanceUIDs?SeriesInstanceUID={u}"))[0]["SOPInstanceUID"]
    d = pydicom.dcmread(io.BytesIO(get(f"{API}/getSingleImage?SeriesInstanceUID={u}&SOPInstanceUID={sop}")), stop_before_pixels=True)
    return {"seriesuid": u, **{k: str(d.get(k, "")) for k in F[1:]}}


from concurrent.futures import ThreadPoolExecutor      # the API answers each request in ~20 s; 8 in flight
with ThreadPoolExecutor(8) as ex:
    for n, row in enumerate(ex.map(one, [u for u in uids if u not in done]), 1):
        w.writerow(row); f.flush()
        if n % 50 == 0:
            print(n, "done", flush=True)
print("all done", len(uids))
