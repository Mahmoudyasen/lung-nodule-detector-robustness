"""CTDIvol of the TCIA Siemens chest cases from the data providers' clinical table (astra env: needs openpyxl).
Output: paper/data/tcia_ctdi.json"""
import os
import json, pandas as pd
from pathlib import Path
x = pd.read_excel(os.path.join(os.environ.get("LUNGCAD_NLST_HDD", "data/raw"), "tcia_ldct/clinical/chest/Chest_v11 2026.xlsx"), sheet_name="Chest", header=1)
s = x[x.Manufacturer.astype(str).str.strip().str.lower() == "siemens"].drop_duplicates("Case")
c = pd.to_numeric(s["CTDIvol_(mGy)"], errors="coerce").dropna()
age = pd.to_numeric(s["Age_(years)"].astype(str).str.extract(r"(\d+)")[0], errors="coerce").dropna()
out = {"n_siemens_cases_in_table": int(len(s)), "n_ctdi": int(len(c)), "ctdi_min": float(c.min()), "ctdi_max": float(c.max()),
       "ctdi_median": float(c.median()), "age_median": float(age.median()), "age_min": float(age.min()), "age_max": float(age.max()), "n_age": int(len(age))}
json.dump(out, open(Path(__file__).with_name("data") / "tcia_ctdi.json", "w"), indent=1); print(out)
