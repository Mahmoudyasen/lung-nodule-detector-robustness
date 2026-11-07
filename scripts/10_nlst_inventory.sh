#!/bin/bash
# Step 0 — NLST local inventory (disk verification of all 203k series, ~30-60 min with 16 threads on the NTFS HDD).
source "$(dirname "$0")/00_env.sh"
$PY -m lungcad.data.nlst_inventory --threads 16
