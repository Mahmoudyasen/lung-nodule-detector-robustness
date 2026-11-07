"""Paths and shared constants. Override the data root with env LUNGCAD_DATA / LUNGCAD_NLST_HDD."""
from __future__ import annotations
import os
from pathlib import Path

THESIS_DATA = Path(os.environ.get("LUNGCAD_DATA", "data"))
NLST_HDD    = Path(os.environ.get("LUNGCAD_NLST_HDD", "data/raw"))

# LUNA16
LUNA_ZIPS  = THESIS_DATA / "luna16" / "zips"
LUNA_RAW   = THESIS_DATA / "luna16" / "raw"
LUNA_SEG   = LUNA_RAW / "seg-lungs-LUNA16"
LUNA_EVAL  = LUNA_RAW / "evaluationScript"
LUNA_INDEX = THESIS_DATA / "luna16" / "index"
LUNA_PRE   = THESIS_DATA / "luna16" / "preprocessed"

# NLST
NLST_DICOM     = NLST_HDD / "nlst"
NLST_METADATA  = NLST_HDD / "metadata" / "metadata.csv"
NLST_CLINICAL  = THESIS_DATA / "nlst" / "clinical" / "nlst_780"
NLST_META      = THESIS_DATA / "nlst" / "meta"
NLST_PRE       = THESIS_DATA / "nlst" / "preprocessed"
NLST_FARINA    = THESIS_DATA / "nlst" / "farina2025" / "nodule_location_globAttCRNN.csv"

# Experiments
WORK   = THESIS_DATA / "work"
STAGE1 = WORK / "wp0" / "stage1"
STAGE2 = WORK / "wp0" / "stage2"
RESULTS = WORK / "wp0" / "results"

# ---- The shared constants (S2). Changing any of these invalidates every trained model. ----
HU_MIN, HU_MAX = -1000.0, 400.0            # the ONE HU window
LUNA_SPACING = (1.25, 0.703125, 0.703125)  # (z, y, x) mm — detector spacing (MONAI LUNA16 tutorial)
NLST_SPACING = (1.5, 1.5, 1.5)             # (z, y, x) mm — NLST archive spacing; resampled to LUNA_SPACING at load
LUNG_MARGIN_MM = 10.0                      # crop margin around the lung mask
LUNA_SEG_LUNG_LABELS = (3, 4)              # seg-lungs-LUNA16: 3 = right lung, 4 = left lung, 5 = trachea/bronchi

SEEDS = (0, 1, 2)
FP_POINTS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)   # LUNA16 CPM operating points
