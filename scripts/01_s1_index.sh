#!/bin/bash
# S1 — build the versioned LUNA16 index + folds + LIDC exclusion list (no GPU). Requires luna16/raw extracted.
source "$(dirname "$0")/00_env.sh"
$PY -m lungcad.data.luna16_index
