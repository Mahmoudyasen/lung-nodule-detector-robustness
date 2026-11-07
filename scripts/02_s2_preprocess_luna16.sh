#!/bin/bash
# S2 — run all 888 scans through the shared module (CPU, ~1-2 h with 10 workers), then the round-trip test.
source "$(dirname "$0")/00_env.sh"
$PY -m lungcad.data.luna16_preprocess --workers ${WORKERS:-10} && $PY tests/test_preprocess_roundtrip.py
