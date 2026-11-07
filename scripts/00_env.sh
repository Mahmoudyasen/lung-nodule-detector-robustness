#!/bin/bash
# Environment used by every shell script. Set PY to the Python of an environment with the packages in requirements.txt,
# LUNGCAD_DATA to the working data root and LUNGCAD_NLST_HDD to the raw-data root (see README.md).
export PY=${PY:-python}
export LUNGCAD_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONUNBUFFERED=1
cd "$LUNGCAD_ROOT"
