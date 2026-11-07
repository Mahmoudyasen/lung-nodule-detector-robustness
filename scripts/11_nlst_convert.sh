#!/bin/bash
# NLST conversion (I/O + CPU bound; runs alongside GPU training). Pilot first, inspect the montage, then the full queue.
#   bash scripts/11_nlst_convert.sh pilot     # 50 random primary series + montage PNG
#   bash scripts/11_nlst_convert.sh full      # all primary series, resumable, refuses to start without free space
source "$(dirname "$0")/00_env.sh"
case "${1:-pilot}" in
  pilot) $PY -m lungcad.data.nlst_convert --pilot 50 --workers ${WORKERS:-8} --montage ;;
  full)  nohup $PY -m lungcad.data.nlst_convert --workers ${WORKERS:-6} >> ${LUNGCAD_DATA:-data}/nlst/preprocessed/convert_stdout.log 2>&1 & echo "pid $!" ;;
esac
