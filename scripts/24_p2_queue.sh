#!/bin/bash
# P2 full plan, mechanical GPU/CPU steps after the projection-noise calibration is accepted (2 Nov 2026), one job at a time:
#   1. projection-domain low-dose LUNA16 test scans, 3 doses, GPU projector (astra env; calibration calib_all_cuda.json)
#   2. E1 on those scans, S3 vs S4, folds 0-3 (condition proj<a>)
#   3. NLST native-resolution conversion of the rest of the fixed 1,955-series sample (CPU)
#   4. NLST descriptive inference, S3 and S4 of fold 0
# Every step is resumable and the queue STOPS at the first failure; relaunching the same command continues where it stopped.
#   systemd-run --user --unit=lungcad-p2q -p OOMPolicy=continue --working-directory=/path/to/repo bash scripts/24_p2_queue.sh
source "$(dirname "$0")/00_env.sh"
ASTRA=${ASTRA_PY:-python}   # Python of the astra environment (requirements-astra.txt)
LOG=${LUNGCAD_DATA:-data}/work/wp0/analysis/p2_queue.log
step() {
  echo "$(date '+%F %T') START $1" | tee -a "$LOG"
  if ! "${@:2}" >> "$LOG" 2>&1; then echo "$(date '+%F %T') FAILED $1 - queue stopped" | tee -a "$LOG"; exit 1; fi
  echo "$(date '+%F %T') DONE $1" | tee -a "$LOG"
}
step "1 projnoise scans" $ASTRA -u scripts/17_luna_projnoise.py --cuda --procs ${PROCS:-4}
step "2 E1 projnoise"    $PY -u scripts/23_projnoise_e1.py
step "3 NLST convert"    $PY -u scripts/20_nlst_native_sample.py --procs 4
step "4 NLST inference"  $PY -u scripts/18_nlst_descriptive.py --folds 0
echo "$(date '+%F %T') P2 queue: all done" | tee -a "$LOG"
