#!/bin/bash
# S4 (augmented arm), all ten folds. Run as:  CFG=configs/stage1_retinanet_aug.yaml bash scripts/05_s5_all_folds.sh
# (the run manifests of the published models record augment.dose_kernel: true).
# S5 — the ten-fold queue, back to back, unattended. Uses the committed S4 config (configs/stage1_retinanet_aug.yaml).
# Each fold: train -> score its test split. A fold counts as done only when done.json AND its test candidate file exist.
# Safety: the queue STOPS at the first failure (so a half-trained fold can never be scored or frozen); relaunching the
# same command resumes that fold from last.pt. If training finished but scoring did not, only the scoring is redone.
source "$(dirname "$0")/00_env.sh"
CFG=${CFG:-configs/stage1_retinanet.yaml}
ALL="0 1 2 3 4 5 6 7 8 9"
for k in ${FOLDS:-$ALL}; do
  OUT=${LUNGCAD_DATA:-data}/work/wp0/stage1/fold$k/seed0; mkdir -p "$OUT"
  scored() { ls "$OUT"/candidates_fold${k}_test_*.csv >/dev/null 2>&1; }
  if [ -f "$OUT/done.json" ] && scored; then echo "fold $k done"; continue; fi
  if [ ! -f "$OUT/done.json" ]; then
    echo "$(date '+%F %T') fold $k: training ($CFG)" | tee -a "$OUT/stdout.log"
    if ! $PY -m lungcad.stage1.train --config "$CFG" --fold $k --seed 0 --resume >> "$OUT/stdout.log" 2>&1; then
      echo "$(date '+%F %T') fold $k: TRAINING FAILED - queue stopped. Relaunch the same command to resume." | tee -a "$OUT/stdout.log"; exit 1
    fi
    [ -f "$OUT/done.json" ] || { echo "fold $k: training ended without done.json - queue stopped" | tee -a "$OUT/stdout.log"; exit 1; }
  fi
  echo "$(date '+%F %T') fold $k: scoring the test split" | tee -a "$OUT/stdout.log"
  if ! $PY -m lungcad.stage1.infer --checkpoint "$OUT/best.pt" --fold $k --split test --score >> "$OUT/stdout.log" 2>&1 || ! scored; then
    echo "$(date '+%F %T') fold $k: TEST SCORING FAILED - queue stopped. Relaunch to redo the scoring only." | tee -a "$OUT/stdout.log"; exit 1
  fi
done
# freeze: only when the full ten-fold queue was requested
if [ "${FOLDS:-$ALL}" = "$ALL" ]; then
  $PY scripts/build_own_candidates.py
else
  echo "FOLDS=${FOLDS}: partial queue, own-candidate freeze not run"
fi
