#!/bin/bash
# S3 (no dose/kernel augmentation) retrained on folds 1-3, so E1 can compare S3 vs. S4 on 4 folds.
# Same config as the fold-0 gate (configs/stage1_retinanet.yaml); S4 for these folds is the ten-fold S4 run (fold$k/seed0).
# Output goes to fold$k/seed0_s3_noaug, NEVER to fold$k/seed0 (the frozen detector). No test scoring here: E1 scores itself.
# Safety: the queue STOPS at the first failure; relaunching the same command resumes that fold from last.pt.
#   systemd-run --user --unit=lungcad-p2s3 -p OOMPolicy=continue --working-directory=/path/to/repo bash scripts/14_p2_s3_folds.sh
source "$(dirname "$0")/00_env.sh"
CFG=configs/stage1_retinanet.yaml
for k in ${FOLDS:-1 2 3}; do
  OUT=${LUNGCAD_DATA:-data}/work/wp0/stage1/fold$k/seed0_s3_noaug; mkdir -p "$OUT"
  if [ -f "$OUT/done.json" ]; then echo "fold $k done"; continue; fi
  echo "$(date '+%F %T') fold $k: training S3 ($CFG)" | tee -a "$OUT/stdout.log"
  if ! $PY -m lungcad.stage1.train --config "$CFG" --fold $k --seed 0 --out "$OUT" --resume >> "$OUT/stdout.log" 2>&1; then
    echo "$(date '+%F %T') fold $k: TRAINING FAILED - queue stopped. Relaunch the same command to resume." | tee -a "$OUT/stdout.log"; exit 1
  fi
  [ -f "$OUT/done.json" ] || { echo "fold $k: training ended without done.json - queue stopped" | tee -a "$OUT/stdout.log"; exit 1; }
done
echo "$(date '+%F %T') S3 folds ${FOLDS:-1 2 3}: all done"
