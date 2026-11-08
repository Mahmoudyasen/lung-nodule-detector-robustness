#!/bin/bash
# Review experiments R1-R4 (lung-nodule-detector-analysis-plans/plans_original/review_prespec.md, hash in analysis/p2_review/PRESPEC_SHA256.txt), one job at a time:
#   1-2  NLST real-kernel pairs: convert (CPU), kernel calibration (CPU)                                  [R3]
#   3    in-plane blur xy1.0 / xy1.5 / xycal, folds 0-3 (GPU)                                              [R1]
#   4-5  texture levels x0.5 and x2 (CPU generation on the raw-data disk, $LUNGCAD_NLST_HDD), inference folds 0-3 (GPU)           [R2]
#   6    NLST pairs, 8 fold models (GPU)                                                                   [R3]
#   7-8  second seed: S3 and S4 of fold 0, seed 1 (GPU, ~19 h each; never writes fold0/seed0)              [R4]
#   9    seed-1 evaluation: E1, projection noise + textures, in-plane blur, TCIA (GPU)                     [R4]
# Every step is resumable and the queue STOPS at the first failure; relaunching the same command continues.
#   systemd-run --user --unit=lungcad-review -p OOMPolicy=continue --working-directory=/path/to/repo bash scripts/34_p2_review_queue.sh
source "$(dirname "$0")/00_env.sh"
A=${LUNGCAD_DATA:-data}/work/wp0
LOG=$A/analysis/p2_review/queue.log
LEV=${LUNGCAD_NLST_HDD:-data/raw}/p2_review/texture_levels
mkdir -p "$(dirname "$LOG")" "$LEV"
step() {
  if grep -q "DONE $1\$" "$LOG" 2>/dev/null; then echo "$(date '+%F %T') skip (done) $1" | tee -a "$LOG"; return; fi
  echo "$(date '+%F %T') START $1" | tee -a "$LOG"
  if ! "${@:2}" >> "$LOG" 2>&1; then echo "$(date '+%F %T') FAILED $1 - queue stopped" | tee -a "$LOG"; exit 1; fi
  echo "$(date '+%F %T') DONE $1" | tee -a "$LOG"
}
train() {   # $1 config, $2 out dir name
  local OUT=$A/stage1/fold0/$2; mkdir -p "$OUT"
  [ "$2" != "seed0" ] || { echo "refusing to write the frozen fold0/seed0"; return 1; }
  [ -f "$OUT/done.json" ] && return 0
  $PY -m lungcad.stage1.train --config "$1" --fold 0 --seed 1 --out "$OUT" --resume >> "$OUT/stdout.log" 2>&1 || return 1
  [ -f "$OUT/done.json" ]
}
step "1 NLST pairs convert"      $PY -u scripts/30_nlst_kernel_pairs.py convert --procs 4
step "2 NLST pairs calibrate"    $PY -u scripts/30_nlst_kernel_pairs.py calibrate --procs 8
step "3 in-plane blur"           $PY -u scripts/31_inplane_blur_e1.py
step "4a texture levels x0.5"    $PY -u scripts/26_luna_texture.py --factor 0.5 --root $LEV --procs 8
step "4b texture levels x2"      $PY -u scripts/26_luna_texture.py --factor 2 --root $LEV --procs 8
step "5a levels inference x0.5"  $PY -u scripts/27_texture_e1.py --factor 0.5 --root $LEV
step "5b levels inference x2"    $PY -u scripts/27_texture_e1.py --factor 2 --root $LEV
step "6 NLST pairs inference"    $PY -u scripts/30_nlst_kernel_pairs.py infer --folds 0 1 2 3
step "7 train S3 fold0 seed1"    train configs/stage1_retinanet.yaml seed1_s3_noaug
step "8 train S4 fold0 seed1"    train configs/stage1_retinanet_aug.yaml seed1_s4_aug
step "9a seed1 E1"               $PY -u scripts/33_seed1_eval.py e1
step "9b seed1 proj + textures"  $PY -u scripts/27_texture_e1.py --seed1 --conds proj0.5 proj0.25 proj0.1 tex_nps tex_gauss tex_white
step "9c seed1 in-plane blur"    $PY -u scripts/31_inplane_blur_e1.py --seed1
step "9d seed1 TCIA"             $PY -u scripts/33_seed1_eval.py tcia
echo "$(date '+%F %T') review queue: all done" | tee -a "$LOG"
