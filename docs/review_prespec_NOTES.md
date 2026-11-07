# Notes on the review pre-specification (`review_prespec.md`)

The plan file is unchanged (SHA-256 in `analysis/p2_review/PRESPEC_SHA256.txt`). Everything below was decided after the
data of the step concerned had been generated or scored, and is labelled "not pre-specified" in the paper.

## Execution (as planned)
- R3 sample: 500 exams (125 per vendor). Conversion: 0 failures. Geometry check: 102 pairs dropped (Philips 80,
  Siemens 14, GE 7, Toshiba 1), almost all with a different shape/origin (different reconstruction field of view);
  398 exams analysed (GE 118, Philips 45, Siemens 111, Toshiba 124). Calibration: median 0.50 mm (IQR 0.45-0.55), no
  exam at the grid edge; `xycal` = 0.50 mm = 0.711 voxels in-plane.
- R1, R2, R3 primary results as pre-specified: F5 both significant, F6 both significant (wording rule: "the spectrum
  mattered more than a four-fold change in level"), F7 significant.

## Additions after seeing the data (3 Nov)
1. **R2 level shortfall at factor 2.** The two-pass level matching measures the SD over the preprocessed lung crop, after
   the clip to [-1000, 400] HU. At factor 2 (median target 204 HU) the clip limited the achievable level: 138 (real NPS),
   156 (white) and 88 (Gaussian grain) of 356 scans fell outside +-3 % (largest -17.7 %); the achieved median was 1.9 %
   (ratio of medians, real NPS) below the target. At factors 0.5 and 1 all or nearly all scans were within tolerance.
   Sensitivity analysis (not pre-specified; `32_review_stats.py levels`, `levels_sens_within3pct.csv`): F6 restricted to
   the 218 scans whose real-NPS x2 noise was within +-3 % (all Gaussian x0.5 scans were). Result: the contrast REVERSED,
   S3 -0.054 [-0.094, -0.012], S4 -0.061 [-0.099, -0.020]. See point 3 for the interpretation.
   Exploration (scratch, then made reproducible as `32_review_stats.py levelskernel` -> `levels_by_kernel.csv`): the 138
   scans outside tolerance are 136 sharp-kernel scans (median x2 target 341 HU vs 164 HU for the kept scans), so the
   restriction is in effect a restriction to soft-kernel scans. F6 by native kernel class: sharp (151 scans, B50f spectrum)
   S3 +0.177, S4 +0.165; soft (205 scans, B30f spectrum) S3 -0.045, S4 -0.061. Splitting at the median target level gives
   the same pattern. A level shortfall of about 2 % cannot explain a swing of this size. Caveat: kernel class is
   confounded with absolute level (median x2 target 327 HU sharp vs 160 HU soft, because the target follows each scan's
   projection noise), so the split cannot attribute the difference to the spectrum; the paper says so.
2. **R3 by-vendor CIs.** The plan lists "A per model and vendor" as secondary without fixing the estimator. Added
   `32_review_stats.py nlstvendor`: the F7 estimator (mean over fold models of S4 - S3) within each vendor, bootstrap over
   that vendor's exams (10,000, seed 0); known-answer check: the "all" row reproduces F7 exactly (estimate and CI).
3. **Interpretation of F6.** The pre-specified result (positive for both models) is reported as such, with its rule-based
   wording, but the paper states next to it that the average hides opposite results by kernel class: the sharp-kernel
   spectrum (at a higher absolute level) beat a four-fold level change, the soft-kernel spectrum did not; spectrum and
   absolute level cannot be separated in this split. The abstract, introduction,
   discussion and conclusion no longer claim that texture matters "more than level" in general. F4 (equal level) holds
   in both kernel classes (soft +0.44/+0.40, sharp +0.68/+0.68), so the texture headline (title) stands.

## Paper
- Figure 2c shows blur against in-plane sigma in mm (3D and in-plane-only); Figure 2e shows the level response (replaces
  the factor-1 texture panel, whose numbers stay in Table 3 and Supplementary Table S6). Figure 5 is the NLST real-kernel
  analysis; the single-series NLST analysis moved to Supplementary Note 1 and Table S2.

## Second review round (3 Nov, after an outside reading of draft v3)
4. **R3 threshold sensitivity (not pre-specified).** S4 scores are shifted slightly lower, so agreement at the fixed 0.5
   threshold could change through calibration alone. `32_review_stats.py nlstthr` -> `nlst_pairs_threshold.csv`: same
   estimator and bootstrap as F7 at thresholds 0.3/0.5/0.7, plus a count-matched variant (each S4 fold model thresholded
   so that its total confident detections over all series equal those of S3 of the same fold at 0.5; S4 thresholds
   0.50-0.62, fold 1 0.22). Known answer: the 0.5 row reproduces F7. Result: +0.023 / +0.021 / +0.023 and +0.020
   [0.011, 0.029] count-matched; all fold differences positive in every variant. Confident detections per series at 0.5:
   S3 2.85, S4 2.76.
5. **Shortfall by kernel class.** `levelskernel` now also writes the median and 90th-percentile shortfall of the real-NPS x2
   level (sharp 4.4 % / 7.2 %, soft 1.0 % / 1.7 %); estimates and CIs unchanged (checked against the previous file).
   The shortfall biases the sharp-kernel F6 contrast towards positive but, on the level curve, is worth far less than its
   size; the abstract no longer states the kernel-class direction.
6. Text corrections: the sentence "augmentation did not change the loss under real-NPS noise at any level" was false
   (x2: +0.036 [0.009, 0.059]) and is replaced; the Discussion's "87 % reproduced" (a Dice-style agreement) is replaced by
   the persistence; "consistently across manufacturers" removed (single fold pairs negative for GE, Philips, Toshiba).

## R4 second seed (5 Nov, executed as planned)
Queue finished 05:48 (S3 seed 1 trained 3 Nov 11:44 - 4 Nov 06:30, S4 seed 1 4 Nov 06:30 - 5 Nov 01:10, evaluation
01:10 - 05:48). `32_review_stats.py seed` -> seed_contrasts.csv, seed_sensitivity_f1f3.csv, seed_results.json.
(iii) Replication (pre-specified criterion, both strengths): NOT met in full. sigma 1.5 voxel (1.05 mm): seed 1 +0.077
[0.027, 0.126] (seed 0 +0.103); sigma 1.0 (0.70 mm): +0.027 [-0.008, 0.064] (seed 0 +0.048). Paper: "met only in part".
(i) Seed differences: clean 0.019 (S3) / 0.004 (S4); median over conditions 0.017 / 0.010; max 0.089 / 0.037, the S3
maximum under correlated noise 80 HU / 1.4 mm (seed-1 S3 more robust, +0.089 [0.045, 0.137]).
(ii) Seed-1 pair: S4 - S3 negative under correlated 80 HU noise (-0.062, -0.075) and Gaussian grain (-0.048).
(iv) Sensitivity (fold-0 S3 replaced by seed-1 S3): F1 all still significant (P_Holm 0.0006); F3 unchanged (ns); F2
correlated 80 HU / 1.4 mm no longer significant (+0.022 [-0.003, 0.044], P_Holm 0.64; main +0.042, P_Holm 0.0014).
The two S3 models differ in seed AND sampler fix; the analysis cannot separate them. Paper: Results (correlated-noise
sentence, seed section), Discussion (noise benefits not robust; blur benefit reproduced at 1.05 mm, CI includes zero
at 0.70 mm) and limitations rewritten from these numbers.
Technical: make_numbers.py sensitivity keys for the two F3 rows renamed (rv.sens.F3DD, rv.sens.F3Dlow) because the raw
condition names contain characters that break LaTeX csnames.
Paper after the seed results (5 Nov, outside reading): abstract now states that the correlated-noise benefit did not
survive retraining a baseline and limits the blur claim to strong blur; Methods state the replication criterion and the
sensitivity analysis; Table 3, Fig. 3 and Table S1 flag the correlated-noise 80 HU / 1.4 mm result with a dagger; the
seed-1 in-plane 0.70 mm result (-0.008, no benefit) is stated; the 0.089 seed difference is compared with the seed-0
benefit on the same 89 scans (+0.033), not with the pooled effect; real-NPS / white x2 benefits marked unreplicated.
