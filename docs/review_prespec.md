# P2 review experiments (pre-specified): blur geometry, noise level, real kernels, second seed

Written 2 Nov 2026, before any of the data below was generated or scored. The SHA-256 hash of this file is recorded in
`analysis/p2_review/PRESPEC_SHA256.txt` before the first generation step; the time of that record is the time of this
plan. Any later change is listed at the end, with the date and the reason. These analyses answer six review points on
draft v2 (2 Nov). They are additional to the plan of 30 Oct and to the texture plan (`p2_texture_prespec.md`); none of
the earlier results is changed, and each family below has its own Holm correction. Only eligibility counts were looked
at before writing (NLST: 29,376 eligible exams), no outcome.

Common to all: same detectors (S3, S4 of folds 0-3; frozen files read only), same inference code and threshold (0.02),
same scorer (LUNA16 protocol, CPM), paired bootstrap with 10,000 common resamples over scans (LUNA16) or patients
(TCIA, NLST), two-sided P with floor 2/10,000, percentile 95 % CIs, α = 0.05.

## R1. Blur geometry (review point 2): in-plane blur (family F5)
The E1 blur is isotropic in voxels on the 1.25 × 0.703 × 0.703 mm grid: σ = 1.0 / 1.5 voxels is 1.25 / 1.88 mm along z
and 0.70 / 1.05 mm in-plane. Reconstruction kernels act in-plane. New conditions, applied exactly like the E1 blur
(preprocessed volume, same per-scan loading, separable Gaussian with reflect padding) but along y and x only:
- `xy1.0`: σ = 1.0 voxel in-plane (0.70 mm), none along z.
- `xy1.5`: σ = 1.5 voxels in-plane (1.05 mm), none along z.
- `xycal` (secondary): in-plane σ equal to the median calibrated σ of R3 (in mm, converted to voxels), rounded to 0.05 mm.
  If it equals 0.70 or 1.05 mm within 0.05 mm, it is still run and reported.

**F5 (Holm, m = 2):** ΔCPM = CPM(S4) − CPM(S3) at `xy1.0` and at `xy1.5`, LUNA16 folds 0-3 pooled (356 scans).
Secondary: each model's loss from clean; ΔCPM at `xycal`; the loss under in-plane blur relative to the loss under the
3D blur of the same σ. The text reports σ in mm everywhere from now on.

## R2. Noise level (review point 3): level response of the three textures (family F6)
The texture test matched the level of all textures to each scan's projection-noise SD at 10 % dose (factor 1). New
levels: factors 0.5 and 2 of the same per-scan target, for `tex_nps`, `tex_gauss` and `tex_white` (6 conditions), made
with the unchanged generator (same noise fields and seeds as factor 1, only the amplitude differs; two-pass matching,
±3 % tolerance, deviations reported). Same insertion path and inference as the texture test.

**F6 (Holm, m = 2):** for each model, CPM(`tex_nps`, factor 2) − CPM(`tex_gauss`, factor 0.5). Prediction > 0: real-NPS
noise at four times the SD (16 times the variance) still harms less than Gaussian-grain noise at the lower level.
Wording rule: significant for both models → "the spectrum mattered more than a four-fold change in level"; for one or
neither → the paper keeps "at equal level" and states that a four-fold level change was (partly) able to offset the
spectrum. Secondary: CPM against level (0, 0.5, 1, 2) per texture and model; the same contrast for white noise at
factor 2 minus Gaussian grain at factor 0.5; ΔCPM S4 − S3 at each condition.

## R3. Real reconstruction kernels in NLST (review points 4 and 6) (family F7)
**Sample.** NLST exams with two axial reconstructions of the same acquisition: the usable archive series (primary;
exclusions applied) with kernel class soft or sharp, and a partner series of the same study with the other class, the
same slice thickness and the same number of images. Thickness ≤ 2.5 mm (the LUNA16 range). If several partners
qualify, the one with the lowest series number. One exam per patient; shuffle with seed 0; up to 125 exams per vendor
(GE, Philips, Siemens, Toshiba): 500 exams, 1,000 series.
**Conversion.** Both series from raw DICOM to detector spacing, the LUNA16 path (`convert_native`), each cropped with the
primary series' archive lung box, so the two volumes cover the same world region. A pair is kept only if the two volumes
have the same shape, origin and spacing; the number dropped is reported. Conversion failures are reported.
**Inference.** All eight fold models (S3, S4 of folds 0-3) on both reconstructions.
**Outcome (no nodule positions in NLST).** For each model and exam, confident detections (score ≥ 0.5) on the soft and on
the sharp reconstruction are matched one-to-one (greedy by score, centre distance ≤ 5 mm). Kernel agreement of a model =
2 × matched / (confident soft + confident sharp), pooled over exams (ratio of sums); exams without confident detections
on either side contribute nothing.
**F7 (m = 1):** ΔA = A(S4) − A(S3), each averaged over the four fold models, bootstrap over patients. Prediction > 0.
Wording rule: significant → "augmentation made detections more consistent across real kernels"; not significant →
"augmentation did not measurably change the consistency across real kernels", with the CI.
**Secondary (descriptive, CIs, no P):** A per model and vendor; persistence = share of confident detections on one
kernel that have any candidate (≥ 0.02) within 5 mm on the other, in both directions; mean score difference
(soft − sharp) of matched candidates; candidates per scan by kernel; the same with fold-0 models only.
**Kernel calibration (descriptive).** Per exam, the in-plane Gaussian σ (mm; grid 0-3 mm, step 0.05) that, applied to the
sharp volume, minimises the mean squared difference to the soft volume over the cropped volume (windowed, normalised
as for the detector). Reported: median and IQR per vendor and overall. The overall median defines `xycal` (R1).
**Paper.** This analysis replaces the current NLST section as the screening-data result; the descriptive single-series
analysis moves to the supplement with a short paragraph.

## R4. Second training seed (review point 5)
Two new fold-0 models, trained with the unchanged code and configs and seed 1: S3 (`configs/stage1_retinanet.yaml`,
`fold0/seed1_s3_noaug`) and S4 (`configs/stage1_retinanet_aug.yaml`, `fold0/seed1_s4_aug`). The frozen `fold0/seed0`
is never written. Both are trained after the sampler fix, so the seed-1 S3 also answers the sampler-fix caveat.
Evaluated on the fold-0 test scans (89) under all E1 conditions, projection noise (3 doses), the three textures at
factor 1, `xy1.0` / `xy1.5`, and on TCIA (both doses).
**Reported (secondary, no new primary family):** (i) seed-to-seed difference in CPM of each configuration per condition
(seed 1 − seed 0) with paired CIs; (ii) ΔCPM S4 − S3 of the seed-1 pair next to the seed-0 pair; (iii) replication
check: the seed-1 pair's ΔCPM at blur σ 1.0 and 1.5 is positive with a CI excluding zero (stated as replicated or not);
(iv) sensitivity analysis of F1-F3: fold 0's S3 replaced by the seed-1 S3 (post-fix), everything else unchanged.

## Interpretation and paper (review points 1 and 6)
Title and abstract are rewritten after R1-R4, following the wording rules above. Claims are limited to what was tested:
Gaussian blur is called a model of softer kernels; "spectrum more than level" is used only if F6 supports it; the real-
kernel claim follows F7 and the calibration. Display items stay at 8 (NLST real kernels replace Fig. 5).

## Changes after this point
(none yet)
