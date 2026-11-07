# P2 task 5: noise-texture test (pre-specified)

Written 2 Nov 2026, ~17:15, before any NPS-matched noise was generated or scored. The SHA-256 hash of this file is
recorded in `analysis/p2_texture/PRESPEC_SHA256.txt` before the generation step. Any later change to this plan is
listed at the end, with the date and the reason.

## Why
In the draft, the claim that noise texture explains the different effects of white, correlated and projection-domain
noise is argued but not tested. The three noise models differ in texture, level, spatial uniformity and insertion
path all at once. For example, "80 HU" image-domain noise adds 67 HU over the lung crop, while projection noise at
10 % dose adds a median of about 102 HU. This test holds the level and the insertion path fixed and varies only the
texture.

## Conditions (LUNA16 test folds 0-3, 356 scans; S3 and S4 of each fold; paired, identical inputs)
All noise is added to the ORIGINAL full-field LUNA16 image (HU, native grid). It then goes through the unchanged
preprocessing with the same lung mask, exactly the path used for the projection-domain noise
(`scripts/17_luna_projnoise.py`). Seeds are fixed per (scan, condition). The noise is stationary, zero-mean,
Gaussian and additive.

- **T1 `tex_nps`**: in-plane noise power spectrum (NPS) equal to the measured NPS of the real TCIA noise
  (10 %-dose minus routine image). The B50f chest pairs give the spectrum for sharp-kernel scans and the B30f
  abdomen pairs for soft-kernel scans, the same kernel-class split as the projection noise. Slices are independent
  along z, as in the projection noise. The radial NPS is measured in cycles/mm and applied at each scan's own pixel
  spacing. It is zero above the highest frequency measured on TCIA.
- **T2 `tex_gauss`**: the texture of the correlated image-domain noise of the draft: white noise low-pass filtered
  with a 3D Gaussian of σ = 1.41 mm in-plane and 2.5 mm along z (2 voxels on the detector grid; the "1.4 mm grain"
  condition), here in physical units on the native grid.
- **T3 `tex_white`**: white noise, independent per native voxel.

**Level:** for every scan and condition, the added-noise SD over the preprocessed lung crop (preprocessed noisy
volume minus the clean archive) is matched to that scan's projection-domain noise SD at 10 % dose
(`noise_sd_added_hu` in `projnoise/a0.1/.../<uid>.json`). The scale is set in two passes; the accepted tolerance is
±3 % per scan, and any scan outside it is reported. T1-T3 and proj0.1 therefore have the same level, crop and
insertion path. They differ in texture (T1 vs T2 vs T3) or, for T1 vs proj0.1, only in the spatial non-uniformity
and streak structure of the projection noise.

## Known-answer checks before scoring
1. The generated T1 noise, measured with the same ROI method, reproduces the target radial NPS shape (normalised L1
   distance < 0.05). The generated T2 and T3 noise have the expected flat or Gaussian shape.
2. With zero amplitude, the output is byte-identical to the clean archive (3 scans).
3. Every scan's achieved SD is within ±3 % of its target.
4. The new inference path reproduces the existing proj0.1 candidates of one scan exactly.

## Primary family F4 (Holm within the family, α = 0.05, m = 2)
For each model M ∈ {S3, S4}: contrast CPM_M(T1) − CPM_M(T2), on the same paired bootstrap as F1-F3 (10,000
resamples over scans, common resamples, two-sided P, floor 2/10,000). The texture hypothesis predicts > 0: at equal
level and path, noise with the real texture harms less than Gaussian-blob noise.

## Pre-specified equivalence check (secondary, not in Holm)
For each model, CPM_M(T1) − CPM_M(proj0.1), 95 % CI inside ±0.02 → "stationary noise with the real NPS reproduces the
projection-domain result".

## Secondary (reported, unadjusted)
- Each model's drop from clean under T1, T2 and T3 (paired CIs).
- CPM_M(T3) − CPM_M(T1).
- ΔCPM S4 − S3 per condition.
- The above by native kernel class.
- Candidates per scan.
- Measured TCIA noise correlation between adjacent slices (descriptive; checks the z-independence assumption).

## Interpretation rule (fixed now; the paper's wording follows it)
- **Supported**: F4 significant for both models AND the equivalence check holds for both. Texture at equal level
  explains most of the difference, and stationary NPS-matched noise suffices to simulate dose.
- **Partly supported**: F4 significant for at least one model, but the equivalence check fails. Texture matters, but
  so does the non-stationary or streak structure of projection noise; NPS-matched stationary noise is not a full
  substitute.
- **Not supported**: F4 significant for neither model. At equal level and path, texture does not explain the
  difference; the draft's explanation must be withdrawn.

The result is reported whichever way it goes.

## Changes after this point
(none yet)
