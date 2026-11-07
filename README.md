# lung-nodule-detector-robustness

Code for a paired evaluation of training-time dose and blur (simulated kernel) augmentation for a 3D RetinaNet
lung-nodule detector trained on LUNA16. Two detectors are trained per cross-validation fold with identical data,
sampling, initialisation and schedule: **S3** without and **S4** with dose/blur augmentation. Each pair is compared on
identical inputs under image-domain noise (white and spatially correlated), Gaussian blur and unsharp masking,
projection-domain noise calibrated on the TCIA LDCT-and-Projection-data pairs, and stationary noise with the measured
real noise power spectrum (noise-texture test). The pairs are also compared on paired full- and 10 %-dose images of the
same patients (TCIA), and described on a stratified sample of NLST screening scans. All statistics follow a pre-specified
plan: paired bootstrap over scans or patients, 10,000 common resamples, and Holm correction within each primary family.

Paper: *[title and DOI to be added on publication]*. Derived results (candidate lists, calibration outputs, statistics):
*[Zenodo DOI to be added]*.

## Environments

| Environment | Used for | Install |
|---|---|---|
| main (Python 3.12) | training, inference, analysis, figures | `pip install -r requirements.txt` (PyTorch with CUDA from pytorch.org) |
| astra (Python 3.12) | projection-domain noise (`proj_noise.py`, `calibrate_tcia.py`, `17_luna_projnoise.py`), LIDC header and TCIA download scripts | `conda install -c astra-toolbox -c conda-forge astra-toolbox=2.5.0`, then `pip install -r requirements-astra.txt` |

Hardware used: one NVIDIA GeForce RTX 3090 (24 GB). Training one detector takes about 19 h (200 epochs).

## Data and paths

All data are public under the providers' Creative Commons terms (check each source; cite the providers):

| Data set | Source | Used for |
|---|---|---|
| LUNA16 | Zenodo [10.5281/zenodo.3723295](https://doi.org/10.5281/zenodo.3723295), [10.5281/zenodo.4121926](https://doi.org/10.5281/zenodo.4121926) | training, perturbation tests |
| LIDC-IDRI DICOM headers | TCIA [10.7937/K9/TCIA.2015.LO9QL9SX](https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX) | kernel, vendor and slice thickness of each LUNA16 scan (`21_lidc_headers.py`) |
| LDCT-and-Projection-data | TCIA [10.7937/9NPB-2637](https://doi.org/10.7937/9NPB-2637) | noise calibration, paired full/10 %-dose test |
| NLST CT | TCIA [10.7937/TCIA.HMQ8-J677](https://doi.org/10.7937/TCIA.HMQ8-J677) | descriptive screening check |

Two environment variables set the data roots (defaults in `lungcad/config.py`):

```bash
export LUNGCAD_DATA=/path/to/working-data      # preprocessed volumes, runs (work/wp0/...), results
export LUNGCAD_NLST_HDD=/path/to/raw-data      # raw downloads (NLST DICOM, tcia_ldct/)
export PY=/path/to/main-env/bin/python         # used by the shell scripts
export ASTRA_PY=/path/to/astra-env/bin/python  # used by scripts/24_p2_queue.sh
```

The expected layout of `LUNGCAD_DATA` (LUNA16 `.mhd` files, lung masks, index) is defined in `lungcad/config.py`.

## Run order

| Step | Command | Output |
|---|---|---|
| 1. LUNA16 index and preprocessing | `bash scripts/01_s1_index.sh`; `bash scripts/02_s2_preprocess_luna16.sh` | resampled, lung-cropped volumes |
| 2. LUNA16 acquisition headers | `python -I scripts/21_lidc_headers.py` (astra env) | kernel / vendor / thickness per scan |
| 3. S3, fold 0 | `bash scripts/03_s3_fold0_gate.sh` | `stage1/fold0/seed0_s3_gate_noaug` |
| 4. S4, all ten folds | `CFG=configs/stage1_retinanet_aug.yaml bash scripts/05_s5_all_folds.sh` | `stage1/fold{k}/seed0`, merged test candidates |
| 5. S3, folds 1-3 | `bash scripts/14_p2_s3_folds.sh` | `stage1/fold{k}/seed0_s3_noaug` |
| 6. Image-domain perturbations | `python scripts/e1_dose_robustness.py` (fold 0); `python scripts/e1_multifold.py --folds 1 2 3` | candidates per condition |
| 7. TCIA download, annotations, preprocessing | `15_tcia_ldct_download.py` (chest; `--region L --cases 10` for abdomen), `19_tcia_annotations.py`, `19b_tcia_lesion_review.py`, `16_tcia_ldct_preprocess.py` | paired volumes and checked lesion list |
| 8. Noise calibration | `python scripts/calibrate_tcia.py --cases all --cuda`; `--cases abdomen --cuda` (astra env) | `tcia_ldct/calibration/` |
| 9. Projection noise, NLST sample | `bash scripts/24_p2_queue.sh` (runs `17_luna_projnoise.py`, `23_projnoise_e1.py`, `20_nlst_native_sample.py`, `18_nlst_descriptive.py`; NLST inventory and conversion first: `10_nlst_inventory.sh`, `11_nlst_convert.sh`) | projection-noise scans and candidates; NLST candidates |
| 10. TCIA inference | `python scripts/22_tcia_e1.py` | TCIA contrasts (F3) |
| 11. Main statistics | `python scripts/p2_e1_final.py` | F1/F2 tables (`analysis/e1_dose_pooled/final_*.csv`) |
| 12. Noise-texture test | `python scripts/25_tcia_nps.py`; `python scripts/26_luna_texture.py --selftest`, then without `--selftest`; `python scripts/27_texture_e1.py --check`, then without `--check`; `python scripts/28_texture_stats.py` | F4 (pre-specified in `analysis_plans/texture_test_prespec.md`) |
| 13. Secondary analyses | `python scripts/29_p2_secondary.py`; `python scripts/p2_nlst_analysis.py`; `python scripts/p2_nlst_figure.py` | kernel/vendor subsets, sensitivity analysis, NLST |
| 14. Missed-nodule review | `python scripts/e5_miss_review.py`; `python scripts/e5b_miss_low_threshold.py` | Supplementary table of misses |
| 15. Review experiments (in-plane blur, noise levels, NLST kernel pairs, second seed) | `bash scripts/34_p2_review_queue.sh` (runs `30_nlst_kernel_pairs.py`, `31_inplane_blur_e1.py`, `26_luna_texture.py --factor`, `27_texture_e1.py --factor`, model training with seed 1, `33_seed1_eval.py`); then `python scripts/32_review_stats.py` with `inplane`, `levels`, `levelskernel`, `nlst`, `nlstvendor`, `nlstthr`, `seed` | F5-F7 and the seed analysis (pre-specified in `analysis_plans/review_prespec.md`; later additions in `analysis_plans/review_prespec_NOTES.md`) |
| 16. Manuscript numbers and figures | `python paper/make_numbers.py`; `python paper/make_figures.py` | `numbers.tex`, figures |

Statistical helpers (paired bootstrap with common resamples, Holm) are in `scripts/p2_stats.py`; run it directly for its
self-checks. The FROC/CPM scorer (`lungcad/eval/froc.py`) reproduces the official LUNA16 evaluation script
(`lungcad/eval/official/`, see its README); `tests/test_froc_official.py` checks the agreement.

## Notes on provenance

- The S4 models were trained first and the S3 models afterwards, with the same code and recipe. The run manifest of
  every model records its configuration (`augment.dose_kernel: true` for S4, `false` for S3).
- The fold-0 S3 model was trained before a fix that made the patch sampler's random sequence reproducible between runs
  (it changes only the order of training patches); this repository contains the code after the fix.
  Fold 0 was retrained with seed 1 after the fix (S3 and S4); the pre-specified sensitivity analysis that replaces the
  fold-0 S3 by its retrain is in `scripts/32_review_stats.py seed`.
- Scripts that were run in the separate `astra` environment say so in their docstring.

## Licence

MIT, see [LICENSE](LICENSE). The LUNA16 evaluation script in `lungcad/eval/official/` keeps its original terms.
