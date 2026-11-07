# Notes on the noise-texture pre-specification

The pre-specification (`texture_test_prespec.md`) is left unchanged so that its SHA-256 hash stays valid. Notes added afterwards:

1. **Time stamp.** The file says "Written 2 Nov 2026, ~17:15". Its hash was recorded at 16:58:38 on 2 Nov 2026
   (the hash file written next to the analysis outputs), so the file was written before 16:58. "~17:15" was an estimate written
   into the text and is wrong. The order of events (plan → hash → generation 17:xx → scoring) is unaffected.
2. **Level tolerance.** 16 of the 356 scans × 3 textures were outside the ±3 % tolerance after two passes
   (real NPS 5, white 11, Gaussian grain 0; largest deviation 5.0 %). As the plan requires, they are reported, and they
   were kept in the analysis.
3. **Result.** Verdict by the pre-specified rule: "partly supported" (F4 significant for both models; equivalence with
   projection noise failed for both). See `texture_results.json` written by `scripts/28_texture_stats.py`.
