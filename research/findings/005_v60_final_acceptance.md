# V60 final acceptance (2026-04-29)

Run ended at V60 (LB 0.97646), accepted by user as final after V61 miadul
experiment showed regression and Kaggle GPU quota blocked V62–V65.

## Final submission
- iter `048_xgb_deeper` / website V60
- CV 0.97608 (5-fold KFold, balanced-accuracy)
- Public LB 0.97646
- recipe: XGB depth=5, lr=0.07, 1000 rounds with early_stopping_rounds=80
  on V34's feature set (fill_na_median + s6e4_digit_extraction_wide +
  s6e4_freq_filter_cats + ordered_te) + Optuna class-weight search.

## V61–V65 closure
- V61 (V60 + miadul w=0.35): CV 0.97612, LB 0.97601 (-0.00045 vs V60). miadul
  hurt — same direction as V35 (V34 + miadul stratified) which lost 0.001.
- V62 attempts (w=1.0, w=1.0 renamed, +stratified): all blocked at Kaggle
  kernel registration after pushing — silent quota throttling. No CV/LB.
- V63–V65: not started.

## Iterations left as SKIPPED on main
- iterations/050_v60_miadul_w10/ (config + SKIPPED.md)
- iterations/051_v60_miadul_full/ (config + SKIPPED.md)
- iterations/052_v60_stratified/ (config but no SKIPPED.md — was the last
  attempt before user accepted V60; queueable when quota recycles)

## Key takeaway
The V34→V60 LB band (0.97617–0.97646) appears to be the practical ceiling for
single-XGB + ordered TE + Optuna postprocess on this dataset. Reaching 0.98
would require a fundamentally different approach (TabPFN-v2, generator
reverse-engineering, or external data with a target-mapping not available
from miadul's same-schema rows).
