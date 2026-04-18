# Next iteration — V22 (iter 010): CatBoost-heavy 3-model blend

## What to change vs V15 (best so far, CV 0.96978)

- Go from **CatBoost-solo** back to **LGBM + XGB + CatBoost** — but with V11's proven **non-equal weights**:
  - `catboost: 0.65, lgbm: 0.25, xgb: 0.10`
- Keep everything else V15 proved: 10-fold stratified CV, miadul external-data augmentation (w=0.35), balanced class weights, V15 feature set (fill_na_median + thresholds + interactions + label_encode + per-fold multiclass TE).
- Additive pipeline change: `cfg['blend_weights']` dict now overrides the default score-proportional blend. No refactor — a single `if isinstance(manual_w, dict)` branch.

## Why this should beat 0.96978 (and approach 0.975)

- V14 result: equal-weight 3-model blend (0.333/0.333/0.334) scored **0.96812** — WORSE than CatBoost-alone (0.96987). The blend itself isn't broken; the equal weights were. V11's hand-tuned CB=0.65/LGBM=0.25/XGB=0.10 is the piece V14 omitted.
- V11 recipe (CB-heavy blend + 10-fold + ext data + TE + logit bias) = **0.97256** — the all-time high. V22 replicates everything except logit bias (V16 proved logit-bias delta = 0.0 on class-weight-balanced models).
- Single-model CV scores observed in V14: LGBM 0.96697, XGB 0.96714, CatBoost 0.96987. Weighted blend CB=0.65/LGBM=0.25/XGB=0.10 on those preds should OOF-score ~0.972 (CatBoost dominates, diversity from weaker models adds ~0.002 via error decorrelation).

## Expected delta

- Base: V15 = 0.96978.
- Expected V22 CV: **0.9720 – 0.9740** (mid of range ~ V11's 0.97256, upper end if CatBoost improves slightly from 10-fold stability).
- To clear 0.975 we'd likely need one more trick (seed ensembling or pseudo-labeling) stacked on top. That becomes V23 if V22 lands as expected.

## Risks

- Runtime: V14 (5-fold 3-model, no ext data) = 74 min. V22 adds 10-fold + miadul, projected ~3.5-4.5 hr on Kaggle P100. Comfortably under the 9-hr poll cap.
- Weight choice is from V11's hand-tuning on a slightly different feature set; transfer could shave a few ten-thousandths. Still expected to beat score-proportional / equal blending.
