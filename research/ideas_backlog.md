# Ideas backlog

Prioritized queue of experiments to try. Strategist pops from the top; reviewer appends to the bottom.

Format:
```
- [P1] short title — why it might help (link to finding if any)
```

## Current

- **V12 (iter 001) — DONE**: CV 0.96360 · LB 0.96152 · LGBM only on threshold booleans + label encoding
- **V13 (iter 002) — DONE**: CV 0.96697 · LB 0.96417 · +balanced weights, per-fold multiclass TE, interactions
- **V14 (iter 003) — IN FLIGHT**: 3-model blend (LGBM+XGB+CatBoost), score-proportional weights
- **V15 (researched)**: 10-fold CV + miadul original-dataset augmentation → [research/findings/001](findings/001_v15_candidate_10fold_plus_original_data.md)
- **V16 (researched)**: Logit-bias post-processing for balanced accuracy → [research/findings/002](findings/002_v16_candidate_logit_bias_tuning.md)
