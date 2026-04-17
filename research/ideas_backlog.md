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
- **V17 (DONE)**: CV 0.96203 — LogReg on cdeotte's 10 features. Confirms comp test has noise beyond the original-data generator; tree models beat linear here.
- **V18 (researched)**: Quantile-bin continuous numerics + target-encode the bins → [research/findings/003](findings/003_v18_candidate_binned_te.md)
- **V19 idea**: swap custom smoothed-TE for `category_encoders.CatBoostEncoder` (per discussion 688866)
