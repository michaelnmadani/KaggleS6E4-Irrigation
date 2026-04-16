# Ideas backlog

Prioritized queue of experiments to try. Strategist pops from the top; reviewer appends to the bottom.

Format:
```
- [P1] short title — why it might help (link to finding if any)
```

## Current

- [P1] Validate pipeline plumbing end-to-end with `001_baseline` (LGBM, default hparams, 5-fold)
- [P2] Try XGBoost with matched hparams — quick sanity on model variance
- [P2] Try CatBoost — usually strongest single model on tabular Playground
- [P3] Seed ensemble of top model × 5 seeds — cheap variance reduction
