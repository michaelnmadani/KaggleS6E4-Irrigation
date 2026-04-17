# Ideas backlog

Prioritized queue of experiments to try. Strategist pops from the top; reviewer appends to the bottom.

Format:
```
- [P1] short title — why it might help (link to finding if any)
```

## Current

- [P1] Validate pipeline plumbing end-to-end with `001_baseline` (threshold booleans + LGBM)
- [P1] If LGBM baseline CV < 1.0, try multinomial LogisticRegression on the same features — cdeotte's notebook shows this hits CV balanced-acc 1.0
- [P2] If baseline already at 1.0 CV, focus on avoiding submission-format mistakes (id alignment, string-label output) and submitting consistently
- [P2] Try XGBoost / CatBoost on same features for robustness against public-LB noise
- [P3] Seed ensemble × 5 for tiebreaking if public LB is close
- [P3] Explore whether the comp's actual test distribution differs from cdeotte's "original" — if so, raw numeric features may matter more than thresholds
