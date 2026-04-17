# V18 research — Unsupervised binning + target encoding on continuous features

*Seeded from Kaggle discussion 688353 "Unsupervised binning method"; Kaggle's JS-rendered UI prevented full scrape but title + first 200 chars are unambiguous.*

## Hypothesis
V13+ target-encodes the 6 string categoricals (Crop_Type, Crop_Growth_Stage, Season, Irrigation_Type, Water_Source, Mulching_Used, Region, Soil_Type), producing 3 per-class probabilities each = 18 TE features. But the dataset has **10 continuous numerics** that TE is not applied to. Binning them into quantile buckets and TE'ing the buckets can capture non-linear target relationships that tree splits on raw numerics miss.

## Why it might help
- Tree models already bin continuous features internally, but they optimize splits on the *full* training set at each node. TE'd bins give the model a per-bucket *prior* that shortcuts shallow splits and helps the minority class (High) where each leaf has few samples.
- Acts as a Bayesian smoothing prior: 5-10 quantile bins × 3 class probs = extra stable per-range features.
- Particularly useful for `Soil_Moisture`, `Temperature_C`, `Rainfall_mm`, `Wind_Speed_kmh` — the four columns whose thresholds V17 confirmed are the core generator. Binning finer than the single 25/30/300/10 thresholds could recover the residual ~3% that pure LogReg missed.

## Expected gain
- **+0.001 to +0.003** balanced-acc on top of V15/V16.
- Biggest impact if paired with logit-bias (V16 infrastructure) for the minority class.

## Implementation sketch (delta from V15/V16)

### `pipeline/src/features.py`
```python
def quantile_bin_numeric(X_tr, X_te, n_bins: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each numeric column, add a `{col}_q{n_bins}` categorical column
    holding the quantile bucket. Edges are fit on X_tr only to avoid test leak."""
    X_tr, X_te = X_tr.copy(), X_te.copy()
    for col in X_tr.select_dtypes(include=[np.number]).columns:
        edges = pd.qcut(X_tr[col], q=n_bins, retbins=True, duplicates="drop")[1]
        X_tr[f"{col}_qbin"] = pd.cut(X_tr[col], bins=edges, labels=False,
                                     include_lowest=True).astype("Int64")
        X_te[f"{col}_qbin"] = pd.cut(X_te[col], bins=edges, labels=False,
                                     include_lowest=True).fillna(0).astype("Int64")
    return X_tr, X_te
```

Register in `BLOCKS` as `quantile_bin_numeric`. Feature pipeline becomes:
```yaml
features:
  - fill_na_median
  - s6e4_threshold_booleans
  - s6e4_interactions
  - quantile_bin_numeric        # NEW: 10 continuous -> 10 integer bin columns
  - label_encode
  - target_encode_multiclass    # TEs both original categoricals AND new qbin columns
```

Outcome: 18 existing TE features + 10 binned continuous × 3 classes = 48 more TE features. Feature count goes from 32 (V15) to ~80.

### Alternative: KMeans or equal-width binning
Quantile is safest; equal-width buckets can concentrate data in rare ranges. KMeans on univariate data is overkill. Stick with quantile unless quantile doesn't move the needle.

## Risks
- **Feature bloat**: 80 features may slow CatBoost by 2-3x. Offset by dropping raw numerics after binning (they're redundant with threshold booleans + binned versions).
- **Overfitting TE on sparse bins**: with n_bins=10 and 630k rows, ~63k per bucket — plenty of samples per bucket. Safe.
- **CV-LB gap widening**: already widening from V12→V15 (0.00208 → 0.00334). More features could widen further. Use V16's nested-CV logit-bias as the decision gate on whether V18 generalizes.

## Decision gates
- Run V18 only after V16 lands AND V16's nested `delta_nested > 0`. If V16 already tops 0.973 CV, V18 may push us into overfit territory — stop.
- If V18 CV is >+0.002 over V17 but public LB is ≤0.96644 (V15), roll back — the new features don't generalize.

## References
- Kaggle discussion 688353 "Unsupervised binning method" (JS-rendered, full content unavailable; title + first 200 chars confirmed the idea)
- Kaggle discussion 688866 "Lightgbm baseline and advanced" — different TargetEncoders compared; could swap our custom TE for `category_encoders.CatBoostEncoder` as a companion experiment to V18
