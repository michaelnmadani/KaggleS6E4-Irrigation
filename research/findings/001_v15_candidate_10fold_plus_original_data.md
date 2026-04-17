# V15 research — CatBoost-solo + 10-fold CV + miadul external-dataset augmentation

*Revised 2026-04-17 after V14 result: the 3-model equal-weight blend (0.96812) under-performed CatBoost-alone (0.96987). V15 pivots to CatBoost-solo rather than continuing to blend.*

## Hypothesis
V14 established CatBoost-alone as the strongest single model on this feature set (CV 0.96987, LB ~0.965 estimated). V11's remaining gap (0.97256) almost certainly comes from:
1. **10-fold CV** (V11 used 10, V14 uses 5) — tighter OOF, ~+0.0005–0.0015
2. **miadul external dataset** appended with `sample_weight=0.35` — V11's top-cited recommendation, expected **+0.002–0.004**
3. Possibly hparam refinements that fit better with more training data

Keeping CatBoost-solo avoids the blend-weight problem that sank V14's gain.

## Expected CV
- Baseline (V14 CatBoost-solo): 0.96987
- + 10-fold CV: ~0.970 ± 0.001
- + external data aug: 0.972 ± 0.002
- Combined V15 target: **0.971–0.973**, right at V11's band

## Risks
- **Runtime**: 10-fold × CatBoost alone ≈ 45–60 min (vs V14's 74 min for 3 models × 5-fold). Actually faster. Still under 9-hr cap.
- **External-data leak**: miadul row set must not intersect the comp's test.csv. Dedupe by row-hash of numeric features before concat; competition data is synthesized *from* miadul so overlap is plausible.
- **Class-balance shift**: recompute `sample_weights` on the combined train + external y (different class priors).
- **miadul hygiene**: the dataset I need is `miadul/irrigation-water-requirement-prediction-dataset` on Kaggle Datasets — verify the column names match (may need renaming).

## Implementation sketch (delta from V14)

### `pipeline/src/data.py`
```python
def load(input_dir, target, id_col, extra_dataset_dir=None, extra_weight=0.35):
    train = pd.read_csv(input_dir/"train.csv")
    test = pd.read_csv(input_dir/"test.csv")
    is_original = np.zeros(len(train), dtype=bool)
    if extra_dataset_dir:
        extra = pd.read_csv(extra_dataset_dir/"irrigation_prediction.csv")
        # Align columns + rename if needed; dedupe against test
        extra = _align_and_dedupe(extra, train.columns, test)
        is_original_new = np.ones(len(extra), dtype=bool)
        train = pd.concat([train, extra], ignore_index=True)
        is_original = np.concatenate([is_original, is_original_new])
    # ... rest as before, also return is_original mask ...
    return X, y, X_test, test_ids, inverse_label_map, is_original
```

### `pipeline/src/train.py`
```python
sw = compute_balanced_sample_weights(y_tr)
if is_original is not None:
    sw = sw * np.where(is_original[tr_idx], extra_weight, 1.0)
```

### `pipeline/kernel_metadata.json` (via build_notebook.py)
Add `dataset_sources: ["miadul/irrigation-water-requirement-prediction-dataset"]` so Kaggle mounts it at `/kaggle/input/irrigation-water-requirement-prediction-dataset/`.

### `iterations/004_catboost_10fold_extdata/config.yaml`
```yaml
target: Irrigation_Need
id_col: id
task: multiclass
metric: balanced_accuracy
class_weights: balanced
extra_dataset:
  slug: miadul/irrigation-water-requirement-prediction-dataset
  file: irrigation_prediction.csv
  weight: 0.35
features: [fill_na_median, s6e4_threshold_booleans, s6e4_interactions,
           label_encode, target_encode_multiclass]
model: catboost
params:
  iterations: 2500
  early_stopping_rounds: 150
  learning_rate: 0.04
  depth: 7
  l2_leaf_reg: 5.0
cv: { n_splits: 10, seed: 42, stratified: true }
```

## Decision gates
- **Go** if V14's CatBoost standalone score (0.96987) holds up as a reproducible single-model baseline — confirmed by V14 metrics.json per_model_cv.
- **Pivot back to blend** (V15b) only if V15 does NOT hit ≥0.970. At that point, implement OOF-optimized weights via `scipy.optimize.minimize(balanced_acc(sum(w_i*oof_i)))` instead of score-proportional.

## Why not 3-model blend again
V14's `blend_weights: {lgbm:0.333, xgb:0.333, catboost:0.334}` shows score-proportional weights degenerated to uniform when raw scores were close (within 0.003). Uniform blend pulled the score down from CatBoost's 0.96987 to 0.96812 (-0.00175). V11's hand-picked `CatBoost=0.65` shows the blend IS viable with correct weights, but tuning those weights is its own optimization problem and adds risk. Simpler to go solo first, then revisit blending if V16 (logit-bias) doesn't close the gap.

## References
- V14 metrics.json (`iterations/003_three_model_blend/kernel_output/metrics.json` on main) — `per_model_cv` block confirms CatBoost's dominance
- V11 `run_v11_steps.py` lines 78–82 — exact `sample_weight = balanced * orig_weight` pattern
- V11 `results.json` improvement block — miadul augmentation top-ranked
