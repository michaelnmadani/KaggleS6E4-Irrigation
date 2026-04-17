# V15 research — 10-fold CV + miadul original-dataset augmentation

## Hypothesis
V14 closes most of the blend gap (~+0.003 expected). The remaining ~0.003 to V11 comes from two techniques V11 used that V14 still doesn't:
1. **10-fold CV** (V11 used 10, we use 5) — more stable OOF estimates, ~5% lower fold_std, usually +0.0005–0.0015 on balanced-acc
2. **miadul external dataset** appended with `sample_weight=0.35` — V11's single largest engineering change, per their results.json `recommendations_applied` list ("Append original dataset … with sample_weight=0.35")

## Expected gain
- 10-fold alone: **+0.001 ± 0.001** (mainly variance reduction)
- Original-data augmentation: **+0.002 to +0.004** (V11 cited this as top-ranked)
- Combined V15 total: **+0.003 to +0.005**, putting us at **~0.970** CV, close to V11's 0.97256

## Risks
- **Runtime 2x**: 10-fold doubles fold count; if V14 is ~90 min on Kaggle, V15 is ~180 min. Still well under the 9-hr cap.
- **Leak risk from augmentation**: the external dataset MUST NOT be in the test-set row space. cdeotte's comp page notes the miadul set is the source for the comp's *synthetic* generation. Need to dedupe by id or by row-hash before appending.
- **Class-imbalance shift**: the miadul dataset may have a different class prior. After append, the `class_weights=balanced` computation should be recomputed on the combined y to avoid mis-weighting.

## Implementation sketch (delta from V14)
- `pipeline/src/data.py` — add optional param `extra_dataset_path: str` and `extra_weight: float` to `load()`. Reads `/kaggle/input/<dataset-slug>/<file>`, aligns columns with train.csv, concatenates, returns an additional `is_original` mask.
- `pipeline/src/train.py` — if loader returned the mask, multiply `sw` by `extra_weight` where mask is True.
- Kernel metadata — add `dataset_sources: ["miadul/irrigation-water-requirement-prediction-dataset"]` so Kaggle mounts it at `/kaggle/input/irrigation-water-requirement-prediction-dataset/`.
- `iterations/004_ext_data_10fold/config.yaml` — flip `cv.n_splits: 10`, add `extra_dataset: {...}` block.

## Decision gates
- Go ahead if V14 lands between 0.969 and 0.971 CV (meaningful improvement that the external data would plausibly stack on).
- Skip and go straight to V16 (logit-bias) if V14 regresses below V13 — signals blend weights need manual tuning instead.

## References
- V11 `run_v11_steps.py` (`claude/data-science-pipeline-agents-gqv4i` branch, lines 78–82) shows the exact `sample_weight = class_weight * orig_weight` pattern
- V11 `results.json` improvement block lists miadul augmentation as the top recommendation_applied
