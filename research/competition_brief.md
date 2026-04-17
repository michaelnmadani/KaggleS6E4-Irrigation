# Competition brief — Predicting Irrigation Need (S6E4)

*Every Claude role reads this. Keep current.*

- **Slug:** `playground-series-s6e4`
- **URL:** <https://www.kaggle.com/competitions/playground-series-s6e4>
- **Title:** Predicting Irrigation Need
- **Task:** `multiclass` (3 classes)
- **Target column:** `Irrigation_Need` — string labels: `Low`, `Medium`, `High`
- **Id column:** `id`
- **Evaluation metric:** `accuracy` — **must match `metric:` in every `config.yaml`**
- **Class encoding (pipeline-internal):** `Low=0, Medium=1, High=2` (auto-applied by `data.py`, reversed on submission)

## Data schema (confirmed via cdeotte's original dataset)

- **Numeric:** `Soil_Moisture`, `Temperature_C`, `Rainfall_mm`, `Wind_Speed_kmh`
- **Categorical:** `Crop_Growth_Stage` (Flowering/Harvest/Sowing/Vegetative), `Mulching_Used` (Yes/No)

## Known strong approach

Four boolean threshold features plus logistic regression hit **CV balanced-accuracy 1.0** on cdeotte's original dataset:

| Feature       | Rule                |
|---------------|---------------------|
| `soil_lt_25`  | `Soil_Moisture < 25` |
| `temp_gt_30`  | `Temperature_C > 30` |
| `rain_lt_300` | `Rainfall_mm < 300`  |
| `wind_gt_10`  | `Wind_Speed_kmh > 10`|

Implemented as the `s6e4_threshold_booleans` feature block in `pipeline/src/features.py`.

## Notable rules / constraints
- External data allowed (cdeotte's original is itself an external-data example used widely).
- Public comp, standard Kaggle Notebook rules (9-hr runtime, no internet required).

## Deadline
- `<TODO: check comp page; set before near-deadline risk rules kick in>`
