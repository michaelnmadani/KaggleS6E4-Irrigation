"""Feature-engineering blocks addressable by name from config.yaml.

Each block takes (X_train, X_test) and returns (X_train, X_test) with new/modified
columns. Some blocks also accept y_tr for supervised encoders (target encoding).
Blocks must be pure and deterministic. Register blocks in BLOCKS at bottom.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _categorical_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def label_encode(X_tr: pd.DataFrame, X_te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_tr, X_te = X_tr.copy(), X_te.copy()
    for col in _categorical_cols(X_tr):
        combined = pd.concat([X_tr[col], X_te[col]], axis=0).astype("category")
        codes = combined.cat.codes
        X_tr[col] = codes.iloc[: len(X_tr)].values
        X_te[col] = codes.iloc[len(X_tr) :].values
    return X_tr, X_te


def fill_na_median(X_tr: pd.DataFrame, X_te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_tr, X_te = X_tr.copy(), X_te.copy()
    for col in _numeric_cols(X_tr):
        med = X_tr[col].median()
        X_tr[col] = X_tr[col].fillna(med)
        X_te[col] = X_te[col].fillna(med)
    return X_tr, X_te


def count_encode_categoricals(X_tr: pd.DataFrame, X_te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_tr, X_te = X_tr.copy(), X_te.copy()
    for col in _categorical_cols(X_tr):
        counts = pd.concat([X_tr[col], X_te[col]]).value_counts()
        X_tr[f"{col}_count"] = X_tr[col].map(counts).fillna(0).astype(int)
        X_te[f"{col}_count"] = X_te[col].map(counts).fillna(0).astype(int)
    return X_tr, X_te


def s6e4_threshold_booleans(X_tr: pd.DataFrame, X_te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """S6E4-specific: four boolean thresholds that separate the 3 irrigation classes
    near-perfectly (cdeotte's original-dataset notebook hits CV balanced-acc 1.0 with
    these + logreg). Drops source columns after thresholding to keep the feature
    space minimal."""
    rules = [
        ("Soil_Moisture",    "<", 25, "soil_lt_25"),
        ("Temperature_C",    ">", 30, "temp_gt_30"),
        ("Rainfall_mm",      "<", 300, "rain_lt_300"),
        ("Wind_Speed_kmh",   ">", 10, "wind_gt_10"),
    ]
    X_tr, X_te = X_tr.copy(), X_te.copy()
    for col, op, val, name in rules:
        if col not in X_tr.columns:
            raise KeyError(f"s6e4_threshold_booleans: missing column {col!r}")
        cmp = (X_tr[col] < val) if op == "<" else (X_tr[col] > val)
        X_tr[name] = cmp.astype(int)
        cmp_te = (X_te[col] < val) if op == "<" else (X_te[col] > val)
        X_te[name] = cmp_te.astype(int)
    return X_tr, X_te


def s6e4_interactions(X_tr: pd.DataFrame, X_te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """S6E4-specific ratio/product/diff features between weather+soil columns.
    Mirrors the interaction set V10/V11 found useful."""
    X_tr, X_te = X_tr.copy(), X_te.copy()
    pairs = [
        ("Soil_Moisture", "Temperature_C",   "moisture_temp_ratio",   "ratio"),
        ("Soil_Moisture", "Temperature_C",   "moisture_temp_product", "product"),
        ("Rainfall_mm",   "Humidity",        "rain_humidity_product", "product"),
        ("Rainfall_mm",   "Humidity",        "rain_humidity_diff",    "diff"),
        ("Soil_pH",       "Organic_Carbon",  "ph_carbon_product",     "product"),
        ("Sunlight_Hours","Wind_Speed_kmh",  "sun_wind_ratio",        "ratio"),
        ("Rainfall_mm",   "Previous_Irrigation_mm", "rain_prev_diff", "diff"),
        ("Rainfall_mm",   "Previous_Irrigation_mm", "rain_prev_ratio","ratio"),
        ("Temperature_C", "Humidity",        "temp_humidity_product", "product"),
    ]
    for a, b, name, op in pairs:
        if a not in X_tr.columns or b not in X_tr.columns:
            continue
        for X in (X_tr, X_te):
            if op == "ratio":
                X[name] = X[a] / (X[b] + 1)
            elif op == "product":
                X[name] = X[a] * X[b]
            elif op == "diff":
                X[name] = X[a] - X[b]
    return X_tr, X_te


def target_encode_multiclass(X_tr: pd.DataFrame, X_te: pd.DataFrame, y_tr=None, smoothing: int = 10, max_nunique: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each categorical col and each target class k, add a smoothed P(y=k|category).

    Column detection: object/category dtype, OR integer dtype with <= max_nunique
    unique values. The integer-dtype fallback catches columns that a prior
    label_encode block converted to int codes (pre-V24 this was a silent no-op)."""
    if y_tr is None:
        raise ValueError("target_encode_multiclass requires y_tr")
    X_tr, X_te = X_tr.copy(), X_te.copy()
    y = pd.Series(y_tr).reset_index(drop=True)
    classes = sorted(y.unique().tolist())

    cat_like = set(_categorical_cols(X_tr))
    # Include int cols that look like label-encoded categoricals (low cardinality).
    for col in X_tr.columns:
        if col in cat_like:
            continue
        dt = X_tr[col].dtype
        if pd.api.types.is_integer_dtype(dt) or str(dt).startswith(("int", "uint")):
            try:
                if X_tr[col].nunique(dropna=True) <= max_nunique:
                    cat_like.add(col)
            except Exception:
                pass

    for col in cat_like:
        # If the source is Categorical, .map can preserve Categorical dtype, which
        # breaks .fillna(float). Convert to object first for safe mapping.
        src_tr = pd.Series(X_tr[col].values).astype(object)
        src_te = pd.Series(X_te[col].values).astype(object)
        for k in classes:
            tgt = (y == k).astype(int)
            global_mean = float(tgt.mean())
            temp = pd.DataFrame({"cat": src_tr, "t": tgt.values})
            agg = temp.groupby("cat")["t"].agg(["mean", "count"])
            smooth = (agg["count"] * agg["mean"] + smoothing * global_mean) / (agg["count"] + smoothing)
            smooth_dict = smooth.to_dict()
            X_tr[f"{col}_te_c{int(k)}"] = np.asarray(src_tr.map(smooth_dict).astype(float).fillna(global_mean))
            X_te[f"{col}_te_c{int(k)}"] = np.asarray(src_te.map(smooth_dict).astype(float).fillna(global_mean))
    return X_tr, X_te


def s6e4_cdeotte_minimal(X_tr: pd.DataFrame, X_te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exactly cdeotte's 10-feature set that perfectly separates the original
    dataset into 3 classes: 4 threshold booleans + Crop_Growth_Stage (4 one-hot
    dummies) + Mulching_Used (2 one-hot dummies) = 10 features, nothing else.

    Use with model: logreg — the LogReg wrapper one-hots categoricals itself,
    so we keep those cols as strings. The Kaggle discussion #687460 notes this
    is the exact data generator.
    """
    rules = [
        ("Soil_Moisture",  "<", 25,  "soil_lt_25"),
        ("Temperature_C",  ">", 30,  "temp_gt_30"),
        ("Rainfall_mm",    "<", 300, "rain_lt_300"),
        ("Wind_Speed_kmh", ">", 10,  "wind_gt_10"),
    ]
    keep_cats = ["Crop_Growth_Stage", "Mulching_Used"]
    out = {}
    for name, X in (("tr", X_tr), ("te", X_te)):
        d = {}
        for col, op, val, new in rules:
            if col not in X.columns:
                raise KeyError(f"s6e4_cdeotte_minimal: missing {col!r}")
            d[new] = ((X[col] < val) if op == "<" else (X[col] > val)).astype(int).values
        for c in keep_cats:
            if c not in X.columns:
                raise KeyError(f"s6e4_cdeotte_minimal: missing {c!r}")
            d[c] = X[c].values
        out[name] = pd.DataFrame(d, index=X.index)
    return out["tr"], out["te"]


def s6e4_realmlp_fe(X_tr: pd.DataFrame, X_te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Full feature-engineering block matching the RealMLP reference notebook
    (mahoganybuttstrings/pg-s6e4-realmlp...). Produces:
      1. Every numeric column duplicated as `{col}_2` (two views per numeric).
      2. All categoricals and numerics factorized to integer codes on the
         train+test union so codes are consistent.
      3. Pairwise-concat interactions between 6 key columns (Soil_Moisture,
         Crop_Growth_Stage, Temperature_C, Mulching_Used, Wind_Speed_kmh,
         Rainfall_mm) = C(6,2) = 15 new int-factorized columns. Sparse combos
         (nunique > half the rows) are dropped.
    The set of 6 columns is per the discussion-687460 "exact formula" finding:
    these are the columns the generator depends on. Pairwise TE on them gives
    the non-linear lift that single-col TE misses.
    """
    X_tr, X_te = X_tr.copy(), X_te.copy()
    nums = ["Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
            "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
            "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm"]
    cats = ["Soil_Type", "Crop_Type", "Crop_Growth_Stage", "Season",
            "Irrigation_Type", "Water_Source", "Mulching_Used", "Region"]
    # 1) duplicate numerics
    for c in nums:
        if c in X_tr.columns:
            X_tr[f"{c}_2"] = X_tr[c]
            X_te[f"{c}_2"] = X_te[c]
    # 2) factorize ALL cats + nums (+ duplicates) on train+test union
    for c in cats + nums:
        if c in X_tr.columns:
            combined = pd.concat([X_tr[c], X_te[c]], axis=0, ignore_index=True)
            codes, _ = pd.factorize(combined)
            X_tr[c] = codes[:len(X_tr)]
            X_te[c] = codes[len(X_tr):]
    # _2 duplicates intentionally NOT factorized — they stay as raw floats so
    # RealMLP's PBLD embedding can learn periodic patterns on the continuous
    # representation, while the factorized-int version goes through the
    # categorical embedding path. Matches the reference notebook.

    # 3) pairwise interactions among the 6 key columns
    from itertools import combinations
    key_cols = ["Soil_Moisture", "Crop_Growth_Stage", "Temperature_C",
                "Mulching_Used", "Wind_Speed_kmh", "Rainfall_mm"]
    n_rows = len(X_tr) + len(X_te)
    for a, b in combinations(key_cols, 2):
        if a not in X_tr.columns or b not in X_tr.columns:
            continue
        name = f"{a}-{b}"
        s_tr = X_tr[a].astype(str) + "_" + X_tr[b].astype(str)
        s_te = X_te[a].astype(str) + "_" + X_te[b].astype(str)
        combined = pd.concat([s_tr, s_te], axis=0, ignore_index=True)
        codes, _ = pd.factorize(combined)
        if pd.Series(codes).nunique() > n_rows // 2:
            continue  # too sparse
        # Cast pairwise interactions as "category" dtype — only these get TE'd
        # by target_encode_multiclass downstream. Keeping originals as int64
        # matches the reference notebook's behavior (TE applied only to pairwise).
        X_tr[name] = pd.Categorical(codes[:len(X_tr)])
        X_te[name] = pd.Categorical(codes[len(X_tr):])
    return X_tr, X_te


def s6e4_pairwise_only(X_tr: pd.DataFrame, X_te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add ONLY the C(6,2)=15 pairwise-concat interactions on the 6 generator
    columns, without touching any other features. Unlike s6e4_realmlp_fe which
    factorizes all columns (destroying continuous signal for tree models), this
    block preserves originals and just APPENDS 6-15 new factorized pairwise
    categoricals. Pairs with nunique > n_rows/2 are dropped as too sparse.
    """
    X_tr, X_te = X_tr.copy(), X_te.copy()
    from itertools import combinations
    key_cols = ["Soil_Moisture", "Crop_Growth_Stage", "Temperature_C",
                "Mulching_Used", "Wind_Speed_kmh", "Rainfall_mm"]
    n_rows = len(X_tr) + len(X_te)
    for a, b in combinations(key_cols, 2):
        if a not in X_tr.columns or b not in X_tr.columns:
            continue
        name = f"{a}-{b}"
        s_tr = X_tr[a].astype(str) + "_" + X_tr[b].astype(str)
        s_te = X_te[a].astype(str) + "_" + X_te[b].astype(str)
        combined = pd.concat([s_tr, s_te], axis=0, ignore_index=True)
        codes, _ = pd.factorize(combined)
        if pd.Series(codes).nunique() > n_rows // 2:
            continue
        # Cast as Categorical so target_encode_multiclass picks it up.
        X_tr[name] = pd.Categorical(codes[:len(X_tr)])
        X_te[name] = pd.Categorical(codes[len(X_tr):])
    return X_tr, X_te


# --- V23 FE ports from V11 ----------------------------------------------------

_V11_NUMERIC_SOURCES = [
    "Soil_pH", "Soil_Moisture", "Organic_Carbon", "Electrical_Conductivity",
    "Temperature_C", "Humidity", "Rainfall_mm", "Sunlight_Hours",
    "Wind_Speed_kmh", "Field_Area_hectare", "Previous_Irrigation_mm",
]


def s6e4_decimal_digits(X_tr: pd.DataFrame, X_te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """V11 synthetic-data fingerprint: the last significant decimal digits of
    each numeric col leak signal about the generator. Extract first + second
    decimal digits as integer features."""
    X_tr, X_te = X_tr.copy(), X_te.copy()
    for col in _V11_NUMERIC_SOURCES:
        if col not in X_tr.columns:
            continue
        for scale, lbl in [(10, "d1"), (100, "d2")]:
            for X in (X_tr, X_te):
                v = X[col].astype(float).fillna(0.0).values
                X[f"{col}_{lbl}"] = (np.round(v * scale).astype(np.int64) % 10).astype(np.int16)
    return X_tr, X_te


def s6e4_field_normalized(X_tr: pd.DataFrame, X_te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """V11 per-hectare ratios. Area-normalizes rainfall/prev-irrigation so field
    size doesn't confound the class signal."""
    X_tr, X_te = X_tr.copy(), X_te.copy()
    if "Field_Area_hectare" not in X_tr.columns:
        return X_tr, X_te
    for num in ("Rainfall_mm", "Previous_Irrigation_mm", "Soil_Moisture"):
        if num not in X_tr.columns:
            continue
        for X in (X_tr, X_te):
            denom = X["Field_Area_hectare"].astype(float).replace(0, np.nan) + 1e-6
            X[f"{num}_per_ha"] = (X[num].astype(float) / denom).fillna(X[num].astype(float).median())
    return X_tr, X_te


def s6e4_groupby_aggs(X_tr: pd.DataFrame, X_te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """V11 groupby aggregations: for each (categorical, continuous) pair below,
    compute mean/std of the continuous within each category level on the union
    of train+test, then map back. Unsupervised so no leakage."""
    X_tr, X_te = X_tr.copy(), X_te.copy()
    cats = [c for c in ("Crop_Type", "Soil_Type", "Region", "Season",
                        "Crop_Growth_Stage", "Mulching_Used") if c in X_tr.columns]
    nums = [n for n in ("Soil_Moisture", "Temperature_C", "Rainfall_mm",
                        "Wind_Speed_kmh", "Previous_Irrigation_mm") if n in X_tr.columns]
    if not cats or not nums:
        return X_tr, X_te
    full = pd.concat([X_tr[cats + nums], X_te[cats + nums]], axis=0, ignore_index=True)
    for c in cats:
        for n in nums:
            agg = full.groupby(c)[n].agg(["mean", "std"])
            mean_map = agg["mean"].to_dict()
            std_map = agg["std"].fillna(0).to_dict()
            for X in (X_tr, X_te):
                X[f"{c}_{n}_gmean"] = X[c].map(mean_map).astype(float)
                X[f"{c}_{n}_gstd"] = X[c].map(std_map).astype(float).fillna(0.0)
    return X_tr, X_te


def s6e4_quantile_bins(X_tr: pd.DataFrame, X_te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """V18 candidate: quantile-bin each continuous numeric into deciles, cast as
    Categorical so a downstream target_encode_multiclass block picks them up and
    learns per-bin P(y=k|bin). Bin edges computed on train only."""
    X_tr, X_te = X_tr.copy(), X_te.copy()
    for col in ("Soil_Moisture", "Temperature_C", "Humidity", "Rainfall_mm",
                "Wind_Speed_kmh", "Sunlight_Hours", "Previous_Irrigation_mm"):
        if col not in X_tr.columns:
            continue
        src = X_tr[col].astype(float)
        edges = np.unique(np.quantile(src.dropna().values, np.linspace(0, 1, 11)))
        if len(edges) < 3:
            continue
        b_tr = np.clip(np.digitize(src.fillna(src.median()).values, edges[1:-1]), 0, 9).astype(np.int16)
        b_te = np.clip(
            np.digitize(X_te[col].astype(float).fillna(src.median()).values, edges[1:-1]), 0, 9
        ).astype(np.int16)
        X_tr[f"{col}_q10"] = pd.Categorical(b_tr, categories=list(range(10)))
        X_te[f"{col}_q10"] = pd.Categorical(b_te, categories=list(range(10)))
    return X_tr, X_te


def s6e4_digit_extraction_wide(X_tr: pd.DataFrame, X_te: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """yunsuxiaozi V34 port: extract digit at positions k in [-4, 4] for each
    numeric column. Each numeric -> 9 digit features.

    For k<0: treats as decimal (10**k acts as shift).
    For k>=0: int-divide by 10**k, modulo 10.
    Plus rounding: if max<10 -> round(3); <100 -> round(2); else round(1).
    Ref: https://www.kaggle.com/code/yunsuxiaozi/pss6e4-xgb-cv-0-979805
    """
    X_tr, X_te = X_tr.copy(), X_te.copy()
    num_cols = [c for c in _V11_NUMERIC_SOURCES if c in X_tr.columns]
    max_vals = {c: float(X_tr[c].max()) for c in num_cols}
    for c in num_cols:
        v_tr = X_tr[c].astype(float).fillna(0.0).values
        v_te = X_te[c].astype(float).fillna(0.0).values
        for k in range(-4, 5):
            p = 10.0 ** k
            X_tr[f"{c}_digit{k}"] = (np.floor(v_tr / p).astype(np.int64) % 10).astype(np.int16)
            X_te[f"{c}_digit{k}"] = (np.floor(v_te / p).astype(np.int64) % 10).astype(np.int16)
        m = max_vals[c]
        if m < 10:
            X_tr[c] = X_tr[c].round(3); X_te[c] = X_te[c].round(3)
        elif m < 100:
            X_tr[c] = X_tr[c].round(2); X_te[c] = X_te[c].round(2)
        else:
            X_tr[c] = X_tr[c].round(1); X_te[c] = X_te[c].round(1)
    return X_tr, X_te


def s6e4_freq_filter_cats(X_tr: pd.DataFrame, X_te: pd.DataFrame, min_count: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """yunsuxiaozi V34 port: for each object/category/digit col, map values with
    train-count >= min_count to indices 0..K-1; rare values -> K (default bucket)."""
    X_tr, X_te = X_tr.copy(), X_te.copy()
    target_cols = [c for c in X_tr.columns
                   if str(X_tr[c].dtype) in ("object", "category") or "digit" in c]
    for c in target_cols:
        freq = X_tr[c].value_counts()
        keep = freq[freq >= min_count].index.tolist()
        mapping = {v: i for i, v in enumerate(keep)}
        default = len(mapping)
        X_tr[c] = X_tr[c].map(lambda x: mapping.get(x, default)).astype(np.int32)
        X_te[c] = X_te[c].map(lambda x: mapping.get(x, default)).astype(np.int32)
    return X_tr, X_te


def ordered_te(X_tr: pd.DataFrame, X_te: pd.DataFrame, y_tr=None, smoothing: float = 1.0, n_shuffles: int = 4, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """yunsuxiaozi V34 port: ordered target-encoding with shuffle augmentation.

    For each category col c and each target class k, computes cumulative
    (sum_so_far - y_current) / (count_so_far - 1), smoothed by class prior.
    Done n_shuffles times with different row orders and concatenated — this
    effectively augments training data by n_shuffles×.

    IMPORTANT: this block returns a LARGER X_tr (n_shuffles × original rows) plus
    a correspondingly stacked y_tr hidden in a special '_y_shuffled' column.
    The train.py caller recognizes the column name and extracts y. Test/val are
    encoded once using train's full per-category priors."""
    if y_tr is None:
        raise ValueError("ordered_te requires y_tr")
    X_tr, X_te = X_tr.copy(), X_te.copy()
    y = pd.Series(y_tr).reset_index(drop=True).astype(int)
    classes = sorted(y.unique().tolist())
    # Identify integer categoricals (already-encoded or low-nunique ints).
    cat_cols = [c for c in X_tr.columns
                if str(X_tr[c].dtype).startswith("int") or str(X_tr[c].dtype) in ("object", "category")]
    # Precompute per-category totals on full train for test/val transform.
    global_prior = {k: float((y == k).mean()) for k in classes}
    stats = {c: {k: None for k in classes} for c in cat_cols}
    for c in cat_cols:
        for k in classes:
            yk = (y == k).astype(int)
            agg = pd.DataFrame({"cat": X_tr[c].values, "t": yk.values}).groupby("cat")["t"].agg(["sum", "count"])
            stats[c][k] = agg

    # Build n_shuffles training copies with ordered cumulative TE features.
    rng = np.random.default_rng(seed)
    aug_parts = []
    aug_y_parts = []
    for s in range(n_shuffles):
        perm = rng.permutation(len(X_tr))
        X_s = X_tr.iloc[perm].reset_index(drop=True).copy()
        y_s = y.iloc[perm].reset_index(drop=True)
        for c in cat_cols:
            for k in classes:
                yk = (y_s == k).astype(int)
                df = pd.DataFrame({"cat": X_s[c].values, "y": yk.values})
                df["cum_cnt"] = df.groupby("cat").cumcount()   # rows before current, same cat
                df["cum_sum"] = df.groupby("cat")["y"].cumsum() - df["y"]
                prior = global_prior[k]
                te = (df["cum_sum"] + smoothing * prior) / (df["cum_cnt"] + smoothing)
                te[df["cum_cnt"] == 0] = prior  # first-seen -> prior
                X_s[f"{c}_ote_c{k}"] = te.values.astype(np.float32)
        aug_parts.append(X_s)
        aug_y_parts.append(y_s)

    X_tr_aug = pd.concat(aug_parts, axis=0, ignore_index=True)
    y_tr_aug = pd.concat(aug_y_parts, axis=0, ignore_index=True)
    # Smuggle y via special col so train.py can pull it back.
    X_tr_aug["_ote_y_shuffled"] = y_tr_aug.values.astype(np.int32)

    # Transform test/val: use full-train stats, smoothed prior.
    X_te_out = X_te.copy()
    for c in cat_cols:
        for k in classes:
            agg = stats[c][k]
            prior = global_prior[k]
            te_vals = ((agg["sum"].values + smoothing * prior) / (agg["count"].values + smoothing))
            mp = dict(zip(agg.index.tolist(), te_vals.tolist()))
            X_te_out[f"{c}_ote_c{k}"] = X_te[c].map(mp).fillna(prior).astype(np.float32).values
    return X_tr_aug, X_te_out


def s6e4_cat_pair_combined_keys(X_tr, X_te):
    """Create combined-key features by joining pairs of the 6 original
    categoricals into single string keys. Run BEFORE s6e4_freq_filter_cats
    so the freq filter maps them to compact int codes; then ordered_te picks
    them up as per-fold per-class TE features.

    V39 FE lever: C(6,2)=15 new cat columns with higher cardinality than any
    single cat. Each has up to 6*6=36 levels per pair. Ordered TE on these
    captures multi-way class-conditional signal the single-cat TE misses.
    """
    X_tr, X_te = X_tr.copy(), X_te.copy()
    CATS = ["Soil_Type", "Crop_Type", "Region", "Season",
            "Crop_Growth_Stage", "Mulching_Used"]
    cats = [c for c in CATS if c in X_tr.columns]
    for i, a in enumerate(cats):
        for b in cats[i+1:]:
            name = f"{a}_X_{b}"
            X_tr[name] = X_tr[a].astype(str) + "|" + X_tr[b].astype(str)
            X_te[name] = X_te[a].astype(str) + "|" + X_te[b].astype(str)
    return X_tr, X_te


BLOCKS = {
    "label_encode": label_encode,
    "fill_na_median": fill_na_median,
    "count_encode_categoricals": count_encode_categoricals,
    "s6e4_threshold_booleans": s6e4_threshold_booleans,
    "s6e4_interactions": s6e4_interactions,
    "s6e4_cdeotte_minimal": s6e4_cdeotte_minimal,
    "s6e4_realmlp_fe": s6e4_realmlp_fe,
    "s6e4_pairwise_only": s6e4_pairwise_only,
    "s6e4_decimal_digits": s6e4_decimal_digits,
    "s6e4_field_normalized": s6e4_field_normalized,
    "s6e4_groupby_aggs": s6e4_groupby_aggs,
    "s6e4_quantile_bins": s6e4_quantile_bins,
    "target_encode_multiclass": target_encode_multiclass,
    "s6e4_digit_extraction_wide": s6e4_digit_extraction_wide,
    "s6e4_cat_pair_combined_keys": s6e4_cat_pair_combined_keys,
    "s6e4_freq_filter_cats": s6e4_freq_filter_cats,
    "ordered_te": ordered_te,
}


def apply_blocks(X_tr: pd.DataFrame, X_te: pd.DataFrame, names: list[str], y_tr=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply feature blocks in order. Blocks whose signature accepts `y_tr`
    get the current training target — enables per-fold target encoding when
    this is called inside a CV loop."""
    for name in names:
        if name not in BLOCKS:
            raise KeyError(f"unknown feature block: {name!r}. known: {sorted(BLOCKS)}")
        fn = BLOCKS[name]
        if "y_tr" in inspect.signature(fn).parameters:
            X_tr, X_te = fn(X_tr, X_te, y_tr=y_tr)
        else:
            X_tr, X_te = fn(X_tr, X_te)
    return X_tr, X_te
