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


def target_encode_multiclass(X_tr: pd.DataFrame, X_te: pd.DataFrame, y_tr=None, smoothing: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each categorical col and each target class k, add a smoothed P(y=k|category).
    Multiclass-aware TE: captures per-class conditional probabilities (V11's biggest lever)."""
    if y_tr is None:
        raise ValueError("target_encode_multiclass requires y_tr")
    X_tr, X_te = X_tr.copy(), X_te.copy()
    y = pd.Series(y_tr).reset_index(drop=True)
    classes = sorted(y.unique().tolist())
    for col in _categorical_cols(X_tr):
        for k in classes:
            tgt = (y == k).astype(int)
            global_mean = float(tgt.mean())
            temp = pd.DataFrame({"cat": X_tr[col].reset_index(drop=True), "t": tgt.values})
            agg = temp.groupby("cat")["t"].agg(["mean", "count"])
            smooth = (agg["count"] * agg["mean"] + smoothing * global_mean) / (agg["count"] + smoothing)
            X_tr[f"{col}_te_c{int(k)}"] = X_tr[col].map(smooth).fillna(global_mean).astype(float).values
            X_te[f"{col}_te_c{int(k)}"] = X_te[col].map(smooth).fillna(global_mean).astype(float).values
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


BLOCKS = {
    "label_encode": label_encode,
    "fill_na_median": fill_na_median,
    "count_encode_categoricals": count_encode_categoricals,
    "s6e4_threshold_booleans": s6e4_threshold_booleans,
    "s6e4_interactions": s6e4_interactions,
    "s6e4_cdeotte_minimal": s6e4_cdeotte_minimal,
    "target_encode_multiclass": target_encode_multiclass,
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
