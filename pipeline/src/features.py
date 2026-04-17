"""Feature-engineering blocks addressable by name from config.yaml.

Each block takes (X_train, X_test) and returns (X_train, X_test) with new/modified
columns. Blocks must be pure and deterministic. Register blocks in BLOCKS at bottom.
"""
from __future__ import annotations

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


BLOCKS = {
    "label_encode": label_encode,
    "fill_na_median": fill_na_median,
    "count_encode_categoricals": count_encode_categoricals,
    "s6e4_threshold_booleans": s6e4_threshold_booleans,
}


def apply_blocks(X_tr: pd.DataFrame, X_te: pd.DataFrame, names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    for name in names:
        if name not in BLOCKS:
            raise KeyError(f"unknown feature block: {name!r}. known: {sorted(BLOCKS)}")
        X_tr, X_te = BLOCKS[name](X_tr, X_te)
    return X_tr, X_te
