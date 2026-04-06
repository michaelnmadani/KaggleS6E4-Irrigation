"""Researcher Agent: EDA and data analysis."""

import numpy as np
import pandas as pd
from pipeline.config import (
    TARGET, CLASS_LABELS, CATEGORICAL_FEATURES, NUMERIC_FEATURES,
    SOIL_FEATURES, WEATHER_FEATURES, CROP_FEATURES, FIELD_FEATURES
)
from pipeline.utils import setup_logging

logger = setup_logging()


def run_eda(train, test):
    """Run exploratory data analysis on the dataset."""
    logger.info("Researcher Agent: Starting EDA...")

    results = {
        "dataset_stats": _dataset_stats(train, test),
        "class_distribution": _class_distribution(train),
        "missing_values": _missing_values(train, test),
        "numeric_stats": _numeric_stats(train),
        "categorical_stats": _categorical_stats(train),
        "correlations": _correlations(train),
        "sample_data": _sample_data(train),
        "feature_categories": {
            "soil": SOIL_FEATURES,
            "weather": WEATHER_FEATURES,
            "crop": CROP_FEATURES,
            "field": FIELD_FEATURES,
        }
    }

    logger.info("Researcher Agent: EDA complete.")
    return results


def _dataset_stats(train, test):
    return {
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "train_columns": train.columns.tolist(),
        "test_columns": test.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in train.dtypes.items()},
    }


def _class_distribution(train):
    dist = train[TARGET].value_counts().to_dict()
    pct = train[TARGET].value_counts(normalize=True).mul(100).round(2).to_dict()
    return {"counts": dist, "percentages": pct}


def _missing_values(train, test):
    train_missing = train.isnull().sum()
    test_missing = test.isnull().sum()
    return {
        "train": {col: int(v) for col, v in train_missing.items() if v > 0},
        "test": {col: int(v) for col, v in test_missing.items() if v > 0},
        "train_total": int(train_missing.sum()),
        "test_total": int(test_missing.sum()),
    }


def _numeric_stats(train):
    num_cols = [c for c in NUMERIC_FEATURES if c in train.columns]
    if not num_cols:
        return {}
    stats = train[num_cols].describe().round(4).to_dict()
    return stats


def _categorical_stats(train):
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in train.columns]
    result = {}
    for col in cat_cols:
        vc = train[col].value_counts().head(10).to_dict()
        result[col] = {
            "unique_count": int(train[col].nunique()),
            "top_values": {str(k): int(v) for k, v in vc.items()}
        }
    return result


def _correlations(train):
    num_cols = [c for c in NUMERIC_FEATURES if c in train.columns]
    if not num_cols:
        return {}
    corr = train[num_cols].corr().round(4)
    return corr.to_dict()


def _sample_data(train, n=5):
    sample = train.head(n)
    return {
        "columns": sample.columns.tolist(),
        "rows": sample.values.tolist()
    }
