"""Load competition data, encode string targets, build CV folds."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


def load(input_dir: Path, target: str, id_col: str, extra_dataset: dict | None = None):
    """Return X, y, X_test, test_ids, inverse_label_map, is_original_mask.

    If `extra_dataset` is set, appends rows from an external Kaggle dataset after
    aligning columns. is_original_mask is True for appended (external) rows,
    False for competition rows — train.py uses it to down-weight externals.
    """
    train = pd.read_csv(input_dir / "train.csv")
    test = pd.read_csv(input_dir / "test.csv")
    is_original = np.zeros(len(train), dtype=bool)

    if extra_dataset:
        ext_dir = Path(extra_dataset["mount_dir"])
        ext_file = ext_dir / extra_dataset["file"]
        ext = pd.read_csv(ext_file)
        # Keep only columns that also exist in the comp train (minus id).
        shared = [c for c in train.columns if c in ext.columns and c != id_col]
        if target not in shared:
            raise KeyError(f"extra_dataset missing target column {target!r}")
        ext_aligned = ext[shared].copy()
        # Dedupe against test by row-hash of numeric columns — extras that
        # appear verbatim in test would leak information.
        num_cols = ext_aligned.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            test_hash = set(map(tuple, test[num_cols].round(6).itertuples(index=False, name=None)))
            ext_hash = list(map(tuple, ext_aligned[num_cols].round(6).itertuples(index=False, name=None)))
            keep = [h not in test_hash for h in ext_hash]
            dropped = len(ext_aligned) - sum(keep)
            ext_aligned = ext_aligned[keep].reset_index(drop=True)
            print(f"extra_dataset: loaded {len(ext)} rows, dropped {dropped} test-row duplicates, kept {len(ext_aligned)}")
        train_aligned = train[[c for c in train.columns if c in shared or c == id_col]].copy()
        if id_col in train_aligned.columns and id_col not in ext_aligned.columns:
            ext_aligned[id_col] = -1 - np.arange(len(ext_aligned))  # synthetic ids
        train = pd.concat([train_aligned, ext_aligned], axis=0, ignore_index=True)
        is_original = np.concatenate([
            np.zeros(len(train) - len(ext_aligned), dtype=bool),
            np.ones(len(ext_aligned), dtype=bool),
        ])

    y = train[target]
    inverse_label_map: dict[int, object] | None = None
    if y.dtype == object:
        classes = sorted(y.dropna().unique().tolist())
        label_map = {c: i for i, c in enumerate(classes)}
        inverse_label_map = {i: c for c, i in label_map.items()}
        y = y.map(label_map).astype(int)

    X = train.drop(columns=[target, id_col], errors="ignore")
    X_test = test.drop(columns=[id_col], errors="ignore")
    # Ensure X and X_test have same columns in same order.
    X = X[[c for c in X.columns if c in X_test.columns]]
    X_test = X_test[X.columns]
    test_ids = test[id_col]
    return X, y, X_test, test_ids, inverse_label_map, is_original


def make_folds(y: pd.Series, n_splits: int, seed: int, stratified: bool) -> np.ndarray:
    splitter_cls = StratifiedKFold if stratified else KFold
    splitter = splitter_cls(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_idx = np.zeros(len(y), dtype=int)
    split_target = y if stratified else np.zeros(len(y))
    for fold, (_, val_idx) in enumerate(splitter.split(np.zeros(len(y)), split_target)):
        fold_idx[val_idx] = fold
    return fold_idx
