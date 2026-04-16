"""Load competition data and build CV folds."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


def load(input_dir: Path, target: str, id_col: str):
    train = pd.read_csv(input_dir / "train.csv")
    test = pd.read_csv(input_dir / "test.csv")
    y = train[target]
    X = train.drop(columns=[target, id_col], errors="ignore")
    X_test = test.drop(columns=[id_col], errors="ignore")
    test_ids = test[id_col]
    return X, y, X_test, test_ids


def make_folds(y: pd.Series, n_splits: int, seed: int, stratified: bool) -> np.ndarray:
    splitter_cls = StratifiedKFold if stratified else KFold
    splitter = splitter_cls(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_idx = np.zeros(len(y), dtype=int)
    split_target = y if stratified else np.zeros(len(y))
    for fold, (_, val_idx) in enumerate(splitter.split(np.zeros(len(y)), split_target)):
        fold_idx[val_idx] = fold
    return fold_idx
