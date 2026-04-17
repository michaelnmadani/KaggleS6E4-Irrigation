"""Load competition data, encode string targets, build CV folds."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


def load(input_dir: Path, target: str, id_col: str):
    """Return X, y, X_test, test_ids, inverse_label_map.

    If the target is string-typed (e.g. multiclass labels like Low/Medium/High)
    we encode it to integers using sorted unique values and return the inverse
    map so submission can write labels back in the original format.
    """
    train = pd.read_csv(input_dir / "train.csv")
    test = pd.read_csv(input_dir / "test.csv")
    y = train[target]

    inverse_label_map: dict[int, object] | None = None
    if y.dtype == object:
        classes = sorted(y.dropna().unique().tolist())
        label_map = {c: i for i, c in enumerate(classes)}
        inverse_label_map = {i: c for c, i in label_map.items()}
        y = y.map(label_map).astype(int)

    X = train.drop(columns=[target, id_col], errors="ignore")
    X_test = test.drop(columns=[id_col], errors="ignore")
    test_ids = test[id_col]
    return X, y, X_test, test_ids, inverse_label_map


def make_folds(y: pd.Series, n_splits: int, seed: int, stratified: bool) -> np.ndarray:
    splitter_cls = StratifiedKFold if stratified else KFold
    splitter = splitter_cls(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_idx = np.zeros(len(y), dtype=int)
    split_target = y if stratified else np.zeros(len(y))
    for fold, (_, val_idx) in enumerate(splitter.split(np.zeros(len(y)), split_target)):
        fold_idx[val_idx] = fold
    return fold_idx
