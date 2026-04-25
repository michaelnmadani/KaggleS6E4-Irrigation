"""Thin wrappers over gradient-boosting libraries with a uniform fit/predict API."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class FitResult:
    model: Any
    val_pred: np.ndarray
    test_pred: np.ndarray


def _lgbm_fit(X_tr, y_tr, X_val, y_val, X_test, params, task, sample_weight=None) -> FitResult:
    import lightgbm as lgb

    base = {
        "objective": "binary" if task == "binary" else ("multiclass" if task == "multiclass" else "regression"),
        "verbosity": -1,
        "seed": 42,
    }
    if task == "multiclass":
        base["num_class"] = int(pd.Series(y_tr).nunique())
    base.update(params)
    dtr = lgb.Dataset(X_tr, label=y_tr, weight=sample_weight)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtr)
    model = lgb.train(
        base,
        dtr,
        num_boost_round=base.pop("num_boost_round", 2000),
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(base.pop("early_stopping_rounds", 100)), lgb.log_evaluation(0)],
    )
    return FitResult(model=model, val_pred=model.predict(X_val), test_pred=model.predict(X_test))


def _xgb_fit(X_tr, y_tr, X_val, y_val, X_test, params, task, sample_weight=None) -> FitResult:
    import xgboost as xgb

    base = {
        "objective": "binary:logistic" if task == "binary" else ("multi:softprob" if task == "multiclass" else "reg:squarederror"),
        "verbosity": 0,
        "seed": 42,
    }
    if task == "multiclass":
        base["num_class"] = int(pd.Series(y_tr).nunique())
    base.update(params)
    num_boost_round = base.pop("num_boost_round", 2000)
    early_stop = base.pop("early_stopping_rounds", 100)
    dtr = xgb.DMatrix(X_tr, label=y_tr, weight=sample_weight)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)
    try:
        model = xgb.train(
            base, dtr, num_boost_round=num_boost_round,
            evals=[(dval, "val")], early_stopping_rounds=early_stop, verbose_eval=False,
        )
    except TypeError:
        # xgboost 2.x removed the early_stopping_rounds kwarg — use callback.
        from xgboost.callback import EarlyStopping
        model = xgb.train(
            base, dtr, num_boost_round=num_boost_round,
            evals=[(dval, "val")], callbacks=[EarlyStopping(rounds=early_stop, save_best=True)],
            verbose_eval=False,
        )
    return FitResult(model=model, val_pred=model.predict(dval), test_pred=model.predict(dtest))


def _catboost_fit(X_tr, y_tr, X_val, y_val, X_test, params, task, sample_weight=None) -> FitResult:
    from catboost import CatBoostClassifier, CatBoostRegressor

    base = {"verbose": 0, "random_seed": 42, "iterations": 2000, "early_stopping_rounds": 100}
    base.update(params)
    cls = CatBoostClassifier if task in ("binary", "multiclass") else CatBoostRegressor
    model = cls(**base)
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), sample_weight=sample_weight)
    if task == "binary":
        val_pred = model.predict_proba(X_val)[:, 1]
        test_pred = model.predict_proba(X_test)[:, 1]
    elif task == "multiclass":
        val_pred = model.predict_proba(X_val)
        test_pred = model.predict_proba(X_test)
    else:
        val_pred = model.predict(X_val)
        test_pred = model.predict(X_test)
    return FitResult(model=model, val_pred=val_pred, test_pred=test_pred)


def _extra_trees_fit(X_tr, y_tr, X_val, y_val, X_test, params, task, sample_weight=None) -> FitResult:
    """ExtraTreesClassifier (Geurts 2006) — extremely randomized trees with
    random split thresholds. Adds hypothesis-space diversity to GBM-dominated
    ensembles. Uses numeric X (label-encoded cats already)."""
    from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
    base = {
        "n_estimators": 800,
        "min_samples_leaf": 50,
        "max_features": 0.85,
        "n_jobs": -1,
        "random_state": 42,
    }
    base.update({k: v for k, v in params.items() if k in base or k in {"max_depth", "bootstrap", "criterion"}})
    X_tr_num = X_tr.select_dtypes(include=[np.number]).fillna(0)
    X_val_num = X_val.reindex(columns=X_tr_num.columns, fill_value=0).select_dtypes(include=[np.number]).fillna(0)
    X_test_num = X_test.reindex(columns=X_tr_num.columns, fill_value=0).select_dtypes(include=[np.number]).fillna(0)
    if task in ("binary", "multiclass"):
        model = ExtraTreesClassifier(**base, class_weight=None)
        model.fit(X_tr_num, y_tr, sample_weight=sample_weight)
        if task == "binary":
            val_pred = model.predict_proba(X_val_num)[:, 1]
            test_pred = model.predict_proba(X_test_num)[:, 1]
        else:
            val_pred = model.predict_proba(X_val_num)
            test_pred = model.predict_proba(X_test_num)
    else:
        model = ExtraTreesRegressor(**base)
        model.fit(X_tr_num, y_tr, sample_weight=sample_weight)
        val_pred = model.predict(X_val_num)
        test_pred = model.predict(X_test_num)
    return FitResult(model=model, val_pred=val_pred, test_pred=test_pred)


def _tabpfn_bagged_fit(X_tr, y_tr, X_val, y_val, X_test, params, task, sample_weight=None) -> FitResult:
    """TabPFN v2 bagged subsample ensemble — orthogonal base to GBM stack.

    TabPFN is a transformer foundation model for tabular. Pretrained on synthetic
    priors, it works via in-context learning on small train-set subsamples.
    For large-data (640k rows) we bag several stratified subsamples of ~10k each,
    predict val+test with each TabPFN instance, and average. Ensemble decision
    boundaries are attention-based (smooth, Bayesian-style) — decorrelated with
    axis-aligned GBM splits. Refs: arXiv:2502.17361 (TabPFN v2), 2511.08667 (TabArena).
    """
    from tabpfn import TabPFNClassifier
    n_bags = int(params.get("n_bags", 8))
    bag_size = int(params.get("bag_size", 10000))
    n_ests = int(params.get("n_estimators", 4))
    device = params.get("device", "cuda")
    rng = np.random.default_rng(int(params.get("random_state", 42)))

    if task not in ("binary", "multiclass"):
        raise ValueError("tabpfn_bagged only supports classification tasks")

    y_tr_arr = np.asarray(y_tr).astype(int)
    classes = np.sort(np.unique(y_tr_arr))
    n_classes = len(classes)

    # TabPFN is sensitive to categorical dtypes — feed pure float32 numerics.
    num_cols = X_tr.select_dtypes(include=[np.number]).columns.tolist()
    X_tr_num = X_tr[num_cols].fillna(0.0).astype(np.float32)
    X_val_num = X_val.reindex(columns=num_cols, fill_value=0).fillna(0.0).astype(np.float32).values
    X_test_num = X_test.reindex(columns=num_cols, fill_value=0).fillna(0.0).astype(np.float32).values

    # Cap feature dim: TabPFN's native support is up to ~500 features; we're well
    # under that, but skip any constant cols that sklearn/TabPFN will drop anyway.
    stds = X_tr_num.std(axis=0)
    keep = stds > 1e-9
    X_tr_num = X_tr_num.loc[:, keep]
    X_val_num = X_val_num[:, keep.values]
    X_test_num = X_test_num[:, keep.values]

    val_sum = np.zeros((X_val_num.shape[0], n_classes), dtype=np.float64)
    test_sum = np.zeros((X_test_num.shape[0], n_classes), dtype=np.float64)
    log_snippet = []

    for bag in range(n_bags):
        # Stratified subsample preserving class ratios.
        indices = []
        for k in classes:
            idx_k = np.where(y_tr_arr == k)[0]
            take = max(50, int(bag_size * len(idx_k) / len(y_tr_arr)))
            take = min(take, len(idx_k))
            indices.extend(rng.choice(idx_k, size=take, replace=False))
        indices = np.asarray(indices)
        X_bag = X_tr_num.iloc[indices].values
        y_bag = y_tr_arr[indices]
        clf = TabPFNClassifier(
            device=device,
            n_estimators=n_ests,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        clf.fit(X_bag, y_bag)
        val_sum += clf.predict_proba(X_val_num)
        test_sum += clf.predict_proba(X_test_num)
        log_snippet.append(f"bag{bag}:{len(indices)}")
    val_pred = val_sum / n_bags
    test_pred = test_sum / n_bags
    return FitResult(model=None, val_pred=val_pred, test_pred=test_pred)


def _logreg_fit(X_tr, y_tr, X_val, y_val, X_test, params, task, sample_weight=None) -> FitResult:
    """Multinomial logistic regression with one-hot categorical expansion.

    Fits `LogisticRegression(multi_class='multinomial')` on the numeric features
    plus one-hot-encoded categorical features. For S6E4 this replicates the
    cdeotte reference notebook that hit CV balanced-acc 1.0 on the original
    dataset (threshold booleans + one-hot categoricals + logreg). The model
    produces per-class logit equations (3 regressions for a 3-class target).
    """
    from sklearn.linear_model import LogisticRegression

    def _prep(df):
        # Detect remaining string categoricals that LGBM/XGB don't need expanded
        # but LogReg does. Keep numerics as-is and get_dummies the rest.
        df = df.copy()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=float)
        return df

    base = {
        "multi_class": "multinomial",
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 42,
        "n_jobs": -1,
    }
    if task in ("binary", "multiclass") and params.get("class_weight") is None and sample_weight is None:
        # If no explicit weights, match sklearn's "balanced" class_weight convention
        # for robustness on imbalanced targets. sample_weight (if set) supersedes.
        pass
    base.update({k: v for k, v in params.items() if k in {"C", "penalty", "l1_ratio", "max_iter", "solver", "multi_class", "class_weight", "random_state"}})

    # Align columns across train/val/test after one-hot (test may have unseen levels).
    all_cols = sorted(set(_prep(X_tr).columns) | set(_prep(X_val).columns) | set(_prep(X_test).columns))
    def _align(df):
        d = _prep(df)
        for c in all_cols:
            if c not in d.columns:
                d[c] = 0.0
        return d[all_cols]

    X_tr_p = _align(X_tr)
    X_val_p = _align(X_val)
    X_test_p = _align(X_test)

    model = LogisticRegression(**base)
    model.fit(X_tr_p, y_tr, sample_weight=sample_weight)
    if task == "binary":
        val_pred = model.predict_proba(X_val_p)[:, 1]
        test_pred = model.predict_proba(X_test_p)[:, 1]
    elif task == "multiclass":
        val_pred = model.predict_proba(X_val_p)
        test_pred = model.predict_proba(X_test_p)
    else:
        val_pred = model.predict(X_val_p)
        test_pred = model.predict(X_test_p)
    return FitResult(model=model, val_pred=val_pred, test_pred=test_pred)


def _realmlp_fit(X_tr, y_tr, X_val, y_val, X_test, params, task, sample_weight=None) -> FitResult:
    """RealMLP (deep-learning MLP with periodic numeric embeddings).

    sample_weight is IGNORED: RealMLP applies class_weight='balanced' internally
    via the smooth_ce loss, which is equivalent for our per-sample balancing.

    Expects params dict with optional RealMLP hyperparams (see realmlp.REALMLP_DEFAULT_CONFIG).
    The special key `cat_cols` (list) tells the model which columns to treat as
    categorical; all others are numeric.
    """
    from . import realmlp as rm

    overrides = {k: v for k, v in params.items() if k != "cat_cols"}
    cat_cols = params.get("cat_cols")
    if cat_cols is None:
        # Default: all columns whose dtype is integer or category are treated as categorical.
        cat_cols = [c for c in X_tr.columns if str(X_tr[c].dtype).startswith("int") or str(X_tr[c].dtype) == "category"]
    model = rm.RealMLPClassifier(**overrides)
    model.fit(X_tr, y_tr, X_val, y_val, cat_cols=list(cat_cols), X_test=X_test)
    val_pred = model.best_val_probs_
    test_pred = model.predict_proba(X_test)
    return FitResult(model=model, val_pred=val_pred, test_pred=test_pred)


def _pytabkit_realmlp_fit(X_tr, y_tr, X_val, y_val, X_test, params, task, sample_weight=None) -> FitResult:
    """Official pytabkit RealMLP_TD_Classifier by the RealMLP paper authors.
    Use this instead of our custom torch port when we want to match published
    benchmarks (my custom port scored ~0.018 below reference for unclear reasons).

    Requires: pip install pytabkit (installs torch-based RealMLP + utilities).
    """
    from pytabkit import RealMLP_TD_Classifier

    cat_cols = params.get("cat_col_names") or params.get("cat_cols")
    if cat_cols is None:
        cat_cols = [c for c in X_tr.columns
                    if str(X_tr[c].dtype).startswith("int") or str(X_tr[c].dtype) == "category"]
    overrides = {k: v for k, v in params.items() if k not in ("cat_col_names", "cat_cols")}

    model = RealMLP_TD_Classifier(**overrides)
    model.fit(X_tr, y_tr, X_val=X_val, y_val=y_val, cat_col_names=list(cat_cols))

    if task == "binary":
        val_pred = model.predict_proba(X_val)[:, 1]
        test_pred = model.predict_proba(X_test)[:, 1]
    else:
        val_pred = model.predict_proba(X_val)
        test_pred = model.predict_proba(X_test)
    return FitResult(model=model, val_pred=val_pred, test_pred=test_pred)


FITTERS = {
    "lgbm": _lgbm_fit,
    # Alias: LGBM in Random-Forest mode (boosting_type=rf in config params).
    # Provides RF-family diversity at native-LGBM speed (vs sklearn ExtraTrees).
    "lgbm_rf": _lgbm_fit,
    "tabpfn_bagged": _tabpfn_bagged_fit,
    "xgb": _xgb_fit,
    # Aliases for multi-seed XGB ensembling; config supplies distinct seed / depth.
    "xgb_a": _xgb_fit,
    "xgb_b": _xgb_fit,
    "xgb_c": _xgb_fit,
    "xgb_d": _xgb_fit,
    "catboost": _catboost_fit,
    # Aliases for multi-seed CatBoost ensembling; config supplies distinct random_seed/rsm.
    "catboost_a": _catboost_fit,
    "catboost_b": _catboost_fit,
    "extra_trees": _extra_trees_fit,
    "logreg": _logreg_fit,
    "realmlp": _realmlp_fit,
    "pytabkit_realmlp": _pytabkit_realmlp_fit,
}


def fit_one_fold(name, X_tr, y_tr, X_val, y_val, X_test, params, task, sample_weight=None) -> FitResult:
    if name not in FITTERS:
        raise KeyError(f"unknown model {name!r}; known: {sorted(FITTERS)}")
    return FITTERS[name](X_tr, y_tr, X_val, y_val, X_test, params, task, sample_weight=sample_weight)


def compute_balanced_sample_weights(y_tr) -> np.ndarray:
    """sklearn-style 'balanced': w_i = N / (n_classes * count(y_i))."""
    y = np.asarray(y_tr)
    classes, counts = np.unique(y, return_counts=True)
    n_classes = len(classes)
    total = len(y)
    class_weight = {c: total / (n_classes * cnt) for c, cnt in zip(classes, counts)}
    return np.array([class_weight[v] for v in y], dtype=float)
