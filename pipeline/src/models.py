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


def _lgbm_fit(X_tr, y_tr, X_val, y_val, X_test, params: dict, task: str) -> FitResult:
    import lightgbm as lgb

    base = {
        "objective": "binary" if task == "binary" else ("multiclass" if task == "multiclass" else "regression"),
        "verbosity": -1,
        "seed": 42,
    }
    if task == "multiclass":
        base["num_class"] = int(pd.Series(y_tr).nunique())
    base.update(params)
    dtr = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtr)
    model = lgb.train(
        base,
        dtr,
        num_boost_round=base.pop("num_boost_round", 2000),
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(base.pop("early_stopping_rounds", 100)), lgb.log_evaluation(0)],
    )
    return FitResult(model=model, val_pred=model.predict(X_val), test_pred=model.predict(X_test))


def _xgb_fit(X_tr, y_tr, X_val, y_val, X_test, params: dict, task: str) -> FitResult:
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
    dtr = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)
    model = xgb.train(
        base, dtr, num_boost_round=num_boost_round,
        evals=[(dval, "val")], early_stopping_rounds=early_stop, verbose_eval=False,
    )
    return FitResult(model=model, val_pred=model.predict(dval), test_pred=model.predict(dtest))


def _catboost_fit(X_tr, y_tr, X_val, y_val, X_test, params: dict, task: str) -> FitResult:
    from catboost import CatBoostClassifier, CatBoostRegressor

    base = {"verbose": 0, "random_seed": 42, "iterations": 2000, "early_stopping_rounds": 100}
    base.update(params)
    cls = CatBoostClassifier if task in ("binary", "multiclass") else CatBoostRegressor
    model = cls(**base)
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
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


FITTERS = {"lgbm": _lgbm_fit, "xgb": _xgb_fit, "catboost": _catboost_fit}


def fit_one_fold(name: str, X_tr, y_tr, X_val, y_val, X_test, params: dict, task: str) -> FitResult:
    if name not in FITTERS:
        raise KeyError(f"unknown model {name!r}; known: {sorted(FITTERS)}")
    return FITTERS[name](X_tr, y_tr, X_val, y_val, X_test, params, task)
