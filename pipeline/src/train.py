"""Orchestrator: reads an iteration config.yaml, runs CV, writes artifacts.

Designed to be called from a Kaggle notebook cell:

    from pipeline.src.train import run
    run(config_path="iterations/001_baseline/config.yaml",
        input_dir="/kaggle/input/<comp-slug>",
        output_dir="/kaggle/working")

Writes into output_dir:
    metrics.json   — CV score, per-fold scores, fit time
    oof.csv        — out-of-fold predictions (id, pred)
    submission.csv — test predictions in competition format
    logs.txt       — human-readable run log
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score, log_loss, mean_absolute_error, mean_squared_error,
    roc_auc_score,
)

from . import data as data_mod
from . import features as feat_mod
from . import models as models_mod


METRICS = {
    "auc": lambda y, p: roc_auc_score(y, p),
    "logloss": lambda y, p: log_loss(y, p),
    "accuracy": lambda y, p: accuracy_score(y, (p > 0.5).astype(int) if p.ndim == 1 else p.argmax(1)),
    "rmse": lambda y, p: float(np.sqrt(mean_squared_error(y, p))),
    "mae": lambda y, p: mean_absolute_error(y, p),
}


def run(config_path: str, input_dir: str, output_dir: str) -> dict:
    t0 = time.time()
    cfg = yaml.safe_load(Path(config_path).read_text())
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = [f"config: {config_path}"]

    X, y, X_test, test_ids = data_mod.load(Path(input_dir), cfg["target"], cfg["id_col"])
    log_lines.append(f"train={X.shape}  test={X_test.shape}")

    X, X_test = feat_mod.apply_blocks(X, X_test, cfg.get("features", []))
    log_lines.append(f"features applied: {cfg.get('features', [])} -> {X.shape[1]} cols")

    task = cfg["task"]
    metric_name = cfg["metric"]
    metric_fn = METRICS[metric_name]

    folds = data_mod.make_folds(y, cfg["cv"]["n_splits"], cfg["cv"]["seed"], cfg["cv"].get("stratified", task != "regression"))

    oof = np.zeros(len(X)) if task != "multiclass" else np.zeros((len(X), int(y.nunique())))
    test_preds = np.zeros(len(X_test)) if task != "multiclass" else np.zeros((len(X_test), int(y.nunique())))
    fold_scores: list[float] = []

    for f in range(cfg["cv"]["n_splits"]):
        tr_idx, va_idx = np.where(folds != f)[0], np.where(folds == f)[0]
        res = models_mod.fit_one_fold(
            cfg["model"], X.iloc[tr_idx], y.iloc[tr_idx], X.iloc[va_idx], y.iloc[va_idx],
            X_test, cfg.get("params", {}), task,
        )
        oof[va_idx] = res.val_pred
        test_preds += res.test_pred / cfg["cv"]["n_splits"]
        s = metric_fn(y.iloc[va_idx], res.val_pred)
        fold_scores.append(float(s))
        log_lines.append(f"fold {f}: {metric_name}={s:.5f}")

    cv_score = float(metric_fn(y, oof))
    elapsed = time.time() - t0

    # artifacts
    pd.DataFrame({cfg["id_col"]: np.arange(len(oof)), "pred": oof if oof.ndim == 1 else oof.tolist()}).to_csv(out / "oof.csv", index=False)
    sub = pd.DataFrame({cfg["id_col"]: test_ids, cfg["target"]: test_preds if test_preds.ndim == 1 else test_preds.tolist()})
    sub.to_csv(out / "submission.csv", index=False)

    metrics = {
        "cv_score": cv_score,
        "metric": metric_name,
        "fold_scores": fold_scores,
        "fold_mean": float(np.mean(fold_scores)),
        "fold_std": float(np.std(fold_scores)),
        "elapsed_sec": round(elapsed, 1),
        "model": cfg["model"],
        "n_features": int(X.shape[1]),
        "n_train": int(len(X)),
        "n_test": int(len(X_test)),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log_lines.append(f"CV {metric_name}={cv_score:.5f}  elapsed={elapsed:.1f}s")
    (out / "logs.txt").write_text("\n".join(log_lines))
    return metrics


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    print(json.dumps(run(args.config, args.input_dir, args.output_dir), indent=2))
