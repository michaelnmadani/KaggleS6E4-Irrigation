"""Orchestrator: reads an iteration config.yaml, runs CV, writes artifacts.

Designed to be called from a Kaggle notebook cell:

    from pipeline.src.train import run
    run(config_path="iterations/001_baseline/config.yaml",
        input_dir="/kaggle/input/<comp-slug>",
        output_dir="/kaggle/working")

Writes into output_dir:
    metrics.json   — CV score, per-fold scores, plain+balanced accuracy
    oof.csv        — out-of-fold predictions (id, per-class probs, pred)
    submission.csv — test predictions in competition format
    logs.txt       — human-readable run log

Config knobs (in iteration's config.yaml):
    features:      — list of block names from features.BLOCKS
    features_per_fold: (optional) list of SUPERVISED blocks (e.g. target encoding)
                     that must run inside the CV loop to avoid leakage
    class_weights: balanced|null  — if "balanced", pass sklearn-style weights to fit
"""
from __future__ import annotations

import inspect
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, log_loss,
    mean_absolute_error, mean_squared_error, roc_auc_score,
)

from . import data as data_mod
from . import features as feat_mod
from . import models as models_mod


def _hard_labels(p: np.ndarray) -> np.ndarray:
    return (p > 0.5).astype(int) if p.ndim == 1 else p.argmax(1)


METRICS = {
    "auc": lambda y, p: roc_auc_score(y, p),
    "logloss": lambda y, p: log_loss(y, p),
    "accuracy": lambda y, p: accuracy_score(y, _hard_labels(p)),
    "balanced_accuracy": lambda y, p: balanced_accuracy_score(y, _hard_labels(p)),
    "rmse": lambda y, p: float(np.sqrt(mean_squared_error(y, p))),
    "mae": lambda y, p: mean_absolute_error(y, p),
}


def _split_feature_blocks(names: list[str]) -> tuple[list[str], list[str]]:
    """Return (global_blocks, per_fold_blocks). Per-fold = supervised = needs y_tr."""
    glob, per = [], []
    for n in names:
        if n not in feat_mod.BLOCKS:
            raise KeyError(f"unknown feature block: {n!r}")
        if "y_tr" in inspect.signature(feat_mod.BLOCKS[n]).parameters:
            per.append(n)
        else:
            glob.append(n)
    return glob, per


def run(config_path: str, input_dir: str, output_dir: str) -> dict:
    t0 = time.time()
    cfg = yaml.safe_load(Path(config_path).read_text())
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = [f"config: {config_path}"]

    X, y, X_test, test_ids, inverse_label_map = data_mod.load(Path(input_dir), cfg["target"], cfg["id_col"])
    log_lines.append(f"train={X.shape}  test={X_test.shape}")
    if inverse_label_map is not None:
        log_lines.append(f"target encoded: {inverse_label_map}")

    # Split supervised (per-fold) from unsupervised (global) feature blocks.
    feat_names = cfg.get("features", [])
    global_blocks, per_fold_blocks = _split_feature_blocks(feat_names)
    X, X_test = feat_mod.apply_blocks(X, X_test, global_blocks)
    log_lines.append(f"global features: {global_blocks} -> {X.shape[1]} cols")
    if per_fold_blocks:
        log_lines.append(f"per-fold features: {per_fold_blocks}")

    task = cfg["task"]
    metric_name = cfg["metric"]
    metric_fn = METRICS[metric_name]

    folds = data_mod.make_folds(y, cfg["cv"]["n_splits"], cfg["cv"]["seed"], cfg["cv"].get("stratified", task != "regression"))

    n_classes = int(pd.Series(y).nunique())
    oof = np.zeros(len(X)) if task != "multiclass" else np.zeros((len(X), n_classes))
    test_preds = np.zeros(len(X_test)) if task != "multiclass" else np.zeros((len(X_test), n_classes))
    fold_scores: list[float] = []
    n_features_final = X.shape[1]

    class_weights_mode = cfg.get("class_weights")

    for f in range(cfg["cv"]["n_splits"]):
        tr_idx, va_idx = np.where(folds != f)[0], np.where(folds == f)[0]
        X_tr, y_tr = X.iloc[tr_idx].reset_index(drop=True), y.iloc[tr_idx].reset_index(drop=True)
        X_va, y_va = X.iloc[va_idx].reset_index(drop=True), y.iloc[va_idx].reset_index(drop=True)
        X_te_fold = X_test.copy()

        # Apply supervised blocks per-fold to avoid leakage. For each supervised
        # block we fit on (X_tr, y_tr) and transform X_va + X_te_fold jointly.
        for name in per_fold_blocks:
            fn = feat_mod.BLOCKS[name]
            val_test = pd.concat([X_va, X_te_fold], axis=0, ignore_index=True)
            X_tr, val_test = fn(X_tr, val_test, y_tr=y_tr)
            X_va = val_test.iloc[: len(X_va)].reset_index(drop=True)
            X_te_fold = val_test.iloc[len(X_va) :].reset_index(drop=True)
        n_features_final = X_tr.shape[1]

        sw = models_mod.compute_balanced_sample_weights(y_tr) if class_weights_mode == "balanced" else None
        res = models_mod.fit_one_fold(
            cfg["model"], X_tr, y_tr, X_va, y_va, X_te_fold,
            cfg.get("params", {}), task, sample_weight=sw,
        )
        oof[va_idx] = res.val_pred
        test_preds += res.test_pred / cfg["cv"]["n_splits"]
        s = metric_fn(y_va, res.val_pred)
        fold_scores.append(float(s))
        log_lines.append(f"fold {f}: {metric_name}={s:.5f}")

    cv_score = float(metric_fn(y, oof))
    plain_acc = float(accuracy_score(y, _hard_labels(oof)))
    bal_acc = float(balanced_accuracy_score(y, _hard_labels(oof)))

    # Per-class recall (multiclass) for reviewer context.
    recalls = None
    if task == "multiclass" and inverse_label_map is not None:
        preds = _hard_labels(oof)
        recalls = {}
        y_arr = np.asarray(y)
        for k in range(n_classes):
            mask = y_arr == k
            recalls[inverse_label_map[int(k)]] = float((preds[mask] == k).mean()) if mask.any() else None

    # OOF CSV
    if oof.ndim == 1:
        oof_df = pd.DataFrame({cfg["id_col"]: np.arange(len(oof)), "pred": oof})
    else:
        oof_df = pd.DataFrame(oof, columns=[f"p_class_{i}" for i in range(oof.shape[1])])
        oof_df.insert(0, cfg["id_col"], np.arange(len(oof)))
        oof_df["pred"] = oof.argmax(1)
    oof_df.to_csv(out / "oof.csv", index=False)

    # Submission: integer labels for hard-label metrics or multiclass;
    # map back to string labels if target was originally string.
    hard_label_metrics = {"accuracy", "balanced_accuracy"}
    if test_preds.ndim == 1:
        test_labels = test_preds
    elif metric_name in hard_label_metrics or task == "multiclass":
        test_labels = test_preds.argmax(1)
    else:
        test_labels = test_preds
    if inverse_label_map is not None and np.ndim(test_labels) == 1:
        test_labels = np.array([inverse_label_map[int(v)] for v in test_labels])
    pd.DataFrame({cfg["id_col"]: test_ids, cfg["target"]: test_labels}).to_csv(out / "submission.csv", index=False)

    metrics = {
        "cv_score": cv_score,
        "metric": metric_name,
        "plain_accuracy": plain_acc,
        "balanced_accuracy": bal_acc,
        "fold_scores": fold_scores,
        "fold_mean": float(np.mean(fold_scores)),
        "fold_std": float(np.std(fold_scores)),
        "per_class_recall": recalls,
        "elapsed_sec": round(time.time() - t0, 1),
        "model": cfg["model"],
        "n_features": int(n_features_final),
        "n_train": int(len(X)),
        "n_test": int(len(X_test)),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    log_lines.append(f"CV {metric_name}={cv_score:.5f}  plain_acc={plain_acc:.5f}  balanced_acc={bal_acc:.5f}")
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
