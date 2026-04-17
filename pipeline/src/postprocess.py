"""Post-processing: per-class logit-bias tuning for balanced accuracy.

Idea: after the CV loop produces per-class OOF probabilities, find a bias
vector b such that softmax(log(probs) + b).argmax scores best on balanced
accuracy. On skewed-class problems this can lift balanced-acc by 0.003-0.006
because argmax under-predicts the minority class.

Naive calibration (learning 3 biases on all OOF) can overfit OOF noise.
We use NESTED CV: hold out one fold at a time, tune bias on the other K-1
folds' OOF, apply to the held-out fold. The OOF-reported score after this
is an unbiased estimate of the post-bias gain.

Final step: refit bias on ALL folds' OOF and apply to the test probabilities
before argmax. The nested-CV number tells us if this is a safe move.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import balanced_accuracy_score


def _apply_bias(probs: np.ndarray, b: np.ndarray) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-9, 1.0)) + b
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    return e / e.sum(axis=1, keepdims=True)


def _neg_balanced_accuracy(b: np.ndarray, probs: np.ndarray, y: np.ndarray) -> float:
    preds = _apply_bias(probs, b).argmax(axis=1)
    return -balanced_accuracy_score(y, preds)


def tune_bias(probs: np.ndarray, y: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
    """Find bias vector maximizing balanced-accuracy via Nelder-Mead."""
    n_classes = probs.shape[1]
    if x0 is None:
        x0 = np.zeros(n_classes)
    res = minimize(
        _neg_balanced_accuracy,
        x0=x0,
        args=(probs, y),
        method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-5, "maxiter": 600},
    )
    return res.x


def tune_bias_nested_cv(probs: np.ndarray, y: np.ndarray, fold_idx: np.ndarray) -> dict:
    """For each fold, tune bias on the other folds' OOF, apply to held-out.
    Returns dict with pre/post nested-CV balanced accuracy and the ALL-FOLDS bias
    (for applying to test).
    """
    y = np.asarray(y)
    fold_idx = np.asarray(fold_idx)
    post_preds = np.zeros_like(y)
    for f in np.unique(fold_idx):
        tr_mask = fold_idx != f
        va_mask = fold_idx == f
        b = tune_bias(probs[tr_mask], y[tr_mask])
        post_preds[va_mask] = _apply_bias(probs[va_mask], b).argmax(axis=1)
    pre = balanced_accuracy_score(y, probs.argmax(axis=1))
    post = balanced_accuracy_score(y, post_preds)
    b_all = tune_bias(probs, y)
    return {"pre_bias_bal_acc": float(pre), "post_bias_bal_acc_nested": float(post),
            "bias_all_folds": b_all.tolist(), "delta_nested": float(post - pre)}


def apply_bias_to_probs(probs: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _apply_bias(probs, np.asarray(b))
