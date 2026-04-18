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


# ------------------------------------------------------------------------
# Meta-stacker (V24): port of arXiv:2602.12469 (Mohammad & Meherab 2026).
# Redundancy-pruned + row-stat-augmented multinomial LogReg over base OOFs.
# ------------------------------------------------------------------------


def _prune_redundant_columns(M: np.ndarray, corr_thr: float = 0.95, mse_rel: float = 0.05) -> np.ndarray:
    """Boolean keep-mask. Drops column j if an earlier column i has
    |corr(i,j)| >= corr_thr AND mean((Mi-Mj)**2) <= (mse_rel * std(Mj)) ** 2."""
    n, p = M.shape
    std = M.std(axis=0)
    std_safe = np.where(std < 1e-12, 1.0, std)
    Z = (M - M.mean(axis=0)) / std_safe
    keep = np.ones(p, dtype=bool)
    for j in range(p):
        if not keep[j]:
            continue
        for i in range(j):
            if not keep[i]:
                continue
            c = float((Z[:, i] * Z[:, j]).mean())
            if abs(c) < corr_thr:
                continue
            mse = float(((M[:, i] - M[:, j]) ** 2).mean())
            if mse <= (mse_rel * max(std[j], 1e-9)) ** 2:
                keep[j] = False
                break
    return keep


def _rowwise_prob_stats(per_model_oof: dict, per_model_test: dict, n_classes: int):
    names = list(per_model_oof.keys())
    feats_tr, feats_te = [], []
    for k in range(n_classes):
        stack_tr = np.stack([per_model_oof[m][:, k] for m in names], axis=1)
        stack_te = np.stack([per_model_test[m][:, k] for m in names], axis=1)
        for arr_tr, arr_te in (
            (stack_tr.mean(1, keepdims=True), stack_te.mean(1, keepdims=True)),
            (stack_tr.std(1, keepdims=True), stack_te.std(1, keepdims=True)),
            (np.median(stack_tr, axis=1, keepdims=True), np.median(stack_te, axis=1, keepdims=True)),
            (stack_tr.max(1, keepdims=True) - stack_tr.min(1, keepdims=True),
             stack_te.max(1, keepdims=True) - stack_te.min(1, keepdims=True)),
        ):
            feats_tr.append(arr_tr)
            feats_te.append(arr_te)
    return np.concatenate(feats_tr, axis=1), np.concatenate(feats_te, axis=1)


def stack_meta_learner(per_model_oof: dict, per_model_test: dict,
                       y: np.ndarray, folds: np.ndarray, metric_fn) -> dict:
    """Redundancy-pruned + stats-augmented multinomial LogReg stacker.
    Trains L1/L2/ElasticNet meta-learners per-fold; inverse-error blends them."""
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y).astype(int)
    folds = np.asarray(folds)
    names = list(per_model_oof.keys())
    n_classes = per_model_oof[names[0]].shape[1]

    base_tr = np.concatenate([per_model_oof[m] for m in names], axis=1)
    base_te = np.concatenate([per_model_test[m] for m in names], axis=1)
    stats_tr, stats_te = _rowwise_prob_stats(per_model_oof, per_model_test, n_classes)

    M_tr = np.concatenate([base_tr, stats_tr], axis=1)
    M_te = np.concatenate([base_te, stats_te], axis=1)

    keep = _prune_redundant_columns(M_tr, corr_thr=0.95, mse_rel=0.05)
    M_tr = M_tr[:, keep]
    M_te = M_te[:, keep]

    M_tr = np.clip(M_tr, 1e-6, 1 - 1e-6)
    M_te = np.clip(M_te, 1e-6, 1 - 1e-6)

    learners = [
        ("l2", {"penalty": "l2", "C": 1.0, "solver": "lbfgs"}),
        ("l1", {"penalty": "l1", "C": 0.5, "solver": "saga"}),
        ("en", {"penalty": "elasticnet", "C": 0.5, "solver": "saga", "l1_ratio": 0.5}),
    ]

    oof_by, test_by, score_by = {}, {}, {}
    uniq_folds = np.unique(folds)
    for name, kw in learners:
        oof = np.zeros((len(y), n_classes))
        test_accum = np.zeros((M_te.shape[0], n_classes))
        for f in uniq_folds:
            tr_mask = folds != f
            va_mask = folds == f
            clf = LogisticRegression(
                max_iter=500, multi_class="multinomial",
                class_weight="balanced", n_jobs=-1, **kw,
            )
            clf.fit(M_tr[tr_mask], y[tr_mask])
            oof[va_mask] = clf.predict_proba(M_tr[va_mask])
            test_accum += clf.predict_proba(M_te) / len(uniq_folds)
        oof_by[name] = oof
        test_by[name] = test_accum
        score_by[name] = float(metric_fn(y, oof))

    raw = np.array([max(1e-6, 1.0 - score_by[n]) for n, _ in learners])
    inv = 1.0 / raw
    w = inv / inv.sum()
    weights = {name: float(wi) for (name, _), wi in zip(learners, w)}
    blended_oof = sum(weights[name] * oof_by[name] for name, _ in learners)
    blended_test = sum(weights[name] * test_by[name] for name, _ in learners)
    blended_score = float(metric_fn(y, blended_oof))

    return {
        "oof": blended_oof,
        "test": blended_test,
        "score": blended_score,
        "per_learner_score": score_by,
        "weights": weights,
        "n_meta_features": int(M_tr.shape[1]),
        "n_meta_features_pre_prune": int(keep.shape[0]),
        "n_pruned": int((~keep).sum()),
    }
