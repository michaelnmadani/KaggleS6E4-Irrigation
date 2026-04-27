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
from . import postprocess as post_mod


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

    # Resolve extra_dataset mount_dir from Kaggle slug if not explicitly set.
    extra_cfg = cfg.get("extra_dataset")
    if extra_cfg and "slug" in extra_cfg and "mount_dir" not in extra_cfg:
        extra_cfg["mount_dir"] = f"/kaggle/input/{extra_cfg['slug'].split('/')[-1]}"
    X, y, X_test, test_ids, inverse_label_map, is_original = data_mod.load(
        Path(input_dir), cfg["target"], cfg["id_col"], extra_dataset=extra_cfg,
    )
    log_lines.append(f"train={X.shape}  test={X_test.shape}  external_rows={int(is_original.sum())}")
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

    # Normalize `model` to a list of model names (single model = list of one).
    model_names = cfg["model"] if isinstance(cfg["model"], list) else [cfg["model"]]
    # `params` may be flat (shared across models) or a dict keyed by model name.
    raw_params = cfg.get("params", {})
    is_per_model_params = (
        isinstance(raw_params, dict)
        and len(raw_params) > 0
        and all(k in models_mod.FITTERS for k in raw_params.keys())
    )

    def _params_for(m: str) -> dict:
        if is_per_model_params:
            return dict(raw_params.get(m, {}))
        return dict(raw_params)

    shape_oof = (len(X),) if task != "multiclass" else (len(X), n_classes)
    shape_test = (len(X_test),) if task != "multiclass" else (len(X_test), n_classes)
    per_model_oof = {m: np.zeros(shape_oof) for m in model_names}
    per_model_test = {m: np.zeros(shape_test) for m in model_names}
    per_model_fold_scores: dict[str, list[float]] = {m: [] for m in model_names}
    n_features_final = X.shape[1]

    class_weights_mode = cfg.get("class_weights")

    # Adversarial-validation reweighting (V44): train binary XGB on
    # concat(train, test) with target=is_test, get per-train-row p(test),
    # use sample_weight_adv = clip(p/(1-p), 0.1, 10) so train rows that
    # "look test-like" get more weight in fold loss.
    adv_weights = None
    if cfg.get("adversarial_reweight"):
        log_lines.append("=== Computing adversarial-validation weights ===")
        try:
            import xgboost as xgb
            X_av_tr = X.select_dtypes(include=[np.number]).fillna(0)
            X_av_te = X_test.reindex(columns=X_av_tr.columns, fill_value=0).fillna(0)
            X_av = pd.concat([X_av_tr, X_av_te], axis=0, ignore_index=True)
            y_av = np.concatenate([np.zeros(len(X_av_tr)), np.ones(len(X_av_te))])
            d_av = xgb.DMatrix(X_av, label=y_av)
            av_params = {"objective": "binary:logistic", "max_depth": 4,
                         "eta": 0.1, "tree_method": "hist", "verbosity": 0,
                         "subsample": 0.8, "colsample_bytree": 0.8}
            n_rounds = int(cfg["adversarial_reweight"].get("n_rounds", 200) if isinstance(cfg["adversarial_reweight"], dict) else 200)
            model_av = xgb.train(av_params, d_av, num_boost_round=n_rounds)
            p_test_train = model_av.predict(xgb.DMatrix(X_av_tr))
            p_clip = np.clip(p_test_train, 1e-6, 1 - 1e-6)
            adv_weights = np.clip(p_clip / (1 - p_clip), 0.1, 10.0)
            adv_weights = adv_weights / adv_weights.mean()
            log_lines.append(
                f"AV weights: mean={adv_weights.mean():.4f} "
                f"median={np.median(adv_weights):.4f} "
                f"p10={np.percentile(adv_weights, 10):.4f} "
                f"p90={np.percentile(adv_weights, 90):.4f}"
            )
        except Exception as e:
            log_lines.append(f"AV reweight failed: {e}")
            adv_weights = None

    for f in range(cfg["cv"]["n_splits"]):
        tr_idx, va_idx = np.where(folds != f)[0], np.where(folds == f)[0]
        X_tr, y_tr = X.iloc[tr_idx].reset_index(drop=True), y.iloc[tr_idx].reset_index(drop=True)
        X_va, y_va = X.iloc[va_idx].reset_index(drop=True), y.iloc[va_idx].reset_index(drop=True)
        X_te_fold = X_test.copy()

        # Apply supervised blocks per-fold to avoid leakage.
        for name in per_fold_blocks:
            fn = feat_mod.BLOCKS[name]
            val_test = pd.concat([X_va, X_te_fold], axis=0, ignore_index=True)
            X_tr, val_test = fn(X_tr, val_test, y_tr=y_tr)
            X_va = val_test.iloc[: len(X_va)].reset_index(drop=True)
            X_te_fold = val_test.iloc[len(X_va) :].reset_index(drop=True)

        # Blocks like ordered_te may augment training rows (e.g. 4x shuffle
        # concatenation) and smuggle the corresponding shuffled labels via
        # _ote_y_shuffled. Extract them here so downstream model.fit sees
        # the augmented (X_tr, y_tr) pair with aligned sample_weight.
        if "_ote_y_shuffled" in X_tr.columns:
            y_tr = pd.Series(X_tr["_ote_y_shuffled"].values).reset_index(drop=True)
            X_tr = X_tr.drop(columns=["_ote_y_shuffled"])
        if "_ote_y_shuffled" in X_va.columns:
            X_va = X_va.drop(columns=["_ote_y_shuffled"])
        if "_ote_y_shuffled" in X_te_fold.columns:
            X_te_fold = X_te_fold.drop(columns=["_ote_y_shuffled"])
        n_features_final = X_tr.shape[1]

        sw = models_mod.compute_balanced_sample_weights(y_tr) if class_weights_mode == "balanced" else None
        # Down-weight external rows (miadul etc.) if configured via extra_dataset.weight.
        extra_weight = (cfg.get("extra_dataset") or {}).get("weight")
        if extra_weight is not None and is_original.any():
            # If ordered_te augmented rows, is_original[tr_idx] is shorter than y_tr;
            # tile it to match the augmented row count.
            is_orig_tr = is_original[tr_idx]
            if len(is_orig_tr) != len(y_tr):
                reps = len(y_tr) // len(is_orig_tr)
                is_orig_tr = np.tile(is_orig_tr, reps)
            if sw is None:
                sw = np.ones(len(y_tr), dtype=float)
            sw = sw * np.where(is_orig_tr, float(extra_weight), 1.0)

        # Adversarial-validation weights: multiply per-row so test-like train
        # rows get more weight in fold loss.
        if adv_weights is not None:
            adv_tr = adv_weights[tr_idx]
            if len(adv_tr) != len(y_tr):
                reps = len(y_tr) // len(adv_tr)
                adv_tr = np.tile(adv_tr, reps)
            if sw is None:
                sw = np.ones(len(y_tr), dtype=float)
            sw = sw * adv_tr

        for m in model_names:
            res = models_mod.fit_one_fold(
                m, X_tr, y_tr, X_va, y_va, X_te_fold,
                _params_for(m), task, sample_weight=sw,
            )
            per_model_oof[m][va_idx] = res.val_pred
            per_model_test[m] += res.test_pred / cfg["cv"]["n_splits"]
            s = metric_fn(y_va, res.val_pred)
            per_model_fold_scores[m].append(float(s))
            log_lines.append(f"fold {f} {m}: {metric_name}={s:.5f}")

    # Per-model CV scores (on full OOF).
    per_model_cv = {m: float(metric_fn(y, per_model_oof[m])) for m in model_names}
    for m in model_names:
        log_lines.append(f"model {m}: CV {metric_name}={per_model_cv[m]:.5f}")

    # Blend: score-proportional weights when multiple models, else pass-through.
    if len(model_names) == 1:
        oof = per_model_oof[model_names[0]]
        test_preds = per_model_test[model_names[0]]
        blend_weights = {model_names[0]: 1.0}
    else:
        manual_w = cfg.get("blend_weights")
        if isinstance(manual_w, dict) and set(manual_w.keys()) == set(model_names):
            raw_w = np.array([float(manual_w[m]) for m in model_names])
            raw_w = np.maximum(raw_w, 0.0)
            weight_mode = "manual"
        else:
            raw_w = np.array([per_model_cv[m] for m in model_names])
            raw_w = np.maximum(raw_w, 1e-9)
            weight_mode = "score_proportional"
        w = raw_w / raw_w.sum()
        blend_weights = {m: float(wi) for m, wi in zip(model_names, w)}
        oof = sum(wi * per_model_oof[m] for m, wi in zip(model_names, w))
        test_preds = sum(wi * per_model_test[m] for m, wi in zip(model_names, w))
        log_lines.append(f"blend weights ({weight_mode}): {blend_weights}")

    # Fold scores for the blend (or single model) for compatibility.
    fold_scores = [
        float(np.mean([per_model_fold_scores[m][f] for m in model_names]))
        for f in range(cfg["cv"]["n_splits"])
    ]
    cv_score = float(metric_fn(y, oof))
    plain_acc = float(accuracy_score(y, _hard_labels(oof)))
    bal_acc = float(balanced_accuracy_score(y, _hard_labels(oof)))

    # Normalize postprocess to list so multiple stages can chain (best-of applied).
    pp_raw = cfg.get("postprocess")
    pp_stages = pp_raw if isinstance(pp_raw, list) else ([pp_raw] if pp_raw else [])

    # Post-processing: logit-bias tuning for balanced accuracy.
    postprocess_info = None
    if "logit_bias" in pp_stages and oof.ndim == 2:
        info = post_mod.tune_bias_nested_cv(oof, np.asarray(y), folds)
        # Apply ALL-FOLDS bias to test probabilities before argmax.
        if test_preds.ndim == 2:
            test_preds = post_mod.apply_bias_to_probs(test_preds, info["bias_all_folds"])
        # Apply ALL-FOLDS bias to OOF too, so downstream uses the biased scores.
        oof_biased = post_mod.apply_bias_to_probs(oof, info["bias_all_folds"])
        post_cv = float(balanced_accuracy_score(np.asarray(y), oof_biased.argmax(axis=1)))
        postprocess_info = {**info, "post_bias_bal_acc_full": post_cv}
        log_lines.append(
            f"logit-bias: pre={info['pre_bias_bal_acc']:.5f}  nested={info['post_bias_bal_acc_nested']:.5f} "
            f"full={post_cv:.5f}  delta_nested={info['delta_nested']:+.5f}  bias={info['bias_all_folds']}"
        )
        # Update reported CV + oof to reflect the applied bias.
        oof = oof_biased
        cv_score = float(metric_fn(y, oof))
        plain_acc = float(accuracy_score(y, _hard_labels(oof)))
        bal_acc = float(balanced_accuracy_score(y, _hard_labels(oof)))

    # Post-processing: meta-stacker (V24). Requires >=2 base models.
    if "meta_stacker" in pp_stages and len(model_names) >= 2 and oof.ndim == 2:
        stack_info = post_mod.stack_meta_learner(
            per_model_oof, per_model_test, np.asarray(y), folds, metric_fn,
        )
        pre_score = cv_score
        meta_score = stack_info["score"]
        log_lines.append(
            f"meta-stacker: pre={pre_score:.5f}  meta={meta_score:.5f}  "
            f"delta={meta_score - pre_score:+.5f}  "
            f"per_learner={stack_info['per_learner_score']}  "
            f"weights={stack_info['weights']}  "
            f"pruned={stack_info['n_pruned']}/{stack_info['n_meta_features_pre_prune']}"
        )
        if meta_score > pre_score:
            oof = stack_info["oof"]
            test_preds = stack_info["test"]
            cv_score = float(metric_fn(y, oof))
            plain_acc = float(accuracy_score(y, _hard_labels(oof)))
            bal_acc = float(balanced_accuracy_score(y, _hard_labels(oof)))
            log_lines.append(f"meta-stacker applied (CV: {pre_score:.5f} -> {cv_score:.5f})")
        else:
            log_lines.append("meta-stacker skipped (did not improve over base blend)")
        postprocess_info = {
            "kind": "meta_stacker",
            "pre_score": pre_score,
            "meta_score": meta_score,
            "per_learner_score": stack_info["per_learner_score"],
            "weights": stack_info["weights"],
            "n_meta_features": stack_info["n_meta_features"],
            "n_pruned": stack_info["n_pruned"],
            "applied": meta_score > pre_score,
        }

    # Post-processing: class-weight Optuna (V34 — yunsuxiaozi port).
    if "class_weight_optuna" in pp_stages and oof.ndim == 2:
        cwo = post_mod.class_weight_optuna(oof, test_preds, np.asarray(y), metric_fn,
                                           n_trials=int(cfg.get("optuna_trials", 200)))
        log_lines.append(
            f"class_weight_optuna: pre={cwo['pre_score']:.5f}  post={cwo['post_score']:.5f}  "
            f"delta={cwo['post_score'] - cwo['pre_score']:+.5f}  weights={cwo['weights']}"
        )
        if cwo["post_score"] > cwo["pre_score"]:
            oof = cwo["oof"]
            test_preds = cwo["test"]
            cv_score = float(metric_fn(y, oof))
            plain_acc = float(accuracy_score(y, _hard_labels(oof)))
            bal_acc = float(balanced_accuracy_score(y, _hard_labels(oof)))
            log_lines.append(f"class_weight_optuna applied (CV: {cwo['pre_score']:.5f} -> {cv_score:.5f})")
        cwo_info = {
            "kind": "class_weight_optuna",
            "pre_score": cwo["pre_score"],
            "post_score": cwo["post_score"],
            "weights": cwo["weights"],
            "applied": cwo["post_score"] > cwo["pre_score"],
        }
        postprocess_info = (
            [postprocess_info, cwo_info] if isinstance(postprocess_info, dict) else
            ([*postprocess_info, cwo_info] if isinstance(postprocess_info, list) else cwo_info)
        )

    # Post-processing: per-class isotonic + additive bias (V53).
    if "per_class_isotonic" in pp_stages and oof.ndim == 2:
        pci = post_mod.per_class_isotonic_calibration(
            oof, test_preds, np.asarray(y), metric_fn,
            n_trials=int(cfg.get("optuna_trials", 200)),
        )
        log_lines.append(
            f"per_class_isotonic: pre={pci['pre_score']:.5f}  iso={pci['iso_score']:.5f}  "
            f"post={pci['post_score']:.5f}  delta={pci['post_score']-pci['pre_score']:+.5f}  "
            f"bias={pci['bias']}"
        )
        if pci["post_score"] > pci["pre_score"]:
            oof = pci["oof"]
            test_preds = pci["test"]
            cv_score = float(metric_fn(y, oof))
            plain_acc = float(accuracy_score(y, _hard_labels(oof)))
            bal_acc = float(balanced_accuracy_score(y, _hard_labels(oof)))
            log_lines.append(f"per_class_isotonic applied (CV: {pci['pre_score']:.5f} -> {cv_score:.5f})")
        pci_info = {
            "kind": "per_class_isotonic",
            "pre_score": pci["pre_score"],
            "iso_score": pci["iso_score"],
            "post_score": pci["post_score"],
            "bias": pci["bias"],
            "applied": pci["post_score"] > pci["pre_score"],
        }
        postprocess_info = (
            [postprocess_info, pci_info] if isinstance(postprocess_info, dict) else
            ([*postprocess_info, pci_info] if isinstance(postprocess_info, list) else pci_info)
        )

    # Post-processing: Caruana greedy hill-climb on balanced_accuracy (V26).
    # Runs AFTER meta-stacker — only accepted if it beats whatever's current.
    if "caruana" in pp_stages and len(model_names) >= 2 and oof.ndim == 2:
        ch = post_mod.caruana_hill_climb(
            per_model_oof, per_model_test, np.asarray(y), folds, metric_fn,
        )
        pre_score = cv_score
        caruana_score = ch["score"]
        log_lines.append(
            f"caruana: pre={pre_score:.5f}  caruana={caruana_score:.5f}  "
            f"delta={caruana_score - pre_score:+.5f}  weights={ch['weights']}"
        )
        if caruana_score > pre_score:
            oof = ch["oof"]
            test_preds = ch["test"]
            cv_score = float(metric_fn(y, oof))
            plain_acc = float(accuracy_score(y, _hard_labels(oof)))
            bal_acc = float(balanced_accuracy_score(y, _hard_labels(oof)))
            log_lines.append(f"caruana applied (CV: {pre_score:.5f} -> {cv_score:.5f})")
        else:
            log_lines.append("caruana skipped (did not improve)")
        caruana_info = {
            "kind": "caruana",
            "pre_score": pre_score,
            "caruana_score": caruana_score,
            "weights": ch["weights"],
            "applied": caruana_score > pre_score,
        }
        postprocess_info = (
            [postprocess_info, caruana_info] if postprocess_info else caruana_info
        )

    # Pseudo-labeling final fit (V45): after CV+postprocess produced reliable
    # test_preds, filter high-confidence test rows, concat to train, do a
    # SINGLE final fit per model on the augmented data, predict X_test, blend
    # using the same weights as the main fold loop. OOF stays from CV.
    pl_cfg = cfg.get("pseudo_label")
    if pl_cfg and oof.ndim == 2:
        threshold = float(pl_cfg.get("threshold", 0.85))
        weight = float(pl_cfg.get("weight", 0.5))
        max_probs = test_preds.max(axis=1)
        pseudo_labels = test_preds.argmax(axis=1)
        sel = max_probs > threshold
        n_pseudo = int(sel.sum())
        log_lines.append(
            f"=== Pseudo-label final fit: threshold={threshold} weight={weight} "
            f"selected {n_pseudo}/{len(X_test)} test rows ({100*n_pseudo/len(X_test):.1f}%) ==="
        )
        if n_pseudo == 0:
            log_lines.append("no pseudo rows selected; skipping final fit")
        else:
            # Build combined train+pseudo. Apply per-fold blocks ONCE on the
            # full data (acceptable for a final fit; CV scores are already in).
            X_orig_full = X.copy()
            y_orig_full = y.copy()
            if "_ote_y_shuffled" in X_orig_full.columns:
                # Already-augmented X — use the original y_tr per-row label that
                # was smuggled. But for final fit we want clean rows; recompute
                # per-fold blocks on raw X (no shuffle).
                pass

            # Pseudo rows from X_test
            X_pseudo = X_test.iloc[sel].copy().reset_index(drop=True)
            y_pseudo = pd.Series(pseudo_labels[sel]).reset_index(drop=True)

            # Run per-fold blocks once with full y as y_tr. CRITICAL: ordered_te
            # uses 4× shuffle augmentation by default which would expand X to
            # 2.5M rows — too memory-heavy when also adding pseudo. Override
            # to n_shuffles=1 for the final fit only.
            X_full_tr = X.copy()
            X_full_te = pd.concat([X_pseudo, X_test], axis=0, ignore_index=True)
            for name in per_fold_blocks:
                fn = feat_mod.BLOCKS[name]
                if name == "ordered_te":
                    X_full_tr, X_full_te = fn(X_full_tr, X_full_te, y_tr=y, n_shuffles=1)
                else:
                    X_full_tr, X_full_te = fn(X_full_tr, X_full_te, y_tr=y)
            if "_ote_y_shuffled" in X_full_tr.columns:
                y_full_tr = pd.Series(X_full_tr["_ote_y_shuffled"].values).reset_index(drop=True)
                X_full_tr = X_full_tr.drop(columns=["_ote_y_shuffled"])
            else:
                y_full_tr = y.copy()
            if "_ote_y_shuffled" in X_full_te.columns:
                X_full_te = X_full_te.drop(columns=["_ote_y_shuffled"])
            X_pseudo_te = X_full_te.iloc[: len(X_pseudo)].reset_index(drop=True)
            X_test_te = X_full_te.iloc[len(X_pseudo) :].reset_index(drop=True)

            X_combined = pd.concat([X_full_tr, X_pseudo_te], axis=0, ignore_index=True)
            y_combined = pd.concat([y_full_tr, y_pseudo], axis=0, ignore_index=True).reset_index(drop=True)

            sw_combined = models_mod.compute_balanced_sample_weights(y_combined) if class_weights_mode == "balanced" else None
            # Pseudo rows get reduced weight.
            n_real = len(X_full_tr)
            pseudo_mask = np.concatenate([np.zeros(n_real, dtype=bool), np.ones(len(X_pseudo_te), dtype=bool)])
            if sw_combined is None:
                sw_combined = np.ones(len(y_combined), dtype=float)
            sw_combined = sw_combined * np.where(pseudo_mask, weight, 1.0)
            log_lines.append(f"final fit rows: {len(X_combined)} (real {n_real} + pseudo {len(X_pseudo_te)})")

            # Fit each base model once on full augmented data, predict X_test.
            new_per_model_test = {m: np.zeros((len(X_test_te), n_classes)) for m in model_names}
            for m in model_names:
                # Use first row as a dummy "val" set (eval_set not used here).
                X_dummy_val = X_combined.iloc[:1].reset_index(drop=True)
                y_dummy_val = y_combined.iloc[:1].reset_index(drop=True)
                try:
                    res = models_mod.fit_one_fold(
                        m, X_combined, y_combined, X_dummy_val, y_dummy_val, X_test_te,
                        _params_for(m), task, sample_weight=sw_combined,
                    )
                    new_per_model_test[m] = res.test_pred
                    log_lines.append(f"pseudo final fit {m}: done")
                except Exception as e:
                    log_lines.append(f"pseudo final fit {m} FAILED: {e}; falling back to CV-averaged test preds")
                    new_per_model_test[m] = per_model_test[m]

            # Blend with same weights used in CV phase.
            if len(model_names) == 1:
                test_preds_pl = new_per_model_test[model_names[0]]
            else:
                test_preds_pl = sum(blend_weights[m] * new_per_model_test[m] for m in model_names)

            # Apply postprocess transforms to the new test_preds where applicable.
            if isinstance(postprocess_info, dict) and postprocess_info.get("kind") == "class_weight_optuna":
                w = np.array(postprocess_info["weights"])
                test_preds_pl = test_preds_pl * w
                test_preds_pl = test_preds_pl / test_preds_pl.sum(axis=1, keepdims=True)
            elif isinstance(postprocess_info, list):
                for info in postprocess_info:
                    if info.get("kind") == "class_weight_optuna" and info.get("applied"):
                        w = np.array(info["weights"])
                        test_preds_pl = test_preds_pl * w
                        test_preds_pl = test_preds_pl / test_preds_pl.sum(axis=1, keepdims=True)

            test_preds = test_preds_pl
            log_lines.append("pseudo final fit applied: test_preds replaced with augmented-fit predictions")

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
        "per_model_cv": per_model_cv,
        "blend_weights": blend_weights,
        "postprocess": postprocess_info,
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
