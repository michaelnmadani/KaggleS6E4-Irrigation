#!/usr/bin/env python3
"""Run V11 in steps — each step runs 5 folds to stay within sandbox timeout.

Usage (each step ~40 min):
    python run_v11_steps.py --step 1a   # LightGBM seed=42, folds 0-4
    python run_v11_steps.py --step 1b   # LightGBM seed=42, folds 5-9 + merge
    python run_v11_steps.py --step 2a   # LightGBM seed=123, folds 0-4
    python run_v11_steps.py --step 2b   # LightGBM seed=123, folds 5-9 + merge
    python run_v11_steps.py --step 3a   # XGBoost seed=42, folds 0-4
    python run_v11_steps.py --step 3b   # XGBoost seed=42, folds 5-9 + merge
    python run_v11_steps.py --step 4a   # CatBoost seed=42, folds 0-4
    python run_v11_steps.py --step 4b   # CatBoost seed=42, folds 5-9 + merge
    python run_v11_steps.py --step 5a   # CatBoost seed=123, folds 0-4
    python run_v11_steps.py --step 5b   # CatBoost seed=123, folds 5-9 + merge
    python run_v11_steps.py --step 6    # Multi-seed avg + blend + logit bias + save submissions
"""

import sys, os, json, time, argparse, pickle
import numpy as np
from scipy.special import softmax, log_softmax
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.data_loader import load_data, save_submission
from pipeline.utils import setup_logging, save_results_json, compute_metrics
from agents.researcher import run_eda
from agents.reviewer import review_results
from agents.iterative_builder import (
    _preprocess, _get_models, _cv_predict, _stacking_ensemble,
    _weighted_ensemble, VERSION_CONFIGS,
)
from agents.orchestrator import _aggregate_results
from pipeline.config import TARGET, ID_COL, CLASS_LABELS, CV_FOLDS_V11, RANDOM_STATE

logger = setup_logging()

INTERMEDIATE_DIR = "outputs/v11_intermediate"
VERSION = 11
N_FOLDS = CV_FOLDS_V11  # 10-fold


def _save_intermediate(name, data):
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    path = os.path.join(INTERMEDIATE_DIR, f"{name}.pkl")
    with open(path, "wb") as f:
        pickle.dump(data, f)
    logger.info(f"  Saved intermediate: {path}")


def _load_intermediate(name):
    path = os.path.join(INTERMEDIATE_DIR, f"{name}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def _get_shared_data():
    """Load or create shared preprocessed data with original data appended."""
    try:
        return _load_intermediate("shared_data")
    except FileNotFoundError:
        logger.info("Preprocessing with original data appended...")
        train, test, _ = load_data(append_original=True, original_weight=0.35)
        logger.info(f"  Train: {train.shape}, Test: {test.shape}")

        # Track which rows are original for sample weighting
        is_original = train.get("_is_original", pd.Series(0, index=train.index)).values
        train = train.drop(columns=["_is_original"], errors="ignore")

        X_train, y_train, X_test, test_ids, label_enc, steps, te_metadata = _preprocess(train, test, VERSION)

        # Compute sample weights with original data discount
        class_counts = np.bincount(y_train)
        total = len(y_train)
        class_weights = {i: total / (len(class_counts) * c) for i, c in enumerate(class_counts)}
        sample_weights = np.array([class_weights[y] for y in y_train])
        # Discount original data rows
        orig_weight = 0.35
        sample_weights = sample_weights * np.where(is_original, orig_weight, 1.0)
        logger.info(f"  Features: {X_train.shape[1]}, Original rows: {is_original.sum()} (weight={orig_weight})")

        shared = {
            "X_train": X_train, "y_train": y_train, "X_test": X_test,
            "test_ids": test_ids, "label_enc": label_enc, "steps": steps,
            "te_metadata": te_metadata, "class_weights": class_weights,
            "sample_weights": sample_weights, "is_original": is_original,
        }
        _save_intermediate("shared_data", shared)
        return shared


import pandas as pd  # needed for _get_shared_data


def _train_model_partial(model_type, seed, fold_start, fold_end, step_name):
    """Train a model on a subset of folds (to stay within sandbox timeout)."""
    t0 = time.time()
    logger.info(f"=== {step_name} ===")

    shared = _get_shared_data()
    X_train = shared["X_train"]
    y_train = shared["y_train"]
    X_test = shared["X_test"]
    sample_weights = shared["sample_weights"]
    te_metadata = shared["te_metadata"]
    class_weights = shared["class_weights"]

    models = _get_models(VERSION, class_weights)
    model = models[model_type]

    # Set the seed
    if hasattr(model, "random_state"):
        model.set_params(random_state=seed)
    elif hasattr(model, "random_seed"):
        model.set_params(random_seed=seed)

    part_name = f"{model_type}_seed{seed}_f{fold_start}_{fold_end}"
    logger.info(f"Training {model_type} seed={seed} (folds {fold_start}-{fold_end-1} of {N_FOLDS})...")
    result = _cv_predict(model, X_train, y_train, X_test, part_name, sample_weights,
                         n_folds=N_FOLDS, te_metadata=te_metadata,
                         fold_range=(fold_start, fold_end))

    _save_intermediate(part_name, result)

    elapsed = round(time.time() - t0, 1)
    logger.info(f"  {step_name} done in {elapsed}s ({elapsed/60:.1f} min)")
    for i, s in enumerate(result["fold_scores"]):
        logger.info(f"    Fold {fold_start + i}: BA = {s:.5f}")


def _merge_partial_results(model_type, seed):
    """Merge two partial fold results into one complete result."""
    part_a = _load_intermediate(f"{model_type}_seed{seed}_f0_5")
    part_b = _load_intermediate(f"{model_type}_seed{seed}_f5_10")

    shared = _load_intermediate("shared_data")
    y_train = shared["y_train"]

    # Merge OOF: each part has zeros where it didn't predict — just add them
    merged_oof = part_a["oof_probabilities"] + part_b["oof_probabilities"]
    # Merge test probs: each part contributed folds/n_folds of the prediction — just add
    merged_test = part_a["test_probabilities"] + part_b["test_probabilities"]

    # Merge fold scores
    fold_scores = part_a["fold_scores"] + part_b["fold_scores"]

    # Compute overall BA
    oof_preds = merged_oof.argmax(axis=1)
    mean_ba = balanced_accuracy_score(y_train, oof_preds)

    # Average feature importance
    fi_merged = {}
    for feat in part_a["feature_importance"]:
        fi_merged[feat] = (part_a["feature_importance"].get(feat, 0) + part_b["feature_importance"].get(feat, 0)) / 2
    fi_merged = dict(sorted(fi_merged.items(), key=lambda x: x[1], reverse=True))

    result_name = f"{model_type}_seed{seed}"
    merged = {
        "fold_scores": fold_scores,
        "mean_balanced_accuracy": round(mean_ba, 5),
        "metrics": compute_metrics(y_train, oof_preds, labels=[0, 1, 2]),
        "test_predictions": merged_test.argmax(axis=1),
        "test_probabilities": merged_test,
        "oof_probabilities": merged_oof,
        "feature_importance": fi_merged,
        "params": part_a["params"],
    }
    _save_intermediate(result_name, merged)
    logger.info(f"  Merged {result_name}: BA = {mean_ba:.5f} ({len(fold_scores)} folds)")
    return merged


# Step functions: each model split into folds 0-4 (a) and folds 5-9 (b)
def step1a(): _train_model_partial("lightgbm", 42, 0, 5, "STEP 1a: LightGBM seed=42 folds 0-4")
def step1b():
    _train_model_partial("lightgbm", 42, 5, 10, "STEP 1b: LightGBM seed=42 folds 5-9")
    _merge_partial_results("lightgbm", 42)

def step2a(): _train_model_partial("lightgbm", 123, 0, 5, "STEP 2a: LightGBM seed=123 folds 0-4")
def step2b():
    _train_model_partial("lightgbm", 123, 5, 10, "STEP 2b: LightGBM seed=123 folds 5-9")
    _merge_partial_results("lightgbm", 123)

def step3a(): _train_model_partial("xgboost", 42, 0, 5, "STEP 3a: XGBoost seed=42 folds 0-4")
def step3b():
    _train_model_partial("xgboost", 42, 5, 10, "STEP 3b: XGBoost seed=42 folds 5-9")
    _merge_partial_results("xgboost", 42)

def step4a(): _train_model_partial("catboost", 42, 0, 5, "STEP 4a: CatBoost seed=42 folds 0-4")
def step4b():
    _train_model_partial("catboost", 42, 5, 10, "STEP 4b: CatBoost seed=42 folds 5-9")
    _merge_partial_results("catboost", 42)

def step5a(): _train_model_partial("catboost", 123, 0, 5, "STEP 5a: CatBoost seed=123 folds 0-4")
def step5b():
    _train_model_partial("catboost", 123, 5, 10, "STEP 5b: CatBoost seed=123 folds 5-9")
    _merge_partial_results("catboost", 123)


def _logit_bias_tuning(oof_probs, y_train, test_probs):
    """Optimize per-class logit-space biases for balanced accuracy using nested CV.

    Instead of tuning on full OOF (which overfits), use 5-fold nested CV:
    tune biases on 4/5 of OOF, evaluate on 1/5, average the found biases.
    """
    logger.info("  Logit bias tuning (nested 5-fold CV)...")
    n_classes = oof_probs.shape[1]

    # Convert to log-probabilities (logit space)
    eps = 1e-10
    oof_logits = np.log(np.clip(oof_probs, eps, 1.0))
    test_logits = np.log(np.clip(test_probs, eps, 1.0))

    best_biases_list = []
    nested_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE + 999)

    for fold_idx, (tune_idx, eval_idx) in enumerate(nested_skf.split(oof_logits, y_train)):
        tune_logits = oof_logits[tune_idx]
        tune_y = y_train[tune_idx]
        eval_logits = oof_logits[eval_idx]
        eval_y = y_train[eval_idx]

        best_ba = -1
        best_bias = np.zeros(n_classes)

        # Grid search in logit space
        for b0 in np.arange(-2.0, 2.5, 0.25):  # HIGH class bias
            for b1 in np.arange(-1.5, 1.0, 0.25):  # LOW class bias
                for b2 in np.arange(-1.5, 1.0, 0.25):  # MEDIUM class bias
                    bias = np.array([b0, b1, b2])
                    adjusted = tune_logits + bias
                    preds = adjusted.argmax(axis=1)
                    ba = balanced_accuracy_score(tune_y, preds)
                    if ba > best_ba:
                        best_ba = ba
                        best_bias = bias.copy()

        # Validate on held-out fold
        eval_adjusted = eval_logits + best_bias
        eval_preds = eval_adjusted.argmax(axis=1)
        eval_ba = balanced_accuracy_score(eval_y, eval_preds)
        logger.info(f"    Nested fold {fold_idx+1}: tune BA={best_ba:.5f}, eval BA={eval_ba:.5f}, bias={best_bias.tolist()}")
        best_biases_list.append(best_bias)

    # Average biases across nested folds
    avg_bias = np.mean(best_biases_list, axis=0)
    logger.info(f"  Average logit bias: {avg_bias.tolist()}")

    # Apply averaged bias to full OOF and test
    adjusted_oof = oof_logits + avg_bias
    adjusted_test = test_logits + avg_bias

    # Convert back to probabilities via softmax
    adjusted_oof_probs = softmax(adjusted_oof, axis=1)
    adjusted_test_probs = softmax(adjusted_test, axis=1)

    final_ba = balanced_accuracy_score(y_train, adjusted_oof_probs.argmax(axis=1))
    logger.info(f"  Logit bias result: BA={final_ba:.5f}")

    return adjusted_oof_probs, adjusted_test_probs, avg_bias.tolist(), final_ba


def step6_blend_and_finalize():
    """Multi-seed avg + CatBoost-dominant blend + logit bias tuning + save submissions."""
    t0 = time.time()
    logger.info("=== STEP 6: Multi-seed Avg + Blend + Logit Bias + Finalize ===")

    shared = _load_intermediate("shared_data")
    X_train = shared["X_train"]
    y_train = shared["y_train"]
    X_test = shared["X_test"]
    test_ids = shared["test_ids"]
    label_enc = shared["label_enc"]
    steps = shared["steps"]
    sample_weights = shared["sample_weights"]

    # Load all individual model results
    lgbm_s42 = _load_intermediate("lightgbm_seed42")
    lgbm_s123 = _load_intermediate("lightgbm_seed123")
    xgb_s42 = _load_intermediate("xgboost_seed42")
    cat_s42 = _load_intermediate("catboost_seed42")
    cat_s123 = _load_intermediate("catboost_seed123")

    # --- Multi-seed averaging within model types ---
    logger.info("Multi-seed averaging...")

    # LGBM average
    lgbm_oof = (lgbm_s42["oof_probabilities"] + lgbm_s123["oof_probabilities"]) / 2
    lgbm_test = (lgbm_s42["test_probabilities"] + lgbm_s123["test_probabilities"]) / 2
    lgbm_ba = balanced_accuracy_score(y_train, lgbm_oof.argmax(axis=1))
    logger.info(f"  LGBM avg BA: {lgbm_ba:.5f} (s42={lgbm_s42['mean_balanced_accuracy']:.5f}, s123={lgbm_s123['mean_balanced_accuracy']:.5f})")

    # XGBoost (single seed)
    xgb_oof = xgb_s42["oof_probabilities"]
    xgb_test = xgb_s42["test_probabilities"]
    xgb_ba = xgb_s42["mean_balanced_accuracy"]
    logger.info(f"  XGBoost BA: {xgb_ba:.5f}")

    # CatBoost average
    cat_oof = (cat_s42["oof_probabilities"] + cat_s123["oof_probabilities"]) / 2
    cat_test = (cat_s42["test_probabilities"] + cat_s123["test_probabilities"]) / 2
    cat_ba = balanced_accuracy_score(y_train, cat_oof.argmax(axis=1))
    logger.info(f"  CatBoost avg BA: {cat_ba:.5f} (s42={cat_s42['mean_balanced_accuracy']:.5f}, s123={cat_s123['mean_balanced_accuracy']:.5f})")

    # Build results dict
    results = {
        "lightgbm": {
            "fold_scores": lgbm_s42["fold_scores"],
            "mean_balanced_accuracy": round(lgbm_ba, 5),
            "metrics": compute_metrics(y_train, lgbm_oof.argmax(axis=1), labels=[0, 1, 2]),
            "test_predictions": lgbm_test.argmax(axis=1),
            "test_probabilities": lgbm_test,
            "oof_probabilities": lgbm_oof,
            "feature_importance": lgbm_s42["feature_importance"],
            "params": {"method": "multi_seed_avg", "seeds": [42, 123]},
        },
        "xgboost": xgb_s42,
        "catboost": {
            "fold_scores": cat_s42["fold_scores"],
            "mean_balanced_accuracy": round(cat_ba, 5),
            "metrics": compute_metrics(y_train, cat_oof.argmax(axis=1), labels=[0, 1, 2]),
            "test_predictions": cat_test.argmax(axis=1),
            "test_probabilities": cat_test,
            "oof_probabilities": cat_oof,
            "feature_importance": cat_s42["feature_importance"],
            "params": {"method": "multi_seed_avg", "seeds": [42, 123]},
        },
    }

    # --- CatBoost-dominant blend (research shows CatBoost=0.65 is optimal) ---
    logger.info("CatBoost-dominant blend...")
    cb_dom_weights = {"catboost": 0.65, "lightgbm": 0.25, "xgboost": 0.10}
    cb_oof = np.zeros_like(lgbm_oof)
    cb_test_blend = np.zeros_like(lgbm_test)
    for n, w in cb_dom_weights.items():
        cb_oof += results[n]["oof_probabilities"] * w
        cb_test_blend += results[n]["test_probabilities"] * w
    cb_ba = balanced_accuracy_score(y_train, cb_oof.argmax(axis=1))
    logger.info(f"  CatBoost-dominant blend BA: {cb_ba:.5f}")

    results["catboost_dominant_blend"] = {
        "fold_scores": [],
        "mean_balanced_accuracy": round(cb_ba, 5),
        "metrics": compute_metrics(y_train, cb_oof.argmax(axis=1), labels=[0, 1, 2]),
        "test_predictions": cb_test_blend.argmax(axis=1),
        "test_probabilities": cb_test_blend,
        "oof_probabilities": cb_oof,
        "feature_importance": cat_s42["feature_importance"],
        "params": {"method": "catboost_dominant_blend", "weights": cb_dom_weights},
    }

    # --- Equal-weight blend ---
    logger.info("Equal-weight blend...")
    eq_w = 1.0 / 3
    eq_oof = lgbm_oof * eq_w + xgb_oof * eq_w + cat_oof * eq_w
    eq_test = lgbm_test * eq_w + xgb_test * eq_w + cat_test * eq_w
    eq_ba = balanced_accuracy_score(y_train, eq_oof.argmax(axis=1))
    logger.info(f"  Equal blend BA: {eq_ba:.5f}")

    results["equal_blend"] = {
        "fold_scores": [],
        "mean_balanced_accuracy": round(eq_ba, 5),
        "metrics": compute_metrics(y_train, eq_oof.argmax(axis=1), labels=[0, 1, 2]),
        "test_predictions": eq_test.argmax(axis=1),
        "test_probabilities": eq_test,
        "oof_probabilities": eq_oof,
        "feature_importance": lgbm_s42["feature_importance"],
        "params": {"method": "equal_blend"},
    }

    # --- Logit-space bias tuning on the best blend ---
    # Pick the best blend/model so far as the base for bias tuning
    best_blend_name = max(results, key=lambda n: results[n]["mean_balanced_accuracy"])
    best_blend_oof = results[best_blend_name]["oof_probabilities"]
    best_blend_test = results[best_blend_name]["test_probabilities"]
    logger.info(f"Applying logit bias tuning to '{best_blend_name}' (BA={results[best_blend_name]['mean_balanced_accuracy']:.5f})...")

    bias_oof, bias_test, bias_values, bias_ba = _logit_bias_tuning(
        best_blend_oof, y_train, best_blend_test
    )

    if bias_ba > results[best_blend_name]["mean_balanced_accuracy"]:
        results["logit_bias_tuned"] = {
            "fold_scores": [],
            "mean_balanced_accuracy": round(bias_ba, 5),
            "metrics": compute_metrics(y_train, bias_oof.argmax(axis=1), labels=[0, 1, 2]),
            "test_predictions": bias_test.argmax(axis=1),
            "test_probabilities": bias_test,
            "oof_probabilities": bias_oof,
            "feature_importance": results[best_blend_name]["feature_importance"],
            "params": {"method": "logit_bias_tuned", "base": best_blend_name, "bias": bias_values},
        }
        logger.info(f"  Logit bias improved: {results[best_blend_name]['mean_balanced_accuracy']:.5f} -> {bias_ba:.5f}")
    else:
        logger.info(f"  Logit bias did not improve (BA={bias_ba:.5f}), skipping")

    # --- Find best overall ---
    best_score = -1
    best_model_name = None
    best_test_preds = None
    for name, r in results.items():
        if r["mean_balanced_accuracy"] > best_score:
            best_score = r["mean_balanced_accuracy"]
            best_model_name = name
            best_test_preds = r["test_predictions"]

    best_score = round(best_score, 5)
    final_preds = label_enc.inverse_transform(best_test_preds)
    logger.info(f"\nV11 Best: {best_model_name} = {best_score:.5f}")

    # --- Save submission files ---
    logger.info("Saving submission files...")

    submissions = {
        "submission_v11_catboost_dom.csv": "catboost_dominant_blend",
        "submission_v11_equal.csv": "equal_blend",
        "submission_v11_catboost.csv": "catboost",
        "submission_v11_lgbm.csv": "lightgbm",
    }
    if "logit_bias_tuned" in results:
        submissions["submission_v11_bias.csv"] = "logit_bias_tuned"

    for fname, model_name in submissions.items():
        if model_name in results:
            preds = label_enc.inverse_transform(results[model_name]["test_predictions"])
            save_submission(test_ids.tolist(), preds.tolist(), filename=fname)
            logger.info(f"  Saved {fname} ({model_name}: BA={results[model_name]['mean_balanced_accuracy']:.5f})")

    # --- Build final output ---
    config = VERSION_CONFIGS[VERSION]
    train_df, test_df, _ = load_data()
    eda = run_eda(train_df, test_df)

    final_results = {
        "models": results,
        "best_model": best_model_name,
        "best_score": best_score,
        "test_ids": test_ids.tolist(),
        "predictions": final_preds.tolist(),
        "pipeline_steps": steps,
        "version": VERSION,
        "version_config": config,
    }

    review = review_results(train_df, test_df, final_results, eda)
    logger.info(f"Verdict: {review['verdict']}")
    for a in review["approvals"]:
        logger.info(f"  + {a}")
    for w in review["warnings"]:
        logger.info(f"  ! {w}")

    elapsed_total = round(time.time() - t0, 1)
    dashboard = _aggregate_results(eda, final_results, review, elapsed_total)

    # Load previous results for comparison
    with open("outputs/results.json") as f:
        prev = json.load(f)
    old_score = prev["best_score"]
    new_score = best_score
    delta = new_score - old_score

    prev_history = prev.get("version_history", [])
    prev_history.append({
        "version": VERSION,
        "score": new_score,
        "best_model": best_model_name,
        "changes": config["changes"],
    })
    dashboard["version_history"] = prev_history
    dashboard["improvement"] = {
        "previous_best_model": prev["best_model"],
        "previous_best_score": old_score,
        "improved_best_model": best_model_name,
        "improved_best_score": new_score,
        "score_change": round(delta, 5),
        "recommendations_applied": config["changes"],
    }

    save_results_json(dashboard)
    save_submission(test_ids.tolist(), final_preds.tolist(), filename="submission_v11.csv")

    logger.info(f"\n{'='*70}")
    logger.info(f"  V{VERSION} RESULTS (Research-driven: original data + pairwise TE + logit bias)")
    logger.info(f"{'='*70}")
    logger.info(f"  Previous:  {prev['best_model']:25s} BA = {old_score:.5f}")
    logger.info(f"  V{VERSION}:       {best_model_name:25s} BA = {new_score:.5f}")
    logger.info(f"  Delta:     {'UP' if delta > 0 else 'DOWN'} {abs(delta):.5f}")
    logger.info(f"\n  Individual model scores:")
    logger.info(f"    LGBM seed42:   {lgbm_s42['mean_balanced_accuracy']:.5f}")
    logger.info(f"    LGBM seed123:  {lgbm_s123['mean_balanced_accuracy']:.5f}")
    logger.info(f"    LGBM avg:      {lgbm_ba:.5f}")
    logger.info(f"    XGBoost:       {xgb_ba:.5f}")
    logger.info(f"    CatBoost s42:  {cat_s42['mean_balanced_accuracy']:.5f}")
    logger.info(f"    CatBoost s123: {cat_s123['mean_balanced_accuracy']:.5f}")
    logger.info(f"    CatBoost avg:  {cat_ba:.5f}")
    logger.info(f"\n  Blend/ensemble scores:")
    for name in sorted(results.keys()):
        if name not in ("lightgbm", "xgboost", "catboost"):
            logger.info(f"    {name:30s}: {results[name]['mean_balanced_accuracy']:.5f}")
    logger.info(f"\n  Submissions saved:")
    for fname in submissions:
        logger.info(f"    {fname}")
    logger.info(f"    submission_v11.csv (best: {best_model_name})")
    logger.info(f"\n  Step 6 done in {elapsed_total}s ({elapsed_total/60:.1f} min)")
    logger.info("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, required=True,
                        choices=["1a", "1b", "2a", "2b", "3a", "3b", "4a", "4b", "5a", "5b", "6"])
    args = parser.parse_args()

    {
        "1a": step1a, "1b": step1b,
        "2a": step2a, "2b": step2b,
        "3a": step3a, "3b": step3b,
        "4a": step4a, "4b": step4b,
        "5a": step5a, "5b": step5b,
        "6": step6_blend_and_finalize,
    }[args.step]()
