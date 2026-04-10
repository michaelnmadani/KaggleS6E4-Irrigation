#!/usr/bin/env python3
"""Run V9 in steps — each step saves intermediate results to avoid sandbox timeouts.

Usage:
    python run_v9_steps.py --step 1   # LightGBM only (~22 min)
    python run_v9_steps.py --step 2   # XGBoost only (~15 min)
    python run_v9_steps.py --step 3   # CatBoost only (~20 min)
    python run_v9_steps.py --step 4   # Blending + calibration + ensembles + save (~5 min)
"""

import sys, os, json, time, argparse, pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.data_loader import load_data, save_submission
from pipeline.utils import setup_logging, save_results_json, compute_metrics
from agents.researcher import run_eda
from agents.reviewer import review_results
from agents.iterative_builder import (
    _preprocess, _get_models, _cv_predict, _stacking_ensemble,
    _weighted_ensemble, VERSION_CONFIGS, _apply_per_fold_te,
)
from agents.orchestrator import _aggregate_results
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from pipeline.config import TARGET, ID_COL, CLASS_LABELS, CV_FOLDS, RANDOM_STATE

logger = setup_logging()

INTERMEDIATE_DIR = "outputs/v9_intermediate"
VERSION = 9


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


def step1_lightgbm():
    """Train LightGBM 5-fold CV and save results."""
    t0 = time.time()
    logger.info("=== STEP 1: LightGBM ===")

    train, test, _ = load_data()
    logger.info(f"  Train: {train.shape}, Test: {test.shape}")

    X_train, y_train, X_test, test_ids, label_enc, steps, te_metadata = _preprocess(train, test, VERSION)

    class_counts = np.bincount(y_train)
    total = len(y_train)
    class_weights = {i: total / (len(class_counts) * c) for i, c in enumerate(class_counts)}
    sample_weights = np.array([class_weights[y] for y in y_train])
    logger.info(f"  Features: {X_train.shape[1]}, Class weights computed")

    models = _get_models(VERSION, class_weights)
    model = models["lightgbm"]

    logger.info("Training lightgbm (5-fold)...")
    result = _cv_predict(model, X_train, y_train, X_test, "lightgbm", sample_weights,
                         n_folds=CV_FOLDS, te_metadata=te_metadata)
    logger.info(f"  LightGBM mean BA: {result['mean_balanced_accuracy']:.5f}")

    # Save everything needed for later steps
    _save_intermediate("shared_data", {
        "X_train": X_train, "y_train": y_train, "X_test": X_test,
        "test_ids": test_ids, "label_enc": label_enc, "steps": steps,
        "te_metadata": te_metadata, "class_weights": class_weights,
        "sample_weights": sample_weights,
    })
    _save_intermediate("lightgbm_result", result)

    elapsed = round(time.time() - t0, 1)
    logger.info(f"  Step 1 done in {elapsed}s ({elapsed/60:.1f} min)")
    logger.info(f"  LightGBM: {result['mean_balanced_accuracy']:.5f}")


def step2_xgboost():
    """Train XGBoost 5-fold CV and save results."""
    t0 = time.time()
    logger.info("=== STEP 2: XGBoost ===")

    shared = _load_intermediate("shared_data")
    X_train = shared["X_train"]
    y_train = shared["y_train"]
    X_test = shared["X_test"]
    sample_weights = shared["sample_weights"]
    te_metadata = shared["te_metadata"]
    class_weights = shared["class_weights"]

    models = _get_models(VERSION, class_weights)
    model = models["xgboost"]

    logger.info("Training xgboost (5-fold)...")
    result = _cv_predict(model, X_train, y_train, X_test, "xgboost", sample_weights,
                         n_folds=CV_FOLDS, te_metadata=te_metadata)
    logger.info(f"  XGBoost mean BA: {result['mean_balanced_accuracy']:.5f}")

    _save_intermediate("xgboost_result", result)

    elapsed = round(time.time() - t0, 1)
    logger.info(f"  Step 2 done in {elapsed}s ({elapsed/60:.1f} min)")


def step3_catboost():
    """Train CatBoost 5-fold CV and save results."""
    t0 = time.time()
    logger.info("=== STEP 3: CatBoost ===")

    shared = _load_intermediate("shared_data")
    X_train = shared["X_train"]
    y_train = shared["y_train"]
    X_test = shared["X_test"]
    sample_weights = shared["sample_weights"]
    te_metadata = shared["te_metadata"]
    class_weights = shared["class_weights"]

    models = _get_models(VERSION, class_weights)
    model = models["catboost"]

    logger.info("Training catboost (5-fold)...")
    result = _cv_predict(model, X_train, y_train, X_test, "catboost", sample_weights,
                         n_folds=CV_FOLDS, te_metadata=te_metadata)
    logger.info(f"  CatBoost mean BA: {result['mean_balanced_accuracy']:.5f}")

    _save_intermediate("catboost_result", result)

    elapsed = round(time.time() - t0, 1)
    logger.info(f"  Step 3 done in {elapsed}s ({elapsed/60:.1f} min)")


def step4_blend_and_finalize():
    """Blending, calibration, ensembles, review, and save final results."""
    t0 = time.time()
    logger.info("=== STEP 4: Blending + Calibration + Ensembles ===")

    shared = _load_intermediate("shared_data")
    X_train = shared["X_train"]
    y_train = shared["y_train"]
    X_test = shared["X_test"]
    test_ids = shared["test_ids"]
    label_enc = shared["label_enc"]
    steps = shared["steps"]
    sample_weights = shared["sample_weights"]

    lgbm_result = _load_intermediate("lightgbm_result")
    xgb_result = _load_intermediate("xgboost_result")
    cat_result = _load_intermediate("catboost_result")

    results = {
        "lightgbm": lgbm_result,
        "xgboost": xgb_result,
        "catboost": cat_result,
    }
    oof_probs_all = {n: r["oof_probabilities"] for n, r in results.items()}

    best_score = -1
    best_model_name = None
    best_test_preds = None
    for name, r in results.items():
        if r["mean_balanced_accuracy"] > best_score:
            best_score = r["mean_balanced_accuracy"]
            best_model_name = name
            best_test_preds = r["test_predictions"]

    # Optimized blending (finer grid for V9)
    logger.info("Optimized blending via grid search on OOF...")
    from itertools import product as iproduct
    blend_models = {n: r for n, r in results.items() if r["mean_balanced_accuracy"] > 0.90}
    blend_names = list(blend_models.keys())
    best_blend_ba = -1
    best_blend_weights = None
    n_models = len(blend_names)
    steps_grid = [round(x * 0.05, 2) for x in range(21)]
    for combo in iproduct(steps_grid, repeat=n_models):
        if abs(sum(combo) - 1.0) > 0.01:
            continue
        if any(w == 0 for w in combo):
            continue
        blended_oof = np.zeros_like(results[blend_names[0]]["oof_probabilities"])
        for i, n in enumerate(blend_names):
            blended_oof += results[n]["oof_probabilities"] * combo[i]
        blend_preds = blended_oof.argmax(axis=1)
        blend_ba = balanced_accuracy_score(y_train, blend_preds)
        if blend_ba > best_blend_ba:
            best_blend_ba = blend_ba
            best_blend_weights = dict(zip(blend_names, combo))

    if best_blend_weights:
        logger.info(f"  Best blend: BA = {best_blend_ba:.5f}, weights = {best_blend_weights}")
        blend_test = np.zeros_like(results[blend_names[0]]["test_probabilities"])
        blend_oof_final = np.zeros_like(results[blend_names[0]]["oof_probabilities"])
        for n, w in best_blend_weights.items():
            blend_test += results[n]["test_probabilities"] * w
            blend_oof_final += results[n]["oof_probabilities"] * w
        results["optimized_blend"] = {
            "fold_scores": [],
            "mean_balanced_accuracy": round(best_blend_ba, 5),
            "metrics": compute_metrics(y_train, blend_oof_final.argmax(axis=1), labels=[0, 1, 2]),
            "test_predictions": blend_test.argmax(axis=1),
            "test_probabilities": blend_test,
            "oof_probabilities": blend_oof_final,
            "feature_importance": results[blend_names[0]]["feature_importance"],
            "params": {"method": "optimized_blend", "weights": best_blend_weights},
        }
        if best_blend_ba > best_score:
            best_score = best_blend_ba
            best_model_name = "optimized_blend"
            best_test_preds = blend_test.argmax(axis=1)

    # V9: Post-blend probability calibration for HIGH class
    if "optimized_blend" in results:
        logger.info("V9: Post-blend HIGH class probability calibration...")
        cal_oof = results["optimized_blend"]["oof_probabilities"].copy()
        cal_test = results["optimized_blend"]["test_probabilities"].copy()
        best_cal_ba = results["optimized_blend"]["mean_balanced_accuracy"]
        best_cal_factor = 1.0
        for factor in [round(0.8 + i * 0.02, 2) for i in range(21)]:
            scaled_oof = cal_oof.copy()
            scaled_oof[:, 0] *= factor
            row_sums = scaled_oof.sum(axis=1, keepdims=True)
            scaled_oof = scaled_oof / row_sums
            cal_preds = scaled_oof.argmax(axis=1)
            cal_ba = balanced_accuracy_score(y_train, cal_preds)
            if cal_ba > best_cal_ba:
                best_cal_ba = cal_ba
                best_cal_factor = factor
        if best_cal_factor != 1.0:
            final_oof = cal_oof.copy()
            final_oof[:, 0] *= best_cal_factor
            final_oof = final_oof / final_oof.sum(axis=1, keepdims=True)
            final_test = cal_test.copy()
            final_test[:, 0] *= best_cal_factor
            final_test = final_test / final_test.sum(axis=1, keepdims=True)
            logger.info(f"  Calibrated: BA = {best_cal_ba:.5f}, HIGH scale = {best_cal_factor}")
            results["calibrated_blend"] = {
                "fold_scores": [],
                "mean_balanced_accuracy": round(best_cal_ba, 5),
                "metrics": compute_metrics(y_train, final_oof.argmax(axis=1), labels=[0, 1, 2]),
                "test_predictions": final_test.argmax(axis=1),
                "test_probabilities": final_test,
                "oof_probabilities": final_oof,
                "feature_importance": results["optimized_blend"]["feature_importance"],
                "params": {"method": "calibrated_blend", "high_class_scale": best_cal_factor,
                           "base_blend_weights": results["optimized_blend"]["params"].get("weights", {})},
            }
            if best_cal_ba > best_score:
                best_score = best_cal_ba
                best_model_name = "calibrated_blend"
                best_test_preds = final_test.argmax(axis=1)
        else:
            logger.info("  No calibration improvement found (factor=1.0 is optimal)")

    # Stacking ensemble
    logger.info("Training stacking ensemble...")
    stack_result = _stacking_ensemble(results, oof_probs_all, X_train, y_train, X_test, sample_weights, VERSION)
    results["stacking_ensemble"] = stack_result
    if stack_result["mean_balanced_accuracy"] > best_score:
        best_score = stack_result["mean_balanced_accuracy"]
        best_model_name = "stacking_ensemble"
        best_test_preds = stack_result["test_predictions"]

    # Weighted ensemble
    logger.info("Training weighted ensemble...")
    we_result = _weighted_ensemble(results, X_test, y_train=y_train)
    results["weighted_ensemble"] = we_result
    if we_result["mean_balanced_accuracy"] > best_score:
        best_score = we_result["mean_balanced_accuracy"]
        best_model_name = "weighted_ensemble"
        best_test_preds = we_result["test_predictions"]

    best_score = round(best_score, 5)
    final_preds = label_enc.inverse_transform(best_test_preds)
    logger.info(f"V9 Best: {best_model_name} = {best_score:.5f}")

    # Build final output (same as run_v9_only.py)
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
        logger.info(f"  ✓ {a}")
    for w in review["warnings"]:
        logger.info(f"  ⚠ {w}")

    elapsed_total = round(time.time() - t0, 1)
    dashboard = _aggregate_results(eda, final_results, review, elapsed_total)

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
    save_submission(test_ids.tolist(), final_preds.tolist(), filename="submission_v9.csv")

    logger.info(f"\n{'='*70}")
    logger.info(f"  V{VERSION} RESULTS")
    logger.info(f"{'='*70}")
    logger.info(f"  Previous: {prev['best_model']:25s} BA = {old_score:.5f}")
    logger.info(f"  V{VERSION}:      {best_model_name:25s} BA = {new_score:.5f}")
    logger.info(f"  Delta:    {'↑' if delta > 0 else '↓'} {abs(delta):.5f}")
    logger.info(f"\n  Model breakdown:")
    for name, r in results.items():
        logger.info(f"    {name:25s}: {r['mean_balanced_accuracy']:.5f}")
    logger.info(f"\n  Step 4 done in {elapsed_total}s ({elapsed_total/60:.1f} min)")
    logger.info("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, required=True, choices=[1, 2, 3, 4])
    args = parser.parse_args()

    {1: step1_lightgbm, 2: step2_xgboost, 3: step3_catboost, 4: step4_blend_and_finalize}[args.step]()
