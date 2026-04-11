#!/usr/bin/env python3
"""Run V10 in steps — multi-seed averaging with fixed blend, no OOF tuning.

Usage:
    python run_v10_steps.py --step 1   # LightGBM seed=42 (~22 min)
    python run_v10_steps.py --step 2   # LightGBM seed=123 (~22 min)
    python run_v10_steps.py --step 3   # XGBoost seed=42 (~15 min)
    python run_v10_steps.py --step 4   # CatBoost seed=42 (~17 min)
    python run_v10_steps.py --step 5   # CatBoost seed=123 (~17 min)
    python run_v10_steps.py --step 6   # Multi-seed avg + fixed blend + save 3 submissions (~5 min)
"""

import sys, os, json, time, argparse, pickle
import numpy as np
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score
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
from pipeline.config import TARGET, ID_COL, CLASS_LABELS, CV_FOLDS, RANDOM_STATE

logger = setup_logging()

INTERMEDIATE_DIR = "outputs/v10_intermediate"
VERSION = 10


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
    """Load or create shared preprocessed data."""
    try:
        return _load_intermediate("shared_data")
    except FileNotFoundError:
        logger.info("Shared data not found, preprocessing...")
        train, test, _ = load_data()
        logger.info(f"  Train: {train.shape}, Test: {test.shape}")
        X_train, y_train, X_test, test_ids, label_enc, steps, te_metadata = _preprocess(train, test, VERSION)
        class_counts = np.bincount(y_train)
        total = len(y_train)
        class_weights = {i: total / (len(class_counts) * c) for i, c in enumerate(class_counts)}
        sample_weights = np.array([class_weights[y] for y in y_train])
        logger.info(f"  Features: {X_train.shape[1]}, Class weights computed")
        shared = {
            "X_train": X_train, "y_train": y_train, "X_test": X_test,
            "test_ids": test_ids, "label_enc": label_enc, "steps": steps,
            "te_metadata": te_metadata, "class_weights": class_weights,
            "sample_weights": sample_weights,
        }
        _save_intermediate("shared_data", shared)
        return shared


def _train_model(model_type, seed, step_name):
    """Train a single model with a specific seed."""
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

    result_name = f"{model_type}_seed{seed}"
    logger.info(f"Training {model_type} seed={seed} (5-fold)...")
    result = _cv_predict(model, X_train, y_train, X_test, result_name, sample_weights,
                         n_folds=CV_FOLDS, te_metadata=te_metadata)
    logger.info(f"  {result_name} mean BA: {result['mean_balanced_accuracy']:.5f}")

    _save_intermediate(result_name, result)

    elapsed = round(time.time() - t0, 1)
    logger.info(f"  {step_name} done in {elapsed}s ({elapsed/60:.1f} min)")


def step1_lgbm_seed42():
    _train_model("lightgbm", 42, "STEP 1: LightGBM seed=42")

def step2_lgbm_seed123():
    _train_model("lightgbm", 123, "STEP 2: LightGBM seed=123")

def step3_xgboost():
    _train_model("xgboost", 42, "STEP 3: XGBoost seed=42")

def step4_catboost_seed42():
    _train_model("catboost", 42, "STEP 4: CatBoost seed=42")

def step5_catboost_seed123():
    _train_model("catboost", 123, "STEP 5: CatBoost seed=123")


def step6_blend_and_finalize():
    """Multi-seed averaging + fixed score-proportional blend + save 3 submissions.
    NO grid search. NO calibration. NO OOF tuning."""
    t0 = time.time()
    logger.info("=== STEP 6: Multi-seed Avg + Fixed Blend + Finalize ===")

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

    # LGBM average (seed 42 + 123)
    lgbm_oof = (lgbm_s42["oof_probabilities"] + lgbm_s123["oof_probabilities"]) / 2
    lgbm_test = (lgbm_s42["test_probabilities"] + lgbm_s123["test_probabilities"]) / 2
    lgbm_ba = balanced_accuracy_score(y_train, lgbm_oof.argmax(axis=1))
    logger.info(f"  LGBM multi-seed avg BA: {lgbm_ba:.5f} (s42={lgbm_s42['mean_balanced_accuracy']:.5f}, s123={lgbm_s123['mean_balanced_accuracy']:.5f})")

    # XGBoost (single seed)
    xgb_oof = xgb_s42["oof_probabilities"]
    xgb_test = xgb_s42["test_probabilities"]
    xgb_ba = xgb_s42["mean_balanced_accuracy"]
    logger.info(f"  XGBoost BA: {xgb_ba:.5f}")

    # CatBoost average (seed 42 + 123)
    cat_oof = (cat_s42["oof_probabilities"] + cat_s123["oof_probabilities"]) / 2
    cat_test = (cat_s42["test_probabilities"] + cat_s123["test_probabilities"]) / 2
    cat_ba = balanced_accuracy_score(y_train, cat_oof.argmax(axis=1))
    logger.info(f"  CatBoost multi-seed avg BA: {cat_ba:.5f} (s42={cat_s42['mean_balanced_accuracy']:.5f}, s123={cat_s123['mean_balanced_accuracy']:.5f})")

    # Build results dict with averaged models
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

    # --- Fixed score-proportional blend (NO grid search) ---
    logger.info("Fixed score-proportional blend (no grid search)...")
    model_bas = {"lightgbm": lgbm_ba, "xgboost": xgb_ba, "catboost": cat_ba}
    total_ba = sum(model_bas.values())
    fixed_weights = {n: ba / total_ba for n, ba in model_bas.items()}
    logger.info(f"  Weights: {{{', '.join(f'{n}: {w:.4f}' for n, w in fixed_weights.items())}}}")

    blend_oof = np.zeros_like(lgbm_oof)
    blend_test = np.zeros_like(lgbm_test)
    for n, w in fixed_weights.items():
        blend_oof += results[n]["oof_probabilities"] * w
        blend_test += results[n]["test_probabilities"] * w

    blend_ba = balanced_accuracy_score(y_train, blend_oof.argmax(axis=1))
    logger.info(f"  Fixed blend BA: {blend_ba:.5f}")

    results["fixed_blend"] = {
        "fold_scores": [],
        "mean_balanced_accuracy": round(blend_ba, 5),
        "metrics": compute_metrics(y_train, blend_oof.argmax(axis=1), labels=[0, 1, 2]),
        "test_predictions": blend_test.argmax(axis=1),
        "test_probabilities": blend_test,
        "oof_probabilities": blend_oof,
        "feature_importance": lgbm_s42["feature_importance"],
        "params": {"method": "fixed_score_proportional_blend", "weights": fixed_weights},
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
        "params": {"method": "equal_blend", "weights": {"lightgbm": eq_w, "xgboost": eq_w, "catboost": eq_w}},
    }

    # --- Stacking ensemble (base models only) ---
    logger.info("Training stacking ensemble...")
    base_names = ("lightgbm", "xgboost", "catboost")
    oof_probs_all = {n: results[n]["oof_probabilities"] for n in base_names}
    stack_results = {n: results[n] for n in base_names}
    stack_result = _stacking_ensemble(stack_results, oof_probs_all, X_train, y_train, X_test, sample_weights, VERSION)
    results["stacking_ensemble"] = stack_result
    logger.info(f"  Stacking BA: {stack_result['mean_balanced_accuracy']:.5f}")

    # --- Weighted ensemble ---
    logger.info("Training weighted ensemble...")
    we_result = _weighted_ensemble(results, X_test, y_train=y_train)
    results["weighted_ensemble"] = we_result
    logger.info(f"  Weighted ensemble BA: {we_result['mean_balanced_accuracy']:.5f}")

    # --- Find best model ---
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
    logger.info(f"\nV10 Best: {best_model_name} = {best_score:.5f}")

    # --- Save 3 submission files ---
    logger.info("Saving 3 submission files...")
    # 1. Fixed blend (primary)
    fixed_preds = label_enc.inverse_transform(results["fixed_blend"]["test_predictions"])
    save_submission(test_ids.tolist(), fixed_preds.tolist(), filename="submission_v10_fixed.csv")
    logger.info("  Saved submission_v10_fixed.csv")

    # 2. Equal blend
    equal_preds = label_enc.inverse_transform(results["equal_blend"]["test_predictions"])
    save_submission(test_ids.tolist(), equal_preds.tolist(), filename="submission_v10_equal.csv")
    logger.info("  Saved submission_v10_equal.csv")

    # 3. LGBM multi-seed (safest single model type)
    lgbm_preds = label_enc.inverse_transform(results["lightgbm"]["test_predictions"])
    save_submission(test_ids.tolist(), lgbm_preds.tolist(), filename="submission_v10_lgbm.csv")
    logger.info("  Saved submission_v10_lgbm.csv")

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

    # Also save the best submission as submission_v10.csv
    save_submission(test_ids.tolist(), final_preds.tolist(), filename="submission_v10.csv")

    logger.info(f"\n{'='*70}")
    logger.info(f"  V{VERSION} RESULTS (Anti-overfit: no calibration, no grid-search blend)")
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
    logger.info(f"\n  Blend scores:")
    for name in ("fixed_blend", "equal_blend", "stacking_ensemble", "weighted_ensemble"):
        if name in results:
            logger.info(f"    {name:25s}: {results[name]['mean_balanced_accuracy']:.5f}")
    logger.info(f"\n  Submissions saved:")
    logger.info(f"    submission_v10_fixed.csv  (score-proportional blend)")
    logger.info(f"    submission_v10_equal.csv  (equal 1/3 weights)")
    logger.info(f"    submission_v10_lgbm.csv   (LGBM multi-seed avg)")
    logger.info(f"    submission_v10.csv        (best model: {best_model_name})")
    logger.info(f"\n  Step 6 done in {elapsed_total}s ({elapsed_total/60:.1f} min)")
    logger.info("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, required=True, choices=[1, 2, 3, 4, 5, 6])
    args = parser.parse_args()

    {
        1: step1_lgbm_seed42,
        2: step2_lgbm_seed123,
        3: step3_xgboost,
        4: step4_catboost_seed42,
        5: step5_catboost_seed123,
        6: step6_blend_and_finalize,
    }[args.step]()
