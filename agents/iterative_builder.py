"""Iterative Builder Agent: Configurable builder that applies version-specific improvements."""

import math
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
from lightgbm import LGBMClassifier, early_stopping as lgb_early_stopping, log_evaluation as lgb_log_eval
from sklearn.metrics import balanced_accuracy_score
from sklearn.base import clone

from pipeline.config import (
    TARGET, ID_COL, CLASS_LABELS, CATEGORICAL_FEATURES, NUMERIC_FEATURES,
    CV_FOLDS, RANDOM_STATE
)
from pipeline.utils import setup_logging, compute_metrics

logger = setup_logging()


# ═══════════════════════════════════════════════════════════════════════
#  VERSION CONFIGS: Each version defines its feature engineering + models
# ═══════════════════════════════════════════════════════════════════════

VERSION_CONFIGS = {
    3: {
        "name": "V3: Domain features + deeper trees",
        "changes": [
            "Add drought_index (Temp*Wind/(Moisture+1)) — huge class separator",
            "Add water_balance (Rainfall+PrevIrrig-Temp*10)",
            "Add Crop_Growth_Stage interaction features",
            "Add binary threshold features (moisture<20, temp>30, wind>12)",
            "Increase LightGBM num_leaves=127, max_depth=10",
            "Increase XGBoost max_depth=9",
            "Remove low-importance freq/target encodings",
        ],
    },
    4: {
        "name": "V4: V2 hyperparams + V3 domain features (best of both)",
        "changes": [
            "Keep V2 hyperparams (num_leaves=63, max_depth=8) — V3 deeper trees overfit",
            "Keep V3 domain features (drought_index, water_balance, thresholds, stage interactions)",
            "Add evapotranspiration proxy (Temp*Sunlight/Humidity)",
            "Add soil_stress = Soil_pH * EC / (Organic_Carbon + 0.1)",
            "Restore all frequency encodings (V3 removal didn't help)",
            "Lower learning rate to 0.01 with more trees (3000) for LGBM",
            "Add early stopping via more estimators but lower lr",
        ],
    },
    5: {
        "name": "V5: CatBoost + multi-seed averaging",
        "changes": [
            "Keep V4 features (67 features, domain + interactions)",
            "Add CatBoost with native categorical handling",
            "Multi-seed LGBM (seeds 42, 123, 777) averaged for stable predictions",
            "Full 5-fold CV, all models",
        ],
    },
    6: {
        "name": "V6: Lower lr + feature selection + rank features + stronger regularization",
        "changes": [
            "Revert learning rate to 0.01 (V4 proved lr=0.01 > lr=0.05)",
            "Increase n_estimators: LGBM 5000, XGB 4000, CatBoost 5000 (compensate lower lr)",
            "Drop features with importance < 1% of max (25 noisy features from V5)",
            "Add rank-based features for top 5 numeric cols (percentile transforms)",
            "Increase regularization (reg_alpha=0.3, reg_lambda=2.0, l2_leaf_reg=5.0)",
            "Add 3-way interaction: moisture * temp * wind (drought severity)",
            "CatBoost depth=8 with l2_leaf_reg=5.0 for better generalization",
        ],
    },
    7: {
        "name": "V7: V2 hyperparams + domain features + per-fold target encoding + bug fixes",
        "changes": [
            "Use V2's winning lr=0.02 with n_estimators=2000 (proven optimal)",
            "Use V2's moderate regularization: reg_alpha=0.1, reg_lambda=1.0",
            "Use V2's moderate complexity: num_leaves=63, max_depth=8",
            "Fix target encoding leakage: per-fold encoding instead of full-train",
            "Keep proven V3 domain features: drought_index, water_balance, threshold flags, growth stage interactions",
            "Keep proven V4 domain features: evapotranspiration, soil_stress, total_water_input",
            "Remove StandardScaler (unnecessary for tree models)",
            "Remove V6 rank features and 3-way interaction (added noise)",
            "Fix weighted ensemble to compute actual OOF balanced accuracy",
        ],
    },
    8: {
        "name": "V8: HIGH class targeting + tuned hyperparams + feature cleanup",
        "changes": [
            "Keep V7 foundation: lr=0.02, per-fold TE, V2 regularization",
            "Increase num_leaves to 80 (more splits for minority class boundaries)",
            "Reduce min_child_samples to 30 (allow smaller leaf nodes for HIGH class)",
            "Increase n_estimators: LGBM 2500, XGB 2000, CatBoost 2500",
            "Add HIGH-class-targeted features: dry_hot_active, moisture_deficit_ratio, rainfall_adequacy",
            "Drop 3 useless features (importance < 100): wind_high, Mulching_Used_freq, moisture_low",
            "Exclude Mulching_Used from target encoding (importance=26)",
            "Increase HIGH class weight by 1.5x in sample_weights",
            "Use finer blend grid (step=0.05 instead of 0.1) for optimized_blend",
        ],
    },
    9: {
        "name": "V9: Revert hyperparams to V7 + CatBoost tuning + probability calibration",
        "changes": [
            "Revert LGBM/XGB to V7 hyperparams (num_leaves=63, min_child_samples=50) — V8 relaxation hurt",
            "Remove HIGH class weight boost (V8's 1.5x was too aggressive, -0.00137 on LGBM)",
            "Keep V8 features: dry_hot_active, moisture_deficit_ratio, rainfall_adequacy",
            "Keep V8 feature drops and Mulching_Used TE exclusion",
            "CatBoost: depth=6, iterations=2000 (V7 params — depth=7 was slower AND worse)",
            "Add post-blend probability calibration: search optimal HIGH class scaling on OOF",
            "Keep finer blend grid (step=0.05)",
        ],
    },
    10: {
        "name": "V10: Anti-overfit — no calibration, fixed blend, multi-seed averaging",
        "changes": [
            "Remove probability calibration entirely (source of CV-LB gap: 0.96999 CV vs 0.96715 LB)",
            "Replace grid-searched blend with fixed score-proportional weights",
            "Multi-seed averaging: LGBM 2 seeds, CatBoost 2 seeds",
            "Add 3 ratio features: moisture_rainfall_ratio, temp_humidity_ratio, wind_temp_product",
            "Drop temp_high, no_mulch (redundant with continuous/categorical originals)",
            "Exclude Soil_Type from TE (low importance across all models)",
            "Increase TE smoothing from 10 to 20 (reduce leakage)",
            "Increase regularization: reg_alpha=0.15, reg_lambda=1.5, l2_leaf_reg=5.0",
        ],
    },
    11: {
        "name": "V11: Research-driven overhaul — pairwise cat TE, decimal digits, groupby, logit bias, original data",
        "changes": [
            "Append original dataset (miadul/irrigation-water-requirement-prediction-dataset) with sample_weight=0.35",
            "Pairwise categorical interaction features with per-fold TE (C(8,2)=28 pairs → ~135 TE features)",
            "Decimal digit extraction for all numeric columns (synthetic data exploit)",
            "GroupBy aggregation features: groupby(cat)[num].agg(mean/std) for top combinations",
            "Multi-class target encoding: P(class=k|category) for each of 3 classes (replaces broken single-value TE)",
            "CatBoost native categorical handling via cat_features parameter",
            "10-fold CV (up from 5-fold) for more stable OOF estimates",
            "Logit-space bias tuning with nested CV: per-class logit offsets for balanced accuracy",
            "CatBoost-dominant blend: CatBoost=0.65, LightGBM=0.25, XGBoost=0.10",
            "Fix CatBoost sample_weight bug (was missing from fit call)",
            "Fix LightGBM bagging_freq=1 (enable actual row subsampling)",
            "Add field-normalized features: Previous_Irrigation/hectare, Rainfall/hectare",
        ],
    },
    12: {
        "name": "V12: 131 pairwise TE, L2 meta-stacker, CatBoost seed fix, XGB dual-seed",
        "changes": [
            "Expand pairwise TE from 10 key pairs to ~131 pairs (28 cat×cat + 88 cat×num + 15 num×num)",
            "Integer-coded fast pairwise TE: code = A*max_B + B, multi-class (3 probs per pair)",
            "Bin numeric features into 10 quantile bins for cat×num and num×num pairs",
            "L2-regularized LogisticRegression meta-stacker on 9 OOF probability columns",
            "Fix CatBoost seed: use get_params() instead of hasattr() for seed detection",
            "XGBoost dual-seed averaging (seed=42 + seed=123)",
            "colsample_bytree=0.5 for LGBM/XGB to handle ~500+ features",
            "Drop raw pair code columns after TE — only keep TE-derived features",
            "float32 TE columns to reduce memory footprint",
        ],
    },
}


def build_versioned_models(train, test, version=3, prev_results=None, fast=False):
    """Build models with version-specific improvements.
    fast=True: LightGBM only with 3-fold CV (~15 min instead of ~90 min)
    """
    logger.info(f"Iterative Builder V{version}: Starting... {'(FAST MODE)' if fast else ''}")
    config = VERSION_CONFIGS.get(version, VERSION_CONFIGS[3])
    logger.info(f"  Config: {config['name']}")
    for c in config["changes"]:
        logger.info(f"    • {c}")

    X_train, y_train, X_test, test_ids, label_enc, pipeline_steps, te_metadata = _preprocess(train, test, version)

    # Class weights
    class_counts = np.bincount(y_train)
    total = len(y_train)
    class_weights = {i: total / (len(class_counts) * c) for i, c in enumerate(class_counts)}
    if version == 8:
        # V8 only: Boost HIGH class weight by 1.5x (V9 reverts — too aggressive)
        class_weights[0] *= 1.5
        logger.info(f"  V8: Boosted HIGH class weight to {class_weights[0]:.2f}")
    sample_weights = np.array([class_weights[y] for y in y_train])
    logger.info(f"  Features: {X_train.shape[1]}, Class weights computed")

    models = _get_models(version, class_weights)
    if fast:
        # Keep only LightGBM model for speed
        lgbm_key = [k for k in models if 'lightgbm' in k or 'lgbm' in k]
        if lgbm_key:
            models = {lgbm_key[0]: models[lgbm_key[0]]}
    results = {}
    best_score = -1
    best_model_name = None
    best_test_preds = None
    oof_probs_all = {}

    n_folds = 3 if fast else CV_FOLDS
    for name, model in models.items():
        logger.info(f"Training {name} ({'3-fold' if fast else '5-fold'})...")
        result = _cv_predict(model, X_train, y_train, X_test, name, sample_weights, n_folds=n_folds, te_metadata=te_metadata)
        results[name] = result
        oof_probs_all[name] = result["oof_probabilities"]
        if result["mean_balanced_accuracy"] > best_score:
            best_score = result["mean_balanced_accuracy"]
            best_model_name = name
            best_test_preds = result["test_predictions"]

    # V5+: Multi-seed averaging for LGBM (more stable predictions)
    # Skip for V6+ to avoid OOM — multi-seed accumulates too much memory
    if version == 5 and not fast and "lightgbm" in results:
        logger.info("Multi-seed LGBM averaging (seeds 42, 123, 777)...")
        base_result = results["lightgbm"]
        multi_test_probs = base_result["test_probabilities"].copy()
        multi_oof_probs = base_result["oof_probabilities"].copy()
        for extra_seed in [123, 777]:
            seed_model = clone(models.get("lightgbm", list(models.values())[0]))
            seed_model.set_params(random_state=extra_seed)
            seed_result = _cv_predict(seed_model, X_train, y_train, X_test,
                                      f"lgbm_seed{extra_seed}", sample_weights, n_folds=n_folds, te_metadata=te_metadata)
            multi_test_probs += seed_result["test_probabilities"]
            multi_oof_probs += seed_result["oof_probabilities"]
            logger.info(f"  Seed {extra_seed}: BA = {seed_result['mean_balanced_accuracy']:.5f}")
        multi_test_probs /= 3
        multi_oof_probs /= 3
        multi_preds = multi_test_probs.argmax(axis=1)
        multi_oof_preds = multi_oof_probs.argmax(axis=1)
        multi_ba = balanced_accuracy_score(y_train, multi_oof_preds)
        logger.info(f"  Multi-seed LGBM avg: BA = {multi_ba:.5f}")
        results["lightgbm_multiseed"] = {
            "fold_scores": base_result["fold_scores"],
            "mean_balanced_accuracy": round(multi_ba, 5),
            "metrics": compute_metrics(y_train, multi_oof_preds, labels=[0, 1, 2]),
            "test_predictions": multi_preds,
            "test_probabilities": multi_test_probs,
            "oof_probabilities": multi_oof_probs,
            "feature_importance": base_result["feature_importance"],
            "params": {"method": "multi_seed_avg", "seeds": [42, 123, 777]},
        }
        if multi_ba > best_score:
            best_score = multi_ba
            best_model_name = "lightgbm_multiseed"
            best_test_preds = multi_preds

    # V7-9: Optimized blending via grid search on OOF predictions (skip V10+ to avoid overfitting)
    if 7 <= version <= 9 and not fast and len([n for n in results if n not in ("stacking_ensemble", "weighted_ensemble", "lightgbm_multiseed")]) >= 2:
        logger.info("Optimized blending via grid search on OOF...")
        blend_models = {n: r for n, r in results.items()
                        if n not in ("stacking_ensemble", "weighted_ensemble", "lightgbm_multiseed")
                        and r["mean_balanced_accuracy"] > 0.90}
        blend_names = list(blend_models.keys())
        best_blend_ba = -1
        best_blend_weights = None
        # Grid search over weight combinations
        from itertools import product as iproduct
        n_models = len(blend_names)
        if version >= 8:
            steps_grid = [round(x * 0.05, 2) for x in range(21)]  # finer grid: 0.05 step
        else:
            steps_grid = [round(x * 0.1, 1) for x in range(11)]  # V7: 0.1 step
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

    # V9 only: Post-blend probability calibration for HIGH class (skip V10+ — caused CV-LB gap)
    if version == 9 and not fast and "optimized_blend" in results:
        logger.info("V9: Post-blend HIGH class probability calibration...")
        cal_oof = results["optimized_blend"]["oof_probabilities"].copy()
        cal_test = results["optimized_blend"]["test_probabilities"].copy()
        best_cal_ba = results["optimized_blend"]["mean_balanced_accuracy"]
        best_cal_factor = 1.0
        # Search for optimal HIGH class (class 0) scaling factor
        for factor in [round(0.8 + i * 0.02, 2) for i in range(21)]:  # 0.80 to 1.20
            scaled_oof = cal_oof.copy()
            scaled_oof[:, 0] *= factor
            # Renormalize rows to sum to 1
            row_sums = scaled_oof.sum(axis=1, keepdims=True)
            scaled_oof = scaled_oof / row_sums
            cal_preds = scaled_oof.argmax(axis=1)
            cal_ba = balanced_accuracy_score(y_train, cal_preds)
            if cal_ba > best_cal_ba:
                best_cal_ba = cal_ba
                best_cal_factor = factor
        if best_cal_factor != 1.0:
            # Apply calibration to test predictions too
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

    # V10+: Fixed score-proportional blend (no grid search, no OOF tuning)
    if version >= 10 and not fast:
        base_model_names = [n for n in results if n in ("lightgbm", "xgboost", "catboost")]
        if len(base_model_names) >= 2:
            logger.info("V10: Fixed score-proportional blend (no grid search)...")
            total_ba = sum(results[n]["mean_balanced_accuracy"] for n in base_model_names)
            fixed_weights = {n: results[n]["mean_balanced_accuracy"] / total_ba for n in base_model_names}
            logger.info(f"  Score-proportional weights: {{{', '.join(f'{n}: {w:.4f}' for n, w in fixed_weights.items())}}}")
            blend_test = np.zeros_like(results[base_model_names[0]]["test_probabilities"])
            blend_oof = np.zeros_like(results[base_model_names[0]]["oof_probabilities"])
            for n, w in fixed_weights.items():
                blend_test += results[n]["test_probabilities"] * w
                blend_oof += results[n]["oof_probabilities"] * w
            blend_ba = balanced_accuracy_score(y_train, blend_oof.argmax(axis=1))
            logger.info(f"  Fixed blend BA: {blend_ba:.5f}")
            results["fixed_blend"] = {
                "fold_scores": [],
                "mean_balanced_accuracy": round(blend_ba, 5),
                "metrics": compute_metrics(y_train, blend_oof.argmax(axis=1), labels=[0, 1, 2]),
                "test_predictions": blend_test.argmax(axis=1),
                "test_probabilities": blend_test,
                "oof_probabilities": blend_oof,
                "feature_importance": results[base_model_names[0]]["feature_importance"],
                "params": {"method": "fixed_score_proportional_blend", "weights": fixed_weights},
            }
            if blend_ba > best_score:
                best_score = blend_ba
                best_model_name = "fixed_blend"
                best_test_preds = blend_test.argmax(axis=1)

            # Also compute equal-weight blend for comparison
            equal_w = 1.0 / len(base_model_names)
            eq_test = np.zeros_like(blend_test)
            eq_oof = np.zeros_like(blend_oof)
            for n in base_model_names:
                eq_test += results[n]["test_probabilities"] * equal_w
                eq_oof += results[n]["oof_probabilities"] * equal_w
            eq_ba = balanced_accuracy_score(y_train, eq_oof.argmax(axis=1))
            logger.info(f"  Equal blend BA: {eq_ba:.5f}")
            results["equal_blend"] = {
                "fold_scores": [],
                "mean_balanced_accuracy": round(eq_ba, 5),
                "metrics": compute_metrics(y_train, eq_oof.argmax(axis=1), labels=[0, 1, 2]),
                "test_predictions": eq_test.argmax(axis=1),
                "test_probabilities": eq_test,
                "oof_probabilities": eq_oof,
                "feature_importance": results[base_model_names[0]]["feature_importance"],
                "params": {"method": "equal_blend", "weights": {n: equal_w for n in base_model_names}},
            }
            if eq_ba > best_score:
                best_score = eq_ba
                best_model_name = "equal_blend"
                best_test_preds = eq_test.argmax(axis=1)

    # Skip ensembles in fast mode (only 1 model)
    if not fast:
        logger.info("Training stacking ensemble...")
        stack_result = _stacking_ensemble(results, oof_probs_all, X_train, y_train, X_test, sample_weights, version)
        results["stacking_ensemble"] = stack_result
        if stack_result["mean_balanced_accuracy"] > best_score:
            best_score = stack_result["mean_balanced_accuracy"]
            best_model_name = "stacking_ensemble"
            best_test_preds = stack_result["test_predictions"]

        logger.info("Training weighted ensemble...")
        we_result = _weighted_ensemble(results, X_test, y_train=y_train)
        results["weighted_ensemble"] = we_result
        if we_result["mean_balanced_accuracy"] > best_score:
            best_score = we_result["mean_balanced_accuracy"]
            best_model_name = "weighted_ensemble"
            best_test_preds = we_result["test_predictions"]

    final_preds = label_enc.inverse_transform(best_test_preds)
    logger.info(f"V{version} Best: {best_model_name} = {best_score:.5f}")

    return {
        "models": results,
        "best_model": best_model_name,
        "best_score": round(best_score, 5),
        "test_ids": test_ids.tolist(),
        "predictions": final_preds.tolist(),
        "pipeline_steps": pipeline_steps,
        "version": version,
        "version_config": config,
    }


# ═══════════════════════════════════════════════════════════════════════
#  PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════

def _preprocess(train, test, version):
    steps = []
    X_train = train.drop(columns=[TARGET, ID_COL], errors="ignore").copy()
    y_series = train[TARGET].copy()
    X_test = test.drop(columns=[ID_COL], errors="ignore").copy()
    test_ids = test[ID_COL].copy()

    label_enc = LabelEncoder()
    label_enc.fit(CLASS_LABELS)
    y_train = label_enc.transform(y_series)
    steps.append({"step": 1, "name": "Encode target", "description": "LabelEncoder: High=0, Low=1, Medium=2"})

    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]
    num_cols = [c for c in NUMERIC_FEATURES if c in X_train.columns]

    # Missing values
    for col in num_cols:
        med = X_train[col].median()
        X_train[col] = X_train[col].fillna(med)
        X_test[col] = X_test[col].fillna(med)
    for col in cat_cols:
        mode = X_train[col].mode()[0] if not X_train[col].mode().empty else "Unknown"
        X_train[col] = X_train[col].fillna(mode)
        X_test[col] = X_test[col].fillna(mode)
    steps.append({"step": 2, "name": "Handle missing values", "description": "Median/mode imputation"})

    # ── Domain features (V3+): BEFORE label encoding so we have raw values ──
    if version >= 3:
        # drought_index — the strongest engineered feature found in research
        X_train["drought_index"] = (X_train["Temperature_C"] * X_train["Wind_Speed_kmh"]) / (X_train["Soil_Moisture"] + 1)
        X_test["drought_index"] = (X_test["Temperature_C"] * X_test["Wind_Speed_kmh"]) / (X_test["Soil_Moisture"] + 1)

        # water_balance
        X_train["water_balance"] = X_train["Rainfall_mm"] + X_train["Previous_Irrigation_mm"] - X_train["Temperature_C"] * 10
        X_test["water_balance"] = X_test["Rainfall_mm"] + X_test["Previous_Irrigation_mm"] - X_test["Temperature_C"] * 10

        # Binary threshold features (from domain analysis)
        X_train["moisture_low"] = (X_train["Soil_Moisture"] < 20).astype(int)
        X_test["moisture_low"] = (X_test["Soil_Moisture"] < 20).astype(int)
        X_train["temp_high"] = (X_train["Temperature_C"] > 30).astype(int)
        X_test["temp_high"] = (X_test["Temperature_C"] > 30).astype(int)
        X_train["wind_high"] = (X_train["Wind_Speed_kmh"] > 12).astype(int)
        X_test["wind_high"] = (X_test["Wind_Speed_kmh"] > 12).astype(int)

        # Crop growth stage interactions (stage is critical splitter)
        # is_active_growth = Flowering or Vegetative (where High irrigation is possible)
        X_train["is_active_growth"] = X_train["Crop_Growth_Stage"].isin(["Flowering", "Vegetative"]).astype(int)
        X_test["is_active_growth"] = X_test["Crop_Growth_Stage"].isin(["Flowering", "Vegetative"]).astype(int)

        # drought during active growth — strongest signal for High class
        X_train["active_drought"] = X_train["is_active_growth"] * X_train["drought_index"]
        X_test["active_drought"] = X_test["is_active_growth"] * X_test["drought_index"]

        # moisture during active growth
        X_train["active_moisture"] = X_train["is_active_growth"] * X_train["Soil_Moisture"]
        X_test["active_moisture"] = X_test["is_active_growth"] * X_test["Soil_Moisture"]

        # Mulching effect (No mulching → higher irrigation need)
        X_train["no_mulch"] = (X_train["Mulching_Used"] == "No").astype(int)
        X_test["no_mulch"] = (X_test["Mulching_Used"] == "No").astype(int)
        X_train["no_mulch_drought"] = X_train["no_mulch"] * X_train["drought_index"]
        X_test["no_mulch_drought"] = X_test["no_mulch"] * X_test["drought_index"]

        steps.append({"step": 3, "name": "Domain features (V3+)",
                       "description": "drought_index, water_balance, threshold flags, growth stage interactions, mulching effects",
                       "code": "drought_index = Temp * Wind / (Moisture + 1)"})

    # ── V4+ additional domain features ──
    if version >= 4:
        # Evapotranspiration proxy
        X_train["evapotrans"] = X_train["Temperature_C"] * X_train["Sunlight_Hours"] / (X_train["Humidity"] + 1)
        X_test["evapotrans"] = X_test["Temperature_C"] * X_test["Sunlight_Hours"] / (X_test["Humidity"] + 1)

        # Soil stress index
        X_train["soil_stress"] = X_train["Soil_pH"] * X_train["Electrical_Conductivity"] / (X_train["Organic_Carbon"] + 0.1)
        X_test["soil_stress"] = X_test["Soil_pH"] * X_test["Electrical_Conductivity"] / (X_test["Organic_Carbon"] + 0.1)

        # Cumulative water input
        X_train["total_water_input"] = X_train["Rainfall_mm"] + X_train["Previous_Irrigation_mm"]
        X_test["total_water_input"] = X_test["Rainfall_mm"] + X_test["Previous_Irrigation_mm"]

        # Evapotrans during active growth
        X_train["active_evapotrans"] = X_train["is_active_growth"] * X_train["evapotrans"]
        X_test["active_evapotrans"] = X_test["is_active_growth"] * X_test["evapotrans"]

        steps.append({"step": "3b", "name": "V4+ domain features",
                       "description": "evapotranspiration proxy, soil stress, total water input",
                       "code": "evapotrans = Temp * Sunlight / (Humidity + 1)"})

    # ── V8+ features: HIGH class targeting ──
    if version >= 8:
        # dry_hot_active: core HIGH irrigation signal (dry soil + hot + active growth)
        X_train["dry_hot_active"] = ((X_train["Soil_Moisture"] < 25) &
                                      (X_train["Temperature_C"] > 28) &
                                      (X_train["is_active_growth"] == 1)).astype(int)
        X_test["dry_hot_active"] = ((X_test["Soil_Moisture"] < 25) &
                                     (X_test["Temperature_C"] > 28) &
                                     (X_test["is_active_growth"] == 1)).astype(int)

        # moisture_deficit_ratio: how far below median moisture, scaled by temperature
        median_moisture = 37.75
        X_train["moisture_deficit_ratio"] = (median_moisture - X_train["Soil_Moisture"]) / (median_moisture + 1) * (X_train["Temperature_C"] / 27)
        X_test["moisture_deficit_ratio"] = (median_moisture - X_test["Soil_Moisture"]) / (median_moisture + 1) * (X_test["Temperature_C"] / 27)

        # rainfall_adequacy: how much rainfall covers the temperature-driven demand
        X_train["rainfall_adequacy"] = X_train["Rainfall_mm"] / (X_train["Temperature_C"] * 50 + 1)
        X_test["rainfall_adequacy"] = X_test["Rainfall_mm"] / (X_test["Temperature_C"] * 50 + 1)

        steps.append({"step": "3c", "name": "V8 HIGH-class features",
                       "description": "dry_hot_active, moisture_deficit_ratio, rainfall_adequacy"})

    # ── V10+ features: ratio features for better generalization ──
    if version >= 10:
        # moisture_rainfall_ratio: strong HIGH class separator (HIGH mean=0.14 vs Low=0.04)
        X_train["moisture_rainfall_ratio"] = X_train["Soil_Moisture"] / (X_train["Rainfall_mm"] + 1)
        X_test["moisture_rainfall_ratio"] = X_test["Soil_Moisture"] / (X_test["Rainfall_mm"] + 1)

        # temp_humidity_ratio: HIGH=0.63 vs Low=0.45
        X_train["temp_humidity_ratio"] = X_train["Temperature_C"] / (X_train["Humidity"] + 1)
        X_test["temp_humidity_ratio"] = X_test["Temperature_C"] / (X_test["Humidity"] + 1)

        # wind_temp_product: combines two key HIGH-class signals (r=+0.25 with HIGH)
        X_train["wind_temp_product"] = X_train["Wind_Speed_kmh"] * X_train["Temperature_C"]
        X_test["wind_temp_product"] = X_test["Wind_Speed_kmh"] * X_test["Temperature_C"]

        steps.append({"step": "3d", "name": "V10 ratio features",
                       "description": "moisture_rainfall_ratio, temp_humidity_ratio, wind_temp_product"})

    # ── V11+ features: decimal digits, groupby, pairwise cats, field-normalized ──
    if version >= 11:
        # Decimal digit extraction — synthetic data has fixed decimal patterns per class
        digit_cols = ["Soil_Moisture", "Temperature_C", "Rainfall_mm", "Humidity",
                      "Wind_Speed_kmh", "Sunlight_Hours", "Soil_pH", "Organic_Carbon",
                      "Electrical_Conductivity", "Previous_Irrigation_mm", "Field_Area_hectare"]
        for col in digit_cols:
            if col in X_train.columns:
                X_train[f"{col}_d1"] = ((X_train[col] * 10) % 10).astype(int)
                X_test[f"{col}_d1"] = ((X_test[col] * 10) % 10).astype(int)
                X_train[f"{col}_d2"] = ((X_train[col] * 100) % 10).astype(int)
                X_test[f"{col}_d2"] = ((X_test[col] * 100) % 10).astype(int)

        # Field-normalized features
        X_train["irrig_per_hectare"] = X_train["Previous_Irrigation_mm"] / (X_train["Field_Area_hectare"] + 0.01)
        X_test["irrig_per_hectare"] = X_test["Previous_Irrigation_mm"] / (X_test["Field_Area_hectare"] + 0.01)
        X_train["rain_per_hectare"] = X_train["Rainfall_mm"] / (X_train["Field_Area_hectare"] + 0.01)
        X_test["rain_per_hectare"] = X_test["Rainfall_mm"] / (X_test["Field_Area_hectare"] + 0.01)

        # GroupBy aggregation features — contextual statistics per category
        groupby_specs = [
            ("Crop_Type", "Soil_Moisture", ["mean", "std"]),
            ("Crop_Type", "Temperature_C", ["mean"]),
            ("Region", "Temperature_C", ["mean", "std"]),
            ("Region", "Rainfall_mm", ["mean"]),
            ("Soil_Type", "Soil_Moisture", ["mean"]),
            ("Season", "Rainfall_mm", ["mean", "std"]),
            ("Irrigation_Type", "Previous_Irrigation_mm", ["mean"]),
            ("Crop_Growth_Stage", "Soil_Moisture", ["mean"]),
        ]
        for cat_col, num_col, aggs in groupby_specs:
            if cat_col in X_train.columns and num_col in X_train.columns:
                for agg in aggs:
                    feat_name = f"{cat_col}_{num_col}_{agg}"
                    mapping = X_train.groupby(cat_col)[num_col].agg(agg)
                    X_train[feat_name] = X_train[cat_col].map(mapping)
                    X_test[feat_name] = X_test[cat_col].map(mapping)
                    # Fill any unmapped test values with global stat
                    fill_val = X_train[num_col].mean() if agg == "mean" else X_train[num_col].std()
                    X_train[feat_name] = X_train[feat_name].fillna(fill_val)
                    X_test[feat_name] = X_test[feat_name].fillna(fill_val)

        steps.append({"step": "3e", "name": "V11 features",
                       "description": "decimal digits, field-normalized, groupby agg, pairwise cats"})

    # ── V6 only features: rank transforms + 3-way interactions ──
    if version == 6:
        # Rank-based features (percentile transforms — captures non-linear relationships)
        rank_cols = ["Soil_Moisture", "Temperature_C", "Wind_Speed_kmh", "Rainfall_mm", "drought_index"]
        for col in rank_cols:
            if col in X_train.columns:
                X_train[f"{col}_rank"] = X_train[col].rank(pct=True)
                X_test[f"{col}_rank"] = X_test[col].rank(pct=True)

        # 3-way interaction: drought severity
        X_train["drought_severity"] = X_train["Soil_Moisture"] * X_train["Temperature_C"] * X_train["Wind_Speed_kmh"]
        X_test["drought_severity"] = X_test["Soil_Moisture"] * X_test["Temperature_C"] * X_test["Wind_Speed_kmh"]

        steps.append({"step": "3c", "name": "V6 rank + 3-way features",
                       "description": "Percentile rank transforms, 3-way drought severity interaction"})

    # Label encode categoricals
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X_train[col], X_test[col]], axis=0).astype(str)
        le.fit(combined)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        label_encoders[col] = le
    steps.append({"step": 4, "name": "Label encode categoricals",
                   "description": f"LabelEncoder for {len(cat_cols)} categorical features"})

    # Target encoding (smoothed)
    te_metadata = None
    if version >= 7:
        # V7+: Per-fold target encoding to prevent leakage
        # Store metadata; actual encoding happens in _cv_predict()
        te_cat_cols = cat_cols[:]
        if version >= 10:
            # V10+: Exclude Mulching_Used and Soil_Type from TE
            te_cat_cols = [c for c in te_cat_cols if c not in ["Mulching_Used", "Soil_Type"]]
            logger.info(f"  V10: Excluded Mulching_Used + Soil_Type from TE ({len(te_cat_cols)} TE cols)")
        elif version >= 8:
            # V8+: Exclude Mulching_Used from TE (importance=26, too noisy)
            te_cat_cols = [c for c in te_cat_cols if c != "Mulching_Used"]
            logger.info(f"  V8: Excluded Mulching_Used from target encoding ({len(te_cat_cols)} TE cols)")
        smoothing = 20 if version >= 10 else 10

        # V11+: Pairwise categorical interaction columns for TE
        pairwise_cat_cols = []
        v12_pair_defs = None  # V12: on-the-fly pair definitions (not stored in DataFrame)
        if version >= 12:
            # V12: Define ~126 pairwise combinations — codes computed ON-THE-FLY during TE
            # This avoids storing 126 extra columns in X_train/X_test (saves ~0.6 GB)
            cat_encoded = [c for c in cat_cols if c in X_train.columns]

            # Compute quantile bin edges for numeric features
            num_bin_edges = {}
            for col in NUMERIC_FEATURES:
                if col in X_train.columns:
                    try:
                        _, edges = pd.qcut(X_train[col], q=10, retbins=True, duplicates="drop")
                        num_bin_edges[col] = edges.tolist()
                    except (ValueError, IndexError):
                        pass

            # Compute max values for cat columns (needed for integer coding)
            cat_max_vals = {}
            for c in cat_encoded:
                cat_max_vals[c] = int(max(X_train[c].max(), X_test[c].max())) + 1

            # Define all pairs
            pair_defs = []
            # 1) cat × cat: C(8,2) = 28
            for c1, c2 in combinations(cat_encoded, 2):
                pair_defs.append(("cc", c1, c2))
            n_cc = len(pair_defs)
            # 2) cat × num_binned: 8 × ~11 = ~88
            for cat_col in cat_encoded:
                for num_col in num_bin_edges:
                    pair_defs.append(("cn", cat_col, num_col))
            n_cn = len(pair_defs) - n_cc
            # 3) num × num: top 5 → C(5,2) = 10
            top_num_for_pairs = ["Soil_Moisture", "Temperature_C", "Rainfall_mm", "Humidity", "Wind_Speed_kmh"]
            top_avail = [c for c in top_num_for_pairs if c in num_bin_edges]
            for i, c1 in enumerate(top_avail):
                for c2 in top_avail[i+1:]:
                    pair_defs.append(("nn", c1, c2))
            n_nn = len(pair_defs) - n_cc - n_cn

            v12_pair_defs = {
                "pair_defs": pair_defs,
                "num_bin_edges": num_bin_edges,
                "cat_max_vals": cat_max_vals,
            }
            logger.info(f"  V12: Defined {len(pair_defs)} on-the-fly pairwise TE "
                         f"({n_cc} cat×cat, {n_cn} cat×num, {n_nn} num×num)")

        elif version >= 11:
            # V11: 10 key pairs with string concat + label encode
            key_pairs = [
                ("Crop_Type", "Crop_Growth_Stage"),
                ("Crop_Type", "Season"),
                ("Region", "Season"),
                ("Crop_Type", "Irrigation_Type"),
                ("Soil_Type", "Crop_Type"),
                ("Region", "Crop_Type"),
                ("Crop_Growth_Stage", "Season"),
                ("Irrigation_Type", "Water_Source"),
                ("Region", "Soil_Type"),
                ("Crop_Growth_Stage", "Irrigation_Type"),
            ]
            for c1, c2 in key_pairs:
                if c1 in X_train.columns and c2 in X_train.columns:
                    pair_name = f"{c1}_x_{c2}"
                    X_train[pair_name] = X_train[c1].astype(str) + "_" + X_train[c2].astype(str)
                    X_test[pair_name] = X_test[c1].astype(str) + "_" + X_test[c2].astype(str)
                    le_pair = LabelEncoder()
                    combined_pair = pd.concat([X_train[pair_name], X_test[pair_name]], axis=0)
                    le_pair.fit(combined_pair)
                    X_train[pair_name] = le_pair.transform(X_train[pair_name])
                    X_test[pair_name] = le_pair.transform(X_test[pair_name])
                    pairwise_cat_cols.append(pair_name)
            logger.info(f"  V11: Created {len(pairwise_cat_cols)} pairwise categorical features")

        te_metadata = {
            "cat_cols": te_cat_cols,
            "pairwise_cat_cols": pairwise_cat_cols,  # V11: pairwise cols in DataFrame
            "global_mean": float(np.mean(y_train)),
            "smoothing": smoothing,
            "multiclass": version >= 11,  # V11+: multi-class TE for base cats
            "n_classes": 3,
            "v12_pair_defs": v12_pair_defs,  # V12: on-the-fly pairwise TE definitions
        }
        steps.append({"step": 5, "name": "Target encoding (per-fold, deferred)",
                       "description": f"Per-fold target encoding for {len(te_cat_cols)} categoricals + {len(pairwise_cat_cols)} pairwise (applied in CV loop)"})
    else:
        # V2-V6: Full-train target encoding (has leakage)
        global_mean = y_train.mean()
        smoothing = 10
        for col in cat_cols:
            temp = pd.DataFrame({"cat": X_train[col], "target": y_train})
            agg = temp.groupby("cat")["target"].agg(["mean", "count"])
            sm = (agg["count"] * agg["mean"] + smoothing * global_mean) / (agg["count"] + smoothing)
            X_train[f"{col}_te"] = X_train[col].map(sm).fillna(global_mean)
            X_test[f"{col}_te"] = X_test[col].map(sm).fillna(global_mean)
        steps.append({"step": 5, "name": "Target encoding",
                       "description": f"Smoothed target encoding (s={smoothing})"})

    # Feature interactions (kept from V2)
    pairs = [
        ("Soil_Moisture", "Temperature_C", "ratio"),
        ("Soil_Moisture", "Temperature_C", "product"),
        ("Rainfall_mm", "Humidity", "product"),
        ("Soil_pH", "Organic_Carbon", "product"),
        ("Sunlight_Hours", "Wind_Speed_kmh", "ratio"),
        ("Rainfall_mm", "Previous_Irrigation_mm", "diff"),
        ("Rainfall_mm", "Previous_Irrigation_mm", "ratio"),
        ("Temperature_C", "Humidity", "product"),
    ]
    for a, b, op in pairs:
        if a in X_train.columns and b in X_train.columns:
            name = f"{a.split('_')[0]}_{b.split('_')[0]}_{op}"
            if op == "ratio":
                X_train[name] = X_train[a] / (X_train[b] + 1)
                X_test[name] = X_test[a] / (X_test[b] + 1)
            elif op == "product":
                X_train[name] = X_train[a] * X_train[b]
                X_test[name] = X_test[a] * X_test[b]
            elif op == "diff":
                X_train[name] = X_train[a] - X_train[b]
                X_test[name] = X_test[a] - X_test[b]
    steps.append({"step": 6, "name": "Feature interactions",
                   "description": "Ratios, products, diffs for key pairs"})

    # Polynomial for top features
    top_num = ["Rainfall_mm", "Temperature_C", "Soil_Moisture", "Wind_Speed_kmh", "Humidity"]
    for col in top_num:
        if col in X_train.columns:
            X_train[f"{col}_sq"] = X_train[col] ** 2
            X_test[f"{col}_sq"] = X_test[col] ** 2
            X_train[f"{col}_log"] = np.log1p(np.abs(X_train[col]))
            X_test[f"{col}_log"] = np.log1p(np.abs(X_test[col]))
    steps.append({"step": 7, "name": "Polynomial features",
                   "description": "Squared + log1p for top 5 numeric"})

    # Frequency encoding
    if version == 3:
        # V3: selective (experiment showed this didn't help)
        freq_cols = ["Crop_Growth_Stage", "Crop_Type", "Soil_Type"]
    else:
        # V2, V4+: all categoricals
        freq_cols = cat_cols
    for col in [c for c in freq_cols if c in X_train.columns]:
        freq = X_train[col].value_counts(normalize=True)
        X_train[f"{col}_freq"] = X_train[col].map(freq).fillna(0)
        X_test[f"{col}_freq"] = X_test[col].map(freq).fillna(0)
    steps.append({"step": 8, "name": "Frequency encoding",
                   "description": f"Frequency encoding for {len(freq_cols)} categoricals"})

    # V8+: Drop useless features (low importance)
    if version >= 10:
        drop_feats = ["wind_high", "Mulching_Used_freq", "moisture_low", "temp_high", "no_mulch"]
        existing_drops = [c for c in drop_feats if c in X_train.columns]
        if existing_drops:
            X_train = X_train.drop(columns=existing_drops)
            X_test = X_test.drop(columns=existing_drops)
            logger.info(f"  V10: Dropped {len(existing_drops)} low-importance features: {existing_drops}")
    elif version >= 8:
        drop_feats = ["wind_high", "Mulching_Used_freq", "moisture_low"]
        existing_drops = [c for c in drop_feats if c in X_train.columns]
        if existing_drops:
            X_train = X_train.drop(columns=existing_drops)
            X_test = X_test.drop(columns=existing_drops)
            logger.info(f"  V8: Dropped {len(existing_drops)} low-importance features: {existing_drops}")

    # Standard scaling — skip for V7+ (tree models don't benefit)
    if version < 7:
        all_num = X_train.select_dtypes(include=[np.number]).columns.tolist()
        scaler = StandardScaler()
        X_train[all_num] = scaler.fit_transform(X_train[all_num])
        X_test[all_num] = scaler.transform(X_test[all_num])
        steps.append({"step": 9, "name": "Standard scaling",
                       "description": f"StandardScaler on {len(all_num)} features"})

    # V6 only: Feature selection — drop low-importance features based on prev results
    if version == 6:
        try:
            import json
            with open("outputs/results.json") as f:
                prev_data = json.load(f)
            best_m = prev_data.get("best_model", "")
            fi = prev_data.get("models", {}).get(best_m, {}).get("feature_importance", {})
            if fi:
                max_imp = max(fi.values())
                threshold = max_imp * 0.01  # drop < 1% of max
                drop_cols = [c for c in X_train.columns if c in fi and fi[c] < threshold]
                if drop_cols:
                    logger.info(f"  V6 feature selection: dropping {len(drop_cols)} low-importance features")
                    X_train = X_train.drop(columns=drop_cols, errors="ignore")
                    X_test = X_test.drop(columns=drop_cols, errors="ignore")
                    steps.append({"step": "9b", "name": "Feature selection (V6)",
                                   "description": f"Dropped {len(drop_cols)} features with importance < 1% of max"})
        except Exception as e:
            logger.warning(f"  Could not load prev results for feature selection: {e}")

    logger.info(f"  Preprocessing done: {X_train.shape[1]} features")
    return X_train, y_train, X_test, test_ids, label_enc, steps, te_metadata


def _apply_per_fold_te(X_tr, y_tr, X_val, X_te, te_metadata):
    """Apply per-fold target encoding: fit on train fold, transform val + test."""
    cat_cols = te_metadata["cat_cols"]
    global_mean = te_metadata["global_mean"]
    smoothing = te_metadata["smoothing"]
    multiclass = te_metadata.get("multiclass", False)
    n_classes = te_metadata.get("n_classes", 3)
    pairwise_cat_cols = te_metadata.get("pairwise_cat_cols", [])

    # All columns to target-encode (base categoricals + pairwise interactions)
    all_te_cols = list(cat_cols) + list(pairwise_cat_cols)

    v12_pair_defs = te_metadata.get("v12_pair_defs", None)

    if multiclass:
        # Multi-class TE for base categoricals: P(class=k | category) for each class
        global_cls_means = [float((y_tr == cls).mean()) for cls in range(n_classes)]
        for col in cat_cols:
            vals = X_tr[col].values
            for cls in range(n_classes):
                te_col = f"{col}_te_c{cls}"
                is_cls = (y_tr == cls).astype(np.float32)
                temp = pd.DataFrame({"cat": vals, "is_cls": is_cls})
                agg = temp.groupby("cat")["is_cls"].agg(["mean", "count"])
                sm = (agg["count"] * agg["mean"] + smoothing * global_cls_means[cls]) / (agg["count"] + smoothing)
                X_tr[te_col] = X_tr[col].map(sm).fillna(global_cls_means[cls]).astype(np.float32)
                X_val[te_col] = X_val[col].map(sm).fillna(global_cls_means[cls]).astype(np.float32)
                X_te[te_col] = X_te[col].map(sm).fillna(global_cls_means[cls]).astype(np.float32)

        # V12: On-the-fly pairwise TE (pair codes computed here, not stored in data)
        if v12_pair_defs is not None:
            pair_defs = v12_pair_defs["pair_defs"]
            num_bin_edges = v12_pair_defs["num_bin_edges"]
            cat_max_vals = v12_pair_defs["cat_max_vals"]

            def _bin_vals(arr, edges):
                """Bin continuous values using pre-computed edges."""
                return np.clip(np.searchsorted(edges[1:-1], arr), 0, len(edges) - 2).astype(np.int32)

            te_names = []
            tr_arrays = []
            val_arrays = []
            te_arrays = []

            for pair_type, c1, c2 in pair_defs:
                # Compute integer pair codes on-the-fly
                if pair_type == "cc":  # cat × cat
                    mx = cat_max_vals[c2]
                    tr_code = X_tr[c1].values * mx + X_tr[c2].values
                    val_code = X_val[c1].values * mx + X_val[c2].values
                    te_code = X_te[c1].values * mx + X_te[c2].values
                elif pair_type == "cn":  # cat × num_binned
                    edges = num_bin_edges[c2]
                    n_bins = len(edges) - 1
                    tr_code = X_tr[c1].values * n_bins + _bin_vals(X_tr[c2].values, edges)
                    val_code = X_val[c1].values * n_bins + _bin_vals(X_val[c2].values, edges)
                    te_code = X_te[c1].values * n_bins + _bin_vals(X_te[c2].values, edges)
                else:  # nn: num × num
                    e1, e2 = num_bin_edges[c1], num_bin_edges[c2]
                    nb2 = len(e2) - 1
                    tr_code = _bin_vals(X_tr[c1].values, e1) * nb2 + _bin_vals(X_tr[c2].values, e2)
                    val_code = _bin_vals(X_val[c1].values, e1) * nb2 + _bin_vals(X_val[c2].values, e2)
                    te_code = _bin_vals(X_te[c1].values, e1) * nb2 + _bin_vals(X_te[c2].values, e2)

                pair_name = f"p_{c1}_x_{c2}"
                # Single-value smoothed TE per pair (memory-efficient)
                temp = pd.DataFrame({"cat": tr_code, "target": y_tr.astype(np.float32)})
                agg = temp.groupby("cat")["target"].agg(["mean", "count"])
                sm = (agg["count"] * agg["mean"] + smoothing * global_mean) / (agg["count"] + smoothing)
                sm_dict = sm.to_dict()
                te_col = f"{pair_name}_te"
                tr_arrays.append(pd.Series(tr_code).map(sm_dict).fillna(global_mean).values.astype(np.float32))
                val_arrays.append(pd.Series(val_code).map(sm_dict).fillna(global_mean).values.astype(np.float32))
                te_arrays.append(pd.Series(te_code).map(sm_dict).fillna(global_mean).values.astype(np.float32))
                te_names.append(te_col)

            # Batch-assign all pairwise TE columns via concat (avoids fragmentation)
            if te_names:
                tr_te = pd.DataFrame(np.column_stack(tr_arrays).astype(np.float32),
                                     columns=te_names, index=X_tr.index)
                val_te = pd.DataFrame(np.column_stack(val_arrays).astype(np.float32),
                                      columns=te_names, index=X_val.index)
                te_te = pd.DataFrame(np.column_stack(te_arrays).astype(np.float32),
                                     columns=te_names, index=X_te.index)
                del tr_arrays, val_arrays, te_arrays
                X_tr = pd.concat([X_tr, tr_te], axis=1); del tr_te
                X_val = pd.concat([X_val, val_te], axis=1); del val_te
                X_te = pd.concat([X_te, te_te], axis=1); del te_te

        elif pairwise_cat_cols:
            # V11: Single-value TE for pairwise cols stored in DataFrame
            for col in pairwise_cat_cols:
                te_col = f"{col}_te"
                temp = pd.DataFrame({"cat": X_tr[col].values, "target": y_tr})
                agg = temp.groupby("cat")["target"].agg(["mean", "count"])
                sm = (agg["count"] * agg["mean"] + smoothing * global_mean) / (agg["count"] + smoothing)
                X_tr[te_col] = X_tr[col].map(sm).fillna(global_mean)
                X_val[te_col] = X_val[col].map(sm).fillna(global_mean)
                X_te[te_col] = X_te[col].map(sm).fillna(global_mean)
    else:
        # Legacy single-value TE
        for col in all_te_cols:
            te_col = f"{col}_te"
            temp = pd.DataFrame({"cat": X_tr[col].values, "target": y_tr})
            agg = temp.groupby("cat")["target"].agg(["mean", "count"])
            sm = (agg["count"] * agg["mean"] + smoothing * global_mean) / (agg["count"] + smoothing)
            X_tr[te_col] = X_tr[col].map(sm).fillna(global_mean)
            X_val[te_col] = X_val[col].map(sm).fillna(global_mean)
            X_te[te_col] = X_te[col].map(sm).fillna(global_mean)

    return X_tr, X_val, X_te


# ═══════════════════════════════════════════════════════════════════════
#  MODELS
# ═══════════════════════════════════════════════════════════════════════

def _get_models(version, class_weights):
    models = {}

    if version >= 12:
        # V12: Higher LR (0.05) for <5min/fold, lower colsample for ~240 features
        models["lightgbm"] = LGBMClassifier(
            n_estimators=2500,
            max_depth=8,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=50,
            subsample=0.8,
            bagging_freq=1,
            colsample_bytree=0.5,  # V12: lower for ~240 features
            reg_alpha=0.15,
            reg_lambda=1.5,
            class_weight="balanced",
            objective="multiclass",
            num_class=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        models["xgboost"] = XGBClassifier(
            n_estimators=2000,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.5,  # V12: lower for ~240 features
            min_child_weight=5,
            reg_alpha=0.15,
            reg_lambda=1.5,
            gamma=0.03,
            objective="multi:softprob",
            num_class=3,
            early_stopping_rounds=50,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
        if HAS_CATBOOST:
            models["catboost"] = CatBoostClassifier(
                iterations=2500,
                depth=7,
                learning_rate=0.05,
                l2_leaf_reg=4.0,  # V12: slightly more regularization
                rsm=0.5,  # V12: random subspace — 50% features per split
                auto_class_weights="Balanced",
                random_seed=RANDOM_STATE,
                verbose=0,
                thread_count=-1,
            )
    elif version >= 11:
        # V11: Research-driven params — bagging_freq fix, more trees, proper regularization
        models["lightgbm"] = LGBMClassifier(
            n_estimators=2500,
            max_depth=8,
            learning_rate=0.02,
            num_leaves=63,
            min_child_samples=50,
            subsample=0.8,
            bagging_freq=1,  # FIX: enable actual row subsampling
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            class_weight="balanced",
            objective="multiclass",
            num_class=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        models["xgboost"] = XGBClassifier(
            n_estimators=2000,
            max_depth=7,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            gamma=0.02,
            objective="multi:softprob",
            num_class=3,
            early_stopping_rounds=50,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
        if HAS_CATBOOST:
            models["catboost"] = CatBoostClassifier(
                iterations=2500,
                depth=7,
                learning_rate=0.02,
                l2_leaf_reg=3.0,
                auto_class_weights="Balanced",
                random_seed=RANDOM_STATE,
                verbose=0,
                thread_count=-1,
            )
    elif version >= 10:
        # V10: Slightly more regularization for better generalization
        models["lightgbm"] = LGBMClassifier(
            n_estimators=2000,
            max_depth=8,
            learning_rate=0.02,
            num_leaves=63,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.15,
            reg_lambda=1.5,
            class_weight="balanced",
            objective="multiclass",
            num_class=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        models["xgboost"] = XGBClassifier(
            n_estimators=1500,
            max_depth=7,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=5,
            reg_alpha=0.15,
            reg_lambda=1.5,
            gamma=0.05,
            objective="multi:softprob",
            num_class=3,
            early_stopping_rounds=50,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
        if HAS_CATBOOST:
            models["catboost"] = CatBoostClassifier(
                iterations=2000,
                depth=6,
                learning_rate=0.02,
                l2_leaf_reg=5.0,
                auto_class_weights="Balanced",
                random_seed=RANDOM_STATE,
                verbose=0,
                thread_count=-1,
            )
    elif version >= 9:
        # V9: Revert LGBM/XGB to V7 proven params, tune CatBoost
        models["lightgbm"] = LGBMClassifier(
            n_estimators=2000,
            max_depth=8,
            learning_rate=0.02,
            num_leaves=63,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            class_weight="balanced",
            objective="multiclass",
            num_class=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        models["xgboost"] = XGBClassifier(
            n_estimators=1500,
            max_depth=7,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="multi:softprob",
            num_class=3,
            early_stopping_rounds=50,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
        if HAS_CATBOOST:
            models["catboost"] = CatBoostClassifier(
                iterations=2000,
                depth=6,
                learning_rate=0.02,
                l2_leaf_reg=3.0,
                auto_class_weights="Balanced",
                random_seed=RANDOM_STATE,
                verbose=0,
                thread_count=-1,
            )
    elif version >= 8:
        # V8: Relaxed hyperparams for better minority class capture
        models["lightgbm"] = LGBMClassifier(
            n_estimators=2500,
            max_depth=8,
            learning_rate=0.02,
            num_leaves=80,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            class_weight="balanced",
            objective="multiclass",
            num_class=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        models["xgboost"] = XGBClassifier(
            n_estimators=2000,
            max_depth=7,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="multi:softprob",
            num_class=3,
            early_stopping_rounds=50,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
        if HAS_CATBOOST:
            models["catboost"] = CatBoostClassifier(
                iterations=2500,
                depth=6,
                learning_rate=0.02,
                l2_leaf_reg=3.0,
                auto_class_weights="Balanced",
                random_seed=RANDOM_STATE,
                verbose=0,
                thread_count=-1,
            )
    elif version >= 7:
        # V7: V2's exact winning hyperparameters + domain features
        models["lightgbm"] = LGBMClassifier(
            n_estimators=2000,
            max_depth=8,
            learning_rate=0.02,
            num_leaves=63,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            class_weight="balanced",
            objective="multiclass",
            num_class=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        models["xgboost"] = XGBClassifier(
            n_estimators=1500,
            max_depth=7,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="multi:softprob",
            num_class=3,
            early_stopping_rounds=50,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
        if HAS_CATBOOST:
            models["catboost"] = CatBoostClassifier(
                iterations=2000,
                depth=6,
                learning_rate=0.02,
                l2_leaf_reg=3.0,
                auto_class_weights="Balanced",
                random_seed=RANDOM_STATE,
                verbose=0,
                thread_count=-1,
            )
    elif version >= 5:
        # V5: lr=0.05; V6+: revert to lr=0.01 (V4 proved lower lr is better)
        lr = 0.05 if version == 5 else 0.01
        models["lightgbm"] = LGBMClassifier(
            n_estimators=3000 if version == 5 else 5000,
            max_depth=8,
            learning_rate=lr,
            num_leaves=63,
            min_child_samples=50 if version < 6 else 30,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1 if version < 6 else 0.3,
            reg_lambda=1.0 if version < 6 else 2.0,
            class_weight="balanced",
            objective="multiclass",
            num_class=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        models["xgboost"] = XGBClassifier(
            n_estimators=2500 if version == 5 else 4000,
            max_depth=7,
            learning_rate=lr,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=5,
            reg_alpha=0.1 if version < 6 else 0.3,
            reg_lambda=1.0 if version < 6 else 2.0,
            objective="multi:softprob",
            num_class=3,
            early_stopping_rounds=50,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
        if HAS_CATBOOST:
            models["catboost"] = CatBoostClassifier(
                iterations=2000 if version == 5 else 3000,
                depth=6 if version == 5 else 8,
                learning_rate=0.05 if version == 5 else 0.01,
                l2_leaf_reg=3.0 if version < 6 else 5.0,
                auto_class_weights="Balanced",
                random_seed=RANDOM_STATE,
                verbose=0,
                thread_count=-1,
            )
    elif version == 4:
        # V4: V2 hyperparams + domain features, lower lr with more trees
        models["lightgbm_v4"] = LGBMClassifier(
            n_estimators=3000,
            max_depth=8,
            learning_rate=0.01,
            num_leaves=63,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            class_weight="balanced",
            objective="multiclass",
            num_class=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        models["xgboost_v4"] = XGBClassifier(
            n_estimators=2500,
            max_depth=7,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="multi:softprob",
            num_class=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
        models["random_forest_v4"] = RandomForestClassifier(
            n_estimators=700,
            max_depth=18,
            min_samples_leaf=2,
            min_samples_split=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    elif version == 3:
        models["lightgbm_v3"] = LGBMClassifier(
            n_estimators=2500, max_depth=10, learning_rate=0.015,
            num_leaves=127, min_child_samples=40, subsample=0.8,
            colsample_bytree=0.65, reg_alpha=0.05, reg_lambda=0.8,
            class_weight="balanced", objective="multiclass", num_class=3,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        )
        models["xgboost_v3"] = XGBClassifier(
            n_estimators=2000, max_depth=9, learning_rate=0.015,
            subsample=0.8, colsample_bytree=0.65, min_child_weight=3,
            reg_alpha=0.05, reg_lambda=0.8, gamma=0.1,
            objective="multi:softprob", num_class=3,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
        )
        models["random_forest_v3"] = RandomForestClassifier(
            n_estimators=500, max_depth=18, min_samples_leaf=2,
            min_samples_split=4, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
    else:
        # V2 fallback
        models["lightgbm_tuned"] = LGBMClassifier(
            n_estimators=2000, max_depth=8, learning_rate=0.02,
            num_leaves=63, min_child_samples=50, subsample=0.8,
            colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
            class_weight="balanced", objective="multiclass", num_class=3,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        )
        models["xgboost_tuned"] = XGBClassifier(
            n_estimators=1500, max_depth=7, learning_rate=0.02,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
            reg_alpha=0.1, reg_lambda=1.0, objective="multi:softprob",
            num_class=3, random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
        )

    return models


# ═══════════════════════════════════════════════════════════════════════
#  CV + PREDICT
# ═══════════════════════════════════════════════════════════════════════

def _cv_predict(model, X_train, y_train, X_test, name, sample_weights=None, n_folds=None, te_metadata=None, fold_range=None):
    """Run CV prediction. fold_range=(start, end) runs only folds [start, end) for splitting long jobs."""
    if n_folds is None:
        n_folds = CV_FOLDS
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(y_train), dtype=int)
    oof_probs = np.zeros((len(y_train), 3))
    test_probs = np.zeros((len(X_test), 3))
    fold_scores = []
    # Feature count may differ if per-fold TE adds columns; track after first fold
    feat_imp = None
    feat_names = None

    fold_start = fold_range[0] if fold_range else 0
    fold_end = fold_range[1] if fold_range else n_folds

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        if fold < fold_start or fold >= fold_end:
            continue

        y_tr = y_train[tr_idx]
        y_val = y_train[val_idx]

        # Per-fold target encoding: copy data and apply TE per fold
        if te_metadata is not None:
            X_tr = X_train.iloc[tr_idx].copy()
            X_val = X_train.iloc[val_idx].copy()
            X_te_fold = X_test.copy()
            X_tr, X_val, X_te_fold = _apply_per_fold_te(X_tr, y_tr, X_val, X_te_fold, te_metadata)
            if fold == 0:
                n_pair_te = len(te_metadata.get("v12_pair_defs", {}).get("pair_defs", [])) if te_metadata.get("v12_pair_defs") else len(te_metadata.get("pairwise_cat_cols", []))
                logger.info(f"  Per-fold TE: {X_tr.shape[1]} features (incl. {len(te_metadata['cat_cols'])} base TE + {n_pair_te} pairwise TE)")
        else:
            X_tr = X_train.iloc[tr_idx]
            X_val = X_train.iloc[val_idx]
            X_te_fold = X_test

        if feat_imp is None:
            feat_imp = np.zeros(X_tr.shape[1])
            feat_names = X_tr.columns.tolist()

        m = clone(model)
        sw = sample_weights[tr_idx] if sample_weights is not None else None

        # Use early stopping for boosting models (huge speedup + prevents overfitting)
        model_type = type(m).__name__
        try:
            if model_type == "LGBMClassifier":
                m.fit(X_tr, y_tr, sample_weight=sw,
                      eval_set=[(X_val, y_val)],
                      callbacks=[lgb_early_stopping(50, verbose=False), lgb_log_eval(-1)])
            elif model_type == "XGBClassifier":
                m.fit(X_tr, y_tr, sample_weight=sw,
                      eval_set=[(X_val, y_val)],
                      verbose=False)
            elif model_type == "CatBoostClassifier":
                cb_kwargs = {"eval_set": (X_val, y_val), "early_stopping_rounds": 50}
                if sw is not None:
                    cb_kwargs["sample_weight"] = sw  # FIX: was missing in V3-V10
                m.fit(X_tr, y_tr, **cb_kwargs)
            else:
                m.fit(X_tr, y_tr, sample_weight=sw)
        except TypeError:
            m.fit(X_tr, y_tr)

        preds = np.asarray(m.predict(X_val)).ravel()
        oof_preds[val_idx] = preds
        ba = balanced_accuracy_score(y_val, preds)
        fold_scores.append(round(ba, 5))

        if hasattr(m, "predict_proba"):
            val_proba = m.predict_proba(X_val)
            test_proba = m.predict_proba(X_te_fold)
            if val_proba.ndim == 1:
                val_proba = val_proba.reshape(-1, 1)
            if test_proba.ndim == 1:
                test_proba = test_proba.reshape(-1, 1)
            oof_probs[val_idx] = val_proba
            test_probs += test_proba / n_folds
        if hasattr(m, "feature_importances_"):
            feat_imp += m.feature_importances_ / n_folds

        logger.info(f"  Fold {fold+1}/{n_folds}: BA = {ba:.5f}")

        # Free memory between folds (critical for CatBoost which is memory-hungry)
        del m, X_tr, X_val, X_te_fold
        try:
            del val_proba, test_proba
        except NameError:
            pass
        import gc; gc.collect()

    # For partial fold runs, scale test_probs by actual folds run
    n_folds_run = fold_end - fold_start
    if fold_range and n_folds_run < n_folds:
        # test_probs was accumulated with /n_folds per fold; rescale to represent partial contribution
        pass  # Already correct — each fold contributes test_proba / n_folds

    # Compute OOF BA only on folds that were actually run
    if fold_range:
        # Partial run — BA on filled indices only
        filled_mask = oof_probs.sum(axis=1) > 0
        if filled_mask.any():
            mean_ba = balanced_accuracy_score(y_train[filled_mask], oof_preds[filled_mask])
            metrics = compute_metrics(y_train[filled_mask], oof_preds[filled_mask], labels=[0, 1, 2])
        else:
            mean_ba = 0.0
            metrics = {}
    else:
        mean_ba = balanced_accuracy_score(y_train, oof_preds)
        metrics = compute_metrics(y_train, oof_preds, labels=[0, 1, 2])

    if feat_names is None:
        feat_names = X_train.columns.tolist()
    if feat_imp is None:
        feat_imp = np.zeros(len(feat_names))
    fi = dict(sorted(zip(feat_names, feat_imp.tolist()), key=lambda x: x[1], reverse=True))

    return {
        "fold_scores": fold_scores,
        "mean_balanced_accuracy": round(mean_ba, 5),
        "metrics": metrics,
        "test_predictions": test_probs.argmax(axis=1),
        "test_probabilities": test_probs,
        "oof_probabilities": oof_probs,
        "feature_importance": fi,
        "params": _safe_params(model),
        "_fold_range": fold_range,  # Track which folds were run
    }


def _stacking_ensemble(results, oof_probs, X_train, y_train, X_test, sw, version):
    base = [n for n in results if n not in ("stacking_ensemble", "weighted_ensemble", "lightgbm_multiseed", "optimized_blend")]
    meta_tr = np.hstack([oof_probs[n] for n in base])
    meta_te = np.hstack([results[n]["test_probabilities"] for n in base])

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # V3+: use LightGBM as meta-learner instead of LR
    if version >= 3:
        meta_model = LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            num_leaves=15, random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
        )
    else:
        meta_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

    oof_preds = np.zeros(len(y_train), dtype=int)
    test_probs = np.zeros((len(X_test), 3))
    fold_scores = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(meta_tr, y_train)):
        m = clone(meta_model)
        m.fit(meta_tr[tr_idx], y_train[tr_idx])
        preds = m.predict(meta_tr[val_idx])
        oof_preds[val_idx] = preds
        ba = balanced_accuracy_score(y_train[val_idx], preds)
        fold_scores.append(round(ba, 5))
        test_probs += m.predict_proba(meta_te) / CV_FOLDS
        logger.info(f"  Stacking Fold {fold+1}/{CV_FOLDS}: BA = {ba:.5f}")

    mean_ba = balanced_accuracy_score(y_train, oof_preds)
    metrics = compute_metrics(y_train, oof_preds, labels=[0, 1, 2])

    fi = {}
    for n in base:
        for feat, imp in results[n]["feature_importance"].items():
            fi[feat] = fi.get(feat, 0) + imp / len(base)
    fi = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))

    return {
        "fold_scores": fold_scores,
        "mean_balanced_accuracy": round(mean_ba, 5),
        "metrics": metrics,
        "test_predictions": test_probs.argmax(axis=1),
        "test_probabilities": test_probs,
        "oof_probabilities": np.zeros((len(y_train), 3)),
        "feature_importance": fi,
        "params": {"method": "stacking", "meta_learner": "LightGBM" if version >= 3 else "LR", "base_models": base},
    }


def _weighted_ensemble(results, X_test, y_train=None):
    _exclude = ("stacking_ensemble", "weighted_ensemble", "lightgbm_multiseed", "optimized_blend")
    base = {n: r for n, r in results.items()
            if n not in _exclude
            and r["mean_balanced_accuracy"] > 0.90}
    if not base:
        base = {n: r for n, r in results.items() if n not in _exclude}

    weights = {n: r["mean_balanced_accuracy"] ** 3 for n, r in base.items()}
    total_w = sum(weights.values())

    test_probs = np.zeros((len(X_test), 3))
    oof_probs = None
    for n, r in base.items():
        w = weights[n] / total_w
        test_probs += r["test_probabilities"] * w
        if r.get("oof_probabilities") is not None and len(r["oof_probabilities"]) > 0:
            if oof_probs is None:
                oof_probs = np.zeros_like(r["oof_probabilities"])
            oof_probs += r["oof_probabilities"] * w

    # Compute actual ensemble OOF balanced accuracy
    if oof_probs is not None and y_train is not None:
        ensemble_oof_preds = oof_probs.argmax(axis=1)
        ensemble_ba = balanced_accuracy_score(y_train, ensemble_oof_preds)
        ensemble_metrics = compute_metrics(y_train, ensemble_oof_preds, labels=[0, 1, 2])
        logger.info(f"  Weighted ensemble actual OOF BA: {ensemble_ba:.5f}")
    else:
        best_n = max(base, key=lambda n: base[n]["mean_balanced_accuracy"])
        ensemble_ba = base[best_n]["mean_balanced_accuracy"]
        ensemble_metrics = base[best_n]["metrics"]
        oof_probs = np.zeros((len(X_test), 3))

    fi = {}
    for n, r in base.items():
        w = weights[n] / total_w
        for feat, imp in r["feature_importance"].items():
            fi[feat] = fi.get(feat, 0) + imp * w
    fi = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))

    return {
        "fold_scores": [round(weights[n], 5) for n in weights],
        "mean_balanced_accuracy": round(ensemble_ba, 5),
        "metrics": ensemble_metrics,
        "test_predictions": test_probs.argmax(axis=1),
        "test_probabilities": test_probs,
        "oof_probabilities": oof_probs if oof_probs is not None else np.zeros((len(X_test), 3)),
        "feature_importance": fi,
        "params": {"method": "weighted_cube", "weights": {k: round(v/total_w, 4) for k, v in weights.items()}},
    }


def _safe_params(model):
    params = model.get_params()
    clean = {}
    for k, v in params.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            clean[k] = None
        elif not isinstance(v, (int, float, str, bool, type(None))):
            clean[k] = str(v)
        else:
            clean[k] = v
    return clean
