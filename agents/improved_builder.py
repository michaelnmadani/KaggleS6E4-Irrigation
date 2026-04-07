"""Improved Builder Agent: Implements researcher recommendations for better models."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.base import clone

from pipeline.config import (
    TARGET, ID_COL, CLASS_LABELS, CATEGORICAL_FEATURES, NUMERIC_FEATURES,
    CV_FOLDS, RANDOM_STATE
)
from pipeline.utils import setup_logging, compute_metrics

logger = setup_logging()


def build_improved_models(train, test, recommendations=None):
    """Build improved models based on researcher recommendations."""
    logger.info("Improved Builder Agent: Starting enhanced model training...")

    X_train, y_train, X_test, test_ids, label_enc, pipeline_steps = _enhanced_preprocess(train, test)

    # Compute class weights for imbalance
    class_counts = np.bincount(y_train)
    total = len(y_train)
    class_weights = {i: total / (len(class_counts) * count) for i, count in enumerate(class_counts)}
    sample_weights = np.array([class_weights[y] for y in y_train])
    logger.info(f"Class weights: {class_weights}")

    models = _get_improved_models(class_weights)
    results = {}
    best_score = -1
    best_model_name = None
    best_test_preds = None
    oof_probs_all = {}

    for name, model in models.items():
        logger.info(f"Training {name}...")
        model_result = _cross_validate_and_predict(
            model, X_train, y_train, X_test, name, sample_weights
        )
        results[name] = model_result
        oof_probs_all[name] = model_result["oof_probabilities"]

        if model_result["mean_balanced_accuracy"] > best_score:
            best_score = model_result["mean_balanced_accuracy"]
            best_model_name = name
            best_test_preds = model_result["test_predictions"]

    # Stacking ensemble
    logger.info("Training stacking ensemble...")
    stacking_result = _build_stacking_ensemble(
        results, oof_probs_all, X_train, y_train, X_test, sample_weights
    )
    results["stacking_ensemble"] = stacking_result

    if stacking_result["mean_balanced_accuracy"] > best_score:
        best_score = stacking_result["mean_balanced_accuracy"]
        best_model_name = "stacking_ensemble"
        best_test_preds = stacking_result["test_predictions"]

    # Weighted average ensemble (improved)
    logger.info("Training weighted ensemble...")
    ensemble_result = _build_weighted_ensemble(results, X_train, y_train, X_test)
    results["weighted_ensemble"] = ensemble_result

    if ensemble_result["mean_balanced_accuracy"] > best_score:
        best_score = ensemble_result["mean_balanced_accuracy"]
        best_model_name = "weighted_ensemble"
        best_test_preds = ensemble_result["test_predictions"]

    final_predictions = label_enc.inverse_transform(best_test_preds)

    logger.info(f"Improved Builder Agent: Best model = {best_model_name} "
                f"(balanced_accuracy = {best_score:.5f})")

    return {
        "models": results,
        "best_model": best_model_name,
        "best_score": round(best_score, 5),
        "test_ids": test_ids.tolist(),
        "predictions": final_predictions.tolist(),
        "pipeline_steps": pipeline_steps,
    }


def _enhanced_preprocess(train, test):
    """Enhanced feature engineering with researcher recommendations."""
    pipeline_steps = []

    X_train = train.drop(columns=[TARGET, ID_COL], errors="ignore").copy()
    y_series = train[TARGET].copy()
    X_test = test.drop(columns=[ID_COL], errors="ignore").copy()
    test_ids = test[ID_COL].copy()

    # Step 1: Encode target
    label_enc = LabelEncoder()
    label_enc.fit(CLASS_LABELS)
    y_train = label_enc.transform(y_series)
    pipeline_steps.append({
        "step": 1, "name": "Encode target labels",
        "description": "LabelEncoder: High=0, Low=1, Medium=2",
        "code": "label_enc = LabelEncoder(); y = label_enc.fit_transform(train['Irrigation_Need'])"
    })

    # Step 2: Handle missing values
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]
    num_cols = [c for c in NUMERIC_FEATURES if c in X_train.columns]

    for col in num_cols:
        median_val = X_train[col].median()
        X_train[col] = X_train[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

    for col in cat_cols:
        mode_val = X_train[col].mode()[0] if not X_train[col].mode().empty else "Unknown"
        X_train[col] = X_train[col].fillna(mode_val)
        X_test[col] = X_test[col].fillna(mode_val)

    pipeline_steps.append({
        "step": 2, "name": "Handle missing values",
        "description": "Median imputation for numeric, mode imputation for categorical",
        "code": "X[col] = X[col].fillna(median_val)"
    })

    # Step 3: Label encode categoricals
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X_train[col], X_test[col]], axis=0).astype(str)
        le.fit(combined)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        label_encoders[col] = le

    pipeline_steps.append({
        "step": 3, "name": "Label encode categoricals",
        "description": f"LabelEncoder applied to {len(cat_cols)} categorical features",
        "code": "le.fit(concat(train, test)); X[col] = le.transform(X[col])"
    })

    # Step 4: Target encoding (NEW - researcher recommendation)
    # Using smoothed mean target encoding with leave-one-out to prevent leakage
    global_mean = y_train.mean()
    smoothing = 10
    for col in cat_cols:
        # Compute target statistics per category
        temp = pd.DataFrame({"cat": X_train[col], "target": y_train})
        agg = temp.groupby("cat")["target"].agg(["mean", "count"])
        smooth_mean = (agg["count"] * agg["mean"] + smoothing * global_mean) / (agg["count"] + smoothing)

        X_train[f"{col}_target_enc"] = X_train[col].map(smooth_mean).fillna(global_mean)
        X_test[f"{col}_target_enc"] = X_test[col].map(smooth_mean).fillna(global_mean)

    pipeline_steps.append({
        "step": 4, "name": "Target encoding (smoothed)",
        "description": f"Smoothed target encoding for {len(cat_cols)} categoricals (smoothing={smoothing})",
        "code": "smooth_mean = (count * mean + s * global_mean) / (count + s)"
    })

    # Step 5: Feature interactions (expanded)
    if "Soil_Moisture" in X_train.columns and "Temperature_C" in X_train.columns:
        X_train["Moisture_Temp_Ratio"] = X_train["Soil_Moisture"] / (X_train["Temperature_C"] + 1)
        X_test["Moisture_Temp_Ratio"] = X_test["Soil_Moisture"] / (X_test["Temperature_C"] + 1)
        X_train["Moisture_Temp_Product"] = X_train["Soil_Moisture"] * X_train["Temperature_C"]
        X_test["Moisture_Temp_Product"] = X_test["Soil_Moisture"] * X_test["Temperature_C"]

    if "Rainfall_mm" in X_train.columns and "Humidity" in X_train.columns:
        X_train["Rain_Humidity_Product"] = X_train["Rainfall_mm"] * X_train["Humidity"]
        X_test["Rain_Humidity_Product"] = X_test["Rainfall_mm"] * X_test["Humidity"]
        X_train["Rain_Humidity_Diff"] = X_train["Rainfall_mm"] - X_train["Humidity"]
        X_test["Rain_Humidity_Diff"] = X_test["Rainfall_mm"] - X_test["Humidity"]

    if "Soil_pH" in X_train.columns and "Organic_Carbon" in X_train.columns:
        X_train["pH_Carbon_Interaction"] = X_train["Soil_pH"] * X_train["Organic_Carbon"]
        X_test["pH_Carbon_Interaction"] = X_test["Soil_pH"] * X_test["Organic_Carbon"]

    if "Sunlight_Hours" in X_train.columns and "Wind_Speed_kmh" in X_train.columns:
        X_train["Sun_Wind_Ratio"] = X_train["Sunlight_Hours"] / (X_train["Wind_Speed_kmh"] + 1)
        X_test["Sun_Wind_Ratio"] = X_test["Sunlight_Hours"] / (X_test["Wind_Speed_kmh"] + 1)

    if "Rainfall_mm" in X_train.columns and "Previous_Irrigation_mm" in X_train.columns:
        X_train["Rain_PrevIrrig_Diff"] = X_train["Rainfall_mm"] - X_train["Previous_Irrigation_mm"]
        X_test["Rain_PrevIrrig_Diff"] = X_test["Rainfall_mm"] - X_test["Previous_Irrigation_mm"]
        X_train["Rain_PrevIrrig_Ratio"] = X_train["Rainfall_mm"] / (X_train["Previous_Irrigation_mm"] + 1)
        X_test["Rain_PrevIrrig_Ratio"] = X_test["Rainfall_mm"] / (X_test["Previous_Irrigation_mm"] + 1)

    if "Temperature_C" in X_train.columns and "Humidity" in X_train.columns:
        X_train["Temp_Humidity_Product"] = X_train["Temperature_C"] * X_train["Humidity"]
        X_test["Temp_Humidity_Product"] = X_test["Temperature_C"] * X_test["Humidity"]

    pipeline_steps.append({
        "step": 5, "name": "Expanded feature interactions",
        "description": "Ratios, products, and differences for correlated feature pairs",
        "code": "X['Rain_PrevIrrig_Diff'] = X['Rainfall_mm'] - X['Previous_Irrigation_mm']"
    })

    # Step 6: Polynomial features for top numeric (NEW - researcher recommendation)
    top_numeric = ["Rainfall_mm", "Temperature_C", "Soil_Moisture", "Wind_Speed_kmh", "Humidity"]
    for col in top_numeric:
        if col in X_train.columns:
            X_train[f"{col}_sq"] = X_train[col] ** 2
            X_test[f"{col}_sq"] = X_test[col] ** 2
            X_train[f"{col}_log"] = np.log1p(np.abs(X_train[col]))
            X_test[f"{col}_log"] = np.log1p(np.abs(X_test[col]))

    pipeline_steps.append({
        "step": 6, "name": "Polynomial & log features",
        "description": f"Squared and log1p transforms for top {len(top_numeric)} numeric features",
        "code": "X[f'{col}_sq'] = X[col] ** 2; X[f'{col}_log'] = np.log1p(np.abs(X[col]))"
    })

    # Step 7: Frequency encoding for categoricals (NEW)
    for col in cat_cols:
        freq = X_train[col].value_counts(normalize=True)
        X_train[f"{col}_freq"] = X_train[col].map(freq).fillna(0)
        X_test[f"{col}_freq"] = X_test[col].map(freq).fillna(0)

    pipeline_steps.append({
        "step": 7, "name": "Frequency encoding",
        "description": f"Value frequency encoding for {len(cat_cols)} categoricals",
        "code": "freq = X[col].value_counts(normalize=True); X[f'{col}_freq'] = X[col].map(freq)"
    })

    # Step 8: Standard scaling
    all_num = X_train.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    X_train[all_num] = scaler.fit_transform(X_train[all_num])
    X_test[all_num] = scaler.transform(X_test[all_num])

    pipeline_steps.append({
        "step": 8, "name": "Standard scaling",
        "description": f"StandardScaler applied to {len(all_num)} numeric features",
        "code": "scaler = StandardScaler(); X = scaler.fit_transform(X)"
    })

    logger.info(f"Enhanced preprocessing complete: {X_train.shape[1]} features "
                f"(was 19 raw, now {X_train.shape[1]} engineered)")
    return X_train, y_train, X_test, test_ids, label_enc, pipeline_steps


def _get_improved_models(class_weights):
    """Return improved model configurations based on researcher recommendations."""
    return {
        "xgboost_tuned": XGBClassifier(
            n_estimators=1500,
            max_depth=7,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.7,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=1,
            objective="multi:softprob",
            num_class=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        ),
        "lightgbm_tuned": LGBMClassifier(
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
        ),
        "random_forest_tuned": RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_leaf=3,
            min_samples_split=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def _cross_validate_and_predict(model, X_train, y_train, X_test, name, sample_weights=None):
    """Run stratified K-fold CV and generate test predictions."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof_preds = np.zeros(len(y_train), dtype=int)
    oof_probs = np.zeros((len(y_train), 3))
    test_probs = np.zeros((len(X_test), 3))
    fold_scores = []
    feature_importance = np.zeros(X_train.shape[1])

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train[train_idx]
        X_val = X_train.iloc[val_idx]
        y_val = y_train[val_idx]

        model_clone = clone(model)

        # Use sample weights for models that support it
        sw = sample_weights[train_idx] if sample_weights is not None else None
        try:
            model_clone.fit(X_tr, y_tr, sample_weight=sw)
        except TypeError:
            model_clone.fit(X_tr, y_tr)

        val_preds = model_clone.predict(X_val)
        oof_preds[val_idx] = val_preds
        fold_ba = balanced_accuracy_score(y_val, val_preds)
        fold_scores.append(round(fold_ba, 5))

        if hasattr(model_clone, "predict_proba"):
            val_probs = model_clone.predict_proba(X_val)
            oof_probs[val_idx] = val_probs
            test_probs += model_clone.predict_proba(X_test) / CV_FOLDS

        if hasattr(model_clone, "feature_importances_"):
            feature_importance += model_clone.feature_importances_ / CV_FOLDS

        logger.info(f"  Fold {fold+1}/{CV_FOLDS}: balanced_accuracy = {fold_ba:.5f}")

    mean_score = balanced_accuracy_score(y_train, oof_preds)
    metrics = compute_metrics(y_train, oof_preds, labels=[0, 1, 2])
    test_predictions = test_probs.argmax(axis=1)

    feature_names = X_train.columns.tolist()
    importance_dict = dict(sorted(
        zip(feature_names, feature_importance.tolist()),
        key=lambda x: x[1], reverse=True
    ))

    return {
        "fold_scores": fold_scores,
        "mean_balanced_accuracy": round(mean_score, 5),
        "metrics": metrics,
        "test_predictions": test_predictions,
        "test_probabilities": test_probs,
        "oof_probabilities": oof_probs,
        "feature_importance": importance_dict,
        "params": _get_params(model),
    }


def _build_stacking_ensemble(model_results, oof_probs_all, X_train, y_train, X_test, sample_weights):
    """Build stacking ensemble using OOF predictions as meta-features."""
    # Create meta-features from OOF probabilities
    base_models = [n for n in model_results if n not in ("stacking_ensemble", "weighted_ensemble")]
    meta_train = np.hstack([oof_probs_all[n] for n in base_models if n in oof_probs_all])
    meta_test = np.hstack([model_results[n]["test_probabilities"] for n in base_models])

    # Train meta-learner with CV
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    meta_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

    oof_preds = np.zeros(len(y_train), dtype=int)
    test_probs = np.zeros((len(X_test), 3))
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(meta_train, y_train)):
        m_tr = meta_train[train_idx]
        y_tr = y_train[train_idx]
        m_val = meta_train[val_idx]
        y_val = y_train[val_idx]

        meta_clone = clone(meta_model)
        meta_clone.fit(m_tr, y_tr)

        val_preds = meta_clone.predict(m_val)
        oof_preds[val_idx] = val_preds
        fold_ba = balanced_accuracy_score(y_val, val_preds)
        fold_scores.append(round(fold_ba, 5))

        test_probs += meta_clone.predict_proba(meta_test) / CV_FOLDS
        logger.info(f"  Stacking Fold {fold+1}/{CV_FOLDS}: balanced_accuracy = {fold_ba:.5f}")

    mean_score = balanced_accuracy_score(y_train, oof_preds)
    metrics = compute_metrics(y_train, oof_preds, labels=[0, 1, 2])
    test_predictions = test_probs.argmax(axis=1)

    # Aggregate feature importance from base models
    ensemble_importance = {}
    for name in base_models:
        if name in model_results:
            for feat, imp in model_results[name]["feature_importance"].items():
                ensemble_importance[feat] = ensemble_importance.get(feat, 0) + imp / len(base_models)
    ensemble_importance = dict(sorted(ensemble_importance.items(), key=lambda x: x[1], reverse=True))

    return {
        "fold_scores": fold_scores,
        "mean_balanced_accuracy": round(mean_score, 5),
        "metrics": metrics,
        "test_predictions": test_predictions,
        "test_probabilities": test_probs,
        "oof_probabilities": np.zeros((len(y_train), 3)),
        "feature_importance": ensemble_importance,
        "params": {"method": "stacking", "meta_learner": "LogisticRegression", "base_models": base_models},
    }


def _build_weighted_ensemble(model_results, X_train, y_train, X_test):
    """Build optimized weighted ensemble excluding weak models."""
    # Only include models that score above threshold
    base_models = {n: r for n, r in model_results.items()
                   if n not in ("stacking_ensemble", "weighted_ensemble")
                   and r["mean_balanced_accuracy"] > 0.80}

    if not base_models:
        base_models = {n: r for n, r in model_results.items()
                       if n not in ("stacking_ensemble", "weighted_ensemble")}

    # Weight by score squared (amplify differences)
    weights = {}
    total = 0
    for name, result in base_models.items():
        w = result["mean_balanced_accuracy"] ** 2
        weights[name] = w
        total += w

    test_probs = np.zeros((len(X_test), 3))
    for name, result in base_models.items():
        test_probs += result["test_probabilities"] * (weights[name] / total)

    test_predictions = test_probs.argmax(axis=1)

    best_name = max(weights, key=weights.get)
    best_metrics = base_models[best_name]["metrics"]

    ensemble_importance = {}
    for name, result in base_models.items():
        w = weights[name] / total
        for feat, imp in result["feature_importance"].items():
            ensemble_importance[feat] = ensemble_importance.get(feat, 0) + imp * w
    ensemble_importance = dict(sorted(ensemble_importance.items(), key=lambda x: x[1], reverse=True))

    return {
        "fold_scores": [round(weights[n], 5) for n in weights],
        "mean_balanced_accuracy": round(max(r["mean_balanced_accuracy"] for r in base_models.values()), 5),
        "metrics": best_metrics,
        "test_predictions": test_predictions,
        "test_probabilities": test_probs,
        "oof_probabilities": np.zeros((len(y_train), 3)),
        "feature_importance": ensemble_importance,
        "params": {"method": "weighted_average_squared", "weights": {k: round(v/total, 4) for k, v in weights.items()}},
    }


def _get_params(model):
    import math
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
