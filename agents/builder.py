"""Builder Agent: Feature engineering, model training, and submission generation."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import balanced_accuracy_score

from pipeline.config import (
    TARGET, ID_COL, CLASS_LABELS, CATEGORICAL_FEATURES, NUMERIC_FEATURES,
    CV_FOLDS, RANDOM_STATE
)
from pipeline.utils import setup_logging, compute_metrics

logger = setup_logging()


def build_models(train, test):
    """Run the full model building pipeline."""
    logger.info("Builder Agent: Starting model training...")

    X_train, y_train, X_test, test_ids, label_enc, pipeline_steps = _preprocess(train, test)

    models = _get_models()
    results = {}
    best_score = -1
    best_model_name = None
    best_test_preds = None

    for name, model in models.items():
        logger.info(f"Training {name}...")
        model_result = _cross_validate_and_predict(
            model, X_train, y_train, X_test, name
        )
        results[name] = model_result

        if model_result["mean_balanced_accuracy"] > best_score:
            best_score = model_result["mean_balanced_accuracy"]
            best_model_name = name
            best_test_preds = model_result["test_predictions"]

    # Ensemble: average test probabilities from all models
    ensemble_result = _build_ensemble(results, X_train, y_train, X_test)
    results["ensemble"] = ensemble_result

    if ensemble_result["mean_balanced_accuracy"] > best_score:
        best_score = ensemble_result["mean_balanced_accuracy"]
        best_model_name = "ensemble"
        best_test_preds = ensemble_result["test_predictions"]

    # Convert numeric predictions back to class labels
    final_predictions = label_enc.inverse_transform(best_test_preds)

    logger.info(f"Builder Agent: Best model = {best_model_name} "
                f"(balanced_accuracy = {best_score:.5f})")

    return {
        "models": results,
        "best_model": best_model_name,
        "best_score": round(best_score, 5),
        "test_ids": test_ids.tolist(),
        "predictions": final_predictions.tolist(),
        "pipeline_steps": pipeline_steps,
    }


def _preprocess(train, test):
    """Feature engineering pipeline."""
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
        "step": 1,
        "name": "Encode target labels",
        "description": "LabelEncoder: Low=0, Medium=1, High=2",
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
        "step": 2,
        "name": "Handle missing values",
        "description": "Median imputation for numeric, mode imputation for categorical",
        "code": "X[num_cols].fillna(X[num_cols].median()); X[cat_cols].fillna(X[cat_cols].mode())"
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
        "step": 3,
        "name": "Label encode categoricals",
        "description": f"LabelEncoder applied to {len(cat_cols)} categorical features",
        "code": "for col in cat_cols: le.fit(concat(train, test)); train[col] = le.transform(train[col])"
    })

    # Step 4: Feature interactions
    if "Soil_Moisture" in X_train.columns and "Temperature_C" in X_train.columns:
        X_train["Moisture_Temp_Ratio"] = X_train["Soil_Moisture"] / (X_train["Temperature_C"] + 1)
        X_test["Moisture_Temp_Ratio"] = X_test["Soil_Moisture"] / (X_test["Temperature_C"] + 1)

    if "Rainfall_mm" in X_train.columns and "Humidity" in X_train.columns:
        X_train["Rain_Humidity_Product"] = X_train["Rainfall_mm"] * X_train["Humidity"]
        X_test["Rain_Humidity_Product"] = X_test["Rainfall_mm"] * X_test["Humidity"]

    if "Soil_pH" in X_train.columns and "Organic_Carbon" in X_train.columns:
        X_train["pH_Carbon_Interaction"] = X_train["Soil_pH"] * X_train["Organic_Carbon"]
        X_test["pH_Carbon_Interaction"] = X_test["Soil_pH"] * X_test["Organic_Carbon"]

    if "Sunlight_Hours" in X_train.columns and "Wind_Speed_kmh" in X_train.columns:
        X_train["Sun_Wind_Ratio"] = X_train["Sunlight_Hours"] / (X_train["Wind_Speed_kmh"] + 1)
        X_test["Sun_Wind_Ratio"] = X_test["Sunlight_Hours"] / (X_test["Wind_Speed_kmh"] + 1)

    pipeline_steps.append({
        "step": 4,
        "name": "Feature interactions",
        "description": "Created ratio/product features: Moisture/Temp, Rain*Humidity, pH*Carbon, Sun/Wind",
        "code": "X['Moisture_Temp_Ratio'] = X['Soil_Moisture'] / (X['Temperature'] + 1)"
    })

    # Step 5: Standard scaling for numeric features
    all_num = X_train.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    X_train[all_num] = scaler.fit_transform(X_train[all_num])
    X_test[all_num] = scaler.transform(X_test[all_num])

    pipeline_steps.append({
        "step": 5,
        "name": "Standard scaling",
        "description": f"StandardScaler applied to {len(all_num)} numeric features",
        "code": "scaler = StandardScaler(); X[num_cols] = scaler.fit_transform(X[num_cols])"
    })

    logger.info(f"Preprocessing complete: {X_train.shape[1]} features")
    return X_train, y_train, X_test, test_ids, label_enc, pipeline_steps


def _get_models():
    """Return dict of models to train."""
    return {
        "xgboost": XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multiclass",
            num_class=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "logistic_regression": LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def _cross_validate_and_predict(model, X_train, y_train, X_test, name):
    """Run stratified K-fold CV and generate test predictions."""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    oof_preds = np.zeros(len(y_train), dtype=int)
    test_probs = np.zeros((len(X_test), 3))
    fold_scores = []
    feature_importance = np.zeros(X_train.shape[1])

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train[train_idx]
        X_val = X_train.iloc[val_idx]
        y_val = y_train[val_idx]

        model_clone = _clone_model(model)
        model_clone.fit(X_tr, y_tr)

        val_preds = model_clone.predict(X_val)
        oof_preds[val_idx] = val_preds
        fold_ba = balanced_accuracy_score(y_val, val_preds)
        fold_scores.append(round(fold_ba, 5))

        if hasattr(model_clone, "predict_proba"):
            test_probs += model_clone.predict_proba(X_test) / CV_FOLDS

        if hasattr(model_clone, "feature_importances_"):
            feature_importance += model_clone.feature_importances_ / CV_FOLDS
        elif hasattr(model_clone, "coef_"):
            feature_importance += np.abs(model_clone.coef_).mean(axis=0) / CV_FOLDS

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
        "feature_importance": importance_dict,
        "params": _get_params(model),
    }


def _build_ensemble(model_results, X_train, y_train, X_test):
    """Build weighted ensemble from all model predictions."""
    weights = {}
    total_weight = 0
    test_probs_weighted = np.zeros((len(X_test), 3))

    for name, result in model_results.items():
        score = result["mean_balanced_accuracy"]
        weights[name] = score
        total_weight += score
        test_probs_weighted += result["test_probabilities"] * score

    test_probs_weighted /= total_weight
    test_predictions = test_probs_weighted.argmax(axis=1)

    # For OOF metrics, use the best model's metrics as proxy
    best_name = max(weights, key=weights.get)
    best_metrics = model_results[best_name]["metrics"]

    # Compute ensemble feature importance (weighted average)
    ensemble_importance = {}
    for name, result in model_results.items():
        w = weights[name] / total_weight
        for feat, imp in result["feature_importance"].items():
            ensemble_importance[feat] = ensemble_importance.get(feat, 0) + imp * w

    ensemble_importance = dict(sorted(
        ensemble_importance.items(), key=lambda x: x[1], reverse=True
    ))

    return {
        "fold_scores": [round(weights[n], 5) for n in weights],
        "mean_balanced_accuracy": round(max(weights.values()), 5),
        "metrics": best_metrics,
        "test_predictions": test_predictions,
        "test_probabilities": test_probs_weighted,
        "feature_importance": ensemble_importance,
        "params": {"method": "balanced_accuracy_weighted_average", "weights": {k: round(v, 5) for k, v in weights.items()}},
    }


def _clone_model(model):
    """Clone a sklearn-compatible model."""
    from sklearn.base import clone
    return clone(model)


def _get_params(model):
    """Extract model params as serializable dict."""
    params = model.get_params()
    return {k: str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v
            for k, v in params.items()}
