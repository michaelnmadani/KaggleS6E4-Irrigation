"""Iterative Builder Agent: Configurable builder that applies version-specific improvements."""

import math
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
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

    X_train, y_train, X_test, test_ids, label_enc, pipeline_steps = _preprocess(train, test, version)

    # Class weights
    class_counts = np.bincount(y_train)
    total = len(y_train)
    class_weights = {i: total / (len(class_counts) * c) for i, c in enumerate(class_counts)}
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
        result = _cv_predict(model, X_train, y_train, X_test, name, sample_weights, n_folds=n_folds)
        results[name] = result
        oof_probs_all[name] = result["oof_probabilities"]
        if result["mean_balanced_accuracy"] > best_score:
            best_score = result["mean_balanced_accuracy"]
            best_model_name = name
            best_test_preds = result["test_predictions"]

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
        we_result = _weighted_ensemble(results, X_test)
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

        steps.append({"step": "3b", "name": "V4 domain features",
                       "description": "evapotranspiration proxy, soil stress, total water input",
                       "code": "evapotrans = Temp * Sunlight / (Humidity + 1)"})

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

    # Standard scaling
    all_num = X_train.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    X_train[all_num] = scaler.fit_transform(X_train[all_num])
    X_test[all_num] = scaler.transform(X_test[all_num])
    steps.append({"step": 9, "name": "Standard scaling",
                   "description": f"StandardScaler on {len(all_num)} features"})

    logger.info(f"  Preprocessing done: {X_train.shape[1]} features")
    return X_train, y_train, X_test, test_ids, label_enc, steps


# ═══════════════════════════════════════════════════════════════════════
#  MODELS
# ═══════════════════════════════════════════════════════════════════════

def _get_models(version, class_weights):
    models = {}

    if version == 4:
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

def _cv_predict(model, X_train, y_train, X_test, name, sample_weights=None, n_folds=None):
    if n_folds is None:
        n_folds = CV_FOLDS
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(y_train), dtype=int)
    oof_probs = np.zeros((len(y_train), 3))
    test_probs = np.zeros((len(X_test), 3))
    fold_scores = []
    feat_imp = np.zeros(X_train.shape[1])

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train[tr_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train[val_idx]
        m = clone(model)
        sw = sample_weights[tr_idx] if sample_weights is not None else None
        try:
            m.fit(X_tr, y_tr, sample_weight=sw)
        except TypeError:
            m.fit(X_tr, y_tr)

        preds = m.predict(X_val)
        oof_preds[val_idx] = preds
        ba = balanced_accuracy_score(y_val, preds)
        fold_scores.append(round(ba, 5))

        if hasattr(m, "predict_proba"):
            oof_probs[val_idx] = m.predict_proba(X_val)
            test_probs += m.predict_proba(X_test) / n_folds
        if hasattr(m, "feature_importances_"):
            feat_imp += m.feature_importances_ / n_folds

        logger.info(f"  Fold {fold+1}/{n_folds}: BA = {ba:.5f}")

    mean_ba = balanced_accuracy_score(y_train, oof_preds)
    metrics = compute_metrics(y_train, oof_preds, labels=[0, 1, 2])
    fi = dict(sorted(zip(X_train.columns.tolist(), feat_imp.tolist()), key=lambda x: x[1], reverse=True))

    return {
        "fold_scores": fold_scores,
        "mean_balanced_accuracy": round(mean_ba, 5),
        "metrics": metrics,
        "test_predictions": test_probs.argmax(axis=1),
        "test_probabilities": test_probs,
        "oof_probabilities": oof_probs,
        "feature_importance": fi,
        "params": _safe_params(model),
    }


def _stacking_ensemble(results, oof_probs, X_train, y_train, X_test, sw, version):
    base = [n for n in results if n not in ("stacking_ensemble", "weighted_ensemble")]
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


def _weighted_ensemble(results, X_test):
    base = {n: r for n, r in results.items()
            if n not in ("stacking_ensemble", "weighted_ensemble")
            and r["mean_balanced_accuracy"] > 0.90}
    if not base:
        base = {n: r for n, r in results.items() if n not in ("stacking_ensemble", "weighted_ensemble")}

    weights = {n: r["mean_balanced_accuracy"] ** 3 for n, r in base.items()}  # cube weights for V3
    total_w = sum(weights.values())

    test_probs = np.zeros((len(X_test), 3))
    for n, r in base.items():
        test_probs += r["test_probabilities"] * (weights[n] / total_w)

    best_n = max(base, key=lambda n: base[n]["mean_balanced_accuracy"])
    fi = {}
    for n, r in base.items():
        w = weights[n] / total_w
        for feat, imp in r["feature_importance"].items():
            fi[feat] = fi.get(feat, 0) + imp * w
    fi = dict(sorted(fi.items(), key=lambda x: x[1], reverse=True))

    return {
        "fold_scores": [round(weights[n], 5) for n in weights],
        "mean_balanced_accuracy": round(max(r["mean_balanced_accuracy"] for r in base.values()), 5),
        "metrics": base[best_n]["metrics"],
        "test_predictions": test_probs.argmax(axis=1),
        "test_probabilities": test_probs,
        "oof_probabilities": np.zeros((len(X_test), 3)),
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
