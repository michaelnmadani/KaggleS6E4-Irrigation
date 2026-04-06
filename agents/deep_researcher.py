"""Deep Researcher Agent: Analyzes current results and recommends improvements."""

import numpy as np
import pandas as pd
from pipeline.config import TARGET, CLASS_LABELS, CATEGORICAL_FEATURES, NUMERIC_FEATURES
from pipeline.utils import setup_logging

logger = setup_logging()


def research_improvements(train, test, current_results):
    """Analyze current pipeline results and recommend specific improvements."""
    logger.info("Deep Researcher Agent: Analyzing results for improvements...")

    recommendations = []
    analysis = {}

    # 1. Analyze class imbalance
    class_dist = train[TARGET].value_counts(normalize=True)
    analysis["class_imbalance"] = {
        label: round(pct, 4) for label, pct in class_dist.items()
    }
    minority_class = class_dist.idxmin()
    minority_pct = class_dist.min()

    if minority_pct < 0.1:
        recommendations.append({
            "id": "class_weight",
            "priority": "HIGH",
            "category": "class_imbalance",
            "title": "Address severe class imbalance",
            "detail": f"Class '{minority_class}' is only {minority_pct:.1%} of data. "
                      f"Use class_weight='balanced' or sample_weight to boost minority performance.",
            "implementation": "Add class_weight='balanced' to all models, or compute sample weights"
        })

    # 2. Analyze feature interactions potential
    num_cols = [c for c in NUMERIC_FEATURES if c in train.columns]
    if len(num_cols) >= 2:
        corr = train[num_cols].corr()
        high_corr_pairs = []
        for i in range(len(num_cols)):
            for j in range(i+1, len(num_cols)):
                c = abs(corr.iloc[i, j])
                if c > 0.3:
                    high_corr_pairs.append((num_cols[i], num_cols[j], round(c, 3)))

        analysis["high_correlation_pairs"] = high_corr_pairs
        if high_corr_pairs:
            recommendations.append({
                "id": "more_interactions",
                "priority": "MEDIUM",
                "category": "feature_engineering",
                "title": "Add more feature interactions",
                "detail": f"Found {len(high_corr_pairs)} correlated feature pairs. "
                          f"Create ratio, product, and difference features for these.",
                "pairs": high_corr_pairs,
                "implementation": "Create pairwise ratio/product/diff for correlated features"
            })

    # 3. Analyze per-class performance gaps
    current_models = current_results.get("models", {})
    best_model_name = current_results.get("best_model", "")
    best_model = current_models.get(best_model_name, {})
    cm = best_model.get("metrics", {}).get("confusion_matrix", [])
    report = best_model.get("metrics", {}).get("classification_report", {})

    if cm and len(cm) == 3:
        analysis["per_class_accuracy"] = {}
        for i, label in enumerate(CLASS_LABELS):
            total = sum(cm[i])
            correct = cm[i][i] if total > 0 else 0
            acc = correct / total if total > 0 else 0
            analysis["per_class_accuracy"][label] = round(acc, 4)

        worst_class = min(analysis["per_class_accuracy"], key=analysis["per_class_accuracy"].get)
        worst_acc = analysis["per_class_accuracy"][worst_class]
        if worst_acc < 0.90:
            recommendations.append({
                "id": "focus_worst_class",
                "priority": "HIGH",
                "category": "model_tuning",
                "title": f"Improve prediction for '{worst_class}' class",
                "detail": f"Class '{worst_class}' has only {worst_acc:.1%} accuracy. "
                          f"This drags down balanced accuracy significantly.",
                "implementation": "Increase class weight for this class, add targeted features"
            })

    # 4. Analyze logistic regression gap
    lr_score = current_models.get("logistic_regression", {}).get("mean_balanced_accuracy", 0)
    best_score = current_results.get("best_score", 0)
    if lr_score < best_score * 0.8:
        recommendations.append({
            "id": "drop_lr",
            "priority": "LOW",
            "category": "model_selection",
            "title": "Replace logistic regression in ensemble",
            "detail": f"LR scores {lr_score:.4f} vs best {best_score:.4f}. "
                      f"It hurts ensemble quality. Replace with ExtraTrees or GradientBoosting.",
            "implementation": "Replace LogisticRegression with ExtraTreesClassifier or GradientBoostingClassifier"
        })

    # 5. Analyze feature engineering opportunities
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in train.columns]
    for col in cat_cols:
        nunique = train[col].nunique()
        if nunique > 2:
            analysis.setdefault("categorical_cardinality", {})[col] = nunique

    recommendations.append({
        "id": "target_encoding",
        "priority": "HIGH",
        "category": "feature_engineering",
        "title": "Add target encoding for categorical features",
        "detail": "Target encoding captures the relationship between categorical values and the target. "
                  "Use smoothed target encoding with CV to avoid leakage.",
        "implementation": "For each categorical, compute mean target per category with smoothing"
    })

    # 6. Recommend hyperparameter tuning
    recommendations.append({
        "id": "tune_lgbm",
        "priority": "HIGH",
        "category": "hyperparameter_tuning",
        "title": "Tune LightGBM hyperparameters",
        "detail": "Current LightGBM uses default-like params. Key params to tune: "
                  "n_estimators (1000-3000), learning_rate (0.01-0.03), "
                  "num_leaves (31-127), min_child_samples (20-100), "
                  "reg_alpha, reg_lambda, subsample, colsample_bytree.",
        "implementation": "Increase n_estimators, lower learning_rate, tune regularization"
    })

    # 7. Analyze if more features could help
    existing_interaction_features = [c for c in ["Moisture_Temp_Ratio", "Rain_Humidity_Product",
                                                   "pH_Carbon_Interaction", "Sun_Wind_Ratio"]]
    recommendations.append({
        "id": "poly_features",
        "priority": "MEDIUM",
        "category": "feature_engineering",
        "title": "Add polynomial and binning features",
        "detail": "Create squared terms for top numeric features, "
                  "bin continuous variables into categories (e.g., temperature ranges), "
                  "and add frequency encoding for categoricals.",
        "implementation": "Add x^2 for top 5 numeric features, pd.cut for binning, value_counts encoding"
    })

    # 8. Recommend better ensemble
    recommendations.append({
        "id": "stacking",
        "priority": "MEDIUM",
        "category": "ensemble",
        "title": "Use stacking instead of weighted average",
        "detail": "Current ensemble uses BA-weighted averaging. "
                  "Stacking with a meta-learner (LR or LightGBM) on OOF predictions "
                  "can capture complementary strengths of base models.",
        "implementation": "Use StackingClassifier with LightGBM/XGB/RF as base, LR as meta"
    })

    # Sort by priority
    priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    recommendations.sort(key=lambda x: priority_order.get(x["priority"], 3))

    logger.info(f"Deep Researcher Agent: Found {len(recommendations)} recommendations")
    for r in recommendations:
        logger.info(f"  [{r['priority']}] {r['title']}")

    return {
        "analysis": analysis,
        "recommendations": recommendations,
        "current_best_score": current_results.get("best_score", 0),
        "current_best_model": current_results.get("best_model", ""),
    }
