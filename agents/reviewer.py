"""Reviewer Agent: Quality assurance and validation."""

import numpy as np
from pipeline.config import TARGET, CLASS_LABELS, ID_COL
from pipeline.utils import setup_logging

logger = setup_logging()


def review_results(train, test, builder_results, eda_results):
    """Review and validate pipeline results."""
    logger.info("Reviewer Agent: Starting review...")

    issues = []
    warnings = []
    approvals = []

    # Check 1: Submission format
    predictions = builder_results["predictions"]
    test_ids = builder_results["test_ids"]

    if len(predictions) != len(test):
        issues.append(f"Prediction count ({len(predictions)}) != test rows ({len(test)})")
    else:
        approvals.append(f"Prediction count matches test set: {len(predictions)} rows")

    if len(test_ids) != len(test):
        issues.append(f"Test ID count ({len(test_ids)}) != test rows ({len(test)})")
    else:
        approvals.append(f"Test ID count matches: {len(test_ids)}")

    # Check 2: Valid class labels
    unique_preds = set(predictions)
    valid_labels = set(CLASS_LABELS)
    invalid = unique_preds - valid_labels
    if invalid:
        issues.append(f"Invalid class labels in predictions: {invalid}")
    else:
        approvals.append(f"All predictions use valid labels: {unique_preds}")

    # Check 3: Prediction distribution vs training
    train_dist = train[TARGET].value_counts(normalize=True).to_dict()
    pred_dist = {}
    for label in CLASS_LABELS:
        pred_dist[label] = round(sum(1 for p in predictions if p == label) / len(predictions), 4)

    dist_diffs = {}
    for label in CLASS_LABELS:
        train_pct = train_dist.get(label, 0)
        pred_pct = pred_dist.get(label, 0)
        diff = abs(train_pct - pred_pct)
        dist_diffs[label] = round(diff, 4)
        if diff > 0.15:
            warnings.append(f"Large distribution shift for '{label}': "
                          f"train={train_pct:.2%} vs pred={pred_pct:.2%}")

    if not any(d > 0.15 for d in dist_diffs.values()):
        approvals.append("Prediction distribution is consistent with training data")

    # Check 4: Model performance
    best_model = builder_results["best_model"]
    best_score = builder_results["best_score"]

    if best_score < 0.5:
        issues.append(f"Best balanced accuracy ({best_score:.4f}) is below 0.5 (worse than random)")
    elif best_score < 0.7:
        warnings.append(f"Best balanced accuracy ({best_score:.4f}) is moderate - consider more tuning")
    else:
        approvals.append(f"Best model '{best_model}' achieves good balanced accuracy: {best_score:.4f}")

    # Check 5: Model comparison
    models = builder_results["models"]
    scores = {name: result["mean_balanced_accuracy"] for name, result in models.items()}
    score_range = max(scores.values()) - min(scores.values())
    if score_range < 0.001:
        warnings.append("All models perform nearly identically - possible data issue")

    # Check 6: Feature importance sanity
    best_importance = models[best_model]["feature_importance"]
    top_features = list(best_importance.keys())[:5]
    approvals.append(f"Top 5 features: {', '.join(top_features)}")

    # Check 7: Missing values
    if eda_results["missing_values"]["train_total"] > 0:
        warnings.append(f"Training data had {eda_results['missing_values']['train_total']} missing values (handled by imputation)")
    else:
        approvals.append("No missing values in training data")

    # Final verdict
    if issues:
        verdict = "REJECT"
    elif len(warnings) > 2:
        verdict = "CONDITIONAL_APPROVE"
    else:
        verdict = "APPROVE"

    review = {
        "verdict": verdict,
        "issues": issues,
        "warnings": warnings,
        "approvals": approvals,
        "model_scores": scores,
        "prediction_distribution": pred_dist,
        "training_distribution": {k: round(v, 4) for k, v in train_dist.items()},
        "distribution_diffs": dist_diffs,
        "best_model": best_model,
        "best_score": best_score,
    }

    logger.info(f"Reviewer Agent: Verdict = {verdict} "
                f"({len(issues)} issues, {len(warnings)} warnings, {len(approvals)} approvals)")

    return review
