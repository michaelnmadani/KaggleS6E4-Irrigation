"""Orchestrator Agent: Coordinates the full pipeline."""

import time
from pipeline.data_loader import load_data, save_submission, submit_predictions
from pipeline.config import COMPETITION_NAME, COMPETITION_SLUG, TARGET, CLASS_LABELS, METRIC
from pipeline.utils import setup_logging, save_results_json
from agents.researcher import run_eda
from agents.builder import build_models
from agents.reviewer import review_results

logger = setup_logging()


def run_pipeline(auto_submit=True):
    """Run the full ML pipeline: EDA -> Build -> Review -> Submit."""
    logger.info("=" * 60)
    logger.info(f"Orchestrator: Starting pipeline for {COMPETITION_NAME}")
    logger.info("=" * 60)
    start_time = time.time()

    # Step 1: Load data
    logger.info("Step 1/4: Loading data via Kaggle API...")
    train, test, sample_sub = load_data()
    logger.info(f"  Train: {train.shape}, Test: {test.shape}")

    # Step 2: Researcher - EDA
    logger.info("Step 2/4: Running EDA (Researcher Agent)...")
    eda_results = run_eda(train, test)

    # Step 3: Builder - Model Training
    logger.info("Step 3/4: Training models (Builder Agent)...")
    builder_results = build_models(train, test)

    # Step 4: Reviewer - Quality Check
    logger.info("Step 4/4: Reviewing results (Reviewer Agent)...")
    review = review_results(train, test, builder_results, eda_results)

    # Aggregate results for dashboard
    elapsed = round(time.time() - start_time, 1)
    results = _aggregate_results(eda_results, builder_results, review, elapsed)

    # Save results JSON for React dashboard
    save_results_json(results)

    # Save submission
    submission_path = save_submission(
        builder_results["test_ids"],
        builder_results["predictions"],
        filename="submission.csv"
    )

    # Auto-submit if approved
    if auto_submit and review["verdict"] != "REJECT":
        try:
            logger.info("Submitting predictions to Kaggle...")
            submit_predictions(
                submission_path,
                message=f"AutoKaggle: {builder_results['best_model']} "
                        f"(BA={builder_results['best_score']:.5f})"
            )
        except Exception as e:
            logger.warning(f"Kaggle submission failed ({e}). Submission saved locally.")
    elif review["verdict"] == "REJECT":
        logger.warning("Submission rejected by Reviewer. Fix issues before submitting.")
    else:
        logger.info(f"Submission saved to {submission_path} (auto-submit disabled)")

    logger.info(f"Pipeline complete in {elapsed}s")
    return results


def _aggregate_results(eda, builder, review, elapsed):
    """Aggregate all results into a single dict for the dashboard."""
    # Clean up model results (remove non-serializable numpy arrays)
    models_clean = {}
    for name, result in builder["models"].items():
        models_clean[name] = {
            "fold_scores": result["fold_scores"],
            "mean_balanced_accuracy": result["mean_balanced_accuracy"],
            "metrics": result["metrics"],
            "feature_importance": result["feature_importance"],
            "params": result["params"],
        }

    return {
        "competition": {
            "name": COMPETITION_NAME,
            "slug": COMPETITION_SLUG,
            "metric": METRIC,
            "target": TARGET,
            "classes": CLASS_LABELS,
        },
        "dataset": eda["dataset_stats"],
        "eda": {
            "class_distribution": eda["class_distribution"],
            "missing_values": eda["missing_values"],
            "numeric_stats": eda["numeric_stats"],
            "categorical_stats": eda["categorical_stats"],
            "correlations": eda["correlations"],
            "sample_data": eda["sample_data"],
            "feature_categories": eda["feature_categories"],
        },
        "pipeline": {
            "steps": builder["pipeline_steps"],
        },
        "models": models_clean,
        "best_model": builder["best_model"],
        "best_score": builder["best_score"],
        "review": review,
        "elapsed_seconds": elapsed,
    }
