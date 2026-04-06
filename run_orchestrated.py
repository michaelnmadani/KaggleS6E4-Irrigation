#!/usr/bin/env python3
"""Orchestrated improvement pipeline: Review → Research → Build → Review → Compare."""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.data_loader import load_data, save_submission
from pipeline.utils import setup_logging, save_results_json, NumpyEncoder
from agents.researcher import run_eda
from agents.reviewer import review_results
from agents.deep_researcher import research_improvements
from agents.improved_builder import build_improved_models
from agents.orchestrator import _aggregate_results

logger = setup_logging()


def main():
    total_start = time.time()

    logger.info("=" * 70)
    logger.info("ORCHESTRATED IMPROVEMENT PIPELINE")
    logger.info("=" * 70)

    # ── Step 1: Load data ─────────────────────────────────────────────
    logger.info("\n[STEP 1/6] Loading data...")
    train, test, sample_sub = load_data()
    logger.info(f"  Train: {train.shape}, Test: {test.shape}")

    # ── Step 2: Load & review current results ─────────────────────────
    logger.info("\n[STEP 2/6] REVIEWER AGENT: Reviewing current results...")
    with open("outputs/results.json") as f:
        current_results = json.load(f)

    logger.info(f"  Current best: {current_results['best_model']} = {current_results['best_score']:.5f}")
    logger.info(f"  Current verdict: {current_results['review']['verdict']}")

    # Detailed review output
    review = current_results["review"]
    logger.info(f"  Approvals ({len(review['approvals'])}):")
    for a in review["approvals"]:
        logger.info(f"    ✓ {a}")
    logger.info(f"  Warnings ({len(review['warnings'])}):")
    for w in review["warnings"]:
        logger.info(f"    ⚠ {w}")
    logger.info(f"  Issues ({len(review['issues'])}):")
    for i in review["issues"]:
        logger.info(f"    ✗ {i}")

    # ── Step 3: Deep research for improvements ────────────────────────
    logger.info("\n[STEP 3/6] DEEP RESEARCHER AGENT: Analyzing for improvements...")
    research = research_improvements(train, test, current_results)

    logger.info(f"\n  Analysis:")
    logger.info(f"    Class imbalance: {research['analysis'].get('class_imbalance', {})}")
    if "per_class_accuracy" in research["analysis"]:
        logger.info(f"    Per-class accuracy: {research['analysis']['per_class_accuracy']}")
    if "high_correlation_pairs" in research["analysis"]:
        logger.info(f"    Correlated pairs: {len(research['analysis']['high_correlation_pairs'])}")

    logger.info(f"\n  Recommendations ({len(research['recommendations'])}):")
    for r in research["recommendations"]:
        logger.info(f"    [{r['priority']:6s}] {r['title']}")
        logger.info(f"            → {r['implementation']}")

    # ── Step 4: Build improved models ─────────────────────────────────
    logger.info("\n[STEP 4/6] IMPROVED BUILDER AGENT: Implementing recommendations...")
    build_start = time.time()
    improved_results = build_improved_models(train, test, research["recommendations"])
    build_elapsed = round(time.time() - build_start, 1)
    logger.info(f"  Build time: {build_elapsed}s")

    # ── Step 5: Review improved results ───────────────────────────────
    logger.info("\n[STEP 5/6] REVIEWER AGENT: Reviewing improved results...")
    eda_results = run_eda(train, test)
    improved_review = review_results(train, test, improved_results, eda_results)

    logger.info(f"  Improved verdict: {improved_review['verdict']}")
    for a in improved_review["approvals"]:
        logger.info(f"    ✓ {a}")
    for w in improved_review["warnings"]:
        logger.info(f"    ⚠ {w}")

    # ── Step 6: Compare & save ────────────────────────────────────────
    logger.info("\n[STEP 6/6] ORCHESTRATOR: Comparing results...")

    old_score = current_results["best_score"]
    new_score = improved_results["best_score"]
    improvement = new_score - old_score

    logger.info(f"\n{'='*70}")
    logger.info(f"  COMPARISON")
    logger.info(f"{'='*70}")
    logger.info(f"  Previous: {current_results['best_model']:25s} BA = {old_score:.5f}")
    logger.info(f"  Improved: {improved_results['best_model']:25s} BA = {new_score:.5f}")
    logger.info(f"  Change:   {'↑' if improvement > 0 else '↓'} {abs(improvement):.5f} "
                f"({'BETTER' if improvement > 0 else 'WORSE'})")

    logger.info(f"\n  Model breakdown:")
    for name, result in improved_results["models"].items():
        old_name_map = {
            "xgboost_tuned": "xgboost",
            "lightgbm_tuned": "lightgbm",
            "random_forest_tuned": "random_forest",
        }
        old_equiv = old_name_map.get(name, "")
        old_s = current_results["models"].get(old_equiv, {}).get("mean_balanced_accuracy", None)
        delta = ""
        if old_s is not None:
            d = result["mean_balanced_accuracy"] - old_s
            delta = f"  (was {old_s:.5f}, {'↑' if d > 0 else '↓'}{abs(d):.5f})"
        logger.info(f"    {name:25s}: {result['mean_balanced_accuracy']:.5f}{delta}")

    # Save improved results
    total_elapsed = round(time.time() - total_start, 1)
    dashboard_results = _aggregate_results(eda_results, improved_results, improved_review, total_elapsed)

    # Add improvement metadata
    dashboard_results["improvement"] = {
        "previous_best_model": current_results["best_model"],
        "previous_best_score": old_score,
        "improved_best_model": improved_results["best_model"],
        "improved_best_score": new_score,
        "score_change": round(improvement, 5),
        "recommendations_applied": [r["title"] for r in research["recommendations"]],
        "research_analysis": research["analysis"],
    }

    save_results_json(dashboard_results)

    # Save submission
    sub_path = save_submission(
        improved_results["test_ids"],
        improved_results["predictions"],
        filename="submission_improved.csv"
    )

    # Try to submit to Kaggle
    try:
        from pipeline.data_loader import submit_predictions
        submit_predictions(sub_path, message=f"Improved: {improved_results['best_model']} BA={new_score:.5f}")
    except Exception as e:
        logger.warning(f"Kaggle submission failed ({e}). Saved locally.")

    logger.info(f"\n{'='*70}")
    logger.info(f"  PIPELINE COMPLETE in {total_elapsed}s")
    logger.info(f"  Best model: {improved_results['best_model']} (BA={new_score:.5f})")
    logger.info(f"  Results: outputs/results.json")
    logger.info(f"  Submission: {sub_path}")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()
