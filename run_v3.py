#!/usr/bin/env python3
"""Run V3 pipeline: domain features + deeper trees."""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.data_loader import load_data, save_submission
from pipeline.utils import setup_logging, save_results_json, NumpyEncoder
from agents.researcher import run_eda
from agents.reviewer import review_results
from agents.iterative_builder import build_versioned_models
from agents.orchestrator import _aggregate_results

logger = setup_logging()

VERSION = 3

def main():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info(f"  RUNNING V{VERSION} PIPELINE")
    logger.info("=" * 70)

    # Load data
    logger.info("\n[1/5] Loading data...")
    train, test, sample_sub = load_data()
    logger.info(f"  Train: {train.shape}, Test: {test.shape}")

    # Load previous results for comparison
    logger.info("\n[2/5] Loading V2 results for comparison...")
    with open("outputs/results.json") as f:
        prev = json.load(f)
    logger.info(f"  Previous best: {prev['best_model']} = {prev['best_score']}")

    # Build V3 models
    logger.info(f"\n[3/5] Building V{VERSION} models...")
    results = build_versioned_models(train, test, version=VERSION, prev_results=prev)

    # Review
    logger.info(f"\n[4/5] Reviewing V{VERSION} results...")
    eda = run_eda(train, test)
    review = review_results(train, test, results, eda)
    logger.info(f"  Verdict: {review['verdict']}")
    for a in review["approvals"]:
        logger.info(f"    ✓ {a}")
    for w in review["warnings"]:
        logger.info(f"    ⚠ {w}")

    # Compare & save
    logger.info(f"\n[5/5] Comparing V{VERSION} vs V2...")
    elapsed = round(time.time() - t0, 1)
    dashboard = _aggregate_results(eda, results, review, elapsed)

    old_score = prev["best_score"]
    new_score = results["best_score"]
    delta = new_score - old_score

    # Build version history
    prev_history = prev.get("version_history", [])
    if not prev_history:
        # Seed with V1 and V2
        prev_history = [
            {"version": 1, "score": 0.96207, "best_model": "lightgbm",
             "changes": ["Baseline: 4 models + weighted ensemble", "23 features"]},
            {"version": 2, "score": prev["best_score"], "best_model": prev["best_model"],
             "changes": prev.get("improvement", {}).get("recommendations_applied", [])},
        ]

    prev_history.append({
        "version": VERSION,
        "score": new_score,
        "best_model": results["best_model"],
        "changes": results["version_config"]["changes"],
    })
    dashboard["version_history"] = prev_history

    dashboard["improvement"] = {
        "previous_best_model": prev["best_model"],
        "previous_best_score": old_score,
        "improved_best_model": results["best_model"],
        "improved_best_score": new_score,
        "score_change": round(delta, 5),
        "recommendations_applied": results["version_config"]["changes"],
        "research_analysis": prev.get("improvement", {}).get("research_analysis", {}),
    }

    logger.info(f"\n{'='*70}")
    logger.info(f"  V{VERSION} RESULTS")
    logger.info(f"{'='*70}")
    logger.info(f"  Previous: {prev['best_model']:25s} BA = {old_score:.5f}")
    logger.info(f"  V{VERSION}:      {results['best_model']:25s} BA = {new_score:.5f}")
    logger.info(f"  Delta:    {'↑' if delta > 0 else '↓'} {abs(delta):.5f}")
    logger.info(f"\n  Model breakdown:")
    for name, r in results["models"].items():
        logger.info(f"    {name:25s}: {r['mean_balanced_accuracy']:.5f}")

    save_results_json(dashboard)

    sub_path = save_submission(results["test_ids"], results["predictions"],
                               filename=f"submission_v{VERSION}.csv")

    logger.info(f"\n  Pipeline complete in {elapsed}s")
    logger.info(f"  Results: outputs/results.json")
    logger.info(f"  Submission: {sub_path}")

if __name__ == "__main__":
    main()
