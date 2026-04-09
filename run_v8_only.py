#!/usr/bin/env python3
"""Run V8 only — full 5-fold CV, all models, no shortcuts."""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.data_loader import load_data, save_submission
from pipeline.utils import setup_logging, save_results_json, NumpyEncoder
from agents.researcher import run_eda
from agents.reviewer import review_results
from agents.iterative_builder import build_versioned_models
from agents.orchestrator import _aggregate_results

logger = setup_logging()


def main():
    t0 = time.time()
    version = 8

    logger.info("Loading data...")
    train, test, sample_sub = load_data()
    logger.info(f"  Train: {train.shape}, Test: {test.shape}")

    logger.info("Running EDA...")
    eda = run_eda(train, test)

    # Load previous results
    with open("outputs/results.json") as f:
        prev = json.load(f)
    logger.info(f"Previous best: {prev['best_model']} = {prev['best_score']}")

    # Build V8 models (full mode)
    logger.info(f"\n[BUILD] Building V{version} models (full 5-fold CV, all models)...")
    logger.info("  Key V8 changes: HIGH class targeting, num_leaves=80, min_child_samples=30, finer blend grid")
    results = build_versioned_models(train, test, version=version, prev_results=prev, fast=False)

    # Review
    logger.info(f"\nReviewing V{version}...")
    review = review_results(train, test, results, eda)
    logger.info(f"Verdict: {review['verdict']}")
    for a in review["approvals"]:
        logger.info(f"  ✓ {a}")
    for w in review["warnings"]:
        logger.info(f"  ⚠ {w}")

    # Save results
    elapsed = round(time.time() - t0, 1)
    dashboard = _aggregate_results(eda, results, review, elapsed)

    old_score = prev["best_score"]
    new_score = results["best_score"]
    delta = new_score - old_score

    # Build version history
    prev_history = prev.get("version_history", [])
    prev_history.append({
        "version": version,
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
    }

    save_results_json(dashboard)
    save_submission(results["test_ids"], results["predictions"], filename="submission_v8.csv")

    logger.info(f"\n{'='*70}")
    logger.info(f"  V{version} RESULTS")
    logger.info(f"{'='*70}")
    logger.info(f"  Previous: {prev['best_model']:25s} BA = {old_score:.5f}")
    logger.info(f"  V{version}:      {results['best_model']:25s} BA = {new_score:.5f}")
    logger.info(f"  Delta:    {'↑' if delta > 0 else '↓'} {abs(delta):.5f}")
    logger.info(f"\n  Model breakdown:")
    for name, r in results["models"].items():
        logger.info(f"    {name:25s}: {r['mean_balanced_accuracy']:.5f}")
    logger.info(f"\n  V{version} complete in {elapsed}s ({elapsed/60:.1f} min)")
    logger.info("DONE")


if __name__ == "__main__":
    main()
