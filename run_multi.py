#!/usr/bin/env python3
"""Run the pipeline multiple times with different random seeds."""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.config import RANDOM_STATE, CV_FOLDS
from pipeline.data_loader import load_data, save_submission
from pipeline.utils import setup_logging, save_results_json, NumpyEncoder
from agents.researcher import run_eda
from agents.builder import build_models
from agents.reviewer import review_results
import pipeline.config as config

logger = setup_logging()


def run_single(seed, run_id, train, test):
    """Run a single pipeline iteration with a specific seed."""
    logger.info(f"\n{'='*60}")
    logger.info(f"RUN {run_id} (seed={seed})")
    logger.info(f"{'='*60}")
    start = time.time()

    # Override the random state for this run
    config.RANDOM_STATE = seed

    # Build models (EDA is the same each run, skip it after first)
    builder_results = build_models(train, test)

    elapsed = round(time.time() - start, 1)

    # Save submission for this run
    sub_path = save_submission(
        builder_results["test_ids"],
        builder_results["predictions"],
        filename=f"submission_run{run_id:02d}_seed{seed}.csv"
    )

    return {
        "run_id": run_id,
        "seed": seed,
        "best_model": builder_results["best_model"],
        "best_score": builder_results["best_score"],
        "model_scores": {
            name: result["mean_balanced_accuracy"]
            for name, result in builder_results["models"].items()
        },
        "submission_file": sub_path,
        "elapsed_seconds": elapsed,
        "builder_results": builder_results,
    }


def main():
    n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    logger.info(f"Starting {n_runs} pipeline runs...")

    # Load data once
    train, test, sample_sub = load_data()
    logger.info(f"Train: {train.shape}, Test: {test.shape}")

    # Run EDA once
    eda_results = run_eda(train, test)

    # Run with different seeds
    seeds = [42 + i * 7 for i in range(n_runs)]
    all_runs = []
    best_overall_score = -1
    best_overall_run = None

    for i, seed in enumerate(seeds):
        result = run_single(seed, i + 1, train, test)
        all_runs.append(result)

        if result["best_score"] > best_overall_score:
            best_overall_score = result["best_score"]
            best_overall_run = result

        logger.info(f"  Run {i+1}/{n_runs}: {result['best_model']} = {result['best_score']:.5f} ({result['elapsed_seconds']}s)")

    # Review the best run
    config.RANDOM_STATE = best_overall_run["seed"]
    review = review_results(train, test, best_overall_run["builder_results"], eda_results)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"ALL {n_runs} RUNS COMPLETE")
    logger.info(f"{'='*60}")
    scores = [r["best_score"] for r in all_runs]
    logger.info(f"Score range: {min(scores):.5f} - {max(scores):.5f}")
    logger.info(f"Mean score: {sum(scores)/len(scores):.5f}")
    logger.info(f"Best: Run {best_overall_run['run_id']} (seed={best_overall_run['seed']}) = {best_overall_score:.5f}")

    # Save best run's results.json for the dashboard
    from agents.orchestrator import _aggregate_results
    total_elapsed = sum(r["elapsed_seconds"] for r in all_runs)
    dashboard_results = _aggregate_results(
        eda_results, best_overall_run["builder_results"], review, round(total_elapsed, 1)
    )

    # Add multi-run summary to results
    dashboard_results["multi_run"] = {
        "total_runs": n_runs,
        "seeds": seeds,
        "scores": [r["best_score"] for r in all_runs],
        "best_models": [r["best_model"] for r in all_runs],
        "mean_score": round(sum(scores) / len(scores), 5),
        "min_score": round(min(scores), 5),
        "max_score": round(max(scores), 5),
        "best_run": best_overall_run["run_id"],
        "best_seed": best_overall_run["seed"],
    }

    save_results_json(dashboard_results)

    # Save run summary as separate JSON
    summary = {
        "total_runs": n_runs,
        "runs": [{
            "run_id": r["run_id"],
            "seed": r["seed"],
            "best_model": r["best_model"],
            "best_score": r["best_score"],
            "model_scores": r["model_scores"],
            "submission_file": r["submission_file"],
            "elapsed_seconds": r["elapsed_seconds"],
        } for r in all_runs],
        "best_run": best_overall_run["run_id"],
        "best_score": best_overall_score,
        "mean_score": round(sum(scores) / len(scores), 5),
    }
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/multi_run_summary.json", "w") as f:
        json.dump(summary, f, cls=NumpyEncoder, indent=2)

    logger.info(f"Results saved to outputs/results.json and outputs/multi_run_summary.json")
    return summary


if __name__ == "__main__":
    main()
