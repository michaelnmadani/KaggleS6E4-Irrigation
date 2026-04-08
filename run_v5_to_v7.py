#!/usr/bin/env python3
"""Run V5, V6, V7 sequentially — full 5-fold CV, all models, no shortcuts."""

import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.data_loader import load_data, save_submission
from pipeline.utils import setup_logging, save_results_json, NumpyEncoder
from agents.researcher import run_eda
from agents.reviewer import review_results
from agents.deep_researcher import research_improvements
from agents.iterative_builder import build_versioned_models
from agents.orchestrator import _aggregate_results

logger = setup_logging()


def run_version(version, train, test, eda):
    """Run a single version: Review → Research → Build → Review → Save."""
    t0 = time.time()
    logger.info("=" * 70)
    logger.info(f"  RUNNING V{version} PIPELINE (FULL MODE)")
    logger.info("=" * 70)

    # ── Step 1: Load & review previous results ──
    with open("outputs/results.json") as f:
        prev = json.load(f)
    logger.info(f"\n  [REVIEW] Previous best: {prev['best_model']} = {prev['best_score']}")
    prev_review = prev.get("review", {})
    for w in prev_review.get("warnings", []):
        logger.info(f"    ⚠ {w}")
    for i in prev_review.get("issues", []):
        logger.info(f"    ✗ {i}")

    # ── Step 2: Deep research for improvements ──
    logger.info(f"\n  [RESEARCH] Analyzing V{version-1} results for improvements...")
    research = research_improvements(train, test, prev)
    logger.info(f"  Found {len(research['recommendations'])} recommendations:")
    for r in research["recommendations"]:
        logger.info(f"    [{r['priority']:6s}] {r['title']}")
        logger.info(f"            → {r['implementation']}")

    # ── Step 3: Build models (full mode) ──
    logger.info(f"\n  [BUILD] Building V{version} models...")
    results = build_versioned_models(train, test, version=version, prev_results=prev, fast=False)

    # Review
    logger.info(f"\n  Reviewing V{version}...")
    review = review_results(train, test, results, eda)
    logger.info(f"  Verdict: {review['verdict']}")
    for a in review["approvals"]:
        logger.info(f"    ✓ {a}")
    for w in review["warnings"]:
        logger.info(f"    ⚠ {w}")

    # Compare & save
    elapsed = round(time.time() - t0, 1)
    dashboard = _aggregate_results(eda, results, review, elapsed)

    old_score = prev["best_score"]
    new_score = results["best_score"]
    delta = new_score - old_score

    # Build version history
    prev_history = prev.get("version_history", [])
    if not prev_history:
        prev_history = [
            {"version": 1, "score": 0.96207, "best_model": "lightgbm",
             "changes": ["Baseline: 4 models + weighted ensemble", "23 features"]},
            {"version": 2, "score": 0.96979, "best_model": "lightgbm_tuned",
             "changes": ["Target encoding", "54 features", "Tuned hyperparameters"]},
        ]

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
        "research_recommendations": [r["title"] for r in research["recommendations"]],
        "research_analysis": research.get("analysis", {}),
    }

    logger.info(f"\n{'='*70}")
    logger.info(f"  V{version} RESULTS")
    logger.info(f"{'='*70}")
    logger.info(f"  Previous: {prev['best_model']:25s} BA = {old_score:.5f}")
    logger.info(f"  V{version}:      {results['best_model']:25s} BA = {new_score:.5f}")
    logger.info(f"  Delta:    {'↑' if delta > 0 else '↓'} {abs(delta):.5f}")
    logger.info(f"\n  Model breakdown:")
    for name, r in results["models"].items():
        logger.info(f"    {name:25s}: {r['mean_balanced_accuracy']:.5f}")

    save_results_json(dashboard)
    sub_path = save_submission(results["test_ids"], results["predictions"],
                               filename=f"submission_v{version}.csv")

    logger.info(f"\n  V{version} complete in {elapsed}s ({elapsed/60:.1f} min)")
    return new_score, results["best_model"]


def main():
    total_start = time.time()

    logger.info("Loading data once for all versions...")
    train, test, sample_sub = load_data()
    logger.info(f"  Train: {train.shape}, Test: {test.shape}")

    logger.info("Running EDA once...")
    eda = run_eda(train, test)

    scores = {}
    for version in [5, 6, 7]:
        try:
            score, model = run_version(version, train, test, eda)
            scores[version] = (score, model)
            logger.info(f"\n  ✓ V{version} done: {model} = {score:.5f}\n")
        except Exception as e:
            logger.error(f"\n  ✗ V{version} FAILED: {e}\n")
            import traceback
            traceback.print_exc()
            scores[version] = (0, f"FAILED: {e}")

    total_elapsed = round(time.time() - total_start, 1)

    logger.info("\n" + "=" * 70)
    logger.info("  ALL VERSIONS COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Total time: {total_elapsed}s ({total_elapsed/60:.1f} min)")
    for v, (s, m) in scores.items():
        logger.info(f"  V{v}: {m:25s} BA = {s:.5f}")

    # Version history summary
    logger.info("\n  Full version history:")
    logger.info(f"  V1: 0.96207  V2: 0.96979  V3: 0.96737  V4: 0.96951")
    for v, (s, m) in scores.items():
        logger.info(f"  V{v}: {s:.5f}")


if __name__ == "__main__":
    main()
