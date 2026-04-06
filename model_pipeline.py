#!/usr/bin/env python3
"""Main entry point for the Irrigation Need Prediction pipeline.

Usage:
    python model_pipeline.py                  # Full pipeline with auto-submit
    python model_pipeline.py --no-submit      # Train only, no Kaggle submission
    python model_pipeline.py --download-only  # Just download the data
    python model_pipeline.py --skip-download  # Skip download, use existing data
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Kaggle S6E4 Irrigation Need Prediction Pipeline")
    parser.add_argument("--no-submit", action="store_true", help="Don't auto-submit to Kaggle")
    parser.add_argument("--download-only", action="store_true", help="Only download competition data")
    parser.add_argument("--skip-download", action="store_true", help="Skip data download, use existing files")
    args = parser.parse_args()

    if args.download_only:
        from pipeline.data_loader import download_competition_data
        download_competition_data(force=True)
        print("Data downloaded. Run without --download-only to train models.")
        return

    from agents.orchestrator import run_pipeline
    results = run_pipeline(auto_submit=not args.no_submit)

    print("\n" + "=" * 60)
    print(f"Best Model: {results['best_model']}")
    print(f"Best Score: {results['best_score']:.5f} (balanced accuracy)")
    print(f"Review Verdict: {results['review']['verdict']}")
    print(f"Time: {results['elapsed_seconds']}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
