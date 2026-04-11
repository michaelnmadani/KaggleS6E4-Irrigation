#!/usr/bin/env python3
"""Submit predictions to Kaggle and download original dataset.

Usage:
    # Submit a specific file
    python submit_to_kaggle.py submit outputs/submissions/submission_v11_catboost_dom.csv "V11 CatBoost-dominant blend"

    # Submit all V11 submissions
    python submit_to_kaggle.py submit-all v11

    # Download the original dataset (for appending to training data)
    python submit_to_kaggle.py download-original

    # Check submission status
    python submit_to_kaggle.py status

Requires: ~/.kaggle/kaggle.json with valid credentials
"""

import os
import sys
import glob
import time
import argparse
import zipfile


COMPETITION_SLUG = "playground-series-s6e4"
ORIGINAL_DATASET = "miadul/irrigation-water-requirement-prediction-dataset"
DATA_DIR = "data"
SUBMISSION_DIR = "outputs/submissions"


def _get_api():
    """Authenticate and return Kaggle API client."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        return api
    except Exception as e:
        print(f"ERROR: Kaggle authentication failed: {e}")
        print("Make sure ~/.kaggle/kaggle.json exists with valid credentials.")
        print("Get your token at: https://www.kaggle.com/settings -> API -> Create New Token")
        sys.exit(1)


def download_original():
    """Download the original dataset used to generate the competition data."""
    api = _get_api()
    dest = os.path.join(DATA_DIR, "original_download")
    os.makedirs(dest, exist_ok=True)

    print(f"Downloading original dataset: {ORIGINAL_DATASET}")
    api.dataset_download_files(ORIGINAL_DATASET, path=dest, unzip=True)

    # Find the CSV file and move it to data/
    csv_files = glob.glob(os.path.join(dest, "*.csv"))
    if csv_files:
        target = os.path.join(DATA_DIR, "irrigation_prediction.csv")
        os.rename(csv_files[0], target)
        print(f"Original dataset saved to: {target}")
        print(f"Rows: {sum(1 for _ in open(target)) - 1}")
        # Clean up
        try:
            os.rmdir(dest)
        except OSError:
            pass
    else:
        print(f"WARNING: No CSV found in download. Check {dest}/")


def submit(filepath, message=None):
    """Submit a prediction file to the competition."""
    api = _get_api()

    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        sys.exit(1)

    # Validate submission format
    with open(filepath) as f:
        header = f.readline().strip()
        if header != "id,Irrigation_Need":
            print(f"WARNING: Unexpected header: {header}")

        n_rows = sum(1 for _ in f)
        if n_rows != 270000:
            print(f"WARNING: Expected 270000 rows, got {n_rows}")

    if message is None:
        basename = os.path.basename(filepath)
        message = f"Auto-submit: {basename}"

    print(f"Submitting {filepath} to {COMPETITION_SLUG}...")
    print(f"  Message: {message}")
    api.competition_submit(
        file_name=filepath,
        message=message,
        competition=COMPETITION_SLUG,
        quiet=False,
    )
    print("Submission complete! Check leaderboard at:")
    print(f"  https://www.kaggle.com/competitions/{COMPETITION_SLUG}/leaderboard")


def submit_all(version):
    """Submit all submission files for a given version."""
    pattern = os.path.join(SUBMISSION_DIR, f"submission_{version}*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No submission files found matching: {pattern}")
        sys.exit(1)

    print(f"Found {len(files)} submission files for {version}:")
    for f in files:
        print(f"  {os.path.basename(f)}")
    print()

    for i, filepath in enumerate(files):
        basename = os.path.basename(filepath)
        message = f"{version.upper()} {basename.replace(f'submission_{version}_', '').replace('.csv', '')}"
        print(f"\n[{i+1}/{len(files)}] Submitting {basename}...")
        submit(filepath, message)

        # Rate limit: wait between submissions
        if i < len(files) - 1:
            print("  Waiting 30s before next submission (rate limit)...")
            time.sleep(30)

    print(f"\nAll {len(files)} submissions complete!")


def check_status():
    """Check recent submission status."""
    api = _get_api()
    print(f"Recent submissions for {COMPETITION_SLUG}:")
    submissions = api.competition_submissions(COMPETITION_SLUG)
    for s in submissions[:10]:
        status = s.get("status", "unknown")
        score = s.get("publicScore", "N/A")
        desc = s.get("description", "")
        date = s.get("date", "")
        fname = s.get("fileName", "")
        print(f"  {date}  {fname:40s}  Score: {score:8s}  Status: {status}  {desc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaggle submission helper")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # submit
    sub_parser = subparsers.add_parser("submit", help="Submit a single file")
    sub_parser.add_argument("filepath", help="Path to submission CSV")
    sub_parser.add_argument("message", nargs="?", default=None, help="Submission message")

    # submit-all
    all_parser = subparsers.add_parser("submit-all", help="Submit all files for a version")
    all_parser.add_argument("version", help="Version prefix, e.g. 'v11'")

    # download-original
    subparsers.add_parser("download-original", help="Download original dataset")

    # status
    subparsers.add_parser("status", help="Check submission status")

    args = parser.parse_args()

    if args.command == "submit":
        submit(args.filepath, args.message)
    elif args.command == "submit-all":
        submit_all(args.version)
    elif args.command == "download-original":
        download_original()
    elif args.command == "status":
        check_status()
    else:
        parser.print_help()
