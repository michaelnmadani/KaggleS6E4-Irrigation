"""Kaggle API data download and submission handler."""

import os
import zipfile
import pandas as pd
from pipeline.config import COMPETITION_SLUG, DATA_DIR, SUBMISSION_DIR


def authenticate_kaggle():
    """Authenticate with Kaggle API using KAGGLE_API_TOKEN or kaggle.json."""
    import kaggle
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def _try_kaggle_cli_download():
    """Fallback: use kaggle CLI directly."""
    import subprocess
    os.makedirs(DATA_DIR, exist_ok=True)
    result = subprocess.run(
        ["kaggle", "competitions", "download", "-c", COMPETITION_SLUG, "-p", DATA_DIR],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Kaggle CLI failed: {result.stderr}")
    return result


def download_competition_data(force=False):
    """Download competition data files from Kaggle."""
    os.makedirs(DATA_DIR, exist_ok=True)

    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")
    sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

    if not force and all(os.path.exists(p) for p in [train_path, test_path, sample_path]):
        print("Data files already exist. Use force=True to re-download.")
        return train_path, test_path, sample_path

    print(f"Downloading data for {COMPETITION_SLUG}...")
    try:
        api = authenticate_kaggle()
        api.competition_download_files(COMPETITION_SLUG, path=DATA_DIR, quiet=False)
    except Exception as e:
        print(f"API download failed ({e}), trying CLI fallback...")
        _try_kaggle_cli_download()

    zip_path = os.path.join(DATA_DIR, f"{COMPETITION_SLUG}.zip")
    if os.path.exists(zip_path):
        print("Extracting zip file...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DATA_DIR)
        os.remove(zip_path)

    print("Data downloaded successfully.")
    return train_path, test_path, sample_path


def load_data(append_original=False, original_weight=0.35):
    """Load train, test, and sample submission DataFrames.

    Args:
        append_original: If True, append the original dataset to training data.
        original_weight: Sample weight for original data rows (0-1). Competition rows get weight 1.0.
    """
    train_path, test_path, sample_path = download_competition_data()
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sample_path)

    if append_original:
        orig_path = os.path.join(DATA_DIR, "irrigation_prediction.csv")
        if os.path.exists(orig_path):
            orig = pd.read_csv(orig_path)
            # Add id column to match train format
            max_id = train["id"].max() + 1
            orig.insert(0, "id", range(max_id, max_id + len(orig)))
            # Mark original vs competition rows for sample weighting
            train["_is_original"] = 0
            orig["_is_original"] = 1
            train = pd.concat([train, orig], axis=0, ignore_index=True)
            print(f"Appended {len(orig)} original rows (weight={original_weight}). "
                  f"Total train: {len(train)}")
        else:
            print(f"WARNING: Original data not found at {orig_path}. Skipping append.")

    return train, test, sample_sub


def submit_predictions(submission_path, message="Auto-submission"):
    """Submit predictions to Kaggle competition."""
    api = authenticate_kaggle()
    print(f"Submitting {submission_path} to {COMPETITION_SLUG}...")
    api.competition_submit(
        file_name=submission_path,
        message=message,
        competition=COMPETITION_SLUG,
        quiet=False
    )
    print("Submission complete.")


def save_submission(test_ids, predictions, filename="submission.csv"):
    """Save submission CSV."""
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    path = os.path.join(SUBMISSION_DIR, filename)
    sub = pd.DataFrame({"id": test_ids, "Irrigation_Need": predictions})
    sub.to_csv(path, index=False)
    print(f"Submission saved to {path}")
    return path
