"""Shared utilities for the ML pipeline."""

import json
import logging
import numpy as np
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix


def setup_logging(level=logging.INFO):
    """Configure logging for the pipeline."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger("irrigation-pipeline")


def compute_metrics(y_true, y_pred, labels=None):
    """Compute classification metrics."""
    ba = balanced_accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {
        "balanced_accuracy": round(ba, 5),
        "classification_report": report,
        "confusion_matrix": cm
    }


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def save_results_json(results, path="outputs/results.json"):
    """Save results dict to JSON for the React dashboard."""
    import os
    import shutil
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, cls=NumpyEncoder, indent=2)
    # Also copy to public/ and src/ for Vercel deployment
    for dest_dir in ["public", "src"]:
        dest_path = os.path.join(dest_dir, "results.json")
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy2(path, dest_path)
    print(f"Results saved to {path}, public/results.json, and src/results.json")
