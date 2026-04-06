#!/usr/bin/env python3
"""Simple wrapper to run the full pipeline."""

import subprocess
import sys

if __name__ == "__main__":
    sys.exit(subprocess.call([sys.executable, "model_pipeline.py"] + sys.argv[1:]))
