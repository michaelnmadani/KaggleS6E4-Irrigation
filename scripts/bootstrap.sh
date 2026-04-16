#!/usr/bin/env bash
# One-time local setup. Idempotent.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[1/3] installing python deps"
python -m pip install --upgrade pip
pip install kaggle jupytext pyyaml lightgbm xgboost catboost scikit-learn pandas numpy

echo "[2/3] checking kaggle credentials"
if [ ! -f "${HOME}/.kaggle/kaggle.json" ] && [ -z "${KAGGLE_KEY:-}" ]; then
  cat <<EOF
kaggle.json not found at ~/.kaggle/kaggle.json and KAGGLE_KEY not set.
  1. Go to https://www.kaggle.com/settings -> "Create New Token"
  2. Move the downloaded kaggle.json to ~/.kaggle/kaggle.json
  3. chmod 600 ~/.kaggle/kaggle.json
EOF
  exit 1
fi

echo "[3/3] verifying kaggle API access"
kaggle competitions list -s playground | head -5

echo "bootstrap OK"
