#!/bin/bash
# Generic chain runner for any version.
# Usage: ./run_chain.sh 12
# Skips steps whose output files already exist.
set -e

export PYTHONUNBUFFERED=1
cd /home/user/KaggleS6E4-Irrigation

VERSION=${1:?"Usage: ./run_chain.sh <version>"}
INTERMEDIATE="outputs/v${VERSION}_intermediate"

run_step() {
    local step=$1
    local check_file=$2
    if [ -n "$check_file" ] && [ -f "$INTERMEDIATE/$check_file" ]; then
        echo "=== SKIP step $step: $check_file already exists ==="
        return 0
    fi
    echo ""
    echo "============================================"
    echo "  Running V${VERSION} step $step at $(date)"
    echo "============================================"
    python -u run_steps.py --version "$VERSION" --step "$step"
    echo "  Step $step completed at $(date)"
}

# Step 1: LightGBM seed=42
run_step 1a "lightgbm_seed42_f0_5.pkl"
run_step 1b "lightgbm_seed42.pkl"

# Step 2: LightGBM seed=123
run_step 2a "lightgbm_seed123_f0_5.pkl"
run_step 2b "lightgbm_seed123.pkl"

# Step 3: XGBoost seed=42
run_step 3a "xgboost_seed42_f0_5.pkl"
run_step 3b "xgboost_seed42.pkl"

# Step 4: CatBoost seed=42
run_step 4a "catboost_seed42_f0_5.pkl"
run_step 4b "catboost_seed42.pkl"

# Step 5: CatBoost seed=123
run_step 5a "catboost_seed123_f0_5.pkl"
run_step 5b "catboost_seed123.pkl"

# Step 6: Blend + stacker + bias + finalize
run_step 6 ""

echo ""
echo "============================================"
echo "  ALL V${VERSION} STEPS COMPLETE at $(date)"
echo "============================================"
echo ""
ls -la outputs/submissions/submission_v${VERSION}*.csv 2>/dev/null
