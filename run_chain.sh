#!/bin/bash
# Generic chain runner — trains each fold individually (<8 min each).
# Usage: ./run_chain.sh 12
# Skips steps whose output files already exist.
set -e

export PYTHONUNBUFFERED=1
cd /home/user/KaggleS6E4-Irrigation

VERSION=${1:?"Usage: ./run_chain.sh <version>"}
INTERMEDIATE="outputs/v${VERSION}_intermediate"

run_fold() {
    local step=$1
    local fold=$2
    local check_file=$3
    if [ -f "$INTERMEDIATE/$check_file" ]; then
        echo "=== SKIP step $step fold $fold: $check_file exists ==="
        return 0
    fi
    echo ""
    echo "============================================"
    echo "  V${VERSION} step $step fold $fold at $(date)"
    echo "============================================"
    python -u run_steps.py --version "$VERSION" --step "$step" --fold "$fold"
    echo "  Step $step fold $fold completed at $(date)"
}

run_merge() {
    local step=$1
    local check_file=$2
    if [ -f "$INTERMEDIATE/$check_file" ]; then
        echo "=== SKIP step $step merge: $check_file exists ==="
        return 0
    fi
    echo ""
    echo "============================================"
    echo "  V${VERSION} step $step MERGE at $(date)"
    echo "============================================"
    python -u run_steps.py --version "$VERSION" --step "$step" --merge
    echo "  Step $step merge completed at $(date)"
}

# Step 1: LightGBM seed=42 (10 folds + merge)
for fold in $(seq 0 9); do
    run_fold 1 "$fold" "lightgbm_seed42_fold${fold}.pkl"
done
run_merge 1 "lightgbm_seed42.pkl"

# Step 2: LightGBM seed=123 (10 folds + merge)
for fold in $(seq 0 9); do
    run_fold 2 "$fold" "lightgbm_seed123_fold${fold}.pkl"
done
run_merge 2 "lightgbm_seed123.pkl"

# Step 3: XGBoost seed=42 (10 folds + merge)
for fold in $(seq 0 9); do
    run_fold 3 "$fold" "xgboost_seed42_fold${fold}.pkl"
done
run_merge 3 "xgboost_seed42.pkl"

# Step 4: CatBoost seed=42 (10 folds + merge)
for fold in $(seq 0 9); do
    run_fold 4 "$fold" "catboost_seed42_fold${fold}.pkl"
done
run_merge 4 "catboost_seed42.pkl"

# Step 5: CatBoost seed=123 (10 folds + merge)
for fold in $(seq 0 9); do
    run_fold 5 "$fold" "catboost_seed123_fold${fold}.pkl"
done
run_merge 5 "catboost_seed123.pkl"

# Step 6: Blend + stacker + bias + finalize
echo ""
echo "============================================"
echo "  V${VERSION} step 6 (blend + finalize) at $(date)"
echo "============================================"
python -u run_steps.py --version "$VERSION" --step 6
echo "  Step 6 completed at $(date)"

echo ""
echo "============================================"
echo "  ALL V${VERSION} STEPS COMPLETE at $(date)"
echo "============================================"
echo ""
ls -la outputs/submissions/submission_v${VERSION}*.csv 2>/dev/null
