#!/bin/bash
# Run RLMRec experiments on ML-20M
#
# Resume support:
#   - Data preparation is skipped if all files exist
#   - Each seed checks for saved results before running
#   - Safe to re-run after interruption
#
# Usage:
#   bash run_rlmrec.sh                        # run all variants × all seeds
#   bash run_rlmrec.sh gene                   # run only generative variant × all seeds
#   bash run_rlmrec.sh gene 42                # run only generative variant, seed 42
#   bash run_rlmrec.sh plus                   # run only contrastive variant × all seeds
#   bash run_rlmrec.sh backbone               # run only backbone × all seeds
#
# Override seeds: SEEDS="12 42 123" bash run_rlmrec.sh gene

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/RLMRec"

DEVICE="${DEVICE:-cuda}"  # override with DEVICE=cpu if no GPU
RESULTS_DIR="$SCRIPT_DIR/../results"
export BENCHMARK_RESULTS_DIR="$RESULTS_DIR"

# Seeds — matches benchmark config.py SEEDS list (first 5)
DEFAULT_SEEDS="12 42 123 456 789"
SEEDS="${SEEDS:-$DEFAULT_SEEDS}"

# Step 1: Prepare data (skips if already done)
echo "=== Checking RLMRec data ==="
cd "$SCRIPT_DIR"
python3 prepare_rlmrec_data.py --item-features profile
cd "$SCRIPT_DIR/RLMRec"

run_variant() {
    local model=$1 label=$2 seed=$3
    local result_dir="${RESULTS_DIR}/rlmrec_${model}__ml20m__seed${seed}"
    if [ -f "${result_dir}/results.json" ]; then
        echo "=== RLMRec: ${label} seed=${seed} — SKIPPED (already complete) ==="
    else
        echo "=== RLMRec: ${label} seed=${seed} ==="
        python3 encoder/train_encoder.py --model "$model" --dataset ml20m --device "$DEVICE" --seed "$seed"
    fi
}

run_all_seeds() {
    local model=$1 label=$2
    for seed in $SEEDS; do
        run_variant "$model" "$label" "$seed"
    done
}

VARIANT="${1:-all}"
SINGLE_SEED="${2:-}"

# If a specific seed is given, override SEEDS
if [ -n "$SINGLE_SEED" ]; then
    SEEDS="$SINGLE_SEED"
fi

if [ "$VARIANT" = "all" ] || [ "$VARIANT" = "backbone" ]; then
    run_all_seeds lightgcn "LightGCN backbone (ML-20M)"
fi

if [ "$VARIANT" = "all" ] || [ "$VARIANT" = "plus" ]; then
    run_all_seeds lightgcn_plus "LightGCN+ contrastive (ML-20M)"
fi

if [ "$VARIANT" = "all" ] || [ "$VARIANT" = "gene" ]; then
    run_all_seeds lightgcn_gene "LightGCN-gene generative (ML-20M)"
fi

echo "=== Done ==="
