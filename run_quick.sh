#!/bin/bash
# Quick ablation: BPR-MF + LightGCN-SF only
#
# Resume support:
#   - Interrupted experiments resume from last saved epoch
#   - Completed experiments with enough epochs are skipped
#   - To extend training: increase EPOCHS and re-run
#   - Safe to re-run after interruption
set -e
cd "$(dirname "$0")"

EPOCHS=${EPOCHS:-10}
SEED=${SEED:-42}

echo "=== Quick Ablation: BPR-MF + LightGCN-SF (${EPOCHS} epochs, seed=${SEED}) ==="
echo "    Resume mode: will continue from last checkpoint"
echo ""

run_exp() {
    local idx=$1 label=$2 model=$3 features=$4
    echo "[${idx}/9] ${label}"
    python3 run_experiment.py --model "$model" --features "$features" --seed $SEED --epochs $EPOCHS
}

# M0: BPR-MF baseline
run_exp 1 "M0: BPR-MF (ID only)" bpr_mf none

# M2-M9: LightGCN-SF with different features
run_exp 2 "M2: + genome PCA(128)" lightgcn_sf genome
run_exp 3 "M3: + BERT title(128)" lightgcn_sf bert_title
run_exp 4 "M4: + LLM profile(128)" lightgcn_sf llm_profile
run_exp 5 "M5: + LLM mood(10)" lightgcn_sf llm_mood
run_exp 6 "M6: + LLM themes(528)" lightgcn_sf llm_themes
run_exp 7 "M7: + LLM profile+mood(138)" lightgcn_sf llm_prof_mood
run_exp 8 "M8: + LLM all(666)" lightgcn_sf llm_all
run_exp 9 "M9: + genome+mood+themes" lightgcn_sf genome_llm

echo ""
echo "=== Collecting results ==="
python3 run_ablation.py --collect-only

echo "=== Done ==="
