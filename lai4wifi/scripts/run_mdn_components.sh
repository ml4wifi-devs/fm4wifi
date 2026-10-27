#!/bin/bash
set -e

cd "$(dirname "$0")/../surrogate"

# MDN mixture components ablation: m ∈ {1,2,4,8,16} × 4 sizes = 20 runs
# Uses AE-encoded prebatched data (MDN+AE)
#
# Produces NLL + effective component metrics for:
#   - Fig: surrogate_nll.tikz
#   - Fig: mdn_effective_components_boxplot.tikz

AE_TRAIN="datasets/prebatched.pkl.lz4"
AE_VAL="datasets/prebatched_val.pkl.lz4"

SIZES=(tiny small medium large)
COMPONENTS=(1 2 4 8 16)

echo "============================================"
echo "MDN Components Ablation: 20 runs"
echo "4 sizes × 5 component counts"
echo "============================================"

for SIZE in "${SIZES[@]}"; do
    for M in "${COMPONENTS[@]}"; do
        echo "[${SIZE}] m=${M}"

        python training.py model=$SIZE optimizer=$SIZE \
            name=sur_${SIZE}_mdn_m${M} \
            model.n_components=$M \
            train_dataset_path=$AE_TRAIN val_dataset_path=$AE_VAL

        CKPT=$(ls -td runs/sur_${SIZE}_mdn_m${M} 2>/dev/null | head -1)
        if [ -n "$CKPT" ]; then
            echo "Evaluating $CKPT"
            python evaluate.py mdn --ckpt-path "$CKPT" --val-dataset-path "$AE_VAL"
        fi
    done
done

echo ""
echo "============================================"
echo "All 20 MDN component ablation runs complete."
echo "Results in runs/sur_*_mdn_m*/mdn_metrics.json"
echo "============================================"
