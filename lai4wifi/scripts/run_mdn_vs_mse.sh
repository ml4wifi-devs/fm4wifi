#!/bin/bash
set -e

cd "$(dirname "$0")/../surrogate"

# MDN vs MSE × AE vs Raw × 4 sizes = 16 runs
#
# Produces metrics for:
#   - Table: surrogate_quality.tex  (R², RMSE, MAE, Correlation, Bias)
#   - Fig: surrogate_quality_bars.tikz  (tail error percentiles P90, P95, P99)

AE_TRAIN="datasets/prebatched.pkl.lz4"
AE_VAL="datasets/prebatched_val.pkl.lz4"
RAW_TRAIN="datasets/prebatched_raw.pkl.lz4"
RAW_VAL="datasets/prebatched_val_raw.pkl.lz4"

SIZES=(tiny small medium large)

echo "============================================"
echo "Surrogate Ablation: 16 runs"
echo "4 sizes × {MDN, MSE} × {AE, Raw}"
echo "============================================"

for SIZE in "${SIZES[@]}"; do
    echo ""
    echo "=== Size: $SIZE ==="

    echo "[1/4] ${SIZE} MDN+AE"
    python training.py model=$SIZE optimizer=$SIZE \
        name=sur_${SIZE}_mdn_ae \
        train_dataset_path=$AE_TRAIN val_dataset_path=$AE_VAL

    echo "[2/4] ${SIZE} MDN+Raw"
    python training.py model=$SIZE optimizer=$SIZE \
        name=sur_${SIZE}_mdn_raw \
        train_dataset_path=$RAW_TRAIN val_dataset_path=$RAW_VAL

    echo "[3/4] ${SIZE} MSE+AE"
    python training.py model=$SIZE optimizer=$SIZE \
        name=sur_${SIZE}_mse_ae model.mse=true \
        train_dataset_path=$AE_TRAIN val_dataset_path=$AE_VAL

    echo "[4/4] ${SIZE} MSE+Raw"
    python training.py model=$SIZE optimizer=$SIZE \
        name=sur_${SIZE}_mse_raw model.mse=true \
        train_dataset_path=$RAW_TRAIN val_dataset_path=$RAW_VAL
done

echo ""
echo "============================================"
echo "All 16 ablation runs complete."
echo "Results saved in runs/sur_*/eval_metrics.json"
echo "============================================"
