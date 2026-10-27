#!/bin/bash
set -e

cd "$(dirname "$0")/../surrogate"

# Risk factor lambda sweep: λ ∈ {0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4}
#
# Surrogate-only ablation using prebatched val data. For each batch,
# scores candidates with MDN at each λ, selects the best, reports its true rate.
#
# Requires: trained surrogate checkpoint + prebatched val dataset
# Produces: risk_factor_results.json for Fig risk_factor.tikz

SUR_CKPT="runs/sur_medium"
VAL_DATA="datasets/prebatched_val.pkl.lz4"

echo "============================================"
echo "Risk Factor Lambda Sweep"
echo "Surrogate: $SUR_CKPT"
echo "Val data:  $VAL_DATA"
echo "============================================"

python evaluate.py risk \
    --ckpt-path "$SUR_CKPT" \
    --val-dataset-path "$VAL_DATA" \
    --output risk_factor_results.json

echo "Done. Results in risk_factor_results.json"
