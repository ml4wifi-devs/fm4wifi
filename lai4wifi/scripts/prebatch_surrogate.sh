#!/bin/bash
set -e

cd "$(dirname "$0")/../surrogate"

DATASET_PATH="../graphs/datasets"
SEED=42

echo "=== Prebatching training data ==="
python data.py generate --dataset-path "$DATASET_PATH" --save-path datasets/prebatched.pkl.lz4 --seed "$SEED"

echo "=== Prebatching validation data ==="
python data.py generate --dataset-path "$DATASET_PATH" --save-path datasets/prebatched_val.pkl.lz4 --seed "$SEED" --val

echo "=== Decoding raw training and validation data ==="
python data.py decode --input-paths datasets/prebatched.pkl.lz4 datasets/prebatched_val.pkl.lz4

echo "=== Done ==="
