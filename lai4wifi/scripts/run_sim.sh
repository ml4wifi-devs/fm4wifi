#!/bin/bash

cd "$(dirname "$0")/../.." && mkdir -p results && cd results

echo "============ Number of samples (simulator scoring, top-1) ============"
for N in 1 2 4 8 16 32 64 128 256 512; do
    python -m lai4wifi.experiments.static --scenarios scaling --use_simulator --top_k 1 --n_samples_eval $N --output fm_scaling_samples_${N}.json
done

echo "============ Top-k selection (128 samples, simulator scoring) ============"
for K in 1 2 4 8 16 32 64 128; do
    python -m lai4wifi.experiments.static --scenarios scaling --use_simulator --top_k $K --output fm_scaling_top_${K}.json
done

echo "============ Static scenarios ============"
for SCENARIOS in residential enterprise random; do
    python -m lai4wifi.experiments.static --scenarios $SCENARIOS --output fm_${SCENARIOS}_results.json
    python -m lai4wifi.experiments.static --scenarios $SCENARIOS --ideal_mcs --output fm_${SCENARIOS}_results_ideal_mcs.json
done

echo "============ Inference time ============"
python -m lai4wifi.experiments.benchmark
