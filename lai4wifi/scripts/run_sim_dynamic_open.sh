#!/bin/bash

cd "$(dirname "$0")/../.." && mkdir -p results && cd results

for SEED in 1 2 3 4 5; do
    python -m lai4wifi.experiments.dynamic --env enterprise --scenario_seed 42 --model_seed $SEED --output results_dynamic_open_${SEED}.json
    python -m lai4wifi.experiments.dynamic --env enterprise --ideal_mcs --scenario_seed 42 --model_seed $SEED --output results_ideal_mcs_dynamic_open_${SEED}.json
    python -m lai4wifi.experiments.dynamic --env enterprise --no_mobility --scenario_seed 42 --model_seed $SEED --output results_no_mobility_dynamic_open_${SEED}.json
    python -m lai4wifi.experiments.dynamic --env enterprise --no_mobility --ideal_mcs --scenario_seed 42 --model_seed $SEED --output results_ideal_mcs_no_mobility_dynamic_open_${SEED}.json
    python -m lai4wifi.baselines.dynamic --env enterprise --scenario_seed 42 --seed $SEED --output results_h_mab_dynamic_open_${SEED}.json
    python -m lai4wifi.baselines.dynamic --env enterprise --no_mobility --scenario_seed 42 --seed $SEED --output results_h_mab_no_mobility_dynamic_open_${SEED}.json
done
