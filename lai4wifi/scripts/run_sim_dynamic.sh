#!/bin/bash

cd "$(dirname "$0")/../.." && mkdir -p results && cd results

run_policies() {
    NAME=$1
    shift

    for SEED in 1 2 3 4 5; do
        python -m lai4wifi.experiments.dynamic "$@" --scenario_seed 42 --model_seed $SEED --regeneration_policy once --output ${NAME}_once_${SEED}.json

        for INTERVAL in 0.5 1 2; do
            python -m lai4wifi.experiments.dynamic "$@" --scenario_seed 42 --model_seed $SEED --regeneration_policy interval --regeneration_interval_s $INTERVAL --output ${NAME}_interval_${INTERVAL}s_${SEED}.json
        done
    done
}

echo "============ Dynamic residential ============"
run_policies results_dynamic_res --env residential
run_policies results_ideal_mcs_dynamic_res --env residential --ideal_mcs

echo "============ Dynamic residential without walls ============"
run_policies results_dynamic_res_no_walls --env residential --no_walls --room_size 20
run_policies results_ideal_mcs_dynamic_res_no_walls --env residential --ideal_mcs --no_walls --room_size 20

echo "============ Dynamic random ============"
run_policies results_dynamic_random --env random
run_policies results_ideal_mcs_dynamic_random --env random --ideal_mcs

echo "============ H-MAB ============"
for SEED in 1 2 3 4 5; do
    python -m lai4wifi.baselines.dynamic --env residential --scenario_seed 42 --seed $SEED --output results_h_mab_dynamic_res_${SEED}.json
    python -m lai4wifi.baselines.dynamic --env residential --no_walls --room_size 20 --scenario_seed 42 --seed $SEED --output results_h_mab_dynamic_res_no_walls_${SEED}.json
done
