#!/bin/bash

python sim.py --n_samples_eval 1 --output results_mcs_surrogate_base_1_1.json
python sim.py --n_samples_eval 1 --use_simulator --output results_mcs_sim_base_1_1.json
python sim.py --n_samples_eval 32 --output results_mcs_surrogate_base_32_1.json
python sim.py --n_samples_eval 32 --use_simulator --output results_mcs_sim_base_32_1.json
python sim.py --n_samples_eval 128 --min_top_configs 16 --output results_mcs_surrogate_base_128_16.json
python sim.py --n_samples_eval 128 --min_top_configs 16 --use_simulator --output results_mcs_sim_base_128_16.json

python sim.py --ideal_mcs --n_samples_eval 1 --output results_ideal_surrogate_base_1_1.json
python sim.py --ideal_mcs --n_samples_eval 1 --use_simulator --output results_ideal_sim_base_1_1.json
python sim.py --ideal_mcs --n_samples_eval 32 --output results_ideal_surrogate_base_32_1.json
python sim.py --ideal_mcs --n_samples_eval 32 --use_simulator --output results_ideal_sim_base_32_1.json
python sim.py --ideal_mcs --n_samples_eval 128 --min_top_configs 16 --output results_ideal_surrogate_base_128_16.json
python sim.py --ideal_mcs --n_samples_eval 128 --min_top_configs 16 --use_simulator --output results_ideal_sim_base_128_16.json
