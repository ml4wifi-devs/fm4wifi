#!/bin/bash

python sim.py --checkpoint runs/flow_matching_small --surrogate ../surrogate/runs/surrogate --n_samples_eval 1 --output results_mcs_surrogate_base_one.json
python sim.py --checkpoint runs/flow_matching_small --surrogate ../surrogate/runs/surrogate --n_samples_eval 32 --output results_mcs_surrogate_base_multi.json
python sim.py --checkpoint runs/flow_matching_small --n_samples_eval 1 --output results_mcs_sim_base_one.json
python sim.py --checkpoint runs/flow_matching_small --n_samples_eval 32 --output results_mcs_sim_base_multi.json

python sim.py --checkpoint runs/flow_matching_small --ideal_mcs --surrogate ../surrogate/runs/surrogate --n_samples_eval 1 --output results_power_surrogate_base_one.json
python sim.py --checkpoint runs/flow_matching_small --ideal_mcs --surrogate ../surrogate/runs/surrogate --n_samples_eval 32 --output results_power_surrogate_base_multi.json
python sim.py --checkpoint runs/flow_matching_small --ideal_mcs --n_samples_eval 1 --output results_power_sim_base_one.json
python sim.py --checkpoint runs/flow_matching_small --ideal_mcs --n_samples_eval 32 --output results_power_sim_base_multi.json

# ------------------------------

python sim.py --checkpoint runs/flow_matching_rl --surrogate ../surrogate/runs/surrogate --n_samples_eval 1 --output results_mcs_surrogate_rl_one.json
python sim.py --checkpoint runs/flow_matching_rl --surrogate ../surrogate/runs/surrogate --n_samples_eval 32 --output results_mcs_surrogate_rl_multi.json
python sim.py --checkpoint runs/flow_matching_rl --n_samples_eval 1 --output results_mcs_sim_rl_one.json
python sim.py --checkpoint runs/flow_matching_rl --n_samples_eval 32 --output results_mcs_sim_rl_multi.json

python sim.py --checkpoint runs/flow_matching_rl --ideal_mcs --surrogate ../surrogate/runs/surrogate --n_samples_eval 1 --output results_power_surrogate_rl_one.json
python sim.py --checkpoint runs/flow_matching_rl --ideal_mcs --surrogate ../surrogate/runs/surrogate --n_samples_eval 32 --output results_power_surrogate_rl_multi.json
python sim.py --checkpoint runs/flow_matching_rl --ideal_mcs --n_samples_eval 1 --output results_power_sim_rl_one.json
python sim.py --checkpoint runs/flow_matching_rl --ideal_mcs --n_samples_eval 32 --output results_power_sim_rl_multi.json

# ------------------------------

python sim_random.py --surrogate ../surrogate/runs/surrogate --n_samples_eval 1 --output results_mcs_surrogate_random_one.json
python sim_random.py --surrogate ../surrogate/runs/surrogate --n_samples_eval 32 --output results_mcs_surrogate_random_multi.json
python sim_random.py --n_samples_eval 1 --output results_mcs_sim_random_one.json
python sim_random.py --n_samples_eval 32 --output results_mcs_sim_random_multi.json

python sim_random.py --ideal_mcs --surrogate ../surrogate/runs/surrogate --n_samples_eval 1 --output results_power_surrogate_random_one.json
python sim_random.py --ideal_mcs --surrogate ../surrogate/runs/surrogate --n_samples_eval 32 --output results_power_surrogate_random_multi.json
python sim_random.py --ideal_mcs --n_samples_eval 1 --output results_power_sim_random_one.json
python sim_random.py --ideal_mcs --n_samples_eval 32 --output results_power_sim_random_multi.json
