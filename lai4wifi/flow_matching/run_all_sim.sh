#!/bin/bash

python sim.py --checkpoint runs/flow_matching_small --surrogate ../surrogate/runs/surrogate --n_samples_eval 1 --output results_mcs_surrogate_base_one.json
python sim.py --checkpoint runs/flow_matching_small --surrogate ../surrogate/runs/surrogate --n_samples_eval 32 --output results_mcs_surrogate_base_multi.json
python sim.py --checkpoint runs/flow_matching_small --n_samples_eval 1 --output results_mcs_sim_base_one.json
python sim.py --checkpoint runs/flow_matching_small --n_samples_eval 32 --output results_mcs_sim_base_multi.json
