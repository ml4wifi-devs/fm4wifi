import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'

import json
from argparse import ArgumentParser

import numpy as np
from mapc_optimal import OptimizationType
from tqdm import tqdm

from lai4wifi.baselines.h_mab import run_h_mab
from lai4wifi.baselines.solvers import run_solver
from lai4wifi.envs.scenarios import TEST_SCENARIOS


def to_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):
        return obj.item()
    return obj


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-o', '--output', type=str, default='baseline_results.json')
    args.add_argument('--scenarios', type=str, choices=list(TEST_SCENARIOS), default='residential')
    args.add_argument('--idx', type=int, default=None)
    args.add_argument('--agents', type=str, nargs='+', choices=['f_optimal', 't_optimal', 'h_mab'], default=['f_optimal', 't_optimal', 'h_mab'])
    args.add_argument('--threads', type=int, default=None)
    args.add_argument('--seed', type=int, default=42)
    args = args.parse_args()

    scenarios = TEST_SCENARIOS[args.scenarios]
    if args.idx is not None:
        scenarios = [scenarios[args.idx]]

    all_results = []

    for scenario in tqdm(scenarios, desc='Scenarios'):
        scenario_results = []

        if 'f_optimal' in args.agents:
            result = run_solver(scenario, OptimizationType.MAX_MIN, args.threads)
            scenario_results.append({'agent': 'f_optimal', **result})

        if 't_optimal' in args.agents:
            result = run_solver(scenario, OptimizationType.SUM, args.threads)
            scenario_results.append({'agent': 't_optimal', **result})

        if 'h_mab' in args.agents:
            result = run_h_mab(scenario, seed=args.seed, n_steps=scenario.n_steps)
            scenario_results.append({'agent': 'h_mab', **result})

        all_results.append(scenario_results)

    with open(args.output, 'w') as file:
        json.dump(to_serializable(all_results), file)
