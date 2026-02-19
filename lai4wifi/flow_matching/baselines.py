import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'

import json
from argparse import ArgumentParser
from itertools import chain

import jax
import numpy as np
import pulp as plp
from mapc_mab import MapcAgentFactory
from mapc_optimal import Solver, OptimizationType, positions_to_path_loss
from reinforced_lib.agents.mab import UCB
from tqdm import tqdm, trange

from lai4wifi.flow_matching.utils import TEST_SCENARIOS, tx_to_action
from lai4wifi.dataset.h_mab import to_python_dict
from lai4wifi.dataset.utils import TX_POWER_LEVELS


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


def run_solver(scenario, opt_type):
    splits = []

    for static, duration in scenario.split_scenario():
        associations = static.associations
        access_points = list(associations.keys())
        stations = list(chain.from_iterable(associations.values()))
        path_loss = positions_to_path_loss(static.pos, static.walls)

        solver = Solver(
            stations, access_points,
            max_tx_power=TX_POWER_LEVELS[0],
            min_tx_power=TX_POWER_LEVELS[-1],
            opt_type=opt_type,
            solver=plp.CPLEX_CMD(msg=False),
            channel_width=80
        )

        configuration, data_rate = solver(path_loss, associations)
        splits.append({
            'duration': duration,
            'data_rate': data_rate,
            'shares': configuration.get('shares', {}),
            'link_rates': configuration.get('link_rates', {}),
            'tx_power': configuration.get('tx_power', {}),
        })

    return {'splits': splits}


def run_h_mab(scenario, seed=42, n_steps=2000):
    agent = MapcAgentFactory(
        associations=to_python_dict(scenario.associations),
        agent_type=UCB,
        agent_params_lvl1={'c': 1.5, 'gamma': 0.5},
        agent_params_lvl2={'c': 0.5, 'gamma': 0.5},
        agent_params_lvl3={'c': 0.2, 'gamma': 0.8},
        hierarchical=True,
        seed=seed
    ).create_mapc_agent()

    key = jax.random.PRNGKey(seed)
    scenario.reset()
    reward = 0.0
    runs = []
    actions = []

    for _ in trange(n_steps, desc='H-MAB steps'):
        key, scenario_key = jax.random.split(key)
        tx, tx_power = agent.sample(reward)
        data_rate, reward, internals = scenario(scenario_key, tx, tx_power, return_internals=True)
        runs.append(data_rate.item())
        actions.append(tx_to_action(scenario.associations, internals, tx, tx_power, internals.mcs))

    return {
        'runs': [runs],
        'actions': [actions]
    }


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('-o', '--output', type=str, default='baseline_results.json')
    args.add_argument('--n_steps', type=int, default=2000)
    args.add_argument('--seed', type=int, default=42)
    args = args.parse_args()

    all_results = []

    for scenario in tqdm(TEST_SCENARIOS, desc='Scenarios'):
        scenario_results = []

        result = run_solver(scenario, OptimizationType.MAX_MIN)
        scenario_results.append({'agent': 'f_optimal', **result})

        result = run_solver(scenario, OptimizationType.SUM)
        scenario_results.append({'agent': 't_optimal', **result})

        result = run_h_mab(scenario, seed=args.seed, n_steps=args.n_steps)
        scenario_results.append({'agent': 'h_mab', **result})

        all_results.append(scenario_results)

    with open(args.output, 'w') as file:
        json.dump(to_serializable(all_results), file)
