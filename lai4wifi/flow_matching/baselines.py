import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
os.environ['JAX_ENABLE_X64'] = 'True'

from itertools import chain

import jax
import numpy as np
import pulp as plp
from mapc_mab import MapcAgentFactory
from mapc_optimal import Solver, OptimizationType, positions_to_path_loss
from reinforced_lib.agents.mab import UCB

from lai4wifi.flow_matching.utils import TEST_SCENARIOS
from lai4wifi.dataset.h_mab import to_python_dict
from lai4wifi.dataset.utils import TX_POWER_LEVELS


def run_f_optimal(scenario):
    associations = scenario.associations
    access_points = list(associations.keys())
    stations = list(chain.from_iterable(associations.values()))
    path_loss = positions_to_path_loss(scenario.pos, scenario.walls)

    solver = Solver(
        stations, access_points,
        max_tx_power=TX_POWER_LEVELS[0],
        min_tx_power=TX_POWER_LEVELS[-1],
        opt_type=OptimizationType.MAX_MIN,
        solver=plp.CPLEX_CMD(msg=False, threads=24)
    )
    
    _, data_rate = solver(path_loss, associations)
    return data_rate


def run_t_optimal(scenario):
    associations = scenario.associations
    access_points = list(associations.keys())
    stations = list(chain.from_iterable(associations.values()))
    path_loss = positions_to_path_loss(scenario.pos, scenario.walls)

    solver = Solver(
        stations, access_points,
        max_tx_power=TX_POWER_LEVELS[0],
        min_tx_power=TX_POWER_LEVELS[-1],
        opt_type=OptimizationType.SUM,
        solver=plp.CPLEX_CMD(msg=False, threads=24)
    )
    
    _, data_rate = solver(path_loss, associations)
    return data_rate


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
    reward = 0.0
    data_rates = []

    for _ in range(n_steps):
        key, scenario_key = jax.random.split(key)
        tx, tx_power = agent.sample(reward)
        data_rate, reward = scenario(scenario_key, tx, tx_power)
        data_rates.append(data_rate.item())

    return np.mean(data_rates[n_steps // 2:])  # average after convergence


if __name__ == '__main__':
    baselines = ['f_optimal', 't_optimal', 'h_mab']
    
    print(f"\n{'Scenario':<12} | " + " | ".join(f"{b:>12}" for b in baselines))
    print("-" * (15 + 15 * len(baselines)))

    for i, scenario in enumerate(TEST_SCENARIOS):
        results = []
        
        for baseline in baselines:
            if baseline == 'f_optimal':
                rate = run_f_optimal(scenario)
            elif baseline == 't_optimal':
                rate = run_t_optimal(scenario)
            else:
                rate = run_h_mab(scenario)
            
            results.append(rate)
        
        rates_str = " | ".join(f"{r:>12.2f}" for r in results)
        print(f"{i+1:<12} | {rates_str}")
