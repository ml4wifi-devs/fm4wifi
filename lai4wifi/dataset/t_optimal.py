import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'False'

from itertools import chain

import jax
import numpy as np
import pulp as plp
from mapc_optimal import Solver, OptimizationType, positions_to_path_loss

from lai4wifi.dataset.utils import generate_dataset, TX_POWER_LEVELS
from lai4wifi.dataset.f_optimal import peek_configuration


def draw_configuration(n_configurations, key, dataset_item):
    associations = dataset_item.scenario.associations
    access_points = list(associations.keys())
    stations = list(chain.from_iterable(associations.values()))
    path_loss = positions_to_path_loss(dataset_item.scenario.pos, dataset_item.scenario.walls)

    solver = Solver(
        stations, access_points,
        max_tx_power=TX_POWER_LEVELS[0],
        min_tx_power=TX_POWER_LEVELS[-1],
        opt_type=OptimizationType.SUM,
        solver=plp.CPLEX_CMD(msg=False, threads=24)
    )
    try:
        configurations, _ = solver(path_loss, associations)
    except Exception:
        # In case the optimization fails, return an empty list of configurations
        return []

    shares = np.array(list(configurations['shares'].values()))
    idx_to_conf = {i: conf for i, conf in enumerate(configurations['links'])}

    for _ in range(n_configurations):
        key, subkey = jax.random.split(key)
        idx = jax.random.choice(subkey, len(shares), p=shares).item()
        yield peek_configuration(configurations, idx_to_conf[idx])


if __name__ == '__main__':
    generate_dataset(
        seed=42, 
        n_realizations=1000, 
        n_configurations=5, 
        draw_configuration_fn=draw_configuration, 
        save_path='datasets/t_opt_dataset.pkl.lz4'
    )
    generate_dataset(
        seed=43, 
        n_realizations=200, 
        n_configurations=5, 
        draw_configuration_fn=draw_configuration, 
        save_path='datasets/t_opt_val_dataset.pkl.lz4'
    )
