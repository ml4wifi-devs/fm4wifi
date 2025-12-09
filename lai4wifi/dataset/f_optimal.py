import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'False'

from itertools import chain

import jax
import numpy as np
import pulp as plp
from mapc_optimal import Solver, OptimizationType, positions_to_path_loss

from lai4wifi.dataset.utils import generate_dataset, rate_to_mcs, tx_power_to_lvl, Configuration, TxPair, TX_POWER_LEVELS


def peek_configuration(configurations, c):
    tx_pairs = [tx for tx in configurations['links'][c]]
    ap = [int(ap.split('_')[1]) for ap, _ in tx_pairs]
    sta = [int(sta.split('_')[1]) for _, sta in tx_pairs]
    mcs = [rate_to_mcs(configurations['link_rates'][c][tx]) for tx in tx_pairs]
    tx_power = [tx_power_to_lvl(configurations['tx_power'][c][tx]) for tx in tx_pairs]
    return Configuration([TxPair(a, s, m, t) for a, s, m, t in zip(ap, sta, mcs, tx_power)])


def draw_configuration(n_configurations, key, dataset_item):
    associations = dataset_item.scenario.associations
    access_points = list(associations.keys())
    stations = list(chain.from_iterable(associations.values()))
    path_loss = positions_to_path_loss(dataset_item.scenario.pos, dataset_item.scenario.walls)

    solver = Solver(
        stations, access_points,
        max_tx_power=TX_POWER_LEVELS[0],
        min_tx_power=TX_POWER_LEVELS[-1],
        opt_type=OptimizationType.MAX_MIN,
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
        n_configurations=30, 
        draw_configuration_fn=draw_configuration, 
        save_path='datasets/f_opt_dataset.pkl.lz4'
    )
    generate_dataset(
        seed=43, 
        n_realizations=200, 
        n_configurations=30, 
        draw_configuration_fn=draw_configuration, 
        save_path='datasets/f_opt_val_dataset.pkl.lz4'
    )
