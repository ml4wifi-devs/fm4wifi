import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'False'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

from itertools import chain

import jax
import numpy as np
import pulp as plp
from mapc_optimal import Solver, OptimizationType, positions_to_path_loss

from lai4wifi.dataset.utils import generate_dataset, rate_to_mcs, tx_power_to_lvl, Configuration, TxPair, TX_POWER_LEVELS


def calculate_succ_prob(ap, sta, mcs, tx_power, dataset_item, key):
    n_nodes = len(dataset_item.scenario.pos)
    tx = np.zeros((n_nodes, n_nodes), dtype=int)
    tx_power_arr = np.zeros(n_nodes, dtype=int)
    mcs_arr = np.zeros(n_nodes, dtype=int)
    
    for a, s, m, t in zip(ap, sta, mcs, tx_power):
        tx[a, s] = 1
        tx_power_arr[a] = t
        mcs_arr[a] = m
        
    _, _, internals = dataset_item.scenario(key, tx, tx_power_arr, mcs_arr, return_internals=True)
    succ_prob = (internals.frames_transmitted / np.maximum(internals.ampdu_size, 1))
    return succ_prob


def peek_configuration(configurations, c, dataset_item, key):
    tx_pairs = [tx for tx in configurations['links'][c]]
    ap = [int(ap.split('_')[1]) for ap, _ in tx_pairs]
    sta = [int(sta.split('_')[1]) for _, sta in tx_pairs]
    mcs = [rate_to_mcs(configurations['link_rates'][c][tx]) for tx in tx_pairs]
    tx_power = [tx_power_to_lvl(configurations['tx_power'][c][tx]) for tx in tx_pairs]
    succ_prob = calculate_succ_prob(ap, sta, mcs, tx_power, dataset_item, key)
    succ_prob = [succ_prob[a].item() for a in ap]
    return Configuration([TxPair(a, s, m, t, p) for a, s, m, t, p in zip(ap, sta, mcs, tx_power, succ_prob)])


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
        solver=plp.CPLEX_CMD(msg=False),
        channel_width=80
    )
    try:
        configurations, _ = solver(path_loss, associations)
    except Exception:
        # In case the optimization fails, return an empty list of configurations
        return []

    shares = np.array(list(configurations['shares'].values()))
    idx_to_conf = {i: conf for i, conf in enumerate(configurations['links'])}

    for _ in range(n_configurations):
        key, random_key, conf_key = jax.random.split(key, 3)
        idx = jax.random.choice(random_key, len(shares), p=shares).item()
        yield peek_configuration(configurations, idx_to_conf[idx], dataset_item, conf_key)


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
