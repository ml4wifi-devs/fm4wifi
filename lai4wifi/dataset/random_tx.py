import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'False'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import jax
import numpy as np

from lai4wifi.dataset.utils import generate_dataset, Configuration, TxPair
from lai4wifi.flow_matching.utils import random_tx


def draw_configuration(n_configurations, key, dataset_item):
    for _ in range(n_configurations):
        key, random_key, scenario_key = jax.random.split(key, 3)
        tx, tx_power, mcs = random_tx(random_key, dataset_item.scenario)
        _, _, internals = dataset_item.scenario(scenario_key, tx, tx_power, mcs, return_internals=True)
        succ_prob = (internals.frames_transmitted / np.maximum(internals.ampdu_size, 1))
        ap, sta = np.where(tx)
        mcs = [mcs[a].item() for a in ap]
        tx_power = [tx_power[a].item() for a in ap]
        succ_prob = [succ_prob[a].item() for a in ap]
        yield Configuration([TxPair(a, s, m, t, p) for a, s, m, t, p in zip(ap, sta, mcs, tx_power, succ_prob)])


if __name__ == '__main__':
    generate_dataset(
        seed=42, 
        n_realizations=1000, 
        n_configurations=30, 
        draw_configuration_fn=draw_configuration, 
        save_path='datasets/random_dataset.pkl.lz4'
    )
    generate_dataset(
        seed=43, 
        n_realizations=200, 
        n_configurations=30, 
        draw_configuration_fn=draw_configuration, 
        save_path='datasets/random_val_dataset.pkl.lz4'
    )
