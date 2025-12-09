import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'False'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import jax
import numpy as np

from lai4wifi.dataset.utils import generate_dataset, Configuration, TxPair
from lai4wifi.flow_matching.utils import random_tx


def draw_configuration(n_configurations, key, dataset_item):
    for _ in range(n_configurations):
        key, subkey = jax.random.split(key)
        tx, tx_power, mcs = random_tx(subkey, dataset_item.scenario)
        ap, sta = np.where(tx)
        mcs = [mcs[a].item() for a in ap]
        tx_power = [tx_power[a].item() for a in ap]
        yield Configuration([TxPair(a, s, m, t) for a, s, m, t in zip(ap, sta, mcs, tx_power)])


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
