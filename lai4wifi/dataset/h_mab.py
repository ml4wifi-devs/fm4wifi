import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'False'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import jax
import numpy as np
from mapc_mab import MapcAgentFactory
from reinforced_lib.agents.mab import UCB

from lai4wifi.dataset.utils import generate_dataset, Configuration, TxPair


def to_python_dict(d):
    result = {}
    for k, v in d.items():
        if hasattr(k, 'item'):
            k = k.item()
        result[k] = np.asarray(v).tolist()
    return result


def draw_configuration(n_configurations, key, dataset_item, n_steps=2000):
    agent = MapcAgentFactory(
        associations=to_python_dict(dataset_item.scenario.associations),
        agent_type=UCB,
        agent_params_lvl1={'c': 1.5, 'gamma': 0.5},
        agent_params_lvl2={'c': 0.5, 'gamma': 0.5},
        agent_params_lvl3={'c': 0.2, 'gamma': 0.8},
        hierarchical=True,
        seed=42
    ).create_mapc_agent()
    reward = 0.0
    tx_pairs = []

    for _ in range(n_steps):
        key, scenario_key = jax.random.split(key)
        tx, tx_power = agent.sample(reward)
        _, reward, internals = dataset_item.scenario(scenario_key, tx, tx_power, return_internals=True)
        succ_prob = (internals.frames_transmitted / np.maximum(internals.ampdu_size, 1))
        ap, sta = np.where(tx)
        mcs = [internals.mcs[a].item() for a in ap]
        tx_power = [tx_power[a].item() for a in ap]
        succ_prob = [succ_prob[a].item() for a in ap]
        tx_pairs.append([TxPair(a, s, m, t, p) for a, s, m, t, p in zip(ap, sta, mcs, tx_power, succ_prob)])

    tx_pairs = tx_pairs[n_steps // 2:]  # configurations after convergence
    idxs = jax.random.choice(key, len(tx_pairs), shape=(n_configurations,), replace=False)

    for i in idxs:
        yield Configuration(tx_pairs[i])


if __name__ == '__main__':
    generate_dataset(
        seed=42, 
        n_realizations=1000, 
        n_configurations=50, 
        draw_configuration_fn=draw_configuration, 
        save_path='datasets/h_mab_dataset.pkl.lz4'
    )
    generate_dataset(
        seed=43, 
        n_realizations=200, 
        n_configurations=50, 
        draw_configuration_fn=draw_configuration, 
        save_path='datasets/h_mab_val_dataset.pkl.lz4'
    )
