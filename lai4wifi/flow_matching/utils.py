import os
os.environ['JAX_ENABLE_X64'] = 'True'

from collections import defaultdict

import jax
import numpy as np
from mapc_research.envs.scenario_impl import residential_scenario

from lai4wifi.dataset.utils import Configuration, TxPair
from lai4wifi.graphs.attributes import *
from lai4wifi.graphs.utils import conf_to_nx, nx_to_jraph


TEST_SCENARIOS = [
    residential_scenario(seed=12, n_steps=500, x_apartments=2, y_apartments=2, n_sta_per_ap=4, size=30.0)
]


def connections_to_tx(associations, key, graph):
    n_nodes = len(associations) + sum(len(stas) for stas in associations.values())
    selected_idxs = defaultdict(list)
    connections = graph.edges

    for idx in range(graph.n_edge[0]):
        ap, sta = graph.senders[idx].item(), graph.receivers[idx].item()

        ap_sta_edge = ap in associations and sta in associations[ap]
        not_na = connections.selected[idx, -1] < 0.5
        selected = np.argmax(connections.selected[idx, :-1]) == IsSelected.TRUE.value
        
        if ap_sta_edge and not_na and selected:
            selected_idxs[ap].append(idx)

    tx_matrix = np.zeros((n_nodes, n_nodes), dtype=int)
    tx_power = np.zeros(n_nodes, dtype=int)
    mcs = np.zeros(n_nodes, dtype=int)
    selected_for_tx = []

    for ap, sta_indices in selected_idxs.items():
        if len(sta_indices) > 1:
            key, subkey = jax.random.split(key)
            selected_sta_idx = jax.random.choice(subkey, len(sta_indices), replace=False).item()
        else:
            selected_sta_idx = 0
        selected_for_tx.append(sta_indices[selected_sta_idx])

    for idx in selected_for_tx:
        ap, sta = graph.senders[idx].item(), graph.receivers[idx].item()
        tx_matrix[ap, sta] = 1
        tx_power[ap] = np.argmax(connections.tx_power[idx, :-1])
        mcs[ap] = np.argmax(connections.mcs[idx, :-1])

    return tx_matrix, tx_power, mcs


def tx_to_action(associations, internals, tx_matrix, tx_power, mcs):
    return {
        ap: (sta, internals.average_data_rate[ap].item() / 1e6, tx_power[ap].item(), mcs[ap].item()) 
        for ap in associations.keys() for sta in associations[ap] if tx_matrix[ap, sta]
    }


def random_tx(key, scenario):
    n_aps = len(scenario.associations)
    n_nodes = n_aps + sum(len(stas) for stas in scenario.associations.values())
    
    tx = np.zeros((n_nodes, n_nodes), dtype=int)
    tx_power = np.zeros(n_nodes, dtype=int)
    mcs = np.zeros(n_nodes, dtype=int)

    for ap, stas in scenario.associations.items():
        key, ap_key, sta_key, power_key, mcs_key = jax.random.split(key, 5)

        if jax.random.uniform(ap_key) < 1 / n_aps:
            sta_idx = jax.random.choice(sta_key, len(stas)).item()
            sta = stas[sta_idx]
            tx[ap, sta] = 1
            tx_power[ap] = jax.random.choice(power_key, len(TxPowerValue) - 1).item()
            mcs[ap] = jax.random.choice(mcs_key, len(McsValue) - 1).item()

    return tx, tx_power, mcs


def tx_to_conf(tx, tx_power, mcs, internals):
    ap, sta = np.where(tx)
    mcs = [mcs[a].item() for a in ap]
    tx_power = [tx_power[a].item() for a in ap]
    
    if internals is not None:
        succ_prob = (internals.frames_transmitted / np.maximum(internals.ampdu_size, 1))
        succ_prob = [succ_prob[a].item() for a in ap]
    else:
        succ_prob = [0.0 for _ in ap]
    
    return Configuration([TxPair(a, s, m, t, p) for a, s, m, t, p in zip(ap, sta, mcs, tx_power, succ_prob)])


def tx_to_jraph(scenario, tx, tx_power, mcs, internals=None, encode_fn=None):
    conf = tx_to_conf(tx, tx_power, mcs, internals)
    G = conf_to_nx(scenario, conf)
    graph = nx_to_jraph(G)
    if encode_fn is not None:
        graph = encode_fn(graph)
    return graph


def gen_sample(graph):
    edges, nodes, globals = jax.tree.map(lambda x: x[:, :-1], (graph.edges, graph.nodes, graph.globals))
    return graph._replace(edges=edges, nodes=nodes, globals=globals)
