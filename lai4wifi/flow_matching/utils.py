from collections import defaultdict
from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np
import networkx as nx
from mapc_sim.constants import DEFAULT_TX_POWER

from lai4wifi.graphs.attributes import *
from lai4wifi.graphs.dataset import RSSI_MEAN, RSSI_STD, RSSI_THR
from lai4wifi.graphs.utils import clear_graph, nx_to_jraph


def associations_to_nx(scenario):
    distance = np.sqrt(np.sum((scenario.pos[:, None, :] - scenario.pos[None, ...]) ** 2, axis=-1))
    path_loss = scenario.path_loss_fn(distance, scenario.walls)

    G = nx.Graph()

    for ap, stas in scenario.associations.items():
        for sta in stas:
            if not isinstance(sta, int):
                sta = sta.item()

            rssi = (DEFAULT_TX_POWER - path_loss[ap, sta] - RSSI_MEAN) / RSSI_STD
            G.add_edge(ap, sta, active=IsActive.TRUE, mcs=McsValue.NA, rssi=rssi, selected=IsSelected.NA, success=0.0, tx_power=TxPowerValue.NA, type=EdgeType.STA)

    for ap_1, ap_2 in combinations(scenario.associations.keys(), r=2):
        rssi = (DEFAULT_TX_POWER - path_loss[ap_1, ap_2] - RSSI_MEAN) / RSSI_STD
        if rssi > RSSI_THR:
            G.add_edge(ap_1, ap_2, active=IsActive.NA, mcs=McsValue.NA, rssi=rssi, selected=IsSelected.NA, success=0.0, tx_power=TxPowerValue.NA, type=EdgeType.AP)

    return G


def get_initial_configurations(scenario, n_timesteps, key, encode_fn):
    G = associations_to_nx(scenario)
    graph = nx_to_jraph(G)
    context = [encode_fn(graph)] * n_timesteps

    ap_sta_idxs = np.where(np.argmax(graph.edges.type, axis=-1) == EdgeType.STA.value)[0]
    n_nodes = len(scenario.associations) + sum(len(stas) for stas in scenario.associations.values())

    for _ in range(n_timesteps):
        key, sel_key, tx_key, scenario_key = jax.random.split(key, 4)

        selected_idx = jax.random.choice(sel_key, ap_sta_idxs).item()
        sharing_ap = graph.senders[selected_idx].item()
        selected_sta = graph.receivers[selected_idx].item()

        G = associations_to_nx(scenario)
        G.edges[sharing_ap, selected_sta]['mcs'] = McsValue.MCS_11
        G.edges[sharing_ap, selected_sta]['selected'] = IsSelected.TRUE
        G.edges[sharing_ap, selected_sta]['success'] = 1.0
        G.edges[sharing_ap, selected_sta]['tx_power'] = TxPowerValue.LVL_1
        graph = nx_to_jraph(G)

        tx = connections_to_tx(scenario.associations, tx_key, graph)
        _, _, internals = scenario(scenario_key, *tx, return_internals=True)

        context = get_next_context(graph, context, internals, encode_fn)

    return context, graph, ap_sta_idxs


def connections_to_tx(associations, key, graph):
    n_nodes = len(associations) + sum(len(stas) for stas in associations.values())
    selected = defaultdict(list)
    connections = graph.edges

    for i in range(graph.n_edge[0]):
        ap, sta = graph.senders[i].item(), graph.receivers[i].item()
        if ap in associations and sta in associations[ap] and np.argmax(connections.selected[i]) == IsSelected.TRUE.value:
            selected[ap].append(i)

    tx_matrix = np.zeros((n_nodes, n_nodes), dtype=int)
    tx_power = np.zeros(n_nodes, dtype=int)
    mcs = np.zeros(n_nodes, dtype=int)
    selected_for_tx = []

    for ap, sta_indices in selected.items():
        key, subkey = jax.random.split(key)
        selected_sta_idx = jax.random.choice(subkey, len(sta_indices), replace=False).item()
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


def get_next_context(graph, context, internals, encode_fn):
    connections = graph.edges
    rssi = jnp.zeros(connections.rssi.shape, dtype=jnp.float64)
    success = jnp.zeros(connections.success.shape, dtype=jnp.float64)

    for i in range(graph.n_edge.item()):
        ap, sta = graph.senders[i].item(), graph.receivers[i].item()
        is_sta_ap = jnp.argmax(connections.type[i]) == EdgeType.STA.value
        rssi = rssi.at[i].set((internals.signal_power[ap, sta] - RSSI_MEAN) / RSSI_STD)

        if is_sta_ap and jnp.argmax(connections.selected[i]) == IsSelected.TRUE.value:
            success = success.at[i].set(internals.frames_transmitted[ap] / max(1, internals.ampdu_size[ap]))

    edges = Connection(
        active=jax.nn.one_hot(jnp.argmax(connections.active, axis=-1), len(IsActive), dtype=jnp.float64),
        mcs=jax.nn.one_hot(jnp.argmax(connections.mcs, axis=-1), len(McsValue), dtype=jnp.float64),
        rssi=rssi,
        selected=jax.nn.one_hot(jnp.argmax(connections.selected, axis=-1), len(IsSelected), dtype=jnp.float64),
        success=success,
        tx_power=jax.nn.one_hot(jnp.argmax(connections.tx_power, axis=-1), len(TxPowerValue), dtype=jnp.float64),
        type=jax.nn.one_hot(jnp.argmax(connections.type, axis=-1), len(EdgeType), dtype=jnp.float64)
    )

    graph = graph._replace(edges=edges)
    graph = clear_graph(graph)
    next_conf = encode_fn(graph)

    context.append(next_conf)
    return context[1:]


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


def tx_to_conf(scenario, tx, tx_power, mcs, internals, encode_fn, add_success=True):
    G = nx.Graph()
    G_base = associations_to_nx(scenario)
    succ_prob = internals.frames_transmitted / np.maximum(internals.ampdu_size, 1)

    if not add_success:
        succ_prob = np.zeros_like(succ_prob)

    for ap, stas in scenario.associations.items():
        for sta in stas:
            if not isinstance(sta, int):
                sta = sta.item()

            rssi = (internals.signal_power[ap, sta] - RSSI_MEAN) / RSSI_STD
            G.add_edge(ap, sta, active=IsActive.TRUE, mcs=McsValue.NA, rssi=rssi, selected=IsSelected.NA, success=0.0, tx_power=TxPowerValue.NA, type=EdgeType.STA)
    
    for ap, sta in zip(*np.where(tx)):
        ap, sta = ap.item(), sta.item()
        G.edges[(ap, sta)]['mcs'] = McsValue(mcs[ap].item())
        G.edges[(ap, sta)]['selected'] = IsSelected.TRUE
        G.edges[(ap, sta)]['success'] = succ_prob[ap].item()
        G.edges[(ap, sta)]['tx_power'] = TxPowerValue(tx_power[ap].item())

    for ap_1, ap_2 in combinations(scenario.associations.keys(), r=2):
        rssi = (internals.signal_power[ap_1, ap_2] - RSSI_MEAN) / RSSI_STD
        if (ap_1, ap_2) in G_base.edges:
            G.add_edge(ap_1, ap_2, active=IsActive.NA, mcs=McsValue.NA, rssi=rssi, selected=IsSelected.NA, success=0.0, tx_power=TxPowerValue.NA, type=EdgeType.AP)

    graph = nx_to_jraph(G)
    return encode_fn(graph)


def gen_sample(graph):
    edges, nodes, globals = jax.tree.map(lambda x: x[:, :-1], (graph.edges, graph.nodes, graph.globals))
    return graph._replace(edges=edges, nodes=nodes, globals=globals)
