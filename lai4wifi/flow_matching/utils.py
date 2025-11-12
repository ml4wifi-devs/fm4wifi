from collections import defaultdict
from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np
import networkx as nx

from lai4wifi.graphs.attributes import *
from lai4wifi.graphs.dataset import MAX_TX_POWER, RSSI_MEAN, RSSI_STD, RSSI_THR
from lai4wifi.graphs.utils import nx_to_jraph


def associations_to_nx(scenario):
    distance = np.sqrt(np.sum((scenario.pos[:, None, :] - scenario.pos[None, ...]) ** 2, axis=-1))
    path_loss = scenario.path_loss_fn(distance, scenario.walls)

    G = nx.Graph()

    for ap, stas in scenario.associations.items():
        for sta in stas:
            if not isinstance(sta, int):
                sta = sta.item()

            rssi = (MAX_TX_POWER - path_loss[ap, sta] - RSSI_MEAN) / RSSI_STD
            G.add_edge(ap, sta, active=IsActive.TRUE, mcs=McsValue.NA, rssi=rssi, selected=IsSelected.NA, success=0.0, tx_power=TxPowerValue.NA, type=EdgeType.STA)

    for ap_1, ap_2 in combinations(scenario.associations.keys(), r=2):
        rssi = (MAX_TX_POWER - path_loss[ap_1, ap_2] - RSSI_MEAN) / RSSI_STD
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

        tx = connections_to_tx(scenario.associations, n_nodes, tx_key, graph)
        _, _, internals = scenario(scenario_key, *tx, return_internals=True)

        context = get_next_context(graph, context, internals, encode_fn)

    return context, graph, ap_sta_idxs


def connections_to_tx(associations, n_nodes, key, graph):
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
    rssi = np.zeros_like(connections.rssi)
    success = np.zeros_like(connections.success)

    for i in range(graph.n_edge.item()):
        ap, sta = graph.senders[i].item(), graph.receivers[i].item()
        is_sta_ap = np.argmax(connections.type[i]) == EdgeType.STA.value
        rssi_val = (internals.signal_power[ap, sta] - RSSI_MEAN) / RSSI_STD

        if is_sta_ap or rssi_val > RSSI_THR:
            rssi[i] = rssi_val

        if is_sta_ap and np.argmax(connections.selected[i]) == IsSelected.TRUE.value:
            success[i] = internals.frames_transmitted[ap] / max(1, internals.ampdu_size[ap])

    edges = Connection(
        active=connections.active,
        mcs=connections.mcs,
        rssi=rssi,
        selected=connections.selected,
        success=success,
        tx_power=connections.tx_power,
        type=connections.type
    )
    edges = jax.tree.map(lambda x: x.astype(jnp.float64), edges)
    graph = graph._replace(edges=edges)
    next_conf = encode_fn(graph)

    context.append(next_conf)
    return context[1:]


def get_network_desc(target_conf, selected_idx, encode_fn):
    edge_types = jnp.argmax(target_conf.edges.type, axis=-1)

    active = jnp.where((edge_types == EdgeType.STA.value)[:, None], jnp.array(IsActive.TRUE)[None], jnp.array(IsActive.NA)[None])
    mcs = jnp.zeros((edge_types.shape[0], len(McsValue)))
    mcs = mcs.at[:].set(jnp.array(McsValue.NA))
    rssi = jnp.zeros((edge_types.shape[0], 1))
    selected = jnp.zeros((edge_types.shape[0], len(IsSelected)))
    selected = selected.at[:].set(jnp.array(IsSelected.NA))
    selected = selected.at[selected_idx].set(jnp.array(IsSelected.TRUE))
    success = jnp.zeros((edge_types.shape[0], 1))
    tx_power = jnp.zeros((edge_types.shape[0], len(TxPowerValue)))
    tx_power = tx_power.at[:].set(jnp.array(TxPowerValue.NA))

    edges = Connection(
        active=active,
        mcs=mcs,
        rssi=rssi,
        selected=selected,
        success=success,
        tx_power=tx_power,
        type=target_conf.edges.type
    )
    edges = jax.tree.map(lambda x: x.astype(jnp.float64), edges)
    desc_graph = target_conf._replace(edges=edges)

    return encode_fn(desc_graph)


def random_tx(key, scenario):
    n_aps = len(scenario.associations)
    n_nodes = n_aps + sum(len(stas) for stas in scenario.associations.values())
    
    tx = np.zeros((n_nodes, n_nodes), dtype=int)
    tx_power = np.zeros(n_nodes, dtype=int)
    mcs = np.zeros(n_nodes, dtype=int)

    for ap, stas in scenario.associations.items():
        key, ap_key, sta_key, power_key, mcs_key = jax.random.split(key, 5)

        if jax.random.uniform(ap_key) < 1 / n_aps:
            sta_idx = jax.random.choice(sta_key, jnp.asarray(stas)).item()
            tx[ap, sta_idx] = 1
            tx_power[ap] = jax.random.choice(power_key, len(TxPowerValue) - 1).item()
            mcs[ap] = jax.random.choice(mcs_key, len(McsValue) - 1).item()

    return tx, tx_power, mcs


def tx_to_conf(scenario, tx, tx_power, mcs, internals, encode_fn):
    G = nx.Graph()
    G_base = associations_to_nx(scenario)
    succ_prob = internals.frames_transmitted / jnp.maximum(internals.ampdu_size, 1)

    for ap, stas in scenario.associations.items():
        for sta in stas:
            if not isinstance(sta, int):
                sta = sta.item()

            rssi = (internals.signal_power[ap, sta] - RSSI_MEAN) / RSSI_STD
            G.add_edge(ap, sta, active=IsActive.TRUE, mcs=McsValue.NA, rssi=rssi, selected=IsSelected.NA, success=0.0, tx_power=TxPowerValue.NA, type=EdgeType.STA)
    
    for ap, sta, power, rate, success in zip(*np.where(tx), tx_power, mcs, succ_prob):
        ap, sta = ap.item(), sta.item()
        G.edges[(ap, sta)]['mcs'] = McsValue(rate.item())
        G.edges[(ap, sta)]['selected'] = IsSelected.TRUE
        G.edges[(ap, sta)]['success'] = success.item()
        G.edges[(ap, sta)]['tx_power'] = TxPowerValue(power.item())

    for ap_1, ap_2 in combinations(scenario.associations.keys(), r=2):
        rssi = (internals.signal_power[ap_1, ap_2] - RSSI_MEAN) / RSSI_STD
        if (ap_1, ap_2) in G_base.edges:
            G.add_edge(ap_1, ap_2, active=IsActive.NA, mcs=McsValue.NA, rssi=rssi, selected=IsSelected.NA, success=0.0, tx_power=TxPowerValue.NA, type=EdgeType.AP)

    graph = nx_to_jraph(G)
    return encode_fn(graph)


def gen_sample(graph):
    edges, nodes, globals = jax.tree.map(lambda x: x[:, :-1], (graph.edges, graph.nodes, graph.globals))
    return graph._replace(edges=edges, nodes=nodes, globals=globals)
