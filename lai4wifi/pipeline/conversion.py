import os
os.environ['JAX_ENABLE_X64'] = 'True'

from dataclasses import dataclass
from functools import partial
from itertools import combinations

import jax
import jax.numpy as jnp
import jraph
import numpy as np
from mapc_sim.constants import DEFAULT_TX_POWER

from lai4wifi.dataset.utils import Configuration, TxPair
from lai4wifi.graphs.attributes import (
    Connection,
    EdgeType,
    IsActive,
    IsSelected,
    McsValue,
    TxPowerValue
)
from lai4wifi.graphs.utils import (
    RSSI_MEAN,
    RSSI_STD,
    RSSI_THR,
    conf_to_nx,
    nx_to_jraph,
    make_batch,
    unbatch
)


def build_associations_arrays(associations):
    ap_ids = jnp.array(list(associations.keys()), dtype=int)
    max_stas = max(len(stas) for stas in associations.values())
    n_aps = len(associations)

    sta_ids = jnp.full((n_aps, max_stas), -1, dtype=int)
    n_stas = jnp.zeros(n_aps, dtype=int)

    for i, (ap, stas) in enumerate(associations.items()):
        sta_ids = sta_ids.at[i, :len(stas)].set(jnp.array(stas, dtype=int))
        n_stas = n_stas.at[i].set(len(stas))

    n_nodes = n_aps + sum(len(stas) for stas in associations.values())
    return ap_ids, sta_ids, n_stas, n_nodes


@partial(jax.jit, static_argnames=['n_nodes', 'deterministic'])
def _connections_to_tx_jax(key, graph, ap_ids, sta_ids, n_stas, n_nodes, deterministic):
    n_edges = graph.senders.shape[0]
    n_aps = ap_ids.shape[0]

    ap_sta_edge = jnp.argmax(graph.edges.type, axis=-1) == EdgeType.STA.value
    not_na = graph.edges.selected[:, -1] <= 0
    selected = jnp.argmax(graph.edges.selected[:, :-1], axis=-1) == IsSelected.TRUE.value
    valid_edge_mask = ap_sta_edge & not_na & selected

    def is_valid_for_ap(ap_idx):
        ap_id = ap_ids[ap_idx]
        stas = sta_ids[ap_idx]
        n_sta = n_stas[ap_idx]
        sender_is_ap = graph.senders == ap_id
        receiver_is_sta = jnp.any(
            (graph.receivers[:, None] == stas[None, :]) & (jnp.arange(stas.shape[0])[None, :] < n_sta),
            axis=-1
        )
        return valid_edge_mask & sender_is_ap & receiver_is_sta

    ap_edge_masks = jax.vmap(is_valid_for_ap)(jnp.arange(n_aps))

    def select_edge_for_ap(key, ap_idx):
        mask = ap_edge_masks[ap_idx]
        n_valid = mask.sum()
        edge_indices = jnp.where(mask, jnp.arange(n_edges), n_edges)
        sorted_indices = jnp.sort(edge_indices)
        select_idx = jax.random.randint(key, (), 0, jnp.maximum(n_valid, 1))
        selected_edge = sorted_indices[select_idx]
        return jnp.where(n_valid > 0, selected_edge, -1)

    keys = jax.random.split(key, n_aps + 1)
    key, ap_keys = keys[0], keys[1:]
    selected_edges = jax.vmap(select_edge_for_ap)(ap_keys, jnp.arange(n_aps))

    tx_matrix = jnp.zeros((n_nodes, n_nodes), dtype=int)
    tx_power = jnp.zeros(n_nodes, dtype=int)
    mcs = jnp.zeros(n_nodes, dtype=int)

    def set_tx_for_edge(carry, inputs):
        tx_matrix, tx_power, mcs, key = carry
        edge_idx, ap_id = inputs
        valid = edge_idx >= 0
        safe_idx = jnp.maximum(edge_idx, 0)
        ap, sta = graph.senders[safe_idx], graph.receivers[safe_idx]

        tx_matrix = jnp.where(valid, tx_matrix.at[ap, sta].set(1), tx_matrix)

        key, power_key, mcs_key = jax.random.split(key, 3)
        if deterministic:
            power_val = jnp.argmax(graph.edges.tx_power[safe_idx, :-1]).astype(int)
            mcs_val = jnp.argmax(graph.edges.mcs[safe_idx, :-1]).astype(int)
        else:
            power_val = jax.random.categorical(power_key, graph.edges.tx_power[safe_idx, :-1]).astype(int)
            mcs_val = jax.random.categorical(mcs_key, graph.edges.mcs[safe_idx, :-1]).astype(int)

        tx_power = jnp.where(valid, tx_power.at[ap].set(power_val), tx_power)
        mcs = jnp.where(valid, mcs.at[ap].set(mcs_val), mcs)
        return (tx_matrix, tx_power, mcs, key), None

    (tx_matrix, tx_power, mcs, _), _ = jax.lax.scan(
        set_tx_for_edge, (tx_matrix, tx_power, mcs, key), (selected_edges, ap_ids)
    )
    return tx_matrix, tx_power, mcs


def connections_to_tx(associations, key, graph, deterministic=False):
    ap_ids, sta_ids, n_stas, n_nodes = build_associations_arrays(associations)
    return _connections_to_tx_jax(key, graph, ap_ids, sta_ids, n_stas, n_nodes, deterministic)


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


def prepare_context(context, encode_fn):
    batched = make_batch(context)
    encoded = encode_fn(batched)
    unbatched = unbatch(encoded)
    ctx = jax.tree.map(lambda *x: jnp.stack(x, axis=1), *unbatched)
    return unbatched[0]._replace(edges=ctx.edges, nodes=ctx.nodes, globals=ctx.globals)


@dataclass
class ScenarioCache:
    scenario: object
    ap_ids: jax.Array
    sta_ids: jax.Array
    n_stas: jax.Array
    n_nodes: int
    ap_sta_senders: np.ndarray
    ap_sta_receivers: np.ndarray
    ap_sta_rssi: np.ndarray
    ap_ap_senders: np.ndarray
    ap_ap_receivers: np.ndarray
    ap_ap_rssi: np.ndarray
    n_ap_sta_edges: int
    n_ap_ap_edges: int
    n_edges: int


def create_scenario_cache(scenario) -> ScenarioCache:
    distance = np.sqrt(np.sum((scenario.pos[:, None, :] - scenario.pos[None, ...]) ** 2, axis=-1))
    path_loss = scenario.path_loss_fn(distance, scenario.walls)
    rssi_matrix = (DEFAULT_TX_POWER - path_loss - RSSI_MEAN) / RSSI_STD
    ap_ids, sta_ids, n_stas, n_nodes = build_associations_arrays(scenario.associations)

    ap_sta_senders = []
    ap_sta_receivers = []
    ap_sta_rssi = []

    for ap, stas in scenario.associations.items():
        for sta in stas:
            sta = int(sta)
            ap_sta_senders.append(ap)
            ap_sta_receivers.append(sta)
            ap_sta_rssi.append(rssi_matrix[ap, sta])

    ap_ap_edges = []

    for ap_1, ap_2 in combinations(scenario.associations.keys(), r=2):
        rssi = rssi_matrix[ap_1, ap_2]
        if rssi > RSSI_THR:
            ap_ap_edges.append((ap_1, ap_2, rssi))

    ap_sta_senders = np.asarray(ap_sta_senders, dtype=np.int64)
    ap_sta_receivers = np.asarray(ap_sta_receivers, dtype=np.int64)
    ap_sta_rssi = np.asarray(ap_sta_rssi, dtype=np.float64)
    ap_ap_senders = np.asarray([e[0] for e in ap_ap_edges], dtype=np.int64)
    ap_ap_receivers = np.asarray([e[1] for e in ap_ap_edges], dtype=np.int64)
    ap_ap_rssi = np.asarray([e[2] for e in ap_ap_edges], dtype=np.float64)

    n_ap_sta_edges = len(ap_sta_senders)
    n_ap_ap_edges = len(ap_ap_edges)
    n_edges = (n_ap_sta_edges + n_ap_ap_edges) * 2

    return ScenarioCache(
        scenario=scenario,
        ap_ids=ap_ids,
        sta_ids=sta_ids,
        n_stas=n_stas,
        n_nodes=n_nodes,
        ap_sta_senders=ap_sta_senders,
        ap_sta_receivers=ap_sta_receivers,
        ap_sta_rssi=ap_sta_rssi,
        ap_ap_senders=ap_ap_senders,
        ap_ap_receivers=ap_ap_receivers,
        ap_ap_rssi=ap_ap_rssi,
        n_ap_sta_edges=n_ap_sta_edges,
        n_ap_ap_edges=n_ap_ap_edges,
        n_edges=n_edges,
    )


@partial(jax.jit, static_argnames=['n_graphs', 'n_nodes', 'n_edges', 'deterministic'])
def _connections_to_tx_batch_jax(
    keys, graph, ap_ids, sta_ids, n_stas, n_graphs, n_nodes, n_edges, deterministic
):
    total_edges = n_graphs * n_edges
    edges = jax.tree.map(lambda x: x[:total_edges].reshape((n_graphs, n_edges) + x.shape[1:]), graph.edges)
    senders = graph.senders[:total_edges].reshape((n_graphs, n_edges))
    receivers = graph.receivers[:total_edges].reshape((n_graphs, n_edges))
    offsets = (jnp.arange(n_graphs, dtype=senders.dtype) * n_nodes)[:, None]
    senders = senders - offsets
    receivers = receivers - offsets
    def per_graph(k, edge_data, s, r):
        graph_i = jraph.GraphsTuple(
            nodes=jnp.empty((n_nodes, 0), dtype=jnp.float32),
            edges=edge_data,
            receivers=r,
            senders=s,
            globals=None,
            n_node=jnp.asarray([n_nodes]),
            n_edge=jnp.asarray([n_edges]),
        )
        return _connections_to_tx_jax(k, graph_i, ap_ids, sta_ids, n_stas, n_nodes, deterministic)

    return jax.vmap(per_graph)(keys, edges, senders, receivers)


def connections_to_tx_batch_fast(keys, graph, cache: ScenarioCache, n_graphs: int, deterministic=False):
    tx, tx_power, mcs = _connections_to_tx_batch_jax(
        keys=keys,
        graph=graph,
        ap_ids=cache.ap_ids,
        sta_ids=cache.sta_ids,
        n_stas=cache.n_stas,
        n_graphs=n_graphs,
        n_nodes=cache.n_nodes,
        n_edges=cache.n_edges,
        deterministic=deterministic,
    )
    return np.asarray(tx), np.asarray(tx_power), np.asarray(mcs)


@partial(jax.jit, static_argnames=['n_nodes', 'n_aps'])
def _random_tx_jax(key, ap_ids, sta_ids, n_stas, n_nodes, n_aps):
    n_tx_power = len(TxPowerValue) - 1
    n_mcs = len(McsValue) - 1
    keys = jax.random.split(key, n_aps * 4)
    ap_keys = keys[0::4]
    sta_keys = keys[1::4]
    power_keys = keys[2::4]
    mcs_keys = keys[3::4]
    active = jax.vmap(lambda k: jax.random.uniform(k) < (1.0 / n_aps))(ap_keys) & (n_stas > 0)

    def sample_sta_idx(k, n):
        return jax.random.randint(k, (), 0, jnp.maximum(n, 1))

    sta_indices = jax.vmap(sample_sta_idx)(sta_keys, n_stas)
    sta_indices = jnp.minimum(sta_indices, n_stas - 1)
    selected_stas = sta_ids[jnp.arange(n_aps), sta_indices]
    power_vals = jax.vmap(lambda k: jax.random.randint(k, (), 0, n_tx_power))(power_keys)
    mcs_vals = jax.vmap(lambda k: jax.random.randint(k, (), 0, n_mcs))(mcs_keys)
    tx_matrix = jnp.zeros((n_nodes, n_nodes), dtype=jnp.int32)
    tx_power = jnp.zeros(n_nodes, dtype=jnp.int32)
    mcs = jnp.zeros(n_nodes, dtype=jnp.int32)

    def update_for_ap(carry, inputs):
        tx_matrix, tx_power, mcs = carry
        ap_id, sta_id, is_active, pwr, mcs_val = inputs
        tx_matrix = jnp.where(is_active, tx_matrix.at[ap_id, sta_id].set(jnp.int32(1)), tx_matrix)
        tx_power = jnp.where(is_active, tx_power.at[ap_id].set(pwr.astype(jnp.int32)), tx_power)
        mcs = jnp.where(is_active, mcs.at[ap_id].set(mcs_val.astype(jnp.int32)), mcs)
        return (tx_matrix, tx_power, mcs), None

    (tx_matrix, tx_power, mcs), _ = jax.lax.scan(
        update_for_ap,
        (tx_matrix, tx_power, mcs),
        (ap_ids, selected_stas, active, power_vals.astype(jnp.int32), mcs_vals.astype(jnp.int32)),
    )
    return tx_matrix, tx_power, mcs


def random_tx_fast(key, cache: ScenarioCache):
    n_aps = cache.ap_ids.shape[0]
    tx, tx_power, mcs = _random_tx_jax(key, cache.ap_ids, cache.sta_ids, cache.n_stas, cache.n_nodes, n_aps)
    return np.asarray(tx), np.asarray(tx_power), np.asarray(mcs)


def tx_to_jraph_direct(cache: ScenarioCache, tx, tx_power, mcs, internals=None):
    n_edges_one_dir = cache.n_ap_sta_edges + cache.n_ap_ap_edges
    n_edges = n_edges_one_dir * 2

    senders = np.concatenate([
        cache.ap_sta_senders, cache.ap_ap_senders, cache.ap_sta_receivers, cache.ap_ap_receivers
    ])
    receivers = np.concatenate([
        cache.ap_sta_receivers, cache.ap_ap_receivers, cache.ap_sta_senders, cache.ap_ap_senders
    ])

    active = np.zeros((n_edges, len(IsActive)), dtype=np.float64)
    mcs_arr = np.zeros((n_edges, len(McsValue)), dtype=np.float64)
    rssi_arr = np.zeros((n_edges, 1), dtype=np.float64)
    selected = np.zeros((n_edges, len(IsSelected)), dtype=np.float64)
    success_arr = np.zeros((n_edges, 1), dtype=np.float64)
    tx_power_arr = np.zeros((n_edges, len(TxPowerValue)), dtype=np.float64)
    edge_type = np.zeros((n_edges, len(EdgeType)), dtype=np.float64)

    for dir_offset in [0, n_edges_one_dir]:
        for i in range(cache.n_ap_sta_edges):
            idx = dir_offset + i
            ap = cache.ap_sta_senders[i]
            sta = cache.ap_sta_receivers[i]

            active[idx, IsActive.TRUE.value] = 1.0
            edge_type[idx, EdgeType.STA.value] = 1.0
            rssi_arr[idx, 0] = cache.ap_sta_rssi[i]

            if tx[ap, sta]:
                selected[idx, IsSelected.TRUE.value] = 1.0
                mcs_arr[idx, mcs[ap]] = 1.0
                tx_power_arr[idx, tx_power[ap]] = 1.0
                if internals is not None:
                    success_arr[idx, 0] = internals.frames_transmitted[ap] / max(internals.ampdu_size[ap], 1)
            else:
                selected[idx, IsSelected.FALSE.value] = 1.0
                mcs_arr[idx, McsValue.NA.value] = 1.0
                tx_power_arr[idx, TxPowerValue.NA.value] = 1.0

    for dir_offset in [0, n_edges_one_dir]:
        for i in range(cache.n_ap_ap_edges):
            idx = dir_offset + cache.n_ap_sta_edges + i
            active[idx, IsActive.NA.value] = 1.0
            edge_type[idx, EdgeType.AP.value] = 1.0
            rssi_arr[idx, 0] = cache.ap_ap_rssi[i]
            selected[idx, IsSelected.NA.value] = 1.0
            mcs_arr[idx, McsValue.NA.value] = 1.0
            tx_power_arr[idx, TxPowerValue.NA.value] = 1.0

    return jraph.GraphsTuple(
        nodes=jnp.empty((cache.n_nodes, 0), dtype=jnp.float64),
        edges=Connection(
            active=jnp.asarray(active),
            mcs=jnp.asarray(mcs_arr),
            rssi=jnp.asarray(rssi_arr),
            selected=jnp.asarray(selected),
            success=jnp.asarray(success_arr),
            tx_power=jnp.asarray(tx_power_arr),
            type=jnp.asarray(edge_type),
        ),
        receivers=jnp.asarray(receivers),
        senders=jnp.asarray(senders),
        globals=None,
        n_node=jnp.asarray([cache.n_nodes]),
        n_edge=jnp.asarray([n_edges]),
    )
