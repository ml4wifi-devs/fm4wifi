import os
os.environ['JAX_ENABLE_X64'] = 'True'

from collections import defaultdict
from functools import partial
from glob import glob

import grain
import jax
import jax.numpy as jnp
import numpy as np
from mapc_research.envs.scenario_impl import residential_scenario

from lai4wifi.dataset.utils import Configuration, TxPair, load_dataset
from lai4wifi.graphs.attributes import *
from lai4wifi.graphs.utils import conf_to_nx, nx_to_jraph, make_batch, unbatch


TEST_SCENARIOS = [
    residential_scenario(seed=12, n_steps=500, x_apartments=2, y_apartments=2, n_sta_per_ap=4, size=30.0)
]


def _connections_to_tx_deprecated(associations, key, graph, deterministic=False):
    n_nodes = len(associations) + sum(len(stas) for stas in associations.values())
    selected_idxs = defaultdict(list)
    
    ap_sta_edge = np.argmax(graph.edges.type, axis=-1) == EdgeType.STA.value
    not_na = graph.edges.selected[:, -1] <= 0
    selected = np.argmax(graph.edges.selected[:, :-1], axis=-1) == IsSelected.TRUE.value
    idxs = np.where(ap_sta_edge & not_na & selected)[0]

    for idx in idxs:
        ap, sta = graph.senders[idx].item(), graph.receivers[idx].item()
        if ap in associations and sta in associations[ap]:
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
        key, power_key, mcs_key = jax.random.split(key, 3)
        ap, sta = graph.senders[idx].item(), graph.receivers[idx].item()
        tx_matrix[ap, sta] = 1

        if deterministic:
            tx_power[ap] = np.argmax(graph.edges.tx_power[idx, :-1])
            mcs[ap] = np.argmax(graph.edges.mcs[idx, :-1])
        else:
            tx_power[ap] = jax.random.categorical(power_key, graph.edges.tx_power[idx, :-1])
            mcs[ap] = jax.random.categorical(mcs_key, graph.edges.mcs[idx, :-1])

    return tx_matrix, tx_power, mcs


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


def gen_sample(graph):
    edges, nodes, globals = jax.tree.map(lambda x: x[:, :-1], (graph.edges, graph.nodes, graph.globals))
    return graph._replace(edges=edges, nodes=nodes, globals=globals)


def prepare_context(context, encode_fn):
    batched = make_batch(context)
    encoded = encode_fn(batched)
    unbatched = unbatch(encoded)
    ctx = jax.tree.map(lambda *x: jnp.stack(x, axis=1), *unbatched)
    return unbatched[0]._replace(edges=ctx.edges, nodes=ctx.nodes, globals=ctx.globals)


@partial(jax.jit, static_argnames=['axis', 'total_repeat_length'])
def jax_repeat(a, repeats, axis, total_repeat_length):
    return jnp.repeat(a, repeats, axis, total_repeat_length=total_repeat_length)


class DataSource(grain.sources.RandomAccessDataSource):
    def __init__(self, dataset_path, filter_fn=None):
        if filter_fn is None:
            filter_fn = lambda _: True
            
        paths = [p for p in glob(f'{dataset_path}/*.pkl.lz4')]
        paths = list(filter(filter_fn, paths))
        self._dataset = sum((load_dataset(p) for p in paths), start=[])
        self._dataset = list(((config, item.scenario) for item in self._dataset for config in item.configurations))

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx]


def default_map_fn(item):
    graph, _ = item
    return graph


def create_data_loader(dataset_path, batch_size, seed, filter_fn=None, map_fn=default_map_fn, batch_fn=make_batch, num_workers=8):
    source = DataSource(dataset_path, filter_fn)
    ds = grain.MapDataset.source(source)
    ds = ds.seed(seed)
    ds = ds.shuffle()
    ds = ds.map(map_fn)
    ds = ds.batch(batch_size, batch_fn=batch_fn)
    ds = ds.repeat()
    ds = ds.to_iter_dataset(grain.ReadOptions(num_threads=0))
    ds = ds.mp_prefetch(grain.MultiprocessingOptions(num_workers, per_worker_buffer_size=16))
    return ds
