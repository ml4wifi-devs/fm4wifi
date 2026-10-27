from glob import glob
from itertools import combinations

import grain
import jax
import jax.numpy as jnp
import jraph
import networkx as nx
import numpy as np
from mapc_sim.constants import DEFAULT_TX_POWER

from lai4wifi.dataset.utils import load_dataset
from lai4wifi.graphs.attributes import *


CCA_THRESHOLD = -82.0
RSSI_MEAN = -51.6
RSSI_STD = 14.3
RSSI_THR = (CCA_THRESHOLD - RSSI_MEAN) / RSSI_STD


def conf_to_nx(scenario, configuration):
    G = nx.Graph()
    
    distance = np.sqrt(np.sum((scenario.pos[:, None, :] - scenario.pos[None, ...]) ** 2, axis=-1))
    path_loss = scenario.path_loss_fn(distance, scenario.walls)

    for ap, stas in scenario.associations.items():
        for sta in stas:
            if not isinstance(sta, int):
                sta = sta.item()

            rssi = (DEFAULT_TX_POWER - path_loss[ap, sta] - RSSI_MEAN) / RSSI_STD
            G.add_edge(ap, sta, active=IsActive.TRUE, mcs=McsValue.NA, rssi=rssi, selected=IsSelected.FALSE, success=0.0, tx_power=TxPowerValue.NA, type=EdgeType.STA)

    for link in configuration.links:
        G.edges[(link.ap, link.sta)]['mcs'] = McsValue(link.mcs)
        G.edges[(link.ap, link.sta)]['selected'] = IsSelected.TRUE
        G.edges[(link.ap, link.sta)]['success'] = link.success
        G.edges[(link.ap, link.sta)]['tx_power'] = TxPowerValue(link.tx_power)

    for ap_1, ap_2 in combinations(scenario.associations.keys(), r=2):
        rssi = (DEFAULT_TX_POWER - path_loss[ap_1, ap_2] - RSSI_MEAN) / RSSI_STD
        if rssi > RSSI_THR:
            G.add_edge(ap_1, ap_2, active=IsActive.NA, mcs=McsValue.NA, rssi=rssi, selected=IsSelected.NA, success=0.0, tx_power=TxPowerValue.NA, type=EdgeType.AP)

    return G


def nearest_power_of_two(n):
    if not isinstance(n, int):
        n = n.item()
    return 2 ** (n - 1).bit_length()


def make_batch(graph_list, pad=True):
    batch = jraph.batch_np(graph_list)

    if pad:
        pad_n_node = nearest_power_of_two(batch.n_node.sum() + 1)
        pad_n_edge = nearest_power_of_two(batch.senders.shape[0])
        pad_n_graph = batch.n_node.shape[0] + 1
        batch = jraph.pad_with_graphs(batch, pad_n_node, pad_n_edge, pad_n_graph)

    return batch


def unbatch(batched_graph):
    graphs = jraph.unpad_with_graphs(batched_graph)
    return jraph.unbatch_np(graphs)


def nx_to_jraph(G):
    edges, senders, receivers = zip(*[[jax.tree.map(np.atleast_1d, Connection(**data)), u, v] for u, v, data in G.edges(data=True)])
    senders, receivers = senders + receivers, receivers + senders
    edges += edges

    return jraph.GraphsTuple(
        nodes=jnp.empty((len(G.nodes), 0), dtype=jnp.float64),
        edges=jax.tree.map(lambda *args: jnp.asarray(np.stack(args, dtype=np.float64)), *edges),
        receivers=jnp.asarray(receivers),
        senders=jnp.asarray(senders),
        globals=None,
        n_node=jnp.asarray([len(G.nodes)]),
        n_edge=jnp.asarray([len(edges)])
    )


def clear_graph(x):
    return x._replace(nodes=jnp.empty((x.nodes.shape[0], 0), dtype=x.nodes.dtype), globals=None)


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


def create_data_loader(
    dataset_path, batch_size, seed, 
    filter_fn=None, map_fn=default_map_fn, batch_fn=make_batch, num_workers=0
):
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
