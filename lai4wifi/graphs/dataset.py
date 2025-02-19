from itertools import combinations

import cloudpickle
import jax.numpy as jnp
import lz4.frame
import networkx as nx

from lai4wifi.graphs.attributes import *
from lai4wifi.graphs.utils import nx_to_jraph

REFERENCE_RSSI = 110.0
RSSI_MEAN = 20.0
RSSI_STD = 20.0


def save_dataset(dataset, path):
    with lz4.frame.open(path, 'wb') as f:
        f.write(cloudpickle.dumps(dataset))


def load_dataset(path):
    with lz4.frame.open(path, 'rb') as f:
        return cloudpickle.loads(f.read())


def dataset_conf_to_nx(associations, path_loss, configuration):
    G = nx.Graph()

    for ap, stas in associations.items():
        for sta in stas:
            rssi = (REFERENCE_RSSI - path_loss[ap, sta] - RSSI_MEAN) / RSSI_STD
            G.add_edge(ap, sta, active=IsActive.NA, mcs=McsValue.NA, rssi=rssi, selected=IsSelected.NA, success=IsSuccessful.NA, tx_power=TxPowerValue.NA, type=EdgeType.STA)

    for link in configuration.links:
        G.edges[(link.ap, link.sta)]['mcs'] = McsValue(link.mcs)
        G.edges[(link.ap, link.sta)]['selected'] = IsSelected.TRUE
        G.edges[(link.ap, link.sta)]['success'] = IsSuccessful.TRUE
        G.edges[(link.ap, link.sta)]['tx_power'] = TxPowerValue(link.tx_power)

    for ap_1, ap_2 in combinations(associations.keys(), r=2):
        rssi = (REFERENCE_RSSI - path_loss[ap_1, ap_2] - RSSI_MEAN) / RSSI_STD
        G.add_edge(ap_1, ap_2, active=IsActive.NA, mcs=McsValue.NA, rssi=rssi, selected=IsSelected.NA, success=IsSuccessful.NA, tx_power=TxPowerValue.NA, type=EdgeType.AP)

    return G


def dataset_item_to_nx(associations, path_loss, configurations, pos, walls):
    for configuration in configurations:
        G = dataset_conf_to_nx(associations, path_loss, configuration)
        yield nx_to_jraph(G, return_misc=True, pos=pos, walls=walls)


def scenario_to_graph_dataset(dataset):
    for dataset_item in dataset:
        distance = jnp.sqrt(jnp.sum((dataset_item.pos[:, None, :] - dataset_item.pos[None, ...]) ** 2, axis=-1))
        path_loss = dataset_item.path_loss_fn(distance, dataset_item.walls)
        yield list(dataset_item_to_nx(dataset_item.associations, path_loss, dataset_item.configurations, dataset_item.pos, dataset_item.walls))


if __name__ == '__main__':
    dataset = load_dataset('dataset.pkl.lz4')
    graph_dataset = list(scenario_to_graph_dataset(dataset))
    save_dataset(graph_dataset, 'graph_dataset.pkl.lz4')
