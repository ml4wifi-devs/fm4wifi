from itertools import combinations

import cloudpickle
import jax.numpy as jnp
import lz4.frame
import networkx as nx
from tqdm import tqdm

from lai4wifi.graphs.attributes import *
from lai4wifi.graphs.utils import nx_to_jraph


MAX_TX_POWER = 20.0
CCA_THRESHOLD = -82.0
RSSI_MEAN = -60.0
RSSI_STD = 20.0
RSSI_THR = (CCA_THRESHOLD - RSSI_MEAN) / RSSI_STD


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
            if not isinstance(sta, int):
                sta = sta.item()

            rssi = (MAX_TX_POWER - path_loss[ap, sta] - RSSI_MEAN) / RSSI_STD
            G.add_edge(ap, sta, active=IsActive.NA, mcs=McsValue.NA, rssi=rssi, selected=IsSelected.NA, success=0.0, tx_power=TxPowerValue.NA, type=EdgeType.STA)

    for link in configuration.links:
        G.edges[(link.ap, link.sta)]['mcs'] = McsValue(link.mcs)
        G.edges[(link.ap, link.sta)]['selected'] = IsSelected.TRUE
        G.edges[(link.ap, link.sta)]['success'] = 1.0
        G.edges[(link.ap, link.sta)]['tx_power'] = TxPowerValue(link.tx_power)

    for ap_1, ap_2 in combinations(associations.keys(), r=2):
        rssi = (MAX_TX_POWER - path_loss[ap_1, ap_2] - RSSI_MEAN) / RSSI_STD
        if rssi > RSSI_THR:
            G.add_edge(ap_1, ap_2, active=IsActive.NA, mcs=McsValue.NA, rssi=rssi, selected=IsSelected.NA, success=0.0, tx_power=TxPowerValue.NA, type=EdgeType.AP)

    return G


def dataset_item_to_nx(associations, path_loss, configurations, pos, walls):
    for configuration in configurations:
        G = dataset_conf_to_nx(associations, path_loss, configuration)
        yield nx_to_jraph(G, return_misc=True, pos=pos, walls=walls)


def scenario_to_graph_dataset(dataset):
    for dataset_item in tqdm(dataset):
        distance = jnp.sqrt(jnp.sum((dataset_item.pos[:, None, :] - dataset_item.pos[None, ...]) ** 2, axis=-1))
        path_loss = dataset_item.path_loss_fn(distance, dataset_item.walls)
        yield list(dataset_item_to_nx(dataset_item.associations, path_loss, dataset_item.configurations, dataset_item.pos, dataset_item.walls))


def convert_dataset(dataset_path, save_path):
    dataset = load_dataset(dataset_path)
    graph_dataset = list(scenario_to_graph_dataset(dataset))
    save_dataset(graph_dataset, save_path)


if __name__ == '__main__':
    convert_dataset('datasets/f_optimal_50_20/dataset.pkl.lz4', 'datasets/f_optimal_50_20/train.pkl.lz4')
    convert_dataset('datasets/f_optimal_50_20/val_dataset.pkl.lz4', 'datasets/f_optimal_50_20/val.pkl.lz4')
    convert_dataset('datasets/t_optimal_100_10/dataset.pkl.lz4', 'datasets/t_optimal_100_10/train.pkl.lz4')
    convert_dataset('datasets/t_optimal_100_10/val_dataset.pkl.lz4', 'datasets/t_optimal_100_10/val.pkl.lz4')
    convert_dataset('datasets/t_optimal_1k_50/dataset.pkl.lz4', 'datasets/t_optimal_1k_50/train.pkl.lz4')
    convert_dataset('datasets/t_optimal_1k_50/val_dataset.pkl.lz4', 'datasets/t_optimal_1k_50/val.pkl.lz4')
