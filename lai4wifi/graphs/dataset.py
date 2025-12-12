import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'False'
os.environ['JAX_ENABLE_X64'] = 'True'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

from itertools import combinations

import numpy as np
import networkx as nx
from mapc_sim.constants import DEFAULT_TX_POWER
from tqdm import tqdm

from lai4wifi.dataset.utils import load_dataset, save_dataset
from lai4wifi.graphs.attributes import *
from lai4wifi.graphs.utils import nx_to_jraph


CCA_THRESHOLD = -82.0
RSSI_MEAN = -60.0
RSSI_STD = 20.0
RSSI_THR = (CCA_THRESHOLD - RSSI_MEAN) / RSSI_STD


def configuration_to_tx(scenario, configuration):
    associations = scenario.associations
    n_nodes = len(associations) + sum(len(stas) for stas in associations.values())
    
    tx = np.zeros((n_nodes, n_nodes), dtype=int)
    tx_power = np.zeros(n_nodes, dtype=int)
    mcs = np.zeros(n_nodes, dtype=int)

    for link in configuration.links:
        ap, sta = link.ap, link.sta
        tx[ap, sta] = 1
        tx_power[ap] = link.tx_power
        mcs[ap] = link.mcs
    
    return tx, tx_power, mcs


def dataset_conf_to_nx(scenario, path_loss, configuration):
    G = nx.Graph()

    for ap, stas in scenario.associations.items():
        for sta in stas:
            if not isinstance(sta, int):
                sta = sta.item()

            rssi = (DEFAULT_TX_POWER - path_loss[ap, sta] - RSSI_MEAN) / RSSI_STD
            G.add_edge(ap, sta, active=IsActive.TRUE, mcs=McsValue.NA, rssi=rssi, selected=IsSelected.FALSE, success=0.0, tx_power=TxPowerValue.NA, type=EdgeType.STA)

    tx = configuration_to_tx(scenario, configuration)
    _, _, internals = scenario(jax.random.key(0), *tx, return_internals=True)
    succ_prob = internals.frames_transmitted / np.maximum(internals.ampdu_size, 1)

    for link in configuration.links:
        G.edges[(link.ap, link.sta)]['mcs'] = McsValue(link.mcs)
        G.edges[(link.ap, link.sta)]['selected'] = IsSelected.TRUE
        G.edges[(link.ap, link.sta)]['success'] = succ_prob[link.ap].item()
        G.edges[(link.ap, link.sta)]['tx_power'] = TxPowerValue(link.tx_power)

    for ap_1, ap_2 in combinations(scenario.associations.keys(), r=2):
        rssi = (DEFAULT_TX_POWER - path_loss[ap_1, ap_2] - RSSI_MEAN) / RSSI_STD
        if rssi > RSSI_THR:
            G.add_edge(ap_1, ap_2, active=IsActive.NA, mcs=McsValue.NA, rssi=rssi, selected=IsSelected.NA, success=0.0, tx_power=TxPowerValue.NA, type=EdgeType.AP)

    return G


def dataset_item_to_nx(dataset_item, path_loss):
    for configuration in dataset_item.configurations:
        G = dataset_conf_to_nx(dataset_item.scenario, path_loss, configuration)
        yield nx_to_jraph(G)


def scenario_to_graph_dataset(dataset):
    for dataset_item in tqdm(dataset):
        pos, walls = dataset_item.scenario.pos, dataset_item.scenario.walls
        distance = np.sqrt(np.sum((pos[:, None, :] - pos[None, ...]) ** 2, axis=-1))
        path_loss = dataset_item.scenario.path_loss_fn(distance, walls)
        dataset_item.configurations = list(dataset_item_to_nx(dataset_item, path_loss))

    return dataset


def convert_dataset(dataset_path, save_path):
    dataset = load_dataset(dataset_path)
    dataset = scenario_to_graph_dataset(dataset)
    save_dataset(dataset, save_path)


if __name__ == '__main__':
    convert_dataset('../dataset/datasets/f_opt_dataset.pkl.lz4', 'datasets/f_opt_dataset.pkl.lz4')
    convert_dataset('../dataset/datasets/f_opt_val_dataset.pkl.lz4', 'datasets/f_opt_val_dataset.pkl.lz4')
    convert_dataset('../dataset/datasets/t_opt_dataset.pkl.lz4', 'datasets/t_opt_dataset.pkl.lz4')
    convert_dataset('../dataset/datasets/t_opt_val_dataset.pkl.lz4', 'datasets/t_opt_val_dataset.pkl.lz4')
    convert_dataset('../dataset/datasets/h_mab_dataset.pkl.lz4', 'datasets/h_mab_dataset.pkl.lz4')
    convert_dataset('../dataset/datasets/h_mab_val_dataset.pkl.lz4', 'datasets/h_mab_val_dataset.pkl.lz4')
    convert_dataset('../dataset/datasets/random_dataset.pkl.lz4', 'datasets/random_dataset.pkl.lz4')
    convert_dataset('../dataset/datasets/random_val_dataset.pkl.lz4', 'datasets/random_val_dataset.pkl.lz4')
