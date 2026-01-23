import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'False'
os.environ['JAX_ENABLE_X64'] = 'True'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

from tqdm import tqdm

from lai4wifi.dataset.utils import load_dataset, save_dataset
from lai4wifi.graphs.utils import conf_to_nx, nx_to_jraph


def dataset_item_to_nx(dataset_item):
    for configuration in dataset_item.configurations:
        G = conf_to_nx(dataset_item.scenario, configuration)
        yield nx_to_jraph(G)


def scenario_to_graph_dataset(dataset):
    for dataset_item in tqdm(dataset):
        dataset_item.configurations = list(dataset_item_to_nx(dataset_item))

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
