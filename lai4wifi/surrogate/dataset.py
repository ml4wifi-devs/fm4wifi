import os
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_ENABLE_X64'] = 'True'

from dataclasses import replace

import jax
import jax.numpy as jnp
from tqdm import tqdm

from lai4wifi.dataset.utils import load_dataset, save_dataset
from lai4wifi.flow_matching.dataset import load_model
from lai4wifi.flow_matching.utils import connections_to_tx
from lai4wifi.graphs.utils import clear_graph, make_batch, unbatch


RATE_MEAN = 207.1
RATE_STD = 158.8


def config_to_rate_dataset(dataset, model, params):
    key = jax.random.key(42)

    for dataset_item in tqdm(dataset):
        if len(dataset_item.configurations) == 0:
            continue

        data_rates, graphs = [], []

        for graph in dataset_item.configurations:
            key, tx_key, scenario_key = jax.random.split(key, 3)
            
            tx = connections_to_tx(dataset_item.scenario.associations, tx_key, graph)
            data_rate, _ = dataset_item.scenario(scenario_key, *tx)
            data_rate = (data_rate - RATE_MEAN) / RATE_STD
            data_rates.append(data_rate.item())
            
            graph = graph._replace(
                edges=replace(graph.edges, success=jnp.zeros_like(graph.edges.success))
            )
            graphs.append(graph)
        
        batch = make_batch(graphs)
        embeddings = jax.jit(model.apply, static_argnames='method')(params, batch, method='encode')
        embeddings = clear_graph(embeddings)
        graphs = unbatch(embeddings)

        dataset_item.configurations = list(zip(graphs, data_rates))
    
    return dataset


def convert_dataset(dataset_path, save_path, model, params):
    dataset = load_dataset(dataset_path)
    dataset = config_to_rate_dataset(dataset, model, params)
    save_dataset(dataset, save_path)


if __name__ == '__main__':
    model, params = load_model('../graphs/runs/gnn_ae_base')
    convert_dataset('../graphs/datasets/f_opt_dataset.pkl.lz4', 'datasets/f_opt_dataset.pkl.lz4', model, params)
    convert_dataset('../graphs/datasets/f_opt_val_dataset.pkl.lz4', 'datasets/f_opt_val_dataset.pkl.lz4', model, params)
    convert_dataset('../graphs/datasets/t_opt_dataset.pkl.lz4', 'datasets/t_opt_dataset.pkl.lz4', model, params)
    convert_dataset('../graphs/datasets/t_opt_val_dataset.pkl.lz4', 'datasets/t_opt_val_dataset.pkl.lz4', model, params)
    convert_dataset('../graphs/datasets/h_mab_dataset.pkl.lz4', 'datasets/h_mab_dataset.pkl.lz4', model, params)
    convert_dataset('../graphs/datasets/h_mab_val_dataset.pkl.lz4', 'datasets/h_mab_val_dataset.pkl.lz4', model, params)
    convert_dataset('../graphs/datasets/random_dataset.pkl.lz4', 'datasets/random_dataset.pkl.lz4', model, params)
    convert_dataset('../graphs/datasets/random_val_dataset.pkl.lz4', 'datasets/random_val_dataset.pkl.lz4', model, params)