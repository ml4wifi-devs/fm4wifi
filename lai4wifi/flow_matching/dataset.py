import os

import jax.random
import jraph
import orbax.checkpoint as ocp
from omegaconf import OmegaConf
from tqdm import tqdm

from lai4wifi.graphs.attributes import *
from lai4wifi.graphs.dataset import load_dataset, save_dataset
from lai4wifi.graphs.model import GnnAutoencoder
from lai4wifi.graphs.training import load
from lai4wifi.graphs.utils import make_batch


def graph_to_embeddings_dataset(dataset, model, params):
    for configurations in tqdm(dataset):
        if len(configurations) == 0:
            continue

        batch = make_batch(list(map(lambda c: c.graph, configurations)))
        batch = jax.jit(model.apply, static_argnames='method')(params, batch, method='encode')
        batch = jraph.unpad_with_graphs(batch)
        batch = jraph.unbatch(batch)

        for i, embeddings in enumerate(batch):
            configurations[i].embedding = embeddings

        yield configurations


def convert_dataset(dataset_path, save_path, model, params):
    dataset = load_dataset(dataset_path)
    graph_dataset = list(graph_to_embeddings_dataset(dataset, model, params))
    save_dataset(graph_dataset, save_path)


def load_model(ckpt_path):
    ckpt_path = os.path.abspath(ckpt_path)
    cfg = OmegaConf.load(os.path.join(ckpt_path, 'config.yaml'))
    model = GnnAutoencoder(**cfg.model)

    with ocp.CheckpointManager(ckpt_path, item_names=['params']) as ckpt_manager:
        params = ckpt_manager.restore(None, args=ocp.args.Composite(params=ocp.args.PyTreeRestore())).params

    return model, params


if __name__ == '__main__':
    model, params = load_model('../graphs/runs/gnn_ae_base')
    convert_dataset('../graphs/datasets/f_opt_dataset_graph.pkl.lz4', 'datasets/f_opt_dataset_graph.pkl.lz4', model, params)
    convert_dataset('../graphs/datasets/f_opt_val_dataset_graph.pkl.lz4', 'datasets/f_opt_val_dataset_graph.pkl.lz4', model, params)
    convert_dataset('../graphs/datasets/t_opt_dataset_graph.pkl.lz4', 'datasets/t_opt_dataset_graph.pkl.lz4', model, params)
    convert_dataset('../graphs/datasets/t_opt_val_dataset_graph.pkl.lz4', 'datasets/t_opt_val_dataset_graph.pkl.lz4', model, params)
    convert_dataset('../graphs/datasets/h_mab_dataset_graph.pkl.lz4', 'datasets/h_mab_dataset_graph.pkl.lz4', model, params)
    convert_dataset('../graphs/datasets/h_mab_val_dataset_graph.pkl.lz4', 'datasets/h_mab_val_dataset_graph.pkl.lz4', model, params)
    convert_dataset('../graphs/datasets/random_dataset_graph.pkl.lz4', 'datasets/random_dataset_graph.pkl.lz4', model, params)
    convert_dataset('../graphs/datasets/random_val_dataset_graph.pkl.lz4', 'datasets/random_val_dataset_graph.pkl.lz4', model, params)
