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
        batch = make_batch(list(map(lambda c: c.graph, configurations)))
        batch = jax.jit(model.apply)(params, batch)
        batch = jraph.unpad_with_graphs(batch)
        yield jraph.unbatch(batch)


def convert_dataset(dataset_path, save_path, model, params):
    dataset = load_dataset(dataset_path)
    graph_dataset = list(graph_to_embeddings_dataset(dataset, model, params))
    save_dataset(graph_dataset, save_path)


def load_model(ckpt_path):
    ckpt_path = os.path.abspath(ckpt_path)
    cfg = OmegaConf.load(os.path.join(ckpt_path, 'config.yaml'))
    model = GnnAutoencoder(**cfg.model)

    with ocp.CheckpointManager(ckpt_path, item_handlers={'params': ocp.StandardCheckpointHandler()}) as ckpt_manager:
        sample = next(iter(load(os.path.join('../graphs', cfg.dataset), cfg.train.batch_size, jax.random.key(0))))
        params = ckpt_manager.restore(None, items={'params': model.init({'params': jax.random.key(0)}, sample)}).params

    return model, params


if __name__ == '__main__':
    model, params = load_model('../graphs/runs/gnn_ae_base')
    convert_dataset('../graphs/datasets/t_optimal_1k_50/train.pkl.lz4', 'datasets/t_optimal_1k_50/train.pkl.lz4', model, params)
    convert_dataset('../graphs/datasets/t_optimal_1k_50/val.pkl.lz4', 'datasets/t_optimal_1k_50/val.pkl.lz4', model, params)
