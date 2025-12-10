import os
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_ENABLE_X64'] = 'True'

import jax
import orbax.checkpoint as ocp
from omegaconf import OmegaConf
from tqdm import tqdm

from lai4wifi.graphs.dataset import load_dataset, save_dataset
from lai4wifi.graphs.model import GnnAutoencoder
from lai4wifi.graphs.utils import make_batch, unbatch


def graph_to_embeddings_dataset(dataset, model, params):
    for dataset_item in tqdm(dataset):
        batch = make_batch(dataset_item.configurations)
        embeddings = jax.jit(model.apply, static_argnames='method')(params, batch, method='encode')
        dataset_item.configurations = unbatch(embeddings)
    
    return dataset


def convert_dataset(dataset_path, save_path, model, params):
    dataset = load_dataset(dataset_path)
    dataset = graph_to_embeddings_dataset(dataset, model, params)
    save_dataset(dataset, save_path)


def load_model(ckpt_path):
    ckpt_path = os.path.abspath(ckpt_path)
    cfg = OmegaConf.load(os.path.join(ckpt_path, 'config.yaml'))
    model = GnnAutoencoder(**cfg.model)

    with ocp.CheckpointManager(ckpt_path, item_names=['params']) as ckpt_manager:
        params = ckpt_manager.restore(None, args=ocp.args.Composite(params=ocp.args.PyTreeRestore())).params

    return model, params


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
