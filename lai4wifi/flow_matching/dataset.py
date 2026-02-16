import os
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_ENABLE_X64'] = 'True'

from functools import partial

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from omegaconf import OmegaConf
from tqdm import tqdm

from lai4wifi.dataset.utils import load_dataset, save_dataset
from lai4wifi.flow_matching.utils import prepare_context, random_tx, tx_to_jraph
from lai4wifi.graphs.model import GnnAutoencoder


def load_encode_fn(gnn_path):
    gnn_cfg = OmegaConf.load(os.path.join(gnn_path, 'config.yaml'))
    with ocp.CheckpointManager(os.path.abspath(gnn_path), item_names=['params']) as ckpt_manager:
        fallback_sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
        restore = ocp.args.Composite(params=ocp.args.StandardRestore(fallback_sharding=fallback_sharding))
        gnn_params = ckpt_manager.restore(None, args=restore).params

    gnn = GnnAutoencoder(**gnn_cfg.model)
    return jax.jit(partial(gnn.apply, gnn_params, method='encode'))


def precompute_sample(target_conf, scenario, key, n_timesteps, encode_fn):
    target_conf = encode_fn(target_conf)
    
    with jax.default_device(jax.devices('cpu')[0]):
        context = []

        for _ in range(n_timesteps):
            key, tx_key, sim_key = jax.random.split(key, 3)
            tx = random_tx(tx_key, scenario)
            _, _, internals = scenario(sim_key, *tx, return_internals=True)
            context.append(tx_to_jraph(scenario, *tx, internals))

    context = prepare_context(context, encode_fn)

    return target_conf._replace(
        edges=jnp.concatenate([context.edges, target_conf.edges[:, None]], axis=1),
        nodes=jnp.concatenate([context.nodes, target_conf.nodes[:, None]], axis=1),
        globals=jnp.concatenate([context.globals, target_conf.globals[:, None]], axis=1),
    )


def precompute_dataset(dataset, n_timesteps, seed, encode_fn):
    key = jax.random.PRNGKey(seed)

    for dataset_item in tqdm(dataset, desc='Scenarios'):
        for i in range(len(dataset_item.configurations)):
            target_conf = dataset_item.configurations[i]
            key, conf_key = jax.random.split(key)
            dataset_item.configurations[i] = precompute_sample(target_conf, dataset_item.scenario, conf_key, n_timesteps, encode_fn)

    return dataset


def convert_dataset(dataset_path, save_path, n_timesteps, seed, encode_fn):
    dataset = load_dataset(dataset_path)
    dataset = precompute_dataset(dataset, n_timesteps, seed, encode_fn)
    save_dataset(dataset, save_path)


if __name__ == '__main__':
    n_timesteps = 7
    seed = 42
    
    encode_fn = load_encode_fn('../graphs/runs/gnn_b0.01_medium')
    
    convert_dataset('../graphs/datasets/f_opt_dataset.pkl.lz4', 'datasets/f_opt_dataset.pkl.lz4', n_timesteps, seed, encode_fn)
    convert_dataset('../graphs/datasets/f_opt_val_dataset.pkl.lz4', 'datasets/f_opt_val_dataset.pkl.lz4', n_timesteps, seed, encode_fn)
    convert_dataset('../graphs/datasets/t_opt_dataset.pkl.lz4', 'datasets/t_opt_dataset.pkl.lz4', n_timesteps, seed, encode_fn)
    convert_dataset('../graphs/datasets/t_opt_val_dataset.pkl.lz4', 'datasets/t_opt_val_dataset.pkl.lz4', n_timesteps, seed, encode_fn)
    convert_dataset('../graphs/datasets/h_mab_dataset.pkl.lz4', 'datasets/h_mab_dataset.pkl.lz4', n_timesteps, seed, encode_fn)
    convert_dataset('../graphs/datasets/h_mab_val_dataset.pkl.lz4', 'datasets/h_mab_val_dataset.pkl.lz4', n_timesteps, seed, encode_fn)
    convert_dataset('../graphs/datasets/random_dataset.pkl.lz4', 'datasets/random_dataset.pkl.lz4', n_timesteps, seed, encode_fn)
    convert_dataset('../graphs/datasets/random_val_dataset.pkl.lz4', 'datasets/random_val_dataset.pkl.lz4', n_timesteps, seed, encode_fn)
