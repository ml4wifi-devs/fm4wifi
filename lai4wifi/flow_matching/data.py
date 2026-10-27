import os
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_ENABLE_X64'] = 'True'

import argparse
from glob import glob
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from lai4wifi.dataset.utils import load_dataset, save_dataset
from lai4wifi.graphs.utils import DataSource, make_batch
from lai4wifi.pipeline.checkpoints import ROOT, load_gnn
from lai4wifi.pipeline.conversion import prepare_context, random_tx, tx_to_jraph


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


def filter_fn_factory(val):
    def filter_fn(item):
        return ('val' in item) == val and 'random' not in item
    return filter_fn


def draw_shuffled_indices(n_items, n_total, seed):
    rng = np.random.default_rng(seed)
    idxs = []
    while len(idxs) < n_total:
        idxs.extend(rng.permutation(n_items).tolist())
    return idxs[:n_total]


def prebatch_dataset(dataset_path, save_path, batch_size, n_batches, val, seed):
    source = DataSource(dataset_path, filter_fn=filter_fn_factory(val))
    shuffled_idxs = draw_shuffled_indices(len(source), n_batches * batch_size, seed)
    batches = []

    for i in tqdm(range(n_batches), desc='Prebatching'):
        batch_idxs = shuffled_idxs[i * batch_size:(i + 1) * batch_size]
        graphs = [source[j][0] for j in batch_idxs]
        batches.append(make_batch(graphs))

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(batches, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flow matching training data')
    subparsers = parser.add_subparsers(dest='command', required=True)

    context = subparsers.add_parser('context', help='Encode configurations with random-observation contexts')
    context.add_argument('--gnn-path', type=str, default=os.path.join(ROOT, 'graphs', 'runs', 'gnn_b0.01_medium'))
    context.add_argument('--dataset-dir', type=str, default='../graphs/datasets')
    context.add_argument('--save-dir', type=str, default='datasets')
    context.add_argument('--n-timesteps', type=int, default=1)
    context.add_argument('--seed', type=int, default=42)

    prebatch = subparsers.add_parser('prebatch', help='Shuffle and batch encoded datasets')
    prebatch.add_argument('--dataset-path', type=str, default='datasets')
    prebatch.add_argument('--save-path', type=str, default='prebatched.pkl.lz4')
    prebatch.add_argument('--batch-size', type=int, default=32)
    prebatch.add_argument('--n-batches', type=int, default=5000)
    prebatch.add_argument('--seed', type=int, default=42)
    prebatch.add_argument('--val', action='store_true', default=False)
    args = parser.parse_args()

    if args.command == 'context':
        encode_fn, _ = load_gnn(args.gnn_path)
        os.makedirs(args.save_dir, exist_ok=True)

        for dataset_path in sorted(glob(os.path.join(args.dataset_dir, '*.pkl.lz4'))):
            save_path = os.path.join(args.save_dir, os.path.basename(dataset_path))
            convert_dataset(dataset_path, save_path, args.n_timesteps, args.seed, encode_fn)
    else:
        prebatch_dataset(args.dataset_path, args.save_path, args.batch_size, args.n_batches, args.val, args.seed)
