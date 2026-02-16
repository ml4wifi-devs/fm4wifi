import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'False'
os.environ['JAX_ENABLE_X64'] = 'True'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from lai4wifi.dataset.utils import save_dataset
from lai4wifi.flow_matching.utils import DataSource
from lai4wifi.graphs.utils import make_batch


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
    args = argparse.ArgumentParser()
    args.add_argument('--dataset-path', type=str, default='datasets')
    args.add_argument('--save-path', type=str, default='prebatched.pkl.lz4')
    args.add_argument('--batch-size', type=int, default=32)
    args.add_argument('--n-batches', type=int, default=5000)
    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--val', action='store_true', default=False)
    args = args.parse_args()
    
    prebatch_dataset(args.dataset_path, args.save_path, args.batch_size, args.n_batches, args.val, args.seed)
