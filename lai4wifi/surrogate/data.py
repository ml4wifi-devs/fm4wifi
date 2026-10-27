import os
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_ENABLE_X64'] = 'True'

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from lai4wifi.dataset.utils import load_dataset, save_dataset
from lai4wifi.envs.scenarios import create_scenario_eval_fn
from lai4wifi.graphs.utils import DataSource, make_batch, unbatch
from lai4wifi.pipeline.checkpoints import FM_PATH, load_fm, load_fm_gnn
from lai4wifi.pipeline.conversion import (
    connections_to_tx_batch_fast,
    create_scenario_cache,
    prepare_context,
    random_tx_fast,
    tx_to_jraph_direct
)
from lai4wifi.surrogate.utils import RATE_MEAN, RATE_STD


def filter_fn_factory(val):
    return lambda item: ('val' in item) == val


def merge_chunks(chunk_dir, save_path, cleanup=True):
    """Merge saved chunk files into a single dataset."""
    chunk_dir = Path(chunk_dir)
    save_path = Path(save_path)
    chunk_files = sorted(chunk_dir.glob('chunk_*.pkl.lz4'))
    if not chunk_files:
        raise FileNotFoundError(f'No chunk files found in {chunk_dir}')

    print(f'Found {len(chunk_files)} chunks in {chunk_dir}')
    all_batches = []
    for cp in chunk_files:
        data = load_dataset(cp)
        print(f'  Loaded {cp.name}: {len(data)} batches')
        all_batches.extend(data)
        del data

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(all_batches, save_path)
    print(f'Saved {len(all_batches)} batches to {save_path}')
    del all_batches

    if cleanup:
        for cp in chunk_files:
            cp.unlink()
        chunk_dir.rmdir()
        print(f'Cleaned up {chunk_dir}')


def generate_fm_dataset(
    dataset_path, fm_path, save_path, n_candidates, max_edges_gen, max_edges_save,
    seed, val, chunk_size
):
    """Generate surrogate training data from FM-generated configs.

    For each scenario in the graph dataset:
    1. Generate random tx as FM context (same as sim.py)
    2. FM generates n_candidates new configs (latent graphs z)
    3. Decode → tx → simulate to get true data rates
    4. Save z (FM latent graph) + normalized rate for surrogate training
    """
    fm_cfg, gen_fn, encode_fn, decode_fn = load_fm(fm_path)
    n_timesteps = fm_cfg.train.n_timesteps

    source = DataSource(dataset_path, filter_fn=filter_fn_factory(val))
    print(f'Loaded {len(source)} samples (val={val})')

    # Extract unique scenarios
    scenario_map = {}
    for i in range(len(source)):
        _, scenario = source[i]
        scenario_map[id(scenario)] = scenario

    scenarios = list(scenario_map.values())
    print(f'Found {len(scenarios)} unique scenarios')

    key = jax.random.PRNGKey(seed)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_dir = save_path.parent / f'.chunks_{save_path.stem}'
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_paths = []

    all_graphs = []
    all_rates = []

    for si, scenario in enumerate(tqdm(scenarios, desc='Scenarios')):
        for split_scenario, _ in scenario.split_scenario():
            cache = create_scenario_cache(split_scenario)
            eval_fn = create_scenario_eval_fn(split_scenario)

            # Generate random context (same as sim.py lines 332-337)
            context_graphs_raw = []
            with jax.default_device(jax.devices('cpu')[0]):
                for _ in range(n_timesteps):
                    key, tx_key, sim_key = jax.random.split(key, 3)
                    tx = random_tx_fast(tx_key, cache)
                    _, _, internals = eval_fn(sim_key, *tx, return_internals=True)
                    context_graphs_raw.append(tx_to_jraph_direct(cache, *tx, internals))

                # Encode context and prepare FM input
                context = prepare_context(context_graphs_raw, encode_fn)
                fm_inputs = [context] * n_candidates

            # FM generation in sub-batches (limited by max_edges_gen for GPU memory)
            edges_per_input = int(np.asarray(fm_inputs[0].n_edge).sum())
            gen_batch_size = max(1, max_edges_gen // edges_per_input)

            for b_start in range(0, n_candidates, gen_batch_size):
                b_end = min(b_start + gen_batch_size, n_candidates)
                curr_size = b_end - b_start

                key, fm_key = jax.random.split(key)
                batch = make_batch(fm_inputs[b_start:b_end])
                z = gen_fn(batch, rngs=fm_key)
                decoded = decode_fn(z)

                with jax.default_device(jax.devices('cpu')[0]):
                    z_list = unbatch(z)
                    tx_keys = []
                    for _ in range(curr_size):
                        key, tk = jax.random.split(key)
                        tx_keys.append(tk)

                    tx_batch, tx_power_batch, mcs_batch = connections_to_tx_batch_fast(
                        jnp.asarray(tx_keys), decoded, cache, curr_size, deterministic=False
                    )

                    for j in range(curr_size):
                        tx = (tx_batch[j], tx_power_batch[j], mcs_batch[j])

                        key, sim_key = jax.random.split(key)
                        data_rate, _, _ = eval_fn(sim_key, *tx, return_internals=True)

                        rate = (float(data_rate) - RATE_MEAN) / RATE_STD

                        all_graphs.append(z_list[j])
                        all_rates.append(rate)

        # Free LLVM compiled code to prevent compilation OOM
        if si % 50 == 0:
            jax.clear_caches()

    # Shuffle and batch by edge count
    print(f'Total samples: {len(all_graphs)}')
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(all_graphs)).tolist()

    def get_n_edges(graph):
        return int(np.asarray(graph.n_edge).sum())

    batches = []
    i = 0
    while i < len(indices):
        batch_graphs = []
        batch_rates = []
        total_edges = 0

        while i < len(indices):
            idx = indices[i]
            n_edges = get_n_edges(all_graphs[idx])
            if batch_graphs and total_edges + n_edges > max_edges_save:
                break
            batch_graphs.append(all_graphs[idx])
            batch_rates.append(all_rates[idx])
            total_edges += n_edges
            i += 1

        if not batch_graphs:
            break

        batch = make_batch(batch_graphs)
        rates_arr = np.asarray(batch_rates + [0.0])  # Padding for jraph
        batches.append((batch, rates_arr))

        if len(batches) >= chunk_size:
            chunk_path = chunk_dir / f'chunk_{len(chunk_paths):04d}.pkl.lz4'
            save_dataset(batches, chunk_path)
            print(f'Saved chunk {len(chunk_paths)} ({len(batches)} batches)')
            chunk_paths.append(chunk_path)
            del batches
            batches = []

    if batches:
        chunk_path = chunk_dir / f'chunk_{len(chunk_paths):04d}.pkl.lz4'
        save_dataset(batches, chunk_path)
        print(f'Saved chunk {len(chunk_paths)} ({len(batches)} batches)')
        chunk_paths.append(chunk_path)
        del batches

    merge_chunks(chunk_dir, save_path)
    print(f'Done! Saved to {save_path}')


def one_hot_from_logits(logits):
    idx = jnp.argmax(logits, axis=-1)
    return jax.nn.one_hot(idx, logits.shape[-1], dtype=logits.dtype)


def to_no_encode_graph(graph):
    edges = graph.edges
    hard_edges = edges.__class__(
        active=one_hot_from_logits(edges.active),
        mcs=one_hot_from_logits(edges.mcs),
        rssi=edges.rssi,
        selected=one_hot_from_logits(edges.selected),
        success=jnp.zeros_like(edges.success),
        tx_power=one_hot_from_logits(edges.tx_power),
        type=one_hot_from_logits(edges.type),
    )
    return graph._replace(edges=hard_edges)


def infer_output_path(input_path, suffix):
    input_path = Path(input_path)
    name = input_path.name
    if name.endswith('.pkl.lz4'):
        return input_path.with_name(f'{name[:-8]}{suffix}.pkl.lz4')
    return input_path.with_name(f'{input_path.stem}{suffix}{input_path.suffix}')


def convert_file(input_path, output_path, decode_fn, keep_soft):
    print(f'Loading {input_path}')
    data = load_dataset(input_path)
    print(f'Loaded {len(data)} batches')

    for i, (batch, rates) in enumerate(tqdm(data, desc=f'Converting {Path(input_path).name}')):
        decoded = decode_fn(batch)
        if not keep_soft:
            decoded = to_no_encode_graph(decoded)
        data[i] = (decoded, rates)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(data, output_path)
    print(f'Saved converted dataset to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Surrogate training data: FM-generated configs with simulated rates')
    subparsers = parser.add_subparsers(dest='command', required=True)

    generate = subparsers.add_parser('generate', help='Generate AE-encoded datasets from FM samples')
    generate.add_argument('--dataset-path', type=str, default='../graphs/datasets',
                          help='Path to graph datasets (used for scenarios + context)')
    generate.add_argument('--fm-path', type=str, default=FM_PATH, help='Path to FM checkpoint')
    generate.add_argument('--save-path', type=str, default='datasets/prebatched.pkl.lz4')
    generate.add_argument('--n-candidates', type=int, default=32,
                          help='Number of FM-generated configs per scenario')
    generate.add_argument('--max-edges-gen', type=int, default=2048,
                          help='Max edges per FM generation batch (GPU memory)')
    generate.add_argument('--max-edges-save', type=int, default=16384,
                          help='Max edges per saved training batch')
    generate.add_argument('--seed', type=int, default=42)
    generate.add_argument('--val', action='store_true', default=False)
    generate.add_argument('--chunk-size', type=int, default=5000)

    decode = subparsers.add_parser('decode', help='Decode AE-encoded datasets into raw graph datasets')
    decode.add_argument('--fm-path', type=str, default=FM_PATH, help='Path to FM checkpoint (used to resolve gnn_path)')
    decode.add_argument('--input-paths', nargs='+', required=True,
                        help='One or more input prebatched AE-encoded datasets')
    decode.add_argument('--output-paths', nargs='*', default=None,
                        help='Optional output paths; must match number of --input-paths')
    decode.add_argument('--output-suffix', type=str, default='_raw',
                        help='Suffix for inferred output names when --output-paths is omitted')
    decode.add_argument('--keep-soft', action='store_true', default=False,
                        help='Keep decoder outputs as soft logits (default: hard no-encode projection)')
    args = parser.parse_args()

    if args.command == 'generate':
        generate_fm_dataset(
            args.dataset_path, args.fm_path, args.save_path,
            args.n_candidates, args.max_edges_gen, args.max_edges_save,
            args.seed, args.val, args.chunk_size
        )
    else:
        if args.output_paths is None or len(args.output_paths) == 0:
            output_paths = [infer_output_path(p, args.output_suffix) for p in args.input_paths]
        else:
            if len(args.output_paths) != len(args.input_paths):
                raise ValueError('--output-paths must have the same number of entries as --input-paths')
            output_paths = [Path(p) for p in args.output_paths]

        _, decode_fn = load_fm_gnn(args.fm_path)
        for input_path, output_path in zip(args.input_paths, output_paths):
            convert_file(input_path, output_path, decode_fn, keep_soft=args.keep_soft)
