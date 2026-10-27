import os
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_ENABLE_X64'] = 'True'

import time
from argparse import ArgumentParser

import jax
import jax.numpy as jnp
import jraph
import numpy as np

from lai4wifi.envs.scenarios import TEST_SCENARIOS, create_scenario_eval_fn
from lai4wifi.pipeline.checkpoints import FM_PATH, load_fm
from lai4wifi.pipeline.conversion import (
    connections_to_tx_batch_fast,
    create_scenario_cache,
    random_tx_fast,
    tx_to_jraph_direct
)


def build_context_graphs(scenario, cache, n_timesteps, key):
    scenario_eval_fn = create_scenario_eval_fn(scenario)
    context_graphs = []

    with jax.default_device(jax.devices('cpu')[0]):
        for _ in range(n_timesteps):
            key, tx_key, sim_key = jax.random.split(key, 3)
            tx = random_tx_fast(tx_key, cache)
            _, _, internals = scenario_eval_fn(sim_key, *tx, return_internals=True)
            context_graphs.append(tx_to_jraph_direct(cache, *tx, internals))

    return context_graphs


def prepare_fm_input(encoded_contexts, n_samples):
    stacked = jax.tree.map(lambda *x: jnp.stack(x, axis=1), *encoded_contexts)
    context = encoded_contexts[0]._replace(
        edges=stacked.edges, nodes=stacked.nodes, globals=stacked.globals
    )
    return [context] * n_samples


def benchmark_fn(fn, n_warmup, n_repeats):
    for _ in range(n_warmup):
        fn()

    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.array(times)


def print_stats(label, times, n_samples):
    print(f"  {label}:  mean={times.mean()*1000:.2f}ms  "
          f"std={times.std()*1000:.2f}ms  "
          f"min={times.min()*1000:.2f}ms  "
          f"max={times.max()*1000:.2f}ms  "
          f"({n_samples / times.mean():.1f} samples/s)")


def main():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=FM_PATH)
    parser.add_argument('--n_samples', type=int, default=128, help='Batch size (number of generated samples)')
    parser.add_argument('--n_warmup', type=int, default=3)
    parser.add_argument('--n_repeats', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--deterministic', default=False, action='store_true')
    args = parser.parse_args()

    key = jax.random.PRNGKey(args.seed)
    fm_cfg, gen_fn, encode_fn, decode_fn = load_fm(args.checkpoint)

    n_timesteps = fm_cfg.train.n_timesteps

    print(f"Device: {jax.devices()[0]}")
    print(f"Batch size (n_samples): {args.n_samples}")
    print(f"Context timesteps: {n_timesteps}")
    print(f"Warmup: {args.n_warmup}, Repeats: {args.n_repeats}")
    print()

    for scenario in TEST_SCENARIOS['scaling']:
        for split_scenario, _ in scenario.split_scenario():
            cache = create_scenario_cache(split_scenario)
            n_aps = len(split_scenario.associations)
            n_stas = sum(len(s) for s in split_scenario.associations.values())

            print(f"=== {split_scenario} ===")
            print(f"  APs: {n_aps}, STAs: {n_stas}, nodes: {cache.n_nodes}, edges/graph: {cache.n_edges}")

            # Pre-build context graphs (CPU)
            key, ctx_key = jax.random.split(key)
            context_graphs = build_context_graphs(split_scenario, cache, n_timesteps, ctx_key)

            # --- Benchmark: encode_fn ---
            ctx_batch = jraph.batch(context_graphs)

            def bench_encode():
                enc = encode_fn(ctx_batch)
                enc.edges.block_until_ready()

            encode_times = benchmark_fn(bench_encode, args.n_warmup, args.n_repeats)
            print_stats("encode", encode_times, args.n_samples)

            # --- Benchmark: gen_fn ---
            enc = encode_fn(ctx_batch)
            enc_ctx = jraph.unbatch(enc)
            fm_inputs = prepare_fm_input(enc_ctx, args.n_samples)
            fm_batch = jraph.batch(fm_inputs)

            def bench_gen():
                nonlocal key
                key, fm_key = jax.random.split(key)
                z = gen_fn(fm_batch, rngs=fm_key)
                z.edges.block_until_ready()

            gen_times = benchmark_fn(bench_gen, args.n_warmup, args.n_repeats)
            print_stats("gen", gen_times, args.n_samples)

            # --- Benchmark: decode_fn ---
            key, fm_key = jax.random.split(key)
            z = gen_fn(fm_batch, rngs=fm_key)

            def bench_decode():
                decoded = decode_fn(z)
                decoded.edges.active.block_until_ready()

            decode_times = benchmark_fn(bench_decode, args.n_warmup, args.n_repeats)
            print_stats("decode", decode_times, args.n_samples)

            # --- Benchmark: connections_to_tx ---
            decoded = decode_fn(z)

            def bench_tx():
                tx_keys = jax.random.split(key, args.n_samples)
                connections_to_tx_batch_fast(
                    jnp.asarray(tx_keys), decoded, cache, args.n_samples, args.deterministic
                )

            tx_times = benchmark_fn(bench_tx, args.n_warmup, args.n_repeats)
            print_stats("connections_to_tx", tx_times, args.n_samples)

            # --- Benchmark: encode + gen + decode (GPU total) ---
            def bench_gpu_total():
                nonlocal key
                key, fm_key = jax.random.split(key)
                enc = encode_fn(ctx_batch)
                enc_c = jraph.unbatch(enc)
                fm_in = prepare_fm_input(enc_c, args.n_samples)
                fm_b = jraph.batch(fm_in)
                z = gen_fn(fm_b, rngs=fm_key)
                decoded = decode_fn(z)
                decoded.edges.active.block_until_ready()

            gpu_total_times = benchmark_fn(bench_gpu_total, args.n_warmup, args.n_repeats)
            print_stats("encode+gen+decode", gpu_total_times, args.n_samples)

            # --- Benchmark: full pipeline (CPU context + all GPU) ---
            def bench_full():
                nonlocal key
                key, ctx_k, fm_key = jax.random.split(key, 3)
                ctx_g = build_context_graphs(split_scenario, cache, n_timesteps, ctx_k)
                ctx_b = jraph.batch(ctx_g)
                enc = encode_fn(ctx_b)
                enc_c = jraph.unbatch(enc)
                fm_in = prepare_fm_input(enc_c, args.n_samples)
                fm_b = jraph.batch(fm_in)
                z = gen_fn(fm_b, rngs=fm_key)
                decoded = decode_fn(z)
                tx_keys = jax.random.split(key, args.n_samples)
                connections_to_tx_batch_fast(
                    jnp.asarray(tx_keys), decoded, cache, args.n_samples, args.deterministic
                )

            full_times = benchmark_fn(bench_full, args.n_warmup, args.n_repeats)
            print_stats("full pipeline", full_times, args.n_samples)

            print()


if __name__ == '__main__':
    main()
