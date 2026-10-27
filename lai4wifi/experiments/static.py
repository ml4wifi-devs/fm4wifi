import os
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_ENABLE_X64'] = 'True'

import json
from argparse import ArgumentParser
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from lai4wifi.envs.scenarios import TEST_SCENARIOS, create_scenario_eval_fn
from lai4wifi.graphs.utils import make_batch, unbatch
from lai4wifi.pipeline.checkpoints import FM_PATH, SURROGATE_PATH, load_fm, load_surrogate
from lai4wifi.pipeline.conversion import (
    create_scenario_cache,
    random_tx_fast,
    tx_to_action,
    tx_to_jraph_direct
)
from lai4wifi.pipeline.inference import (
    generate_candidates,
    iter_edge_batched_ranges,
    score_candidates,
    select_configs
)


def run_scenario(
        scenario, n_steps, n_timesteps, n_samples_eval, seed, ideal_mcs, deterministic,
        gen_fn, encode_fn, decode_fn, surrogate_fn, batch_size, use_simulator, top_k, n_eval_repeats
):
    key = jax.random.PRNGKey(seed)
    cache = create_scenario_cache(scenario)
    scenario_eval_fn = create_scenario_eval_fn(scenario)
    total_contexts = n_steps * n_timesteps
    context_graphs_raw = []

    with jax.default_device(jax.devices('cpu')[0]):
        for _ in range(total_contexts):
            key, tx_key, sim_key = jax.random.split(key, 3)
            tx = random_tx_fast(tx_key, cache)
            _, _, internals = scenario_eval_fn(sim_key, *tx, return_internals=True)
            context_graphs_raw.append(tx_to_jraph_direct(cache, *tx, internals))

    encoded_contexts = []

    for batch_start, batch_end in iter_edge_batched_ranges(context_graphs_raw, batch_size):
        batch = make_batch(context_graphs_raw[batch_start:batch_end])
        encoded = encode_fn(batch)
        encoded_contexts += unbatch(encoded)

    fm_inputs = []

    for step in range(n_steps):
        ctx_graphs = encoded_contexts[step * n_timesteps:(step + 1) * n_timesteps]
        stacked = jax.tree.map(lambda *x: jnp.stack(x, axis=1), *ctx_graphs)
        context = ctx_graphs[0]._replace(edges=stacked.edges, nodes=stacked.nodes, globals=stacked.globals)
        fm_inputs += [context] * n_samples_eval

    key, z_batches, all_txs = generate_candidates(key, fm_inputs, cache, gen_fn, decode_fn, batch_size, deterministic)
    key, scores_array = score_candidates(key, z_batches, all_txs, surrogate_fn, scenario_eval_fn, use_simulator)

    all_results = []

    with jax.default_device(jax.devices('cpu')[0]):
        for step in range(n_steps):
            step_start = step * n_samples_eval
            step_end = step_start + n_samples_eval
            selected = select_configs(
                scores_array[step_start:step_end],
                top_k,
            )
            results = []

            for step_idx, score in selected:
                flat_idx = step_start + step_idx
                eval_tx = all_txs[flat_idx]
                repeats = []

                for _ in range(n_eval_repeats):
                    key, scenario_key = jax.random.split(key)

                    if ideal_mcs:
                        tx_mat, tx_power, _ = eval_tx
                        data_rate, _, internals = scenario_eval_fn(
                            scenario_key, tx_mat, tx_power, return_internals=True
                        )
                        eval_tx = (tx_mat, tx_power, np.asarray(internals.mcs))
                    else:
                        data_rate, _, internals = scenario_eval_fn(scenario_key, *eval_tx, return_internals=True)

                    repeats.append({
                        'data_rate': data_rate.item(),
                        'action': tx_to_action(scenario.associations, internals, *eval_tx)
                    })

                results.append({
                    'score': score,
                    'runs': repeats
                })

            all_results.append({'configs': results})

    return all_results


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--checkpoint', type=str, default=FM_PATH)
    args.add_argument('--surrogate', type=str, default=SURROGATE_PATH)
    args.add_argument('--n_samples_eval', type=int, default=128)
    args.add_argument('--batch_size', type=int, default=4096)
    args.add_argument('--ideal_mcs', default=False, action='store_true')
    args.add_argument('--deterministic', default=False, action='store_true')
    args.add_argument('--use_simulator', default=False, action='store_true')
    args.add_argument('--scenarios', type=str, choices=list(TEST_SCENARIOS), default='residential')
    args.add_argument('--output', type=str, default='fm_results.json')
    args.add_argument('--seed', type=int, default=42)
    args.add_argument('--top_k', type=int, default=8)
    args.add_argument('--n_eval_repeats', type=int, default=5)
    args.add_argument('--n_steps', type=int, default=32)
    args = args.parse_args()

    surrogate_fn = None

    if not args.use_simulator:
        surrogate, s_params = load_surrogate(args.surrogate)
        surrogate_fn = jax.jit(partial(surrogate.apply, s_params, training=False))

    fm_cfg, gen_fn, encode_fn, decode_fn = load_fm(args.checkpoint)
    all_results = []

    for scenario in tqdm(TEST_SCENARIOS[args.scenarios], desc='Scenarios'):
        split_results = []

        for split_scenario, _ in scenario.split_scenario():
            split_results.append([run_scenario(
                split_scenario, args.n_steps, fm_cfg.train.n_timesteps, args.n_samples_eval,
                args.seed, args.ideal_mcs, args.deterministic, gen_fn, encode_fn, decode_fn,
                surrogate_fn, args.batch_size, args.use_simulator, args.top_k, args.n_eval_repeats
            )])

        all_results.append({'splits': split_results})

    with open(args.output, 'w') as file:
        json.dump(all_results, file)
