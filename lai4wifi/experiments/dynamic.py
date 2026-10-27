import os
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_ENABLE_X64'] = 'True'

import json
from argparse import ArgumentParser
from functools import partial

import jax
import jax.numpy as jnp
from tqdm import trange

from lai4wifi.envs.mobility import add_env_args, create_env
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


def generate_config_pool(
        key, scenario_key, scenario, scenario_eval_fn, n_timesteps, n_samples_eval, deterministic,
        gen_fn, encode_fn, decode_fn, surrogate_fn, batch_size, use_simulator, top_k
):
    cache = create_scenario_cache(scenario)
    context_graphs_raw = []

    with jax.default_device(jax.devices('cpu')[0]):
        for _ in range(n_timesteps):
            key, tx_key = jax.random.split(key)
            scenario_key, sim_key = jax.random.split(scenario_key)
            tx = random_tx_fast(tx_key, cache)
            _, _, internals = scenario_eval_fn(sim_key, *tx, return_internals=True)
            context_graphs_raw.append(tx_to_jraph_direct(cache, *tx, internals))

    encoded_contexts = []
    for batch_start, batch_end in iter_edge_batched_ranges(context_graphs_raw, batch_size):
        batch = make_batch(context_graphs_raw[batch_start:batch_end])
        encoded_contexts += unbatch(encode_fn(batch))

    stacked = jax.tree.map(lambda *x: jnp.stack(x, axis=1), *encoded_contexts)
    context = encoded_contexts[0]._replace(edges=stacked.edges, nodes=stacked.nodes, globals=stacked.globals)
    fm_inputs = [context] * n_samples_eval

    key, z_batches, all_txs = generate_candidates(key, fm_inputs, cache, gen_fn, decode_fn, batch_size, deterministic)
    _, scores_array = score_candidates(scenario_key, z_batches, all_txs, surrogate_fn, scenario_eval_fn, use_simulator)
    selected = select_configs(scores_array, top_k)

    config_pool = [all_txs[idx] for idx, _ in selected]
    config_scores = [float(score) for _, score in selected]
    return config_pool, config_scores


def run_scenario(
        env, n_steps, n_timesteps, n_samples_eval, scenario_seed, model_seed, ideal_mcs, deterministic,
        gen_fn, encode_fn, decode_fn, surrogate_fn, batch_size, use_simulator, top_k,
        regeneration_policy, regeneration_interval_steps
):
    scenario_key = env.reset(jax.random.PRNGKey(scenario_seed))
    model_key = jax.random.PRNGKey(model_seed)

    config_pool = []
    config_scores = []
    rr_idx = 0
    last_regeneration_step = None
    all_results = []

    for step in trange(n_steps):
        scenario_key, config_key, info = env.step(scenario_key, step)
        regenerate = step == 0 or info.get('topology_changed', False)

        if not regenerate and regeneration_policy == 'interval':
            regenerate = (step - last_regeneration_step) >= regeneration_interval_steps

        if regenerate:
            generate_key, model_key = jax.random.split(model_key)
            config_pool, config_scores = generate_config_pool(
                generate_key, config_key, env.scenario, env.eval_fn, n_timesteps, n_samples_eval, deterministic,
                gen_fn, encode_fn, decode_fn, surrogate_fn, batch_size, use_simulator, top_k,
            )
            rr_idx = 0
            last_regeneration_step = step

        if not config_pool:
            raise ValueError('Empty configuration pool after regeneration.')

        pool_idx = rr_idx % len(config_pool)
        rr_idx += 1
        eval_tx = config_pool[pool_idx]

        scenario_key, eval_key = jax.random.split(scenario_key)
        if ideal_mcs:
            tx_mat, tx_power, _ = eval_tx
            data_rate, _, internals = env.eval_fn(eval_key, tx_mat, tx_power, return_internals=True)
            eval_tx = (tx_mat, tx_power, internals.mcs)
        else:
            data_rate, _, internals = env.eval_fn(eval_key, *eval_tx, return_internals=True)

        all_results.append({
            'step': step,
            'regenerated': regenerate,
            **info,
            'pool_index': pool_idx,
            'pool_size': len(config_pool),
            'score': config_scores[pool_idx],
            'data_rate': float(data_rate.item()),
            'action': tx_to_action(env.scenario.associations, internals, *eval_tx),
        })

    return all_results


if __name__ == '__main__':
    args = ArgumentParser()
    add_env_args(args)
    args.add_argument('--checkpoint', type=str, default=FM_PATH)
    args.add_argument('--surrogate', type=str, default=SURROGATE_PATH)
    args.add_argument('--output', type=str, default='fm_dynamic_results.json')
    args.add_argument('--model_seed', type=int, default=0)
    args.add_argument('--n_samples_eval', type=int, default=128)
    args.add_argument('--batch_size', type=int, default=2048)
    args.add_argument('--ideal_mcs', default=False, action='store_true')
    args.add_argument('--deterministic', default=False, action='store_true')
    args.add_argument('--use_simulator', default=False, action='store_true')
    args.add_argument('--top_k', type=int, default=8)
    args.add_argument('--regeneration_policy', choices=['once', 'interval'], default='once')
    args.add_argument('--regeneration_interval_s', type=float, default=1.0)
    args = args.parse_args()

    env, n_steps, metadata = create_env(args)
    regeneration_interval_steps = max(1, int(round(args.regeneration_interval_s / args.sim_dt)))

    surrogate_fn = None

    if not args.use_simulator:
        surrogate, s_params = load_surrogate(args.surrogate)
        surrogate_fn = jax.jit(partial(surrogate.apply, s_params, training=False))

    fm_cfg, gen_fn, encode_fn, decode_fn = load_fm(args.checkpoint)

    results = run_scenario(
        env, n_steps, fm_cfg.train.n_timesteps, args.n_samples_eval, args.scenario_seed, args.model_seed,
        args.ideal_mcs, args.deterministic, gen_fn, encode_fn, decode_fn, surrogate_fn, args.batch_size,
        args.use_simulator, args.top_k, args.regeneration_policy, regeneration_interval_steps,
    )

    if args.env == 'enterprise':
        metadata['topology_change_steps'] = env.topology_change_steps
    else:
        metadata = {
            'regeneration_policy': args.regeneration_policy,
            'regeneration_interval_s': args.regeneration_interval_s,
            **metadata,
        }

    with open(args.output, 'w') as file:
        json.dump({
            'scenario': str(env.scenario),
            'n_steps': n_steps,
            'sim_time_s': n_steps * args.sim_dt,
            'scenario_seed': args.scenario_seed,
            'model_seed': args.model_seed,
            **metadata,
            'results': results,
        }, file)
