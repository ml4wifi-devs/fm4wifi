import os
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_ENABLE_X64'] = 'True'

import json
from argparse import ArgumentParser
from functools import partial

import jax
import jax.numpy as jnp
import jraph
import orbax.checkpoint as ocp
from mapc_research.envs.scenario_impl import residential_scenario
from omegaconf import OmegaConf
from tqdm import tqdm

from lai4wifi.flow_matching.model import FlowMatching, SurrogateModel
from lai4wifi.flow_matching.utils import connections_to_tx, tx_to_action, tx_to_conf, random_tx
from lai4wifi.graphs.attributes import *
from lai4wifi.graphs.model import GnnAutoencoder


def eval_with_sim(scenario, z, key, decode_fn):
    best, best_val = None, float('-inf')

    for zt in jraph.unbatch(decode_fn(z)):
        key, tx_key, scenario_key = jax.random.split(key, 3)
        tx = connections_to_tx(scenario.associations, tx_key, zt)
        data_rate, _ = scenario(scenario_key, *tx, return_internals=False)

        if data_rate > best_val:
            best = tx
            best_val = data_rate

    return best


def eval_with_surrogate(scenario, z, key, surrogate_fn):
    model_key, tx_key = jax.random.split(key)
    pred = surrogate_fn(z, rngs=model_key)
    best_idx = jnp.argmax(pred)
    best_zt = jraph.unbatch(z)[best_idx]
    best = connections_to_tx(scenario.associations, tx_key, best_zt)
    return best


def run_scenario(scenario, n_reps, n_steps, n_timesteps, n_samples_eval, seed, gen_fn, encode_fn, decode_fn, surrogate_fn):
    key = jax.random.PRNGKey(seed)
    runs = []
    actions = []

    for _ in range(n_reps):
        scenario.reset()
        runs.append([])
        actions.append([])
        context = []

        for _ in range(n_timesteps + n_steps):
            key, tx_key, sim_key = jax.random.split(key, 3)
            tx = random_tx(tx_key, scenario)
            data_rate, _, internals = scenario(sim_key, *tx, return_internals=True)
            context.append(tx_to_conf(scenario, *tx, internals, encode_fn))

        for step in range(n_steps):
            key, fm_key, eval_key, scenario_key = jax.random.split(key, 4)

            all = jax.tree.map(lambda *x: jnp.stack(x, axis=1), *context[step:step + n_timesteps])
            x = context[0]._replace(
                edges=all.edges,
                nodes=all.nodes,
                globals=all.globals
            )

            # https://arxiv.org/abs/2412.08296
            x_eval = jraph.batch([x for _ in range(n_samples_eval)])
            z = gen_fn(x_eval, rngs=fm_key)

            if surrogate_fn is not None:
                best = eval_with_surrogate(scenario, z, eval_key, surrogate_fn)
            else:
                best = eval_with_sim(scenario, z, eval_key, decode_fn)

            data_rate, _, internals = scenario(scenario_key, *best, return_internals=True)
            runs[-1].append(data_rate.item())
            actions[-1].append(tx_to_action(scenario.associations, internals, *best))

    return runs, actions


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--checkpoint', type=str, default='runs/flow_matching_base')
    args.add_argument('--surrogate', type=str)
    args.add_argument('--n_reps', type=int, default=1)
    args.add_argument('--n_samples_eval', type=int, default=1)
    args.add_argument('--output', type=str, default='all_results.json')
    args.add_argument('--seed', type=int, default=42)
    args = args.parse_args()

    if args.surrogate is not None:
        s_cfg = OmegaConf.load(os.path.join(args.surrogate, 'config.yaml'))
        with ocp.CheckpointManager(os.path.abspath(args.surrogate), item_names=['params']) as ckpt_manager:
            s_params = ckpt_manager.restore(None, args=ocp.args.Composite(params=ocp.args.PyTreeRestore())).params

        surrogate = SurrogateModel(**s_cfg.model)
        surrogate_fn = jax.jit(partial(surrogate.apply, s_params, training=False))
    else:
        surrogate_fn = None

    fm_cfg = OmegaConf.load(os.path.join(args.checkpoint, 'config.yaml'))
    with ocp.CheckpointManager(os.path.abspath(args.checkpoint), item_names=['params']) as ckpt_manager:
        fm_params = ckpt_manager.restore(None, args=ocp.args.Composite(params=ocp.args.PyTreeRestore())).params

    fm = FlowMatching(**fm_cfg.model)
    gen_fn = jax.jit(partial(fm.apply, fm_params, method='gen'))

    gnn_cfg = OmegaConf.load(os.path.join(fm_cfg.gnn_path, 'config.yaml'))
    with ocp.CheckpointManager(os.path.abspath(fm_cfg.gnn_path), item_names=['params']) as ckpt_manager:
        gnn_params = ckpt_manager.restore(None, args=ocp.args.Composite(params=ocp.args.PyTreeRestore())).params

    gnn = GnnAutoencoder(**gnn_cfg.model)
    encode_fn = jax.jit(partial(gnn.apply, gnn_params, method='encode'))
    decode_fn = jax.jit(partial(gnn.apply, gnn_params, method='decode'))

    scenarios = [
        residential_scenario(seed=12, n_steps=500, x_apartments=2, y_apartments=2, n_sta_per_ap=4, size=30.0)
    ]
    all_results = []

    for scenario in tqdm(scenarios, desc='Scenarios'):
        runs, actions = run_scenario(
            scenario, args.n_reps, scenario.n_steps, fm_cfg.train.n_timesteps, args.n_samples_eval, args.seed, 
            gen_fn, encode_fn, decode_fn, surrogate_fn
        )
        all_results.append([{
            'runs': runs,
            'actions': actions
        }])

    with open(args.output, 'w') as file:
        json.dump(all_results, file)
