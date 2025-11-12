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
from tqdm import tqdm, trange

from lai4wifi.flow_matching.model import FlowMatching
from lai4wifi.flow_matching.utils import connections_to_tx, tx_to_action, tx_to_conf, random_tx, get_initial_configurations, get_network_desc
from lai4wifi.graphs.attributes import *
from lai4wifi.graphs.model import GnnAutoencoder


def run_scenario(scenario, n_reps, n_steps, n_timesteps, n_samples_eval, seed, gen_fn, encode_fn, decode_fn):
    key = jax.random.PRNGKey(seed)
    runs = []
    actions = []

    n_nodes = len(scenario.associations) + sum(len(stas) for stas in scenario.associations.values())

    for _ in trange(n_reps):
        scenario.reset()
        runs.append([])
        actions.append([])

        key, conf_key, context_key = jax.random.split(key, 3)
        _, graph, ap_sta_idxs = get_initial_configurations(scenario, n_timesteps, conf_key, encode_fn)
        
        context = []

        for _ in range(n_timesteps + n_steps):
            context_key, tx_key, sim_key = jax.random.split(context_key, 3)
            tx = random_tx(tx_key, scenario)
            data_rate, _, internals = scenario(sim_key, *tx, return_internals=True)
            context.append(tx_to_conf(scenario, *tx, internals, encode_fn))

        for step in range(n_steps):
            key, sel_key, fm_key, tx_key, scenario_key = jax.random.split(key, 5)

            selected_idx = jax.random.choice(sel_key, ap_sta_idxs).item()
            network_desc = get_network_desc(graph, selected_idx, encode_fn)
            all = jax.tree.map(lambda *x: jnp.stack(x, axis=1), network_desc, *context[step:step + n_timesteps])
            x = network_desc._replace(
                edges=all.edges,
                nodes=all.nodes,
                globals=all.globals
            )

            # https://arxiv.org/abs/2412.08296
            x_eval = jraph.batch([x for _ in range(n_samples_eval)])
            z = gen_fn(x_eval, rngs=fm_key)
            best, best_val = None, float('-inf')

            for zt in jraph.unbatch(decode_fn(z)):
                tx_key, subkey = jax.random.split(tx_key)
                tx = connections_to_tx(scenario.associations, n_nodes, subkey, zt)

                scenario_key, subkey = jax.random.split(scenario_key)
                data_rate, _ = scenario(subkey, *tx, return_internals=False)

                if data_rate > best_val:
                    best = tx
                    best_val = data_rate

            data_rate, _, internals = scenario(scenario_key, *best, return_internals=True)
            runs[-1].append(data_rate.item())
            actions[-1].append(tx_to_action(scenario.associations, internals, *best))

    return runs, actions


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--checkpoint', type=str, default='runs/flow_matching_base')
    args.add_argument('--n_reps', type=int, default=1)
    args.add_argument('--output', type=str, default='all_results.json')
    args.add_argument('--seed', type=int, default=42)
    args = args.parse_args()

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
            scenario, args.n_reps, scenario.n_steps, fm_cfg.train.n_timesteps, fm_cfg.train.n_eval_samples, args.seed, gen_fn, encode_fn, decode_fn
        )
        all_results.append([{
            'runs': runs,
            'actions': actions
        }])

    with open(args.output, 'w') as file:
        json.dump(all_results, file)
