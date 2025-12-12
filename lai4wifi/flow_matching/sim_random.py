import os
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_ENABLE_X64'] = 'True'

import json
from argparse import ArgumentParser
from functools import partial

import jax
import jraph
import orbax.checkpoint as ocp
from mapc_research.envs.scenario_impl import random_scenario
from omegaconf import OmegaConf
from tqdm import tqdm

from lai4wifi.flow_matching.model import SurrogateModel
from lai4wifi.flow_matching.sim import eval_with_sim, eval_with_surrogate
from lai4wifi.flow_matching.utils import random_tx, tx_to_action, tx_to_conf
from lai4wifi.graphs.model import GnnAutoencoder
from lai4wifi.graphs.utils import clear_graph


def run_scenario(scenario, n_reps, n_steps, n_samples_eval, seed, surrogate_fn, encode_fn, decode_fn):
    key = jax.random.PRNGKey(seed)
    runs = []
    actions = []

    for _ in range(n_reps):
        scenario.reset()
        runs.append([])
        actions.append([])

        for _ in range(n_steps):
            key, rand_key, gnn_key, eval_key, scenario_key = jax.random.split(key, 5)
            graphs = []

            for _ in range(n_samples_eval):
                rand_key, tx_key, sim_key = jax.random.split(rand_key, 3)
                tx = random_tx(tx_key, scenario)
                _, _, internals = scenario(sim_key, *tx, return_internals=True)
                graph = clear_graph(tx_to_conf(scenario, *tx, internals, encode_fn))
                graphs.append(graph)

            x_eval = jraph.batch(graphs)
            z = encode_fn(x_eval, rngs=gnn_key)

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
    args.add_argument('--checkpoint', type=str, default='runs/surrogate_base')
    args.add_argument('--n_reps', type=int, default=1)
    args.add_argument('--n_samples_eval', type=int, default=32)
    args.add_argument('--output', type=str, default='all_results.json')
    args.add_argument('--seed', type=int, default=42)
    args = args.parse_args()

    cfg = OmegaConf.load(os.path.join(args.checkpoint, 'config.yaml'))
    with ocp.CheckpointManager(os.path.abspath(args.checkpoint), item_names=['params']) as ckpt_manager:
        s_params = ckpt_manager.restore(None, args=ocp.args.Composite(params=ocp.args.PyTreeRestore())).params

    surrogate = SurrogateModel(**cfg.model)
    surrogate_fn = jax.jit(partial(surrogate.apply, s_params, training=False))

    gnn_cfg = OmegaConf.load(os.path.join(cfg.gnn_path, 'config.yaml'))
    with ocp.CheckpointManager(os.path.abspath(cfg.gnn_path), item_names=['params']) as ckpt_manager:
        gnn_params = ckpt_manager.restore(None, args=ocp.args.Composite(params=ocp.args.PyTreeRestore())).params

    gnn = GnnAutoencoder(**gnn_cfg.model)
    encode_fn = jax.jit(partial(gnn.apply, gnn_params, method='encode'))
    decode_fn = jax.jit(partial(gnn.apply, gnn_params, method='decode'))

    scenarios = [
        random_scenario(seed=100, d_ap=75., d_sta=8., n_ap=2, n_sta_per_ap=5, n_steps=1000),
    ]
    all_results = []

    for scenario in tqdm(scenarios, desc='Scenarios'):
        runs, actions = run_scenario(
            scenario, args.n_reps, scenario.n_steps, args.n_samples_eval, args.seed, surrogate_fn, encode_fn, decode_fn
        )
        all_results.append([{
            'runs': runs,
            'actions': actions
        }])

    with open(args.output, 'w') as file:
        json.dump(all_results, file)
