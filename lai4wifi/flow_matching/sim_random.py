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
from omegaconf import OmegaConf
from tqdm import tqdm, trange

from lai4wifi.flow_matching.sim import eval_with_sim, eval_with_surrogate
from lai4wifi.flow_matching.utils import TEST_SCENARIOS, random_tx, tx_to_action, tx_to_conf
from lai4wifi.graphs.model import GnnAutoencoder
from lai4wifi.graphs.utils import make_batch
from lai4wifi.surrogate.model import SurrogateModel


def run_scenario(
        scenario, n_reps, n_steps, n_samples_eval, seed, ideal_mcs, use_surrogate,
        surrogate_fn, encode_fn, decode_fn
):
    key = jax.random.PRNGKey(seed)
    runs = []
    actions = []

    for _ in range(n_reps):
        scenario.reset()
        runs.append([])
        actions.append([])

        for _ in trange(n_steps, desc='Steps'):
            key, rand_key, eval_key, scenario_key = jax.random.split(key, 4)
            graphs = []

            for _ in range(n_samples_eval):
                rand_key, tx_key, sim_key = jax.random.split(rand_key, 3)
                tx = random_tx(tx_key, scenario)
                _, _, internals = scenario(sim_key, *tx, return_internals=True)
                graphs.append(tx_to_conf(scenario, *tx, internals, encode_fn, add_success=False))

            z_eval = make_batch(graphs)
            configs = decode_fn(z_eval)

            if use_surrogate:
                best = eval_with_surrogate(scenario, configs, internals, eval_key, surrogate_fn, encode_fn)
            else:
                best = eval_with_sim(scenario, configs, eval_key)
            
            if ideal_mcs:
                best_tx, best_tx_power, _ = best
                best = (best_tx, best_tx_power)
                data_rate, _, internals = scenario(scenario_key, *best, return_internals=True)
                best = (*best, internals.mcs)
            else:
                data_rate, _, internals = scenario(scenario_key, *best, return_internals=True)

            runs[-1].append(data_rate.item())
            actions[-1].append(tx_to_action(scenario.associations, internals, *best))

    return runs, actions


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--surrogate', type=str)
    args.add_argument('--use_surrogate', default=False, action='store_true')
    args.add_argument('--n_reps', type=int, default=1)
    args.add_argument('--n_samples_eval', type=int, default=1)
    args.add_argument('--ideal_mcs', default=False, action='store_true')
    args.add_argument('--output', type=str, default='all_results.json')
    args.add_argument('--seed', type=int, default=42)
    args = args.parse_args()

    cfg = OmegaConf.load(os.path.join(args.surrogate, 'config.yaml'))
    with ocp.CheckpointManager(os.path.abspath(args.surrogate), item_names=['params']) as ckpt_manager:
        s_params = ckpt_manager.restore(None, args=ocp.args.Composite(params=ocp.args.PyTreeRestore())).params

    surrogate = SurrogateModel(**cfg.model)
    surrogate_fn = jax.jit(partial(surrogate.apply, s_params, training=False))

    gnn_cfg = OmegaConf.load(os.path.join(cfg.gnn_path, 'config.yaml'))
    with ocp.CheckpointManager(os.path.abspath(cfg.gnn_path), item_names=['params']) as ckpt_manager:
        gnn_params = ckpt_manager.restore(None, args=ocp.args.Composite(params=ocp.args.PyTreeRestore())).params

    gnn = GnnAutoencoder(**gnn_cfg.model)
    encode_fn = jax.jit(partial(gnn.apply, gnn_params, method='encode'))
    decode_fn = jax.jit(partial(gnn.apply, gnn_params, method='decode'))

    all_results = []

    for scenario in tqdm(TEST_SCENARIOS, desc='Scenarios'):
        runs, actions = run_scenario(
            scenario, args.n_reps, scenario.n_steps, args.n_samples_eval, args.seed, 
            args.ideal_mcs, args.use_surrogate, surrogate_fn, encode_fn, decode_fn
        )
        all_results.append([{
            'runs': runs,
            'actions': actions
        }])

    with open(args.output, 'w') as file:
        json.dump(all_results, file)
