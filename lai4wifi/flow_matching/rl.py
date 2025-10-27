import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'

import json
from argparse import ArgumentParser
from copy import deepcopy
from functools import partial

import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
import orbax.checkpoint as ocp
from mapc_research.envs.test_scenarios import EXEMPLARY_SCENARIOS
from omegaconf import OmegaConf
from tqdm import tqdm, trange

from lai4wifi.autoencoder.model import GnnAutoencoder
from lai4wifi.flow_matching.model import FlowMatching
from lai4wifi.flow_matching.sim import get_initial_configurations, get_next_context, connections_to_tx
from lai4wifi.flow_matching.training import get_network_desc


# https://www.arxiv.org/abs/2509.16117
N_SAMPLES_TRAIN = 128
BS_TRAIN = 32
N_SAMPLES_EVAL = 32
BETA = 1.0


def run_scenario(scenario, n_reps, n_steps, n_timesteps, seed, v_fn, gen_fn, fm_params, encode_fn, decode_fn):
    key = jax.random.PRNGKey(seed)
    runs, actions = [], []
    n_nodes = len(scenario.associations) + sum(len(stas) for stas in scenario.associations.values())

    fm_params, fm_params_old = deepcopy(fm_params), deepcopy(fm_params)
    optimizer = optax.adam(3e-5, b1=0.95, b2=0.95)
    opt_state = optimizer.init(fm_params)

    for _ in range(n_reps):
        scenario.reset()
        runs.append([])
        actions.append([])

        key, conf_key = jax.random.split(key)
        context, graph, ap_sta_idxs = get_initial_configurations(scenario, n_timesteps + 1, conf_key, encode_fn)

        for step in trange(n_steps):
            key, sel_key, fm_key, tx_key, scenario_key, train_key = jax.random.split(key, 6)

            selected_idx = jax.random.choice(sel_key, ap_sta_idxs).item()
            network_desc = get_network_desc(graph, selected_idx, encode_fn)

            all_gen = jax.tree.map(lambda *x: np.stack(x, axis=1), network_desc, *context[:-1])
            x_gen = network_desc._replace(
                edges=all_gen.edges,
                nodes=all_gen.nodes,
                globals=all_gen.globals
            )
            x_gen_train = jraph.batch([x_gen for _ in range(N_SAMPLES_TRAIN)])
            x_gen_eval = jraph.batch([x_gen for _ in range(N_SAMPLES_EVAL)])

            all_train = jax.tree.map(lambda *x: np.stack(x, axis=1), network_desc, *context)
            x_train = network_desc._replace(
                edges=all_train.edges,
                nodes=all_train.nodes,
                globals=all_train.globals
            )
            x_train = jraph.batch([x_train for _ in range(BS_TRAIN)])

            dec = decode_fn(gen_fn(fm_params_old, x_gen_train, rngs=fm_key))
            rewards = np.zeros(N_SAMPLES_TRAIN)

            for i, zt in enumerate(jraph.unbatch(dec)):
                tx_key, subkey = jax.random.split(tx_key)
                tx = connections_to_tx(scenario.associations, n_nodes, subkey, zt)

                scenario_key, subkey = jax.random.split(scenario_key)
                data_rate, _ = scenario(subkey, *tx, return_internals=False)

                rewards[i] = data_rate

            # DiffusionNFT paper:
            # rewards = rewards - np.mean(rewards)
            # rewards = 0.5 + 0.5 * np.clip(rewards / Z_c, -1.0, 1.0)

            # Our scaling (for max 3 concurrent tx):
            rewards /= 3 * 143.0

            @jax.jit
            def loss_fn(params, key, batch, rewards, params_old):
                z_key, t_key, model_key = jax.random.split(key, 3)
                t = jax.random.uniform(t_key, minval=0.0, maxval=0.99)

                def noisy_example(batch, t, z_key):
                    z = jax.random.normal(z_key, (batch.shape[0], batch.shape[-1]))
                    x = batch[:, -1]
                    x_t = (1 - t) * z + t * x
                    v_t = x - z
                    return batch.at[:, -1].set(x_t), v_t

                edges_key, nodes_key, globals_key = jax.random.split(z_key, 3)
                noisy_edges, v_t = noisy_example(batch.edges, t, edges_key)
                cx = batch._replace(
                    edges=noisy_edges,
                    nodes=noisy_example(batch.nodes, 0, nodes_key)[0],
                    globals=noisy_example(batch.globals, 0, globals_key)[0]
                )

                v_pred = v_fn(params, cx, t, rngs=model_key)
                v_pred_old = v_fn(params_old, cx, t, rngs=model_key)

                v_plus = (1 - BETA) * v_pred_old.edges + BETA * v_pred.edges
                v_minus = (1 + BETA) * v_pred_old.edges - BETA * v_pred.edges

                mask = jraph.get_edge_padding_mask(v_pred)
                edges_per_graph = mask.sum() // rewards.shape[0]
                rewards_edges = jnp.arange(mask.shape[0]) // edges_per_graph
                rewards_edges = rewards_edges * mask
                rewards = rewards[rewards_edges] * mask

                loss_1 = rewards * optax.squared_error(v_plus, v_t).mean(axis=-1)
                loss_2 = (1 - rewards) * optax.squared_error(v_minus, v_t).mean(axis=-1)
                loss = loss_1 + loss_2
                loss = (mask * loss).sum() / mask.sum()
                return loss

            @jax.jit
            def step_fn(params, opt_state, key, *batch):
                loss, grads = jax.value_and_grad(loss_fn)(params, key, *batch)
                updates, opt_state = optimizer.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                return params, opt_state, loss

            for i in range(N_SAMPLES_TRAIN // BS_TRAIN):
                start, end = i * BS_TRAIN, (i + 1) * BS_TRAIN
                batch_rewards = rewards[start:end]

                train_key, subkey = jax.random.split(train_key)
                fm_params, opt_state, loss = step_fn(fm_params, opt_state, subkey, x_train, batch_rewards, fm_params_old)

            fm_params_old = optax.incremental_update(fm_params, fm_params_old, step_size=max(1 - step * 0.001, 0.5))

            # https://arxiv.org/abs/2412.08296
            dec = decode_fn(gen_fn(fm_params, x_gen_eval, rngs=fm_key))
            best, best_val = dec[0], float('-inf')

            for zt in jraph.unbatch(dec):
                tx_key, subkey = jax.random.split(tx_key)
                tx = connections_to_tx(scenario.associations, n_nodes, subkey, zt)

                scenario_key, subkey = jax.random.split(scenario_key)
                data_rate, _ = scenario(subkey, *tx, return_internals=False)

                if data_rate > best_val:
                    best = tx
                    best_val = data_rate
                    graph = graph._replace(edges=zt.edges)

            data_rate, _, internals = scenario(scenario_key, *best, return_internals=True)
            runs[-1].append(data_rate.item())
            actions[-1].append(scenario.tx_to_action(*best))
            context = get_next_context(graph, context, internals, encode_fn)

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
    gen_fn = jax.jit(partial(fm.apply, method='gen'))
    v_fn = jax.jit(partial(fm.apply))

    gnn_cfg = OmegaConf.load(os.path.join(fm_cfg.gnn_path, 'config.yaml'))
    with ocp.CheckpointManager(os.path.abspath(fm_cfg.gnn_path), item_names=['params']) as ckpt_manager:
        gnn_params = ckpt_manager.restore(None, args=ocp.args.Composite(params=ocp.args.PyTreeRestore())).params

    gnn = GnnAutoencoder(**gnn_cfg.model)
    encode_fn = jax.jit(partial(gnn.apply, gnn_params, method='encode'))
    decode_fn = jax.jit(partial(gnn.apply, gnn_params, method='decode'))

    all_results = []

    for scenario in tqdm([EXEMPLARY_SCENARIOS[-1]], desc='Scenarios'):
        runs, actions = run_scenario(
            scenario, args.n_reps, scenario.n_steps, fm_cfg.train.n_timesteps, args.seed, v_fn, gen_fn, fm_params, encode_fn, decode_fn
        )
        all_results.append([{
            'runs': runs,
            'actions': actions
        }])

    with open(args.output, 'w') as file:
        json.dump(all_results, file)
