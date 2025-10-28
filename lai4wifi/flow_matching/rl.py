import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'

import json
from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy
from functools import partial

import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
import orbax.checkpoint as ocp
from mapc_research.envs.scenario_impl import residential_scenario
from omegaconf import OmegaConf
from tqdm import tqdm, trange

from lai4wifi.autoencoder.model import GnnAutoencoder
from lai4wifi.flow_matching.model import FlowMatching
from lai4wifi.flow_matching.sim import connections_to_tx, get_initial_configurations, get_next_context, tx_to_action
from lai4wifi.flow_matching.training import get_network_desc


# https://www.arxiv.org/abs/2509.16117
N_SAMPLES_TRAIN = 128
BS_TRAIN = 32
N_SAMPLES_EVAL = 32
BETA = 1.0


def gen_sample(graph):
    edges, nodes, globals = jax.tree.map(lambda x: x[:, :-1], (graph.edges, graph.nodes, graph.globals))
    return graph._replace(edges=edges, nodes=nodes, globals=globals)


def run_scenario(scenario, n_steps, seed, context_len, model, fm_params, encode_fn, decode_fn):
    key = jax.random.PRNGKey(seed)
    n_nodes = len(scenario.associations) + sum(len(stas) for stas in scenario.associations.values())
    logs = defaultdict(list)

    fm_params, fm_params_old = deepcopy(fm_params), deepcopy(fm_params)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(3e-5, b1=0.95, b2=0.95)
    )
    opt_state = optimizer.init(fm_params)

    key, conf_key = jax.random.split(key)
    context, graph, ap_sta_idxs = get_initial_configurations(scenario, context_len + 1, conf_key, encode_fn)

    for step in trange(n_steps, desc='Training steps'):
        key, sample_key, fm_gen_key, fm_eval_key, scenario_key, train_key, eval_key = jax.random.split(key, 7)
        xs = []

        for _ in range(N_SAMPLES_TRAIN):
            sample_key, subkey = jax.random.split(sample_key)
            selected_idx = jax.random.choice(subkey, ap_sta_idxs).item()
            network_desc = get_network_desc(graph, selected_idx, encode_fn)
            all = jax.tree.map(lambda *x: np.stack(x, axis=1), network_desc, *context)
            xs.append(network_desc._replace(edges=all.edges, nodes=all.nodes, globals=all.globals))

        x_gen = jraph.batch([gen_sample(x) for x in xs])
        z = jax.jit(model.apply, static_argnames='method')(fm_params_old, x_gen, rngs=fm_gen_key, method='gen')
        rewards = np.zeros(N_SAMPLES_TRAIN)

        for i, zt in enumerate(jraph.unbatch(decode_fn(z))):
            scenario_key, tx_key, sim_key = jax.random.split(scenario_key, 3)
            tx = connections_to_tx(scenario.associations, n_nodes, tx_key, zt)
            data_rate, _ = scenario(sim_key, *tx, return_internals=False)
            rewards[i] = data_rate

        logs['mean_reward'].append(rewards.mean().item())
        logs['std_reward'].append(rewards.std().item())
        logs['max_reward'].append(rewards.max().item())
        logs['min_reward'].append(rewards.min().item())

        # DiffusionNFT paper:
        # Z_c = global reward stddev
        # rewards = rewards - np.mean(rewards)
        # rewards = 0.5 + 0.5 * np.clip(rewards / Z_c, -1.0, 1.0)

        # Our scaling (assume max 3 concurrent tx):
        Z_c = 4 * 143.0
        rewards = np.clip(rewards / Z_c, 0.0, 1.0)

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

            v_pred = model.apply(params, cx, t, rngs=model_key)
            v_pred_old = model.apply(params_old, cx, t, rngs=model_key)

            v_plus = (1 - BETA) * v_pred_old.edges + BETA * v_pred.edges
            v_minus = (1 + BETA) * v_pred_old.edges - BETA * v_pred.edges

            # Mask out padding edges and match rewards to edges
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
            train_key, subkey = jax.random.split(train_key)
            start, end = i * BS_TRAIN, (i + 1) * BS_TRAIN
            batch_rewards = rewards[start:end]
            batch_train = jraph.batch(xs[start:end])
            fm_params, opt_state, loss = step_fn(fm_params, opt_state, subkey, batch_train, batch_rewards, fm_params_old)
            logs['loss'].append(loss.item())

        fm_params_old = optax.incremental_update(fm_params, fm_params_old, step_size=max(1 - step / n_steps, 0.5))

        # https://arxiv.org/abs/2412.08296
        x_eval = gen_sample(xs[0])
        x_eval = jraph.batch([x_eval for _ in range(N_SAMPLES_EVAL)])
        z = jax.jit(model.apply, static_argnames='method')(fm_params, x_eval, rngs=fm_eval_key, method='gen')
        best, best_val = None, float('-inf')

        for zt in jraph.unbatch(decode_fn(z)):
            eval_key, tx_key, sim_key = jax.random.split(eval_key, 3)
            tx = connections_to_tx(scenario.associations, n_nodes, tx_key, zt)
            data_rate, _ = scenario(sim_key, *tx, return_internals=False)

            if data_rate > best_val:
                best = tx
                best_val = data_rate
                graph = graph._replace(edges=zt.edges)

        data_rate, _, internals = scenario(scenario_key, *best, return_internals=True)
        context = get_next_context(graph, context, internals, encode_fn)

        logs['sim_data_rate'].append(data_rate.item())
        logs['actions'].append(tx_to_action(scenario.associations, *best))

        print({k: v[-1] for k, v in logs.items()})

    return logs


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('--checkpoint', type=str, default='runs/flow_matching_base')
    args.add_argument('--n_steps', type=int, default=200)
    args.add_argument('--output', type=str, default='rl_results.json')
    args.add_argument('--seed', type=int, default=42)
    args = args.parse_args()

    fm_cfg = OmegaConf.load(os.path.join(args.checkpoint, 'config.yaml'))
    with ocp.CheckpointManager(os.path.abspath(args.checkpoint), item_names=['params']) as ckpt_manager:
        fm_params = ckpt_manager.restore(None, args=ocp.args.Composite(params=ocp.args.PyTreeRestore())).params

    fm = FlowMatching(**fm_cfg.model)

    gnn_cfg = OmegaConf.load(os.path.join(fm_cfg.gnn_path, 'config.yaml'))
    with ocp.CheckpointManager(os.path.abspath(fm_cfg.gnn_path), item_names=['params']) as ckpt_manager:
        gnn_params = ckpt_manager.restore(None, args=ocp.args.Composite(params=ocp.args.PyTreeRestore())).params

    gnn = GnnAutoencoder(**gnn_cfg.model)
    encode_fn = jax.jit(partial(gnn.apply, gnn_params, method='encode'))
    decode_fn = jax.jit(partial(gnn.apply, gnn_params, method='decode'))

    scenarios = [
        residential_scenario(seed=101, x_apartments=2, y_apartments=2, n_sta_per_ap=4, size=30.0)
    ]
    all_results = []

    for scenario in tqdm(scenarios, desc='Scenarios'):
        all_results.append(run_scenario(scenario, args.n_steps, args.seed, fm_cfg.train.n_timesteps, fm, fm_params, encode_fn, decode_fn))

    with open(args.output, 'w') as file:
        json.dump(all_results, file)
