import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'

import logging
from copy import deepcopy
from functools import partial

import hydra
import jax
import jax.numpy as jnp
import jraph
import numpy as np
import optax
import orbax.checkpoint as ocp
import wandb
from mapc_research.envs.scenario_impl import residential_scenario
from omegaconf import OmegaConf
from tqdm import tqdm, trange

from lai4wifi.autoencoder.model import GnnAutoencoder
from lai4wifi.flow_matching.model import FlowMatching
from lai4wifi.flow_matching.sim import connections_to_tx, get_initial_configurations, get_next_context, tx_to_action
from lai4wifi.flow_matching.training import get_network_desc


def gen_sample(graph):
    edges, nodes, globals = jax.tree.map(lambda x: x[:, :-1], (graph.edges, graph.nodes, graph.globals))
    return graph._replace(edges=edges, nodes=nodes, globals=globals)


def run_scenario(scenario, cfg, fm_cfg, fm, fm_params, encode_fn, decode_fn):
    n_nodes = len(scenario.associations) + sum(len(stas) for stas in scenario.associations.values())

    fm_params, fm_params_old = deepcopy(fm_params), deepcopy(fm_params)
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.optimizer.grad_clip),
        optax.adamw(cfg.optimizer.learning_rate, weight_decay=cfg.optimizer.weight_decay, b1=cfg.optimizer.b1, b2=cfg.optimizer.b2)
    )
    opt_state = optimizer.init(fm_params)

    ckpt_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    ckpt_manager = ocp.CheckpointManager(
        directory=ckpt_path,
        options=ocp.CheckpointManagerOptions(max_to_keep=None if cfg.train.ckpt.intermediate else 1, create=True),
        item_names=['params', 'params_old', 'opt_state']
    )
    OmegaConf.save(config=cfg, f=os.path.join(ckpt_path, 'config.yaml'))

    if cfg.train.log.type == 'wandb':
        wandb.init(project='lai4wifi', name=cfg.name)

    key = jax.random.PRNGKey(cfg.train.seed)
    key, conf_key = jax.random.split(key)
    context, graph, ap_sta_idxs = get_initial_configurations(scenario, fm_cfg.train.n_timesteps + 1, conf_key, encode_fn)

    for step in trange(cfg.train.n_steps, desc='Training steps'):
        key, sample_key, fm_gen_key, fm_eval_key, scenario_key, train_key, eval_key = jax.random.split(key, 7)
        xs = []

        for _ in range(cfg.train.nft.n_samples):
            sample_key, subkey = jax.random.split(sample_key)
            selected_idx = jax.random.choice(subkey, ap_sta_idxs).item()
            network_desc = get_network_desc(graph, selected_idx, encode_fn)
            all = jax.tree.map(lambda *x: jnp.stack(x, axis=1), network_desc, *context)
            xs.append(network_desc._replace(edges=all.edges, nodes=all.nodes, globals=all.globals))

        x_gen = jraph.batch([gen_sample(x) for x in xs])
        z = jax.jit(fm.apply, static_argnames='method')(fm_params_old, x_gen, rngs=fm_gen_key, method='gen')
        rewards = np.zeros(len(xs))

        for i, zt in enumerate(jraph.unbatch(decode_fn(z))):
            scenario_key, tx_key, sim_key = jax.random.split(scenario_key, 3)
            tx = connections_to_tx(scenario.associations, n_nodes, tx_key, zt)
            data_rate, _ = scenario(sim_key, *tx, return_internals=False)
            rewards[i] = data_rate

        log_dict = {}
        log_dict['step'] = step
        log_dict['reward/mean'] = rewards.mean().item()
        log_dict['reward/std'] = rewards.std().item()
        log_dict['reward/max'] = rewards.max().item()
        log_dict['reward/min'] = rewards.min().item()

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

            # https://www.arxiv.org/abs/2509.16117
            v_pred = fm.apply(params, cx, t, rngs=model_key)
            v_pred_old = fm.apply(params_old, cx, t, rngs=model_key)

            beta = cfg.train.nft.beta
            v_plus = (1 - beta) * v_pred_old.edges + beta * v_pred.edges
            v_minus = (1 + beta) * v_pred_old.edges - beta * v_pred.edges

            # Assume all graphs in the batch are the same size
            rewards = jax.jit(jnp.repeat, static_argnames=['axis', 'total_repeat_length'])(rewards, v_pred.n_edge, axis=0, total_repeat_length=v_pred.edges.shape[0])

            loss_1 = rewards * optax.squared_error(v_plus, v_t).mean(axis=-1)
            loss_2 = (1 - rewards) * optax.squared_error(v_minus, v_t).mean(axis=-1)
            loss = loss_1 + loss_2
            return loss.mean()

        @jax.jit
        def step_fn(params, opt_state, key, *batch):
            loss, grads = jax.value_and_grad(loss_fn)(params, key, *batch)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            grad_norm = optax.global_norm(grads)
            return params, opt_state, loss, grad_norm

        batch_size = cfg.train.nft.batch_size
        loss_val, gn_val, grad_steps = 0.0, 0.0, cfg.train.nft.n_samples // batch_size

        for i in range(grad_steps):
            train_key, subkey = jax.random.split(train_key)
            start, end = i * batch_size, (i + 1) * batch_size
            batch_rewards = rewards[start:end]
            batch_train = jraph.batch(xs[start:end])
            fm_params, opt_state, loss, grad_norm = step_fn(fm_params, opt_state, subkey, batch_train, batch_rewards, fm_params_old)
            loss_val += loss.item()
            gn_val += grad_norm.item()

        log_dict['rl/loss'] = loss_val / grad_steps
        log_dict['rl/grad_norm'] = gn_val / grad_steps

        fm_params_old = optax.incremental_update(fm_params, fm_params_old, step_size=max(1 - step / cfg.train.n_steps, 0.5))

        # https://arxiv.org/abs/2412.08296
        x_eval = gen_sample(xs[0])
        x_eval = jraph.batch([x_eval for _ in range(fm_cfg.train.n_eval_samples)])
        z = jax.jit(fm.apply, static_argnames='method')(fm_params, x_eval, rngs=fm_eval_key, method='gen')
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

        log_dict['sim/data_rate'] = data_rate.item()
        log_dict['sim/actions'] = tx_to_action(scenario.associations, *best)

        if step % cfg.train.log.freq == 0 or step == cfg.train.n_steps - 1:
            logging.info(log_dict)
            if cfg.train.log.type == 'wandb':
                wandb.log(log_dict)

        if step % cfg.train.ckpt.freq == 0 or step == cfg.train.n_steps - 1:
            ckpt_manager.save(step, args=ocp.args.Composite(
                params=ocp.args.PyTreeSave(fm_params),
                params_old=ocp.args.PyTreeSave(fm_params_old),
                opt_state=ocp.args.PyTreeSave(opt_state)
            ))

    ckpt_manager.wait_until_finished()
    ckpt_manager.close()


@hydra.main(config_path='configs', config_name='rl', version_base=None)
def main(cfg):
    fm_cfg = OmegaConf.load(os.path.join(cfg.fm_path, 'config.yaml'))
    with ocp.CheckpointManager(os.path.abspath(cfg.fm_path), item_names=['params']) as ckpt_manager:
        fm_params = ckpt_manager.restore(None, args=ocp.args.Composite(params=ocp.args.PyTreeRestore())).params

    fm = FlowMatching(**fm_cfg.model)

    gnn_cfg = OmegaConf.load(os.path.join(fm_cfg.gnn_path, 'config.yaml'))
    with ocp.CheckpointManager(os.path.abspath(fm_cfg.gnn_path), item_names=['params']) as ckpt_manager:
        gnn_params = ckpt_manager.restore(None, args=ocp.args.Composite(params=ocp.args.PyTreeRestore())).params

    gnn = GnnAutoencoder(**gnn_cfg.model)
    encode_fn = jax.jit(partial(gnn.apply, gnn_params, method='encode'))
    decode_fn = jax.jit(partial(gnn.apply, gnn_params, method='decode'))

    for scenario in tqdm([
        residential_scenario(seed=101, x_apartments=2, y_apartments=2, n_sta_per_ap=4, size=30.0)
    ], desc='Scenarios'):
        run_scenario(scenario, cfg, fm_cfg, fm, fm_params, encode_fn, decode_fn)


if __name__ == '__main__':
    main()
