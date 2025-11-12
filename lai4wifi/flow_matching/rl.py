import os
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_ENABLE_X64'] = 'True'

import logging
from copy import deepcopy
from functools import partial
from itertools import combinations

import hydra
import jax
import jax.numpy as jnp
import jraph
import networkx as nx
import numpy as np
import optax
import orbax.checkpoint as ocp
import wandb
from mapc_research.envs.scenario_impl import residential_scenario
from omegaconf import OmegaConf
from tqdm import tqdm, trange

from lai4wifi.graphs.attributes import *
from lai4wifi.graphs.dataset import RSSI_MEAN, RSSI_STD
from lai4wifi.graphs.utils import nx_to_jraph
from lai4wifi.flow_matching.model import FlowMatching
from lai4wifi.flow_matching.sim import associations_to_nx, connections_to_tx, get_initial_configurations, tx_to_action
from lai4wifi.flow_matching.training import draw_t, get_network_desc, jax_repeat, noisy_example
from lai4wifi.graphs.model import GnnAutoencoder


def random_tx(key, scenario):
    n_aps = len(scenario.associations)
    n_nodes = n_aps + sum(len(stas) for stas in scenario.associations.values())
    
    tx = np.zeros((n_nodes, n_nodes), dtype=int)
    tx_power = np.zeros(n_nodes, dtype=int)
    mcs = np.zeros(n_nodes, dtype=int)

    for ap, stas in scenario.associations.items():
        key, ap_key, sta_key, power_key, mcs_key = jax.random.split(key, 5)

        if jax.random.uniform(ap_key) < 1 / n_aps:
            sta_idx = jax.random.choice(sta_key, jnp.asarray(stas)).item()
            tx[ap, sta_idx] = 1
            tx_power[ap] = jax.random.choice(power_key, len(TxPowerValue) - 1).item()
            mcs[ap] = jax.random.choice(mcs_key, len(McsValue) - 1).item()

    return tx, tx_power, mcs


def tx_to_conf(scenario, tx, tx_power, mcs, internals, encode_fn):
    G = nx.Graph()
    G_base = associations_to_nx(scenario)
    succ_prob = internals.frames_transmitted / jnp.maximum(internals.ampdu_size, 1)

    for ap, stas in scenario.associations.items():
        for sta in stas:
            if not isinstance(sta, int):
                sta = sta.item()

            rssi = (internals.signal_power[ap, sta] - RSSI_MEAN) / RSSI_STD
            G.add_edge(ap, sta, active=IsActive.TRUE, mcs=McsValue.NA, rssi=rssi, selected=IsSelected.NA, success=0.0, tx_power=TxPowerValue.NA, type=EdgeType.STA)
    
    for ap, sta, power, rate, success in zip(*np.where(tx), tx_power, mcs, succ_prob):
        ap, sta = ap.item(), sta.item()
        G.edges[(ap, sta)]['mcs'] = McsValue(rate.item())
        G.edges[(ap, sta)]['selected'] = IsSelected.TRUE
        G.edges[(ap, sta)]['success'] = success.item()
        G.edges[(ap, sta)]['tx_power'] = TxPowerValue(power.item())

    for ap_1, ap_2 in combinations(scenario.associations.keys(), r=2):
        rssi = (internals.signal_power[ap_1, ap_2] - RSSI_MEAN) / RSSI_STD
        if (ap_1, ap_2) in G_base.edges:
            G.add_edge(ap_1, ap_2, active=IsActive.NA, mcs=McsValue.NA, rssi=rssi, selected=IsSelected.NA, success=0.0, tx_power=TxPowerValue.NA, type=EdgeType.AP)

    graph = nx_to_jraph(G)
    return encode_fn(graph)


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
    key, conf_key, xs_key = jax.random.split(key, 3)
    _, graph, ap_sta_idxs = get_initial_configurations(scenario, 0, conf_key, encode_fn)
    all_xs, all_rewards = [], np.zeros(cfg.train.nft.n_examples)

    for i in trange(cfg.train.nft.n_examples, desc='Generating dataset'):
        xs_key, tx_key, sim_key = jax.random.split(xs_key, 3)
        tx = random_tx(tx_key, scenario)
        data_rate, _, internals = scenario(sim_key, *tx, return_internals=True)
        all_xs.append(tx_to_conf(scenario, *tx, internals, encode_fn))
        all_rewards[i] = data_rate

    for step in trange(cfg.train.n_steps, desc='Training steps'):
        key, sample_key, train_key, eval_key = jax.random.split(key, 4)
        xs, rewards = [], []

        for i in range(cfg.train.nft.n_samples):
            selected_key, context_key, sample_key = jax.random.split(sample_key, 3)
            selected_idx = jax.random.choice(selected_key, ap_sta_idxs).item()
            graphs_idxs = jax.random.choice(context_key, len(all_xs), (fm_cfg.train.n_timesteps + 1,)).tolist()
            
            graphs_ctx = [all_xs[i] for i in graphs_idxs]
            network_desc = get_network_desc(graph, selected_idx, encode_fn)
            all = jax.tree.map(lambda *x: jnp.stack(x, axis=1), network_desc, *graphs_ctx)
            
            xs.append(network_desc._replace(edges=all.edges, nodes=all.nodes, globals=all.globals))
            rewards.append(all_rewards[graphs_idxs[-1]])

        # DiffusionNFT paper:
        # Z_c = global reward stddev
        # rewards = rewards - np.mean(rewards)
        # rewards = 0.5 + 0.5 * np.clip(rewards / Z_c, -1.0, 1.0)

        # Our scaling (T-Optimal as an upper bound):
        Z_c = 315.4
        rewards = np.array(rewards)
        rewards = np.clip(rewards / Z_c, 0.0, 1.0)

        log_dict = {}
        log_dict['step'] = step
        log_dict['reward/mean'] = rewards.mean().item()
        log_dict['reward/std'] = rewards.std().item()
        log_dict['reward/max'] = rewards.max().item()
        log_dict['reward/min'] = rewards.min().item()

        @jax.jit
        def loss_fn(params, key, batch, rewards, params_old):
            t_key, edges_key, nodes_key, globals_key, model_key = jax.random.split(key, 5)

            t = draw_t(t_key, batch)
            noisy_edges, v_t = noisy_example(edges_key, batch.edges, t)
            cx = batch._replace(
                edges=noisy_edges,
                nodes=noisy_example(nodes_key, batch.nodes)[0],
                globals=noisy_example(globals_key, batch.globals)[0]
            )

            # https://www.arxiv.org/abs/2509.16117
            fm_key, fm_key_old = jax.random.split(model_key)
            v_pred = fm.apply(params, cx, t, rngs=fm_key)
            v_pred_old = fm.apply(params_old, cx, t, rngs=fm_key_old)

            beta = cfg.train.nft.beta
            v_plus = (1 - beta) * v_pred_old.edges + beta * v_pred.edges
            v_minus = (1 + beta) * v_pred_old.edges - beta * v_pred.edges

            # Assume all graphs in the batch are the same size
            rewards = jax_repeat(rewards, v_pred.n_edge, axis=0, total_repeat_length=v_pred.edges.shape[0])

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

        x_eval = gen_sample(xs[0])
        fm_key, tx_key, sim_key = jax.random.split(eval_key, 3)

        z = jax.jit(fm.apply, static_argnames='method')(fm_params, x_eval, rngs=fm_key, method='gen')
        tx = connections_to_tx(scenario.associations, n_nodes, tx_key, decode_fn(z))
        data_rate, _, internals = scenario(sim_key, *tx, return_internals=True)

        log_dict['sim/data_rate'] = data_rate.item()
        log_dict['sim/actions'] = tx_to_action(scenario.associations, internals, *tx)

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
        residential_scenario(seed=12, x_apartments=2, y_apartments=2, n_sta_per_ap=4, size=30.0)
    ], desc='Scenarios'):
        run_scenario(scenario, cfg, fm_cfg, fm, fm_params, encode_fn, decode_fn)


if __name__ == '__main__':
    main()
