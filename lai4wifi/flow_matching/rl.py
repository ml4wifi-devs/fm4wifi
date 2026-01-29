import os
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_ENABLE_X64'] = 'True'

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
from mapc_research.envs.scenario_impl import *
from omegaconf import OmegaConf
from tqdm import trange

from lai4wifi.flow_matching.model import FlowMatching
from lai4wifi.flow_matching.training import draw_t, jax_repeat, noisy_example
from lai4wifi.flow_matching.utils import connections_to_tx, random_tx, tx_to_action, tx_to_jraph
from lai4wifi.graphs.model import GnnAutoencoder
from lai4wifi.graphs.utils import make_batch, unbatch


TRAIN_SCENARIOS = [
    toy_scenario_1(20.0),
    toy_scenario_2(10.0, 1.0),
    distance_scenario(5.0),
    distance_scenario(20.0),
    distance_scenario(50.0),
    flow_in_the_middle_scenario(5.0),
    flow_in_the_middle_scenario(15.0),
    flow_in_the_middle_scenario(30.0),
    residential_scenario(seed=100, x_apartments=2, y_apartments=2, n_sta_per_ap=4, size=20.0),
    residential_scenario(seed=101, x_apartments=3, y_apartments=1, n_sta_per_ap=3, size=20.0),
    residential_scenario(seed=102, x_apartments=3, y_apartments=2, n_sta_per_ap=2, size=30.0),
    residential_scenario(seed=103, x_apartments=2, y_apartments=1, n_sta_per_ap=5, size=10.0),
    small_office_scenario(10.0, 3),
    small_office_scenario(20.0, 3),
    small_office_scenario(40.0, 2),
    random_scenario(seed=100, d_ap=75., d_sta=8., n_ap=2, n_sta_per_ap=5, n_steps=1000),
    random_scenario(seed=101, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=3, n_steps=1000),
    random_scenario(seed=102, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=4, n_steps=1000),
    random_scenario(seed=103, d_ap=75., d_sta=5., n_ap=4, n_sta_per_ap=3, n_steps=2000),
    random_scenario(seed=104, d_ap=75., d_sta=4., n_ap=4, n_sta_per_ap=4, n_steps=1000),
    random_scenario(seed=105, d_ap=75., d_sta=4., n_ap=5, n_sta_per_ap=3, n_steps=3000),
    random_scenario(seed=106, d_ap=75., d_sta=8., n_ap=2, n_sta_per_ap=5, n_steps=1000),
    random_scenario(seed=107, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=3, n_steps=1000),
    random_scenario(seed=108, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=4, n_steps=1000),
    random_scenario(seed=109, d_ap=75., d_sta=5., n_ap=4, n_sta_per_ap=3, n_steps=2000),
    random_scenario(seed=110, d_ap=75., d_sta=4., n_ap=4, n_sta_per_ap=4, n_steps=1000),
    random_scenario(seed=111, d_ap=75., d_sta=4., n_ap=5, n_sta_per_ap=3, n_steps=3000),
    random_scenario(seed=112, d_ap=75., d_sta=8., n_ap=2, n_sta_per_ap=5, n_steps=1000),
    random_scenario(seed=113, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=3, n_steps=1000),
    random_scenario(seed=114, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=4, n_steps=1000),
    random_scenario(seed=115, d_ap=75., d_sta=5., n_ap=4, n_sta_per_ap=3, n_steps=2000),
    random_scenario(seed=116, d_ap=75., d_sta=4., n_ap=4, n_sta_per_ap=4, n_steps=1000),
    random_scenario(seed=117, d_ap=75., d_sta=4., n_ap=5, n_sta_per_ap=3, n_steps=3000),
    random_scenario(seed=118, d_ap=75., d_sta=8., n_ap=2, n_sta_per_ap=5, n_steps=1000),
    random_scenario(seed=119, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=3, n_steps=1000),
    random_scenario(seed=120, d_ap=75., d_sta=5., n_ap=3, n_sta_per_ap=4, n_steps=1000),
    random_scenario(seed=121, d_ap=75., d_sta=5., n_ap=4, n_sta_per_ap=3, n_steps=2000),
    random_scenario(seed=122, d_ap=75., d_sta=4., n_ap=4, n_sta_per_ap=4, n_steps=2000),
    random_scenario(seed=123, d_ap=75., d_sta=4., n_ap=5, n_sta_per_ap=3, n_steps=3000),
    symm_residential_scenario(seed=42, n_steps=2000, x_apartments=1, y_apartments=4, n_sta_per_ap=4, size=30, d_sta=2),
    symm_residential_scenario(seed=42, n_steps=2000, x_apartments=2, y_apartments=4, n_sta_per_ap=4, size=30, d_sta=2),
    symm_residential_scenario(seed=42, n_steps=2000, x_apartments=3, y_apartments=4, n_sta_per_ap=4, size=30, d_sta=2),
    symm_residential_scenario(seed=42, n_steps=2000, x_apartments=4, y_apartments=4, n_sta_per_ap=4, size=30, d_sta=2)
]
EVAL_SCENARIO = residential_scenario(seed=12, x_apartments=2, y_apartments=2, n_sta_per_ap=4, size=30.0)


@hydra.main(config_path='configs', config_name='rl', version_base=None)
def main(cfg):
    beta = cfg.train.nft.beta
    batch_size = cfg.train.nft.batch_size
    grad_steps = cfg.train.nft.n_samples // batch_size

    @jax.jit
    def loss_fn(params, key, batch, rewards, params_old):
        # DiffusionNFT - https://www.arxiv.org/abs/2509.16117
        t_key, edges_key, nodes_key, globals_key, fm_key, fm_key_old = jax.random.split(key, 6)

        t = draw_t(t_key, batch)
        noisy_edges, v_t = noisy_example(edges_key, batch.edges, t)
        cx = batch._replace(
            edges=noisy_edges,
            nodes=noisy_example(nodes_key, batch.nodes)[0],
            globals=noisy_example(globals_key, batch.globals)[0]
        )

        v_pred = fm.apply(params, cx, t, rngs=fm_key)
        v_pred_old = fm.apply(params_old, cx, t, rngs=fm_key_old)

        v_plus = (1 - beta) * v_pred_old.edges + beta * v_pred.edges
        v_minus = (1 + beta) * v_pred_old.edges - beta * v_pred.edges

        rewards = jnp.concatenate([rewards, jnp.array([0.])])
        rewards = jax_repeat(rewards, v_pred.n_edge, axis=0, total_repeat_length=v_pred.edges.shape[0])

        mask = jraph.get_edge_padding_mask(v_pred)
        loss_p = rewards * optax.squared_error(v_plus, v_t).mean(axis=-1)
        loss_m = (1 - rewards) * optax.squared_error(v_minus, v_t).mean(axis=-1)
        loss = loss_p + loss_m
        loss = (loss * mask).sum() / mask.sum()
        return loss

    @jax.jit
    def step_fn(params, opt_state, key, batch, rewards, params_old):
        loss, grads = jax.value_and_grad(loss_fn)(params, key, batch, rewards, params_old)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        grad_norm = optax.global_norm(grads)
        param_dist = optax.global_norm(jax.tree.map(lambda a, b: a - b, params, params_old))
        return params, opt_state, loss, grad_norm, param_dist
    
    def prepare_context(context):
        batched = make_batch(context)
        encoded = encode_fn(batched)
        unbatched = unbatch(encoded)
        ctx = jax.tree.map(lambda *x: jnp.stack(x, axis=1), *unbatched)
        return unbatched[0]._replace(edges=ctx.edges, nodes=ctx.nodes, globals=ctx.globals)

    key = jax.random.PRNGKey(cfg.train.seed)
    eval_key, dataset_key, train_key = jax.random.split(key, 3)

    fm_cfg = OmegaConf.load(os.path.join(cfg.fm_path, 'config.yaml'))
    with ocp.CheckpointManager(os.path.abspath(cfg.fm_path), item_names=['ema_params']) as ckpt_manager:
        fm_params = ckpt_manager.restore(None, args=ocp.args.Composite(ema_params=ocp.args.PyTreeRestore())).ema_params

    fm = FlowMatching(**fm_cfg.model)
    gen_fn = jax.jit(partial(fm.apply, n_steps=cfg.train.gen.n_steps, method='gen'))

    gnn_cfg = OmegaConf.load(os.path.join(fm_cfg.gnn_path, 'config.yaml'))
    with ocp.CheckpointManager(os.path.abspath(fm_cfg.gnn_path), item_names=['params']) as ckpt_manager:
        gnn_params = ckpt_manager.restore(None, args=ocp.args.Composite(params=ocp.args.PyTreeRestore())).params

    gnn = GnnAutoencoder(**gnn_cfg.model)
    encode_fn = jax.jit(partial(gnn.apply, gnn_params, method='encode'))
    decode_fn = jax.jit(partial(gnn.apply, gnn_params, method='decode'))

    fm_params, fm_params_old = deepcopy(fm_params), deepcopy(fm_params)
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.optimizer.grad_clip),
        optax.adamw(
            cfg.optimizer.learning_rate, weight_decay=cfg.optimizer.weight_decay, 
            b1=cfg.optimizer.b1, b2=cfg.optimizer.b2, eps=cfg.optimizer.eps
        )
    )
    opt_state = optimizer.init(fm_params)

    ckpt_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    ckpt_manager = ocp.CheckpointManager(
        directory=ckpt_path,
        options=ocp.CheckpointManagerOptions(max_to_keep=None if cfg.train.ckpt.intermediate else 1, create=True),
        item_names=['ema_params', 'params_old', 'opt_state']
    )
    OmegaConf.save(config=cfg, f=os.path.join(ckpt_path, 'rl_config.yaml'))
    OmegaConf.save(config=fm_cfg, f=os.path.join(ckpt_path, 'config.yaml'))

    if cfg.train.log.type == 'wandb':
        wandb.init(project='lai4wifi', name=cfg.name)
    
    eval_ctx = []

    for _ in range(fm_cfg.train.n_timesteps):
        eval_key, tx_key, sim_key = jax.random.split(eval_key, 3)
        tx = random_tx(tx_key, EVAL_SCENARIO)
        _, _, internals = EVAL_SCENARIO(sim_key, *tx, return_internals=True)
        eval_ctx.append(tx_to_jraph(EVAL_SCENARIO, *tx, internals))

    eval_ctx = prepare_context(eval_ctx)
    all_scenarios, all_ctx, all_rewards = [], [], []

    for i in trange(cfg.train.nft.n_examples, desc='Generating context dataset'):
        dataset_key, scenario_key, gen_key = jax.random.split(dataset_key, 3)
        scenario_idx = jax.random.choice(scenario_key, len(TRAIN_SCENARIOS))
        scenario = TRAIN_SCENARIOS[scenario_idx.item()]
        context = []

        for _ in range(fm_cfg.train.n_timesteps):
            gen_key, tx_key, sim_key = jax.random.split(gen_key, 3)
            tx = random_tx(tx_key, scenario)
            data_rate, _, internals = scenario(sim_key, *tx, return_internals=True)
            context.append(tx_to_jraph(scenario, *tx, internals))
            all_rewards.append(data_rate.item())

        context = prepare_context(context)
        all_scenarios.append(scenario)
        all_ctx.append(context)

    all_rewards = np.array(all_rewards)
    Z_c = all_rewards.std()

    for step in trange(cfg.train.n_steps, desc='Training steps'):
        train_key, sample_key, fm_key, target_key, step_key, dataset_key = jax.random.split(train_key, 6)
        
        idxs = jax.random.choice(sample_key, len(all_ctx), shape=(cfg.train.nft.n_samples,))
        contexts, scenarios = [all_ctx[i] for i in idxs.tolist()], [all_scenarios[i] for i in idxs.tolist()]
        decoded = []

        for i in range(0, cfg.train.nft.n_samples, batch_size):
            fm_key, subkey = jax.random.split(fm_key)
            batch = make_batch(contexts[i:i + batch_size])
            generated = gen_fn(fm_params_old, batch, rngs=subkey)
            decoded += unbatch(decode_fn(generated))

        targets, rewards = [], []

        for scenario, zt in zip(scenarios, decoded):
            target_key, tx_key, sim_key = jax.random.split(target_key, 3)
            tx = connections_to_tx(scenario.associations, tx_key, zt)
            data_rate, _, internals = scenario(sim_key, *tx, return_internals=True)
            targets.append(tx_to_jraph(scenario, *tx, internals))
            rewards.append(data_rate.item())

        targets = make_batch(targets)
        targets = encode_fn(targets)
        targets = unbatch(targets)
        xs = []
            
        for ctx, target in zip(contexts, targets):
            full = ctx._replace(
                edges=jnp.concatenate([ctx.edges, target.edges[:, None]], axis=1),
                nodes=jnp.concatenate([ctx.nodes, target.nodes[:, None]], axis=1),
                globals=jnp.concatenate([ctx.globals, target.globals[:, None]], axis=1)
            )
            xs.append(full)

        rewards_raw = np.array(rewards)
        rewards = rewards_raw - rewards_raw.mean()
        rewards = 0.5 + 0.5 * np.clip(rewards / Z_c, -1.0, 1.0)

        loss_val, gn_val, pd_val = 0.0, 0.0, 0.0

        for i in range(grad_steps):
            step_key, subkey = jax.random.split(step_key)

            start, end = i * batch_size, (i + 1) * batch_size
            batch = make_batch(xs[start:end])
            batch_rewards = rewards[start:end]

            fm_params, opt_state, loss, grad_norm, param_dist = step_fn(
                fm_params, opt_state, subkey, batch, batch_rewards, fm_params_old
            )

            loss_val += loss.item()
            gn_val += grad_norm.item()
            pd_val += param_dist.item()

        fm_params_old = optax.incremental_update(fm_params, fm_params_old, step_size=max(1 - step / cfg.train.n_steps, 0.5))

        fm_key, tx_key, sim_key = jax.random.split(dataset_key, 3)
        z = gen_fn(fm_params, eval_ctx, rngs=fm_key)
        tx = connections_to_tx(EVAL_SCENARIO.associations, tx_key, decode_fn(z))
        data_rate, _, internals = EVAL_SCENARIO(sim_key, *tx, return_internals=True)

        log_dict = {}
        log_dict['step'] = step
        log_dict['rl/loss'] = loss_val / grad_steps
        log_dict['rl/grad_norm'] = gn_val / grad_steps
        log_dict['rl/mean_reward'] = rewards_raw.mean()
        log_dict['rl/param_dist'] = pd_val / grad_steps
        log_dict['sim/data_rate'] = data_rate.item()
        log_dict['sim/actions'] = tx_to_action(EVAL_SCENARIO.associations, internals, *tx)

        if step % cfg.train.log.freq == 0 or step == cfg.train.n_steps - 1:
            logging.info(log_dict)
            if cfg.train.log.type == 'wandb':
                wandb.log(log_dict)

        if step % cfg.train.ckpt.freq == 0 or step == cfg.train.n_steps - 1:
            ckpt_manager.save(step, args=ocp.args.Composite(
                ema_params=ocp.args.PyTreeSave(fm_params),
                params_old=ocp.args.PyTreeSave(fm_params_old),
                opt_state=ocp.args.PyTreeSave(opt_state)
            ))

    ckpt_manager.wait_until_finished()
    ckpt_manager.close()


if __name__ == '__main__':
    main()
