import os
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_ENABLE_X64'] = 'True'

import json
import logging
import time

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import wandb
from omegaconf import OmegaConf
from tqdm import trange

from lai4wifi.envs.scenarios import TEST_SCENARIOS, create_scenario_batch_eval_fn, create_scenario_eval_fn
from lai4wifi.flow_matching.model import FlowMatching
from lai4wifi.graphs.utils import clear_graph
from lai4wifi.pipeline.checkpoints import load_fm_gnn, load_surrogate, restore_params
from lai4wifi.pipeline.conversion import (
    connections_to_tx,
    create_scenario_cache,
    designate_stations,
    designated_edges,
    random_tx_fast,
    tx_to_jraph_direct
)
from lai4wifi.surrogate.utils import RATE_MEAN, RATE_STD, mdn_loss


def per_scenario(value, i):
    return value[i] if OmegaConf.is_list(value) else value


def get_scenario(name):
    split, idx = name.split('/')
    return TEST_SCENARIOS[split][int(idx)]


def stack_graphs(graphs):
    return jax.tree.map(lambda *x: jnp.stack(x), *graphs)


def take_graphs(graphs, idxs):
    return jax.tree.map(lambda x: x[idxs], graphs)


def select_top(scores, served, top_k, coverage_ratio):
    return np.argsort(-scores)[:top_k]


def select_covering(scores, served, top_k, coverage_ratio):
    order = np.argsort(-scores)
    covered = np.zeros(served.shape[-1], dtype=bool)
    n_covering = int(np.ceil(top_k * coverage_ratio))
    chosen = []

    for i in order:
        if len(chosen) < n_covering and (served[i] & ~covered).any():
            chosen.append(i)
            covered |= served[i]

    chosen += [i for i in order if i not in chosen][:top_k - len(chosen)]
    return np.asarray(chosen)


def rank_correlation(a, b):
    ranks_a, ranks_b = a.argsort(axis=-1).argsort(axis=-1), b.argsort(axis=-1).argsort(axis=-1)
    return np.mean([np.corrcoef(x, y)[0, 1] for x, y in zip(ranks_a, ranks_b)]).item()


def latent_graph(context, z):
    return clear_graph(context._replace(edges=z))


def add_condition_input(params, edge_dim):
    params = jax.tree.map(lambda x: x, params)
    dense = params['params']['Transformer_0']['Dense_0']
    dense['kernel'] = jnp.insert(dense['kernel'], edge_dim, 0.0, axis=0)
    return params


def interleaved_stations(key, associations):
    ap_key, sta_key = jax.random.split(key)
    aps = np.asarray(jax.random.permutation(ap_key, np.asarray(list(associations))))
    shifts = np.asarray(jax.random.randint(sta_key, (len(aps),), 0, 1000))
    n_rounds = max(len(stas) for stas in associations.values())
    return np.asarray([
        [ap, associations[ap][(r + shift) % len(associations[ap])]]
        for r in range(n_rounds) for ap, shift in zip(aps, shifts) if r < len(associations[ap])
    ])


def sample_designated(key, associations, n_designated, n):
    return jnp.stack([interleaved_stations(k, associations)[:n_designated] for k in jax.random.split(key, n)])


def round_robin_designated(key, associations, n_designated, n):
    stations = interleaved_stations(key, associations)
    idxs = (np.arange(n)[:, None] * n_designated + np.arange(n_designated)) % len(stations)
    return jnp.asarray(stations[idxs])


def append_noise(key, x):
    return jnp.concatenate([x, jax.random.normal(key, x[:, -1:].shape, dtype=x.dtype)], axis=1)


def create_contexts(key, cache, scenario_eval_fn, encode_fn, n_contexts):
    graphs = []

    with jax.default_device(jax.devices('cpu')[0]):
        for _ in range(n_contexts):
            key, tx_key, sim_key = jax.random.split(key, 3)
            tx = random_tx_fast(tx_key, cache)
            _, _, internals = scenario_eval_fn(sim_key, *tx, return_internals=True)
            graphs.append(tx_to_jraph_direct(cache, *tx, internals))

    contexts = jax.vmap(encode_fn)(stack_graphs(graphs))
    return contexts._replace(
        edges=contexts.edges[:, :, None],
        nodes=contexts.nodes[:, :, None],
        globals=contexts.globals[:, :, None]
    )


@hydra.main(config_path='configs', config_name='rl', version_base=None)
def main(cfg):
    beta = cfg.train.nft.beta
    adv_clip = cfg.train.nft.adv_clip
    kl_coef = cfg.train.nft.kl_coef
    chunk_size = cfg.train.rollout.chunk_size
    sim_chunk_size = cfg.train.rollout.sim_chunk_size
    deterministic = cfg.train.rollout.deterministic

    def condition(context, designated):
        if designated is None:
            return None
        return designated_edges(context, designated[:, 0], designated[:, 1])

    @jax.jit
    def sample_fn(params, key, contexts, designated):
        def sample(args):
            context, key, designated = args
            return fm_sample.apply(
                params, context, n_steps=cfg.train.rollout.gen_steps, method='gen', rngs=key,
                cond=condition(context, designated)
            ).edges

        keys = jax.random.split(key, contexts.n_node.shape[0])
        return jax.lax.map(sample, (contexts, keys, designated), batch_size=chunk_size).astype(contexts.edges.dtype)

    @jax.jit
    def score_fn(s_params, contexts, designated, z):
        def score(args):
            context, designated, z = args
            logits, means, _ = surrogate.apply(
                s_params, latent_graph(context, z), training=False, cond=condition(context, designated)
            )
            return jnp.sum(jax.nn.softmax(logits, axis=-1) * means, axis=-1)[0]

        return jax.lax.map(score, (contexts, designated, z), batch_size=chunk_size)

    def surrogate_loss_fn(s_params, contexts, designated, z, rates):
        def graph_loss(context, designated, z, rate):
            logits, means, scales = surrogate.apply(
                s_params, latent_graph(context, z), training=False, cond=condition(context, designated)
            )
            return mdn_loss(logits, means, scales, rate)[0]

        return jax.vmap(graph_loss)(contexts, designated, z, (rates - RATE_MEAN) / RATE_STD).mean()

    @jax.jit
    def surrogate_step_fn(s_params, s_opt_state, contexts, designated, z, rates):
        loss, grads = jax.value_and_grad(surrogate_loss_fn)(s_params, contexts, designated, z, rates)
        updates, s_opt_state = s_optimizer.update(grads, s_opt_state, s_params)
        s_params = optax.apply_updates(s_params, updates)
        return s_params, s_opt_state, loss

    def create_tx_fn(associations):
        @jax.jit
        def tx_fn(keys, contexts, z, designated):
            def to_tx(args):
                key, context, z, designated = args
                graph = decode_fn(latent_graph(context, z))

                if designated is None:
                    return connections_to_tx(associations, key, graph, deterministic)

                tx_key, designate_key = jax.random.split(key)
                tx = connections_to_tx(associations, tx_key, graph, deterministic)
                return designate_stations(designate_key, graph, *tx, designated[:, 0], designated[:, 1], deterministic)

            return jax.lax.map(to_tx, (keys, contexts, z, designated), batch_size=chunk_size)

        return tx_fn

    def x0_loss(v, x_t, x, t):
        err = x_t + (1 - t) * v - x
        return (err ** 2).mean() / jax.lax.stop_gradient(jnp.maximum(jnp.abs(err).mean(), 1e-5))

    def nft_loss(params, key, context, designated, x, reward, params_old, params_ref):
        # DiffusionNFT - https://www.arxiv.org/abs/2509.16117
        t_key, edges_key, nodes_key, globals_key = jax.random.split(key, 4)

        t = jnp.full((x.shape[0], 1), jax.random.uniform(t_key, maxval=0.99, dtype=x.dtype))
        noise = jax.random.normal(edges_key, x.shape, dtype=x.dtype)
        x_t = (1 - t) * noise + t * x
        cx = context._replace(
            edges=jnp.concatenate([context.edges, x_t[:, None]], axis=1),
            nodes=append_noise(nodes_key, context.nodes),
            globals=append_noise(globals_key, context.globals)
        )

        cond = condition(context, designated)
        v_pred = fm.apply(params, cx, t, training=False, cond=cond).edges
        v_old = fm.apply(params_old, cx, t, training=False, cond=cond).edges
        v_ref = fm.apply(params_ref, cx, t, training=False, cond=cond).edges

        v_plus = (1 - beta) * v_old + beta * v_pred
        v_minus = (1 + beta) * v_old - beta * v_pred

        loss_p = x0_loss(v_plus, x_t, x, t)
        loss_m = x0_loss(v_minus, x_t, x, t)
        kl = ((v_pred - v_ref) ** 2).mean()

        loss = (reward * loss_p + (1 - reward) * loss_m) * adv_clip / beta + kl_coef * kl
        return loss, kl

    def loss_fn(params, key, contexts, designated, x, rewards, params_old, params_ref):
        keys = jax.random.split(key, rewards.shape[0])
        loss, kl = jax.vmap(nft_loss, in_axes=(None, 0, 0, 0, 0, 0, None, None))(
            params, keys, contexts, designated, x, rewards, params_old, params_ref
        )
        return loss.mean(), kl.mean()

    @jax.jit
    def step_fn(params, opt_state, key, contexts, designated, x, rewards, params_old, params_ref):
        (loss, kl), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            params, key, contexts, designated, x, rewards, params_old, params_ref
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, kl, optax.global_norm(grads)

    def simulate(key, env, tx):
        keys = jax.random.split(key, tx[0].shape[0])
        return np.concatenate([
            np.asarray(env['batch_eval_fn'](keys[i:i + sim_chunk_size], *[a[i:i + sim_chunk_size] for a in tx]))
            for i in range(0, len(keys), sim_chunk_size)
        ])

    def rollout(key, params, contexts, designated, env, n_sim_repeats):
        sample_key, tx_key, sim_key = jax.random.split(key, 3)
        n_samples = contexts.n_node.shape[0]

        z = sample_fn(params, sample_key, contexts, designated)
        tx = env['tx_fn'](jax.random.split(tx_key, n_samples), contexts, z, designated)
        rewards = [simulate(key, env, tx) for key in jax.random.split(sim_key, n_sim_repeats)]
        return z, tx, np.mean(rewards, axis=0)

    def resimulate(key, env, tx, idxs, n_repeats):
        tx = [np.repeat(np.asarray(a)[idxs], n_repeats, axis=0) for a in tx]
        return simulate(key, env, tx).reshape(len(idxs), n_repeats)

    def evaluate(key, params, env):
        n_samples, n_repeats = cfg.train.eval.n_samples, cfg.train.eval.n_repeats
        selection = cfg.train.eval.selection
        select_fn = select_top if selection == 'top' else select_covering
        rollout_key, sim_key = jax.random.split(key)

        z, tx, rewards = rollout(rollout_key, params, env['eval_contexts'], env['eval_designated'], env, 1)
        rewards = rewards.reshape(-1, n_samples)
        scores = np.asarray(score_fn(s_params, env['eval_contexts'], env['eval_designated'], z)).reshape(-1, n_samples)
        served = np.asarray(tx[0])[..., env['sta_ids']].any(axis=-2).reshape(*scores.shape, -1)
        cover = served

        if selection == 'designated':
            designated = np.asarray(env['eval_designated'])[..., 1]
            cover = (designated[..., None] == env['sta_ids']).any(axis=-2).reshape(served.shape)

        offsets = n_samples * np.arange(rewards.shape[0])[:, None]

        metrics = {
            'sample_mean': rewards.mean().item(),
            'surrogate_rank_corr': rank_correlation(scores, rewards)
        }

        for top_k in cfg.train.eval.top_ks:
            idxs = np.stack([select_fn(s, c, top_k, env['coverage_ratio']) for s, c in zip(scores, cover)])
            metrics[f'top_{top_k}'] = resimulate(sim_key, env, tx, (idxs + offsets).reshape(-1), n_repeats).mean().item()
            metrics[f'top_{top_k}_coverage'] = np.take_along_axis(served, idxs[..., None], axis=1).any(axis=1).mean().item()

        return metrics

    key = jax.random.PRNGKey(cfg.train.seed)
    ctx_key, train_key = jax.random.split(key)

    fm_cfg = OmegaConf.load(os.path.join(cfg.fm_path, 'config.yaml'))
    fm = FlowMatching(**{**fm_cfg.model, 'dtype': cfg.train.nft.dtype})
    fm_sample = FlowMatching(**{**fm_cfg.model, 'dtype': cfg.train.rollout.dtype})
    encode_fn, decode_fn = load_fm_gnn(os.path.abspath(cfg.fm_path))
    surrogate, s_params = load_surrogate(os.path.abspath(cfg.surrogate_path))
    surrogate = surrogate.clone(dtype=cfg.train.surrogate.dtype)

    envs = []

    for i, name in enumerate(cfg.scenarios):
        scenario = get_scenario(name)
        n_designated = per_scenario(cfg.train.n_designated, i)
        cache = create_scenario_cache(scenario)
        scenario_eval_fn = create_scenario_eval_fn(scenario)
        ctx_key, pool_key = jax.random.split(ctx_key)

        sta_ids = np.asarray([sta for stas in scenario.associations.values() for sta in stas])

        eval_contexts = create_contexts(
            jax.random.PRNGKey(cfg.train.eval.seed), cache, scenario_eval_fn, encode_fn, cfg.train.eval.n_steps
        )
        eval_designated = None

        if n_designated > 0:
            eval_designated = jnp.concatenate([
                round_robin_designated(k, scenario.associations, n_designated, cfg.train.eval.n_samples)
                for k in jax.random.split(jax.random.PRNGKey(cfg.train.eval.seed + 1), cfg.train.eval.n_steps)
            ])

        envs.append({
            'name': name,
            'n_designated': n_designated,
            'coverage_ratio': per_scenario(cfg.train.eval.coverage_ratio, i),
            'sta_ids': sta_ids,
            'associations': scenario.associations,
            'tx_fn': create_tx_fn(scenario.associations),
            'batch_eval_fn': create_scenario_batch_eval_fn(scenario),
            'contexts': create_contexts(pool_key, cache, scenario_eval_fn, encode_fn, cfg.train.rollout.n_contexts),
            'eval_contexts': take_graphs(eval_contexts, np.repeat(np.arange(cfg.train.eval.n_steps), cfg.train.eval.n_samples)),
            'eval_designated': eval_designated
        })

    params_ref = restore_params(os.path.abspath(cfg.fm_path), 'ema_params')

    if any(env['n_designated'] > 0 for env in envs):
        edges_shape = envs[0]['contexts'].edges.shape
        params_ref = add_condition_input(params_ref, (edges_shape[-2] + 1) * edges_shape[-1])
        s_params = add_condition_input(s_params, edges_shape[-1])

    params, params_old = params_ref, params_ref

    if cfg.init_path is not None:
        params = params_old = restore_params(os.path.abspath(cfg.init_path), 'ema_params')
        s_params = restore_params(os.path.abspath(cfg.init_path), 'surrogate_params')

    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.optimizer.grad_clip),
        optax.adamw(
            cfg.optimizer.learning_rate, weight_decay=cfg.optimizer.weight_decay,
            b1=cfg.optimizer.b1, b2=cfg.optimizer.b2, eps=cfg.optimizer.eps
        )
    )
    opt_state = optimizer.init(params)

    s_optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.optimizer.grad_clip),
        optax.adamw(cfg.train.surrogate.learning_rate, weight_decay=0.0)
    )
    s_opt_state = s_optimizer.init(s_params)

    run_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    ckpt_manager = ocp.CheckpointManager(
        directory=run_path,
        options=ocp.CheckpointManagerOptions(max_to_keep=None if cfg.train.ckpt.intermediate else 1, create=True),
        item_names=['ema_params', 'params_old', 'opt_state', 'surrogate_params']
    )
    OmegaConf.save(config=cfg, f=os.path.join(run_path, 'rl_config.yaml'))
    OmegaConf.save(config=fm_cfg, f=os.path.join(run_path, 'config.yaml'))
    log_file = open(os.path.join(run_path, 'log.jsonl'), 'a')

    if cfg.train.log.type == 'wandb':
        wandb.init(project='lai4wifi', name=cfg.name, config=OmegaConf.to_container(cfg))

    n_groups = cfg.train.rollout.n_groups // len(envs)
    group_size = cfg.train.rollout.group_size
    minibatch_size = cfg.train.nft.minibatch_size

    start_time = time.time()

    for step in trange(cfg.train.n_steps + 1, desc='Training steps'):
        log_dict = {'step': step, 'time': time.time() - start_time}

        if step % cfg.train.eval.freq == 0 or step == cfg.train.n_steps:
            for env in envs:
                metrics = evaluate(jax.random.PRNGKey(cfg.train.eval.seed), params, env)
                log_dict.update({f'eval/{env["name"]}/{k}': v for k, v in metrics.items()})

        if step < cfg.train.n_steps:
            minibatches = []

            for env in envs:
                train_key, ctx_key, rollout_key = jax.random.split(train_key, 3)
                idxs = jax.random.choice(ctx_key, env['contexts'].n_node.shape[0], shape=(n_groups,))
                contexts = take_graphs(env['contexts'], jnp.repeat(idxs, group_size))
                designated = None

                if env['n_designated'] > 0:
                    train_key, designated_key = jax.random.split(train_key)
                    designated = sample_designated(designated_key, env['associations'], env['n_designated'], n_groups)
                    designated = jnp.repeat(designated, group_size, axis=0)

                z, tx, rewards_raw = rollout(
                    rollout_key, params_old, contexts, designated, env, cfg.train.rollout.n_sim_repeats
                )
                groups = rewards_raw.reshape(n_groups, group_size)
                served = np.asarray(tx[0])[..., env['sta_ids']].any(axis=-2).reshape(n_groups, group_size, -1)
                adv = (groups - groups.mean(axis=-1, keepdims=True)) / (rewards_raw.std() + 1e-4)
                rewards = 0.5 + 0.5 * np.clip(adv.reshape(-1) / adv_clip, -1.0, 1.0)

                for i in range(0, len(rewards), minibatch_size):
                    idxs = np.arange(i, min(i + minibatch_size, len(rewards)))
                    minibatches.append((
                        take_graphs(contexts, idxs), take_graphs(designated, idxs), z[idxs], rewards[idxs], rewards_raw[idxs]
                    ))

                log_dict[f'train/{env["name"]}/reward_mean'] = rewards_raw.mean().item()
                log_dict[f'train/{env["name"]}/reward_best'] = groups.max(axis=-1).mean().item()
                log_dict[f'train/{env["name"]}/reward_group_std'] = groups.std(axis=-1).mean().item()
                log_dict[f'train/{env["name"]}/group_coverage'] = served.any(axis=1).mean().item()

            losses, kls, grad_norms, s_losses = [], [], [], []

            for _ in range(cfg.train.nft.n_epochs):
                train_key, perm_key = jax.random.split(train_key)

                for i in jax.random.permutation(perm_key, len(minibatches)).tolist():
                    train_key, step_key = jax.random.split(train_key)
                    contexts, designated, z, rewards, _ = minibatches[i]
                    params, opt_state, loss, kl, grad_norm = step_fn(
                        params, opt_state, step_key, contexts, designated, z, rewards, params_old, params_ref
                    )
                    losses.append(loss)
                    kls.append(kl)
                    grad_norms.append(grad_norm)

            for _ in range(cfg.train.surrogate.n_epochs):
                train_key, perm_key = jax.random.split(train_key)

                for i in jax.random.permutation(perm_key, len(minibatches)).tolist():
                    contexts, designated, z, _, rates = minibatches[i]
                    s_params, s_opt_state, s_loss = surrogate_step_fn(s_params, s_opt_state, contexts, designated, z, rates)
                    s_losses.append(s_loss)

            eta = min(cfg.train.nft.eta_rate * step, cfg.train.nft.eta_max)
            params_old = optax.incremental_update(params, params_old, step_size=1 - eta)

            log_dict['rl/loss'] = np.mean(losses).item()
            log_dict['rl/kl'] = np.mean(kls).item()
            log_dict['rl/grad_norm'] = np.mean(grad_norms).item()
            log_dict['rl/eta'] = eta

            if s_losses:
                log_dict['surrogate/loss'] = np.mean(s_losses).item()

        if step % cfg.train.log.freq == 0 or step == cfg.train.n_steps:
            logging.info(log_dict)
            log_file.write(json.dumps(log_dict) + '\n')
            log_file.flush()
            if cfg.train.log.type == 'wandb':
                wandb.log(log_dict)

        if step % cfg.train.ckpt.freq == 0 or step == cfg.train.n_steps:
            ckpt_manager.save(step, args=ocp.args.Composite(
                ema_params=ocp.args.PyTreeSave(params),
                params_old=ocp.args.PyTreeSave(params_old),
                opt_state=ocp.args.PyTreeSave(opt_state),
                surrogate_params=ocp.args.PyTreeSave(s_params)
            ))

    log_file.close()
    ckpt_manager.wait_until_finished()
    ckpt_manager.close()


if __name__ == '__main__':
    main()
