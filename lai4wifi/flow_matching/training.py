import os
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_ENABLE_X64'] = 'True'

import logging
from copy import deepcopy
from functools import partial
from glob import glob

import hydra
import jax
import jax.numpy as jnp
import jraph
import optax
import orbax.checkpoint as ocp
import wandb
from omegaconf import OmegaConf

from lai4wifi.flow_matching.model import FlowMatching
from lai4wifi.flow_matching.mmd import compute_mmd
from lai4wifi.flow_matching.utils import gen_sample, random_tx, tx_to_jraph
from lai4wifi.graphs.dataset import load_dataset
from lai4wifi.graphs.model import GnnAutoencoder
from lai4wifi.graphs.utils import make_batch, unbatch


def prepare_context(context, encode_fn):
    batched = make_batch(context)
    encoded = encode_fn(batched)
    unbatched = unbatch(encoded)
    ctx = jax.tree.map(lambda *x: jnp.stack(x, axis=1), *unbatched)
    return unbatched[0]._replace(edges=ctx.edges, nodes=ctx.nodes, globals=ctx.globals)


def draw_configurations(dataset_item, n_timesteps, key, encode_fn):
    idx_key, *ctx_keys = jax.random.split(key, n_timesteps + 2)
    
    target_idx = jax.random.choice(idx_key, len(dataset_item.configurations))
    target_conf = dataset_item.configurations[target_idx]
    context = []
    
    for i in range(n_timesteps):
        tx_key, sim_key = jax.random.split(ctx_keys[i])
        tx = random_tx(tx_key, dataset_item.scenario)
        _, _, internals = dataset_item.scenario(sim_key, *tx, return_internals=True)
        context.append(tx_to_jraph(dataset_item.scenario, *tx, internals))
        
    context = prepare_context(context, encode_fn)

    all_desc = jax.tree.map(lambda *x: jnp.stack(x, axis=1), *context, target_conf)
    return target_conf._replace(
        edges=all_desc.edges,
        nodes=all_desc.nodes,
        globals=all_desc.globals
    )


def load(dataset_path, batch_size, n_timesteps, key, val, encode_fn):
    dataset_paths = glob(f'{dataset_path}/*.pkl.lz4')
    dataset_paths = filter(lambda p: ('val' in p) == val, dataset_paths)
    dataset_paths = filter(lambda p: 'random' not in p, dataset_paths)
    dataset = sum((load_dataset(path) for path in dataset_paths), start=[])

    while True:
        edges = 0
        batch = []

        while True:
            key, idx_key, conf_key = jax.random.split(key, 3)
            idx = jax.random.choice(idx_key, len(dataset))

            if len(dataset[idx].configurations) == 0:
                continue
            
            graph_size = dataset[idx].configurations[0].edges.shape[0]

            if edges + graph_size < batch_size:
                batch.append(draw_configurations(dataset[idx], n_timesteps, conf_key, encode_fn))
                edges += graph_size
            elif edges == 0:
                continue
            else:
                break

        yield make_batch(batch)


@partial(jax.jit, static_argnames=['axis', 'total_repeat_length'])
def jax_repeat(a, repeats, axis, total_repeat_length):
    return jnp.repeat(a, repeats, axis, total_repeat_length=total_repeat_length)


def draw_t(key, batch, schedule='uniform'):
    if schedule == 'log-normal':
        t = jax.random.normal(key, (batch.n_edge.shape[0], 1))
        t = 1 / (1 + jnp.exp(-t))
    elif schedule == 'uniform':
        t = jax.random.uniform(key, (batch.n_edge.shape[0], 1), minval=0.0, maxval=0.99)
    else:
        raise ValueError(f'Invalid schedule: {schedule}')
    return jax_repeat(t, batch.n_edge, axis=0, total_repeat_length=batch.edges.shape[0])


def noisy_example(key, batch, t=0):
    z = jax.random.normal(key, (batch.shape[0], batch.shape[-1]))
    x = batch[:, -1]
    x_t = (1 - t) * z + t * x
    v_t = x - z
    return batch.at[:, -1].set(x_t), v_t


@hydra.main(config_path='configs', config_name='fm', version_base=None)
def main(cfg):

    @jax.jit
    def loss_fn(params, key, batch):
        t_key, edges_key, nodes_key, globals_key, model_key = jax.random.split(key, 5)

        t = draw_t(t_key, batch)
        noisy_edges, v_t = noisy_example(edges_key, batch.edges, t)
        cx = batch._replace(
            edges=noisy_edges,
            nodes=noisy_example(nodes_key, batch.nodes)[0],
            globals=noisy_example(globals_key, batch.globals)[0]
        )

        v_pred = model.apply(params, cx, t, rngs=model_key)
        mask = jraph.get_edge_padding_mask(v_pred)
        loss = optax.squared_error(v_pred.edges, v_t).mean(axis=-1)
        loss = (mask * loss).sum() / mask.sum()
        return loss

    @jax.jit
    def step_fn(params, opt_state, key, batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, key, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @partial(jax.jit, static_argnames=['n_samples'])
    def mmd_fn(params, batch, key, n_samples):
        def _mmd(params, batch, mask, key):
            generated = model.apply(params, gen_sample(batch), method='gen', rngs=key)
            return compute_mmd(batch.edges[:, -1], generated.edges, mask, mask)

        mask = jraph.get_edge_padding_mask(batch)
        mmds = jax.vmap(_mmd, in_axes=(None, None, None, 0))(params, batch, mask, jax.random.split(key, n_samples))
        return mmds.mean()

    key = jax.random.PRNGKey(cfg.train.seed)
    init_key, train_key, val_key, mmd_key, data_key, val_data_key = jax.random.split(key, 6)

    gnn_cfg = OmegaConf.load(os.path.join(cfg.gnn_path, 'config.yaml'))
    with ocp.CheckpointManager(os.path.abspath(cfg.gnn_path), item_names=['params']) as ckpt_manager:
        gnn_params = ckpt_manager.restore(None, args=ocp.args.Composite(params=ocp.args.PyTreeRestore())).params
    
    gnn = GnnAutoencoder(**gnn_cfg.model)
    encode_fn = jax.jit(partial(gnn.apply, gnn_params, method='encode'))

    data = load(cfg.dataset_path, cfg.train.batch_size, cfg.train.n_timesteps, data_key, val=False, encode_fn=encode_fn)
    val_data = iter(load(cfg.dataset_path, cfg.train.batch_size, cfg.train.n_timesteps, val_data_key, val=True, encode_fn=encode_fn))
    sample = next(iter(data))
    sample = (sample, jnp.zeros((sample.edges.shape[0], 1)))

    model = FlowMatching(**cfg.model)
    params = model.init({'params': init_key}, *sample)
    ema_params = deepcopy(params)
    print(model.tabulate(jax.random.PRNGKey(0), *sample))

    lr = optax.cosine_onecycle_schedule(cfg.train.n_steps, cfg.optimizer.learning_rate, pct_start=0.1)
    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.optimizer.grad_clip),
        optax.adamw(lr, weight_decay=cfg.optimizer.weight_decay, b1=cfg.optimizer.b1, b2=cfg.optimizer.b2)
    )
    opt_state = optimizer.init(params)

    ckpt_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    ckpt_manager = ocp.CheckpointManager(
        directory=ckpt_path,
        options=ocp.CheckpointManagerOptions(max_to_keep=None if cfg.train.ckpt.intermediate else 1, create=True),
        item_names=['params', 'ema_params', 'opt_state', 'train_key', 'val_key']
    )
    OmegaConf.save(config=cfg, f=os.path.join(ckpt_path, 'config.yaml'))

    if cfg.train.log.type == 'wandb':
        wandb.init(project='lai4wifi', name=cfg.name)

    for step, batch in zip(range(cfg.train.n_steps), data):
        log_dict = {}

        train_key, subkey = jax.random.split(train_key)
        params, opt_state, loss = step_fn(params, opt_state, subkey, batch)
        ema_params = optax.incremental_update(params, ema_params, 1.0 - cfg.optimizer.ema_decay)

        log_dict['step'] = step
        log_dict['train/loss'] = loss.item()

        if step % cfg.train.val.freq == 0 or step == cfg.train.n_steps - 1:
            val_loss, val_loss_ema = 0.0, 0.0

            for _ in range(cfg.train.val.n_steps):
                val_key, subkey = jax.random.split(val_key)
                val_batch = next(val_data)
                loss = loss_fn(params, subkey, val_batch)
                loss_ema = loss_fn(ema_params, subkey, val_batch)
                val_loss += loss.item()
                val_loss_ema += loss_ema.item()

            log_dict['val/loss'] = val_loss / cfg.train.val.n_steps
            log_dict['val/loss_ema'] = val_loss_ema / cfg.train.val.n_steps

        if cfg.train.mmd.enabled and (step % cfg.train.mmd.freq == 0 or step == cfg.train.n_steps - 1):
            mmd_key, mmd_subkey, mmd_ema_subkey = jax.random.split(mmd_key, 3)
            mmd_batch = next(val_data)
            
            mmd_val = mmd_fn(params, mmd_batch, mmd_subkey, cfg.train.mmd.n_samples)
            mmd_val_ema = mmd_fn(ema_params, mmd_batch, mmd_ema_subkey, cfg.train.mmd.n_samples)
            
            log_dict['val/mmd'] = mmd_val.item()
            log_dict['val/mmd_ema'] = mmd_val_ema.item()

        if step % cfg.train.log.freq == 0 or step == cfg.train.n_steps - 1:
            logging.info(log_dict)
            if cfg.train.log.type == 'wandb':
                wandb.log(log_dict)

        if step % cfg.train.ckpt.freq == 0 or step == cfg.train.n_steps - 1:
            ckpt_manager.save(step, args=ocp.args.Composite(
                params=ocp.args.PyTreeSave(params),
                ema_params=ocp.args.PyTreeSave(ema_params),
                opt_state=ocp.args.PyTreeSave(opt_state),
                train_key=ocp.args.JaxRandomKeySave(train_key),
                val_key=ocp.args.JaxRandomKeySave(val_key)
            ))

    ckpt_manager.wait_until_finished()
    ckpt_manager.close()


if __name__ == '__main__':
    main()
