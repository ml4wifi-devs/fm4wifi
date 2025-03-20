import logging
import os
from copy import deepcopy
from functools import partial

import hydra
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import wandb
from omegaconf import OmegaConf

from lai4wifi.flow_matching.model import FlowMatching
from lai4wifi.graphs.attributes import *
from lai4wifi.graphs.dataset import load_dataset
from lai4wifi.graphs.model import GnnAutoencoder
from lai4wifi.graphs.utils import eig_laplacian, nearest_power_of_two


def get_network_desc(target_conf, encode_fn, eig_k):
    desc_graph = deepcopy(target_conf.graph)
    edge_types = jnp.argmax(desc_graph.nodes.type, axis=-1)

    active = jnp.where((edge_types == EdgeType.STA.value)[:, None], jnp.array(IsActive.TRUE)[None], jnp.array(IsActive.NA)[None])
    mcs = jnp.zeros((edge_types.shape[0], len(McsValue)))
    mcs = mcs.at[:].set(jnp.array(McsValue.NA))
    rssi = jnp.zeros((edge_types.shape[0], 1))
    selected = jnp.zeros((edge_types.shape[0], len(IsSelected)))
    selected = selected.at[:].set(jnp.array(IsSelected.NA))
    selected = selected.at[target_conf.selected_idx].set(jnp.array(IsSelected.TRUE))
    success = jnp.zeros((edge_types.shape[0], 1))
    tx_power = jnp.zeros((edge_types.shape[0], len(TxPowerValue)))
    tx_power = tx_power.at[:].set(jnp.array(TxPowerValue.NA))

    desc_graph = desc_graph._replace(nodes=Connection(
        active=active,
        mcs=mcs,
        rssi=rssi,
        selected=selected,
        success=success,
        tx_power=tx_power,
        type=desc_graph.nodes.type
    ))
    network_desc = encode_fn(desc_graph).nodes
    evals, evects = eig_laplacian(desc_graph.senders, desc_graph.receivers, desc_graph.n_node[0], eig_k)

    return network_desc, evals, evects


def draw_configurations(scenario, n_timesteps, eig_k, key, encode_fn):
    *context_idxs, target_idx = jax.random.choice(key, len(scenario), (n_timesteps + 1,))
    context = [scenario[i].embedding.nodes for i in context_idxs]
    target_conf = scenario[target_idx].embedding.nodes
    network_desc, evals, evects = get_network_desc(scenario[target_idx], encode_fn, eig_k)
    return jnp.stack([network_desc, *context, target_conf], axis=1), evals, evects


def make_pos_emb(evals, evects):
    evals = jnp.expand_dims(evals, axis=(0, -1))
    evals = jnp.repeat(evals, evects.shape[0], axis=0)
    evects = jnp.expand_dims(evects, axis=-1)
    return jnp.concatenate((evects, evals), axis=-1)


def pad_and_mask(batch):
    batch, evals, evects = zip(*batch)

    sizes = [b.shape[0] for b in batch]
    padding = nearest_power_of_two(sum(sizes)) - sum(sizes)
    sizes = [0] + sizes + [padding]
    sizes = jnp.cumsum(jnp.array(sizes))

    batch += (jnp.zeros((padding,) + batch[0].shape[1:]),)
    evals += (jnp.full_like(evals[0], fill_value=jnp.nan),)
    evects += (jnp.full((padding, evects[0].shape[1]), fill_value=jnp.nan),)
    pos_emb = jnp.vstack([make_pos_emb(e, v) for e, v in zip(evals, evects)])
    batch = jnp.vstack(batch)

    attn_mask = jnp.zeros((1, batch.shape[0], batch.shape[0]))
    for start, end in zip(sizes[:-1], sizes[1:]):
        attn_mask = attn_mask.at[:, start:end, start:end].set(1)

    pad_mask = jnp.ones(batch.shape[0])
    pad_mask = pad_mask.at[-padding:].set(0.)

    return jax.tree.map(lambda x: x[None], (batch, pos_emb, attn_mask, pad_mask))


def load(dataset_path, batch_size, n_timesteps, eig_k, key, encode_fn):
    dataset = load_dataset(dataset_path)

    while True:
        nodes = 0
        batch = []

        while True:
            key, idx_key, conf_key = jax.random.split(key, 3)
            idx = jax.random.choice(idx_key, len(dataset))
            graph_size = dataset[idx][0].embedding.nodes.shape[0]

            if nodes + graph_size < batch_size:
                batch.append(draw_configurations(dataset[idx], n_timesteps, eig_k, conf_key, encode_fn))
                nodes += graph_size
            else:
                break

        yield pad_and_mask(batch)


@hydra.main(config_path="configs", config_name="fm", version_base=None)
def main(cfg):

    @jax.jit
    def loss_fn(params, key, batch, schedule='uniform'):
        cx, pos_emb, attn_mask, pad_mask = batch
        z_key, t_key, model_key = jax.random.split(key, 3)

        if schedule == 'log-normal':
            t = jax.random.normal(t_key, (cx.shape[1],))
            t = 1 / (1 + jnp.exp(-t))
        elif schedule == 'uniform':
            t = jax.random.uniform(t_key, (cx.shape[1],), minval=0.0, maxval=0.99)
        else:
            raise ValueError(f'Invalid schedule: {schedule}')

        z = jax.random.normal(z_key, cx.shape[:2] + (cfg.model.latent_size,))
        t_b = t.reshape((1, -1, 1))

        x = cx[..., -1, :]
        x_t = (1 - t_b) * z + t_b * x
        v_t = x - z
        cx = cx.at[..., -1, :].set(x_t)

        v_pred = model.apply(params, cx, pos_emb, t, attn_mask, rngs=model_key)
        loss = optax.squared_error(v_pred, v_t).mean(axis=-1)
        loss = (pad_mask * loss).sum() / pad_mask.sum()
        return loss

    @jax.jit
    def step_fn(params, opt_state, key, batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, key, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def generate_fn(params, key, batch):
        return model.apply(params, *batch, rngs=key, method='gen')

    with ocp.CheckpointManager(os.path.abspath(cfg.gnn_path), item_names=['params']) as ckpt_manager:
        gnn_params = ckpt_manager.restore(None, args=ocp.args.Composite(params=ocp.args.PyTreeRestore())).params

    gnn_cfg = OmegaConf.load(os.path.join(cfg.gnn_path, 'config.yaml'))
    gnn = GnnAutoencoder(**gnn_cfg.model)
    encode_fn = jax.jit(partial(gnn.apply, gnn_params, method='encode'))

    key = jax.random.PRNGKey(cfg.train.seed)
    init_key, train_key, val_key, data_key, val_data_key = jax.random.split(key, 5)

    data = load(cfg.dataset, cfg.train.batch_size, cfg.train.n_timesteps, cfg.train.eig_k, data_key, encode_fn)
    val_data = iter(load(cfg.val_dataset, cfg.train.batch_size, cfg.train.n_timesteps, cfg.train.eig_k, val_data_key, encode_fn))
    sample = next(iter(data))
    sample = (sample[0], sample[1], jnp.zeros((1, sample[0].shape[1])), sample[2])

    model = FlowMatching(**cfg.model)
    params = model.init({'params': init_key}, *sample)
    print(model.tabulate(jax.random.PRNGKey(0), *sample))

    optimizer = optax.adamw(**cfg.optimizer)
    opt_state = optimizer.init(params)

    ckpt_path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    ckpt_manager = ocp.CheckpointManager(
        directory=ckpt_path,
        options=ocp.CheckpointManagerOptions(max_to_keep=None if cfg.train.ckpt.intermediate else 1, create=True),
        item_names=['params', 'opt_state', 'train_key', 'val_key']
    )
    OmegaConf.save(config=cfg, f=os.path.join(ckpt_path, 'config.yaml'))

    if cfg.train.log.type == 'wandb':
        wandb.init(project='lai4wifi', name=cfg.name)

    for step, batch in zip(range(cfg.train.n_steps), data):
        log_dict = {}

        train_key, subkey = jax.random.split(train_key)
        params, opt_state, loss = step_fn(params, opt_state, subkey, batch)

        log_dict['step'] = step
        log_dict['train/loss'] = loss.item()

        if step % cfg.train.val.freq == 0 or step == cfg.train.n_steps - 1:
            val_loss = 0.0

            for _ in range(cfg.train.val.n_steps):
                val_key, subkey = jax.random.split(val_key)
                val_batch = next(val_data)
                loss = loss_fn(params, subkey, val_batch)
                val_loss += loss.item()

            log_dict['val/loss'] = val_loss / cfg.train.val.n_steps

        if step % cfg.train.log.freq == 0 or step == cfg.train.n_steps - 1:
            logging.info(log_dict)
            if cfg.train.log.type == 'wandb':
                wandb.log(log_dict)

        if step % cfg.train.ckpt.freq == 0 or step == cfg.train.n_steps - 1:
            ckpt_manager.save(step, args=ocp.args.Composite(
                params=ocp.args.PyTreeSave(params),
                opt_state=ocp.args.PyTreeSave(opt_state),
                train_key=ocp.args.JaxRandomKeySave(train_key),
                val_key=ocp.args.JaxRandomKeySave(val_key)
            ))

    ckpt_manager.wait_until_finished()
    ckpt_manager.close()


if __name__ == '__main__':
    main()
