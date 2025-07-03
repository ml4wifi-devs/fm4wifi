import os
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'

import logging
from functools import partial

import hydra
import jax
import jax.numpy as jnp
import jraph
import optax
import orbax.checkpoint as ocp
import wandb
from omegaconf import OmegaConf

from lai4wifi.flow_matching.model import FlowMatching
from lai4wifi.autoencoder.attributes import *
from lai4wifi.autoencoder.dataset import load_dataset
from lai4wifi.autoencoder.model import GnnAutoencoder
from lai4wifi.autoencoder.utils import make_batch


def get_network_desc(target_conf, selected_idx, encode_fn):
    edge_types = jnp.argmax(target_conf.edges.type, axis=-1)

    active = jnp.where((edge_types == EdgeType.STA.value)[:, None], jnp.array(IsActive.TRUE)[None], jnp.array(IsActive.NA)[None])
    mcs = jnp.zeros((edge_types.shape[0], len(McsValue)))
    mcs = mcs.at[:].set(jnp.array(McsValue.NA))
    rssi = jnp.zeros((edge_types.shape[0], 1))
    selected = jnp.zeros((edge_types.shape[0], len(IsSelected)))
    selected = selected.at[:].set(jnp.array(IsSelected.NA))
    selected = selected.at[selected_idx].set(jnp.array(IsSelected.TRUE))
    success = jnp.zeros((edge_types.shape[0], 1))
    tx_power = jnp.zeros((edge_types.shape[0], len(TxPowerValue)))
    tx_power = tx_power.at[:].set(jnp.array(TxPowerValue.NA))

    desc_graph = target_conf._replace(edges=Connection(
        active=active,
        mcs=mcs,
        rssi=rssi,
        selected=selected,
        success=success,
        tx_power=tx_power,
        type=target_conf.edges.type
    ))

    return encode_fn(desc_graph)


def draw_configurations(scenario, n_timesteps, key, encode_fn):
    *context_idxs, target_idx = jax.random.choice(key, len(scenario), (n_timesteps + 1,))

    context = [scenario[i].embedding for i in context_idxs]
    target_conf = scenario[target_idx].embedding
    network_desc = get_network_desc(scenario[target_idx].graph, scenario[target_idx].selected_idx, encode_fn)

    all_desc = jax.tree.map(lambda *x: jnp.stack(x, axis=1), network_desc, *context, target_conf)
    return network_desc._replace(
        edges=all_desc.edges,
        nodes=all_desc.nodes,
        globals=all_desc.globals
    )


def load(dataset_path, batch_size, n_timesteps, key, encode_fn):
    dataset = load_dataset(dataset_path)

    while True:
        edges = 0
        batch = []

        while True:
            key, idx_key, conf_key = jax.random.split(key, 3)
            idx = jax.random.choice(idx_key, len(dataset))
            graph_size = dataset[idx][0].embedding.edges.shape[0]

            if edges + graph_size < batch_size:
                batch.append(draw_configurations(dataset[idx], n_timesteps, conf_key, encode_fn))
                edges += graph_size
            else:
                break

        yield make_batch(batch)


@hydra.main(config_path='configs', config_name='fm', version_base=None)
def main(cfg):

    @jax.jit
    def loss_fn(params, key, batch, schedule='uniform'):
        z_key, t_key, model_key = jax.random.split(key, 3)

        if schedule == 'log-normal':
            t = jax.random.normal(t_key)
            t = 1 / (1 + jnp.exp(-t))
        elif schedule == 'uniform':
            t = jax.random.uniform(t_key, minval=0.0, maxval=0.99)
        else:
            raise ValueError(f'Invalid schedule: {schedule}')

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
            nodes=noisy_example(batch.nodes, t, nodes_key)[0],
            globals=noisy_example(batch.globals, t, globals_key)[0]
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

    @jax.jit
    def generate_fn(params, key, batch):
        return model.apply(params, *batch, rngs=key, method='gen')

    gnn_cfg = OmegaConf.load(os.path.join(cfg.gnn_path, 'config.yaml'))
    with ocp.CheckpointManager(os.path.abspath(cfg.gnn_path), item_names=['params']) as ckpt_manager:
        gnn_params = ckpt_manager.restore(None, args=ocp.args.Composite(params=ocp.args.PyTreeRestore())).params

    gnn = GnnAutoencoder(**gnn_cfg.model)
    encode_fn = jax.jit(partial(gnn.apply, gnn_params, method='encode'))

    key = jax.random.PRNGKey(cfg.train.seed)
    init_key, train_key, val_key, data_key, val_data_key = jax.random.split(key, 5)

    data = load(cfg.dataset, cfg.train.batch_size, cfg.train.n_timesteps, data_key, encode_fn)
    val_data = iter(load(cfg.val_dataset, cfg.train.batch_size, cfg.train.n_timesteps, val_data_key, encode_fn))
    sample = next(iter(data))
    sample = (sample, jnp.zeros(sample.edges.shape[0]))

    model = FlowMatching(**cfg.model)
    params = model.init({'params': init_key}, *sample)
    print(model.tabulate(jax.random.PRNGKey(0), *sample))

    lr = optax.cosine_onecycle_schedule(cfg.train.n_steps, cfg.optimizer.learning_rate, pct_start=0.1)
    optimizer = optax.adamw(lr, weight_decay=cfg.optimizer.weight_decay)
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
