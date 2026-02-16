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
from omegaconf import OmegaConf

from lai4wifi.dataset.utils import load_dataset
from lai4wifi.flow_matching.model import FlowMatching
from lai4wifi.flow_matching.mmd import compute_mmd
from lai4wifi.flow_matching.utils import gen_sample, jax_repeat
from lai4wifi.graphs.model import GnnAutoencoder


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
    def accuracy_fn(params, key, batch):
        generated = model.apply(params, gen_sample(batch), method='gen', rngs={'lai': key})
        target = batch._replace(edges=batch.edges[:, -1], nodes=batch.nodes[:, -1], globals=batch.globals[:, -1])
        pred_decoded, target_decoded = decode_fn(generated), decode_fn(target)

        def mixture_accuracy(logits, target_logits):
            target_cls = jnp.argmax(target_logits, axis=-1)
            na_idx = target_logits.shape[-1] - 1
            target_na = target_cls == na_idx
            na_acc = (logits[..., -1] > 0.0) == target_na
            pred_acc = (jnp.argmax(logits[..., :-1], axis=-1) == jnp.argmax(target_logits[..., :-1], axis=-1))
            pred_acc = (1 - target_na.astype(float)) * pred_acc.astype(float)
            return jnp.asarray([na_acc, pred_acc, 1 - target_na])

        def isclose_accuracy(values, target_values, atol=0.1):
            pred_acc = jnp.isclose(values, target_values, atol=atol).mean(axis=-1)
            na_acc = jnp.ones_like(pred_acc)
            return jnp.asarray([na_acc, pred_acc, na_acc])

        def metrics_fn(target, pred):
            mask = jraph.get_edge_padding_mask(target)[None]
            active_acc   = mask * mixture_accuracy(pred.edges.active, target.edges.active)
            mcs_acc      = mask * mixture_accuracy(pred.edges.mcs, target.edges.mcs)
            rssi_acc     = mask * isclose_accuracy(pred.edges.rssi, target.edges.rssi)
            selected_acc = mask * mixture_accuracy(pred.edges.selected, target.edges.selected)
            success_acc  = mask * isclose_accuracy(jax.nn.sigmoid(pred.edges.success), jax.nn.sigmoid(target.edges.success))
            tx_power_acc = mask * mixture_accuracy(pred.edges.tx_power, target.edges.tx_power)
            type_acc     = mask * mixture_accuracy(pred.edges.type, target.edges.type)
            metrics = jnp.array([active_acc, mcs_acc, rssi_acc, selected_acc, success_acc, tx_power_acc, type_acc])

            metrics, pred_mask = metrics[:, :-1], metrics[:, -1]
            metrics = metrics.at[:, -1].mul(pred_mask)
            metrics = metrics.sum(axis=-1)
            na_metrics = metrics[:, 0] / mask.sum(axis=-1)
            pred_metrics = metrics[:, 1] / pred_mask.sum(axis=-1)
            return na_metrics, pred_metrics
            
        na_metrics, pred_metrics = metrics_fn(target_decoded, pred_decoded)
        return na_metrics, pred_metrics

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

    def log_vector(log_dict, split, type, values, n_steps=1):
        log_dict[f'{split}/{type}/active'] = values[0] / n_steps
        log_dict[f'{split}/{type}/mcs'] = values[1] / n_steps
        log_dict[f'{split}/{type}/rssi'] = values[2] / n_steps
        log_dict[f'{split}/{type}/selected'] = values[3] / n_steps
        log_dict[f'{split}/{type}/success'] = values[4] / n_steps
        log_dict[f'{split}/{type}/tx_power'] = values[5] / n_steps
        log_dict[f'{split}/{type}/type'] = values[6] / n_steps

    key = jax.random.PRNGKey(cfg.train.seed)
    init_key, train_key, val_key, mmd_key = jax.random.split(key, 4)

    gnn_cfg = OmegaConf.load(os.path.join(cfg.gnn_path, 'config.yaml'))
    with ocp.CheckpointManager(os.path.abspath(cfg.gnn_path), item_names=['params']) as ckpt_manager:
        gnn_params = ckpt_manager.restore(None, args=ocp.args.Composite(params=ocp.args.PyTreeRestore())).params
    
    gnn = GnnAutoencoder(**gnn_cfg.model)
    decode_fn = jax.jit(partial(gnn.apply, gnn_params, method='decode'))

    data = load_dataset(cfg.dataset_path)
    val_data = load_dataset(cfg.val_dataset_path)
    val_data_idx = 0
    sample = (data[0], jnp.zeros((data[0].edges.shape[0], 1)))

    model = FlowMatching(**cfg.model)
    params = model.init({'params': init_key}, *sample)
    ema_params = deepcopy(params)
    print(model.tabulate(jax.random.PRNGKey(0), *sample))

    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.optimizer.grad_clip),
        optax.adamw(cfg.optimizer.learning_rate, weight_decay=cfg.optimizer.weight_decay, b1=cfg.optimizer.b1, b2=cfg.optimizer.b2)
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
        log_dict = {'step': step}

        train_key, subkey = jax.random.split(train_key)
        params, opt_state, loss = step_fn(params, opt_state, subkey, batch)
        ema_params = optax.incremental_update(params, ema_params, 1.0 - cfg.optimizer.ema_decay)

        if step % cfg.train.val.freq == 0 or step == cfg.train.n_steps - 1:
            val_loss, val_loss_ema = 0.0, 0.0
            val_na_metrics, val_pred_metrics = np.zeros(7), np.zeros(7)
            val_na_metrics_ema, val_pred_metrics_ema = np.zeros(7), np.zeros(7)

            for _ in range(cfg.train.val.n_steps):
                val_key, subkey, acc_subkey, acc_ema_subkey = jax.random.split(val_key, 4)
                val_batch = val_data[val_data_idx]
                val_data_idx = (val_data_idx + 1) % len(val_data)
                val_loss += loss_fn(params, subkey, val_batch)
                val_loss_ema += loss_fn(ema_params, subkey, val_batch)
                na_acc, pred_acc = accuracy_fn(params, acc_subkey, val_batch)
                na_acc_ema, pred_acc_ema = accuracy_fn(ema_params, acc_ema_subkey, val_batch)
                val_na_metrics += na_acc
                val_pred_metrics += pred_acc
                val_na_metrics_ema += na_acc_ema
                val_pred_metrics_ema += pred_acc_ema

            log_dict['val/loss'] = val_loss / cfg.train.val.n_steps
            log_dict['val/loss_ema'] = val_loss_ema / cfg.train.val.n_steps
            log_vector(log_dict, 'val', 'na_acc', val_na_metrics, cfg.train.val.n_steps)
            log_vector(log_dict, 'val', 'pred_acc', val_pred_metrics, cfg.train.val.n_steps)
            log_vector(log_dict, 'val', 'na_acc_ema', val_na_metrics_ema, cfg.train.val.n_steps)
            log_vector(log_dict, 'val', 'pred_acc_ema', val_pred_metrics_ema, cfg.train.val.n_steps)

        if cfg.train.mmd.enabled and (step % cfg.train.mmd.freq == 0 or step == cfg.train.n_steps - 1):
            mmd_key, mmd_subkey, mmd_ema_subkey = jax.random.split(mmd_key, 3)
            mmd_batch = val_data[val_data_idx]
            val_data_idx = (val_data_idx + 1) % len(val_data)
            
            mmd_val = mmd_fn(params, mmd_batch, mmd_subkey, cfg.train.mmd.n_samples)
            mmd_val_ema = mmd_fn(ema_params, mmd_batch, mmd_ema_subkey, cfg.train.mmd.n_samples)
            
            log_dict['val/mmd'] = mmd_val
            log_dict['val/mmd_ema'] = mmd_val_ema

        if step % cfg.train.log.freq == 0 or step == cfg.train.n_steps - 1:
            log_dict['train/loss'] = loss
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
