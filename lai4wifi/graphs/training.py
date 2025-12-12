import os
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_ENABLE_X64'] = 'True'

import logging
from glob import glob

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
from lai4wifi.graphs.model import GnnAutoencoder
from lai4wifi.graphs.utils import make_batch


def load(batch_size, key, val_dataset=False):
    all_datasets = []

    for file in glob(f'{os.path.dirname(__file__)}/datasets/*.pkl.lz4'):
        if (val_dataset and 'val' not in file) or (not val_dataset and 'val' in file):
            continue

        dataset = load_dataset(file)
        dataset = sum((d.configurations for d in dataset), start=[])
        all_datasets += dataset

    while True:
        key, subkey = jax.random.split(key)
        idxs = jax.random.choice(subkey, len(all_datasets), (batch_size,))
        yield make_batch([all_datasets[i] for i in idxs])


@hydra.main(config_path="configs", config_name="gnn", version_base=None)
def main(cfg):
    loss_weights = jnp.array(cfg.train.loss_weights, dtype=jnp.float64)
    loss_weights /= loss_weights.sum()
    kl_loss_weight = cfg.train.kl_loss_weight

    @jax.jit
    def loss_fn(params, key, batch):
        pred, z_mean, z_log_var = model.apply(params, batch, rngs={'lai': key})

        def kl_loss(mean, log_var):
            return -0.5 * (1. + log_var - mean ** 2 - jnp.exp(log_var)).mean(axis=-1)

        def mixture_categorical_loss(logits, labels):
            na_logits, na_labels = logits[..., -1], labels[..., -1]
            softmax_logits, softmax_labels = logits[..., :-1], labels[..., :-1]
            na_loss = optax.sigmoid_binary_cross_entropy(na_logits, na_labels)
            softmax_loss = optax.softmax_cross_entropy(softmax_logits, softmax_labels)
            return na_loss + (1 - na_labels) * softmax_loss

        def weighted_loss_fn(batch, pred, z_mean, z_log_var):
            mask = jraph.get_edge_padding_mask(pred)
            active_loss   = mask * mixture_categorical_loss(pred.edges.active, batch.edges.active)
            mcs_loss      = mask * mixture_categorical_loss(pred.edges.mcs, batch.edges.mcs)
            rssi_loss     = mask * optax.squared_error(pred.edges.rssi, batch.edges.rssi).mean(axis=-1)
            selected_loss = mask * mixture_categorical_loss(pred.edges.selected, batch.edges.selected)
            success_loss  = mask * optax.sigmoid_binary_cross_entropy(pred.edges.success, batch.edges.success).mean(axis=-1)
            tx_power_loss = mask * mixture_categorical_loss(pred.edges.tx_power, batch.edges.tx_power)
            type_loss     = mask * mixture_categorical_loss(pred.edges.type, batch.edges.type)
            losses = jnp.array([active_loss, mcs_loss, rssi_loss, selected_loss, success_loss, tx_power_loss, type_loss]).sum(axis=-1)
            losses /= mask.sum()
            kl_reg_loss   = mask * kl_loss(z_mean.edges, z_log_var.edges)
            kl_reg_loss   = kl_reg_loss.sum() / mask.sum()
            return losses @ loss_weights + kl_loss_weight * kl_reg_loss, kl_reg_loss, losses

        def mixture_accuracy(logits, labels):
            na_logits, na_labels = logits[..., -1], labels[..., -1]
            softmax_logits, softmax_labels = logits[..., :-1], labels[..., :-1]
            na_acc = (na_logits > 0.5) == na_labels
            softmax_acc = (softmax_logits.argmax(axis=-1) == softmax_labels.argmax(axis=-1))
            softmax_acc = (1 - na_labels) * softmax_acc
            return jnp.asarray([na_acc, softmax_acc, 1 - na_labels])

        def isclose_accuracy(value, labels):
            pred_acc = jnp.isclose(value, labels, atol=0.1).mean(axis=-1)
            na_acc = jnp.ones_like(pred_acc)
            return jnp.asarray([na_acc, pred_acc, na_acc])

        def metrics_fn(batch, pred):
            mask = jraph.get_edge_padding_mask(pred)[None]
            active_acc   = mask * mixture_accuracy(pred.edges.active, batch.edges.active)
            mcs_acc      = mask * mixture_accuracy(pred.edges.mcs, batch.edges.mcs)
            rssi_acc     = mask * isclose_accuracy(pred.edges.rssi, batch.edges.rssi)
            selected_acc = mask * mixture_accuracy(pred.edges.selected, batch.edges.selected)
            success_acc  = mask * isclose_accuracy(jax.nn.sigmoid(pred.edges.success), batch.edges.success)
            tx_power_acc = mask * mixture_accuracy(pred.edges.tx_power, batch.edges.tx_power)
            type_acc     = mask * mixture_accuracy(pred.edges.type, batch.edges.type)
            metrics = jnp.array([active_acc, mcs_acc, rssi_acc, selected_acc, success_acc, tx_power_acc, type_acc])

            metrics, pred_mask = metrics[:, :-1], metrics[:, -1]
            metrics = metrics.at[:, -1].mul(pred_mask)
            metrics = metrics.sum(axis=-1)
            na_metrics = metrics[:, 0] / mask.sum(axis=-1)
            pred_metrics = metrics[:, 1] / pred_mask.sum(axis=-1)
            return na_metrics, pred_metrics

        loss, kl_loss, other_losses = weighted_loss_fn(batch, pred, z_mean, z_log_var)
        na_metrics, pred_metrics = metrics_fn(batch, pred)
        return loss, (kl_loss, other_losses, na_metrics, pred_metrics)

    @jax.jit
    def step_fn(params, opt_state, key, batch):
        (loss, *other), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, key, batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, *other

    def log_vector(log_dict, split, type, values, n_steps=1):
        log_dict[f'{split}/{type}/active'] = values[0].item() / n_steps
        log_dict[f'{split}/{type}/mcs'] = values[1].item() / n_steps
        log_dict[f'{split}/{type}/rssi'] = values[2].item() / n_steps
        log_dict[f'{split}/{type}/selected'] = values[3].item() / n_steps
        log_dict[f'{split}/{type}/success'] = values[4].item() / n_steps
        log_dict[f'{split}/{type}/tx_power'] = values[5].item() / n_steps
        log_dict[f'{split}/{type}/type'] = values[6].item() / n_steps

    key = jax.random.PRNGKey(cfg.train.seed)
    init_key, train_key, val_key, data_key, val_data_key = jax.random.split(key, 5)

    data = load(cfg.train.batch_size, data_key)
    val_data = iter(load(cfg.train.batch_size, val_data_key, val_dataset=True))
    sample = next(iter(data))

    model = GnnAutoencoder(**cfg.model)
    params = model.init({'params': init_key, 'lai': init_key}, sample)
    print(model.tabulate(jax.random.PRNGKey(0), sample))

    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.optimizer.grad_clip),
        optax.adamw(cfg.optimizer.learning_rate, weight_decay=cfg.optimizer.weight_decay, b1=cfg.optimizer.b1, b2=cfg.optimizer.b2)
    )
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
        params, opt_state, loss, (kl_loss, other_losses, na_metrics, pred_metrics) = step_fn(params, opt_state, subkey, batch)

        log_dict['step'] = step
        log_dict['train/loss'] = loss.item()
        log_dict['train/kl_loss'] = kl_loss.item()
        log_vector(log_dict, 'train', 'loss', other_losses)
        log_vector(log_dict, 'train', 'na_acc', na_metrics)
        log_vector(log_dict, 'train', 'pred_acc', pred_metrics)

        if step % cfg.train.val.freq == 0 or step == cfg.train.n_steps - 1:
            val_loss, val_kl_loss = 0.0, 0.0
            val_other_losses, val_na_metrics, val_pred_metrics = np.zeros(7), np.zeros(7), np.zeros(7)

            for _ in range(cfg.train.val.n_steps):
                val_key, subkey = jax.random.split(val_key)
                val_batch = next(val_data)
                loss, (kl_loss, other_losses, na_metrics, pred_metrics) = loss_fn(params, subkey, val_batch)
                val_loss += loss.item()
                val_kl_loss += kl_loss.item()
                val_other_losses += other_losses
                val_na_metrics += na_metrics
                val_pred_metrics += pred_metrics

            log_dict['val/loss'] = val_loss / cfg.train.val.n_steps
            log_dict['val/kl_loss'] = val_kl_loss / cfg.train.val.n_steps
            log_vector(log_dict, 'val', 'loss', val_other_losses, cfg.train.val.n_steps)
            log_vector(log_dict, 'val', 'na_acc', val_na_metrics, cfg.train.val.n_steps)
            log_vector(log_dict, 'val', 'pred_acc', val_pred_metrics, cfg.train.val.n_steps)

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
