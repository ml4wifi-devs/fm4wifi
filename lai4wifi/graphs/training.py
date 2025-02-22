import logging
import os
from itertools import chain

import hydra
import jax
import jax.numpy as jnp
import jraph
import optax
import orbax.checkpoint as ocp
import wandb
from omegaconf import OmegaConf

from lai4wifi.graphs.dataset import load_dataset
from lai4wifi.graphs.model import GnnAutoencoder
from lai4wifi.graphs.utils import make_batch


def load(dataset_path, batch_size, key):
    dataset = load_dataset(dataset_path)
    dataset = chain.from_iterable(dataset)
    dataset = list(map(lambda g: g.graph, dataset))

    while True:
        key, subkey = jax.random.split(key)
        idxs = jax.random.choice(key, len(dataset), (batch_size,), replace=False)
        yield make_batch([dataset[i] for i in idxs])


@hydra.main(config_path="configs", config_name="gnn", version_base=None)
def main(cfg):
    loss_weights = jnp.array(cfg.train.loss_weights)
    loss_weights /= loss_weights.sum()

    @jax.jit
    def loss_fn(params, key, batch):
        pred = model.apply(params, batch)

        def mixture_categorical_loss(logits, labels):
            na_logits, na_labels = logits[..., -1], labels[..., -1]
            softmax_logits, softmax_labels = logits[..., :-1], labels[..., :-1]
            na_loss = optax.sigmoid_binary_cross_entropy(na_logits, na_labels)
            softmax_loss = optax.softmax_cross_entropy(softmax_logits, softmax_labels)
            return na_loss + na_labels * softmax_loss

        @jax.vmap
        def vmap_loss_fn(batch, pred):
            mask = jraph.get_node_padding_mask(pred)
            active_loss   = mask * mixture_categorical_loss(pred.nodes.active, batch.nodes.active)
            mcs_loss      = mask * mixture_categorical_loss(pred.nodes.mcs, batch.nodes.mcs)
            rssi_loss     = mask * optax.squared_error(pred.nodes.rssi, batch.nodes.rssi).mean(axis=-1)
            selected_loss = mask * optax.sigmoid_binary_cross_entropy(pred.nodes.selected, batch.nodes.selected).mean(axis=-1)
            success_loss  = mask * mixture_categorical_loss(pred.nodes.success, batch.nodes.success)
            tx_power_loss = mask * mixture_categorical_loss(pred.nodes.tx_power, batch.nodes.tx_power)
            type_loss     = mask * mixture_categorical_loss(pred.nodes.type, batch.nodes.type)
            losses        = jnp.array([active_loss, mcs_loss, rssi_loss, selected_loss, success_loss, tx_power_loss, type_loss]).mean(axis=-1)
            return losses @ loss_weights, losses

        @jax.vmap
        def vmap_metrics_fn(batch, pred):
            mask = jraph.get_node_padding_mask(pred)
            active_acc   = mask * (pred.nodes.active.argmax(axis=-1) == batch.nodes.active.argmax(axis=-1))
            mcs_acc      = mask * (pred.nodes.mcs.argmax(axis=-1) == batch.nodes.mcs.argmax(axis=-1))
            rssi_acc     = mask * jnp.isclose(pred.nodes.rssi, batch.nodes.rssi, atol=0.1).mean(axis=-1)
            selected_acc = mask * jnp.isclose(pred.nodes.selected, batch.nodes.selected, atol=0.1).mean(axis=-1)
            success_acc  = mask * (pred.nodes.success.argmax(axis=-1) == batch.nodes.success.argmax(axis=-1))
            tx_power_acc = mask * (pred.nodes.tx_power.argmax(axis=-1) == batch.nodes.tx_power.argmax(axis=-1))
            type_acc     = mask * (pred.nodes.type.argmax(axis=-1) == batch.nodes.type.argmax(axis=-1))
            metrics      = jnp.array([active_acc, mcs_acc, rssi_acc, selected_acc, success_acc, tx_power_acc, type_acc]).mean(axis=-1)
            return metrics

        loss, other_losses = vmap_loss_fn(batch, pred)
        loss, other_losses = loss.mean(), other_losses.mean(axis=0)
        metrics = vmap_metrics_fn(batch, pred).mean(axis=0)
        return loss, (other_losses, metrics)

    @jax.jit
    def step_fn(params, opt_state, key, batch):
        (loss, *other), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, key, batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, *other

    def log_vector(log_dict, split, type, values, n_steps=1):
        log_dict[f'{split}/active_{type}'] = values[0].item() / n_steps
        log_dict[f'{split}/mcs_{type}'] = values[1].item() / n_steps
        log_dict[f'{split}/rssi_{type}'] = values[2].item() / n_steps
        log_dict[f'{split}/selected_{type}'] = values[3].item() / n_steps
        log_dict[f'{split}/success_{type}'] = values[4].item() / n_steps
        log_dict[f'{split}/tx_power_{type}'] = values[5].item() / n_steps
        log_dict[f'{split}/type_{type}'] = values[6].item() / n_steps

    key = jax.random.PRNGKey(cfg.train.seed)
    init_key, train_key, val_key, data_key, val_data_key = jax.random.split(key, 5)

    data = load(cfg.dataset, cfg.train.batch_size, data_key)
    val_data = iter(load(cfg.val_dataset, cfg.train.batch_size, val_data_key))
    sample = next(iter(data))

    model = GnnAutoencoder(**cfg.model)
    params = model.init({'params': init_key}, sample)
    print(model.tabulate(jax.random.PRNGKey(0), sample))

    optimizer = optax.adam(**cfg.optimizer)
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
        params, opt_state, loss, (other_losses, metrics) = step_fn(params, opt_state, subkey, batch)

        log_dict['step'] = step
        log_vector(log_dict, 'train', 'loss', other_losses)
        log_vector(log_dict, 'train', 'acc', metrics)

        if step % cfg.train.val.freq == 0 or step == cfg.train.n_steps - 1:
            val_loss, val_other_losses = 0.0, jnp.zeros(7)

            for _ in range(cfg.train.val.n_steps):
                val_key, subkey = jax.random.split(val_key)
                val_batch = next(val_data)
                loss, (other_losses, metrics) = loss_fn(params, subkey, val_batch)
                val_loss += loss.item()
                val_other_losses += other_losses

            log_dict['val/loss'] = val_loss / cfg.train.val.n_steps
            log_vector(log_dict, 'val', 'loss', val_other_losses, cfg.train.val.n_steps)
            log_vector(log_dict, 'val', 'acc', metrics, cfg.train.val.n_steps)

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
