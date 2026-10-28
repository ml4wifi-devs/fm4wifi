import os
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_ENABLE_X64'] = 'True'

import json
import logging

import hydra
import jax
import jax.numpy as jnp
import jraph
import optax
import orbax.checkpoint as ocp
import wandb
from omegaconf import OmegaConf

from lai4wifi.dataset.utils import load_dataset
from lai4wifi.surrogate.model import SurrogateModel
from lai4wifi.surrogate.utils import evaluate_model, mdn_loss


@hydra.main(config_path='configs', config_name='surrogate', version_base=None)
def main(cfg):
    @jax.jit
    def mse_loss_fn(params, key, batch, rates):
        pred = model.apply(params, batch, rngs=key)
        mask = jraph.get_graph_padding_mask(batch)
        loss = optax.squared_error(pred, rates)
        loss = (mask * loss).sum() / mask.sum()
        return loss, pred

    @jax.jit
    def mdn_loss_fn(params, key, batch, rates):
        logits, means, scales = model.apply(params, batch, rngs=key)
        probs = jax.nn.softmax(logits, axis=-1)
        expected_value = jnp.sum(probs * means, axis=-1)

        mask = jraph.get_graph_padding_mask(batch)
        loss = mdn_loss(logits, means, scales, rates)
        loss = (mask * loss).sum() / mask.sum()
        return loss, expected_value

    loss_fn = mse_loss_fn if cfg.model.mse else mdn_loss_fn

    @jax.jit
    def step_fn(params, opt_state, key, batch, rates):
        (loss, pred), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, key, batch, rates)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, pred

    key = jax.random.PRNGKey(cfg.train.seed)
    init_key, train_key, val_key = jax.random.split(key, 3)

    with jax.default_device(jax.devices('cpu')[0]):
        data = load_dataset(cfg.train_dataset_path)
        val_data = load_dataset(cfg.val_dataset_path)

    val_idx = 0
    sample = data[0][0]

    model = SurrogateModel(**cfg.model)
    params = model.init({'params': init_key}, sample)
    print(model.tabulate(jax.random.PRNGKey(0), sample))

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
        item_names=['params', 'opt_state', 'train_key', 'val_key']
    )
    OmegaConf.save(config=cfg, f=os.path.join(ckpt_path, 'config.yaml'))

    if cfg.train.log.type == 'wandb':
        wandb.init(project='lai4wifi', name=cfg.name)

    for step, (batch, rates) in zip(range(cfg.train.n_steps), data):
        log_dict = {'step': step}

        train_key, subkey = jax.random.split(train_key)
        params, opt_state, loss, pred = step_fn(params, opt_state, subkey, batch, rates)

        if step % cfg.train.val.freq == 0 or step == cfg.train.n_steps - 1:
            val_loss = 0.0

            for _ in range(cfg.train.val.n_steps):
                val_key, subkey = jax.random.split(val_key)
                val_batch, val_rates = val_data[val_idx]
                val_idx = (val_idx + 1) % len(val_data)
                val_loss += loss_fn(params, subkey, val_batch, val_rates)[0]

            log_dict['val/loss'] = val_loss / cfg.train.val.n_steps

        if step % cfg.train.log.freq == 0 or step == cfg.train.n_steps - 1:
            log_dict['train/loss'] = loss
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

    metrics = evaluate_model(model, params, val_data)
    logging.info(metrics)

    with open(os.path.join(ckpt_path, 'eval_metrics.json'), 'w') as file:
        json.dump(metrics, file, indent=2)

    if cfg.train.log.type == 'wandb':
        wandb.log({f'eval/{k}': v for k, v in metrics.items()})


if __name__ == '__main__':
    main()
