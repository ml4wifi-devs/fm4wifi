import logging
import os

import hydra
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import wandb
from omegaconf import OmegaConf

from lai4wifi.flow_matching.model import FMUnet


def load():
    import tensorflow as tf
    import tensorflow_datasets as tfds
    ds = tfds.load(name="mnist", split="train", batch_size=128, as_supervised=False)
    ds = ds.map(lambda x: (tf.cast(x['image'], tf.float32) / 255, tf.zeros((tf.shape(x['image'])[0], 10))))
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    ds = ds.repeat()
    return tfds.as_numpy(ds)


@hydra.main(config_path="configs", config_name="fm", version_base=None)
def main(cfg):

    @jax.jit
    def loss_fn(params, key, x, cond, schedule='uniform'):
        z_key, t_key = jax.random.split(key)

        if schedule == 'log-normal':
            t = jax.random.normal(t_key, (x.shape[0],))
            t = 1 / (1 + jnp.exp(-t))
        elif schedule == 'uniform':
            t = jax.random.uniform(t_key, (x.shape[0],), minval=0.0, maxval=0.99)
        else:
            raise ValueError(f'Invalid schedule: {schedule}')

        z = jax.random.normal(z_key, x.shape)
        t_b = t.reshape((-1,) + (1,) * (len(x.shape) - 1))

        x_t = (1 - t_b) * z + t_b * x
        v_t = x - z

        v_pred = model.apply(params, x_t, cond, t)
        return optax.squared_error(v_pred, v_t).mean()

    @jax.jit
    def step_fn(params, opt_state, key, *batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, key, *batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def generate_fn(params, key, *batch):
        return model.apply(params, *batch, rngs={'lai': key}, method='gen')

    key = jax.random.PRNGKey(cfg.train.seed)
    init_key, train_key, val_key = jax.random.split(key, 3)

    data = load()
    val_data = iter(load())
    sample = next(iter(data))
    sample = sample + (jnp.zeros(sample[0].shape[0]),)

    model = FMUnet(**cfg.model)
    params = model.init({'params': init_key}, *sample)
    print(model.tabulate(jax.random.PRNGKey(0), *sample))

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
        params, opt_state, loss = step_fn(params, opt_state, subkey, *batch)

        log_dict['step'] = step
        log_dict['train/loss'] = loss.item()

        if step % cfg.train.val.freq == 0 or step == cfg.train.n_steps - 1:
            val_loss = 0.0

            for _ in range(cfg.train.val.n_steps):
                val_key, subkey = jax.random.split(val_key)
                val_batch = next(val_data)
                loss = loss_fn(params, subkey, *val_batch)
                val_loss += loss.item()

                # visualize samples, to be removed
                import matplotlib.pyplot as plt
                sample = generate_fn(params, subkey, jnp.zeros((1, 10), jnp.float32))
                sample = jnp.clip(sample, 0, 1)
                plt.imshow(sample[0], cmap='gray')
                plt.show()

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
