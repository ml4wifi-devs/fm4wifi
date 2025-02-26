import logging
import os

import hydra
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import wandb
from omegaconf import OmegaConf

from lai4wifi.flow_matching.model import FlowMatching
from lai4wifi.graphs.dataset import load_dataset


def get_network_desc(target_conf, context):
    # TODO: Implement network description
    network_desc = jnp.zeros(target_conf.embedding.nodes.shape)
    return target_conf.embedding.nodes, network_desc


def draw_configurations(scenario, n_timesteps, key):
    *context_idxs, target_idx = jax.random.choice(key, len(scenario), (n_timesteps + 1,))
    context = [scenario[i].embedding.nodes for i in context_idxs]
    target_conf = scenario[target_idx]
    target_conf, network_desc = get_network_desc(target_conf, context)
    return jnp.stack([network_desc, *context, target_conf], axis=1)


def load(dataset_path, batch_size, n_timesteps, key):
    dataset = load_dataset(dataset_path)

    while True:
        key, idx_key, conf_key = jax.random.split(key, 3)
        idxs = jax.random.choice(idx_key, len(dataset), (batch_size,))
        batch = [draw_configurations(dataset[i], n_timesteps, conf_key) for i in idxs]
        yield jnp.vstack(batch)


@hydra.main(config_path="configs", config_name="debug", version_base=None)
def main(cfg):

    @jax.jit
    def loss_fn(params, key, cx, schedule='uniform'):
        z_key, t_key, model_key = jax.random.split(key, 3)

        if schedule == 'log-normal':
            t = jax.random.normal(t_key, (cx.shape[0],))
            t = 1 / (1 + jnp.exp(-t))
        elif schedule == 'uniform':
            t = jax.random.uniform(t_key, (cx.shape[0],), minval=0.0, maxval=0.99)
        else:
            raise ValueError(f'Invalid schedule: {schedule}')

        z = jax.random.normal(z_key, cx.shape[:2] + (cfg.model.latent_size,))
        t_b = t.reshape((-1,) + (1,) * (len(z.shape) - 1))

        x = cx[..., -1, :]
        x_t = (1 - t_b) * z + t_b * x
        v_t = x - z
        cx = cx.at[..., -1, :].set(x_t)

        v_pred = model.apply(params, cx, t, rngs=model_key)
        return optax.squared_error(v_pred, v_t).mean()

    @jax.jit
    def step_fn(params, opt_state, key, batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, key, batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def generate_fn(params, key, batch):
        return model.apply(params, batch, rngs=key, method='gen')

    key = jax.random.PRNGKey(cfg.train.seed)
    init_key, train_key, val_key, data_key, val_data_key = jax.random.split(key, 5)

    data = load(cfg.dataset, cfg.train.batch_size, cfg.train.n_timesteps, data_key)
    val_data = iter(load(cfg.val_dataset, cfg.train.batch_size, cfg.train.n_timesteps, val_data_key))
    sample = next(iter(data))
    sample = (sample, jnp.zeros(sample.shape[0]))

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
