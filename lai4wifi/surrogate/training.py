import os
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_ENABLE_X64'] = 'True'

import logging
from dataclasses import replace

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import jraph
import optax
import orbax.checkpoint as ocp
import wandb
from omegaconf import OmegaConf

from lai4wifi.flow_matching.utils import create_data_loader
from lai4wifi.surrogate.model import SurrogateModel
from lai4wifi.surrogate.utils import RATE_MEAN, RATE_STD, calculate_data_rate
from lai4wifi.graphs.utils import make_batch


def filter_fn_factory(val):
    return lambda item: ('val' in item) == val


def map_fn(item):
    with jax.default_device(jax.devices('cpu')[0]):
        graph, _ = item
        rate = (calculate_data_rate(graph) - RATE_MEAN) / RATE_STD
        graph = graph._replace(edges=replace(graph.edges, success=jnp.zeros_like(graph.edges.success)))
        return graph, rate


def batch_fn(batch):
    graphs, rates = zip(*batch)
    return make_batch(graphs), np.asarray(list(rates) + [0.0])


@hydra.main(config_path='configs', config_name='surrogate', version_base=None)
def main(cfg):
    
    def quantile_loss(predictions, targets, tau=0.9):
        """aka pinball loss https://en.wikipedia.org/wiki/Quantile_regression"""
        errors = targets - predictions
        return jnp.where(errors < 0, (tau - 1) * errors, tau * errors)

    def soft_quantile_loss(predictions, targets, tau=0.9, b=4.0):
        """Soft approximation to quantile loss using squareplus function - this avoids transcendental functions"""
        errors = targets - predictions
        return ((2.0 * tau - 1.0) * errors + jnp.sqrt(errors ** 2 + b)) / 2.0

    @jax.jit
    def loss_fn(params, key, batch, rates):
        pred = model.apply(params, batch, rngs=key)
        mask = jraph.get_graph_padding_mask(batch)
        loss = optax.squared_error(pred, rates)
        loss = (mask * loss).sum() / mask.sum()
        return loss, pred

    def mdn_loss(logits, means, scales, rates):
        """Mixture Density Network https://research.aston.ac.uk/en/publications/mixture-density-networks"""
        log_probs_pi = jax.nn.log_softmax(logits, axis=-1)
        log_probs_gauss = -0.5 * jnp.log(2 * jnp.pi) - jnp.log(scales) - 0.5 * ((rates[..., None] - means) / scales) ** 2
        log_probs_total = log_probs_pi + log_probs_gauss
        return -jax.nn.logsumexp(log_probs_total, axis=-1)

    @jax.jit
    def mdn_loss_fn(params, key, batch, rates):
        logits, means, scales = model.apply(params, batch, rngs=key)
        probs = jax.nn.softmax(logits, axis=-1)
        expected_value = jnp.sum(probs * means, axis=-1)        
        
        mask = jraph.get_graph_padding_mask(batch)
        loss = mdn_loss(logits, means, scales, rates)
        loss = (mask * loss).sum() / mask.sum()
        return loss, expected_value

    @jax.jit
    def step_fn(params, opt_state, key, batch, rates):
        (loss, pred), grads = jax.value_and_grad(mdn_loss_fn, has_aux=True)(params, key, batch, rates)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, pred

    key = jax.random.PRNGKey(cfg.train.seed)
    init_key, train_key, val_key = jax.random.split(key, 3)

    data = create_data_loader(
        cfg.dataset_path, cfg.train.batch_size, cfg.train.seed, filter_fn=filter_fn_factory(val=False), map_fn=map_fn, batch_fn=batch_fn
    )
    val_data = iter(create_data_loader(
        cfg.dataset_path, cfg.train.batch_size, cfg.train.seed, filter_fn=filter_fn_factory(val=True), map_fn=map_fn, batch_fn=batch_fn
    ))
    sample = next(iter(data))[0]

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
                val_batch, val_rates = next(val_data)
                loss, pred = mdn_loss_fn(params, subkey, val_batch, val_rates)
                val_loss += loss

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


if __name__ == '__main__':
    main()
