import os
os.environ['XLA_FLAGS'] = '--xla_gpu_enable_triton_gemm=false'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['JAX_ENABLE_X64'] = 'True'

import logging
from functools import partial
from glob import glob

import hydra
import jax
import numpy as np
import jraph
import optax
import orbax.checkpoint as ocp
import wandb
from omegaconf import OmegaConf

from lai4wifi.flow_matching.model import SurrogateModel
from lai4wifi.flow_matching.utils import tx_to_conf
from lai4wifi.graphs.dataset import configuration_to_tx, load_dataset
from lai4wifi.graphs.model import GnnAutoencoder
from lai4wifi.graphs.utils import clear_graph, make_batch


RATE_MEAN = 200
RATE_STD = 200


def load(dataset_path, batch_size, encode_fn, key, val):
    dataset_paths = glob(f'{dataset_path}/*.pkl.lz4')
    dataset_paths = filter(lambda p: ('val' in p) == val, dataset_paths)
    dataset = sum((load_dataset(path) for path in dataset_paths), start=[])

    while True:
        graph_batch, rate_batch = [], []

        while len(graph_batch) < batch_size:
            key, dataset_key, config_key, scenario_key = jax.random.split(key, 4)
            dataset_idx = jax.random.choice(dataset_key, len(dataset))
            dataset_item = dataset[dataset_idx]
            
            if len(dataset_item.configurations) == 0:
                continue

            config_idx = jax.random.choice(config_key, len(dataset_item.configurations))
            configuration = dataset_item.configurations[config_idx]

            tx = configuration_to_tx(dataset_item.scenario, configuration)
            data_rate, _, internals = dataset_item.scenario(scenario_key, *tx, return_internals=True)
            data_rate = (data_rate - RATE_MEAN) / RATE_STD

            graph = tx_to_conf(dataset_item.scenario, *tx, internals, encode_fn)
            graph = clear_graph(graph)
            
            graph_batch.append(graph)
            rate_batch.append(data_rate)

        yield make_batch(graph_batch), np.asarray(rate_batch + [0.0])


@hydra.main(config_path='configs', config_name='surrogate', version_base=None)
def main(cfg):

    @jax.jit
    def loss_fn(params, key, batch, rates):
        pred = model.apply(params, batch, rngs=key)
        mask = jraph.get_graph_padding_mask(batch)
        loss = optax.squared_error(pred, rates)
        loss = (mask * loss).sum() / mask.sum()
        return loss, pred

    @jax.jit
    def step_fn(params, opt_state, key, batch, rates):
        (loss, pred), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, key, batch, rates)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, pred

    key = jax.random.PRNGKey(cfg.train.seed)
    init_key, train_key, val_key, data_key, val_data_key = jax.random.split(key, 5)

    gnn_cfg = OmegaConf.load(os.path.join(cfg.gnn_path, 'config.yaml'))
    with ocp.CheckpointManager(os.path.abspath(cfg.gnn_path), item_names=['params']) as ckpt_manager:
        gnn_params = ckpt_manager.restore(None, args=ocp.args.Composite(params=ocp.args.PyTreeRestore())).params

    gnn = GnnAutoencoder(**gnn_cfg.model)
    encode_fn = jax.jit(partial(gnn.apply, gnn_params, method='encode'))

    data = load(cfg.dataset_path, cfg.train.batch_size, encode_fn, data_key, val=False)
    val_data = iter(load(cfg.dataset_path, cfg.train.batch_size, encode_fn, val_data_key, val=True))
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
        log_dict = {}

        train_key, subkey = jax.random.split(train_key)
        params, opt_state, loss, pred = step_fn(params, opt_state, subkey, batch, rates)

        log_dict['step'] = step
        log_dict['train/loss'] = loss.item()

        if step % cfg.train.val.freq == 0 or step == cfg.train.n_steps - 1:
            val_loss = 0.0

            for _ in range(cfg.train.val.n_steps):
                val_key, subkey = jax.random.split(val_key)
                val_batch, val_rates = next(val_data)
                loss, pred = loss_fn(params, subkey, val_batch, val_rates)
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
