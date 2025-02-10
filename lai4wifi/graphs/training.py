import logging
import os

import hydra
import jax
import jraph
import optax
import orbax.checkpoint as ocp
import wandb
from omegaconf import OmegaConf

from lai4wifi.graphs.model import GnnAutoencoder


def load(batch_size, n_node, n_edge):
    import networkx as nx
    from lai4wifi.graphs.attributes import IsActive, McsValue, IsSelected, TxPowerValue, EdgeType
    from lai4wifi.graphs.utils import nx_to_jraph, make_batch

    G = nx.Graph([
        (0, 1, {'active': IsActive.TRUE, 'mcs': McsValue.NAN, 'rate': 0., 'rssi': 0., 'selected': IsSelected.FALSE, 'tx_power': TxPowerValue.NAN, 'type': EdgeType.STA}),
        (0, 2, {'active': IsActive.TRUE, 'mcs': McsValue.NAN, 'rate': 0., 'rssi': 0., 'selected': IsSelected.FALSE, 'tx_power': TxPowerValue.NAN, 'type': EdgeType.STA}),
        (0, 3, {'active': IsActive.FALSE, 'mcs': McsValue.NAN, 'rate': 0., 'rssi': 0., 'selected': IsSelected.FALSE, 'tx_power': TxPowerValue.NAN, 'type': EdgeType.STA}),
        (0, 4, {'active': IsActive.TRUE, 'mcs': McsValue.NAN, 'rate': 0., 'rssi': 0., 'selected': IsSelected.FALSE, 'tx_power': TxPowerValue.NAN, 'type': EdgeType.STA}),
        (5, 6, {'active': IsActive.TRUE, 'mcs': McsValue.NAN, 'rate': 0., 'rssi': 0., 'selected': IsSelected.FALSE, 'tx_power': TxPowerValue.NAN, 'type': EdgeType.STA}),
        (5, 7, {'active': IsActive.TRUE, 'mcs': McsValue.NAN, 'rate': 0., 'rssi': 0., 'selected': IsSelected.TRUE, 'tx_power': TxPowerValue.NAN, 'type': EdgeType.STA}),
        (5, 8, {'active': IsActive.TRUE, 'mcs': McsValue.NAN, 'rate': 0., 'rssi': 0., 'selected': IsSelected.FALSE, 'tx_power': TxPowerValue.NAN, 'type': EdgeType.STA}),
        (5, 9, {'active': IsActive.FALSE, 'mcs': McsValue.NAN, 'rate': 0., 'rssi': 0., 'selected': IsSelected.FALSE, 'tx_power': TxPowerValue.NAN, 'type': EdgeType.STA}),
        (10, 11, {'active': IsActive.TRUE, 'mcs': McsValue.NAN, 'rate': 0., 'rssi': 0., 'selected': IsSelected.FALSE, 'tx_power': TxPowerValue.NAN, 'type': EdgeType.STA}),
        (10, 12, {'active': IsActive.TRUE, 'mcs': McsValue.NAN, 'rate': 0., 'rssi': 0., 'selected': IsSelected.FALSE, 'tx_power': TxPowerValue.NAN, 'type': EdgeType.STA}),
        (10, 13, {'active': IsActive.TRUE, 'mcs': McsValue.NAN, 'rate': 0., 'rssi': 0., 'selected': IsSelected.FALSE, 'tx_power': TxPowerValue.NAN, 'type': EdgeType.STA}),
        (10, 14, {'active': IsActive.TRUE, 'mcs': McsValue.NAN, 'rate': 0., 'rssi': 0., 'selected': IsSelected.FALSE, 'tx_power': TxPowerValue.NAN, 'type': EdgeType.STA}),
        (15, 16, {'active': IsActive.FALSE, 'mcs': McsValue.NAN, 'rate': 0., 'rssi': 0., 'selected': IsSelected.FALSE, 'tx_power': TxPowerValue.NAN, 'type': EdgeType.STA}),
        (15, 17, {'active': IsActive.FALSE, 'mcs': McsValue.NAN, 'rate': 0., 'rssi': 0., 'selected': IsSelected.FALSE, 'tx_power': TxPowerValue.NAN, 'type': EdgeType.STA}),
        (15, 18, {'active': IsActive.TRUE, 'mcs': McsValue.NAN, 'rate': 0., 'rssi': 0., 'selected': IsSelected.FALSE, 'tx_power': TxPowerValue.NAN, 'type': EdgeType.STA}),
        (15, 19, {'active': IsActive.TRUE, 'mcs': McsValue.NAN, 'rate': 0., 'rssi': 0., 'selected': IsSelected.FALSE, 'tx_power': TxPowerValue.NAN, 'type': EdgeType.STA}),
        (0, 5, {'active': IsActive.NAN, 'mcs': McsValue.NAN, 'rate': 0., 'rssi': -50.0, 'selected': IsSelected.NAN, 'tx_power': TxPowerValue.NAN, 'type': EdgeType.AP}),
        (0, 10, {'active': IsActive.NAN, 'mcs': McsValue.NAN, 'rate': 0., 'rssi': -55.0, 'selected': IsSelected.NAN, 'tx_power': TxPowerValue.NAN, 'type': EdgeType.AP}),
        (0, 15, {'active': IsActive.NAN, 'mcs': McsValue.NAN, 'rate': 0., 'rssi': -60.0, 'selected': IsSelected.NAN, 'tx_power': TxPowerValue.NAN, 'type': EdgeType.AP}),
        (5, 10, {'active': IsActive.NAN, 'mcs': McsValue.NAN, 'rate': 0., 'rssi': -53.0, 'selected': IsSelected.NAN, 'tx_power': TxPowerValue.NAN, 'type': EdgeType.AP}),
        (5, 15, {'active': IsActive.NAN, 'mcs': McsValue.NAN, 'rate': 0., 'rssi': -57.0, 'selected': IsSelected.NAN, 'tx_power': TxPowerValue.NAN, 'type': EdgeType.AP}),
        (10, 15, {'active': IsActive.NAN, 'mcs': McsValue.NAN, 'rate': 0., 'rssi': -51.0, 'selected': IsSelected.NAN, 'tx_power': TxPowerValue.NAN, 'type': EdgeType.AP})
    ])

    example_graph = nx_to_jraph(G, n_node=n_node, n_edge=n_edge)
    batch = [example_graph] * batch_size
    batch = make_batch(batch)

    while True:
        yield batch


@hydra.main(config_path="configs", config_name="gnn", version_base=None)
def main(cfg):

    @jax.jit
    def loss_fn(params, key, batch):
        pred = model.apply(params, batch)

        @jax.vmap
        def vmap_loss_fn(batch, pred):
            mask = jraph.get_node_padding_mask(pred)[..., None]
            active_loss = optax.sigmoid_binary_cross_entropy(pred.nodes.active, batch.nodes.active).mean(axis=-1)
            mcs_loss = optax.sigmoid_binary_cross_entropy(pred.nodes.mcs, batch.nodes.mcs).mean(axis=-1)
            rate_loss = optax.squared_error(pred.nodes.rate, batch.nodes.rate).mean(axis=-1)
            rssi_loss = optax.squared_error(pred.nodes.rssi, batch.nodes.rssi).mean(axis=-1)
            selected_loss = optax.sigmoid_binary_cross_entropy(pred.nodes.selected, batch.nodes.selected).mean(axis=-1)
            tx_power_loss = optax.sigmoid_binary_cross_entropy(pred.nodes.tx_power, batch.nodes.tx_power).mean(axis=-1)
            type_loss = optax.sigmoid_binary_cross_entropy(pred.nodes.type, batch.nodes.type).mean(axis=-1)
            loss = mask * (active_loss + mcs_loss + rate_loss + rssi_loss + selected_loss + tx_power_loss + type_loss) / 7
            return loss

        return vmap_loss_fn(batch, pred).mean()

    @jax.jit
    def step_fn(params, opt_state, key, batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, key, batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    key = jax.random.PRNGKey(cfg.train.seed)
    init_key, train_key, val_key = jax.random.split(key, 3)

    data = load(cfg.train.batch_size, cfg.train.pad_n_node, cfg.train.pad_n_edge)
    val_data = iter(load(cfg.train.batch_size, cfg.train.pad_n_node, cfg.train.pad_n_edge))
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
