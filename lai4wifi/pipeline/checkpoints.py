import os
from functools import partial

import jax
import orbax.checkpoint as ocp
from omegaconf import OmegaConf

from lai4wifi.flow_matching.model import FlowMatching
from lai4wifi.graphs.model import GnnAutoencoder
from lai4wifi.surrogate.model import SurrogateModel


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FM_PATH = os.path.join(ROOT, 'flow_matching', 'runs', 'fm_medium')
SURROGATE_PATH = os.path.join(ROOT, 'surrogate', 'runs', 'sur_medium')


def restore_params(path, item_name='params'):
    fallback_sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    restore = ocp.args.Composite(**{item_name: ocp.args.StandardRestore(fallback_sharding=fallback_sharding)})

    with ocp.CheckpointManager(os.path.abspath(path), item_names=[item_name]) as ckpt_manager:
        return getattr(ckpt_manager.restore(None, args=restore), item_name)


def load_gnn(path):
    cfg = OmegaConf.load(os.path.join(path, 'config.yaml'))
    gnn = GnnAutoencoder(**cfg.model)
    params = restore_params(path)

    encode_fn = jax.jit(partial(gnn.apply, params, method='encode'))
    decode_fn = jax.jit(partial(gnn.apply, params, method='decode'))
    return encode_fn, decode_fn


def load_fm_gnn(path=FM_PATH):
    cfg = OmegaConf.load(os.path.join(path, 'config.yaml'))
    return load_gnn(os.path.normpath(os.path.join(ROOT, 'flow_matching', cfg.gnn_path)))


def load_fm(path=FM_PATH):
    cfg = OmegaConf.load(os.path.join(path, 'config.yaml'))
    fm = FlowMatching(**cfg.model)
    gen_fn = jax.jit(partial(fm.apply, restore_params(path, 'ema_params'), method='gen'))
    return cfg, gen_fn, *load_fm_gnn(path)


def load_surrogate(path=SURROGATE_PATH):
    cfg = OmegaConf.load(os.path.join(path, 'config.yaml'))
    return SurrogateModel(**cfg.model), restore_params(path)
