import os
os.environ['JAX_COMPILATION_CACHE_DIR'] = '/tmp/jax_cache'
os.environ['JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES'] = '-1'
os.environ['JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS'] = '0'

import jax
import jax.numpy as jnp
import networkx as nx
import matplotlib.pyplot as plt
from mapc_sim.sim import network_data_rate
from evojax.algo import CMA_ES_JAX
from tqdm import trange

from lai4wifi.graphs.attributes import IsActive, McsValue, IsSelected, TxPowerValue, EdgeType
from lai4wifi.graphs.model import GnnAutoencoder
from lai4wifi.graphs.utils import nx_to_jraph


def load_example():
    G = nx.Graph([
        (0, 4, {'active': IsActive.TRUE, 'mcs': McsValue.NA, 'rate': 0., 'rssi': 0., 'selected': IsSelected.FALSE, 'tx_power': TxPowerValue.NA, 'type': EdgeType.STA}),
        (0, 5, {'active': IsActive.TRUE, 'mcs': McsValue.NA, 'rate': 0., 'rssi': 0., 'selected': IsSelected.FALSE, 'tx_power': TxPowerValue.NA, 'type': EdgeType.STA}),
        (0, 6, {'active': IsActive.FALSE, 'mcs': McsValue.NA, 'rate': 0., 'rssi': 0., 'selected': IsSelected.FALSE, 'tx_power': TxPowerValue.NA, 'type': EdgeType.STA}),
        (0, 7, {'active': IsActive.TRUE, 'mcs': McsValue.NA, 'rate': 0., 'rssi': 0., 'selected': IsSelected.FALSE, 'tx_power': TxPowerValue.NA, 'type': EdgeType.STA}),
        (1, 8, {'active': IsActive.TRUE, 'mcs': McsValue.NA, 'rate': 0., 'rssi': 0., 'selected': IsSelected.FALSE, 'tx_power': TxPowerValue.NA, 'type': EdgeType.STA}),
        (1, 9, {'active': IsActive.TRUE, 'mcs': McsValue.NA, 'rate': 0., 'rssi': 0., 'selected': IsSelected.TRUE, 'tx_power': TxPowerValue.NA, 'type': EdgeType.STA}),
        (1, 10, {'active': IsActive.TRUE, 'mcs': McsValue.NA, 'rate': 0., 'rssi': 0., 'selected': IsSelected.FALSE, 'tx_power': TxPowerValue.NA, 'type': EdgeType.STA}),
        (1, 11, {'active': IsActive.FALSE, 'mcs': McsValue.NA, 'rate': 0., 'rssi': 0., 'selected': IsSelected.FALSE, 'tx_power': TxPowerValue.NA, 'type': EdgeType.STA}),
        (2, 12, {'active': IsActive.TRUE, 'mcs': McsValue.NA, 'rate': 0., 'rssi': 0., 'selected': IsSelected.FALSE, 'tx_power': TxPowerValue.NA, 'type': EdgeType.STA}),
        (2, 13, {'active': IsActive.TRUE, 'mcs': McsValue.NA, 'rate': 0., 'rssi': 0., 'selected': IsSelected.FALSE, 'tx_power': TxPowerValue.NA, 'type': EdgeType.STA}),
        (2, 14, {'active': IsActive.TRUE, 'mcs': McsValue.NA, 'rate': 0., 'rssi': 0., 'selected': IsSelected.FALSE, 'tx_power': TxPowerValue.NA, 'type': EdgeType.STA}),
        (2, 15, {'active': IsActive.TRUE, 'mcs': McsValue.NA, 'rate': 0., 'rssi': 0., 'selected': IsSelected.FALSE, 'tx_power': TxPowerValue.NA, 'type': EdgeType.STA}),
        (3, 16, {'active': IsActive.FALSE, 'mcs': McsValue.NA, 'rate': 0., 'rssi': 0., 'selected': IsSelected.FALSE, 'tx_power': TxPowerValue.NA, 'type': EdgeType.STA}),
        (3, 17, {'active': IsActive.FALSE, 'mcs': McsValue.NA, 'rate': 0., 'rssi': 0., 'selected': IsSelected.FALSE, 'tx_power': TxPowerValue.NA, 'type': EdgeType.STA}),
        (3, 18, {'active': IsActive.TRUE, 'mcs': McsValue.NA, 'rate': 0., 'rssi': 0., 'selected': IsSelected.FALSE, 'tx_power': TxPowerValue.NA, 'type': EdgeType.STA}),
        (3, 19, {'active': IsActive.TRUE, 'mcs': McsValue.NA, 'rate': 0., 'rssi': 0., 'selected': IsSelected.FALSE, 'tx_power': TxPowerValue.NA, 'type': EdgeType.STA}),
        (0, 1, {'active': IsActive.NA, 'mcs': McsValue.NA, 'rate': 0., 'rssi': -50.0, 'selected': IsSelected.NA, 'tx_power': TxPowerValue.NA, 'type': EdgeType.AP}),
        (0, 2, {'active': IsActive.NA, 'mcs': McsValue.NA, 'rate': 0., 'rssi': -55.0, 'selected': IsSelected.NA, 'tx_power': TxPowerValue.NA, 'type': EdgeType.AP}),
        (0, 3, {'active': IsActive.NA, 'mcs': McsValue.NA, 'rate': 0., 'rssi': -60.0, 'selected': IsSelected.NA, 'tx_power': TxPowerValue.NA, 'type': EdgeType.AP}),
        (1, 2, {'active': IsActive.NA, 'mcs': McsValue.NA, 'rate': 0., 'rssi': -53.0, 'selected': IsSelected.NA, 'tx_power': TxPowerValue.NA, 'type': EdgeType.AP}),
        (1, 3, {'active': IsActive.NA, 'mcs': McsValue.NA, 'rate': 0., 'rssi': -57.0, 'selected': IsSelected.NA, 'tx_power': TxPowerValue.NA, 'type': EdgeType.AP}),
        (2, 3, {'active': IsActive.NA, 'mcs': McsValue.NA, 'rate': 0., 'rssi': -51.0, 'selected': IsSelected.NA, 'tx_power': TxPowerValue.NA, 'type': EdgeType.AP})
    ])

    d_ap, d_sta = 20.0, 2.0
    ap_pos = [
        [0 * d_ap, 0 * d_ap],
        [1 * d_ap, 0 * d_ap],
        [1 * d_ap, 1 * d_ap],
        [0 * d_ap, 1 * d_ap],
    ]
    dx = jnp.array([-1, 1, 1, -1]) * d_sta / jnp.sqrt(2)
    dy = jnp.array([-1, -1, 1, 1]) * d_sta / jnp.sqrt(2)
    sta_pos = [[x + dx[i], y + dy[i]] for x, y in ap_pos for i in range(len(dx))]
    pos = jnp.array(ap_pos + sta_pos)

    Gl = nx.line_graph(G)
    nx.set_node_attributes(Gl, dict(map(lambda e: (e[0:2], e[2]), G.edges(data=True))))

    idx_to_edge = jnp.array([(u, v) for (u, v), d in Gl.nodes(data=True) if d['type'] == EdgeType.STA])
    idx_to_ap = jnp.array([u for (u, v), d in Gl.nodes(data=True) if d['type'] == EdgeType.STA])
    selected_idx = next(i for i, ((_, _), d) in enumerate(Gl.nodes(data=True)) if d['selected'] == IsSelected.TRUE)
    n_nodes, n_conns = len(G.nodes), len(Gl.nodes)
    n_stas = len(idx_to_edge)
    aps = jnp.unique(idx_to_ap)

    return nx_to_jraph(G, pad=False), idx_to_edge, idx_to_ap, selected_idx, n_nodes, n_stas, n_conns, pos, aps


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, emb_key = jax.random.split(key)
    graph, idx_to_edge, idx_to_ap, selected_idx, n_nodes, n_stas, n_conns, pos, aps = load_example()

    model = GnnAutoencoder(latent_size=32, encoder_features=(), decoder_features=(), dtype=jnp.float32)
    params = model.init({'params': init_key}, graph)

    noise_std = 0.1
    emb = model.apply(params, graph, method='encode').nodes
    emb_noise = emb + noise_std * jax.random.normal(emb_key, emb.shape)

    solver = CMA_ES_JAX(
        param_size=emb_noise.size,
        pop_size=2048,
        mean=emb_noise.reshape(-1),
        init_stdev=10.0
    )

    @jax.jit
    @jax.vmap
    def fitness_fn(z):
        z = graph._replace(nodes=z.reshape(n_conns, -1))
        x = model.apply(params, z, method='decode')

        key = jax.random.key(0)
        tx = jnp.zeros((n_nodes, n_nodes), dtype=int)
        walls = jnp.zeros_like(tx)
        mcs = jnp.zeros(n_nodes, dtype=int)
        tx_power = jnp.zeros_like(mcs)

        def scan_fn(carry, i):
            tx, mcs, tx_power = carry
            tx = tx.at[*idx_to_edge[i]].set(jnp.argmax(x.nodes.selected[i]) == IsSelected.TRUE.value)
            mcs = mcs.at[idx_to_ap[i]].set(jnp.argmax(x.nodes.mcs[i]))
            tx_power = tx_power.at[idx_to_ap[i]].set(jnp.argmax(x.nodes.tx_power[i]))
            return (tx, mcs, tx_power), None

        (tx, mcs, tx_power), _ = jax.lax.scan(scan_fn, (tx, mcs, tx_power), jnp.arange(n_stas))
        data_rate = network_data_rate(key, tx, pos, mcs, tx_power, sigma=0., walls=walls, return_sample=False)

        multiple_tx_per_ap = jnp.any(tx[aps].sum(axis=1) > 1)
        designated_sta_not_selected = jnp.argmax(x.nodes.selected[selected_idx]) == IsSelected.FALSE.value

        return jax.lax.select(jax.lax.bitwise_or(multiple_tx_per_ap, designated_sta_not_selected), -1., data_rate)

    history = []

    for _ in (pbar := trange(500)):
        solutions = solver.ask()
        fitness = fitness_fn(solutions)
        solver.tell(fitness)

        best_trial = jnp.array([solver.best_params])
        best_val = fitness_fn(best_trial).item()
        history.append(best_val)
        pbar.set_postfix({'Fitness': best_val})

    plt.plot(history)
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.ylim(bottom=0)
    plt.grid()
    plt.tight_layout()
    plt.show()
