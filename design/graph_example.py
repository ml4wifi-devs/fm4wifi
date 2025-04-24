import jax
import networkx as nx

from lai4wifi.graphs.attributes import *
from lai4wifi.graphs.model import GnnAutoencoder
from lai4wifi.graphs.utils import nx_to_jraph, make_batch


G = nx.Graph([
    (0, 1, {'active': IsActive.TRUE, 'mcs': McsValue.NA, 'rssi': 0., 'selected': IsSelected.FALSE, 'success': 0.0, 'tx_power': TxPowerValue.NA, 'type': EdgeType.STA}),
    (0, 2, {'active': IsActive.TRUE, 'mcs': McsValue.NA, 'rssi': 0., 'selected': IsSelected.FALSE, 'success': 0.0, 'tx_power': TxPowerValue.NA, 'type': EdgeType.STA}),
    (0, 3, {'active': IsActive.FALSE, 'mcs': McsValue.NA, 'rssi': 0., 'selected': IsSelected.FALSE, 'success': 0.0, 'tx_power': TxPowerValue.NA, 'type': EdgeType.STA}),
    (0, 4, {'active': IsActive.TRUE, 'mcs': McsValue.NA, 'rssi': 0., 'selected': IsSelected.FALSE, 'success': 0.0, 'tx_power': TxPowerValue.NA, 'type': EdgeType.STA}),
    (5, 6, {'active': IsActive.TRUE, 'mcs': McsValue.NA, 'rssi': 0., 'selected': IsSelected.FALSE, 'success': 0.0, 'tx_power': TxPowerValue.NA, 'type': EdgeType.STA}),
    (5, 7, {'active': IsActive.TRUE, 'mcs': McsValue.NA, 'rssi': 0., 'selected': IsSelected.TRUE, 'success': 0.0, 'tx_power': TxPowerValue.NA, 'type': EdgeType.STA}),
    (5, 8, {'active': IsActive.TRUE, 'mcs': McsValue.NA, 'rssi': 0., 'selected': IsSelected.FALSE, 'success': 0.0, 'tx_power': TxPowerValue.NA, 'type': EdgeType.STA}),
    (5, 9, {'active': IsActive.FALSE, 'mcs': McsValue.NA, 'rssi': 0., 'selected': IsSelected.FALSE, 'success': 0.0, 'tx_power': TxPowerValue.NA, 'type': EdgeType.STA}),
    (10, 11, {'active': IsActive.TRUE, 'mcs': McsValue.NA, 'rssi': 0., 'selected': IsSelected.FALSE, 'success': 0.0, 'tx_power': TxPowerValue.NA, 'type': EdgeType.STA}),
    (10, 12, {'active': IsActive.TRUE, 'mcs': McsValue.NA, 'rssi': 0., 'selected': IsSelected.FALSE, 'success': 0.0, 'tx_power': TxPowerValue.NA, 'type': EdgeType.STA}),
    (10, 13, {'active': IsActive.TRUE, 'mcs': McsValue.NA, 'rssi': 0., 'selected': IsSelected.FALSE, 'success': 0.0, 'tx_power': TxPowerValue.NA, 'type': EdgeType.STA}),
    (10, 14, {'active': IsActive.TRUE, 'mcs': McsValue.NA, 'rssi': 0., 'selected': IsSelected.FALSE, 'success': 0.0, 'tx_power': TxPowerValue.NA, 'type': EdgeType.STA}),
    (15, 16, {'active': IsActive.FALSE, 'mcs': McsValue.NA, 'rssi': 0., 'selected': IsSelected.FALSE, 'success': 0.0, 'tx_power': TxPowerValue.NA, 'type': EdgeType.STA}),
    (15, 17, {'active': IsActive.FALSE, 'mcs': McsValue.NA, 'rssi': 0., 'selected': IsSelected.FALSE, 'success': 0.0, 'tx_power': TxPowerValue.NA, 'type': EdgeType.STA}),
    (15, 18, {'active': IsActive.TRUE, 'mcs': McsValue.NA, 'rssi': 0., 'selected': IsSelected.FALSE, 'success': 0.0, 'tx_power': TxPowerValue.NA, 'type': EdgeType.STA}),
    (15, 19, {'active': IsActive.TRUE, 'mcs': McsValue.NA, 'rssi': 0., 'selected': IsSelected.FALSE, 'success': 0.0, 'tx_power': TxPowerValue.NA, 'type': EdgeType.STA}),
    (0, 5, {'active': IsActive.NA, 'mcs': McsValue.NA, 'rssi': -50.0, 'selected': IsSelected.NA, 'success': 0.0, 'tx_power': TxPowerValue.NA, 'type': EdgeType.AP}),
    (0, 10, {'active': IsActive.NA, 'mcs': McsValue.NA, 'rssi': -55.0, 'selected': IsSelected.NA, 'success': 0.0, 'tx_power': TxPowerValue.NA, 'type': EdgeType.AP}),
    (0, 15, {'active': IsActive.NA, 'mcs': McsValue.NA, 'rssi': -60.0, 'selected': IsSelected.NA, 'success': 0.0, 'tx_power': TxPowerValue.NA, 'type': EdgeType.AP}),
    (5, 10, {'active': IsActive.NA, 'mcs': McsValue.NA, 'rssi': -53.0, 'selected': IsSelected.NA, 'success': 0.0, 'tx_power': TxPowerValue.NA, 'type': EdgeType.AP}),
    (5, 15, {'active': IsActive.NA, 'mcs': McsValue.NA, 'rssi': -57.0, 'selected': IsSelected.NA, 'success': 0.0, 'tx_power': TxPowerValue.NA, 'type': EdgeType.AP}),
    (10, 15, {'active': IsActive.NA, 'mcs': McsValue.NA, 'rssi': -51.0, 'selected': IsSelected.NA, 'success': 0.0, 'tx_power': TxPowerValue.NA, 'type': EdgeType.AP})
])


if __name__ == '__main__':
    example_graph = nx_to_jraph(G)
    batch = [example_graph] * 16
    batch = make_batch(batch)

    model = GnnAutoencoder(embed_features=32, encoder_features=(), decoder_features=(), latent_features=32, n_layers=0, dtype=jnp.float32)
    params = model.init(jax.random.key(0), batch)
    ug = model.apply(params, batch, rngs={'lai': jax.random.key(42)})

    print(jax.tree.map(jnp.shape, batch))
    print(jax.tree.map(jnp.shape, params))
    print(jax.tree.map(jnp.shape, ug))

    ...
