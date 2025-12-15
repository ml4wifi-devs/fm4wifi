from functools import partial

import flax.linen as nn
import jax.numpy as jnp
import jraph
from jraph._src.utils import segment_mean

from lai4wifi.flow_matching.model import make_transformer
from lai4wifi.graphs.model import make_embed_fn
from lai4wifi.graphs.utils import clear_graph


class SurrogateModel(nn.Module):
    gnn_layers: int
    dim: int
    ff_dim: int
    n_layers: int
    n_heads: int
    drop_rate: float
    embed_features: None
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x, training=True):
        transformer_fn = partial(
            make_transformer,
            self.dim, self.ff_dim, self.n_layers, self.n_heads, self.drop_rate, self.dtype, training=training
        )

        for _ in range(self.gnn_layers):
            x = jraph.GraphNetwork(
                update_edge_fn=transformer_fn(),
                update_node_fn=transformer_fn(),
                update_global_fn=transformer_fn(),
                aggregate_edges_for_nodes_fn=segment_mean,
                aggregate_nodes_for_globals_fn=segment_mean,
                aggregate_edges_for_globals_fn=segment_mean
            )(x)

        x = jraph.GraphMapFeatures(embed_global_fn=make_embed_fn(1, self.dtype))(x)
        return x.globals.squeeze(-1)
