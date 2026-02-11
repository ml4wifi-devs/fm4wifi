import flax.linen as nn
import jax.numpy as jnp
import jraph
from jraph._src.utils import segment_mean

from lai4wifi.flow_matching.model import make_transformer, compute_segment_ids
from lai4wifi.graphs.model import make_embed_fn


class SurrogateModel(nn.Module):
    gnn_layers: int
    dim: int
    ff_dim: int
    n_layers: int
    n_heads: int
    n_components: int
    drop_rate: float
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x, training=True):
        transformer_fn = lambda segment_ids: make_transformer(
            self.dim, self.ff_dim, self.n_layers, self.n_heads, self.drop_rate, self.dtype, segment_ids, training=training
        )

        edge_segment_ids = compute_segment_ids(x.n_edge, x.senders.shape[0])
        node_segment_ids = compute_segment_ids(x.n_node, x.nodes.shape[0])
        global_segment_ids = jnp.arange(x.n_node.shape[0])

        for _ in range(self.gnn_layers):
            x = jraph.GraphNetwork(
                update_edge_fn=transformer_fn(edge_segment_ids),
                update_node_fn=transformer_fn(node_segment_ids),
                update_global_fn=transformer_fn(global_segment_ids),
                aggregate_edges_for_nodes_fn=segment_mean,
                aggregate_nodes_for_globals_fn=segment_mean,
                aggregate_edges_for_globals_fn=segment_mean
            )(x)

        x = jraph.GraphMapFeatures(embed_global_fn=make_embed_fn(self.n_components * 3, self.dtype))(x)

        logits, means, raw_scales = jnp.split(x.globals, 3, axis=-1)
        scales = nn.softplus(raw_scales) + 1e-4
        return logits, means, scales
