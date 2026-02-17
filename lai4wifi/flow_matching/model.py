import jax
import jax.numpy as jnp
import jraph
import flax.linen as nn
from jraph._src.utils import segment_mean

from lai4wifi.graphs.model import make_embed_fn
from lai4wifi.graphs.utils import clear_graph


class FeedForwardBlock(nn.Module):
    ff_dim: int
    drop_rate: float
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x, training=True):
        out_dim = x.shape[-1]
        x = nn.Dense(self.ff_dim, dtype=self.dtype, use_bias=False)(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.drop_rate)(x, deterministic=not training)
        x = nn.Dense(out_dim, dtype=self.dtype, use_bias=False)(x)
        x = nn.Dropout(self.drop_rate)(x, deterministic=not training)
        return x


class TransformerBlock(nn.Module):
    n_heads: int
    ff_dim: int
    drop_rate: float
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x, mask, training=True):
        residual = x
        x = nn.LayerNorm(dtype=self.dtype)(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=x.shape[-1],
            dtype=self.dtype,
            use_bias=False
        )(x, mask=mask)
        x = x + residual

        residual = x
        x = nn.LayerNorm(dtype=self.dtype)(x)
        x = FeedForwardBlock(self.ff_dim, self.drop_rate, self.dtype)(x, training=training)
        x = x + residual

        return x


class Transformer(nn.Module):
    dim: int
    ff_dim: int
    n_layers: int
    n_heads: int
    drop_rate: float
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x, mask=None, training=True):
        x = nn.Dense(self.dim, dtype=self.dtype, use_bias=False)(x)

        for _ in range(self.n_layers):
            x = TransformerBlock(self.n_heads, self.ff_dim, self.drop_rate, self.dtype)(x, mask=mask, training=training)

        x = nn.LayerNorm(dtype=self.dtype)(x)
        return x


def compute_segment_ids(n_elements, total_length):
    return jnp.searchsorted(jnp.cumsum(n_elements), jnp.arange(total_length), side='right')


def make_transformer(dim, ff_dim, n_layers, n_heads, drop_rate, dtype, segment_ids=None, training=True):
    if segment_ids is not None:
        mask = segment_ids[:, None] == segment_ids[None, :]
    else:
        mask = None

    @jraph.concatenated_args
    def update_fn(inputs):
        return Transformer(dim, ff_dim, n_layers, n_heads, drop_rate, dtype)(inputs, mask=mask, training=training)

    return update_fn


class TimeEmbedding(nn.Module):
    # https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/embeddings_flax.py

    ff_dim: int
    latent_size: int
    drop_rate: float
    dtype: jnp.dtype

    @staticmethod
    def get_sinusoidal_embeddings(timesteps, embedding_dim, min_timescale=1, max_timescale=1.0e4):
        num_timescales = float(embedding_dim // 2)
        log_timescale_increment = jnp.log(max_timescale / min_timescale) / num_timescales
        inv_timescales = min_timescale * jnp.exp(jnp.arange(num_timescales, dtype=jnp.float64) * -log_timescale_increment)
        time = timesteps * inv_timescales[None, :]
        signal = jnp.concatenate([jnp.cos(time), jnp.sin(time)], axis=-1)
        return signal

    @nn.compact
    def __call__(self, t, training=True):
        t = self.get_sinusoidal_embeddings(t, self.latent_size)
        return FeedForwardBlock(self.ff_dim, drop_rate=self.drop_rate, dtype=self.dtype)(t, training=training)


class FlowMatching(nn.Module):
    gnn_layers: int
    dim: int
    ff_dim: int
    n_layers: int
    n_heads: int
    drop_rate: float
    embed_features: int
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x, t, training=True):
        transformer_fn = lambda segment_ids: make_transformer(
            self.dim, self.ff_dim, self.n_layers, self.n_heads, self.drop_rate, self.dtype, segment_ids, training=training
        )
        
        edge_segment_ids = compute_segment_ids(x.n_edge, x.edges.shape[0])
        node_segment_ids = compute_segment_ids(x.n_node, x.nodes.shape[0])
        global_segment_ids = jnp.arange(x.n_node.shape[0])

        x = x._replace(
            edges=x.edges.reshape(x.edges.shape[0], -1),
            nodes=x.nodes.reshape(x.nodes.shape[0], -1),
            globals=x.globals.reshape(x.globals.shape[0], -1)
        )
        t = TimeEmbedding(x.edges.shape[-1], x.edges.shape[-1], self.drop_rate, self.dtype)(t, training=training)
        x = x._replace(edges=x.edges + t)

        for _ in range(self.gnn_layers):
            x = jraph.GraphNetwork(
                update_edge_fn=transformer_fn(edge_segment_ids),
                update_node_fn=transformer_fn(node_segment_ids),
                update_global_fn=transformer_fn(global_segment_ids),
                aggregate_edges_for_nodes_fn=segment_mean,
                aggregate_nodes_for_globals_fn=segment_mean,
                aggregate_edges_for_globals_fn=segment_mean
            )(x)

        x = jraph.GraphMapFeatures(embed_edge_fn=make_embed_fn(self.embed_features, self.dtype))(x)
        return x

    def gen(self, context, n_steps=6):
        def scan_fn(unet, x, t):
            cx = context._replace(
                edges=jnp.concatenate([context.edges, x.edges[:, None]], axis=1),
                nodes=jnp.concatenate([context.nodes, x.nodes[:, None]], axis=1),
                globals=jnp.concatenate([context.globals, x.globals[:, None]], axis=1)
            )
            v = unet(cx, t, training=False)
            x = x._replace(edges=x.edges + v.edges / n_steps)
            return x, None

        key = self.make_rng('lai')
        edges_key, nodes_key, globals_key = jax.random.split(key, 3)

        z_edges = jax.random.normal(edges_key, (context.edges.shape[0], context.edges.shape[-1]), dtype=context.edges.dtype)
        z_nodes = jax.random.normal(nodes_key, (context.nodes.shape[0], context.nodes.shape[-1]), dtype=context.nodes.dtype)
        z_globals = jax.random.normal(globals_key, (context.globals.shape[0], context.globals.shape[-1]), dtype=context.globals.dtype)
        z = context._replace(edges=z_edges, nodes=z_nodes, globals=z_globals)

        scan = nn.scan(scan_fn, variable_broadcast='params')
        x, _ = scan(self, z, jnp.linspace(0.0, 1.0, n_steps, endpoint=False))

        return clear_graph(x)
