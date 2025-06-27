from typing import Sequence

import jax
import jax.numpy as jnp
import jraph
import flax.linen as nn
from jraph._src.utils import segment_mean

from lai4wifi.autoencoder.model import make_mlp, make_embed_fn
from lai4wifi.autoencoder.utils import clear_graph


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
    num_heads: int
    ff_dim: int
    drop_rate: float
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x, mask, training=True):
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=x.shape[-1],
            dtype=self.dtype,
            use_bias=False
        )(x, mask=mask)
        x = x + residual

        residual = x
        x = nn.LayerNorm()(x)
        x = FeedForwardBlock(self.ff_dim, self.drop_rate, self.dtype)(x, training=training)
        x = x + residual

        return x


class TimeEmbedding(nn.Module):
    # https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/embeddings_flax.py

    ff_dim: int
    latent_size: int
    dtype: jnp.dtype

    @staticmethod
    def get_sinusoidal_embeddings(timesteps, embedding_dim, min_timescale=1, max_timescale=1.0e4):
        num_timescales = float(embedding_dim // 2)
        log_timescale_increment = jnp.log(max_timescale / min_timescale) / num_timescales
        inv_timescales = min_timescale * jnp.exp(jnp.arange(num_timescales, dtype=jnp.float32) * -log_timescale_increment)
        time = jnp.expand_dims(timesteps, -1) * jnp.expand_dims(inv_timescales, 0)
        signal = jnp.concatenate([jnp.cos(time), jnp.sin(time)], axis=-1)
        signal = signal.reshape([*timesteps.shape, embedding_dim])
        return signal

    @nn.compact
    def __call__(self, t):
        t = self.get_sinusoidal_embeddings(t, self.latent_size)
        return FeedForwardBlock(self.ff_dim, drop_rate=0., dtype=self.dtype)(t)


class FlowMatching(nn.Module):
    mlp_features: Sequence[int]
    embed_features: int
    n_layers: int
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x, t, training=True):
        x = x._replace(
            edges=x.edges.reshape(x.edges.shape[0], -1),
            nodes=x.nodes.reshape(x.nodes.shape[0], -1),
            globals=x.globals.reshape(x.globals.shape[0], -1)
        )
        t = TimeEmbedding(4 * x.edges.shape[-1], x.edges.shape[-1], self.dtype)(t)
        x = x._replace(edges=x.edges + t)

        for _ in range(self.n_layers):
            x = jraph.GraphNetwork(
                update_edge_fn=make_mlp(self.mlp_features, self.dtype),
                update_node_fn=make_mlp(self.mlp_features, self.dtype),
                update_global_fn=make_mlp(self.mlp_features, self.dtype),
                aggregate_edges_for_nodes_fn=segment_mean,
                aggregate_nodes_for_globals_fn=segment_mean,
                aggregate_edges_for_globals_fn=segment_mean
            )(x)

        x = jraph.GraphMapFeatures(embed_edge_fn=make_embed_fn(self.embed_features, self.dtype))(x)
        return x

    def gen(self, context, n_steps=10):
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

        z_edges = jax.random.normal(edges_key, (context.edges.shape[0::2]), dtype=self.dtype)
        z_nodes = jax.random.normal(nodes_key, (context.nodes.shape[0::2]), dtype=self.dtype)
        z_globals = jax.random.normal(globals_key, (context.globals.shape[0::2]), dtype=self.dtype)
        z = context._replace(edges=z_edges, nodes=z_nodes, globals=z_globals)

        scan = nn.scan(scan_fn, variable_broadcast='params')
        x, _ = scan(self, z, jnp.linspace(0.0, 1.0, n_steps, endpoint=False))

        return clear_graph(x)
