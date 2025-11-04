from typing import Sequence

import jax.numpy as jnp
import jraph
import flax.linen as nn
from jraph._src.utils import segment_mean

from lai4wifi.graphs.attributes import *
from lai4wifi.graphs.utils import clear_graph


class MLP(nn.Module):
    features: list
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, x):
        residual = x
        x = nn.LayerNorm(dtype=self.dtype)(x)

        for i, size in enumerate(self.features):
            x = nn.Dense(size, dtype=self.dtype)(x)
            x = nn.gelu(x)

        r = nn.Dense(self.features[-1], dtype=self.dtype)(residual)
        x = x + r

        return x


def make_embed_fn(latent_size, dtype):
    @jraph.concatenated_args
    def embed_fn(inputs):
        return nn.Dense(latent_size, dtype=dtype)(inputs)

    return embed_fn


def make_mlp(features, dtype):
    @jraph.concatenated_args
    def update_fn(inputs):
        return MLP(features, dtype)(inputs)

    return update_fn


class Encoder(nn.Module):
    # https://arxiv.org/pdf/2107.09422v1
    # https://github.com/google-deepmind/jraph/blob/master/jraph/ogb_examples/train_flax.py

    embed_features: int
    mlp_features: Sequence[int]
    latent_features: int
    n_layers: int
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, graph):
        x = jraph.GraphMapFeatures(embed_edge_fn=make_embed_fn(self.embed_features, self.dtype))(graph)

        for _ in range(self.n_layers):
            x = jraph.GraphNetwork(
                update_edge_fn=make_mlp(self.mlp_features, self.dtype),
                update_node_fn=make_mlp(self.mlp_features, self.dtype),
                update_global_fn=make_mlp(self.mlp_features, self.dtype),
                aggregate_edges_for_nodes_fn=segment_mean,
                aggregate_nodes_for_globals_fn=segment_mean,
                aggregate_edges_for_globals_fn=segment_mean
            )(x)
            
        z_mean = jraph.GraphMapFeatures(embed_edge_fn=make_embed_fn(self.latent_features, self.dtype))(x)
        z_log_var = jraph.GraphMapFeatures(embed_edge_fn=make_embed_fn(self.latent_features, self.dtype))(x)

        return z_mean, z_log_var


class Decoder(nn.Module):
    mlp_features: Sequence[int]
    n_layers: int
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, z):
        for _ in range(self.n_layers):
            z = jraph.GraphNetwork(
                update_edge_fn=make_mlp(self.mlp_features, self.dtype),
                update_node_fn=make_mlp(self.mlp_features, self.dtype),
                update_global_fn=make_mlp(self.mlp_features, self.dtype),
                aggregate_edges_for_nodes_fn=segment_mean,
                aggregate_nodes_for_globals_fn=segment_mean,
                aggregate_edges_for_globals_fn=segment_mean
            )(z)

        x = nn.gelu(z.edges)

        edges = Connection(
            active=nn.Dense(len(IsActive), dtype=self.dtype)(x),
            mcs=nn.Dense(len(McsValue), dtype=self.dtype)(x),
            rssi=nn.Dense(1, dtype=self.dtype)(x),
            selected=nn.Dense(len(IsSelected), dtype=self.dtype)(x),
            success=nn.Dense(1, dtype=self.dtype)(x),
            tx_power=nn.Dense(len(TxPowerValue), dtype=self.dtype)(x),
            type=nn.Dense(len(EdgeType), dtype=self.dtype)(x)
        )

        return z._replace(edges=edges)


class GnnAutoencoder(nn.Module):
    embed_features: int
    encoder_features: Sequence[int]
    decoder_features: Sequence[int]
    latent_features: int
    n_layers: int
    dtype: jnp.dtype

    def setup(self):
        self.encoder = Encoder(self.embed_features, self.encoder_features, self.latent_features, self.n_layers, self.dtype)
        self.decoder = Decoder(self.decoder_features, self.n_layers, self.dtype)

    def __call__(self, graph):
        z_mean, z_log_var = self.encoder(graph)

        epsilon = jax.random.normal(self.make_rng('lai'), z_mean.edges.shape, dtype=z_mean.edges.dtype)
        edges = z_mean.edges + jnp.exp(0.5 * z_log_var.edges) * epsilon
        z = clear_graph(z_mean._replace(edges=edges))

        x = self.decoder(z)
        return x, z_mean, z_log_var

    def encode(self, graph):
        z_mean, _ = self.encoder(graph)
        return z_mean

    def decode(self, z):
        return self.decoder(z)
