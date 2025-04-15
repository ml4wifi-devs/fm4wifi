from typing import Callable, Sequence

import jax.numpy as jnp
import jraph
import flax.linen as nn

from lai4wifi.graphs.attributes import *


class MLP(nn.Module):
    features: list
    dtype: jnp.dtype
    act_fn: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.features):
            x = nn.Dense(size, dtype=self.dtype)(x)

            if i < len(self.features) - 1:
                x = self.act_fn(x)

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

    latent_size: int
    features: Sequence[int]
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, graph):
        embed = jraph.GraphMapFeatures(
            embed_node_fn=make_embed_fn(self.latent_size, self.dtype)
        )
        encode = jraph.GraphNetwork(
            update_edge_fn=None,
            update_node_fn=make_mlp(self.features[:-1], self.dtype)
        )
        mean_map = jraph.GraphMapFeatures(
            embed_node_fn=make_embed_fn(self.features[-1], self.dtype)
        )
        log_var_map = jraph.GraphMapFeatures(
            embed_node_fn=make_embed_fn(self.features[-1], self.dtype)
        )

        x = embed(graph)
        z = encode(x)
        z_mean = mean_map(z)
        z_log_var = log_var_map(z)

        return z_mean, z_log_var


class Decoder(nn.Module):
    features: Sequence[int]
    dtype: jnp.dtype

    @nn.compact
    def __call__(self, z):
        decode = jraph.GraphNetwork(
            update_edge_fn=None,
            update_node_fn=make_mlp(self.features, self.dtype)
        )

        graph = decode(z)
        x = nn.relu(graph.nodes)

        nodes = Connection(
            active=nn.Dense(len(IsActive), dtype=self.dtype)(x),
            mcs=nn.Dense(len(McsValue), dtype=self.dtype)(x),
            rssi=nn.Dense(1, dtype=self.dtype)(x),
            selected=nn.Dense(len(IsSelected), dtype=self.dtype)(x),
            success=nn.Dense(1, dtype=self.dtype)(x),
            tx_power=nn.Dense(len(TxPowerValue), dtype=self.dtype)(x),
            type=nn.Dense(len(EdgeType), dtype=self.dtype)(x)
        )

        return jraph.GraphsTuple(
            nodes=nodes,
            edges=None,
            receivers=graph.receivers,
            senders=graph.senders,
            globals=None,
            n_node=graph.n_node,
            n_edge=graph.n_edge
        )


class GnnAutoencoder(nn.Module):
    latent_size: int
    encoder_features: Sequence[int]
    decoder_features: Sequence[int]
    dtype: jnp.dtype

    def setup(self):
        self.encoder = Encoder(self.latent_size, self.encoder_features, self.dtype)
        self.decoder = Decoder(self.decoder_features, self.dtype)

    def __call__(self, graph):
        z_mean, z_log_var = self.encoder(graph)

        epsilon = jax.random.normal(self.make_rng('lai'), z_mean.nodes.shape, dtype=z_mean.nodes.dtype)
        nodes = z_mean.nodes + jnp.exp(0.5 * z_log_var.nodes) * epsilon
        z = jraph.GraphsTuple(
            nodes=nodes,
            edges=None,
            receivers=z_mean.receivers,
            senders=z_mean.senders,
            globals=None,
            n_node=z_mean.n_node,
            n_edge=z_mean.n_edge
        )

        x = self.decoder(z)
        return x, z_mean, z_log_var

    def encode(self, graph):
        z_mean, _ = self.encoder(graph)
        return z_mean

    def decode(self, z):
        return self.decoder(z)
